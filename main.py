#!/usr/bin/env python3
"""
Roblox Trending Chart Forecaster
=================================
Reads a history of trending-chart snapshots (history.ndjson) and produces a
probability forecast over the next chart's top-N places (forecast.json).

Architecture overview
---------------------
  load_snapshots()        → parse NDJSON into a list of raw snapshot dicts
  build_game_registry()   → collect every game's name, appearances, rank history
  compute_recency_weights()→ exponential decay curve over snapshot indices
  compute_velocity()      → rank-change rate (momentum) per game, with drop detection
  score_games()           → weighted rank score per game, incorporating velocity
  apply_absence_penalty() → discount games that have stopped appearing recently
  build_place_distributions() → per-place probability dicts, normalised to 1.0
  rank_top_n()            → ordered list of most-likely top-N universe IDs
  write_forecast()        → serialise and write forecast.json

Each stage is a pure function that takes explicit inputs and returns a value.
Nothing mutates global state; main() is just an orchestrator that wires them.
"""

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants — tune these without touching any logic
# ---------------------------------------------------------------------------

# Exponential decay half-life in number of snapshots.
# A snapshot RECENCY_HALFLIFE steps old contributes exactly half as much weight
# as the most-recent snapshot.  Smaller = more aggressive recency bias.
RECENCY_HALFLIFE: int = 5

# When computing velocity, only look at this many most-recent snapshots.
# A larger window smooths out noise; a smaller window is more reactive.
VELOCITY_WINDOW: int = 6

# How much the velocity bonus is scaled before being added to the rank score.
# 0.0 = ignore velocity entirely.  1.0 = velocity bonus equals one full
# snapshot's worth of weight.
VELOCITY_WEIGHT: float = 0.6

# Absence penalty half-life in snapshots.
# If a game was last seen N snapshots ago, its entire score is multiplied by
# 2^(-N / ABSENCE_HALFLIFE).  Set to a large number to disable absence decay.
# Example: ABSENCE_HALFLIFE=3 → missing for 3 snaps → score halved.
ABSENCE_HALFLIFE: int = 3

# A single-step rank change larger than this (in absolute rank units) is
# treated as a sudden drop rather than normal drift, and velocity regression
# is bypassed in favour of a direct penalty proportional to the drop size.
# Example: DROP_THRESHOLD=5 → jumping from #2 to #8 triggers drop detection.
DROP_THRESHOLD: int = 5

# Default number of ranked places to produce a forecast for.
DEFAULT_FORECAST_PLACES: int = 10

# Default length of the top-N universe-ID list in the output.
DEFAULT_TOP_N: int = 10


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GameRecord:
    """Everything we know about a single game across all snapshots."""
    universe_id: int
    name: str = ""

    # Parallel lists — index i corresponds to snapshot index i.
    # Only populated for snapshots in which this game appeared.
    snapshot_indices: list[int] = field(default_factory=list)
    ranks: list[int] = field(default_factory=list)

    @property
    def latest_rank(self) -> Optional[int]:
        """The rank from the most-recent snapshot this game appeared in."""
        return self.ranks[-1] if self.ranks else None

    @property
    def latest_snapshot_index(self) -> Optional[int]:
        """Index of the most-recent snapshot this game appeared in."""
        return self.snapshot_indices[-1] if self.snapshot_indices else None

    def gaps_missed(self, total_snapshots: int) -> int:
        """
        How many of the most-recent snapshots is this game absent from?

        This is simply (total_snapshots - 1) - latest_snapshot_index.
        A game present in the very last snapshot returns 0.
        A game last seen 3 snapshots ago returns 3.
        A game that never appeared returns total_snapshots (maximum penalty).
        """
        if self.latest_snapshot_index is None:
            return total_snapshots
        return (total_snapshots - 1) - self.latest_snapshot_index


# ---------------------------------------------------------------------------
# Stage 1 — I/O: load raw snapshots
# ---------------------------------------------------------------------------

def load_snapshots(history_path: Path) -> list[dict]:
    """
    Parse history.ndjson into a list of snapshot dicts (one per line).

    Lines that are blank or contain malformed JSON are silently skipped so a
    single bad write to the history file doesn't take down the whole run.
    The returned list is in file order, i.e. index 0 is the oldest snapshot.

    Raises ValueError if the file is missing or contains zero valid snapshots.
    """
    if not history_path.exists():
        raise ValueError(f"History file not found: {history_path}")

    snapshots: list[dict] = []
    with open(history_path, "r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                snapshots.append(json.loads(line))
            except json.JSONDecodeError as exc:
                # Log the bad line but keep going
                print(f"  [warn] Skipping malformed JSON on line {lineno}: {exc}")

    if not snapshots:
        raise ValueError(
            f"No valid snapshots found in {history_path}. "
            "Cannot produce a forecast with no data."
        )

    return snapshots


def load_task(task_path: Path) -> dict:
    """
    Load optional task metadata from task.json.

    Returns an empty dict if the file doesn't exist so callers can always use
    .get() without checking for None first.
    """
    if not task_path.exists():
        return {}
    with open(task_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Stage 2 — Parse: build a registry of every game that ever appeared
# ---------------------------------------------------------------------------

def build_game_registry(snapshots: list[dict]) -> dict[int, GameRecord]:
    """
    Walk every snapshot and accumulate per-game rank histories.

    For each snapshot at index `idx` we look at its `top_trending` list.
    Each entry must have at minimum `universe_id` and `rank`; `name`/`title`/
    `game_name` are optional and we keep whichever we see most recently.

    Returns a dict mapping universe_id → GameRecord, with `snapshot_indices`
    and `ranks` lists in chronological order (index 0 = oldest appearance).
    """
    registry: dict[int, GameRecord] = {}

    for snap_idx, snapshot in enumerate(snapshots):
        entries = snapshot.get("top_trending", [])

        for entry in entries:
            uid: Optional[int] = entry.get("universe_id")
            rank: Optional[int] = entry.get("rank")

            # Both fields are mandatory — skip if either is absent
            if uid is None or rank is None:
                continue

            # Create a new record the first time we see this game
            if uid not in registry:
                registry[uid] = GameRecord(universe_id=uid)

            rec = registry[uid]

            # Update name from whatever field the snapshot uses
            for name_key in ("name", "title", "game_name", "gameName"):
                name_val = entry.get(name_key)
                if name_val and isinstance(name_val, str):
                    rec.name = name_val.strip()
                    break

            rec.snapshot_indices.append(snap_idx)
            rec.ranks.append(rank)

    return registry


# ---------------------------------------------------------------------------
# Stage 3 — Weights: exponential decay over snapshot age
# ---------------------------------------------------------------------------

def compute_recency_weights(num_snapshots: int) -> list[float]:
    """
    Return a list of length `num_snapshots` where index i is the weight for
    snapshot i.  The most-recent snapshot (index num_snapshots-1) always has
    weight 1.0.  Older snapshots decay exponentially:

        w(age) = 2 ^ (-age / RECENCY_HALFLIFE)

    where age = (num_snapshots - 1) - i  (0 for the newest snapshot).

    Example with RECENCY_HALFLIFE=5 and 11 snapshots:
        index 10 → age 0  → w = 1.000
        index  5 → age 5  → w = 0.500
        index  0 → age 10 → w = 0.250
    """
    weights: list[float] = []
    for snapshot_idx in range(num_snapshots):
        age = (num_snapshots - 1) - snapshot_idx
        decay_weight = math.pow(2.0, -age / RECENCY_HALFLIFE)
        weights.append(decay_weight)
    return weights


# ---------------------------------------------------------------------------
# Stage 4 — Velocity: how fast is each game climbing or falling?
# ---------------------------------------------------------------------------

def compute_velocity(
    record: GameRecord,
    total_snapshots: int,
    window: int = VELOCITY_WINDOW,
) -> float:
    """
    Estimate a game's rank-change velocity (rank units per snapshot).

    Three cases are handled in priority order:

    1. SUDDEN DROP — if the game's last recorded rank change in a single step
       exceeded DROP_THRESHOLD places downward, regression is unreliable (one
       huge outlier dominates the slope).  Instead we return a strong negative
       velocity directly proportional to the drop magnitude.  Example: a game
       that fell from #2 to #9 in one tick gets velocity = -(9-2) = -7.

    2. COMPLETE DISAPPEARANCE — if the game is absent from the most-recent
       snapshot entirely, we treat the gap as a worst-case drop to rank
       (forecast_places + 1), i.e. off-chart.  This turns absence into an
       explicit strong negative signal rather than leaving the score to coast.

    3. NORMAL DRIFT — weighted least-squares regression over the recent window,
       with exponential decay weighting so the most-recent points dominate.

    Convention throughout: positive return value = game is RISING (rank number
    dropping), negative = game is FALLING (rank number increasing or gone).

    Returns 0.0 if the game has fewer than 2 data points.
    """
    if len(record.ranks) < 2:
        return 0.0

    # ------------------------------------------------------------------
    # Case 2: game has vanished from recent snapshots entirely.
    # Treat the missing appearance as a drop to just off-chart (rank = very
    # large) so the velocity is strongly negative.
    # ------------------------------------------------------------------
    snapshots_absent = record.gaps_missed(total_snapshots)
    if snapshots_absent > 0:
        # Synthesise a "last seen at latest_rank, now off-chart" drop.
        # We use snapshots_absent as a multiplier so longer absences are penalised harder.
        synthesised_off_chart_rank = record.latest_rank + (DROP_THRESHOLD * snapshots_absent)
        # Return a negative velocity: the game has effectively fallen far.
        return -(synthesised_off_chart_rank - record.latest_rank) / snapshots_absent

    # ------------------------------------------------------------------
    # Case 1: check for a sudden single-step drop in the last appearance.
    # Only check the final step — earlier large moves are historical noise.
    # ------------------------------------------------------------------
    last_step_rank_change = record.ranks[-1] - record.ranks[-2]  # positive = rank number rose = game fell
    if last_step_rank_change >= DROP_THRESHOLD:
        # Bypass regression entirely; the drop IS the velocity signal.
        # Negate so the convention holds (positive = rising).
        return -float(last_step_rank_change)

    # ------------------------------------------------------------------
    # Case 3: normal weighted least-squares regression.
    # ------------------------------------------------------------------
    recent_indices = record.snapshot_indices[-window:]
    recent_ranks = record.ranks[-window:]
    window_size = len(recent_indices)

    if window_size < 2:
        return 0.0

    # Build per-point weights using the same exponential decay, normalised so
    # the most-recent point always has weight 1.0.
    most_recent_snap_idx = recent_indices[-1]
    point_weights = [
        math.pow(2.0, -(most_recent_snap_idx - snap_idx) / RECENCY_HALFLIFE)
        for snap_idx in recent_indices
    ]

    # Weighted least-squares slope:  slope = Σw(x-x̄)(y-ȳ) / Σw(x-x̄)²
    # where x = snapshot_index, y = rank, w = point weight
    sum_of_weights = sum(point_weights)
    weighted_mean_snap_idx = sum(point_weights[i] * recent_indices[i] for i in range(window_size)) / sum_of_weights
    weighted_mean_rank     = sum(point_weights[i] * recent_ranks[i]   for i in range(window_size)) / sum_of_weights

    numerator = sum(
        point_weights[i] * (recent_indices[i] - weighted_mean_snap_idx) * (recent_ranks[i] - weighted_mean_rank)
        for i in range(window_size)
    )
    denominator = sum(
        point_weights[i] * (recent_indices[i] - weighted_mean_snap_idx) ** 2
        for i in range(window_size)
    )

    if denominator == 0.0:
        return 0.0

    rank_change_per_snapshot = numerator / denominator   # rank units per snapshot index

    # Negate: a falling rank number means the game is rising.
    return -rank_change_per_snapshot


# ---------------------------------------------------------------------------
# Stage 5 — Scoring: accumulate a weighted rank-score for every game
# ---------------------------------------------------------------------------

def score_games(
    registry: dict[int, GameRecord],
    snapshot_weights: list[float],
    forecast_places: int,
) -> dict[int, dict[int, float]]:
    """
    For each game, accumulate a weighted "evidence score" for every rank it
    has ever held within [1, forecast_places].

    Then apply a velocity bonus: if a game is trending upward, boost its
    probability at the rank it's projected to reach on the next tick.

    Returns a nested dict:  raw_scores[universe_id][place] = accumulated weight

    These are raw (unnormalised) weights — normalisation happens in a later
    stage so each place's distribution is treated independently.
    """
    # raw_scores[uid][place] = sum of snapshot weights for every snapshot where
    # uid held that place
    raw_scores: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))

    for uid, game_record in registry.items():
        for snap_idx, rank in zip(game_record.snapshot_indices, game_record.ranks):
            if 1 <= rank <= forecast_places:
                raw_scores[uid][rank] += snapshot_weights[snap_idx]

    total_snapshots = len(snapshot_weights)

    # Velocity bonus/penalty pass
    for uid, game_record in registry.items():
        velocity = compute_velocity(game_record, total_snapshots)

        # Only apply an adjustment if the game is actually moving
        if abs(velocity) < 0.1:
            continue

        current_rank = game_record.latest_rank
        if current_rank is None:
            continue

        # Project one tick forward: where will this game be next?
        projected_rank = round(current_rank - velocity)   # subtract because
        # positive velocity = rising = rank number drops

        projected_rank = max(1, min(forecast_places, projected_rank))

        # The adjustment magnitude is proportional to velocity and the recency
        # weight of the last snapshot this game appeared in
        last_snap_weight = snapshot_weights[game_record.latest_snapshot_index]
        velocity_score_adjustment = abs(velocity) * VELOCITY_WEIGHT * last_snap_weight

        raw_scores[uid][projected_rank] += velocity_score_adjustment

    return raw_scores


# ---------------------------------------------------------------------------
# Stage 5a — Absence penalty: discount games that have gone quiet
# ---------------------------------------------------------------------------

def apply_absence_penalty(
    raw_scores: dict[int, dict[int, float]],
    registry: dict[int, GameRecord],
    total_snapshots: int,
) -> dict[int, dict[int, float]]:
    """
    Multiply every score for a game by an exponential absence penalty if that
    game has not appeared in recent snapshots.

    The penalty factor is:

        penalty(gaps) = 2 ^ (-gaps / ABSENCE_HALFLIFE)

    where `gaps` is the number of trailing snapshots the game is absent from.
    A game present in the latest snapshot is unaffected (penalty = 1.0).
    A game missing for ABSENCE_HALFLIFE snapshots has all its scores halved.
    A game missing for 2×ABSENCE_HALFLIFE snapshots has scores quartered, etc.

    This handles the "strong historical record but suddenly gone" case: without
    this stage, such a game coasts on its accumulated weight indefinitely.
    The velocity pass in score_games already applies a negative bonus for
    absent games; this stage compounds that by directly discounting the base
    score so the effect is visible in the final probabilities.

    Returns a new raw_scores dict (does not mutate the input).
    """
    penalised: dict[int, dict[int, float]] = {}

    for uid, place_weights in raw_scores.items():
        rec = registry.get(uid)
        if rec is None:
            penalised[uid] = dict(place_weights)
            continue

        snapshots_absent = rec.gaps_missed(total_snapshots)

        if snapshots_absent == 0:
            # Game is current — no penalty, copy scores unchanged
            penalised[uid] = dict(place_weights)
        else:
            # Apply exponential decay to every place score for this game
            absence_penalty_factor = math.pow(2.0, -snapshots_absent / ABSENCE_HALFLIFE)
            penalised[uid] = {place: w * absence_penalty_factor for place, w in place_weights.items()}

            if snapshots_absent >= ABSENCE_HALFLIFE:
                # Log noticeably penalised games so it's easy to spot in output
                name = rec.name or str(uid)
                print(
                    f"  [absence] {name} (uid={uid}) missing {snapshots_absent} snapshots "
                    f"→ score × {absence_penalty_factor:.3f}"
                )

    return penalised


# ---------------------------------------------------------------------------
# Stage 6 — Distributions: per-place probability dicts, normalised
# ---------------------------------------------------------------------------

def normalise_to_probability_distribution(
    unnormalised_weights: dict[int, float],
) -> dict[int, float]:
    """
    Convert a dict of raw {uid: weight} scores into a proper probability
    distribution where all values sum to exactly 1.0.

    Each value is divided by the total weight, so the relative ordering of
    games is preserved — only the scale changes.

    Returns an empty dict if the input is empty (no candidates for this place),
    rather than dividing by zero.
    """
    if not unnormalised_weights:
        return {}

    total_weight = sum(unnormalised_weights.values())
    return {uid: weight / total_weight for uid, weight in unnormalised_weights.items()}


def build_place_distributions(
    raw_scores: dict[int, dict[int, float]],
    forecast_places: int,
) -> dict[int, dict[int, float]]:
    """
    Invert the uid→place→weight structure into place→uid→probability.

    For each place, all candidate games and their raw weights are collected,
    then passed to normalise_to_probability_distribution() so that the
    probabilities for each place sum to exactly 1.0.

    Games that never appeared at a given rank will have no entry in that
    place's distribution (rather than a 0.0 entry), keeping the output lean.

    Returns:  place_dist[place][universe_id] = probability (0 < p ≤ 1)
    """
    # Accumulate raw weights keyed by place
    place_raw: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))

    for uid, place_weights in raw_scores.items():
        for place, weight in place_weights.items():
            if 1 <= place <= forecast_places:
                place_raw[place][uid] += weight

    # Convert each place's raw weights into a normalised probability distribution
    return {
        place: normalise_to_probability_distribution(dict(place_raw.get(place, {})))
        for place in range(1, forecast_places + 1)
    }


# ---------------------------------------------------------------------------
# Stage 7 — Ranking: produce an ordered top-N list of universe IDs
# ---------------------------------------------------------------------------

def rank_top_n(
    place_dist: dict[int, dict[int, float]],
    registry: dict[int, GameRecord],
    top_n: int,
    forecast_places: int,
) -> list[int]:
    """
    Derive a single ranked list of the most-likely top-N universe IDs.

    Scoring method: for each game, sum p(place) * (1 / place) across all
    places.  Dividing by place means strong probability at #1 is worth much
    more than strong probability at #10.  This collapses the full probability
    matrix into one sortable scalar per game.

    If fewer than top_n games appear in the distributions (sparse data), pad
    the list by appending remaining games ordered by their most-recent rank.
    """
    composite_score_by_uid: dict[int, float] = defaultdict(float)

    for place, dist in place_dist.items():
        for uid, probability in dist.items():
            composite_score_by_uid[uid] += probability / place   # higher place = higher divisor = less value

    top_uid_list = sorted(composite_score_by_uid.keys(), key=lambda uid: -composite_score_by_uid[uid])[:top_n]

    # Pad if needed — sort unseen games by latest rank (best rank first)
    if len(top_uid_list) < top_n:
        already_ranked_uids = set(top_uid_list)
        fallback_records = sorted(
            [rec for uid, rec in registry.items() if uid not in already_ranked_uids and rec.latest_rank],
            key=lambda rec: rec.latest_rank,
        )
        for rec in fallback_records:
            top_uid_list.append(rec.universe_id)
            if len(top_uid_list) >= top_n:
                break

    return top_uid_list[:top_n]


# ---------------------------------------------------------------------------
# Stage 8 — Output: serialise and write forecast.json
# ---------------------------------------------------------------------------

def build_forecast_payload(
    place_dist: dict[int, dict[int, float]],
    top10: list[int],
    registry: dict[int, GameRecord],
    num_snapshots: int,
    forecast_places: int,
) -> dict:
    """
    Assemble the final JSON-serialisable forecast dict.

    Output schema
    -------------
    {
      "probabilities": [
        {"universe_id": int, "place": int, "p": float, "name": str},
        ...   // sorted by place asc, then p desc within each place
      ],
      "top10": [int, ...],               // ordered universe IDs
      "top10_named": [                   // same list, with names
        {"universe_id": int, "name": str}, ...
      ],
      "meta": {
        "snapshots_used": int,
        "recency_halflife_snapshots": int,
        "velocity_window": int,
        "velocity_weight": float,
        "forecast_places": int
      }
    }
    """
    probabilities = []
    for place in range(1, forecast_places + 1):
        dist = place_dist.get(place, {})
        # Sort by probability descending so the most likely game is first
        for uid, probability in sorted(dist.items(), key=lambda uid_prob_pair: -uid_prob_pair[1]):
            prob_entry: dict = {
                "universe_id": uid,
                "place": place,
                "p": round(probability, 6),
            }
            name = registry[uid].name if uid in registry else ""
            if name:
                prob_entry["name"] = name
            probabilities.append(prob_entry)

    top10_named = [
        {
            "universe_id": uid,
            "name": registry[uid].name if uid in registry else "",
        }
        for uid in top10
    ]

    return {
        "probabilities": probabilities,
        "top10": top10,
        "top10_named": top10_named,
        "meta": {
            "snapshots_used": num_snapshots,
            "recency_halflife_snapshots": RECENCY_HALFLIFE,
            "absence_halflife_snapshots": ABSENCE_HALFLIFE,
            "drop_threshold_ranks": DROP_THRESHOLD,
            "velocity_window": VELOCITY_WINDOW,
            "velocity_weight": VELOCITY_WEIGHT,
            "forecast_places": forecast_places,
        },
    }


def write_forecast(payload: dict, output_path: Path) -> None:
    """Write the forecast payload to disk as indented JSON (UTF-8)."""
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"Forecast written → {output_path}")


# ---------------------------------------------------------------------------
# Main — wire the pipeline together
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate the full forecasting pipeline:

        load → parse → weight → score → penalise → distribute → rank → write

    All heavy logic lives in the stage functions above; this function only
    reads config, calls them in order, and handles top-level error reporting.
    """
    # -- Resolve paths from environment variables or fall back to defaults --
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load task config (optional) ---------------------------------------
    task_data = load_task(input_dir / "task.json")
    forecast_places: int = int(task_data.get("forecast_places", DEFAULT_FORECAST_PLACES))
    top_n: int = int(task_data.get("top_n", DEFAULT_TOP_N))

    print(f"Config: forecast_places={forecast_places}, top_n={top_n}")

    # -- Stage 1: Load raw snapshot history --------------------------------
    snapshots = load_snapshots(input_dir / "history.ndjson")
    print(f"Loaded {len(snapshots)} snapshots")

    # -- Stage 2: Build game registry from all snapshots -------------------
    registry = build_game_registry(snapshots)
    print(f"Registry contains {len(registry)} unique games")

    # -- Stage 3: Compute per-snapshot recency weights ---------------------
    snapshot_weights = compute_recency_weights(len(snapshots))
    # (no print here — it's just a list of floats, nothing interesting to log)

    # -- Stage 4+5: Score games (velocity is computed inside score_games) --
    raw_scores = score_games(registry, snapshot_weights, forecast_places)

    # -- Stage 5a: Discount games absent from recent snapshots -------------
    raw_scores = apply_absence_penalty(raw_scores, registry, len(snapshots))

    # -- Stage 6: Normalise into per-place probability distributions -------
    place_dist = build_place_distributions(raw_scores, forecast_places)

    # -- Stage 7: Derive an ordered top-N list -----------------------------
    top10 = rank_top_n(place_dist, registry, top_n, forecast_places)

    # -- Stage 8: Serialise and write output -------------------------------
    payload = build_forecast_payload(
        place_dist, top10, registry, len(snapshots), forecast_places
    )
    write_forecast(payload, output_dir / "forecast.json")


if __name__ == "__main__":
    main()
