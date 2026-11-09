#!/usr/bin/env python3
"""
Roblox Trending Chart Forecaster  —  Sticky-Dropout-Velocity Edition
======================================================================
Winning strategy implementation based on the -20.6 log-score approach:

  1. STICKY CHART        — games already on the chart have very high probability
                           of remaining.  Top positions are "stickier" than lower.
  2. DROPOUT MIXTURE     — when a game suddenly disappears (or is on the verge of
                           leaving), its probability mass is redistributed to its
                           nearest-rank neighbours rather than evaporating.
  3. RANK-VELOCITY       — a 14-hour velocity estimate (rank-change per hour) is
                           computed via exponentially-weighted regression and used
                           to shift each game's distribution forward in time.
  4. BOTTOM-HEAVY PRIOR  — lower ranks receive a heavier uniform prior so the
                           forecast acknowledges that any game can bubble up from
                           outside the chart, not just current occupants.
  5. BRADLEY-TERRY PAIRS — pairwise win probabilities between all candidate games
                           per rank position are aggregated into final per-place
                           marginal probabilities, capturing relative strength
                           better than a simple normalisation.
  6. EXPONENTIAL DECAY   — recent snapshots are weighted geometrically more than
                           older ones; half-life is expressed in hours so it is
                           robust to irregular snapshot intervals.

Pipeline
--------
  load_snapshots()                → list[Snapshot]
  build_game_registry()           → dict[universe_id → GameRecord]
  compute_snapshot_weights()      → list[float]   (per snapshot, by real timestamp)
  estimate_velocity_14h()         → dict[universe_id → rank_displacement_float]
  compute_sticky_scores()         → dict[universe_id → dict[place → evidence_score]]
  apply_dropout_mixture()         → dict[universe_id → dict[place → evidence_score]]
  shift_by_velocity()             → dict[universe_id → dict[place → evidence_score]]
  add_bottom_heavy_prior()        → dict[universe_id → dict[place → evidence_score]]
  bradley_terry_marginals()       → dict[place → dict[universe_id → probability]]
  normalize_place_distributions() → dict[place → dict[universe_id → probability]]
  verify_normalization()          → list[str]   (audit warnings; empty = all clear)
  rank_top_n()                    → list[universe_id]
  build_forecast_payload()        → dict
  write_forecast()                → (side-effect only)
"""

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ============================================================================
# TUNABLE CONSTANTS
# ============================================================================

# Recency decay — expressed as a human-readable "boost" goal rather than
# an arbitrary half-life:
#
#   RECENCY_BOOST_HOURS  — the age boundary in hours below which snapshots
#                          are considered "recent".
#   RECENCY_BOOST_FACTOR — how many times more weight a snapshot at age 0
#                          should carry vs one at exactly RECENCY_BOOST_HOURS.
#
# The half-life is derived automatically so the decay curve hits exactly
# the requested ratio at the requested boundary:
#
#   decay_halflife = RECENCY_BOOST_HOURS * ln(2) / ln(RECENCY_BOOST_FACTOR)
#
# Example (defaults):  3 h boundary, 5× boost
#   decay_halflife ≈ 3 * 0.6931 / 1.6094 ≈ 1.292 hours
#   → a snapshot taken 3 h ago has weight 1/5 of the most-recent snapshot
#   → a snapshot taken 6 h ago has weight 1/25  (5× again)
#   → a snapshot taken 1 h ago has weight ≈ 0.59  (still well above the 3 h level)
RECENCY_BOOST_HOURS:  float = 3.0    # recent window boundary in hours
RECENCY_BOOST_FACTOR: float = 5.0    # snapshots inside the window are this many × heavier

import math as _math
RECENCY_HALFLIFE_HOURS: float = (
    RECENCY_BOOST_HOURS * _math.log(2) / _math.log(RECENCY_BOOST_FACTOR)
)   # ≈ 1.292 h with the defaults above

# Maximum number of recent snapshots used when fitting the rank-velocity
# regression line per game.
VELOCITY_REGRESSION_WINDOW: int = 12

# How many hours ahead to project the fitted velocity when computing each
# game's predicted rank displacement.
VELOCITY_HORIZON_HOURS: float = 14.0

# Fraction of a dropping game's total evidence score that is redistributed
# to its neighbouring-rank games during the dropout mixture pass.
# The remaining (1 - fraction) stays with the dropping game itself.
DROPOUT_MASS_REDISTRIBUTION_FRACTION: float = 0.45

# Number of immediate rank neighbours on each side of a dropping game's
# anchor rank that receive redistributed probability mass.
# e.g. with value 2: ranks (anchor-2),(anchor-1),(anchor+1),(anchor+2).
DROPOUT_NEIGHBOUR_RANK_WINDOW: int = 2

# Stickiness weight applied to each (game, place) observation.
# sticky_weight(rank) = clamp(BASE - rank * PER_RANK_DECAY, MIN, 1.0)
# A game at rank 1 gets weight BASE; each rank lower reduces it by PER_RANK_DECAY.
STICKINESS_WEIGHT_BASE:          float = 0.92
STICKINESS_WEIGHT_PER_RANK_DECAY: float = 0.012
STICKINESS_WEIGHT_MIN:           float = 0.60

# Bottom-heavy prior weight added to every (game, place) pair.
# prior_weight(rank) = BASE + rank * RANK_SLOPE
# Higher-numbered (lower) ranks receive a larger prior, reflecting that new
# games tend to enter the chart from outside the top positions.
PRIOR_WEIGHT_BASE:       float = 0.005
PRIOR_WEIGHT_RANK_SLOPE: float = 0.0008

# Scaling factor for Bradley-Terry strength conversion.
# A game's BT strength = exp(evidence_score * BT_STRENGTH_SCALE).
# Higher values make the strongest game's probability more dominant.
BT_STRENGTH_SCALE: float = 3.5

# Maximum number of candidate games per place included in the Bradley-Terry
# strength comparison.  Games are ranked by evidence score and only the top
# BT_CANDIDATE_POOL_SIZE are compared, keeping the computation tractable.
BT_CANDIDATE_POOL_SIZE: int = 30

# A single-step rank change equal to or larger than this value (in rank
# positions) is treated as a sudden drop rather than normal drift.
# Example: DROP_DETECTION_THRESHOLD=4 means jumping from #2 to #6 triggers
# the dropout treatment instead of the normal velocity regression.
DROP_DETECTION_THRESHOLD: int = 4

# Default number of ranked chart positions to generate forecasts for.
DEFAULT_FORECAST_PLACES: int = 10

# Default length of the top-N universe-ID list written to forecast.json.
DEFAULT_TOP_N: int = 10

# When a snapshot carries no timestamp in its JSON, assume it was taken this
# many seconds after the previous snapshot (used only as a fallback).
FALLBACK_SNAPSHOT_GAP_SECONDS: float = 1800.0   # 30 minutes


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Snapshot:
    """A single scraped chart state read from one line of history.ndjson."""
    index: int                    # 0 = oldest snapshot in the file
    timestamp_unix: float         # Unix epoch seconds; estimated when absent
    chart_entries: list[dict]     # raw top_trending list from the NDJSON line


@dataclass
class GameRecord:
    """Full chart appearance history of a single Roblox game."""
    universe_id: int
    display_name: str = ""

    # Parallel lists in chronological order.
    # Only snapshots where this game appeared are recorded.
    snapshot_indices:    list[int]   = field(default_factory=list)
    snapshot_timestamps: list[float] = field(default_factory=list)  # Unix seconds
    chart_ranks:         list[int]   = field(default_factory=list)

    @property
    def latest_rank(self) -> Optional[int]:
        """Chart rank from the most recent snapshot this game appeared in."""
        return self.chart_ranks[-1] if self.chart_ranks else None

    @property
    def latest_snapshot_index(self) -> Optional[int]:
        """Index of the most recent snapshot this game appeared in."""
        return self.snapshot_indices[-1] if self.snapshot_indices else None

    @property
    def latest_appearance_time(self) -> Optional[float]:
        """Unix timestamp of the most recent snapshot this game appeared in."""
        return self.snapshot_timestamps[-1] if self.snapshot_timestamps else None

    def absent_trailing_snapshots(self, total_snapshot_count: int) -> int:
        """
        How many of the most-recent snapshots is this game absent from?
        0 = present in the very last snapshot.
        N = last seen N snapshots ago.
        total_snapshot_count = never appeared.
        """
        if self.latest_snapshot_index is None:
            return total_snapshot_count
        return (total_snapshot_count - 1) - self.latest_snapshot_index

    def is_on_current_chart(self, total_snapshot_count: int) -> bool:
        """True if this game appeared in the most recent snapshot."""
        return self.absent_trailing_snapshots(total_snapshot_count) == 0


# ============================================================================
# STAGE 1  —  I/O: load history.ndjson and task.json
# ============================================================================

def _extract_unix_timestamp(
    snapshot_dict: dict,
    fallback_time: float,
    fallback_gap_seconds: float,
) -> float:
    """
    Try to extract a Unix timestamp from a raw snapshot dict.
    Checks several common field names; falls back to
    fallback_time + fallback_gap_seconds when none is found.
    """
    for candidate_field in ("timestamp", "ts", "created_at", "time", "snapshot_time"):
        raw_value = snapshot_dict.get(candidate_field)
        if raw_value is None:
            continue
        if isinstance(raw_value, (int, float)):
            # Distinguish milliseconds (> year 2001 in ms) from seconds
            if raw_value > 1e12:
                return raw_value / 1000.0
            return float(raw_value)
        if isinstance(raw_value, str):
            try:
                import datetime
                parsed_datetime = datetime.datetime.fromisoformat(raw_value.rstrip("Z"))
                return parsed_datetime.timestamp()
            except ValueError:
                pass
    return fallback_time + fallback_gap_seconds


def load_snapshots(history_path: Path) -> list[Snapshot]:
    """
    Parse history.ndjson into a list of Snapshot objects, oldest first.
    Malformed JSON lines are skipped with a printed warning.
    Raises ValueError if the file is missing or entirely unreadable.
    """
    if not history_path.exists():
        raise ValueError(f"History file not found: {history_path}")

    parsed_lines: list[dict] = []
    with open(history_path, "r", encoding="utf-8") as history_file:
        for line_number, raw_line in enumerate(history_file, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                parsed_lines.append(json.loads(stripped))
            except json.JSONDecodeError as parse_error:
                print(f"  [warn] Skipping malformed JSON on line {line_number}: {parse_error}")

    if not parsed_lines:
        raise ValueError(
            f"No valid snapshots found in {history_path}. "
            "Cannot forecast with no data."
        )

    snapshots: list[Snapshot] = []
    running_timestamp: float = 0.0

    for snapshot_index, raw_dict in enumerate(parsed_lines):
        estimated_timestamp = _extract_unix_timestamp(
            raw_dict, running_timestamp, FALLBACK_SNAPSHOT_GAP_SECONDS
        )
        running_timestamp = estimated_timestamp
        chart_entries = raw_dict.get("top_trending", [])
        snapshots.append(
            Snapshot(
                index=snapshot_index,
                timestamp_unix=estimated_timestamp,
                chart_entries=chart_entries,
            )
        )

    return snapshots


def load_task_config(task_path: Path) -> dict:
    """Load optional task.json configuration; return {} if the file is absent."""
    if not task_path.exists():
        return {}
    with open(task_path, "r", encoding="utf-8") as task_file:
        return json.load(task_file)


# ============================================================================
# STAGE 2  —  Parse: build GameRecord registry from all snapshots
# ============================================================================

def build_game_registry(snapshots: list[Snapshot]) -> dict[int, GameRecord]:
    """
    Walk every snapshot and build a per-game chart history.
    Returns universe_id → GameRecord with lists in chronological order.
    """
    registry: dict[int, GameRecord] = {}

    for snapshot in snapshots:
        for chart_entry in snapshot.chart_entries:
            universe_id: Optional[int] = chart_entry.get("universe_id")
            chart_rank:  Optional[int] = chart_entry.get("rank")
            if universe_id is None or chart_rank is None:
                continue

            if universe_id not in registry:
                registry[universe_id] = GameRecord(universe_id=universe_id)

            game_record = registry[universe_id]

            # Accept several common name field spellings
            for candidate_name_field in ("name", "title", "game_name", "gameName"):
                candidate_name = chart_entry.get(candidate_name_field)
                if candidate_name and isinstance(candidate_name, str):
                    game_record.display_name = candidate_name.strip()
                    break

            game_record.snapshot_indices.append(snapshot.index)
            game_record.snapshot_timestamps.append(snapshot.timestamp_unix)
            game_record.chart_ranks.append(chart_rank)

    return registry


# ============================================================================
# STAGE 3  —  Recency weights keyed to real elapsed hours
# ============================================================================

def compute_snapshot_weights(snapshots: list[Snapshot]) -> list[float]:
    """
    For each snapshot, compute an exponential-decay weight relative to the
    most-recent snapshot's timestamp.

    Decay formula
    -------------
        snapshot_age_hours = (latest_timestamp - snap.timestamp_unix) / 3600
        decay_weight(age)  = 2 ^ ( -snapshot_age_hours / RECENCY_HALFLIFE_HOURS )

    The half-life is derived from the user-facing boost constants so that:
        decay_weight(0)                   = 1.0           (most-recent snapshot)
        decay_weight(RECENCY_BOOST_HOURS) = 1 / RECENCY_BOOST_FACTOR

    With defaults RECENCY_BOOST_HOURS=3, RECENCY_BOOST_FACTOR=5:
        halflife ≈ 1.292 h
        decay_weight(0 h)   = 1.000   ← snapshot taken right now
        decay_weight(1 h)   ≈ 0.588
        decay_weight(2 h)   ≈ 0.346
        decay_weight(3 h)   = 0.200   ← exactly 5× less than weight(0 h)
        decay_weight(6 h)   = 0.040   ← 25× less (5² less)
        decay_weight(12 h)  ≈ 0.002   ← 625× less

    This means the last 3 hours carry the vast majority of total evidential weight.
    """
    if not snapshots:
        return []

    latest_snapshot_time = snapshots[-1].timestamp_unix
    snapshot_weights: list[float] = []

    for snapshot in snapshots:
        snapshot_age_hours = (latest_snapshot_time - snapshot.timestamp_unix) / 3600.0
        decay_weight = math.pow(2.0, -snapshot_age_hours / RECENCY_HALFLIFE_HOURS)
        snapshot_weights.append(decay_weight)

    return snapshot_weights


# ============================================================================
# STAGE 4  —  Rank-velocity over the last 14 hours
# ============================================================================

def _weighted_least_squares_slope(
    time_points_hours: list[float],
    rank_values:       list[float],
    point_weights:     list[float],
) -> float:
    """
    Compute the weighted least-squares slope (rank change per hour).

    Uses the standard weighted linear regression formula:
        slope = (Σw·x·y  −  Σw·x · Σw·y / Σw)
              / (Σw·x²   −  (Σw·x)² / Σw)

    Returns 0.0 for degenerate inputs (fewer than 2 points, or zero total weight,
    or zero variance in the x values).
    """
    num_points = len(time_points_hours)
    if num_points < 2:
        return 0.0

    sum_weights = sum(point_weights)
    if sum_weights == 0.0:
        return 0.0

    sum_wx  = sum(point_weights[i] * time_points_hours[i] for i in range(num_points))
    sum_wy  = sum(point_weights[i] * rank_values[i]       for i in range(num_points))
    sum_wxx = sum(point_weights[i] * time_points_hours[i] * time_points_hours[i]
                  for i in range(num_points))
    sum_wxy = sum(point_weights[i] * time_points_hours[i] * rank_values[i]
                  for i in range(num_points))

    # Denominator = weighted variance of x, scaled by Σw
    weighted_x_variance = sum_weights * sum_wxx - sum_wx * sum_wx
    if abs(weighted_x_variance) < 1e-12:
        return 0.0

    return (sum_weights * sum_wxy - sum_wx * sum_wy) / weighted_x_variance


def estimate_velocity_14h(
    registry:  dict[int, GameRecord],
    snapshots: list[Snapshot],
) -> dict[int, float]:
    """
    Estimate each game's rank-change velocity and project it 14 hours forward.

    Returns universe_id → predicted_rank_displacement
    (positive = rank number rises = game falls in the chart;
     negative = rank number falls = game rises in the chart)

    Three cases handled in priority order:
      1. Sudden drop  — last visible rank jump >= DROP_DETECTION_THRESHOLD:
                        bypass regression; use the jump magnitude directly.
      2. Absent game  — not on the latest chart: synthesise an off-chart point
                        at a large rank number so the velocity is strongly positive.
      3. Normal drift — weighted least-squares regression over recent appearances.
    """
    total_snapshot_count = len(snapshots)
    latest_snapshot_time = snapshots[-1].timestamp_unix if snapshots else 0.0
    velocity_by_uid: dict[int, float] = {}

    def snapshot_decay_weight(snapshot_time: float) -> float:
        snapshot_age_hours = (latest_snapshot_time - snapshot_time) / 3600.0
        return math.pow(2.0, -snapshot_age_hours / RECENCY_HALFLIFE_HOURS)

    for universe_id, game_record in registry.items():
        if not game_record.chart_ranks:
            velocity_by_uid[universe_id] = 0.0
            continue

        absent_count = game_record.absent_trailing_snapshots(total_snapshot_count)

        # Case 1 — sudden single-step rank drop
        if len(game_record.chart_ranks) >= 2:
            last_step_rank_change = game_record.chart_ranks[-1] - game_record.chart_ranks[-2]
            if last_step_rank_change >= DROP_DETECTION_THRESHOLD and absent_count == 0:
                # Project that drop rate across the full horizon
                velocity_by_uid[universe_id] = float(last_step_rank_change) * VELOCITY_HORIZON_HOURS
                continue

        # Build regression inputs from recent appearances
        recent_times = game_record.snapshot_timestamps[-VELOCITY_REGRESSION_WINDOW:]
        recent_ranks = game_record.chart_ranks[-VELOCITY_REGRESSION_WINDOW:]

        regression_time_hours: list[float] = [t / 3600.0 for t in recent_times]
        regression_rank_values: list[float] = list(recent_ranks)
        regression_point_weights: list[float] = [
            snapshot_decay_weight(t) for t in recent_times
        ]

        # Case 2 — absent game: inject a synthetic "off-chart" observation at now
        if absent_count > 0:
            off_chart_rank = (
                (game_record.latest_rank or DEFAULT_FORECAST_PLACES)
                + DROP_DETECTION_THRESHOLD * absent_count
            )
            regression_time_hours.append(latest_snapshot_time / 3600.0)
            regression_rank_values.append(float(off_chart_rank))
            regression_point_weights.append(1.0)   # full weight — most recent signal

        # Case 3 — normal weighted regression
        rank_change_per_hour = _weighted_least_squares_slope(
            regression_time_hours,
            regression_rank_values,
            regression_point_weights,
        )
        velocity_by_uid[universe_id] = rank_change_per_hour * VELOCITY_HORIZON_HOURS

    return velocity_by_uid


# ============================================================================
# STAGE 5  —  Sticky scoring: weight of each game at each place
# ============================================================================

def compute_sticky_scores(
    registry:         dict[int, GameRecord],
    snapshot_weights: list[float],
    snapshots:        list[Snapshot],
    forecast_places:  int,
) -> dict[int, dict[int, float]]:
    """
    For each (game, place) pair, accumulate a weighted evidence score from
    the historical chart data, amplified by a stickiness multiplier that
    favours games already near the top.

    stickiness_multiplier(rank) = clamp(
        STICKINESS_WEIGHT_BASE - rank * STICKINESS_WEIGHT_PER_RANK_DECAY,
        STICKINESS_WEIGHT_MIN,
        1.0,
    )

    Returns evidence_scores[universe_id][place] = accumulated weighted score.
    """
    evidence_scores: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))

    for universe_id, game_record in registry.items():
        for snapshot_index, chart_rank in zip(
            game_record.snapshot_indices, game_record.chart_ranks
        ):
            if not (1 <= chart_rank <= forecast_places):
                continue
            stickiness_multiplier = max(
                STICKINESS_WEIGHT_MIN,
                min(
                    1.0,
                    STICKINESS_WEIGHT_BASE - chart_rank * STICKINESS_WEIGHT_PER_RANK_DECAY,
                ),
            )
            evidence_scores[universe_id][chart_rank] += (
                snapshot_weights[snapshot_index] * stickiness_multiplier
            )

    return evidence_scores


# ============================================================================
# STAGE 6  —  Dropout mixture: redistribute probability for disappearing games
# ============================================================================

def apply_dropout_mixture(
    evidence_scores:      dict[int, dict[int, float]],
    registry:             dict[int, GameRecord],
    total_snapshot_count: int,
    forecast_places:      int,
) -> dict[int, dict[int, float]]:
    """
    For every game that has EITHER:
        (a) disappeared from the latest snapshot, or
        (b) suffered a sudden rank drop >= DROP_DETECTION_THRESHOLD in its last step,

    redistribute DROPOUT_MASS_REDISTRIBUTION_FRACTION of its total evidence
    score to games currently holding the nearest-rank positions on each side.

    Models the empirical observation that when a game leaves the chart, the
    games at neighbouring ranks absorb its vacated audience and momentum.

    Returns a new score dict (the input is not mutated).
    """
    # Shallow-copy so we don't mutate the caller's data
    result: dict[int, dict[int, float]] = {
        uid: dict(place_score_map)
        for uid, place_score_map in evidence_scores.items()
    }

    for universe_id, game_record in registry.items():
        if universe_id not in result or not result[universe_id]:
            continue

        absent_count  = game_record.absent_trailing_snapshots(total_snapshot_count)
        is_sudden_drop = (
            len(game_record.chart_ranks) >= 2
            and (game_record.chart_ranks[-1] - game_record.chart_ranks[-2])
                >= DROP_DETECTION_THRESHOLD
            and absent_count == 0
        )

        if absent_count == 0 and not is_sudden_drop:
            continue   # healthy, present game — nothing to redistribute

        # Rank this game is most historically associated with
        peak_evidence_rank = max(result[universe_id], key=result[universe_id].get)
        game_total_evidence = sum(result[universe_id].values())
        mass_to_redistribute = game_total_evidence * DROPOUT_MASS_REDISTRIBUTION_FRACTION

        # Scale down the dropping game's own scores
        retained_fraction = 1.0 - DROPOUT_MASS_REDISTRIBUTION_FRACTION
        result[universe_id] = {
            place: score * retained_fraction
            for place, score in result[universe_id].items()
        }

        # Collect neighbour ranks within the forecast range
        neighbour_ranks: list[int] = []
        for rank_offset in range(1, DROPOUT_NEIGHBOUR_RANK_WINDOW + 1):
            rank_above = peak_evidence_rank - rank_offset
            rank_below = peak_evidence_rank + rank_offset
            if 1 <= rank_above <= forecast_places:
                neighbour_ranks.append(rank_above)
            if 1 <= rank_below <= forecast_places:
                neighbour_ranks.append(rank_below)

        if not neighbour_ranks:
            continue

        # Find which games currently hold each neighbour rank on the latest chart
        current_rank_holders: dict[int, list[int]] = defaultdict(list)
        for other_uid, other_game_record in registry.items():
            if (
                other_game_record.is_on_current_chart(total_snapshot_count)
                and other_game_record.latest_rank is not None
            ):
                current_rank_holders[other_game_record.latest_rank].append(other_uid)

        mass_per_neighbour_rank = mass_to_redistribute / len(neighbour_ranks)

        for neighbour_rank in neighbour_ranks:
            holders_at_rank = current_rank_holders.get(neighbour_rank, [])

            if holders_at_rank:
                mass_per_holder = mass_per_neighbour_rank / len(holders_at_rank)
                for holder_uid in holders_at_rank:
                    if holder_uid not in result:
                        result[holder_uid] = {}
                    result[holder_uid][neighbour_rank] = (
                        result[holder_uid].get(neighbour_rank, 0.0) + mass_per_holder
                    )
            else:
                # No known holder at this rank — keep mass with the dropping game
                result[universe_id][neighbour_rank] = (
                    result[universe_id].get(neighbour_rank, 0.0) + mass_per_neighbour_rank
                )

    return result


# ============================================================================
# STAGE 7  —  Velocity shift: move each game's probability mass forward 14 h
# ============================================================================

def shift_by_velocity(
    evidence_scores:          dict[int, dict[int, float]],
    velocity_by_uid:          dict[int, float],
    forecast_places:          int,
    velocity_blend_weight:    float = 0.55,
) -> dict[int, dict[int, float]]:
    """
    For each game, project its current distribution forward by its estimated
    velocity and blend the result with its historical distribution.

    velocity_blend_weight controls how much the velocity projection overrides
    history (0.0 = ignore velocity, 1.0 = use velocity only):

        blended_score[place] = (1 - velocity_blend_weight) * historical_prob[place]
                             + velocity_blend_weight        * velocity_projected_prob[place]

    The velocity-projected distribution is a Gaussian centred on the
    predicted future rank, with spread proportional to the displacement
    magnitude (larger predicted jump → wider, less certain distribution).

    Returns a new score dict.
    """
    result: dict[int, dict[int, float]] = {}

    for universe_id, place_score_map in evidence_scores.items():
        if not place_score_map:
            result[universe_id] = {}
            continue

        predicted_rank_displacement = velocity_by_uid.get(universe_id, 0.0)
        total_evidence_weight = sum(place_score_map.values())

        # Current weighted-average rank (centroid of the historical distribution)
        weighted_mean_rank = (
            sum(place * weight for place, weight in place_score_map.items())
            / total_evidence_weight
            if total_evidence_weight > 0
            else 5.0
        )

        predicted_future_rank = max(
            1.0,
            min(float(forecast_places), weighted_mean_rank + predicted_rank_displacement),
        )

        # Gaussian spread: larger displacement → wider, less certain distribution
        gaussian_spread = max(0.8, 0.5 + 0.15 * abs(predicted_rank_displacement))

        velocity_projected_dist: dict[int, float] = {}
        for place in range(1, forecast_places + 1):
            gaussian_weight = math.exp(
                -0.5 * ((place - predicted_future_rank) / gaussian_spread) ** 2
            )
            velocity_projected_dist[place] = gaussian_weight

        # Normalise the projected distribution to sum to 1.0
        projected_total = sum(velocity_projected_dist.values())
        if projected_total > 0:
            velocity_projected_dist = {
                place: weight / projected_total
                for place, weight in velocity_projected_dist.items()
            }

        # Normalise the historical distribution to sum to 1.0
        historical_dist: dict[int, float] = {}
        if total_evidence_weight > 0:
            historical_dist = {
                place: weight / total_evidence_weight
                for place, weight in place_score_map.items()
            }

        # Blend historical and velocity-projected distributions
        blended_dist: dict[int, float] = {}
        for place in range(1, forecast_places + 1):
            historical_prob = historical_dist.get(place, 0.0)
            velocity_prob   = velocity_projected_dist.get(place, 0.0)
            blended_dist[place] = (
                (1.0 - velocity_blend_weight) * historical_prob
                + velocity_blend_weight        * velocity_prob
            )

        # Re-scale back to original total weight (preserves relative game importance)
        blended_sum = sum(blended_dist.values())
        if blended_sum > 0:
            result[universe_id] = {
                place: score * total_evidence_weight / blended_sum
                for place, score in blended_dist.items()
            }
        else:
            result[universe_id] = dict(place_score_map)

    return result


# ============================================================================
# STAGE 8  —  Bottom-heavy prior: inject extra probability mass for lower ranks
# ============================================================================

def add_bottom_heavy_prior(
    evidence_scores:      dict[int, dict[int, float]],
    registry:             dict[int, GameRecord],
    total_snapshot_count: int,
    forecast_places:      int,
) -> dict[int, dict[int, float]]:
    """
    Add a small non-zero background weight to every (game, place) pair so that:

        1. Games outside the current chart can still win lower positions.
        2. Lower ranks are more "accessible", so the prior at rank r is
           PRIOR_WEIGHT_BASE + r * PRIOR_WEIGHT_RANK_SLOPE  (larger at lower ranks).

    The prior for absent games is further attenuated by an absence penalty:
        absence_penalty_factor = 2 ^ (-absent_count / 3)
    so games that have been off the chart for many snapshots contribute very little.

    Returns a new dict (the input is not mutated).
    """
    result: dict[int, dict[int, float]] = {
        uid: dict(place_score_map)
        for uid, place_score_map in evidence_scores.items()
    }

    for universe_id, game_record in registry.items():
        if universe_id not in result:
            result[universe_id] = {}

        absent_count = game_record.absent_trailing_snapshots(total_snapshot_count)
        absence_penalty_factor = (
            math.pow(2.0, -absent_count / 3.0)
            if absent_count > 0
            else 1.0
        )

        for place in range(1, forecast_places + 1):
            prior_weight = (
                (PRIOR_WEIGHT_BASE + place * PRIOR_WEIGHT_RANK_SLOPE)
                * absence_penalty_factor
            )
            result[universe_id][place] = (
                result[universe_id].get(place, 0.0) + prior_weight
            )

    return result


# ============================================================================
# STAGE 9  —  Bradley-Terry marginals: pairwise strength → per-place probability
# ============================================================================

def _bradley_terry_strength(evidence_score: float) -> float:
    """Convert a raw evidence score to a Bradley-Terry strength parameter."""
    return math.exp(evidence_score * BT_STRENGTH_SCALE)


def bradley_terry_marginals(
    evidence_scores: dict[int, dict[int, float]],
    forecast_places: int,
) -> dict[int, dict[int, float]]:
    """
    For each rank position, estimate the probability that each candidate game
    occupies it using Bradley-Terry pairwise win probabilities.

    For place r with candidate games g₁ … gₙ:
        P(gᵢ wins place r) ≈ strength(gᵢ) / Σⱼ strength(gⱼ)

    Only the top BT_CANDIDATE_POOL_SIZE games by evidence score are compared.

    Returns place_probabilities[place][universe_id] = raw probability
    (before the final normalisation pass).
    """
    # Invert uid → place → score  to  place → uid → score
    scores_by_place: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))

    for universe_id, place_score_map in evidence_scores.items():
        for place, score in place_score_map.items():
            if 1 <= place <= forecast_places and score > 0:
                scores_by_place[place][universe_id] += score

    place_probabilities: dict[int, dict[int, float]] = {}

    for place in range(1, forecast_places + 1):
        candidate_scores = scores_by_place.get(place, {})

        if not candidate_scores:
            place_probabilities[place] = {}
            continue

        # Keep only the strongest BT_CANDIDATE_POOL_SIZE candidates
        ranked_candidates = sorted(
            candidate_scores.items(), key=lambda uid_score: -uid_score[1]
        )[:BT_CANDIDATE_POOL_SIZE]

        bradley_terry_strengths = {
            uid: _bradley_terry_strength(score) for uid, score in ranked_candidates
        }
        sum_bt_strengths = sum(bradley_terry_strengths.values())

        if sum_bt_strengths == 0.0:
            # Degenerate fallback: uniform distribution
            num_candidates = len(ranked_candidates)
            place_probabilities[place] = {
                uid: 1.0 / num_candidates for uid, _ in ranked_candidates
            }
        else:
            place_probabilities[place] = {
                uid: strength / sum_bt_strengths
                for uid, strength in bradley_terry_strengths.items()
            }

    return place_probabilities


# ============================================================================
# STAGE 9b  —  Normalisation: guarantee each place sums to exactly 1.0
# ============================================================================

def normalize_place_distributions(
    place_probabilities: dict[int, dict[int, float]],
    forecast_places:     int,
    residual_tolerance:  float = 1e-9,
) -> dict[int, dict[int, float]]:
    """
    Ensure that for every rank position, the probabilities across all candidate
    games sum to *exactly* 1.0.

    Why this step is needed
    -----------------------
    The Bradley-Terry step divides each game's strength by the pool's total.
    Floating-point arithmetic means those divisions rarely produce a sum of
    exactly 1.0.  The error is usually < 1e-14, but can be larger when:

      * the candidate pool was capped at BT_CANDIDATE_POOL_SIZE (games outside
        the cap contribute zero, so the remaining probabilities sum to < 1),
      * the bottom-heavy prior added weight outside the top-candidate window,
      * the dropout redistribution sent mass to games outside the BT pool.

    Three-pass correction algorithm
    --------------------------------
    Pass 1 — Clamp negatives
        Any negative probability (from floating-point cancellation) is clamped
        to 0.0 before the total is computed.

    Pass 2 — Renormalise
        Every value is divided by the clamped total so the distribution is
        proportionally correct.

    Pass 3 — Residual correction
        math.fsum gives the exact post-division sum.  The tiny residual
        (1.0 − exact_sum) is added to the highest-probability entry so the
        invariant holds to machine precision.  This changes the last few
        mantissa bits of the dominant entry and is invisible in practice.

    Parameters
    ----------
    place_probabilities : place → {universe_id: probability}
    forecast_places     : number of rank positions to normalise
    residual_tolerance  : skip residual correction when |1 − sum| < this value

    Returns
    -------
    A new dict with the same structure; the input is not mutated.

    Raises
    ------
    ValueError : if a place has candidates but all probabilities are zero
                 or negative after clamping — indicates an upstream logic error.
    """
    normalised_distributions: dict[int, dict[int, float]] = {}

    for place in range(1, forecast_places + 1):
        unnormalised_dist = place_probabilities.get(place, {})

        if not unnormalised_dist:
            normalised_distributions[place] = {}
            continue

        # Pass 1 — clamp any negative floating-point artefacts to zero
        nonnegative_probs: dict[int, float] = {
            uid: max(0.0, probability)
            for uid, probability in unnormalised_dist.items()
        }

        clamped_total = sum(nonnegative_probs.values())

        if clamped_total == 0.0:
            raise ValueError(
                f"Place {place} has {len(nonnegative_probs)} candidate(s) but all "
                "probabilities are zero or negative after clamping.  "
                "This indicates a logic error in an upstream pipeline stage."
            )

        # Pass 2 — divide by total to produce a proportionally correct distribution
        renormalised_probs: dict[int, float] = {
            uid: probability / clamped_total
            for uid, probability in nonnegative_probs.items()
        }

        # Pass 3 — correct the remaining sub-epsilon residual using exact summation
        renormed_sum = math.fsum(renormalised_probs.values())
        normalisation_residual = 1.0 - renormed_sum

        if abs(normalisation_residual) > residual_tolerance:
            # Apply residual correction to the dominant (highest-probability) game
            dominant_uid = max(renormalised_probs, key=renormalised_probs.__getitem__)
            renormalised_probs[dominant_uid] += normalisation_residual

        normalised_distributions[place] = renormalised_probs

    return normalised_distributions


def verify_normalization(
    place_probabilities: dict[int, dict[int, float]],
    forecast_places:     int,
    allowed_deviation:   float = 1e-12,
) -> list[str]:
    """
    Audit every place and return warning strings for any distribution that:
      - sums to outside [1 - allowed_deviation, 1 + allowed_deviation], or
      - contains a negative probability, or
      - contains a probability exceeding 1.0.

    Returns an empty list when all places pass.  Intended to be called
    after normalize_place_distributions() as a belt-and-suspenders check.

    Parameters
    ----------
    place_probabilities : place → {universe_id: probability}
    forecast_places     : number of rank positions to audit
    allowed_deviation   : maximum permitted |sum - 1.0|  (default: 1e-12)

    Returns
    -------
    List of human-readable warning strings; empty means all clear.
    """
    audit_warnings: list[str] = []

    for place in range(1, forecast_places + 1):
        place_probs = place_probabilities.get(place, {})
        if not place_probs:
            continue   # empty places (no candidates) are acceptable

        prob_sum     = math.fsum(place_probs.values())
        sum_deviation = abs(prob_sum - 1.0)

        if sum_deviation > allowed_deviation:
            audit_warnings.append(
                f"place={place}: probabilities sum to {prob_sum:.17f} "
                f"(deviation={sum_deviation:.2e}, {len(place_probs)} candidates)"
            )

        for universe_id, probability in place_probs.items():
            if probability < 0.0:
                audit_warnings.append(
                    f"place={place} uid={universe_id}: "
                    f"negative probability {probability:.6e}"
                )
            elif probability > 1.0 + allowed_deviation:
                audit_warnings.append(
                    f"place={place} uid={universe_id}: "
                    f"probability {probability:.6e} exceeds 1.0"
                )

    return audit_warnings


# ============================================================================
# STAGE 10  —  Rank top-N: derive an ordered forecast list
# ============================================================================

def rank_top_n(
    place_probabilities: dict[int, dict[int, float]],
    registry:            dict[int, GameRecord],
    top_n:               int,
    forecast_places:     int,
) -> list[int]:
    """
    Collapse the full probability matrix into a single ranked list of universe IDs.

    Composite score per game:
        composite_score(game) = Σ_place  probability(game | place) × (1 / place^1.5)

    The 1/place^1.5 exponent makes top-rank probability worth much more than
    lower-rank probability, reflecting that #1 is rarer and more valuable.

    Ties are broken by the game's most-recent observed rank (lower number = better).

    If fewer than top_n games appear in any distribution, the list is padded
    with the remaining games ordered by most-recent chart rank.
    """
    composite_scores: dict[int, float] = defaultdict(float)

    for place, uid_probs in place_probabilities.items():
        rank_position_weight = 1.0 / (place ** 1.5)
        for universe_id, probability in uid_probs.items():
            composite_scores[universe_id] += probability * rank_position_weight

    def composite_sort_key(universe_id: int) -> tuple[float, int]:
        latest_rank_as_tiebreak = (
            registry[universe_id].latest_rank
            if universe_id in registry and registry[universe_id].latest_rank
            else 9999
        )
        return (-composite_scores[universe_id], latest_rank_as_tiebreak)

    ranked_uid_list = sorted(composite_scores.keys(), key=composite_sort_key)[:top_n]

    # Pad with fallback games if fewer than top_n made it into the distributions
    if len(ranked_uid_list) < top_n:
        already_ranked_uids = set(ranked_uid_list)
        fallback_records = sorted(
            [
                game_record
                for uid, game_record in registry.items()
                if uid not in already_ranked_uids and game_record.latest_rank is not None
            ],
            key=lambda game_record: game_record.latest_rank,
        )
        for game_record in fallback_records:
            ranked_uid_list.append(game_record.universe_id)
            if len(ranked_uid_list) >= top_n:
                break

    return ranked_uid_list[:top_n]


# ============================================================================
# STAGE 11  —  Assemble and write forecast.json
# ============================================================================

def build_forecast_payload(
    place_probabilities: dict[int, dict[int, float]],
    top_uid_list:        list[int],
    registry:            dict[int, GameRecord],
    total_snapshot_count: int,
    forecast_places:     int,
    velocity_by_uid:     dict[int, float],
) -> dict:
    """
    Assemble the final JSON-serialisable forecast dict.

    Output schema
    -------------
    {
      "probabilities": [
        {
          "universe_id": int,
          "place": int,
          "p": float,                 // guaranteed: 0 < p <= 1; sum per place = 1.0
          "name": str,
          "velocity_rank_delta": float  // predicted rank displacement over 14 h
        }, ...
      ],
      "top10": [int, ...],
      "top10_named": [{"universe_id": int, "name": str}, ...],
      "meta": { ... }
    }
    """
    probability_rows = []
    for place in range(1, forecast_places + 1):
        uid_probs = place_probabilities.get(place, {})
        for universe_id, probability in sorted(
            uid_probs.items(), key=lambda uid_prob: -uid_prob[1]
        ):
            prob_entry: dict = {
                "universe_id": universe_id,
                "place": place,
                "p": round(probability, 7),
            }
            if universe_id in registry:
                if registry[universe_id].display_name:
                    prob_entry["name"] = registry[universe_id].display_name
                rank_displacement = velocity_by_uid.get(universe_id)
                if rank_displacement is not None:
                    prob_entry["velocity_rank_delta"] = round(rank_displacement, 3)
            probability_rows.append(prob_entry)

    top_named_list = [
        {
            "universe_id": uid,
            "name": registry[uid].display_name if uid in registry else "",
        }
        for uid in top_uid_list
    ]

    return {
        "probabilities": probability_rows,
        "top10": top_uid_list,
        "top10_named": top_named_list,
        "meta": {
            "strategy": "sticky-dropout-velocity-bt",
            "snapshots_used": total_snapshot_count,
            "recency_boost_hours": RECENCY_BOOST_HOURS,
            "recency_boost_factor": RECENCY_BOOST_FACTOR,
            "recency_halflife_hours": round(RECENCY_HALFLIFE_HOURS, 6),
            "velocity_horizon_hours": VELOCITY_HORIZON_HOURS,
            "velocity_regression_window": VELOCITY_REGRESSION_WINDOW,
            "dropout_mass_redistribution_fraction": DROPOUT_MASS_REDISTRIBUTION_FRACTION,
            "dropout_neighbour_rank_window": DROPOUT_NEIGHBOUR_RANK_WINDOW,
            "stickiness_weight_base": STICKINESS_WEIGHT_BASE,
            "stickiness_weight_per_rank_decay": STICKINESS_WEIGHT_PER_RANK_DECAY,
            "bt_strength_scale": BT_STRENGTH_SCALE,
            "bt_candidate_pool_size": BT_CANDIDATE_POOL_SIZE,
            "forecast_places": forecast_places,
            "top_n": len(top_uid_list),
            "normalization": "three-pass (clamp, renorm, residual-correct)",
        },
    }


def write_forecast(payload: dict, output_path: Path) -> None:
    """Write the forecast payload to disk as indented UTF-8 JSON."""
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    print(f"Forecast written → {output_path}")


# ============================================================================
# DIAGNOSTIC HELPERS
# ============================================================================

def _print_top_games_summary(
    place_probabilities: dict[int, dict[int, float]],
    registry:            dict[int, GameRecord],
    velocity_by_uid:     dict[int, float],
    top_n:               int = 10,
) -> None:
    """
    Print a human-readable per-place summary table.
    Set environment variable FORECAST_VERBOSE=1 to enable.
    """
    print("\n--- TOP FORECASTED GAMES BY PLACE ---")
    for place in range(1, top_n + 1):
        uid_probs = place_probabilities.get(place, {})
        if not uid_probs:
            print(f"  Place {place:>2}: (no candidates)")
            continue
        leading_uid, leading_prob = max(uid_probs.items(), key=lambda uid_prob: uid_prob[1])
        game_name        = (
            registry[leading_uid].display_name
            if leading_uid in registry else str(leading_uid)
        )
        rank_displacement = velocity_by_uid.get(leading_uid, 0.0)
        velocity_label   = (
            f"Δ{rank_displacement:+.1f}" if abs(rank_displacement) >= 0.1 else "stable"
        )
        prob_sum = math.fsum(uid_probs.values())
        print(
            f"  Place {place:>2}: {game_name:<40s}  "
            f"p={leading_prob:.4f}  velocity={velocity_label}  Σp={prob_sum:.15f}"
        )
    print()


# ============================================================================
# MAIN  —  orchestration
# ============================================================================

def main() -> None:
    """
    Full sticky-dropout-velocity pipeline:

        load → parse → weight → velocity →
        sticky score → dropout mixture → velocity shift →
        bottom-heavy prior → Bradley-Terry marginals →
        NORMALISE + VERIFY → rank → write
    """
    input_dir  = Path(os.environ.get("MAKEFUN_INPUT_DIR",  "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    show_verbose_summary = os.environ.get("FORECAST_VERBOSE", "0") == "1"

    task_config     = load_task_config(input_dir / "task.json")
    forecast_places = int(task_config.get("forecast_places", DEFAULT_FORECAST_PLACES))
    top_n           = int(task_config.get("top_n",           DEFAULT_TOP_N))

    print(f"Config: forecast_places={forecast_places}, top_n={top_n}")

    # Stage 1 — load history
    snapshots = load_snapshots(input_dir / "history.ndjson")
    print(f"Loaded {len(snapshots)} snapshots")

    # Stage 2 — build per-game registry
    registry = build_game_registry(snapshots)
    print(f"Registry: {len(registry)} unique games")

    # Stage 3 — compute per-snapshot recency weights
    snapshot_weights = compute_snapshot_weights(snapshots)

    # Stage 4 — estimate 14-hour rank-velocity per game
    print("Computing 14h rank-velocity …")
    velocity_by_uid = estimate_velocity_14h(registry, snapshots)

    # Stage 5 — accumulate sticky historical evidence scores
    print("Computing sticky scores …")
    evidence_scores = compute_sticky_scores(
        registry, snapshot_weights, snapshots, forecast_places
    )

    # Stage 6 — redistribute mass from dropping/absent games to neighbours
    print("Applying dropout mixture …")
    evidence_scores = apply_dropout_mixture(
        evidence_scores, registry, len(snapshots), forecast_places
    )

    # Stage 7 — shift distributions forward by velocity
    print("Shifting distributions by velocity …")
    evidence_scores = shift_by_velocity(evidence_scores, velocity_by_uid, forecast_places)

    # Stage 8 — add bottom-heavy background prior
    evidence_scores = add_bottom_heavy_prior(
        evidence_scores, registry, len(snapshots), forecast_places
    )

    # Stage 9 — compute Bradley-Terry marginal probabilities per place
    print("Computing Bradley-Terry marginals …")
    place_probabilities = bradley_terry_marginals(evidence_scores, forecast_places)

    # Stage 9b — guarantee Σp = 1.0 for every place
    print("Normalising place distributions …")
    place_probabilities = normalize_place_distributions(place_probabilities, forecast_places)

    # Verification pass — catches any residual inconsistency before writing
    normalisation_warnings = verify_normalization(place_probabilities, forecast_places)
    if normalisation_warnings:
        print(f"  [warn] Normalisation audit found {len(normalisation_warnings)} issue(s):")
        for warning_text in normalisation_warnings:
            print(f"    {warning_text}")
    else:
        print(f"  Normalisation OK — all {forecast_places} places sum to 1.0 ✓")

    # Stage 10 — derive ordered top-N list
    top_uid_list = rank_top_n(place_probabilities, registry, top_n, forecast_places)

    if show_verbose_summary:
        _print_top_games_summary(place_probabilities, registry, velocity_by_uid, top_n)

    # Stage 11 — assemble and write output
    payload = build_forecast_payload(
        place_probabilities,
        top_uid_list,
        registry,
        len(snapshots),
        forecast_places,
        velocity_by_uid,
    )
    write_forecast(payload, output_dir / "forecast.json")

    total_probability_entries = len(payload["probabilities"])
    print(
        f"Output: {total_probability_entries} probability entries "
        f"across {forecast_places} places"
    )
    print(
        f"Top-{top_n}: "
        f"{[row['name'] or str(row['universe_id']) for row in payload['top10_named']]}"
    )


if __name__ == "__main__":
    main()
