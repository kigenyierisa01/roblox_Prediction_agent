# Roblox Trending Forecast — make.fun Competition Entry

A Python forecasting agent that reads hourly Roblox Top Trending chart history and predicts which games will hold each of the top 10 places over the next 24 hours.

## Strategy

- Reads all historical snapshots from `history.ndjson`
- Applies exponential decay weighting — recent hours count more than older ones
- Calculates each game's trend velocity (rising or falling)
- For each place, assigns honest probabilities based on each game's recent history at that position
- Outputs a deterministic forecast for every hour in the 24-hour window

## Requirements

- Python 3.13
- Standard library only (no pip installs)
- No network access needed — runs fully offline

## Environment Variables

| Variable | Description |
|---|---|
| `MAKEFUN_INPUT_DIR` | Path to input directory (default: `/input`) |
| `MAKEFUN_OUTPUT_DIR` | Path to output directory (default: `/output`) |

## Usage

```bash
python main.py
```

## Output

Writes `/output/forecast.json` with probabilities for each game at each of the top 10 places, for every hour in the forecast window.

## License

MIT
