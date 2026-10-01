# F1 Race Strategy Optimizer

A lap-by-lap Formula 1 race strategy engine built on FastF1 data. It predicts tyre
degradation with uncertainty, simulates all 20 cars (gaps, traffic, overtakes, undercuts),
runs Monte Carlo over Safety Car / VSC timing and rival strategies, and re-optimizes live
during a replayed race, using [Fast Flag](https://github.com/Pseudocoder28/Fast-Flag)'s
Safety Car calls as early triggers.

Status: Session 1 of 6 (data foundation) done. See [HANDOFF.md](HANDOFF.md) for the full
state, decisions and findings.

## Roadmap

1. Data foundation: ingest, cleaning, pit loss, Safety Car rates, overtakes (done)
2. Tyre degradation model with uncertainty (LightGBM + PyTorch quantile, p10/p50/p90)
3. Overtaking model and race rules
4. 20-car race engine, DP optimizer, Monte Carlo
5. Live replay strategy engine with the Fast Flag hook, backtest on holdout races
6. Pit-wall dashboard, Arduino "BOX" pit board, demo

## Data

27 dry races from 2023 to 2025 ([config/races.yaml](config/races.yaml)). Wet races are
excluded automatically.

- **Train:** 23 races.
- **Holdout:** 2025 Japanese, United States, Singapore and Abu Dhabi. The holdout is
  never used by any fit, and none of these races is in Fast Flag's training set.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.ingest --set all
python -m src.clean
python -m src.pitloss
python -m src.safety_car
python -m src.overtakes
python -m src.plots
pytest -q
```

The first ingest downloads race data into `data/cache/`. Outputs land in
`data/processed/` as parquet, and plots in `figures/`.

## Pipeline

| Module | Output | What it does |
|---|---|---|
| `src/ingest.py` | `raw/<race>/*.parquet` | Laps, weather, race control, track and session status per race |
| `src/clean.py` | `laps_clean.parquet` | Flags clean laps (no pit, lap 1, SC/VSC/red, deleted, inaccurate, 107% outliers) |
| `src/pitloss.py` | `pit_stops.parquet`, `pitloss_by_track.parquet` | Green pit loss measured per race; SC/VSC loss from a stated phi |
| `src/safety_car.py` | `sc_events.parquet`, `sc_rates.parquet` | SC/VSC/red episodes, per-lap deployment rates per track |
| `src/overtakes.py` | `overtakes.parquet`, `battles.parquet` | On-track passes and close battles with gap, pace and tyre deltas |
| `src/plots.py` | `figures/*.png` | Sanity plots |

## Findings that shaped the model

- **Pit loss under Safety Car can't be measured from lap times.** On laps where the SC or
  VSC starts or ends, lap times of cars that stayed out spread 22 to 31 s, tracking
  running position. So the SC discount uses a stated phi (0.08, swept 0.05 to 0.12 in the
  Monte Carlo), to be validated against observed position loss in Session 4.
- **FastF1 `PitInTime` fires inside the pit lane, not at the entry line**, so pit lane
  "transit" can't stand in for the geometry.
- **2023 Austria lap 2 was the Safety Car leading the field through the pit lane, not 20
  pit stops.** A stop now requires a tyre change.
- **Zandvoort raised its pit lane limit to 80 km/h in 2025.** Green pit loss is 18.1 s,
  against 23.2 s in 2024, so pit loss is keyed by season and event.
- **Race control message times are wall clock.** Timed events come from the track status
  feed, which is on session time: the same clock as lap data and Fast Flag.

## Credits

- [Fast Flag](https://github.com/Pseudocoder28/Fast-Flag), the AI race control assistant
  whose Safety Car and VSC recommendations trigger re-optimization, was built mainly by
  **Naman** (team We Are So Back, FormulaTech Hacks 2026). How we consume it:
  [docs/fast_flag_recon.md](docs/fast_flag_recon.md).
- Race data from [FastF1](https://github.com/theOehrly/Fast-F1).

Everything is a replay of historical data.
