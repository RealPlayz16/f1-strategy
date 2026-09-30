# HANDOFF: F1 Race Strategy Optimizer (Insane Version)
Session 1 of 6 | Started 1:40pm ET Sep 30 | Deadline Oct 1 11:59pm ET
Environment: Windows 11, PowerShell, venv at .venv, Python 3.11

## Project summary
A lap-by-lap F1 race strategy engine built on FastF1 data. It predicts tire
degradation with uncertainty, simulates all 20 cars (gaps, traffic, overtakes,
undercuts), runs Monte Carlo over Safety Car/VSC and rival strategies, and
re-optimizes live during a replayed race, using Fast Flag's Safety Car calls as
triggers. Delivered with a pit-wall dashboard and an Arduino "BOX" pit board.

## Session roadmap
1. Data foundation (IN PROGRESS)
2. Tire degradation model with uncertainty (LightGBM + PyTorch quantile)
3. Overtaking model + race rules
4. 20-car race engine + DP optimizer + Monte Carlo
5. Live replay strategy engine + Fast Flag hook + backtest
6. Pit-wall dashboard + Arduino pit board + README/demo

## Working style
- Minimal explanation, direct bullets, step-by-step commands
- No em-dashes
- Edit files directly; run pytest and ruff yourself, do not ask me to paste output
- Do not patch symptoms. If a number looks wrong, find the cause before changing code
- State assumptions before writing code when a formula or model choice is involved

## Stack and conventions
- Python 3.11, fastf1, pandas, pyarrow, numpy, pyyaml, pytest, ruff
- Later sessions: scikit-learn, lightgbm, torch, fastapi, uvicorn
- All times stored as float seconds; all intermediate data saved as parquet
- FastF1 cache at data/cache/ (gitignored); outputs at data/processed/
- fastf1.set_log_level('WARNING'); random seed 42 everywhere
- Every module runnable as a CLI: python -m src.<module>
- ruff line-length 100, select E F I UP B
- PowerShell writes BOMs with Set-Content -Encoding utf8. Use
  [System.IO.File]::WriteAllText to avoid breaking the TOML parser
- Holdout races (2025 Japanese, Dutch, Singapore, Abu Dhabi) are excluded from
  every fit: phi, pit loss medians, SC rates, tire models

## Current state

### Done and verified
- Repo scaffold, requirements.txt, .gitignore, pyproject.toml, .github/workflows/ci.yml
- config/races.yaml: 22 train + 4 holdout races, 2023-2025
- src/ingest.py: 26 races ingested clean, per-race laps/weather/rcm/meta parquet
  under data/processed/raw/, plus data/processed/ingest_manifest.csv.
  Wet-compound auto-exclusion verified against 2024 British GP (INTERMEDIATE).
- src/clean.py: laps_clean.parquet, 29818 rows, 87.9% clean, 26 races.
  Median clean lap by compound SOFT 83.6 < MEDIUM 84.8 < HARD 86.7, correct ordering.
  Carries pit_in_time_s and pit_out_time_s.
- src/pitloss.py: transit-based model built and run. Result: transit-window phi is
  about 0 at every track (see "Pit loss model" below). Being replaced by phi fitted
  from the SC discount.
- tests/test_clean.py, tests/test_pitloss.py: 12 tests passing
- ruff clean (smoke.py had a BOM and an import-sort error, fixed)

### In progress
- Pit loss: fit phi from the SC discount, apply to all tracks, key by
  (event, season), exclude holdout. Plan under "Pit loss model".

### Not started
- src/safety_car.py
- src/overtakes.py
- Fast Flag recon (clone github.com/Pseudocoder28/Fast-Flag, document how SC/VSC
  recommendations are emitted: format, timing, transport). Credit Naman in README.
  Hard dependency for Session 5, do early.
- Sanity plots (lap time vs tire age per compound; pit loss per track)
- README.md

## Pit loss model, and why

pit_loss = transit - phi * L

- transit = time spent in the pit lane between the pit entry and pit exit lines.
- phi = fraction of a lap, in time, of the track section between pit entry and
  pit exit. A per-track geometric constant.
- L = the lap time the car would have run had it stayed out, under the conditions
  in force. Under SC this is roughly 1.4x green, so phi * L is larger and the
  loss is smaller. The SC discount falls out of the physics.

Equivalent form used for SC / VSC, needing only phi and a measured green loss:
  loss_cond = green_loss - phi * (L_cond - L_green)

### Finding: FastF1 PitInTime is not the pit entry line
Measured in Session 1 from sector session times on green stops:
- At 10 of 13 tracks the car has already spent more than a full normal S3 when
  PitInTime fires, by 0.3 to 3.1 s. PitInTime fires about 1.8 s (median) before
  the car crosses the timing line inside the pit lane.
- The [PitInTime, PitOutTime] window therefore spans only about 0 to 3 s of track,
  and the entry-road slowdown before PitInTime cancels it. Fitted
  phi = 2 - (lap_sum - transit) / L comes out -0.027 to 0.028 at 12 of 13 tracks.
  Silverstone, where PitInTime fires early (6.4 s of S3 left), is the one track
  with phi 0.091.
- At green, transit (23.09 s median) equals lap-sum loss (23.13 s). With phi about 0
  the transit model gives SC loss equal to green loss, so no SC discount.
- Telemetry to locate the true pit lines was considered and declined.

### Current plan: phi from the SC discount
- D = measured loss = in_lap + out_lap - 2 * L, with L trusted
- phi_implied = (D_green - D_sc) / (L_sc - L_green), per (event, season)
- Fit only on train races with a real stay-out sample under SC. Exclude Austria
  (discount has the wrong sign) and damage repairs (2023 Japan ALB, BOT, ZHO lap 1;
  2024 US ALB lap 3)
- Pool across tracks, report the spread. Expect about 0.07. If the pooled value is
  outside 0.04 to 0.12, stop and report rather than fitting harder
- Apply pooled phi to every track. Green loss stays as measured. No clamps.

### Rejected approach, do not reintroduce
The original formula was (in_lap + out_lap) - 2 * reference_clean_lap, with the
reference interpolated from the field when the field mass-pitted. It produced an
84 s SC pit loss at 2023 Austria. Attempted fixes (minimum running cars,
status-aware interpolation, mass-pit exclusion, and a clamp forcing sc_s < green_s)
each patched the symptom. The clamp in particular was wrong and must not come back.

Root cause found in Session 1: 2023 Austria lap 2 was not a mass pit. Race control:
"SAFETY CAR THROUGH THE PIT LANE". 19 of 20 cars drove through without changing
tyres; only MAG stopped. The old model treated drive-throughs as stops.

## Stop definition (src/pitloss.py)
- In-lap with PitInTime followed by an out-lap with PitOutTime
- Tyres must change: compound changes or TyreLife does not simply continue.
  FastF1 starts a new Stint on any pit lane passage, so Stint alone is not a stop.
  Removes 25 passages: Austria 2023 lap 2 (19) plus penalties and repairs (6)
- Red flag on the in-lap or out-lap: skipped
- Transit longer than the stay-out lap: garage visit, not a stop
  (2023 Japan PER lap 13, 41 minutes)
- Missing LapTime on the in-lap or out-lap: stop kept (common on SC laps, e.g.
  2024 Saudi lap 7, 2025 Bahrain lap 32), lap_sum_s is NaN

## Output schemas

laps_clean.parquet
  season, round, event, driver, team, lap, lap_time_s, sector1_s, sector2_s,
  sector3_s, compound, tyre_life, stint, fresh_tyre, position, session_time_s,
  pit_in_time_s, pit_out_time_s, gap_ahead_s, track_status, is_pit_in,
  is_pit_out, is_sc, is_vsc, is_yellow, is_red, is_lap1, is_deleted,
  is_accurate, is_outlier, is_clean, track_temp, air_temp, laps_remaining

pit_stops.parquet
  season, round, event, driver, lap, compound_in, compound_out, condition,
  transit_s, ref_lap_s, lap_sum_s, pace_trusted, phi, pit_loss_s

pitloss_by_track.parquet (being rekeyed to event, season)
  event, green_s, sc_s, vsc_s, transit_s, phi, n_stops, n_green, n_sc, n_vsc

pit_phi.parquet (being replaced by the SC-discount fit)
  event, phi, n_fit, phi_season_spread

sc_rates.parquet (not built yet)
  event, race_laps, sc_deployments, vsc_deployments, p_sc_per_lap, p_vsc_per_lap

overtakes.parquet (not built yet)
  season, round, event, lap, overtaker, overtaken, pace_delta_s, tyre_age_delta,
  compound_pair, gap_before_s, drs_likely (gap < 1.0 s)

## Cleaning rules
- Drop from is_clean: pit in/out laps, lap 1, SC/VSC/red flag laps, deleted laps,
  laps with IsAccurate == False
- Outliers: more than 107% of that driver's stint median
- Track status is a concatenation of single digit codes, e.g. "14" is green + SC.
  1 green, 2 yellow, 4 SC, 5 red, 6 VSC, 7 VSC ending. Must be parsed as a
  string containing each digit, never cast to int.
- Fuel correction is NOT applied in Session 1 (fitted from data in Session 2);
  laps_remaining is kept for it

## Data notes
- 2023 Monza and 2024 Bahrain used only two compounds. Expected, not a bug.
- Ergast warnings on session load are harmless and expected for recent sessions
- 2023 Austria lap 2: SC led the field through the pit lane. Not a mass pit stop.
- 2025 Dutch: pit lane limit raised to 80 km/h. Transit 17.8 s vs 21.2 s in 2024,
  green loss 19.9 s vs 23.2 s. Pit loss must be keyed by (event, season).
- 2023 Japan and 2024 US: lap 1-3 SC stops include damage repairs (40 to 56 s transit)
- Seasons 2023-2025 only. 2026 excluded: new regulations change tire behaviour.

## Decisions log
- Timedelta columns converted to float seconds at ingest, before parquet write,
  to avoid duration round-trip issues on Windows
- meta.parquet per race carries race_laps, needed for laps_remaining and sc_rates
- Pit loss switched from lap-time subtraction to transit-based measurement
- Transit-window phi found to be about 0 (FastF1 PitInTime fires inside the pit lane).
  phi to be fitted from the SC discount instead. Telemetry route declined.
- Holdout races excluded from all fits

## Next session (2) preview
Tire model: per-race baseline fit, cross-race LightGBM, PyTorch quantile model
(p10/p50/p90), cliff detection, held-out MAE and calibration.
Interface target: predict_laptime(features) -> (p10, p50, p90)
