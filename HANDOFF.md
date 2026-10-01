# HANDOFF: F1 Race Strategy Optimizer (Insane Version)
Session 1 of 6 done | Started 1:40pm ET Sep 30 | Deadline Oct 1 11:59pm ET
Environment: Windows 11, PowerShell, venv at .venv, Python 3.11

## Project summary
A lap-by-lap F1 race strategy engine built on FastF1 data. It predicts tire
degradation with uncertainty, simulates all 20 cars (gaps, traffic, overtakes,
undercuts), runs Monte Carlo over Safety Car/VSC and rival strategies, and
re-optimizes live during a replayed race, using Fast Flag's Safety Car calls as
triggers. Delivered with a pit-wall dashboard and an Arduino "BOX" pit board.

## Session roadmap
1. Data foundation (DONE)
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
- config/races.yaml: 23 train + 4 holdout races, 2023-2025. Holdout: 2025 Japanese,
  United States, Singapore, Abu Dhabi. 2025 Dutch moved to train in Session 1 because
  Fast Flag trained on it; 2025 United States replaced it (dry, VSC lap 7, not in Fast
  Flag's training or check races).
- src/splits.py: split per race from config/races.yaml, the single source of truth.
  meta.parquet also stores a split, written at ingest, which goes stale. Do not use it.
- src/ingest.py: 27 races ingested clean, per-race laps/weather/rcm/meta parquet
  under data/processed/raw/, plus data/processed/ingest_manifest.csv.
  Wet-compound auto-exclusion verified against 2024 British GP (INTERMEDIATE).
- src/clean.py: laps_clean.parquet, 30885 rows, 87.9% clean, 27 races, carries split,
  pit_in_time_s and pit_out_time_s.
  Compound pace ordering is NOT verified in Session 1. The earlier "SOFT < MEDIUM < HARD"
  check pooled lap times across tracks and flipped when 2025 Austin was added. Within
  race and uncorrected for fuel, HARD is 0.38 s faster than MEDIUM (it runs late on low
  fuel). A race + fuel + tyre-age regression gives SOFT +0.11, HARD +0.10 vs MEDIUM,
  fuel 0.051 s per lap remaining. Session 2 separates compound pace properly.
- src/pitloss.py: green loss measured per (season, event), SC / VSC from stated phi.
  Train only. See "Pit loss model".
- docs/fast_flag_recon.md: rec schema, time base, transport, lead times, overlap
- ruff clean, pytest passing

- src/safety_car.py: SC / VSC / RED events from the track status feed (counts match race
  control in all 27 races) and train-only deployment rates
- src/overtakes.py: passes and close battles from lap-end order
- src/plots.py: figures/laptime_vs_tyre_age.png and figures/pitloss_by_race.png
  (figures/ is gitignored, regenerate with python -m src.plots). Degradation about
  0.04 to 0.05 s per lap on every compound after a 0.05 s/lap fuel adjustment. Soft laps
  at tyre age 2 to 4 sit above the soft trend, likely race laps 2 to 4 in DRS trains:
  check in Session 2.
- README.md (credits Naman for Fast Flag)

Session 1 complete. Next: Session 2.

## Session 4 task: validate PHI against position loss
PHI = 0.08 is a stated assumption, not a measurement. In Session 4, validate it against
the positions actually lost when stopping under SC in train races (race engine replay of
those stops vs observed running order after the stop), not lap-time seconds. Lap times
cannot identify SC pit loss (see below). Sweep PHI over 0.05 to 0.12 in the Monte Carlo.

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

### Finding: SC pit loss is not identifiable from lap times with this sample
Tried phi_implied = (D_green - D_sc) / (L_sc - L_green), D = in_lap + out_lap - 2 * L,
on train races with a stay-out sample under SC:
- British 2023 0.035 (8 stops), Saudi 2023 0.319 (7), Bahrain 2025 0.179 (1),
  Spain 2025 1.459 (1). Pooled 0.249, outside the 0.04 to 0.12 guard. Stopped there.
- Cause: on laps where SC or VSC is deployed or ends, stay-out lap times spread 22 to
  31 s across the field with rank correlation about 0.97 with running position (green
  laps: 3 s). The stay-out median is not any one car's counterfactual.
- Only 3 train stops have both laps fully under SC with lap times and a stay-out sample.
- The earlier "British and Dutch near 0.07" came from a model table (British had the
  transit-window phi built in; Dutch pooled 2024 green at 60 km/h with 2025 SC at 80).

### Current model (src/pitloss.py)
- green_s: median over trusted green stops of in_lap + out_lap - 2 * L_green, per
  (season, event), train races only. Measured.
- PHI = 0.08 stated; Monte Carlo sweeps PHI_RANGE = (0.05, 0.12).
- L_cond = L_green * r_cond. r_cond = median over train races of stay-out lap on laps
  entirely under the condition (status exactly "4" or "6") over that race's median clean
  lap: SC 1.426, VSC 1.326.
- sc_s, vsc_s = green_s - PHI * (L_cond - L_green). Per (season, event), never per stop.
- No clamps anywhere.

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

pit_stops.parquet (train races only)
  season, round, event, driver, lap, compound_in, compound_out, condition,
  transit_s, lap_sum_s, driver_pace_s, ref_lap_s, pace_trusted, measured_loss_s,
  phi, pit_loss_s
  measured_loss_s: green stops with trusted pace only. ref_lap_s under SC / VSC is the
  stay-out median, a diagnostic (position confounded). pit_loss_s: green = measured,
  SC / VSC = the (season, event) value.

pitloss_by_track.parquet (train races only, key season + event)
  season, event, green_s, transit_s, lap_green_s, lap_sc_s, lap_vsc_s, sc_s, vsc_s,
  phi, n_stops, n_green, n_sc, n_vsc
  For another phi: loss = green_s - phi * (lap_cond_s - lap_green_s)

sc_events.parquet (all races, split column; ground truth for the backtest)
  season, round, event, split, race_laps, kind (SC, VSC, RED), lap_deploy, t_deploy_s,
  lap_end, t_end_s, t_ending_s, ended_by, duration_s, duration_laps
  From track_status.parquet (session time). Train medians: SC 3 laps / 436 s, VSC 96 s.

sc_rates.parquet (train races only)
  event, n_races, race_laps, sc_deployments, vsc_deployments, p_sc_per_lap,
  p_vsc_per_lap, p_sc_per_lap_raw, p_vsc_per_lap_raw, p_sc_lap1
  Lap-1 SC is a start hazard counted apart (p_sc_lap1 = 2 of 23 = 0.087). Per-lap
  rates shrunk to the global train rate with a 60-lap prior (1 to 3 races per track).

overtakes.parquet (all races, split column; fit on train only)
  season, round, event, split, lap, overtaker, overtaken, pace_delta_s, tyre_age_delta,
  compound_pair, gap_before_s, drs_likely (gap < 1.0 s)
  Pass = swap in lap-end order between laps k-1 and k, same lap number. Excludes lap 1,
  SC / VSC / red laps and any car with an in-lap on k-1 or k or an out-lap on k.
  pace_delta_s = overtaken minus overtaker recent pace (median of last 3 green laps),
  tyre_age_delta = overtaken minus overtaker tyre life. Positive favours the overtaker.
  953 passes, 15 to 60 per race; FastF1 Position agrees on 100% of them.

battles.parquet (all races, split column)
  season, round, event, split, lap, driver, ahead, gap_before_s, pace_delta_s,
  tyre_age_delta, compound_pair, drs_likely, passed
  Every car within 2.0 s of the eligible car directly ahead at the end of lap k-1.
  11473 rows. Pass rate: 32% under 0.5 s, 6% at 0.5 to 1.0, 1% at 1.0 to 1.5.
  Session 3 fits the pass model on these (failed attempts included).

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
- Race control message Time (rcm.parquet) is wall clock. LapStartDate is empty because
  FastF1 needs telemetry for t0_date. Use track_status.parquet and session_status.parquet
  (session time, saved by ingest since Session 1) for anything timed.
- 2023 Austria lap 2: SC led the field through the pit lane. Not a mass pit stop.
- 2025 Dutch: pit lane limit raised to 80 km/h. Transit 17.8 s vs 21.2 s in 2024,
  green loss 18.1 s vs 23.2 s. Pit loss is keyed by (season, event).
- Holdout races have no pitloss_by_track row (train only). Session 5 must pick a
  pre-race estimate, e.g. the same event's latest train season. Watch for pit lane
  rule changes like Zandvoort 2025.
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

## Session 2: race-progress correction (src/fuel.py, done)
- lap_time_fc_s = lap_time_s - K * laps_remaining / race_laps. K = 3.249 s per race,
  fitted on 22 train races (2024 Saudi excluded from the K fit only: one stint after the
  lap 7 SC, no between-stint variation). 0.0565 s/lap at a median 58-lap race.
- K is fuel plus track evolution, NOT fuel. Never call it a fuel coefficient.
- By season 3.18 / 3.26 / 3.32 (slight upward drift, watch it). Free air K 3.07
  (0.0473 s/lap per-lap form): traffic inflates the slope a little.
- Scaling check: per-race slopes vs 1 / race_laps r = 0.54. Built into the module.
- Rejected: per (race, driver, stint, compound) intercepts. tyre_life and laps_remaining
  are collinear within a stint, so it estimates fuel minus degradation (0.0043).
- Any error in K moves within-stint degradation estimates one for one (K / race_laps).
- Outputs: laps_fuel_corrected.parquet (all laps, adds race_laps, progress_s,
  lap_time_fc_s), fuel_fit.json.

## Session 2: tyre model (src/tyre.py) - calibration FAILS, awaiting decision
- Target y = lap_time_fc_s - race_ref_s (field median fc clean lap, laps 2-10, causal);
  driver_pace_s from the same window. Modelled laps > 10. GroupKFold(5) by race, train only.
- Held-out: below p10 27.5%, above p90 21.9% (target 10 / 10). MAE LightGBM 0.646 s,
  net 0.718, predicting zero 0.876. Crossing 0.1%.
- Cause: per-race offset sd 0.66 s held out vs 0.043 in sample. The net identifies each
  training race (event + temps) and learns its level, so quantiles carry only within-race
  noise. Residual drift vs lap correlates 0.58 with the race's progress-slope deviation.
  The laps 2-10 reference also carries the race-start bump below.
- Do not tune until the user picks a fix.

## Session 2: degradation (src/degradation.py)
- Within stint, never pooled. Median slope per (race, compound), A pooled-K / B joint:
  SOFT 0.079 / 0.087, MEDIUM 0.052 / 0.049, HARD 0.059 / 0.053 s per lap. A vs B r 0.88.
- Paired within race: SOFT minus MEDIUM +0.017 (A) / +0.012 (B), 82% / 73% of 11 races.
  HARD minus MEDIUM 0.000, 50% of 20 races: no deg difference between hard and medium.
- Curves beyond about 25 laps drop or flatten: survivorship (only low-deg long stints).
- Early-stint effect (renamed; "early-soft anomaly" was wrong): ages 2-4 sit above the
  within-stint trend on 5-15 in EVERY compound, ordered soft < medium < hard, and
  gap_ahead_s does not explain it. Race-start stints: SOFT 0.26, MEDIUM 0.58, HARD 0.70 s
  (with gap controls 0.32 / 0.59 / 0.71). Stints starting mid-race: 0.07 to 0.25 s.
  Two parts, both inversely related to grip: fresh-tyre warm-up in every stint (harder
  compounds warm slower; present mid-race on a rubbered track) and a larger race-start
  part (green track / non-linear early track evolution plausible). Not separable from lap
  data. Consequence: any anchor or reference window must skip laps 2-4.
- Anchor window moved from laps 2-10 to 5-10: misses 27.5/21.9 -> 26.9/21.5, offset
  removed 21.6/16.8 -> 20.9/16.9, race offset sd 0.66 -> 0.60, MAE 0.646 -> 0.625.
  The start-compound bias was real but small.

## Next session (2) preview
Tire model: per-race baseline fit, cross-race LightGBM, PyTorch quantile model
(p10/p50/p90), cliff detection, held-out MAE and calibration.
Interface target: predict_laptime(features) -> (p10, p50, p90)
