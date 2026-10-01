# HANDOFF: F1 Race Strategy Optimizer (Insane Version)
Sessions 1-3 of 6 done | Started 1:40pm ET Sep 30 | Deadline Oct 1 11:59pm ET
Environment: Windows 11, PowerShell, venv at .venv, Python 3.11

## Project summary
A lap-by-lap F1 race strategy engine built on FastF1 data. It predicts tire
degradation with uncertainty, simulates all 20 cars (gaps, traffic, overtakes,
undercuts), runs Monte Carlo over Safety Car/VSC and rival strategies, and
re-optimizes live during a replayed race, using Fast Flag's Safety Car calls as
triggers. Delivered with a pit-wall dashboard and an Arduino "BOX" pit board.

## Session roadmap
1. Data foundation (DONE)
2. Tire degradation model with uncertainty (LightGBM + PyTorch quantile) (DONE)
3. Overtaking model + race rules (DONE)
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

## Session 2: first tyre model, fixed reference (SUPERSEDED by the re-anchored model below)
- Target y = lap_time_fc_s - race_ref_s (field median fc clean lap, laps 2-10, causal);
  driver_pace_s from the same window. Modelled laps > 10. GroupKFold(5) by race, train only.
- Held-out: below p10 27.5%, above p90 21.9% (target 10 / 10). MAE LightGBM 0.646 s,
  net 0.718, predicting zero 0.876. Crossing 0.1%.
- Cause: per-race offset sd 0.66 s held out vs 0.043 in sample. The net identifies each
  training race (event + temps) and learns its level, so quantiles carry only within-race
  noise. Residual drift vs lap correlates 0.58 with the race's progress-slope deviation.
  The laps 2-10 reference also carries the race-start bump below.
- User chose: fix the anchor window, then re-anchor (1), then quantiles on out-of-race
  residuals (2) only if 1 leaves misses above about 15%.

## Session 2: re-anchored tyre model (current src/tyre.py)
INTERFACE CHANGE from the original brief. The brief said predict_laptime(features) ->
(p10, p50, p90). A fixed reference cannot carry the race level to unseen races (offset sd
0.60 s), so the model is now anchored on the live race and horizon dependent:
- compute_anchors(laps up to t) -> anchor_s per (driver, lap t): field median of clean fc
  laps over laps max(5, t-4)..t plus the driver's gap to the per-lap field median, current
  stint only (>= 2 laps). Never uses laps 2-4.
- predict_laptime(features) needs anchor_s, h, tyre now (tyre_life_t, compound_t), tyre
  plan at t+h (tyre_life_f, compound_f, fresh_tyre_f, stints_ahead), temps at t, event,
  driver, team, laps_remaining at t+h, race_laps. Returns absolute p10/p50/p90 for lap t+h.
- Session 4's DP optimizer must call it with h up to a full stint (trained on h in
  1,2,3,5,8,10,15,20,25,30).

FINAL MODEL (two stages, ACCEPTED):
- p50 = LightGBM median, all features.
- p10 / p90 = p50 + offsets from a quantile net trained on LightGBM residuals from races
  left out of an inner GroupKFold. The spread net only sees features that cannot identify
  a race (h, tyre ages, compounds, fresh, stints_ahead). Do not give it event, driver,
  team or temperatures: with them it learns each training race's offset and misses
  24.4 / 14.7% (tested).
- Models: data/models/tyre_lgb.txt and tyre_model.pt (plain state, not pickled
  objects). Refit: python -m src.tyre --final-only.

Held-out results (GroupKFold by race, train races), acceptance <15% / <15%, h=1 near 10%:
            below p10  above p90  width  median resid  MAE    race offset sd
  all          11.6%      9.8%    1.51     +0.013     0.508   0.17
  h=1          10.7       9.0     1.14     +0.018     0.368   0.11
  h=5          11.4       9.8     1.55     +0.005     0.503   0.19
  h=15         11.4       9.6     1.86     +0.009     0.585   0.24
  h=30         13.1      10.5     1.87     +0.006     0.630   0.36
- Width grows with h but flattens from h=15 to 30 (1.86 -> 1.87) while misses creep up.
  Fewer rows at h=30 (8408) and survivorship (only laps still clean 30 ahead). Watch it
  in Session 4.
- Quantile crossing 0.04%.

SOFT BIAS (survives fix 2, not tuned away): soft p50 is too fast by 0.070 to 0.085 s at
every horizon (median resid y - p50 = +0.075; MEDIUM -0.009, HARD +0.013). Calibration
looks fine (soft 13.1 / 9.5%) because the intervals widened, and per-race offset sd for
soft is 0.41 s vs 0.19-0.20. Consequence: the optimizer, which runs on p50, will favour
soft strategies by about 0.075 s per soft lap (about 1.1 s over a 15-lap stint). Session 4
must check soft-strategy choices against this, e.g. a sensitivity run with soft p50 + 0.075.

History of the fix: fixed laps 5-10 reference 26.9 / 21.5%; re-anchored single net
22.2 / 14.3% (net p50 biased 0.036 to 0.104 s slow, growing with h); two-stage 11.6 / 9.8%.

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

Cliff detection was skipped in Session 2 (user decision: time).

## Session 3 brief: overtaking model and race rules
Mostly modelling on data we already have. No new extraction needed.

Inputs (train split only, via src/splits.py; holdout never touches a fit):
- battles.parquet: 11473 car-behind / car-ahead pairs within 2.0 s at the end of lap k-1,
  with passed flag. Includes the failed attempts. Base pass rates: 32% under 0.5 s, 6% at
  0.5-1.0, 1% at 1.0-1.5, 0.6% at 1.5-2.0.
- overtakes.parquet: 953 passes (all pairs, not only adjacent), FastF1 Position agrees.
- laps_fuel_corrected.parquet with gap_ahead_s, for the cost of following.

Deliverables:
1. src/overtake_model.py: P(pass on lap | gap_before_s, pace_delta_s, tyre_age_delta,
   compound_pair, drs_likely, track). GroupKFold by race. Report reliability (predicted vs
   observed by probability bin) on held-out races; reproduce the base rates by gap bin.
   Track overtaking difficulty from 1 to 3 races per event: shrink toward the global rate.
2. Following cost: lap time lost when running within X s of the car ahead, as a function
   of gap, within stint on fuel-corrected laps (degradation.py found gap < 1 s costs
   0.14 to 0.53 s). The race engine needs it to make undercuts and traffic real.
3. Race rules module for the Session 4 engine: two dry compounds required, no passing
   under SC / VSC, DRS off for the first 2 laps and after restarts, SC queue compression
   and lapped cars, pit stop under SC (pit loss from pitloss_by_track with PHI). Write
   each rule as a tested function; cite the source of each rule.

Identification risks to check before fitting (Session 1-2 lesson: an uncontrolled
variable was larger than the effect three times):
- pace_delta_s comes from recent laps of a car that may be stuck behind (dirty air), so
  the follower's true pace is censored, and the censoring is worst exactly in close battles.
- pace_delta_s and tyre_age_delta are correlated (fresher tyres are faster).
- Track difficulty is confounded with which battles occur (Monza vs Hungary).
- Selection: pairs that stay close for many laps are the ones where passing failed.
- The passed flag compares lap-end order; a pass and re-pass within a lap is invisible.

Carry forward to Session 4:
- PHI validation against observed position loss (see above).
- Soft p50 bias +0.075 s (see the tyre model section).
- Interval width flattening between h=15 and h=30.
- predict_laptime is horizon dependent and needs anchors (interface change above).

## Session 3 results (done)
Identification, worked before fitting:
1. Dirty-air censoring of pace delta: large. Within 0.5 s, observed recent-lap delta 0.22 s
   vs free-air delta 0.62 s (r 0.50). The pass model uses free-air pace delta (what the
   engine knows). Free-air pace exists for 52% of train battles (70% of passes); the rest
   are stints with no clear-air lap, skewed to stuck pairs, and get a missing flag.
2. Free-air pace delta vs tyre age delta r 0.56: prediction fine, coefficients not
   separately interpretable.
3. Track difficulty: pass rate within 1 s 8% (Silverstone, Zandvoort) to 24% (Abu Dhabi,
   Spain), 1-3 races per track: L2-shrunk event terms.
4. Selection in long battles: pass rate 20% (battle laps 1-2) -> 5% (lap 6+). Modelled
   with log laps_in_battle; the engine applies the average decay to every pair.
5. Pass and re-pass within one lap: not identifiable; lap-end order is what scores.

src/traffic.py: free-air pace per (driver, stint) from laps with > 3 s clear air (leaders
count as clear air), age-adjusted with the joint per-race deg slope.
- Dirty air, lap time minus own free-air pace, by gap at end of previous lap:
    gap        aero-only (not faster)   held up (faster)   pooled
    0-0.5         +0.35 (0.24-0.66)        +0.96            +0.80
    0.5-1         -0.11                    +0.62            +0.33
    1-1.5         -0.07                    +0.41            +0.21
    1.5-2         -0.12                    +0.33            +0.14
    2-3           -0.09                    +0.17            +0.04
- NOT SEPARABLE CLEANLY: at 2-3 s both effects should be about 0, but the split by
  estimated free-air delta leaves -0.085 / +0.167 s of selection bias, as big as the effect.
- PARAMETERISED for the engine: dirty_air_penalty(gap, d0): d0 up to 0.5 s, linear to 0 at
  1.0 s. d0 = 0.35 s, Monte Carlo sweep 0.24 to 0.66. Being held up comes from the engine
  (a car that cannot pass cannot lap faster than the car ahead), never from this curve.
- The pooled curve is the validation target for Session 4 (see brief).

src/overtake_model.py: logistic P(pass on lap | state at end of previous lap).
- Held-out (GroupKFold by race): AUC 0.926 (gap only 0.836), Brier 0.045 (0.057).
- Calibration, predicted vs observed:
    [0,0.02) 0.006/0.005  [0.02,0.05) 0.033/0.024  [0.05,0.1) 0.071/0.059
    [0.1,0.2) 0.144/0.125  [0.2,0.3) 0.246/0.240  [0.3,0.5) 0.392/0.363
    [0.5,1] 0.748/0.641 (n 504, about 45% of all passes)
- Base rates reproduced without being given: within 0.5 s 34.9% predicted / 31.8% observed,
  0.5-1 7.1 / 5.9, 1-1.5 1.5 / 1.1, 1.5-2 1.0 / 0.6.
- MISS, reported not fixed: 15% too many passes overall (828 vs 720). Tracks seen in
  training are calibrated (7.37% vs 7.27%); unseen tracks over-predict (15.2% vs 8.9%);
  the top bucket over-predicts by 11 points. Holdout: only Singapore is an unseen track.
- Model: data/models/pass_model.json (coefficients, standardisation, events, gap edges).

src/rules.py: compound rule, no overtaking under SC / VSC / red, DRS availability, pit loss
by condition (free under red), stint caps SOFT 28 / MEDIUM 46 / HARD 53 (longest pit-ended
train stints; a floor), optional race tyre limit. Article numbers unverified.

## SESSION 4 GOVERNING RISK: the optimizer selects for our biases
A DP optimizer searches for the strategy that maximises the modelled outcome, so it
systematically selects the strategies our known biases flatter. These do not average out;
the optimizer actively steers toward them:
- softs: tyre model soft p50 understated by 0.075 s/lap
- overtake-dependent strategies: pass model 15% high overall, about 70% high at unseen
  tracks (15.2% vs 8.9%), top bucket 0.75 vs 0.64
- long stints: h = 30 tyre intervals are a floor (survivorship)
Checks:
- On train races compare the optimizer's choice with what teams ran (stops, compounds,
  passes needed). Report it whatever it shows. Differences also contain team constraints
  we do not model (tyre-set availability above all), so also run it with the bias knobs
  neutralised (soft +0.075, pass probabilities scaled down): the part of the gap that
  closes is attributable to our bias.
- Tyre model predictions for a train race come from the CV fold model that did not see it,
  so the bias is present as in deployment.
- Joint (not one-at-a-time) Monte Carlo sweep over phi, D0, soft bias and pass scaling;
  report how much the strategy ranking changes.

## Session 4 status: engine built, VALIDATION FAILED TWICE, documented limitation
src/engine.py, replay of all 23 train races with each car's measured free-air pace (oracle,
actual strategies), 20 replays each:
- Following curve, simulated vs target (lap time minus free-air pace by gap):
    0-0.5 1.055 vs 0.80 | 0.5-1 1.028 vs 0.33 | 1-1.5 0.516 vs 0.21 | 1.5-2 0.196 vs 0.14
    | 2-3 0.001 vs 0.04. FAILS at 0.5-1.5 s.
- Passes per race 54.7 simulated vs 35.1 actual. Close-battle laps 512 vs 415 per race.
- Finishing order plausible (Spearman 0.945, mean position error 1.2, winner right 64%,
  median gap error 12 s), but with oracle pace this is a weak test.
- CAUSE (confirmed on simulated laps): the no-pass rule puts the follower a "held gap"
  behind, sampled from gaps in long battles (median 0.94 s). That is the distribution of
  where stuck cars end up, used as a floor, so it binds on followers that were never held
  up: followers NOT faster than the car ahead lose 0.54 s at 0.5-1 s and 0.20 s at 1-1.5 s,
  where only the aero penalty (about 0.17 and 0) should apply.
- Fix applied (user decision): stated constant min_gap = 0.2 s (not from data: every gap
  statistic comes from the same following laps as the validation curve), to be swept 0.1
  to 0.4. Re-run, same validation unchanged:
    0-0.5 0.48 | 0.5-1 0.26 | 1-1.5 0.02 | 1.5-2 0.01 | 2-3 0.00; passes 124.9 per race.
  MISSES in the other direction: with a 0.2 s floor cars bunch into trains and every
  bunched pair draws a pass each lap. The two versions bracket reality; the true following
  distance is state dependent (aero keeps a held car about 0.5 s+ back). The clean 2-3 s
  bin misses in both (0.00 vs 0.04).
- DECISION: stop, no third variant. Ship as a DOCUMENTED ENGINE LIMITATION and spend the
  remaining time on Sessions 5 and 6. The optimizer, joint Monte Carlo and the comparison
  with real strategies were NOT built.
- What the engine can still be trusted for: pit loss, SC / VSC neutralisation and compound
  rules on free-air pace. Not for traffic- or overtake-dependent strategy claims.
- If revisited: a following model where the gap a held car settles at depends on its
  free-air pace advantage and the dirty air penalty (no fixed floor), validated on the same
  curve and on passes per race, with the 2-3 s bin as the clean test.

## Session 4 brief: race engine, DP optimizer, Monte Carlo
The biggest build left. Inputs all exist; this session is simulation, not modelling.

Engine (src/engine.py), one lap at a time, 20 cars:
- Free-air lap time per car: tyre model predict_laptime (anchored, horizon dependent;
  sample between p10 and p90, do not use p50 alone for uncertainty) plus K * progress.
- Traffic: car within 1 s of the car ahead adds dirty_air_penalty(gap, d0). A car that does
  not pass cannot finish the lap ahead of the car in front (held up). Passes drawn from the
  pass model, using free-air pace delta, gap, DRS (rules.drs_enabled), tyre and compound
  deltas, track, laps in battle.
- Pit stops: pit loss from pitloss_by_track (season, event) via rules.pit_loss_s.
- SC / VSC: deployments from sc_rates (lap-1 SC apart via p_sc_lap1), durations from
  sc_events train medians or their empirical distribution; no passing; field compresses.
- Rules: every plan checked with rules.strategy_violations.

Optimizer: DP over (lap, compound, tyre age, stops made) for one car, rivals on fixed or
sampled strategies; Monte Carlo over SC timing, rival strategies and lap time noise.

Validation, before any strategy claim:
- Engine vs history on train races: replay real strategies and compare finishing order and
  gaps.
- Traffic: simulate following pairs and check the engine reproduces the pooled dirty-air
  curve (0.80 / 0.33 / 0.21 / 0.14 / 0.04 s by gap bin). Separately, passes per race.
- PHI: pit loss under SC uses phi = 0.08, swept 0.05 to 0.12 in the Monte Carlo. Validate it
  against observed position loss when stopping under SC on train races, not lap-time
  seconds (see "Session 4 task" above).

Carries, each a required check:
- Soft p50 + 0.075 s sensitivity: soft p50 is too fast by about 0.075 s at every horizon.
  Re-run soft-strategy recommendations with soft p50 + 0.075 and report whether they flip.
- h = 30 tyre intervals are a floor, not an estimate: survivorship (only laps still clean
  30 laps later are in that bucket). Do not read long-horizon width as the true uncertainty.
- Pass model over-predicts on unseen tracks (15.2% vs 8.9%) and in its top bucket (0.75 vs
  0.64): sensitivity run with pass probabilities scaled down, and report Singapore (the
  holdout's unseen track) separately.
- Dirty air d0 sweep 0.24 to 0.66 in the Monte Carlo.
- Holdout races touch nothing until Session 5's backtest.
