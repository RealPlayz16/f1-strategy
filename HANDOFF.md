# HANDOFF: F1 Race Strategy Optimizer (Insane Version)
All 7 sessions done (Session 7 reworked the engine following model; it still fails
validation, now with an identified cause; no optimizer) | Started
1:40pm ET Sep 30 | Deadline Oct 1 11:59pm ET
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
4. 20-car race engine + DP optimizer + Monte Carlo (engine only, failed validation,
   documented limitation; optimizer and Monte Carlo not built)
5. Live replay strategy engine + Fast Flag hook + backtest (DONE)
6. Pit-wall dashboard + README/demo (DONE; Arduino pit board cut, listed as planned work)
7. Engine following model rework (DONE as a rework; validation still fails, cause now
   identified and split between the engine and the pass model. See Session 7 results.)
8. free_missing routing (DONE, misses its own criterion in 1 of 4 bins, does not fix the
   engine, kept for correctness); race set 27 -> 38 and full rebuild (DONE); Session 5
   backtest re-run on the 9-race holdout (DONE)
9. Pass model refit on causal coverage (DONE, correctly specified, makes the engine
   worse, kept); DP optimizer built and run end to end (DONE, first strategy
   recommendation in the project; stop count robust, compound choice is not)

## Working style
- Minimal explanation, direct bullets, step-by-step commands
- No em-dashes
- Edit files directly; run pytest and ruff yourself, do not ask me to paste output
- Do not patch symptoms. If a number looks wrong, find the cause before changing code
- State assumptions before writing code when a formula or model choice is involved

## Method: every real defect so far was found by two measurements disagreeing

Four instances now, none of them found by a test. This is the project's most reliable
debugging tool and it is worth applying deliberately rather than by luck.

1. 2023 Austria SC pit loss (Session 1). Lap-time subtraction said 84 s; the transit
   measurement and the physics said about 23 s. Cause: FastF1 starts a new Stint on any pit
   lane passage, so 19 drive-throughs were counted as stops, while race control said "SAFETY
   CAR THROUGH THE PIT LANE". Three successive fixes patched the symptom first, one of them a
   clamp forcing sc_s < green_s, before anyone looked for the cause.
2. Pit-now plans ignored the stop at pit_lap (Session 5, 035cd85). Found by consistency
   checks on the break-even table.
3. The false-call branch dropped the would-be neutralised laps (Session 5, d28803b). SC and
   VSC "pit anyway" were 7% and 38% for a quantity that should have been the same. Both read
   38% after the fix.
4. Inverted pass push-back (Session 7, 506cb2a). Drawn passes 73 against 28 swaps in lap-end
   order on the same race, for a quantity where the second should never be far below the
   first.

Two things this buys that a test does not. It catches errors in the measurement as readily as
in the model, and it does not need to know the right answer in advance, only that two routes
to the same number must agree. Mechanism tests would not have caught any of the four.

### Corollaries, all three learned the hard way
- THE DIRECTION A NUMBER MOVES AFTER A FIX IS NO EVIDENCE THE FIX WAS RIGHT. Fixing the
  inverted push-back made the headline worse, 28 to 72 scored passes against 35.1 actual. It
  was still correct. Conversely the Session 1 clamp made the pit loss look right and was
  wrong. Judge a fix on the mechanism, never on whether the number improved.
- A PLAUSIBLE MECHANISM WILL ABSORB A REAL DEFECT IF YOU LET IT. The push-back bug was about
  to be written up as a limit of the model class, with a physical story (a static equilibrium
  cannot represent a car dropping back and taking a run) that fitted the symptom and was
  false. Session 7 then produced three more inferred claims that the instrumented run
  contradicted outright. When a tidy explanation arrives before the measurement, treat the
  tidiness as a warning.
- WHERE IT FAILS: it needs two independent routes to the same quantity. Session 4's following
  curve had only one, which is part of why two attempts failed without the cause surfacing.
  Introducing the battle-lap distribution in Session 7 was partly about restoring a second
  route. When a quantity has only one measurement, that is a gap in the method, not a sign
  the quantity is fine.

## ENGINE LIMITATION: what the engine can and cannot be trusted for

Read this before using the engine for anything. Sessions 4, 7 and 8 all feed it and it has
survived three reworks.

### The sharpest form of it, and the best quantified (Session 8)
TRAFFIC COST ON PIT EXIT SITS AT 3.4% CAUSAL COVERAGE, THE WORST-COVERED REGION IN THE
DATASET. Causal coverage is whether a car has had clear air so far in its current stint,
which is what decides whether the pass model gets a real free-air pace delta or falls back to
its missing flag (src/traffic.py causal_coverage). It is mostly a function of
lap-within-stint, not of gap:
    laps 1-3 of a stint   3.4%        laps 11-20   32.0%
    laps 4-6             11.7%        laps 21+     47.5%
    laps 7-10            18.0%                              (gap bins only span 17 to 33%)
So an undercut compares pace LATE in a stint, 47.5% covered, against pace on the OUT-LAP and
the three laps after it, 3.4% covered. The engine's pass probabilities in exactly the region
where undercut and overcut modelling lives come almost entirely from the missing branch.

And that branch is fitted on the wrong population. The flag behind it is NOT causal:
free_pace uses the clear-air laps of the whole stint, so a car covered only later already
counts as covered at the start. Session 8 measured the consequence at 0-0.5 s: routing
causally makes the engine query a 19.1%-rate branch for about 83% of pairs, standing in for a
population that actually passes at 27.4%. The mismatch inverts rather than closing, and no
amount of engine work fixes it. THE COMPLETE FIX IS A REFIT OF data/models/pass_model.json
WITH A CAUSAL COVERAGE FLAG. Until that happens, the region the engine is least able to model
is the region the strategy work most needs.

This is a better statement than the Session 4 one, which said only that the engine failed to
reproduce following behaviour. It names a mechanism, a location and a size.

### The counts
- Scored passes 60.6 per race against 35.1 actual, 1.73x. Was 2.05x before the Session 8
  free_missing routing.
- Car-laps within 0.5 s of the car ahead 134.2 per race against 63.2, 2.12x. UNCHANGED by
  every rework so far, and its mechanism is NOT IDENTIFIED. Not to be hunted without a
  mechanism to test (user decision, Session 7).
- Battle-lap distribution total 477.4 against 415.0. The other three gap bins are within 10,
  7 and 22%.

### Trustworthy
Pit loss, SC / VSC neutralisation, compound rules, anything that runs on free-air pace.

### NOT trustworthy, BLOCKED
Traffic-dependent and overtake-dependent strategy claims. Undercut and overcut modelling,
traffic cost on pit exit, and any claim that depends on track position. The optimizer, the
joint Monte Carlo and the comparison with what teams actually ran were not built in Session 4
and have not been built since.

## VALIDATION CEILING: n=1 on the real-call path is Fast Flag's, not ours

THE PREMISE THAT MORE RACES FIXES n=1 IS FALSE. It was the stated reason for the Session 8
holdout expansion and the screening falsified it. Record this before anyone plans around the
decision path again.

src/live.py reads Fast Flag recs from ~/fast-flag/data/timeline/<race>.json.gz and has no
fallback: no timeline, no recs, no decision. Fast Flag has timelines for FIVE races:
    2021_Azerbaijan   2025_Abu_Dhabi   2025_Japanese   2025_Singapore   2025_United_States
Four of those are already our holdout. data/case_studies/ holds the same five. So the number
of holdout races on which a REAL Fast Flag call can be scored is capped at 4, and only one of
those four has a real neutralisation (2025 US, VSC lap 7). Ingesting more races does not move
it. OUR INGEST WAS NEVER THE CONSTRAINT.

Closing it means building a Fast Flag case study and running their pipeline per race: a
different repository, telemetry-weight ingest, and the standing constraint that Naman's
models stay untouched. It is a cross-project job and it is NOT closeable from this repo.

What expanding our holdout does and does not buy:
    real-call analysis (precision, false calls, lead time, blind window)   UNCHANGED, n=1
    break-even / hypothetical calls (Session 5, every 10th lap, synthetic) scales with races
    tyre calibration, pit loss, SC rates, engine replay                    genuinely improves
Session 5 built the break-even work off 724 hypothetical calls specifically so the project
would not rest on n=1. That reasoning is now load-bearing rather than cautious.

### OPEN RISK: Fast Flag contamination of candidate holdout races
Fast Flag's own race lists are readable without a rebuild, and both were checked:
- TRAINING, 20 races, ~/fast-flag/data/models/risk_meta.json key "races". The 2025 entries
  are Azerbaijan, Belgian, British, Dutch, Miami, Australian.
- CHECK, 20 races, ~/fast-flag/docs/charts/escalation_check.json key "check_races". The 2025
  entries are Saudi Arabian, Emilia Romagna, Sao Paulo, Las Vegas.
Training races are excluded from our holdout by the rule that moved 2025 Dutch in Session 1.
Check races were not fitted on (the README says the fixes were found on the 2021 Azerbaijan
replay and checked on the TRAINING races, never on the check set), so the machine learning
models never saw them, but results have been reported on them. They are weaker than clean.
CONDITIONAL RULE, CHECK IT BEFORE ANY REAL-CALL WORK: the four Fast Flag CHECK races stayed
parity-eligible in the Session 8 split and 2025 Las Vegas is in the holdout. That is safe only
because no Fast Flag timeline exists for any of them, so no real call can be scored on them
and the contamination is inert. IF A FAST FLAG TIMELINE IS EVER BUILT FOR A CHECK RACE, THAT
RACE MOVES TO TRAIN AND THE AFFECTED FITS ARE REBUILT. The risk goes live the moment a
timeline appears, which is exactly the kind of thing that gets forgotten. Checking costs one
ls of ~/fast-flag/data/timeline.

STILL UNVERIFIED: ~/fast-flag/data/features/ holds only 3 of the 20 training races locally
(2023_Australian, 2024_Canadian, 2025_Azerbaijan), so the lists above come from committed
metadata rather than from the data. If a race list changes upstream, ours goes stale silently.
Re-check both files before trusting any holdout race for real-call work.

## Gates do not get written until data breaks them

Expanding the race set from 27 to 40 candidates turned up THREE data defects in the 13 new
races, and found more bad data than bad code. The original 27 had none of them, which is
precisely why no gate existed for any of them. Two were invisible to every check the project
had, and in both cases the cheap fix would have buried the corruption and left the pipeline
green.

1. CORRUPT TYRE AGES THAT LOOK LIKE MISSING DATA (2025 Miami). No compound, tyre_life or
   stint for laps 1-24, for 19 of 20 cars. The trap is that where the feed resumes the counter
   RESTARTS: 15 of 20 drivers show an age below the laps they had already run with no stop
   before it. The obvious fix, drop the NaN laps, would have kept 559 laps whose ages are
   understated by up to 24 and fed them to every age-based model silently. Surfaced only as
   "SVD did not converge" deep inside a least-squares call.
2. A WET RACE WITH NO INTERMEDIATES RECORDED (2025 Belgian). The wet exclusion tests
   compounds, and a drying track after a wet delay never shows one: the field runs slicks on a
   damp track. It looks dry to a compound test and is not dry to any lap-time model, 122.8 s
   at lap 5 against 107.9 s by lap 22. The cheap fix, add it to fuel.py's FIT_EXCLUDE beside
   2024 Saudi, would have left it corrupting degradation and the tyre model.
3. STALE RAW DIRECTORIES (self-inflicted). Screening candidates through a throwaway config
   wrote into the real data/processed/raw/.

What the three have in common: each was caught by something refusing to proceed, not by a
test. A crash in fuel.py, fuel.py's own output gate, and a KeyError in split_of. None of the
56 tests failed at any point. That is the same pattern as "## Method: every real defect so far
was found by two measurements disagreeing", in its other form: a component that refuses rather
than guesses turns silent corruption into a loud stop. fuel.py's gate had looked like
belt-and-braces for two sessions and earned its place here.

The lesson for the next expansion: new data is where the gaps are, the existing gates encode
only the defects the existing data happened to contain, and a fix that makes the symptom go
away without explaining it is how corruption gets in.

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
- Holdout races (2025 Japanese, United States, Singapore, Abu Dhabi) are excluded from
  every fit: phi, pit loss medians, SC rates, tyre models

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
SUPERSEDED IN SESSION 8, kept as the record of what 23 races showed. On 29 races the soft
median residual is +0.022, so most of this was small sample. The live bias is horizon shaped:
h = 30 MEDIUM +0.087, SOFT +0.091, HARD -0.020. Use src/live.py P50_BIAS_LONG_H_S.

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
- DEAD, do not use: soft p50 bias +0.075 s. Session 8 on 29 races puts the soft median
  residual at +0.022. The live bias is HORIZON shaped: at h = 30, MEDIUM +0.087 and
  SOFT +0.091, HARD -0.020. Spec and code: src/live.py P50_BIAS_LONG_H_S = 0.09 applied
  to MEDIUM and SOFT beyond h = 15.
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
- long stints on the softer compounds: tyre p50 runs about 0.09 s slow on MEDIUM and
  SOFT beyond h = 15 and not on HARD (Session 8). The flat soft 0.075 is dead.
- overtake-dependent strategies: pass model 15% high overall, about 70% high at unseen
  tracks (15.2% vs 8.9%), top bucket 0.75 vs 0.64
- long stints: h = 30 tyre intervals are a floor (survivorship)
Checks:
- On train races compare the optimizer's choice with what teams ran (stops, compounds,
  passes needed). Report it whatever it shows. Differences also contain team constraints
  we do not model (tyre-set availability above all), so also run it with the bias knobs
  neutralised (p50 + 0.09 on MEDIUM and SOFT beyond h = 15, pass probabilities scaled down):
  the part of the gap that closes is attributable to our bias.
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
- REVISITED IN SESSION 7 along exactly these lines (pace rule, no fixed floor). It did not
  pass. The 2-3 s bin turned out NOT to be a clean test under a pace rule: it is
  structurally 0.00 and cannot fail. See "Session 7 results" for what replaced it.

## Session 7 results (done): following model reworked, still fails, cause identified

Replaced the position-rule following model with a pace rule. The engine is better and still
fails the pass count. Two separate defects were found, one in the engine and one in the pass
model, and one coding bug nearly got attributed to a model limitation. Nothing was tuned.

### The rule (src/engine.py)
There is no minimum gap anywhere in the engine. For car d with the nearest non-pitting car a
ahead and gap g at the end of the previous lap:

    unconstrained   L0_d = free pace + lap noise (+ soft bias) + dirty_air_penalty(g, d0)
    held iff        t_d + L0_d < arrival of a   (would finish ahead of a car it did not pass)
    held            L_d = L_a + dirty_air_penalty(g, d0)

Held cars end the lap g + aero(g) behind. L_d > L0_d always (held implies L0_d < L_a - g), so
the rule can only slow a car down; it is not a clamp. A car settles where aero(g*) equals its
pace in hand; a car with more than d0 in hand has no equilibrium. Trains propagate because
L_a already carries a's own held time.
- The held test is on ARRIVAL, not pace, so a slower follower is never held and drops back on
  its own. No case analysis needed for it.
- The blocking car is the resolved car with the LATEST arrival, not always the nearest: a car
  shoved back by a pass still blocks the cars behind it.
- Pass lap: the passer keeps its own lap time and the passed car pays the swap. ASSUMPTION
  about who pays, taken in the direction that does not compound the pass model's known 15%
  over-prediction. PASS_MARGIN_S = 0.1, stated.
- A car behind a pitting car references the nearest non-pitting car ahead, matching how
  battles.parquet defines the eligible car directly ahead.
- SC_RESTART_GAP_S 1.0 -> 0.41, MEASURED: median gap to the car ahead on the last lap of each
  train SC, 11 events, 173 gaps, per-race medians 0.28 to 0.54. The old 1.0 was a stated
  constant and was 2.5x the data. Isolation run (--restart-gap 1.0, same seed) moved passes
  by 8% of the miss and nothing else by more than 1%, so it is not a cause either way; keep
  the measured value.

### VALIDATION TARGET CHANGED: battle-lap distribution, not the following curve
The following curve is no longer a test and is DEMOTED to a printed sanity check. Under the
pace rule an unheld car's dev is exactly dirty_air_penalty(gap), which is zero past 1.0 s by
construction, so the 1.5-2 and 2-3 bins read about 0.000 and CANNOT FAIL. A near-match there
is worth nothing. The bins that can still move take d0, the aero shape and free-air pace from
the same following laps that produce the target, so no part of the curve is out of sample.

The primary target is now the BATTLE-LAP DISTRIBUTION by gap bin: car-laps per race within
2.0 s of the eligible car directly ahead. Nothing in the following model was fitted to it
(d0 and the aero shape come from conditional mean lap times, free-air pace from clear-air
laps, the pass model from pass outcomes), and it is a distribution over four bins rather than
five conditional means, so it constrains where cars spend their time, which is what drives
pass counts and traffic cost.
  Both sides are constructed identically. src/engine.py eligible_sim reproduces
  overtakes.eligible_pairs on simulated laps: ineligible cars removed from the ordering, lap 1
  and neutral laps on lap k dropped, in-lap and out-lap dropped, and a lap k-1 under a
  neutralisation KEPT (overtakes.py flags lap k only), so restart laps count on both sides.
  passes_sim is now scored as overtakes.py scores it, any swap in lap-end order between
  eligible cars. passes_drawn is kept beside it; the two differ a lot and only the first is
  comparable to 35.1.

### Result, 23 train races x 20 replays, oracle pace
Battle laps per race by gap bin (PRIMARY):
    bin        sim    actual   ratio
    0-0.5    133.1      63.2    2.11      <- the one real miss
    0.5-1    148.7     162.5    0.92
    1-1.5     98.2     102.3    0.96
    1.5-2     69.0      87.0    0.79
    total    448.9     415.0    1.08
Passes per race: 72.0 scored (82.5 drawn) vs 35.1 actual.
Finishing order: Spearman 0.951, mean position error 1.17, winner right 67%, median gap error
12.0 s. Weak test under oracle pace, unchanged in meaning from Session 4.
Following curve (SANITY ONLY): 0.64 / 0.48 / 0.15 / 0.03 / 0.00 vs 0.80 / 0.33 / 0.21 / 0.14
/ 0.04. Best of the three attempts and correctly ordered for the first time, but see above:
it is not evidence.

The distribution is within 10% in two bins and 21% under in one. The entire failure is
2.11x too many car-laps at 0-0.5 s.

### Pass decomposition, adjacent-pair drawn passes per race
    actual                       31.3
    sim laps x blended rate      55.4    too many battle laps at 0-0.5
    sim laps x covered rate      80.8    + the engine is always on the covered branch
    sim laps x sim model rate    79.7    + the simulated pair population (slightly negative)
So the miss is about 1.77x from lap counts and about 1.46x from the pass model branch, and
the simulated pair population contributes nothing (0.99x). The two causes are independent and
live in different modules.

### FINDING 1 (engine): 2.11x too many car-laps at minimum gap, MECHANISM NOT IDENTIFIED
Checked and excluded:
- SC restart compression. Only 12.5 of 141.1 close laps per race fall within 3 laps of a
  neutralisation ending (9%). It is a green-flag phenomenon.
- The restart gap value. The 1.0 s isolation run moved it by under 5%.
I do not have an identified mechanism for the remaining excess. Do not guess one.

RETRACTED, measured and false. Three claims were inferred mid-session from the actual-side
data alone and then contradicted by the instrumented sim-side run. They are recorded here so
they are not re-derived:
1. "The engine cannot produce close followers with no pace advantage, because the only way to
   get close is to be faster." FALSE. Simulated 0-0.5 pairs are 45.8% not-faster against
   24.4% in the actual covered population, and the simulated median free-air delta there is
   0.063 s against 0.616 s. The engine produces MORE slow close followers than reality,
   because the held rule propagates down trains.
2. "The 1 to 2 s band is unpopulated because aero(g) is zero past 1.0 s." FALSE. 0.5-1 is at
   0.92x and 1-1.5 at 0.96x of actual. Only 1.5-2 is meaningfully under, at 0.79x.
3. "The two absences are one defect, a single fixed point per pace delta, and therefore a
   limit of the model class." WITHDRAWN. Both absences were measured and neither exists.
   There may still be a model-class limit here, but this evidence does not show one and the
   oscillation story is not supported by anything measured.

### FINDING 2 (pass model, independent of the engine): the covered branch is a selection
overtake_model.py has a covered branch (free-air pace delta known) and a missing-flag branch.
Session 3 recorded that uncovered pairs are "skewed to stuck pairs" but never sized it.
Train battles, by gap bin:
    bin      blended  covered  uncovered  coverage  covered/blend
    0-0.5      0.318    0.454      0.191     0.482          1.43
    0.5-1      0.059    0.095      0.031     0.441          1.60
    1-1.5      0.011    0.016      0.004     0.555          1.51
    1.5-2      0.006    0.007      0.003     0.644          1.27
Coverage is SELECTION, not data availability. Uncovered pairs have lower observed pace delta
in every bin (0-0.5: median 0.057 vs 0.215) and longer battles in three of four (1.5-2: mean
4.81 vs 3.67 laps), and no tyre-age advantage where covered pairs at 0-0.5 have three laps of
fresher rubber. Caveat that strengthens it: pace_delta_s is itself dirty-air censored and the
censoring is worst for uncovered pairs, so the measured separation is a LOWER BOUND.

Consequence: the covered branch was fitted on the easier-passing half of the data, and every
consumer that always has a pace estimate always takes it. That is the engine, the optimizer
and the live decision path. It is a defect of data/models/pass_model.json, not of
src/engine.py, and it stacks on the two over-predictions already on record (15% overall,
about 70% at unseen tracks).

WHY THE TARGET WAS NOT MOVED TO THE COVERED RATE. It was considered and rejected. The engine
simulates whole races, every adjacent pair, so the correct answer for a race is the census
count, 31.3 adjacent-pair passes / 35.1 scored swaps. Comparing a whole-race simulator to the
rate of an easier-passing subpopulation would redefine the target to match a known bias. The
covered rate is reported as a decomposition term only.

IDENTIFIED FIX, NOT BUILT: the engine already knows whether a car has had more than
FREE_AIR_GAP_S of clear air in its current stint, so it could set free_missing = 1 for cars
that have not and route them to the branch reality would have put them in. That is a new
variant and needs its own validation. It is the single most promising next step on the pass
count, because it addresses the 1.46x term directly and is derived rather than tuned.

### BUG: inverted pass push-back, and the pattern it repeats
    if d in passes and arr[a] > arr[d] - PASS_MARGIN_S:    # wrong
When the passed car was comfortably ahead (arr[a] much less than arr[d]) the test was false,
so it was never pushed back and the drawn pass never happened. Only draws where the pair was
already nearly level resolved into a swap. On 2023 Bahrain, 73 drawn passes produced 28 swaps
in lap-end order. Fast cars also stayed stuck behind cars they had nominally passed, which
inflated occupancy at minimum gap.

Found by checking scored_passes (swaps in lap-end order, how 35.1 is measured) against the
drawn-pass count the headline had been using. The earlier reported 76.8 per race was drawn
passes, is not comparable to 35.1, and is withdrawn.

This is instance 4 of the cross-measure pattern. See "Method: every real defect so far was
found by two measurements disagreeing" near the top, which carries the full argument and both
corollaries; do not re-derive it here.

### What the engine can and cannot be trusted for
Moved to "## ENGINE LIMITATION" near the top of this file, where it belongs. Sessions 4, 7
and 8 all feed it.

### Artefacts
engine_validation.json, engine_following.parquet, engine_battles.parquet (per-replay battle
gaps), engine_pairs.parquet (per-pair gap, free-air delta, predicted p, eligibility).
Suffixed _restart1 for the isolation run. python -m src.engine [--restart-gap X].
About 20 minutes for 23 races x 20 replays.

## Session 5 results (done): live Fast Flag decision system
FRAMING (use these words): a working live decision system, demonstrated end to end, with
honest uncertainty. Not a validated strategy system. The holdout has one real
neutralisation (2025 United States, VSC lap 7); one real neutralisation cannot validate
anything. Full report: docs/session5_results.md (python -m src.live_report).

Built:
- src/live.py: Fast Flag recs from ~/fast-flag/data/timeline/<race>.json.gz (--case
  builds, Naman's models untouched), clock check vs sc_events (2025 US: 0.28 s),
  call_probability from Fast Flag's out-of-sample scorecard (SC 10/23 = 0.44, VSC 12/21 =
  0.57, Beta intervals), ReplayDriver (state as of session time; leak tests), decide()
  (pit-now vs stay-out on free-air time, tyre p10/p50/p90 through split normals, plans
  fully correlated within, P(pit better) at rho 0 / 0.5 / 0.9, phi swept, break-even p*,
  accounts-for / does-not-account-for lists in every output).
- src/backtest.py: call episodes, matching, decisions at calls and at deployments, early
  call windows, team actions, --break-even hypothetical calls every 10 laps.
- src/live_report.py: docs/session5_results.md.

Findings:
- BLIND WINDOW (headline): Fast Flag called the 2025 US incident at the end of lap 5, 61.2 s
  before race control's VSC. The tyre model anchors on >= 2 clean laps from lap 5 (laps
  2-4 skipped, early-stint effect), so no decision is possible before about lap 6-7: the
  lead time bought nothing on the one real event. 4 of 19 train neutralisations deployed
  before lap 6, 1 more on lap 7. Fix (future work): a shorter anchor or a prior-race /
  qualifying pace fallback for the opening laps, validated like the Session 2 model.
- Early call: 17 of 20 cars gained a pit window from the 61 s lead; for none was the
  no-call window outside the VSC (213 s). The call confirmed earlier, it changed no car's
  options. 32.6 s (Fast Flag median lead) is 0.33-0.36 laps at the holdout tracks.
- False calls: one in the holdout (Singapore, SC call about lap 44, no neutralisation).
  16 of 20 cars would have pitted on it, median free-air gain +1.0 s (in their window
  anyway). Japan: no recs; Abu Dhabi: yellows only.
- Break-even (724 hypothetical calls, every 10th lap, every car): pitting wins even if the
  call is false in 38%; loses even if real in 44-58%. At phi 0.08 acting on Fast Flag's
  precision beats ignoring in 44% (SC) / 49% (VSC), but the call changes the decision in
  only 6 / 11 points of that. Every p* is an upper bound (track position not priced).
- 2025 US no-call path (synthetic, P = 1 at the deployment): all 15 decidable cars
  "overlapping"; pitting at lap 7 wins on the median only for soft starters. Soft + 0.075
  flips 1 of 15. 41% of plan laps are beyond h = 30 (floor).
- Two analysis bugs found and fixed mid-session, both from consistency checks on the
  break-even table: (1) pit-now plans ignored the stop at pit_lap (035cd85); (2) the
  false-call branch dropped the would-be neutralised laps (d28803b; SC and VSC "pit anyway"
  disagreed 7% vs 38%, now both 38%). Every number above is after both fixes.

## Session 6 results (done): dashboard, README, hygiene
- README rewritten: what the system does before what it cannot do; n = 1 framed as the
  binding constraint (and why the break-even work is built off 724 hypothetical calls so it
  does not rest on n = 1); a section on how the two Session 5 analysis bugs surfaced (a
  result too clean to believe, then an internal inconsistency), including that the
  mechanism tests would not have caught either; three-command demo note.
- src/dashboard.py + dashboard/ (FastAPI, vanilla JS, no CDN): lap-by-lap replay of
  2025 United States and Singapore. Nothing after the current lap renders. Decision cards
  show P(pit better) as a range over rho 0 / 0.5 / 0.9, gains if real / false, both plans,
  the soft p50 sensitivity and the h > 30 floor note. "overlapping" is neutral grey, never
  coloured as a recommendation. The blind window renders as "no decision: model has no
  anchor yet (laps 2-4 skipped)" on all 20 cars at the US call. A call's outcome appears
  only once the replay passes the matching window, so the Singapore false call teaches
  itself without leaking.
- src/dashboard_data.py writes committed snapshots to data/demo/ (about 200 KB), so the
  demo runs from a fresh clone with no FastF1 download and no Fast Flag timeline.
- Fix: decide() early returns now carry soft_bias_s. Without it the 20 blind-window cars
  were dropped by any filter on that field (they vanished from the export and the report
  printed an empty count). No analysis numbers changed.
- Hygiene: scipy added to requirements (src/live.py imports it directly; it was only
  present transitively via scikit-learn), fastapi and uvicorn added, stray ingest.log
  untracked, .claude/ and *.log ignored. Fresh clone verified: 55 tests pass and the
  dashboard serves with no data rebuild.
- NOT done: push to GitHub (needs interactive credentials; run `git push origin main`),
  and CI therefore unverified since dcd4b22.

## Session 6 brief: pit-wall dashboard, README, demo
The demo is what makes Sessions 4-5 showable. Protect the time; build nothing new in the
models.

Dashboard (FastAPI + a static page, uvicorn, localhost):
- Replay a holdout race lap by lap using ReplayDriver (never shows the future): running
  order, tyres, Fast Flag recs as they arrive, official SC / VSC.
- On a call or a deployment, show the decide() output per car: P(pit better) at the three
  rho values as a range, gain distributions (p10 / p50 / p90), the verdict
  ("overlapping" shown as such, never coloured as a recommendation), and the accounts-for /
  does-not-account-for lists on screen.
- Show the blind window explicitly: before the anchor exists, the card says "no decision:
  model has no anchor yet (laps 2-4 skipped)", not a blank.
- Demo races: 2025 United States (real VSC, lap 5 call, blind window) and 2025 Singapore
  (false SC call). Precompute decisions (backtest_decisions.parquet) so the page is fast.
- Arduino "BOX" board: optional, only after the page works. Fast Flag's serial protocol is
  in PROJECT_BRIEF 7.7 of the Fast Flag repo; ours would be a separate one-line message.

README (what a reviewer opens first):
- Top: what it is, the n=1 statement in the framing words above, the blind-window finding
  as a finding (not under limitations).
- Credits: Fast Flag is a separate two-person project (Naman, main builder; Ishaan) that
  this project consumes as an input; the strategy work here is solo. The 0.435 SC precision
  is Fast Flag's own measured out-of-sample number.
- Honest limitations: engine failed validation (traffic), SC pit loss phi stated not
  measured, soft p50 bias, h=30 floor, pass model over-predicts on unseen tracks.

Carries: none of the Session 4 optimizer work exists; do not demo strategy rankings.

## Session 9 results, item 2 (done): the DP optimizer runs, and what it is worth

src/optimizer.py. The first actual strategy recommendation the project has produced. Exact DP,
no heuristics, 458 car-races over 28 of the 29 train races, from DECISION_LAP = 10 to the flag.
READ THE MODULE DOCSTRING BEFORE ANY NUMBER HERE: it prices traffic, overtaking, track
position and rivals' reactions at ZERO, and plans on green pit loss.

### Results
                                      measured   neutralised   actual
  stop-count agreement with the team     69.1%        69.1%
  optimiser stops / actual stops          1.36         1.33      1.61
  modelled gain over the team's plan    2.5 s        2.7 s
    p10 / p90                        0.3 / 12.9   0.2 / 14.1
  no-neutralisation subset (169 car-races, 10 races): agreement 69.2%, gain median 2.9 s
"neutralised" = tyre p50 + 0.09 s on MEDIUM and SOFT beyond h = 15, the Session 8 spec.
14 of 458 car-races are not comparable: the team's actual plan is not scorable under the model
(a stint or tyre age outside the grid). 2025 Qatar is dropped entirely because NO car has an
anchor at lap 10 there, anchors existing at lap 6 and then from lap 12.

### TAKE 1: the model thinks teams were within about 2.5 s of free-air optimal
Over roughly 50 remaining laps, median 2.5 s, p90 12.9 s. That is a small number and it is the
most believable thing here. It says the free-air part of strategy, which is all this optimizer
sees, is close to solved by the teams, and that whatever separates a good strategy call from a
bad one lives in the part the optimizer prices at zero. An optimizer that found 30 s lying
around would have been evidence against itself.

### TAKE 2: stop COUNT is robust to the bias knob, COMPOUND CHOICE IS NOT
Stop count barely moves when the knob is neutralised (1.36 -> 1.33, agreement identical at
69.1%), but 63.3% of plans change, and the compounds swing violently:
                      HARD   SOFT   MEDIUM
  optimiser measured   295    224      102      HARD 47% of stops
  optimiser neutralised 530     55       26      HARD 87%
  what teams actually ran 404    122      199      HARD 56%
The reason is mechanical: the Session 8 bias is HORIZON shaped and spares HARD, so penalising
MEDIUM and SOFT beyond h = 15 penalises most of a plan's laps on two of three compounds and
hands the race to HARD. THE TWO RUNS BRACKET, NEITHER IS THE ANSWER. Compound recommendations
from this optimizer are not trustworthy and should not be quoted; the stop count is the only
part of its output that survives its own sensitivity.
Note also that the measured run over-picks SOFT against the teams (224 against 122) and
under-picks MEDIUM (102 against 199), which is the direction the old soft-bias story predicted
even though that bias is now measured much smaller.

### TAKE 3: the Session 4 bias comparison answers NEGATIVE
The brief said to run the comparison with the knobs measured and neutralised, because "the
part of the gap that closes is attributable to our bias". Nothing closes. The gain widens
slightly (2.5 -> 2.7 s) and stop-count agreement is identical. The tyre p50 bias is not what
separates the optimizer from the teams.
And the biases that most likely do are the ones that CANNOT BE SWEPT HERE. The pass model is
21% high overall and about 74% high at unseen tracks, and this objective has no traffic term
for either knob to act on. So the bias sweep the brief specified is incomplete by construction,
not by omission: an optimizer that ignores overtaking cannot be made sensitive to the error in
its overtaking model, while still emitting plans that need overtaking.

### Also not driven by the Safety Car confound
The no-neutralisation subset (10 races with no SC or VSC at all) gives 69.2% agreement and a
2.9 s median gain, against 69.1% and 2.5 s on everything. So planning on green pit loss while
teams could pit under a real neutralisation is not what produces the disagreement either.

### Gaps in the optimizer itself, recorded not fixed
- NO tyre_limit_laps. rules.strategy_violations supports a per-set lap limit (2023 Qatar) and
  the optimizer never passes one, so on a race with such a limit it would emit illegal plans.
  No illegal plan was emitted here only because the one Qatar race in train is dropped for
  lack of an anchor. Close this before running the optimizer on a limited-set race.
- NO Monte Carlo over SC timing, rival strategies or lap-time noise (Session 4 brief). The
  headline is a single deterministic plan per car on green pit loss.
- The decision lap is fixed at 10 and drops any race without an anchor there.
- h > 30 intervals are a floor, and about 40% of plan laps sit beyond h = 30, so the p50 the
  DP minimises is least reliable exactly where most of the plan lives.

## Session 9 results, item 1 (done): pass model refit on causal coverage

The complete fix named in Session 8. The pass model's covered branch was estimated on a
population defined with information the engine does not have, and the engine always takes
that branch. Both the FLAG and the VALUE are now causal: src/traffic.py adds
free_pace_causal, an expanding median over the stint's clear-air laps STRICTLY BEFORE this
lap, and free_delta_causal from it. src/overtake_model.py fits on that.
Only making the flag causal would have left a causally-covered row carrying a delta computed
from future laps, which is a second leak the Session 8 note did not name.
PASS_WHOLE_STINT_PACE=1 refits the old way, which is how the control below was run.

### The refit is correctly specified and it makes the engine WORSE
Isolated properly: both runs on the same 29-race data, only the pass model differs. The
Session 8 figure of 60.6 passes was measured on 23 races and is NOT comparable to either.
                              whole-stint    causal    actual
  scored passes per race          55.8        62.6      35.1
  battle laps per race           468.2       461.4     418.9
  mean p vs blended rate, 0-0.5    1.22        1.33      1.00
                          0.5-1    0.71        0.94      1.00
                          1-1.5    1.03        1.29      1.00
                          1.5-2    1.19        1.87      1.00
  bins within the +-15% criterion  1 of 4      1 of 4
PRIMARY CRITERION (pre-stated): engine mean predicted p per battle lap within 15% of the
BLENDED actual rate per bin, because the causal partition is a partition of the same rows, so
a correctly specified two-branch model at the true coverage mix must reproduce the blend.
MISS, 1 of 4 bins, and the refit improves only the 0.5-1 bin (0.71 -> 0.94) while worsening
the other three. Nothing was tuned.

PREDICTIONS STATED BEFORE THE RUN, one falsified:
1. WRONG in attribution, right in magnitude. I predicted passes would fall to about 56. The
   control is 55.8, almost exactly that, but the fall came from the RACE-SET EXPANSION
   (60.6 on 23 races -> 55.8 on 29) and the causal refit moved it back UP to 62.6. Predicting
   the right number for the wrong reason is how a confounded comparison passes unnoticed; the
   only thing that caught it was running the control.
2. Right: AUC fell, 0.9240 -> 0.9167, since causal coverage is 21.4% of rows against 51.3%.
3. Right: the unseen-track defect did not move, 0.1360 / 0.0770 against 0.1338 / 0.0770. It
   is untouched by this and remains the pass model's largest known error.

### Why it is kept anyway, and what it changes
Each branch now serves its own population. At 0-0.5 s: covered observed 0.516 and predicted
0.555, missing observed 0.272 and predicted 0.315, with coverage shares 16.7 / 15.8 / 23.8 /
32.8% matching causal_coverage exactly. That is a correctness property, not a performance one,
and it decides the question the same way it did for the Session 8 routing.

THE MECHANISM FOR THE WORSE NUMBER IS IDENTIFIED. The engine over-covers in every bin
(simulated 0.246 / 0.211 / 0.295 / 0.335 against causal 0.167 / 0.158 / 0.238 / 0.328) and the
refit moved the two branches further apart (branch_mismatch 1.41 -> 1.65 at 0-0.5, 1.69 ->
2.13 at 0.5-1), so the same coverage error costs more than it did. TWO ERRORS WERE PARTIALLY
CANCELLING: branches too close together, and an engine that takes the hot branch too often.
Removing the first exposed the second. Reverting would restore the cancellation, which is the
worst available reason to prefer a model.
CONSEQUENCE: the Session 8 coverage miss is now the BINDING defect rather than a footnote. It
was +7.3 pp with no identified mechanism and is +7.9 pp now, and the pass count hinges on it.

### Stale input found, not changed
DIRTY_AIR_D0_S is hardcoded 0.35 from 23 races. On 29 the aero-only median at 0-0.5 s measures
0.311. It was held at 0.35 so the pass model was the only thing that moved in this comparison.
It is inside the stated sweep (0.24 to 0.66) but it is now a stale measurement feeding the
engine, and it should be revisited on its own.

## Session 8 results, item 1 (done): free_missing routing

Built the routing identified in Session 7. The engine now tracks, per car, whether it has had
a green non-pit lap more than traffic.FREE_AIR_GAP_S (3.0 s) behind the car ahead SO FAR in
its current stint, resets that on every stop, and sets free_delta to NaN for a battle pair
unless BOTH cars qualify. The pass model then takes its missing-flag branch, as it would have
in the data. Params.route_missing, default True; python -m src.engine --no-route-missing
reproduces the pre-Session 8 behaviour exactly (no RNG draws were added, so the off path is
bit-identical to the Session 7 run).

### PARTIAL FIX BY CONSTRUCTION, not a fix that happened to fall short
The pass model's free_missing flag was fitted NON-CAUSALLY. traffic.py builds free_pace from
the clear-air laps of the WHOLE stint, so a car that only gets clear air later in a stint is
already "covered" at the start of it. A forward-running engine cannot reproduce that, and
nothing in this session changes it. Measured on train battles at 0-0.5 s:
    whole-stint covered    45.4%        causal covered      53.7%
    whole-stint uncovered  19.1%        causal uncovered    27.4%
Routing causally moves the engine from querying a 45.4% branch for every pair to querying a
19.1% branch for about 83% of them, while the population it is standing in for passes at
27.4%. The mismatch INVERTS rather than closing. The complete fix is a refit of
data/models/pass_model.json with a causal coverage flag (src/traffic.py causal_coverage is
the definition). THAT IS THE NEXT STEP ON THE PASS MODEL AND IT WAS NOT ATTEMPTED HERE.

### Success criterion, fixed before the run: MISS, 3 of 4 bins
Simulated coverage within 5 pp of the real CAUSAL rate per bin, not the whole-stint rate.
Threshold and stint window were NOT tuned.
    bin      whole-stint   causal   simulated   error
    0-0.5          0.482    0.167       0.240   +7.3 pp   MISS
    0.5-1          0.441    0.163       0.178   +1.5 pp
    1-1.5          0.555    0.237       0.246   +0.8 pp
    1.5-2          0.644    0.330       0.302   -2.8 pp
Checked and excluded as the cause of the 0-0.5 overshoot: the engine's clear-air test cannot
model is_clean's deleted, inaccurate and outlier exclusions, but those account for only 1.4%
of the laps the engine would count as clear air, nowhere near 7.3 pp. MECHANISM NOT
IDENTIFIED. It sits in the same bin as the open Session 7 finding and is not to be hunted
without a mechanism to test.

### Effect on the engine (reported whether or not it helps, per the brief)
Scored passes 72.0 -> 60.6 against 35.1 actual, so 2.05x -> 1.73x. Decomposition, adjacent
pairs per race:
    actual                     31.3
    sim laps, blended rate     56.6
    sim laps, covered rate     82.8     the extreme the engine used to sit at
    sim laps, SIM rate         65.4     was 79.7 before routing
The routing did what it was designed to do: it moved the engine off the covered extreme and
roughly two thirds of the way to the blend. It did NOT fix the engine.

Battle-lap distribution per race, before -> after against target:
    0-0.5   133.1 -> 134.2   (63.2)   2.11 -> 2.12   unchanged, still the whole failure
    0.5-1   148.7 -> 174.7  (162.5)   0.92 -> 1.07   crossed over
    1-1.5    98.2 -> 101.1  (102.3)   0.96 -> 0.99
    1.5-2    69.0 ->  67.4   (87.0)   0.79 -> 0.78
    total   448.9 -> 477.4  (415.0)   1.08 -> 1.15   WORSE
Finishing order slightly worse: Spearman 0.951 -> 0.941, mean position error 1.17 -> 1.27,
median gap error 12.0 -> 13.7 s, winner right 67% -> 69%.
The demoted following curve now reads 0.82 at 0-0.5 against a target of 0.80 and 0.78 at
0.5-1 against 0.33. The first is not evidence of anything (see why it was demoted) and the
second got considerably worse: fewer passes means cars are held longer.

KEPT ON BY DEFAULT ANYWAY. The brief is right that this is a defect in how every consumer
queries the pass model, independent of the engine, so correctness decides it rather than the
headline number. Note the Session 7 corollary: the direction a number moves is no evidence
about a fix, and here it moved both ways at once.

### Two facts about coverage worth carrying
- COVERAGE IS MOSTLY STINT POSITION, NOT GAP (3.4% over laps 1-3 of a stint against 47.5%
  past lap 21, a 14x range, against roughly 2x across gap bins). This is the most important
  thing in Session 8 and it is PROMOTED to "## ENGINE LIMITATION" near the top of this file.
  Do not read it only as a Session 8 detail.
- WHY BOTH CAUSAL RATES ARE HIGHER THAN THEIR WHOLE-STINT COUNTERPARTS. It is arithmetic, not
  an effect. Four cells at 0-0.5 s:
      whole-stint uncovered, causal uncovered   n 753   rate 0.191
      whole-stint COVERED,   causal uncovered   n 458   rate 0.410
      whole-stint covered,   causal covered     n 242   rate 0.537
      whole-stint uncovered, causal covered     n   0             (confirms causal is a
                                                                   strict subset, no leak)
  The "covered later, not yet" group sits BETWEEN the two whole-stint rates. Reclassifying it
  removes the lowest-rate members from the covered group (0.454 -> 0.537) and adds the
  highest-rate members to the uncovered group (0.191 -> 0.274), so both averages rise from
  one group moving. There is nothing further to read into it.

### New in the code
- src/traffic.py causal_coverage(state): the single definition, per (race, driver, lap), of
  whether a car has had clear air so far in its current stint. Use it rather than reinventing
  it; the docstring records how it differs from free_pace and by how much.
- src/engine.py: Params.route_missing, clear-air tracking in simulate(), covered flag in the
  pair log, and a COVERAGE ROUTING table in the run output carrying the criterion.

## Session 8 results, item 2 (done): race set expanded, pipeline rebuilt

27 races (23 train / 4 holdout) -> 38 races (29 train / 9 holdout). 15 new 2025 races
screened, 2 wet-excluded automatically, 2 more excluded for data quality found during the
rebuild (below). Split by ROUND PARITY, never by deployment count. Holdout 9 races with 6
SC / VSC deployments; see "## VALIDATION CEILING" for why 6 is enough and why more would not
have helped.

### Against the Session 1 to 3 baselines
                                was        now     reading
  races                      27          38
  clean lap share            87.9%       87.7%    resampling, not a change
  green pit loss, median     22.94 s     22.30 s  29 (season, event) rows, was 23
  fuel K                     3.249       3.206    gate passed; per-lap 0.0565 -> 0.0558
  K by season        3.18/3.26/3.32  3.18/3.26/3.18
  tyre below p10 / above p90 11.6/9.8%   10.7/10.2%  IMPROVED, both nearer the 10/10 target
  tyre MAE p50               0.508       0.506    flat
  tyre race offset sd        0.17        0.168    flat
  soft p50 bias             +0.075 s    +0.022 s  largely gone, see below
  pass AUC / Brier        0.926/0.045  0.924/0.0435  flat
  pass over-prediction        15%         21%     WORSE, see below
Train grew 26%, so clean share, pit loss and K move for resampling reasons and none of those
moves means anything about model quality. They are recorded so the next session does not read
them as drift.

### RESOLVED: the K season drift was small-sample
Session 2 recorded K by season as 3.18 / 3.26 / 3.32 and said "slight upward drift, watch it".
2023 and 2024 are unchanged because their races are unchanged; 2025 moved 3.32 -> 3.179 once
it had more than five races. There is no drift. Stop watching it.

### Tyre model: calibration IMPROVED on a larger, more varied train set
10.7% below p10 and 10.2% above p90, against 11.6 / 9.8 on 23 races. Better at every horizon,
h=1 10.1/9.3 and h=30 11.7/10.9 (was 13.1/10.5). Width, MAE and race offset sd are flat.
  CORRECTION, and it was the stated reason for doing this work. The Session 8 brief said the
  number that tests something is "tyre calibration on 9 holdout races instead of 4". THAT IS
  NOT WHAT THIS IS and it is not what the expansion produced. src/tyre.py never loads the
  holdout at all: it prints "holdout rows: 0", and that is correct, because the holdout
  touches nothing but the backtest. 10.7 / 10.2 is GroupKFold over the 29 TRAIN races. The
  improvement is real and it is a different improvement: the folds now leave out more
  distinct tracks, so the held-out-fold population is harder than it was on 23 races.
  REMAINING GAP: the tyre model has never been scored on the holdout, on 4 races or on 9.
  Nothing in Sessions 1 to 8 measures its error on a race no fit has seen. Closing that means
  deciding whether scoring the tyre model on the holdout is allowed under "holdout touches
  nothing but the backtest"; evaluation is not fitting, but the rule has been read strictly
  so far and should not be loosened silently.

SOFT BIAS LARGELY GONE, AND REPLACED BY A HORIZON BIAS. Soft p50 median residual is +0.022 s,
against +0.075 recorded in Session 2, with MEDIUM +0.007 and HARD +0.008. The Session 2 soft
bias was substantially a small-sample artefact of 23 races. What is there now is different and
HORIZON shaped, not compound shaped:
    h=30   HARD -0.020   MEDIUM +0.087   SOFT +0.091
    h=15   HARD +0.013   MEDIUM +0.009   SOFT +0.037
So long-horizon p50 is slow by about 0.09 s on the two softer compounds and not on HARD.
Anything that was going to apply a soft + 0.075 sensitivity should now apply a long-horizon
sensitivity on MEDIUM and SOFT instead. The old correction is the wrong size and the wrong
axis.

### Pass model: overall over-prediction got WORSE, and I predicted the opposite
21% over on train battles (1097 predicted against 910 observed), against 15% on 23 races. I
said before the rebuild that it should improve because 6 of the new train races are at tracks
the model had never seen. That was wrong and is recorded as wrong.
The per-population numbers show why the headline moved:
    seen tracks     predicted 0.0775  observed 0.0743    4% over  (was 1% over)
    unseen tracks   predicted 0.1338  observed 0.0770   74% over  (was 71% over)
Per population almost nothing changed. The overall figure is a mixture of the two, and adding
distinct tracks raises the share of CV rows that sit at a track the fold never saw, so the
worse mixture is mostly composition, not a worse model. Free-air coverage 51.3%, was 51.8%.
The unseen-track defect is unchanged at about 70% over and remains the pass model's largest
known error, alongside the covered-branch selection in "## ENGINE LIMITATION".

### THREE DATA DEFECTS IN THE 13 NEW RACES, none of which the existing gates caught
The original 27 races had none of these, which is exactly why no gate existed for them.
1. 2025 MIAMI, excluded. FastF1 carries no compound, tyre_life or stint for laps 1-24 for 19
   of 20 cars, and where the feed resumes the tyre counter RESTARTS: 15 of 20 drivers show an
   age below the laps they had already completed with no pit stop before it. The data is
   wrong, not missing. Surfaced as "SVD did not converge" inside fuel.py. Dropping the NaN
   laps, the obvious fix, would have kept 559 laps with ages understated by up to 24.
   Now caught by ingest.tyre_counter_resets. Checked across all 40 races: Miami only.
2. 2025 BELGIAN, excluded. A drying track after a wet delay, which looks dry to a compound
   test and is not dry to any lap-time model: no INTERMEDIATE appears in its recorded laps.
   Field median clean lap 122.8 s at lap 5 and 107.9 s by lap 22. Surfaced as fuel.py's own
   gate refusing to write, with Belgian showing a per-race slope of 1.2169 s/lap against a
   median of 0.0571. Now caught by clean.early_lap_excess_s at DRYING_EXCESS_S = 6.0 s, a
   threshold set from the measured distribution (39 other races: median 2.08 s, max 3.52 s;
   Belgian 16.04 s) and above what K plus the early-stint effect can produce, under 4 s.
3. STALE RAW DIRECTORIES, self-inflicted. Screening candidate races through a throwaway
   config wrote into data/processed/raw/, so a race the config no longer listed stayed on
   disk and every raw/ scanner crashed in split_of. splits.is_configured now skips them and
   clean.py prints what it skipped. The crash was the mild outcome: a scanner that had
   tolerated the missing split by guessing would have folded an excluded race into a fit.
   IF SCREENING CANDIDATES AGAIN, ingest into a separate tree or clean up afterwards.

Both race defects had a cheap fix that would have left the corruption in place and the
pipeline running. Both were caught by a gate or a crash pointing at a race rather than at
code. Expanding a dataset by 48% found more bad data than bad code.

### Session 5 backtest re-run on the 9-race holdout (hypothetical calls only)
1616 hypothetical calls, against 724 on 4 races. Full report: docs/session5_results.md.

THE REAL-CALL PATH DID NOT MOVE AND CANNOT. The backtest printed the ceiling itself: 5 of the
9 holdout races have no Fast Flag timeline and were skipped by name. Of the 4 that have one,
Abu Dhabi and Japanese produce 0 call episodes, Singapore 1 (the false call) and United States
1 (the real VSC). Still ONE real neutralisation. The report now carries a labelled section,
"Two different sample sizes in this report, do not mix them", so a 2.2x larger break-even
sample cannot be read as 2.2x more validation of the live decision.

Break-even at phi 0.08, then -> now:
    pit anyway regardless of the call      38%  ->  33%
    never pit even if the call is real     44-58%  ->  SC 61%, VSC 51%
    acting on FF precision beats ignoring  SC 44% / VSC 49%  ->  SC 38% / VSC 44%
    of which the CALL changes the decision SC 6 / VSC 11 points  ->  SC 5 / VSC 11 points
THE CONCLUSIONS ARE STABLE under a 2.2x larger and more varied sample, which is the useful
result: the Session 5 finding that the call changes the decision in only about 5 to 11
percentage points was not an artefact of four races. Every p* remains an upper bound because
track position is still not priced.

Two smaller changes from the rebuilt models:
- The new long-horizon sensitivity (p50 + 0.09 on MEDIUM and SOFT beyond h = 15) flips 0 of 15
  cars at the real event, where the dead soft + 0.075 flipped 1 of 15. The sensitivity is
  milder here because the US decision window is short-horizon.
- Singapore false call: 14 of 20 cars would have pitted at a median +1.3 s free-air gain, was
  16 of 20 at +1.0 s.

### Not done in Session 8
- The tyre model has still never been scored on the holdout. See the correction above.
- The engine's 0-0.5 s battle-lap excess (2.12x) still has no identified mechanism, and by
  user decision is not to be hunted without one to test.
- DONE in Session 9: the pass model refit on causal coverage. It is correctly specified
  and made the engine worse, and it promoted the engine's coverage error to the binding
  defect. See "Session 9 results, item 1".
- DIRTY_AIR_D0_S is stale: 0.35 from 23 races, 0.311 measured on 29.

## Session 8 brief: free_missing routing, then expand the holdout

### 1. Route cars with no clear-air history to the uncovered branch
Session 7 Finding 2: the pass model's covered branch was fitted on the easier-passing half of
the data (covered pairs pass 2.4x more often at 0-0.5 s), and the engine always takes it
because it always has a pace estimate. Worth 1.46x of the pass miss. This is a defect of
data/models/pass_model.json that exists independently of the engine, so FIX IT WHETHER OR NOT
IT HELPS THE ENGINE MISS.

The engine already has what it needs: per (driver, stint), whether any lap so far ran with
more than traffic.FREE_AIR_GAP_S (3.0 s) of clear air. Cars without it get free_missing = 1
and free_delta = 0 (what overtake_model.features does with a missing delta).

IDENTIFICATION RISK, STATE IT BEFORE RUNNING. The real coverage flag is not causal.
traffic.py builds free_pace per (driver, stint) from the clear-air laps of the WHOLE stint, so
a pair early in a stint that only gets clear air later is "covered" in battles.parquet. The
engine, running forward, cannot know that. So the engine's causal coverage will come out
BELOW the real 48 / 44 / 56 / 64% by construction, and an exact match is not the right
success criterion. Decide before running what the criterion is, and do not tune the clear-air
threshold or the stint window to close that gap: the gap is the causality difference, not an
error. If a causal version of the real coverage rate is wanted for comparison, recompute it
from traffic.py with an expanding window and report both.

Validate on its own terms first, then on the engine:
- simulated covered / uncovered split per gap bin against the real rates (48 / 44 / 56 / 64%),
  and against the causal recomputation if built
- then the battle-lap distribution (133.1 / 148.7 / 98.2 / 69.0 vs 63.2 / 162.5 / 102.3 /
  87.0) and the pass count (72.0 scored vs 35.1)
REPORT BOTH EVEN IF THE PASS COUNT BARELY MOVES. The decomposition says the branch is worth
1.46x and lap counts 1.77x, so a correct fix here cannot close the miss on its own.

Do NOT hunt the 0 to 0.5 bin further without a mechanism to test (user decision, Session 7).
Finding 1 stays open with the mechanism unidentified.

### 2. Expand the holdout
The decision path is n = 1: one real neutralisation (2025 US VSC lap 7) across four holdout
races, so nothing downstream of it is validated. The ingest pipeline already handles wet
compound exclusion automatically.
- Add races to config/races.yaml. 2023 to 2025 has far more dry races than the 27 currently
  there (23 train + 4 holdout).
- Target a holdout with at least 8 real SC or VSC deployments. CHECK DEPLOYMENT COUNTS FROM
  THE DATA, not from memory, before fixing the split.
- Keep the split logic in src/splits.py. Any race already used in a fit STAYS IN TRAIN (phi,
  pit loss medians, SC rates, tyre models, K, pass model). New races can go either way.
- TELL THE USER THE PROPOSED SPLIT AND ITS NEUTRALISATION COUNT BEFORE REBUILDING ANYTHING.
- Then rebuild: ingest, clean, pitloss, safety_car, overtakes. Report whether the Session 1
  to 3 numbers hold: clean-lap share (was 87.9%), green pit loss (was 22.94 s), fuel K (was
  3.249 s per race of progress), pass model calibration especially at tracks now seen more
  than once.
- Retrain the tyre model and report held-out calibration against Session 2 (11.6% below p10,
  9.8% above p90). More races and more unseen tracks make this the real generalisation test.
  IF CALIBRATION DEGRADES THAT IS A FINDING, NOT A REGRESSION TO FIX.
- Re-run the Session 5 backtest on the larger holdout. Break-even, false-call cost and
  blind-window counts only mean something at n > 1.
- Ingest is roughly 1 to 2 minutes per race, so start it early and in the background.

Carries: holdout touches nothing but the backtest; no clamping or post-processing; say so if a
spec is not identified from the data; stop and report rather than iterating past a second
attempt; commit and push after each piece.

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
- SUPERSEDED IN SESSION 8. Was: soft p50 + 0.075 s at every horizon. Now: p50 + 0.09 s on
  MEDIUM and SOFT beyond h = 15, nothing on HARD. Re-run and report whether choices flip.
- h = 30 tyre intervals are a floor, not an estimate: survivorship (only laps still clean
  30 laps later are in that bucket). Do not read long-horizon width as the true uncertainty.
- Pass model over-predicts on unseen tracks (15.2% vs 8.9%) and in its top bucket (0.75 vs
  0.64): sensitivity run with pass probabilities scaled down, and report Singapore (the
  holdout's unseen track) separately.
- Dirty air d0 sweep 0.24 to 0.66 in the Monte Carlo.
- Holdout races touch nothing until Session 5's backtest.
