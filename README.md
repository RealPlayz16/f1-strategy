# F1 Race Strategy: live Safety Car decisions under uncertainty

A Formula 1 strategy system built on FastF1 data. It consumes a live, predictive race-control
feed ([Fast Flag](https://github.com/Pseudocoder28/Fast-Flag)), and when that feed calls a
Safety Car or VSC it answers, for every car on track: pit now, or stay out?

The answer is a probability with its spread, not a verdict, and it carries an explicit list of
what it does and does not account for.

**The strategy work in this repository is Ishaan's, solo**: the race engine, the tyre and pass
models, the DP optimizer, the live decision path and the dashboard. Fast Flag is a separate
project by Naman that this one consumes as an input and never modifies; the boundary is set out
in [Fast Flag, and who built what](#fast-flag-and-who-built-what).

## The headline: teams were within about 2.5 seconds of optimal

The DP optimizer plans the rest of a race from lap 10 and compares its choice with what the
team actually did, over 458 car-races on 28 training races. The model's verdict is that the
teams left a **median of 2.5 seconds** on the table across roughly fifty remaining laps (p90
12.9 s), and it agreed with their number of pit stops 69% of the time.

That small number is the believable one, and the reason is worth stating plainly: **an
optimiser that found 30 seconds lying around would have been evidence against itself.** Real F1
strategists are extremely good at the part of this problem that can be written down. A model
claiming otherwise would be describing its own errors, not their mistakes. What the result
actually says is that the free-air part of strategy is close to solved, so whatever separates a
good call from a bad one lives in the part this optimizer cannot see.

### The 2.5 s is an upper bound on the free-air component only

This is not a caveat sitting beside the known biases. It inverts them.

Every earlier stage of this project worried that the optimizer would favour strategies needing
overtaking, because the pass model predicts about 21% too many passes overall and about 74% too
many at tracks it has not seen. The optimizer prices traffic, overtaking and track position at
**zero**, and pricing passing as free *is that same bias at infinite strength*. So the worry has
not been managed, it has been maximised.

Two consequences follow, and both bound the headline:

- The 2.5 s is an upper bound on the **free-air** gap, not on the real one. A plan that gains
  time by stopping into traffic has that cost counted as nothing.
- It says nothing at all about whether those plans are **executable**. The optimizer cannot tell
  a plan that works from one that needs three passes it would never complete.

It is also why the bias sweep the design called for is incomplete *by construction* rather than
by omission: an objective with no traffic term gives the overtaking-bias knob nothing to act on,
while still emitting overtake-dependent plans.

## What it does

- **Reads a predictive race-control feed.** Fast Flag's Safety Car and VSC calls arrive before
  race control's, with a measured median lead of 32.6 s. Each call is carried as a probability
  (Fast Flag's own out-of-sample precision), never as a deployed Safety Car.
- **Replays a race without leaking the future.** The replay driver serves the race as of a
  session time: laps completed, calls issued, neutralisations deployed, and nothing after it. A
  Safety Car still running does not reveal when it ends. This is tested directly: mutating
  every future row leaves the state byte-identical.
- **Predicts lap times with calibrated uncertainty.** A LightGBM median plus a quantile network
  trained on out-of-race residuals, anchored on the live race and horizon dependent. On
  held-out training folds, 10.7% of laps fall below p10 and 10.2% above p90 against a 10 / 10
  target. Scored once on the holdout itself, that becomes 18.4% below p10 and 10.7% above: a
  location error, not an interval error. See the limitations.
- **Measures pit loss per race** from pit lane transit, keyed by season and event, with the
  Safety Car discount as a stated, swept parameter rather than a fitted number that the data
  cannot support.
- **Decides under uncertainty.** Pit-now and stay-out plans are scored to the flag on free-air
  time, with per-lap quantiles carried through, pit loss swept, and the call's probability
  applied. The output is P(pit better) across three assumptions about how the two plans' errors
  correlate, plus the gain distribution. When the distributions overlap, the system says they
  overlap instead of picking a side.
- **Optimises the rest of a race.** An exact DP over lap, compound, tyre age, stops made and
  whether the two-compound rule is satisfied, minimising free-air time to the flag.
- **Backtests on nine holdout races** that touched no fit, and reports the cost of acting on a
  false call alongside the benefit of acting on a true one.

## Method: every defect in this project was found by two measurements disagreeing

This is the most transferable thing here, so it goes before the results it produced. Four real
defects, none of them caught by a test. The 71 tests in this repo check mechanisms
(no leakage, correct plan construction, quantile loss) and they would not have caught any of the
four.

1. **A pit loss of 84 seconds.** Lap-time subtraction said 84 s for a Safety Car stop at 2023
   Austria; the transit measurement and the physics said about 23. Cause: FastF1 starts a new
   stint on *any* pit lane passage, so 19 cars following the Safety Car through the pit lane
   were counted as pit stops, while race control's own message said "SAFETY CAR THROUGH THE PIT
   LANE". Three fixes patched the symptom first, one of them a clamp forcing the Safety Car loss
   below the green loss, before anyone looked for the cause.
2. **A break-even table too clean to believe.** It reported that pitting on a call never beats
   staying out, in 99 to 100% of situations. A real strategy question is not that one-sided.
   Cause: pit-now plans were scored on the old tyres to the flag while still paying for the
   stop, because the stop fell on the lap before scoring began. (`035cd85`)
3. **An internal inconsistency.** "Pitting wins even if the call is false" came out at 7% for
   Safety Car calls and 38% for VSC calls. If the call is false nothing is neutralised, so the
   number cannot depend on the call type. Cause: the false-call branch was still excluding the
   laps that would only have been neutralised had the call been real. (`d28803b`)
4. **73 passes drawn, 28 passes scored.** On one simulated race the engine drew 73 overtakes
   and only 28 appeared as position swaps. Cause: the condition that realises a pass was
   inverted, so a car that was comfortably ahead was never pushed back and the pass silently
   did not happen. (`506cb2a`)

What this buys that a test does not: it catches errors in the *measurement* as readily as in
the model, and it needs no knowledge of the right answer, only that two routes to the same
number must agree. A fifth case went the other way and is worth keeping for that reason. A
check of the optimizer's plans against the rules module failed, and the rules module was right
while the *check* was wrong, because it reconstructed a stint from lap 1 when the tyre was
already five laps old.

### Three corollaries, each learned the hard way

- **The direction a number moves after a fix is no evidence the fix was right.** Fixing the
  inverted pass condition made the headline *worse*, from 28 to 72 simulated passes against 35
  actual. It was still correct. Conversely the clamp in defect 1 made the pit loss look right
  and was wrong. Judge a fix on its mechanism, never on whether the number improved.
- **A plausible mechanism will absorb a real defect if you let it.** Defect 4 was about to be
  written up as a fundamental limit of the model, with a tidy physical story about cars needing
  to drop back and take a run at each other. The story fitted the symptom and was false. When a
  satisfying explanation arrives before the measurement, treat the satisfaction as a warning.
- **It needs two independent routes, and sometimes there is only one.** The engine's following
  model was validated against a single curve for two failed attempts, which is part of why the
  cause took so long to surface. Adding a second, independent target, the distribution of
  car-laps spent at each following distance, which nothing in the model was fitted to, was
  what finally localised it.

### A related habit: components that refuse rather than guess

Expanding the race set from 27 to 38 turned up three data defects, and **none of the 71 tests
failed at any point**. Each was caught by something declining to proceed: a crash inside a
least-squares call, a fit gate refusing to write a result outside its stated bounds, and a
lookup raising on a race it could not classify. Two of the three were invisible to every check
that existed, because the original 27 races had never contained that kind of corruption. Gates
do not get written until data breaks them.

Both of the data defects had a cheap fix that would have left the corruption in place and the
pipeline green. 2025 Miami carries no tyre data for its first 24 laps *and restarts the tyre
age counter* where the data resumes, so dropping the empty laps would have kept 559 laps whose
ages were understated by up to 24. 2025 Belgian is a drying track after a wet delay, which
looks dry to a compound test and is not dry to any lap-time model.

## Findings that are also limitations

### The system is blind exactly when an early call arrives

On the one real event, Fast Flag called the incident at the end of lap 5, **61 s before race
control deployed the VSC**. The tyre model could not use that lead. It anchors on at least two
clean laps from lap 5 onward (laps 2 to 4 are skipped because of a measured race-start effect
on lap times), so it cannot predict anything before about lap 6 or 7. The call arrived inside
that blind window and bought nothing.

This generalises. Across the 27 train-race neutralisations, 8 deployed before lap 6 and 3 more
on lap 7, so 11 of 27 land in or next to the blind window. Early Safety Cars are common, and
in F1 they are often the strategically important ones, because the field is bunched and nobody
has stopped yet (a track-position effect this model does not price). A system that cannot act
before lap 6 is blind in that window.

Future work, named: a shorter anchor, or a prior-race / qualifying pace fallback for the opening
laps, validated the way the tyre model was.

### The evidence ceiling is not mine to raise

The nine holdout races contain six real neutralisations, but only four of those races have a
Fast Flag timeline, and among those four there is exactly **one real neutralisation** (2025
United States, VSC on lap 7). So the decision path has n = 1, and that is the binding constraint
on every claim here: it is why this is a working live decision system demonstrated end to end
with honest uncertainty, and not a validated strategy system.

The obvious fix is more races, and it does not work. A Fast Flag recommendation exists for a
race only if Fast Flag has built a timeline for it. It has built five, four of which are already
these holdout races. Expanding my own pipeline was worth doing and was done: 27 races became
38, the holdout went from four to nine, the hypothetical-call sample from 724 to 1616, and the
tyre model could be scored on a real holdout for the first time. It moved n = 1 **not at
all**, because none of the five new holdout races has a Fast Flag timeline.

So the ceiling is set by a dependency I do not control. Raising it means running someone else's
pipeline over new races in their repository, which is a different project with its own owner and
its own compute. That is worth naming plainly, because it is a structural property of building
on an upstream system rather than an oversight in this one: **you inherit your dependency's
coverage as your own evidence ceiling, and no amount of work on your side of the boundary moves
it.** I found it by checking what the upstream actually had on disk rather than assuming my own
pipeline was the constraint, and that assumption had survived three sessions unexamined.

The analysis is built to be informative anyway, which is why the break-even work below runs off
1616 hypothetical calls rather than the single observed one.

### Two more results from the backtest

Full report: [docs/session5_results.md](docs/session5_results.md).

- **Early calls mostly confirm, they rarely change.** 32.6 s is about a third of a lap at the
  holdout tracks. On the real VSC, 17 of 20 cars got an extra pit window from the call, but for
  none of them did the later no-call window fall outside the VSC (it ran 213 s). A Safety Car
  (median 3 laps) outlasts a third of a lap easily; a short VSC is the only case where the lead
  can change a car's options.
- **Break-even precision: the call rarely decides the stop.** Over 1616 hypothetical calls
  (every 10th lap, every car, all nine holdout races), pitting now wins in 33% of situations even
  if the call is false (the car is in its window anyway), and loses in 51% (VSC) to 61% (SC) even
  if the call is real. Only in between does precision matter. At phi 0.08, acting on Fast Flag's
  measured precision beats ignoring it in 38% of situations for Safety Car calls and 44% for VSC
  calls, but only **5 points (SC) and 11 points (VSC)** of that are cases where the call changes
  the decision. Every break-even here is an upper bound, because track position is not priced.
  These numbers held when the sample grew: on four holdout races and 724 calls they were 38%, 44
  to 58%, 44 / 49% and 6 / 11 points. The conclusion was not an artefact of four races.
- **False calls.** Fast Flag made one false call across the holdout (Singapore, SC, late in the
  race). 14 of 20 cars would have pitted on it, at a median gain of 1.3 s in free-air time,
  because they were in their window anyway. Japan had no calls; Abu Dhabi only yellows.

## Results this project talked itself out of

### A recommendation made, then withdrawn by its own sensitivity

The optimizer's **stop count** is stable: neutralising the tyre model's known p50 bias moves it
from 1.36 to 1.33 stops per car and leaves stop-count agreement identical at 69.1%.

Its **compound choice is not**, and the sensitivity run is the only reason anyone knows:

| | HARD | SOFT | MEDIUM |
|---|---|---|---|
| optimiser, measured knobs | 295 (47%) | 224 | 102 |
| optimiser, bias neutralised | **530 (87%)** | 55 | 26 |
| what teams actually ran | 404 (56%) | 122 | 199 |

63% of plans change, and the HARD share nearly doubles. The cause is mechanical: the tyre bias
is horizon-shaped and spares HARD, so penalising MEDIUM and SOFT beyond fifteen laps ahead
penalises most of a plan on two of the three compounds. **The two runs bracket; neither is the
answer.** The compound recommendation was produced, tested against its own stated uncertainty,
and withdrawn on that evidence. Without the sensitivity run it would have shipped as a result,
and it would have looked like one: the measured-knobs row alone is close enough to what teams
ran to pass for a finding.

### Five predictions written down in advance, then refuted

Each was recorded before the run that tested it, which is the only reason any of them could
fail rather than be explained away afterwards.

1. **That the bias sweep would close the gap.** The design said to run the optimizer comparison
   twice, measured and neutralised, on the reasoning that *the part of the gap that closes is
   attributable to our bias*. Nothing closed: the gap widened slightly, 2.5 s to 2.7 s, and
   stop-count agreement was identical to a tenth of a percent.
2. **That fixing the pass model's coverage flag would reduce the engine's pass count.** It
   raised it, 55.8 to 62.6 per race. Only a control run on identical data showed that the fall
   I had predicted came from the larger race set instead. The right number for the wrong
   reason, which is how a confounded comparison passes unnoticed.
3. **That the tyre model's holdout degradation would be symmetric.** It was one-sided: 18.4%
   below p10 against a calibrated 10.7% above. I had the shape wrong, not just the size, and
   the shape is the informative part.
4. **That between-race variance would widen on the holdout.** It did not move at all.
5. **That unseen tracks would drive the degradation.** A split fixed *before* the run says they
   drive the median error (MAE 0.725 against 0.524) and not the calibration (18.8% against
   17.6%). With nine races and two groups there is always a split that flatters whichever story
   you arrived with; fixed in advance, it could refute, and it did.

All are kept in the record, because a prediction that fails tells you more than one that lands.

## What every decision accounts for, and what it does not

**Accounts for:** tyre model p10 / p50 / p90 per lap (anchored on laps up to the call); pit loss
by condition with phi swept 0.05 to 0.12; the call as a probability; the compound rule and stint
caps.

**Does not account for:** traffic, track position and overtaking; queue position under the
Safety Car and rivals' reactions; tyre set availability; the correlation between the two plans'
errors (P(pit better) is reported at three values instead); tyre intervals beyond 30 laps ahead,
which are a floor. On the real VSC, every car's answer was "overlapping: the model cannot
separate the two plans", and the system reports that rather than picking a side.

## Limitations, and what each one cost to find

Every item below is a measurement, not an apology. Most took a session to establish, several
survived an attempted fix, and three of them bound the headline result above. They are listed
because a limitation you can state precisely is worth more than one you have not found.

- **The 20-car race engine failed validation, three times.** It does not reproduce how cars
  follow each other. The first version over-penalised following; the second under-penalised it
  and tripled the passes; the third reworked it from a position rule into a pace constraint,
  which fixed the *direction* of the error and still over-predicts passes by 1.78x. The
  remaining defect is sharp and unexplained: the engine puts **2.12x too many car-laps within
  half a second of the car ahead**, and after excluding Safety Car restarts and the restart gap
  as causes, the mechanism is still not identified. It is recorded as open rather than guessed
  at. Consequence: nothing here makes a traffic or overtaking claim, and the optimizer prices
  traffic at zero rather than using this engine.
- **The tyre model is a location error out of sample, not an interval error.** 10.7% below p10
  and 10.2% above p90 on held-out training folds; scored once on the real holdout, 18.4% below
  p10 and 10.7% above, MAE 0.665 s against 0.506. The upper tail stays calibrated and only the
  lower tail fails, with a median residual of −0.099 s. A pooled miss rate cannot tell those two
  faults apart and they want opposite fixes. Monaco is much the worst race (MAE 1.374 s against
  0.542 elsewhere, both tails blown), which is this model failing at the one circuit where lap
  time is dominated by track position rather than tyre state: the missing physics surfacing
  where it cannot be ignored. That also predicts where else to distrust it: wherever position
  beats pace.
- **The tyre sensitivity knob is the wrong size, and deliberately left alone.** The holdout says
  the long-horizon bias wants roughly double for MEDIUM, a third for SOFT, and a large negative
  term for HARD that it does not get. Re-deriving it from the holdout would make every
  downstream sensitivity a function of the evaluation set, at which point it stops being a
  sensitivity. Its job was sensitivity analysis, not accuracy, so this weakens the results that
  used it without making them wrong. It makes them untested at the magnitude the holdout
  suggests.
- **Pit loss under a Safety Car is a stated assumption** (phi = 0.08, swept 0.05 to 0.12), not a
  fitted number. It is not identifiable from lap times with this data: on laps where a Safety
  Car starts or ends, stay-out lap times spread 22 to 31 s with running position, so the
  stay-out median is not any one car's counterfactual.
- **The pass model over-predicts on tracks it has not seen**, 13.4% against 7.7% observed, and
  21% overall. A refit that corrected how its two branches are specified made the engine worse
  rather than better, and was kept anyway because it is correct: two errors had been partially
  cancelling, and removing one exposed the other.
- **One real neutralisation on the decision path.** See the evidence ceiling above. This is the
  only limitation here that no amount of further work on this repository can lift.
- **Most components have still never been scored on the holdout.** The tyre model has, once.
  The engine, the pass model, the optimizer, the dirty-air constants, the fuel and degradation
  fits and the Safety Car rates have not. The gap is narrowed, not closed.
- **The Arduino "BOX" pit board** is planned, not built. The serial path would be a one-line
  message to an Uno, separate from Fast Flag's own protocol.

## Fast Flag, and who built what

Fast Flag is a **separate two-person project** (team We Are So Back, FormulaTech Hacks 2026),
built mainly by **Naman**, with Ishaan on the bridge and firmware. This project **consumes it as
an input**: it reads Fast Flag's precomputed timeline for each holdout race and never retrains
or edits Fast Flag. The strategy work in this repository, meaning the models, engine,
optimizer, decision path and dashboard, is Ishaan's, solo.

The call precision used here, **0.44 for SC calls (10 of 23) and 0.57 for VSC calls (12 of
21)**, is Fast Flag's own measured out-of-sample number from its check-race scorecard, not an
assumption made here. How this project consumes Fast Flag:
[docs/fast_flag_recon.md](docs/fast_flag_recon.md).

## Data

38 dry races, 2023 to 2025 ([config/races.yaml](config/races.yaml)). 29 train, 9 holdout
(2025 Abu Dhabi, Canadian, Chinese, Japanese, Las Vegas, Mexico City, Monaco, Singapore,
United States). Holdout races touch no fit.

Three rules decided the split, and none of them is "pick the interesting races":

- **New races were split by round parity**, odd to train and even to holdout. Choosing holdout
  races by how many Safety Cars they contained would have selected on the very thing the
  backtest measures.
- **Fast Flag's own training races are forced into train**, so its models never see a holdout
  race: 2025 Dutch, Azerbaijan, Belgian and Miami.
- **Two races were excluded for data quality**, each caught by a component refusing to run
  rather than by a test. 2025 Miami because FastF1's tyre counter restarts mid-race, making
  tyre ages wrong rather than missing; 2025 Belgian because it is a drying track after a wet
  delay, which looks dry to a compound test and is not dry to any lap-time model.

## See it work (three commands)

```bash
pip install -r requirements.txt
python -m src.dashboard
# open http://127.0.0.1:8000
```

No data rebuild and no FastF1 download: the two demo races ship as committed snapshots in
`data/demo/`, written by `python -m src.dashboard_data` from the backtest outputs. Run from
the repository root.

What to look at:

1. **2025 United States**, press *jump to the call*. The replay lands on the end of lap 6,
   where Fast Flag calls a Safety Car 61 s before race control deploys a VSC (right event,
   wrong kind). Every one of the 20 cars reads **"no decision: model has no anchor yet (laps
   2-4 skipped)"**. That is the blind window above, on screen.
2. Step one lap on to lap 7. Race control deploys the VSC, the system can now decide, and
   **15 of the 20 cars** come back **"overlapping: the model cannot separate the two
   plans"**, with P(pit better) shown as a range across the three correlation assumptions
   rather than a single number. The other five still have no anchor.
3. **2025 Singapore**, *jump to the call*, then step forward two laps: the Safety Car call
   turns out to be false, and the page says so only once the replay reaches the point where
   that would be known.

The page never renders anything after the current lap, and every decision on it was
computed by `src.live.decide` from the race state as of that moment.

## Rebuild the pipeline from scratch

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.ingest --set all
python -m src.clean
python -m src.pitloss
python -m src.safety_car
python -m src.overtakes
python -m src.fuel
python -m src.degradation
python -m src.tyre
python -m src.traffic
python -m src.overtake_model
python -m src.backtest
python -m src.backtest --break-even
python -m src.live_report
python -m src.engine
python -m src.optimizer
python -m src.tyre_holdout
pytest -q
```

The first ingest downloads race data into `data/cache/`. The backtest needs Fast Flag
timelines for the holdout races, built in a Fast Flag clone at `~/fast-flag` with
`python -m src.ingest.build --case <race>` then `python -m src.replay.timeline <race>`.

## Pipeline

| Module | What it does |
|---|---|
| `src/ingest.py`, `src/clean.py` | Laps, weather, race control, track and session status; clean-lap flags |
| `src/pitloss.py` | Green pit loss measured per race; SC / VSC loss from a stated phi |
| `src/safety_car.py` | SC / VSC / red episodes from the track status feed; deployment rates |
| `src/fuel.py` | Race-progress correction. K is fuel burn plus track evolution, not a fuel coefficient |
| `src/degradation.py` | Within-race degradation curves; the early-stint effect |
| `src/tyre.py` | Anchored, horizon-dependent lap time model: LightGBM median, quantile net for p10 / p90 |
| `src/traffic.py`, `src/overtake_model.py` | Free-air pace, dirty air (parameterised), pass probability |
| `src/rules.py` | Compound rule, DRS, overtaking under SC, pit loss by condition, stint caps |
| `src/engine.py` | 20-car engine (failed validation, documented) |
| `src/optimizer.py` | DP strategy optimizer; free-air objective, traffic priced at zero |
| `src/tyre_holdout.py` | One terminal scoring pass of the frozen tyre model on the holdout |
| `src/live.py` | Fast Flag hook, leak-tested replay driver, pit-now vs stay-out decision |
| `src/backtest.py`, `src/live_report.py` | Holdout backtest, break-even precision, results report |
| `src/dashboard.py`, `src/dashboard_data.py`, `dashboard/` | Pit-wall replay page and its committed demo snapshots |

Design history, every model decision and every failed check: [HANDOFF.md](HANDOFF.md).

## Credits

- [Fast Flag](https://github.com/Pseudocoder28/Fast-Flag): Naman (main builder) and Ishaan,
  team We Are So Back. Consumed as an input, see above.
- Race data from [FastF1](https://github.com/theOehrly/Fast-F1).

Everything here is a replay of historical data.
