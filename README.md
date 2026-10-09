# F1 Race Strategy: live Safety Car decisions under uncertainty

A Formula 1 strategy system built on FastF1 data. It consumes a live, predictive
race-control feed ([Fast Flag](https://github.com/Pseudocoder28/Fast-Flag)), and when that
feed calls a Safety Car or VSC it answers, for every car on track: pit now, or stay out?

The answer is a probability with its spread, not a verdict, and it carries an explicit list
of what it does and does not account for.

## What it does

- **Reads a predictive race-control feed.** Fast Flag's Safety Car and VSC calls arrive
  before race control's, with a measured median lead of 32.6 s. Each call is carried as a
  probability (Fast Flag's own out-of-sample precision), never as a deployed Safety Car.
- **Replays a race without leaking the future.** The replay driver serves the race as of a
  session time: laps completed, calls issued, neutralisations deployed, and nothing after
  it. A Safety Car still running does not reveal when it ends. This is tested directly:
  mutating every future row leaves the state byte-identical.
- **Predicts lap times with calibrated uncertainty.** A LightGBM median plus a quantile
  network trained on out-of-race residuals, anchored on the live race and horizon
  dependent. On held-out training folds, 10.7% of laps fall below p10 and 10.2% above p90
  against a 10 / 10 target. Scored once on the holdout itself, that becomes 18.4% below p10
  and 10.7% above: a location error, not an interval error. See the limitations.
- **Measures pit loss per race** from pit lane transit, keyed by season and event, with the
  Safety Car discount as a stated, swept parameter rather than a fitted number that the
  data cannot support.
- **Decides under uncertainty.** Pit-now and stay-out plans are scored to the flag on
  free-air time, with per-lap quantiles carried through, pit loss swept, and the call's
  probability applied. The output is P(pit better) across three assumptions about how the
  two plans' errors correlate, plus the gain distribution. When the distributions overlap,
  the system says they overlap instead of picking a side.
- **Backtests on nine holdout races** that touched no fit, and reports the cost of acting
  on a false call alongside the benefit of acting on a true one.

## The scope of the evidence

The nine holdout races contain six real neutralisations, but only four of those races have a
Fast Flag timeline, and among those four there is exactly **one real neutralisation** (2025
United States, VSC on lap 7). So the decision path has n = 1.

That number is stated up front because it is the binding constraint on every claim here: it
is why this is a working live decision system demonstrated end to end, with honest
uncertainty, and not a validated strategy system. One real neutralisation cannot validate
anything. The analysis is built to be informative anyway: the break-even work below is
driven off 1616 hypothetical calls rather than the single observed one, precisely so the
conclusions do not rest on n = 1.

### That ceiling is not mine to raise

The obvious fix is more races, and it does not work. This project consumes Fast Flag as an
upstream input, and a Fast Flag recommendation for a race exists only if Fast Flag has built
a timeline for it. It has built five, four of which are already these holdout races. Adding
races to my own pipeline, which I did in Session 8 and which took fifteen minutes, moves the
count not at all.

So the validation ceiling here is set by a dependency I do not control. Raising it means
running someone else's pipeline over new races in their repository, which is a different
project with its own owner and its own compute. That is worth naming plainly, because it is
a structural property of building on top of an upstream system rather than an oversight in
this one: **you inherit your dependency's coverage as your own evidence ceiling, and no
amount of work on your side of the boundary moves it.** I found this by checking what the
upstream actually had on disk rather than assuming my own pipeline was the constraint, and
the assumption had already survived three sessions unexamined.

Expanding the race set was still worth doing, just not for that reason, and it was done: 27
races became 38, and the holdout went from four races to nine. That scaled the
hypothetical-call sample from 724 to 1616, let the tyre model be scored on a real holdout for
the first time, and added five more neutralisations to the holdout. It moved n = 1 not at all,
because none of the five new holdout races has a Fast Flag timeline.

## The finding: the system is blind exactly when an early call arrives

On the one real event, Fast Flag called the incident at the end of lap 5, **61 s before race
control deployed the VSC**. The tyre model could not use that lead. It anchors on at least
two clean laps from lap 5 onward (laps 2 to 4 are skipped because of a measured race-start
effect on lap times), so it cannot predict anything before about lap 6 or 7. The call
arrived inside that blind window and bought nothing.

This generalises. Across the 27 train-race neutralisations, 8 deployed before lap 6 and 3
more on lap 7, so 11 of 27 land in or next to the blind window. Early Safety Cars are common, and in F1 they are often the strategically important
ones, because the field is bunched and nobody has stopped yet (a track-position effect this
model does not price). A system that cannot act before lap 6 is blind in that window.

Future work, named: a shorter anchor, or a prior-race / qualifying pace fallback for the
opening laps, validated the way the tyre model was (held-out races, calibration of p10/p90).

Two more results from the backtest ([docs/session5_results.md](docs/session5_results.md)):

- **Early calls mostly confirm, they rarely change.** 32.6 s is about a third of a lap at
  the holdout tracks. On the real VSC, 17 of 20 cars got an extra pit window from the call,
  but for none of them did the later no-call window fall outside the VSC (it ran 213 s). A
  Safety Car (median 3 laps) outlasts a third of a lap easily; a short VSC is the only case
  where the lead can change a car's options.
- **Break-even precision: the call rarely decides the stop.** Over 1616 hypothetical calls
  (every 10th lap, every car, all nine holdout races), pitting now wins in 33% of situations
  even if the call is false (the car is in its window anyway), and loses in 51% (VSC) to 61%
  (SC) even if the call is real. Only in between does precision matter. At phi 0.08, acting
  on Fast Flag's measured precision beats ignoring it in 38% of situations for Safety Car
  calls and 44% for VSC calls, but only **5 points (SC) and 11 points (VSC)** of that are
  cases where the call changes the decision. Every break-even here is an upper bound,
  because track position is not priced.
  These numbers held up when the sample grew: on four holdout races and 724 calls the same
  figures were 38%, 44 to 58%, 44 / 49% and 6 / 11 points. The conclusion that the call
  decides the stop in only about 5 to 11 points of cases was not an artefact of four races.
- **False calls.** Fast Flag made one false call across the holdout (Singapore, SC, late in
  the race). 14 of 20 cars would have pitted on it, at a median gain of 1.3 s in free-air
  time, because they were in their window anyway. Japan had no calls; Abu Dhabi only
  yellows.

## How two analysis bugs surfaced

Both decision-analysis bugs in this project were caught by knowing roughly what the answer
should look like, not by a test:

1. **A break-even table too clean to believe.** It reported that pitting on a call never
   beats staying out, in 99 to 100% of situations. A real strategy question is not that
   one-sided. The cause: pit-now plans were scored on the old tyres to the flag while still
   paying for the stop, because the stop fell on the lap before scoring began.
2. **An internal inconsistency.** "Pitting wins even if the call is false" came out at 7%
   for Safety Car calls and 38% for VSC calls. If the call is false, nothing is
   neutralised, so the number cannot depend on the call type. The cause: the false-call
   branch was still excluding the laps that would only have been neutralised had the call
   been real.

Both invalidated every decision number computed before them, and both were fixed and the
analysis re-run (commits `035cd85`, `d28803b`). The tests in this repo check mechanisms
(no leakage, correct plan construction, quantile loss); they would not have caught either
of these. Sanity-checking the shape of a result against what the domain says it should look
like is a separate and necessary layer, and it is the layer that caught both.

## Fast Flag, and who built what

Fast Flag is a **separate two-person project** (team We Are So Back, FormulaTech Hacks
2026), built mainly by **Naman**, with Ishaan on the bridge and firmware. This project
**consumes it as an input**: it reads Fast Flag's precomputed timeline for each holdout race
and never retrains or edits Fast Flag. The strategy work in this repository is solo.

The call precision used here, **0.44 for SC calls (10 of 23) and 0.57 for VSC calls (12 of
21)**, is Fast Flag's own measured out-of-sample number from its check-race scorecard, not
an assumption made here. How this project consumes Fast Flag:
[docs/fast_flag_recon.md](docs/fast_flag_recon.md).

## What every decision accounts for, and what it does not

**Accounts for:** tyre model p10 / p50 / p90 per lap (anchored on laps up to the call); pit
loss by condition with phi swept 0.05 to 0.12; the call as a probability; the compound rule
and stint caps.

**Does not account for:** traffic, track position and overtaking; queue position under the
Safety Car and rivals' reactions; tyre set availability; the correlation between the two
plans' errors (P(pit better) is reported at three values instead); tyre intervals beyond 30
laps ahead, which are a floor. On the real VSC, every car's answer was "overlapping: the
model cannot separate the two plans", and the system reports that rather than picking a side.

## The headline: teams were within about 2.5 seconds of optimal

The DP optimizer plans the rest of a race from lap 10 and compares its choice with what the
team actually did, over 458 car-races on 28 training races. The model's verdict is that the
teams left a **median of 2.5 seconds** on the table across roughly fifty remaining laps (p90
12.9 s), and it agreed with their number of pit stops 69% of the time.

That small number is the believable one, and the reason is worth stating plainly: **an
optimiser that found 30 seconds lying around would have been evidence against itself.** Real
F1 strategists are extremely good at the part of this problem that can be written down. A
model claiming otherwise would be describing its own errors, not their mistakes. What the
result actually says is that the free-air part of strategy is close to solved, so whatever
separates a good call from a bad one lives in the part this optimizer cannot see.

### The 2.5 s is an upper bound on the free-air component only

This is not a caveat sitting beside the known biases. It inverts them.

Every earlier stage of this project worried that the optimizer would favour strategies needing
overtaking, because the pass model predicts about 21% too many passes overall and about 74%
too many at tracks it has not seen. The optimizer prices traffic, overtaking and track
position at **zero**, and pricing passing as free *is that same bias at infinite strength*.
So the worry has not been managed, it has been maximised.

Two consequences follow, and both bound the headline:

- The 2.5 s is an upper bound on the **free-air** gap, not on the real one. A plan that gains
  time by stopping into traffic has that cost counted as nothing.
- It says nothing at all about whether those plans are **executable**. The optimizer cannot
  tell a plan that works from one that needs three passes it would never complete.

It is also why the bias sweep the design called for is incomplete *by construction* rather
than by omission: an objective with no traffic term gives the overtaking-bias knob nothing to
act on, while still emitting overtake-dependent plans.

### A recommendation made, then withdrawn by its own sensitivity

The optimizer's **stop count** is stable: neutralising the tyre model's known p50 bias moves
it from 1.36 to 1.33 stops per car and leaves stop-count agreement identical at 69.1%.

Its **compound choice is not**, and the sensitivity run is the only reason anyone knows:

| | HARD | SOFT | MEDIUM |
|---|---|---|---|
| optimiser, measured knobs | 295 (47%) | 224 | 102 |
| optimiser, bias neutralised | **530 (87%)** | 55 | 26 |
| what teams actually ran | 404 (56%) | 122 | 199 |

63% of plans change, and the HARD share nearly doubles. The cause is mechanical: the tyre
bias is horizon-shaped and spares HARD, so penalising MEDIUM and SOFT beyond fifteen laps
ahead penalises most of a plan on two of the three compounds. **The two runs bracket; neither
is the answer.** The compound recommendation was produced, tested against its own stated
uncertainty, and withdrawn on that evidence. Without the sensitivity run it would have shipped
as a result, and it would have looked like one.

### A premise stated in advance, and refuted

The design said to run the comparison twice, with the bias knobs at measured values and
neutralised, on the reasoning that *the part of the gap that closes under neutralisation is
attributable to our bias*. Nothing closed. The gap widened slightly, 2.5 s to 2.7 s, and
stop-count agreement was identical to the tenth of a percent. The tyre bias is not what
separates this optimizer from the teams.

That is one of **five** predictions this project wrote down in advance and then refuted with
its own data. The others: that routing the pass model's coverage flag causally would reduce
the engine's pass count (it raised it, and only a control run on identical data showed the
fall had come from somewhere else); and three of five expectations about the tyre model's
holdout score, including that the degradation would be symmetric when it was one-sided. All
are kept in the record, because a prediction that fails tells you more than one that lands.

## Honest limitations

- **The 20-car race engine failed validation**, three times. It does not reproduce how cars
  follow each other: the first version over-penalised following, the second under-penalised
  it and tripled the passes, and a third reworked it as a pace constraint rather than a
  position rule, which fixed the direction of the error and still over-predicts passes by
  1.78x. So nothing here makes a traffic or overtaking claim. The strategy optimizer exists
  and is the headline above, but it prices traffic at zero rather than using this engine, and
  there is still no Monte Carlo ranking.
- **Pit loss under a Safety Car is a stated assumption** (phi = 0.08, swept 0.05 to 0.12).
  It is not identifiable from lap times with this data: on laps where a Safety Car starts or
  ends, stay-out lap times spread 22 to 31 s with running position.
- **Tyre model:** 10.7% below p10 and 10.2% above p90 on held-out *training* folds. Scored
  once on the real holdout it gives 18.4% below p10, 10.7% above, and MAE 0.665 s against
  0.506. The upper tail stays calibrated and only the lower tail fails, with a median
  residual of -0.099 s, so this is a **location** error and not an interval error; a pooled
  miss rate cannot tell those apart. Monaco is much the worst race (MAE 1.374 s against 0.542
  elsewhere, both tails blown), which is the model failing at the one circuit where lap time
  is dominated by track position rather than tyre state: the missing physics surfacing where
  it cannot be ignored. Intervals beyond 30 laps ahead remain a floor.
- **Pass model** over-predicts on tracks it has not seen (13.4% vs 7.7%), and 21% overall.
- **One real neutralisation** on the decision path. See the scope section above.
- **Arduino "BOX" pit board:** planned, not built. The serial path would be a one-line
  message to an Uno, separate from Fast Flag's own protocol.

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

1. **2025 United States**, press *jump to the call*. Fast Flag calls a Safety Car at the end
   of lap 5, 61 s before race control deploys a VSC (right event, wrong kind). Every one of the 20 cars reads **"no decision: model has no
   anchor yet (laps 2-4 skipped)"**. That is the blind window above, on screen.
2. Step one lap on. Race control deploys the VSC, the system can now decide, and every car
   comes back **"overlapping: the model cannot separate the two plans"**, with P(pit better)
   shown as a range across the three correlation assumptions rather than a single number.
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
