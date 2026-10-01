# F1 Race Strategy: live Safety Car decisions under uncertainty

A Formula 1 strategy system built on FastF1 data. It takes a live, predictive race-control
feed ([Fast Flag](https://github.com/Pseudocoder28/Fast-Flag)), and when that feed calls a
Safety Car or VSC it asks, for every car: pit now or stay out? The answer is a probability
with its spread, plus an explicit list of what the answer does and does not account for.

**What this is, and what it is not.** This is a working live decision system, demonstrated
end to end, with honest uncertainty. It is not a validated strategy system. The four holdout
races contain exactly one real neutralisation (2025 United States, VSC on lap 7), so the
decision path has n = 1, and one real neutralisation cannot validate anything.

## The finding: the system is blind exactly when an early call arrives

On the one real event, Fast Flag called the incident at the end of lap 5, **61 s before race
control deployed the VSC**. Our tyre model could not use that lead. It anchors on at least
two clean laps from lap 5 onward (laps 2 to 4 are skipped because of a measured race-start
effect on lap times), so it cannot predict anything before about lap 6 or 7. The call
arrived inside that blind window and bought nothing.

This generalises. In our 19 train-race neutralisations, 4 deployed before lap 6 and one
more on lap 7. Early Safety Cars are common, and in F1 they are often the strategically
important ones, because the field is bunched and nobody has stopped yet (a track-position
effect our model does not price). A system that cannot act before lap 6 is blind in that
window.

Future work, named: a shorter anchor, or a prior-race / qualifying pace fallback for the
opening laps, validated the same way the tyre model was (held-out races, calibration of
p10 / p90).

Two more results from the backtest ([docs/session5_results.md](docs/session5_results.md)):

- **Early calls mostly confirm, they rarely change.** Fast Flag's median lead is 32.6 s,
  about a third of a lap at the holdout tracks. On the real VSC, 17 of 20 cars got an extra
  pit window from the call, but for none of them did the later no-call window fall outside
  the VSC (it ran 213 s). An SC (median 3 laps) outlasts a third of a lap easily; a short
  VSC is the only case where the lead can change a car's options.
- **Break-even precision: the call rarely decides the stop.** Over 724 hypothetical calls
  (every 10th lap, every car, all four holdout races), pitting now wins in 38% of situations
  even if the call is false (the car is in its window anyway), and loses in 44 to 58% even
  if the call is real. Only in between does the precision matter. At phi 0.08, acting on
  Fast Flag's measured precision beats ignoring it in 44% of situations for SC calls and 49%
  for VSC calls, but only **6 points (SC) and 11 points (VSC)** of that are cases where the
  call changes the decision. Every break-even here is an upper bound, because track position
  is not priced.
- **False calls.** Fast Flag made one false call across the holdout (Singapore, SC, late in
  the race). 16 of 20 cars would have pitted on it, at a median gain of 1.0 s in free-air
  time, because they were in their window anyway. Japan had no calls; Abu Dhabi only
  yellows.

## Fast Flag, and who built what

Fast Flag is a separate two-person project (team We Are So Back, FormulaTech Hacks 2026),
built mainly by **Naman**, with Ishaan on the bridge and firmware. This project consumes it
as an input: it reads Fast Flag's precomputed timeline for each holdout race and never
retrains or edits Fast Flag. The strategy work in this repository is solo.

The call precision used here, **0.44 for SC calls (10 of 23) and 0.57 for VSC calls (12 of
21)**, is Fast Flag's own measured out-of-sample number (its check-race scorecard), not an
assumption of ours. Each call is carried as a probability with a Beta interval, never as a
deployed Safety Car. How we consume Fast Flag: [docs/fast_flag_recon.md](docs/fast_flag_recon.md).

## What every decision accounts for, and what it does not

Accounts for: tyre model p10 / p50 / p90 per lap (anchored on laps up to the call); pit
loss by condition with phi swept 0.05 to 0.12; the call as a probability; the compound rule
and stint caps.

Does not account for: traffic, track position and overtaking; queue position under the
Safety Car and rivals' reactions; tyre set availability; the correlation between the two
plans' errors (P(pit better) is reported at three values instead); tyre intervals beyond 30
laps ahead, which are a floor. On the real VSC, every car's answer was "overlapping: the
model cannot separate the two plans", and the system says so rather than picking one.

## Honest limitations

- **The 20-car race engine failed validation** (Session 4). It does not reproduce how cars
  follow each other: one version over-penalised following, the fix under-penalised it and
  tripled passes. So nothing here makes a traffic or overtaking claim, and there is no
  strategy optimizer or Monte Carlo ranking.
- **Pit loss under a Safety Car is a stated assumption** (phi = 0.08, swept 0.05 to 0.12).
  It is not identifiable from lap times with this data: on laps where an SC starts or ends,
  stay-out lap times spread 22 to 31 s with running position.
- **Tyre model:** calibrated on held-out races (11.6% below p10, 9.8% above p90), but soft
  p50 is about 0.075 s/lap too fast, and intervals beyond 30 laps ahead are a floor.
- **Pass model** over-predicts on tracks it has not seen (15.2% vs 8.9%).
- One real neutralisation on the decision path. See the top of this page.

## Data

27 dry races, 2023 to 2025 ([config/races.yaml](config/races.yaml)). 23 train, 4 holdout
(2025 Japanese, United States, Singapore, Abu Dhabi). Holdout races touch no fit. 2025
Dutch was moved out of the holdout because Fast Flag trained on it.

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
python -m src.fuel
python -m src.degradation
python -m src.tyre
python -m src.traffic
python -m src.overtake_model
python -m src.backtest
python -m src.backtest --break-even
python -m src.live_report
pytest -q
```

The backtest needs Fast Flag timelines for the holdout races, built in a Fast Flag clone at
`~/fast-flag` with `python -m src.ingest.build --case <race>` then
`python -m src.replay.timeline <race>`.

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
| `src/live.py` | Fast Flag hook, leak-tested replay driver, pit-now vs stay-out decision |
| `src/backtest.py`, `src/live_report.py` | Holdout backtest, break-even precision, results report |

Design history, every model decision and every failed check: [HANDOFF.md](HANDOFF.md).

## Credits

- [Fast Flag](https://github.com/Pseudocoder28/Fast-Flag): Naman (main builder) and Ishaan,
  team We Are So Back. Consumed as an input, see above.
- Race data from [FastF1](https://github.com/theOehrly/Fast-F1).

Everything is a replay of historical data.
