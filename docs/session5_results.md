# Session 5 results: live Fast Flag decision system

**This is a working live decision system, demonstrated end to end, with honest uncertainty. It is not a validated strategy system. The holdout contains one real neutralisation (2025 United States, VSC lap 7), and one real neutralisation cannot validate anything.**

Every decision output carries these lists:

Accounts for:
- tyre model p10/p50/p90 per lap (anchored on laps up to the call)
- pit loss by condition, phi swept 0.05 to 0.12
- Fast Flag call as a probability (out-of-sample precision, Beta interval)
- compound rule and stint caps (src/rules.py)

Does not account for:
- traffic, track position and overtaking (engine failed validation, Session 4)
- queue position under SC and rivals' reactions
- tyre set availability (a fresh set of each compound assumed)
- correlation between the two plans' errors (reported at rho 0 / 0.5 / 0.9)
- h > 30 tyre intervals are a floor (survivorship), so P is overconfident there

Fast Flag call precision (its out-of-sample scorecard): SC 10/23, P = 0.44 (90% 0.28-0.60); VSC 12/21, P = 0.57 (90% 0.40-0.73).

Holdout races with a Fast Flag timeline: 2025_Abu_Dhabi, 2025_Japanese, 2025_Singapore, 2025_United_States.

## Break-even precision (headline)

For a hypothetical Fast Flag call at the end of every 10th lap, for every running car, p* is the call precision above which pitting on the call beats ignoring it. Free-air time only: the track-position benefit of stopping under a neutralisation is not modelled, so the gain from a real call is understated and **every p* here is an upper bound** on the precision really needed.

Hypothetical calls: 730; with a decision: 724 (the rest have no tyre anchor yet or the race is ending).

| kind | phi | n | pit anyway (p*=0) | never pit (no p*) | p* median (when 0<p*<inf) | p* 10-90% | acting beats ignoring at FF precision 0.44 | acting beats ignoring at FF precision 0.57 |
|---|---|---|---|---|---|---|---|---|
| SC | 0.05 | 362 | 38% | 58% | 0.26 | 0.07-0.68 | 41% | nan |
| SC | 0.08 | 362 | 38% | 55% | 0.11 | 0.03-0.48 | 44% | nan |
| SC | 0.12 | 362 | 38% | 45% | 0.44 | 0.02-0.82 | 47% | nan |
| VSC | 0.05 | 362 | 38% | 50% | 0.22 | 0.03-0.85 | nan | 47% |
| VSC | 0.08 | 362 | 38% | 44% | 0.36 | 0.02-0.89 | nan | 49% |
| VSC | 0.12 | 362 | 38% | 40% | 0.36 | 0.02-0.82 | nan | 54% |

Split at phi 0.08, SC calls: cars whose stay-out plan still needs a stop (n=238): p* median 0.00, never-pit 37%; cars that can run to the flag (n=124): never-pit 90%.

## False calls

| race | episode | kind called | real | actual | lead_s |
|---|---|---|---|---|---|
| 2025_Singapore | 0 | SC | False |  |  |
| 2025_United_States | 0 | SC | True | VSC | 61.190 |

- 2025_Singapore episode 0: 20 cars with a decision, 16 would have pitted (P >= 0.5 at rho 0.5); median free-air time change from pitting on the false call: +1.0 s (positive = a gain: those cars were in their pit window anyway).

## The one real neutralisation: 2025 United States, VSC lap 7

- Fast Flag called **SC** at t = 3998.5 s ("car 12 stopped for 3.0 s after an impact"); race control deployed a **VSC** at 4059.7 s. Lead 61.2 s = 0.61 laps. Right event, wrong kind.
- Clock check: Fast Flag's official VSC vs our sc_events differ by -0.28 s.
- Decisions at the call: {'no anchor (fewer than 2 clean laps in current stint)': 20}. The call came at the end of lap 5; the tyre model anchors on at least 2 clean laps from lap 5 on (laps 2-4 are skipped, early-stint effect), so **the system was blind when the call arrived**. The lead time bought nothing on the one real event.
- Decisions at the deployment (no-call path, P = 1): {'overlapping: the model cannot separate the two plans': 15, 'no anchor (fewer than 2 clean laps in current stint)': 5}.

| driver | compound_now | p_pit_better_rho0.0 | p_pit_better_rho0.5 | p_pit_better_rho0.9 | gain_if_real_p50_rho0.5 | share_laps_h_gt_30 | pit_now_plan | stay_out_plan |
|---|---|---|---|---|---|---|---|---|
| ALB | HARD | 0.376 | 0.327 | 0.163 | -16.633 | 0.408 | {7: 'SOFT', 29: 'SOFT'} | {33: 'SOFT'} |
| ALO | MEDIUM | 0.446 | 0.440 | 0.353 | -5.289 | 0.408 | {7: 'HARD'} | {33: 'SOFT'} |
| BEA | MEDIUM | 0.505 | 0.527 | 0.526 | 2.576 | 0.408 | {7: 'HARD'} | {35: 'SOFT'} |
| BOR | SOFT | 0.523 | 0.555 | 0.591 | 4.660 | 0.408 | {7: 'HARD'} | {8: 'HARD'} |
| GAS | MEDIUM | 0.482 | 0.489 | 0.444 | -1.293 | 0.408 | {7: 'HARD'} | {44: 'SOFT'} |
| HAD | HARD | 0.367 | 0.319 | 0.147 | -17.142 | 0.408 | {7: 'SOFT', 31: 'SOFT'} | {31: 'SOFT'} |
| HAM | MEDIUM | 0.481 | 0.490 | 0.457 | -0.732 | 0.408 | {7: 'HARD'} | {33: 'SOFT'} |
| HUL | MEDIUM | 0.480 | 0.487 | 0.447 | -1.142 | 0.408 | {7: 'HARD'} | {29: 'HARD'} |
| LAW | MEDIUM | 0.471 | 0.477 | 0.432 | -1.888 | 0.408 | {7: 'HARD'} | {30: 'SOFT'} |
| LEC | SOFT | 0.528 | 0.568 | 0.621 | 5.395 | 0.408 | {7: 'HARD'} | {10: 'MEDIUM'} |
| NOR | MEDIUM | 0.432 | 0.425 | 0.320 | -6.949 | 0.408 | {7: 'HARD'} | {29: 'SOFT'} |
| OCO | HARD | 0.395 | 0.361 | 0.203 | -13.394 | 0.408 | {7: 'MEDIUM', 33: 'SOFT'} | {28: 'MEDIUM'} |
| RUS | MEDIUM | 0.470 | 0.475 | 0.427 | -2.127 | 0.408 | {7: 'HARD'} | {33: 'SOFT'} |
| TSU | MEDIUM | 0.465 | 0.468 | 0.415 | -2.668 | 0.408 | {7: 'HARD'} | {31: 'SOFT'} |
| VER | MEDIUM | 0.409 | 0.381 | 0.231 | -10.167 | 0.408 | {7: 'SOFT', 29: 'SOFT'} | {30: 'SOFT'} |

- Soft p50 + 0.075 s sensitivity: 1 of 15 cars change side of P = 0.5 at rho 0.5.
- Share of plan laps beyond h = 30 (tyre intervals a floor there): 41%. The probabilities above are overconfident by that much.
- Early call windows: 17 of 20 cars got an extra pit window from the call; for 0 of them the no-call window fell outside the neutralisation. **The call confirmed earlier; it changed no car's options.** Teams that pitted during the VSC: 1.
- Synthetic no-call path (**SYNTHETIC**): the deployment decisions above are what the system does when Fast Flag misses or is ignored. The holdout has no real missed neutralisation.

## Lead time in laps

Fast Flag's median lead over race control is 32.6 s (its out-of-sample scorecard). In laps of advance notice per holdout track:

| race | median lap (s) | 32.6 s in laps |
|---|---|---|
| 2025_Abu_Dhabi | 89.600 | 0.360 |
| 2025_Japanese | 93.200 | 0.350 |
| 2025_Singapore | 97.900 | 0.330 |
| 2025_United_States | 100.100 | 0.330 |

A third of a lap. An SC (median 3 laps) outlasts that easily, so an early SC call mostly confirms a decision earlier. A VSC (median 96 s, about one lap) is where a third of a lap can decide whether a car's next pit window is still inside the neutralisation; in the one real VSC it did not (the VSC ran 213 s).

