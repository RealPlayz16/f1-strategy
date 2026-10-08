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

### Two different sample sizes in this report, do not mix them

The holdout is **9 races**, and the hypothetical-call analysis below uses all of them. The **real-call** evidence does not: a Fast Flag recommendation exists only for a race Fast Flag has built a timeline for, and it has built **4**. Expanding the race set in Session 8 scaled the hypothetical sample and moved the real-call sample not at all.

**The real-call path is unchanged at n = 1**: one real neutralisation, 2025 United States, VSC lap 7. A larger break-even sample is more coverage of race *situations*, not more validation of the live decision system. The ceiling is set by an upstream dependency this project does not control, so no amount of work here raises it. See "VALIDATION CEILING" in HANDOFF.md.

## Break-even precision (headline)

For a hypothetical Fast Flag call at the end of every 10th lap, for every running car, p* is the call precision above which pitting on the call beats ignoring it. Free-air time only: the track-position benefit of stopping under a neutralisation is not modelled, so the gain from a real call is understated and **every p* here is an upper bound** on the precision really needed.

Hypothetical calls: 1616; with a decision: 1604 (the rest have no tyre anchor yet or the race is ending).

| kind | phi | n | pit anyway (p*=0) | never pit (no p*) | p* median (when 0<p*<inf) | p* 10-90% | acting beats ignoring at FF precision 0.44 | acting beats ignoring at FF precision 0.57 |
|---|---|---|---|---|---|---|---|---|
| SC | 0.05 | 802 | 33% | 63% | 0.19 | 0.05-0.50 | 36% | nan |
| SC | 0.08 | 802 | 33% | 61% | 0.10 | 0.02-0.82 | 38% | nan |
| SC | 0.12 | 802 | 33% | 53% | 0.44 | 0.02-0.89 | 40% | nan |
| VSC | 0.05 | 802 | 33% | 55% | 0.37 | 0.04-0.84 | nan | 41% |
| VSC | 0.08 | 802 | 33% | 51% | 0.35 | 0.03-0.87 | nan | 44% |
| VSC | 0.12 | 802 | 33% | 48% | 0.31 | 0.03-0.83 | nan | 47% |

Split at phi 0.08, SC calls: cars whose stay-out plan still needs a stop (n=541): p* median 0.00, never-pit 47%; cars that can run to the flag (n=261): never-pit 92%.

## False calls

| race | episode | kind called | real | actual | lead_s |
|---|---|---|---|---|---|
| 2025_Singapore | 0 | SC | False |  |  |
| 2025_United_States | 0 | SC | True | VSC | 61.190 |

- 2025_Singapore episode 0: 20 cars with a decision, 14 would have pitted (P >= 0.5 at rho 0.5); median free-air time change from pitting on the false call: +1.3 s (positive = a gain: those cars were in their pit window anyway).

## The one real neutralisation: 2025 United States, VSC lap 7

- Fast Flag called **SC** at t = 3998.5 s ("car 12 stopped for 3.0 s after an impact"); race control deployed a **VSC** at 4059.7 s. Lead 61.2 s = 0.61 laps. Right event, wrong kind.
- Clock check: Fast Flag's official VSC vs our sc_events differ by -0.28 s.
- Decisions at the call: {'no anchor (fewer than 2 clean laps in current stint)': 20}. The call came at the end of lap 5; the tyre model anchors on at least 2 clean laps from lap 5 on (laps 2-4 are skipped, early-stint effect), so **the system was blind when the call arrived**. The lead time bought nothing on the one real event.
- Decisions at the deployment (no-call path, P = 1): {'overlapping: the model cannot separate the two plans': 15, 'no anchor (fewer than 2 clean laps in current stint)': 5}.

| driver | compound_now | p_pit_better_rho0.0 | p_pit_better_rho0.5 | p_pit_better_rho0.9 | gain_if_real_p50_rho0.5 | share_laps_h_gt_30 | pit_now_plan | stay_out_plan |
|---|---|---|---|---|---|---|---|---|
| ALB | HARD | 0.409 | 0.394 | 0.263 | -11.305 | 0.408 | {7: 'HARD', 37: 'SOFT'} | {33: 'SOFT'} |
| ALO | MEDIUM | 0.420 | 0.394 | 0.250 | -10.456 | 0.408 | {7: 'HARD'} | {29: 'SOFT'} |
| BEA | MEDIUM | 0.446 | 0.426 | 0.314 | -7.434 | 0.408 | {7: 'HARD'} | {36: 'SOFT'} |
| BOR | SOFT | 0.538 | 0.591 | 0.670 | 8.523 | 0.408 | {7: 'HARD'} | {19: 'HARD'} |
| GAS | MEDIUM | 0.427 | 0.399 | 0.266 | -10.112 | 0.408 | {7: 'HARD'} | {38: 'SOFT'} |
| HAD | HARD | 0.402 | 0.381 | 0.238 | -12.492 | 0.408 | {7: 'HARD', 34: 'SOFT'} | {30: 'SOFT'} |
| HAM | MEDIUM | 0.398 | 0.356 | 0.199 | -14.272 | 0.408 | {7: 'HARD'} | {30: 'HARD'} |
| HUL | MEDIUM | 0.413 | 0.378 | 0.230 | -11.951 | 0.408 | {7: 'HARD'} | {31: 'SOFT'} |
| LAW | MEDIUM | 0.435 | 0.413 | 0.288 | -8.716 | 0.408 | {7: 'HARD'} | {29: 'SOFT'} |
| LEC | SOFT | 0.523 | 0.562 | 0.608 | 5.124 | 0.408 | {7: 'HARD'} | {24: 'HARD'} |
| NOR | MEDIUM | 0.382 | 0.334 | 0.169 | -16.212 | 0.408 | {7: 'HARD'} | {30: 'SOFT'} |
| OCO | HARD | 0.400 | 0.372 | 0.217 | -12.763 | 0.408 | {7: 'SOFT', 34: 'SOFT'} | {31: 'SOFT'} |
| RUS | MEDIUM | 0.444 | 0.422 | 0.308 | -7.692 | 0.408 | {7: 'HARD'} | {33: 'SOFT'} |
| TSU | MEDIUM | 0.429 | 0.406 | 0.274 | -9.486 | 0.408 | {7: 'HARD'} | {33: 'SOFT'} |
| VER | MEDIUM | 0.368 | 0.318 | 0.146 | -17.940 | 0.408 | {7: 'HARD'} | {31: 'SOFT'} |

- Long-horizon p50 + 0.09 s on MEDIUM and SOFT beyond h = 15 (Session 8 spec, replaces the dead soft + 0.075): 0 of 15 cars change side of P = 0.5 at rho 0.5.
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

