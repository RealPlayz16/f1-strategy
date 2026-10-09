"""DP strategy optimizer, and the bias comparison against what teams actually ran.

Specified in the Session 4 brief, never run until Session 9. This is the first thing in the
project that produces an actual strategy recommendation, and the limits below are not caveats
bolted on afterwards: they decide what the output can mean.

WHAT IT OPTIMISES. Free-air race time from a decision lap to the flag: tyre model p50 per lap
(src/tyre.py, anchored at the decision lap), pit loss by condition (src/pitloss.py via
src/rules.py) and the compound and stint-cap rules (src/rules.py). Exact DP, no search
heuristics, over states (lap, compound, tyre age, stops made, two-compound rule satisfied).

WHAT IT PRICES AT ZERO, AND WHY THAT IS NOT A BIAS KNOB.
- TRAFFIC AND OVERTAKING. The engine cannot sit in a DP inner loop: a 20-car replay is about
  3 s and this searches tens of thousands of states per car. The engine is also the component
  that failed validation three times (see "## ENGINE LIMITATION" in HANDOFF.md). So passing is
  FREE in this objective, and the optimizer will systematically prefer plans that need
  overtaking. That is absent physics, not a parameter: of the three Session 4 bias knobs, only
  the tyre p50 one can be swept here, because the objective has no traffic term for the other
  two to act on. An optimizer that cannot be bias-swept on overtaking but still emits
  overtake-dependent plans is the Session 4 governing risk in its sharpest form.
- TRACK POSITION, rivals' reactions, tyre-set availability, and the undercut or overcut
  itself, all of which are traffic.
- SAFETY CARS. The headline run plans with GREEN pit loss everywhere, which is what a car
  knows at the decision lap. A team that pitted under a real neutralisation will therefore
  look wrong to this optimizer, and that is a confound in the comparison rather than a finding
  about the team, so a neutralisation-free subset is reported beside the full set. The
  principled treatment is a Monte Carlo over SC timing (Session 4 brief); it is not run here.

NO PRE-RACE PLAN EXISTS. predict_laptime needs an anchor from compute_anchors, which needs at
least MIN_DRIVER_LAPS clean laps in the window from lap WINDOW_START, laps 2 to 4 being
skipped for the measured early-stint effect. So the optimizer cannot plan from the grid. It
plans from DECISION_LAP with the car's real state there, and the comparison is therefore
"best remaining plan given the real opening laps", never "best race strategy". This is the
same blind window Session 5 found on the live path.

Read every number here as what the model recommends under these assumptions. It is not what a
team should have done, and the gap between the two is mostly the traffic term that is missing.

Usage:
    python -m src.optimizer                 (measured knobs and neutralised, train races)
    python -m src.optimizer --decision-lap 15
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.live import P50_BIAS_LONG_H_S, P50_BIAS_MIN_H
from src.rules import DRY_COMPOUNDS, MAX_STINT_LAPS
from src.splits import add_split
from src.tyre import compute_anchors, predict_laptime

LAPS_PATH = Path("data/processed/laps_fuel_corrected.parquet")
PITLOSS_PATH = Path("data/processed/pitloss_by_track.parquet")
SC_PATH = Path("data/processed/sc_events.parquet")
PLANS_PATH = Path("data/processed/optimizer_plans.parquet")
REPORT_PATH = Path("data/processed/optimizer_report.json")

DECISION_LAP = 10
MAX_STOPS = 3
MAX_AGE = 90
BIG = 1e9
COMPOUNDS = list(DRY_COMPOUNDS)
C_IX = {c: i for i, c in enumerate(COMPOUNDS)}


# ---------- lap time table ----------

def lap_time_grid(base: dict, decision_lap: int, race_laps: int, age_now: float,
                  p50_bias_s: float = 0.0) -> np.ndarray:
    """lt[h, compound, age, stops]: modelled p50 lap time for lap decision_lap + h run on
    `compound` at `age`, having made `stops` stops since the decision lap. BIG where the
    combination is unreachable or breaks a stint cap.

    stops drives fresh_tyre_f and stints_ahead, exactly as live.plan_rows builds them, so the
    tyre model sees the same features it was fitted on.
    """
    n_h = race_laps - decision_lap
    lt = np.full((n_h + 1, len(COMPOUNDS), MAX_AGE + 1, MAX_STOPS + 1), BIG)
    rows, index = [], []
    for h in range(1, n_h + 1):
        lap = decision_lap + h
        for ci, comp in enumerate(COMPOUNDS):
            cap = MAX_STINT_LAPS[comp]
            for stops in range(MAX_STOPS + 1):
                if stops == 0:
                    ages = [age_now + h]            # still on the tyre it had at the decision
                else:
                    ages = range(1, min(cap, h) + 1)  # fitted during the horizon
                for age in ages:
                    if age > cap or age > MAX_AGE:
                        continue
                    index.append((h, ci, int(age), stops))
                    rows.append({**base, "h": h, "tyre_life_f": float(age),
                                 "compound_f": comp, "fresh_tyre_f": float(stops > 0),
                                 "stints_ahead": stops,
                                 "laps_remaining": race_laps - lap})
    if not rows:
        return lt
    f = pd.DataFrame(rows)
    _, p50, _ = predict_laptime(f)
    if p50_bias_s:
        # Horizon shaped, not compound shaped. See live.P50_BIAS_LONG_H_S.
        bump = (f["compound_f"].isin(("MEDIUM", "SOFT")).to_numpy()
                & (f["h"].to_numpy() > P50_BIAS_MIN_H)) * p50_bias_s
        p50 = p50 + bump
    for (h, ci, age, stops), v in zip(index, p50, strict=True):
        lt[h, ci, age, stops] = v
    return lt


# ---------- DP ----------

def solve(lt: np.ndarray, decision_lap: int, race_laps: int, compound_now: str,
          age_now: float, stops_now: int, used: set[str],
          pit_loss_by_lap: np.ndarray) -> tuple[float, dict[int, str]]:
    """Exact backward induction. Returns (modelled time for the remaining laps, stops made).

    State is (compound, tyre age, stops since the decision lap, two-compound rule satisfied),
    where the age is the one the tyre carries at the END of a lap. V[h] is the remaining time
    for laps h..n_h as a function of the state at the end of lap h-1, so the lap about to be
    run ages the tyre by one before it is timed. `used` is the dry compounds the car has
    already run, so the rule can be satisfied only by one it has not.
    """
    n_h = race_laps - decision_lap
    n_c, n_s, n_a = len(COMPOUNDS), MAX_STOPS + 1, MAX_AGE + 1
    age_next = np.minimum(np.arange(n_a) + 1, MAX_AGE)
    caps = np.array([MAX_STINT_LAPS[c] for c in COMPOUNDS])
    over_cap = (np.arange(n_a)[None, :] + 1) > caps[:, None]          # (compound, age)

    nxt = np.full((n_c, n_a, n_s, 2), BIG)
    nxt[:, :, :, 1] = 0.0                       # at the flag, legal only once the rule is met
    choice = np.zeros((n_h + 1, n_c, n_a, n_s, 2), dtype=np.int8)

    for h in range(n_h, 0, -1):
        cur = np.full((n_c, n_a, n_s, 2), BIG)
        best = np.zeros((n_c, n_a, n_s, 2), dtype=np.int8)
        for ci, comp in enumerate(COMPOUNDS):
            run = np.where(over_cap[ci][:, None], BIG, lt[h, ci, age_next, :])   # (age, stops)
            # run on to the end of the race on this tyre
            cur[ci] = run[:, :, None] + nxt[ci, age_next, :, :]
            if h == n_h:
                continue                                        # never pit on the final lap
            for xi, x in enumerate(COMPOUNDS):
                fresh_rule = x not in (used | {comp})            # does this stop satisfy it
                for stops in range(n_s - 1):
                    after = nxt[xi, 0, stops + 1, :]             # value by sat after the stop
                    sat_after = np.array([1 if fresh_rule else 0, 1])
                    cost = run[:, stops] + pit_loss_by_lap[h]    # in-lap, then the stop
                    cand = cost[:, None] + after[sat_after][None, :]
                    better = cand < cur[ci, :, stops, :]
                    cur[ci, :, stops, :] = np.where(better, cand, cur[ci, :, stops, :])
                    best[ci, :, stops, :] = np.where(better, xi + 1, best[ci, :, stops, :])
        nxt, choice[h] = cur, best

    ci0 = C_IX[compound_now]
    a0 = min(int(round(age_now)), MAX_AGE)
    s0 = min(stops_now, MAX_STOPS)
    sat0 = 1 if len(used | {compound_now}) >= 2 else 0
    total = float(nxt[ci0, a0, s0, sat0])
    if not np.isfinite(total) or total >= BIG:
        return float("nan"), {}

    stops: dict[int, str] = {}
    ci, age, st, sat = ci0, a0, s0, sat0
    seen = set(used) | {compound_now}
    for h in range(1, n_h + 1):
        k = int(choice[h, ci, age, st, sat])
        age = min(age + 1, MAX_AGE)
        if k:
            x = COMPOUNDS[k - 1]
            stops[decision_lap + h] = x
            if x not in seen:
                sat = 1
            seen.add(x)
            ci, age, st = C_IX[x], 0, min(st + 1, MAX_STOPS)
    return total, stops


def plan_time(lt: np.ndarray, decision_lap: int, race_laps: int, compound_now: str,
              age_now: float, stops: dict[int, str],
              pit_loss_by_lap: np.ndarray) -> float:
    """Modelled time of a given plan under the same lap-time table, for the comparison."""
    ci, age, st, total = C_IX[compound_now], age_now, 0, 0.0
    for h in range(1, race_laps - decision_lap + 1):
        lap, age = decision_lap + h, age + 1
        t = lt[h, ci, min(int(round(age)), MAX_AGE), min(st, MAX_STOPS)]
        if t >= BIG:
            return float("nan")
        total += t
        if lap in stops:
            total += pit_loss_by_lap[h]
            x = stops[lap]
            if x not in C_IX:
                return float("nan")
            ci, age, st = C_IX[x], 0.0, st + 1
    return float(total)


# ---------- per car setup from the data ----------

def car_state(mine: pd.DataFrame, decision_lap: int) -> dict | None:
    """Compound, tyre age and compounds already used at the end of the decision lap, plus the
    stops the team actually made after it. None if the car is not running by then."""
    at = mine[mine["lap"] == decision_lap]
    if at.empty or pd.isna(at["compound"].iloc[0]) or pd.isna(at["tyre_life"].iloc[0]):
        return None
    after = mine[mine["lap"] > decision_lap]
    actual: dict[int, str] = {}
    by_lap = mine.set_index("lap")
    for lap in after.loc[after["is_pit_in"], "lap"]:
        if lap + 1 in by_lap.index and not pd.isna(by_lap.loc[lap + 1, "compound"]):
            actual[int(lap)] = str(by_lap.loc[lap + 1, "compound"])
    return {
        "compound_now": str(at["compound"].iloc[0]),
        "age_now": float(at["tyre_life"].iloc[0]),
        "used": {str(c) for c in mine.loc[mine["lap"] <= decision_lap, "compound"].dropna()},
        "actual_stops": actual,
        "last_lap": int(mine["lap"].max()),
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    decision_lap = DECISION_LAP
    if "--decision-lap" in argv:
        decision_lap = int(argv[argv.index("--decision-lap") + 1])

    laps = add_split(pd.read_parquet(LAPS_PATH).drop(columns=["split"], errors="ignore"))
    laps = laps[laps["split"] == "train"].copy()
    pitloss = pd.read_parquet(PITLOSS_PATH).set_index(["season", "event"])
    sc = pd.read_parquet(SC_PATH)
    neutralised = {(int(s), int(r)) for s, r, k in
                   zip(sc["season"], sc["round"], sc["kind"], strict=True)
                   if k in ("SC", "VSC")}

    anchors = compute_anchors(laps[laps["is_clean"]])
    anchors = anchors[anchors["lap"] == decision_lap]
    print(f"decision lap {decision_lap}: {len(anchors)} car-races with an anchor", flush=True)

    rows = []
    for (season, rnd), race_anchors in anchors.groupby(["season", "round"], sort=True):
        race = laps[(laps["season"] == season) & (laps["round"] == rnd)]
        event = str(race["event"].iloc[0])
        race_laps = int(race["race_laps"].iloc[0])
        if (season, event) not in pitloss.index:
            print(f"  {season} {event}: no pit loss row, skipped", flush=True)
            continue
        green = float(pitloss.loc[(season, event), "green_s"])
        # GREEN everywhere: what a car knows at the decision lap. See the module docstring.
        pit_loss_by_lap = np.full(race_laps - decision_lap + 1, green)
        for _, a in race_anchors.iterrows():
            drv = str(a["driver"])
            mine = race[race["driver"] == drv].sort_values("lap")
            st = car_state(mine, decision_lap)
            if st is None:
                continue
            base = {"event": event, "driver": drv, "team": str(mine["team"].iloc[0]),
                    "compound_t": str(a["compound_t"]), "tyre_life_t": float(a["tyre_life_t"]),
                    "track_temp": float(a["track_temp"]), "air_temp": float(a["air_temp"]),
                    "anchor_s": float(a["anchor_s"]), "race_laps": race_laps}
            row = {"season": season, "round": rnd, "event": event, "driver": drv,
                   "race_laps": race_laps, "decision_lap": decision_lap,
                   "compound_now": st["compound_now"], "age_now": st["age_now"],
                   "finished": st["last_lap"] >= race_laps,
                   "neutralised_race": (int(season), int(rnd)) in neutralised,
                   "actual_stops_n": len(st["actual_stops"]),
                   "actual_plan": json.dumps(st["actual_stops"])}
            for label, bias in (("measured", 0.0), ("neutralised", P50_BIAS_LONG_H_S)):
                lt = lap_time_grid(base, decision_lap, race_laps, st["age_now"], bias)
                best, stops = solve(lt, decision_lap, race_laps, st["compound_now"],
                                    st["age_now"], 0, st["used"], pit_loss_by_lap)
                act = plan_time(lt, decision_lap, race_laps, st["compound_now"],
                                st["age_now"], st["actual_stops"], pit_loss_by_lap)
                row[f"{label}_opt_s"] = best
                row[f"{label}_actual_s"] = act
                row[f"{label}_gain_s"] = act - best if np.isfinite(act) else float("nan")
                row[f"{label}_stops_n"] = len(stops)
                row[f"{label}_plan"] = json.dumps(stops)
            rows.append(row)
        print(f"  {season} {event}: {len(race_anchors)} cars", flush=True)

    d = pd.DataFrame(rows)
    d.to_parquet(PLANS_PATH, index=False)
    report = summarise(d)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {PLANS_PATH} and {REPORT_PATH}")
    return 0


def summarise(d: pd.DataFrame) -> dict:
    pd.set_option("display.width", 200)
    out: dict = {"n_car_races": int(len(d)), "n_races": int(d.groupby(["season", "round"]).ngroups)}
    print(f"\noptimised {len(d)} car-races over {out['n_races']} train races")
    print("\nEVERY NUMBER BELOW PRICES TRAFFIC AT ZERO. The optimizer has no overtaking term,")
    print("so it prefers plans that need passing and the gains are upper bounds. It is what")
    print("the model recommends under stated assumptions, not what a team should have done.")
    for label in ("measured", "neutralised"):
        ok = d[np.isfinite(d[f"{label}_gain_s"])]
        agree = (ok[f"{label}_stops_n"] == ok["actual_stops_n"]).mean() if len(ok) else np.nan
        out[label] = {
            "n_comparable": int(len(ok)),
            "stop_count_agreement": float(agree),
            "gain_median_s": float(ok[f"{label}_gain_s"].median()),
            "gain_p10_s": float(ok[f"{label}_gain_s"].quantile(0.1)),
            "gain_p90_s": float(ok[f"{label}_gain_s"].quantile(0.9)),
            "opt_stops_mean": float(ok[f"{label}_stops_n"].mean()),
            "actual_stops_mean": float(ok["actual_stops_n"].mean()),
        }
        print(f"\n--- knobs {label}  (n={len(ok)})")
        print(f"  stop-count agreement with the team   {agree:.1%}")
        print(f"  optimiser stops / actual stops       {ok[f'{label}_stops_n'].mean():.2f}"
              f" / {ok['actual_stops_n'].mean():.2f}")
        g = ok[f"{label}_gain_s"]
        print(f"  modelled gain over the actual plan   median {g.median():.1f} s"
              f"  (p10 {g.quantile(0.1):.1f}, p90 {g.quantile(0.9):.1f})")
        print(f"  optimiser stop count distribution    "
              f"{ok[f'{label}_stops_n'].value_counts().sort_index().to_dict()}")
        print(f"  actual stop count distribution       "
              f"{ok['actual_stops_n'].value_counts().sort_index().to_dict()}")
    # The SC confound: a team that pitted under a real neutralisation is being compared with
    # an optimizer that planned on green pit loss. The clean subset has no neutralisation.
    clean = d[~d["neutralised_race"] & np.isfinite(d["measured_gain_s"])]
    out["no_neutralisation_subset"] = {
        "n_car_races": int(len(clean)),
        "n_races": int(clean.groupby(["season", "round"]).ngroups) if len(clean) else 0,
        "stop_count_agreement": float((clean["measured_stops_n"]
                                       == clean["actual_stops_n"]).mean()) if len(clean) else None,
        "gain_median_s": float(clean["measured_gain_s"].median()) if len(clean) else None,
    }
    if len(clean):
        print(f"\n--- no-neutralisation subset, measured knobs (n={len(clean)} car-races, "
              f"{clean.groupby(['season', 'round']).ngroups} races)")
        print("  removes the confound that the optimizer planned on green pit loss while the")
        print("  team could pit under a real SC or VSC.")
        print(f"  stop-count agreement                 "
              f"{(clean['measured_stops_n'] == clean['actual_stops_n']).mean():.1%}")
        print(f"  modelled gain median                 {clean['measured_gain_s'].median():.1f} s")
    flips = d[np.isfinite(d["measured_gain_s"]) & np.isfinite(d["neutralised_gain_s"])]
    changed = (flips["measured_plan"] != flips["neutralised_plan"]).mean() if len(flips) else np.nan
    out["plan_changed_when_neutralised"] = float(changed)
    print(f"\nplans that change when the tyre p50 bias is neutralised: {changed:.1%}")
    print("  (the pass-model knobs cannot be swept here: the objective has no traffic term)")
    return out


if __name__ == "__main__":
    sys.exit(main())
