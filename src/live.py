"""Live replay: Fast Flag calls -> pit-now vs stay-out decision under uncertainty.

The race is served as of a session time t: laps completed by t, Fast Flag recs issued by t,
official neutralisations deployed by t, nothing later (tested in tests/test_live.py).

Fast Flag recs come from its precomputed timeline (~/fast-flag/data/timeline/<race>.json.gz,
built with --case for our holdout races; Fast Flag's models untouched). Their t is FastF1
session time, the clock of our laps and sc_events.parquet; check_clock() compares Fast Flag's
own official events with sc_events.

A call is a probability, not a deployment: P(a neutralisation follows | call) from Fast
Flag's out-of-sample scorecard (escalation_check_ours.csv): SC calls 10 of 23, VSC calls 12
of 21 matched an official escalation. Beta(1 + hits, 1 + misses) carries the small sample.

Decision for one car at time t:
  pit-now   stop at the car's next pit opportunity under the called condition
  stay-out  continue; stop later under green if the compound rule or stint cap needs it
Both plans are scored on free-air lap time to the flag: tyre model p10 / p50 / p90 per lap
(src/tyre.py, anchored on laps up to t) plus pit loss (phi 0.08 for SC / VSC, swept 0.05 to
0.12). Laps under the neutralisation run at the same pace on both plans and cancel.
Assumptions, stated in every output:
  - per-lap quantiles -> split normal; laps within a plan fully correlated (widest)
  - correlation between the two plans' errors is not identified: P(pit better) is
    reported at rho = 0, 0.5, 0.9
  - a fresh set of every dry compound is available (tyre allocation not modelled)
  - h > 30 tyre intervals are a floor: the share of laps beyond h = 15 / 30 is reported
NOT accounted for: traffic, track position, overtaking, queue position under SC (the engine
cannot support them; Session 4), rivals' reactions, tyre wear under the neutralisation.

Usage:
    python -m src.live --race 2025_United_States
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import norm

from src.rules import DRY_COMPOUNDS, MAX_STINT_LAPS
from src.tyre import compute_anchors, predict_laptime

FAST_FLAG_DIR = Path.home() / "fast-flag"
LAPS_PATH = Path("data/processed/laps_fuel_corrected.parquet")
SC_PATH = Path("data/processed/sc_events.parquet")
PITLOSS_PATH = Path("data/processed/pitloss_by_track.parquet")
FF_SCORECARD = FAST_FLAG_DIR / "docs/charts/escalation_check_ours.csv"

SEED = 42
PHI = 0.08
PHI_RANGE = (0.05, 0.12)
RHOS = (0.0, 0.5, 0.9)
N_DRAWS = 4000
REACTION_S = 10.0       # call -> decision -> driver told -> pit entry (stated assumption)
CLOCK_TOL_S = 2.0
Z90 = norm.ppf(0.9)
NEUTRAL_LAPS = {"SC": 3, "VSC": 1}  # train medians (sc_events): SC 3 laps, VSC about 96 s

ACCOUNTS_FOR = [
    "tyre model p10/p50/p90 per lap (anchored on laps up to the call)",
    "pit loss by condition, phi swept 0.05 to 0.12",
    "Fast Flag call as a probability (out-of-sample precision, Beta interval)",
    "compound rule and stint caps (src/rules.py)",
]
NOT_ACCOUNTED = [
    "traffic, track position and overtaking (engine failed validation, Session 4)",
    "queue position under SC and rivals' reactions",
    "tyre set availability (a fresh set of each compound assumed)",
    "correlation between the two plans' errors (reported at rho 0 / 0.5 / 0.9)",
    "h > 30 tyre intervals are a floor (survivorship), so P is overconfident there",
]


def race_id(season: int, event: str) -> str:
    return f"{season}_{event.replace('Grand Prix', '').strip().replace(' ', '_')}"


# ---------- Fast Flag ----------

def load_recs(rid: str, ff_dir: Path = FAST_FLAG_DIR) -> pd.DataFrame:
    """Track-wide recs and TRACK CLEAR from Fast Flag's precomputed timeline."""
    with gzip.open(ff_dir / "data/timeline" / f"{rid}.json.gz", "rt", encoding="utf-8") as f:
        ticks = json.load(f)["ticks"]
    recs = [e["data"] for tick in ticks for e in tick if e["kind"] == "rec"]
    cols = ["id", "t", "msector", "flag", "confidence", "reason", "message",
            "source_detections"]
    df = pd.DataFrame(recs, columns=cols) if recs else pd.DataFrame(columns=cols)
    keep = df["flag"].isin(["SC", "VSC", "RED"]) | (df["message"] == "TRACK CLEAR")
    return df[keep].sort_values("t").reset_index(drop=True)


def check_clock(rid: str, sc_events: pd.DataFrame, season: int, event: str,
                ff_dir: Path = FAST_FLAG_DIR) -> pd.DataFrame:
    """Fast Flag's official SC / VSC deployments vs our sc_events, same session clock."""
    official = json.loads((ff_dir / "data/case_studies" / f"{rid}_official.json").read_text(
        encoding="utf-8"))
    ff = pd.DataFrame([o for o in official if o["flag"] in ("SC", "VSC")
                       and "DEPLOYED" in o["message"]])
    ours = sc_events[(sc_events["season"] == season) & (sc_events["event"] == event)
                     & sc_events["kind"].isin(["SC", "VSC"])]
    rows = []
    for _, e in ours.iterrows():
        cand = ff[ff["flag"] == e["kind"]]
        if cand.empty:
            rows.append({"kind": e["kind"], "ours_t": e["t_deploy_s"], "ff_t": np.nan,
                         "diff_s": np.nan})
            continue
        j = (cand["t"] - e["t_deploy_s"]).abs().idxmin()
        rows.append({"kind": e["kind"], "ours_t": e["t_deploy_s"], "ff_t": cand.loc[j, "t"],
                     "diff_s": cand.loc[j, "t"] - e["t_deploy_s"]})
    return pd.DataFrame(rows)


def call_probability(kind: str, scorecard: Path = FF_SCORECARD) -> dict:
    """P(neutralisation follows | Fast Flag call of this kind), Beta(1 + hit, 1 + miss)."""
    o = pd.read_csv(scorecard)
    o = o[o["flag"] == kind]
    hit = int((o["category"] == "matched").sum())
    miss = len(o) - hit
    a, b = 1 + hit, 1 + miss
    return {"p": a / (a + b), "q05": float(beta_dist.ppf(0.05, a, b)),
            "q95": float(beta_dist.ppf(0.95, a, b)), "hits": hit, "n": len(o)}


# ---------- replay driver ----------

@dataclass
class RaceState:
    t: float
    laps: pd.DataFrame       # laps completed by t
    recs: pd.DataFrame       # Fast Flag recs issued by t
    neutral: pd.DataFrame    # official neutralisations deployed by t


class ReplayDriver:
    """Serves one race as of a session time. Nothing after t is ever returned."""

    def __init__(self, laps: pd.DataFrame, recs: pd.DataFrame, sc_events: pd.DataFrame):
        self._laps = laps.sort_values(["session_time_s"]).reset_index(drop=True)
        self._recs = recs.sort_values("t").reset_index(drop=True)
        self._neutral = sc_events.sort_values("t_deploy_s").reset_index(drop=True)

    def lap_end_time(self, lap: int) -> float:
        """Leader's session time at the end of a lap."""
        return float(self._laps.loc[self._laps["lap"] == lap, "session_time_s"].min())

    def at(self, t: float) -> RaceState:
        laps = self._laps[self._laps["session_time_s"] <= t].copy()
        recs = self._recs[self._recs["t"] <= t].copy()
        neutral = self._neutral[self._neutral["t_deploy_s"] <= t].copy()
        # an event still running at t must not reveal how it ends
        running = neutral["t_end_s"] > t
        neutral.loc[running, ["t_end_s", "lap_end", "duration_s", "duration_laps",
                              "t_ending_s", "ended_by"]] = np.nan
        return RaceState(t, laps, recs, neutral)

    def at_lap(self, lap: int) -> RaceState:
        return self.at(self.lap_end_time(lap))


# ---------- decision ----------

def pit_loss_estimate(pitloss: pd.DataFrame, season: int, event: str, phi: float) -> dict:
    """Holdout races have no row (train only): the same event's latest earlier train season,
    else the train median (flagged)."""
    rows = pitloss[(pitloss["event"] == event) & (pitloss["season"] < season)]
    if len(rows):
        row, source = rows.sort_values("season").iloc[-1], f"{event} {int(rows['season'].max())}"
    else:
        row, source = pitloss.median(numeric_only=True), "train median (no train race here)"
    out = {"green": float(row["green_s"]), "source": source}
    for c in ("sc", "vsc"):
        out[f"delta_lap_{c}"] = float(row[f"lap_{c}_s"] - row["lap_green_s"])
        out[c] = out["green"] - phi * out[f"delta_lap_{c}"]
    return out


def quantile_sum(q: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Plan total at quantile levels u, laps fully correlated: sum of per-lap split-normal
    quantiles. q: (n_laps, 3) p10 / p50 / p90."""
    z = norm.ppf(u)[:, None]
    lo = (q[:, 1] - q[:, 0]) / Z90
    hi = (q[:, 2] - q[:, 1]) / Z90
    per_lap = q[None, :, 1] + np.where(z < 0, z * lo[None, :], z * hi[None, :])
    return per_lap.sum(axis=1)


def plan_rows(base: dict, laps: range, stops: dict[int, str], compound_now: str,
              age_now: float, anchor_lap: int) -> list[dict]:
    """Feature rows for every lap of a plan. stops: in-lap -> compound fitted."""
    rows, comp, age, n_stops = [], compound_now, age_now, 0
    # A stop at or before the first scored lap (pit-now stops at pit_lap; laps start at
    # pit_lap + 1) is applied before scoring. Missing this scored pit-now plans on the old
    # tyres to the flag (Session 5 bug, found from a 100% never-pit break-even table).
    for lap in sorted(s for s in stops if s < laps.start):
        comp, age, n_stops = stops[lap], 0.0, n_stops + 1
    for lap in laps:
        age += 1
        rows.append({**base, "h": lap - anchor_lap, "tyre_life_f": age, "compound_f": comp,
                     "fresh_tyre_f": float(n_stops > 0), "stints_ahead": n_stops,
                     "lap": lap})
        if lap in stops:
            comp, age, n_stops = stops[lap], 0.0, n_stops + 1
    return rows


def candidate_plans(compound_now: str, age_now: float, used: set[str], pit_lap: int,
                    race_laps: int) -> dict[str, list[dict[int, str]]]:
    """pit-now family and stay-out family, each a list of stop dicts."""
    rem_after = race_laps - pit_lap
    pit_now, stay = [], []
    for c in DRY_COMPOUNDS:
        if rem_after <= MAX_STINT_LAPS[c] and len(used | {c}) >= 2:
            pit_now.append({pit_lap: c})
        for later in range(pit_lap + 1, race_laps):
            for c2 in DRY_COMPOUNDS:
                ok = (race_laps - later <= MAX_STINT_LAPS[c2]
                      and later - pit_lap <= MAX_STINT_LAPS[c]
                      and len(used | {c, c2}) >= 2)
                if ok and len(pit_now) < 400:
                    pit_now.append({pit_lap: c, later: c2})
    if len(used) >= 2 and age_now + (race_laps - pit_lap + 1) <= MAX_STINT_LAPS[compound_now]:
        stay.append({})
    for later in range(pit_lap + 1, race_laps):
        for c in DRY_COMPOUNDS:
            if (race_laps - later <= MAX_STINT_LAPS[c] and len(used | {c}) >= 2
                    and age_now + (later - pit_lap + 1) <= MAX_STINT_LAPS[compound_now]):
                stay.append({later: c})
    return {"pit_now": pit_now, "stay_out": stay}


def decide(state: RaceState, driver: str, laps_all_cols: pd.DataFrame, kind: str,
           p_real: dict, pitloss: dict, race_laps: int, soft_bias: float = 0.0,
           rng: np.random.Generator | None = None) -> dict:
    """Pit-now vs stay-out for one car at state.t. Returns a self-describing result."""
    rng = rng or np.random.default_rng(SEED)
    mine = state.laps[state.laps["driver"] == driver].sort_values("lap")
    out = {"driver": driver, "t": state.t, "kind": kind, "accounts_for": ACCOUNTS_FOR,
           "does_not_account_for": NOT_ACCOUNTED, "pit_loss_source": pitloss["source"]}
    if mine.empty:
        return {**out, "decision": "no data"}
    last = mine.iloc[-1]
    # next pit opportunity: end of the current lap if the car can still be told in time
    lap_now = int(last["lap"]) + 1
    expected_end = float(last["session_time_s"]) + float(mine["lap_time_s"].dropna().iloc[-1])
    pit_lap = lap_now if expected_end - state.t > REACTION_S else lap_now + 1
    if pit_lap >= race_laps:
        return {**out, "decision": "race ending, no decision"}
    clean = state.laps[state.laps["is_clean"]]
    anchors = compute_anchors(clean) if len(clean) else pd.DataFrame()
    a = anchors[anchors["driver"] == driver] if len(anchors) else anchors
    if a.empty:
        return {**out, "decision": "no anchor (fewer than 2 clean laps in current stint)"}
    a = a.sort_values("lap").iloc[-1]
    used = set(mine.loc[mine["compound"].isin(DRY_COMPOUNDS), "compound"])
    compound_now, age_now = str(last["compound"]), float(last["tyre_life"])
    age_at_pit = age_now + (pit_lap - int(last["lap"]) - 1)
    base = {"anchor_s": a["anchor_s"], "tyre_life_t": a["tyre_life_t"],
            "compound_t": a["compound_t"], "track_temp": a["track_temp"],
            "air_temp": a["air_temp"], "event": last["event"], "driver": driver,
            "team": last["team"], "race_laps": race_laps}
    plans = candidate_plans(compound_now, age_at_pit, used, pit_lap, race_laps)
    neutral_laps = set(range(pit_lap, pit_lap + NEUTRAL_LAPS.get(kind, 1)))
    laps = range(pit_lap + 1, race_laps + 1)

    if not plans["pit_now"] or not plans["stay_out"]:
        return {**out, "decision": "no legal plan in one family"}
    # one batched prediction for every plan of this car
    frames, meta = [], []
    for family, options in plans.items():
        for k, stops in enumerate(options):
            rows = pd.DataFrame(plan_rows(base, laps, stops, compound_now, age_at_pit,
                                          int(a["lap"])))
            rows["plan"] = len(meta)
            frames.append(rows)
            meta.append((family, k, stops))
    rows = pd.concat(frames, ignore_index=True)
    rows["laps_remaining"] = race_laps - rows["lap"]
    # Laps under the neutralisation run at the same pace on both plans, but only if the call
    # is real. The real-call branch drops them; the false-call branch scores every lap.
    rows["neutral"] = rows["lap"].isin(neutral_laps)
    p10, p50, p90 = predict_laptime(rows)
    soft = (rows["compound_f"] == "SOFT").to_numpy() * soft_bias
    rows["q10"], rows["q50"], rows["q90"] = p10 + soft, p50 + soft, p90 + soft
    scored = {}
    for pid, g in rows.groupby("plan"):
        family, _, stops = meta[pid]
        later = sum(1 for lap in stops if lap != pit_lap)
        total = g["q50"].sum() + later * pitloss["green"]
        if family not in scored or total < scored[family]["p50"]:
            real = ~g["neutral"].to_numpy()
            scored[family] = {"stops": stops, "q_all": g[["q10", "q50", "q90"]].to_numpy(),
                              "q": g.loc[real, ["q10", "q50", "q90"]].to_numpy(),
                              "p50": total, "h": g["h"].to_numpy(), "later_stops": later}
    # Selecting the best of many plans on p50 favours plans whose p50 is optimistic
    # (winner's curse); the governing risk in HANDOFF applies here in small form.

    pn, so = scored["pit_now"], scored["stay_out"]
    phi_draw = rng.uniform(*PHI_RANGE, N_DRAWS)
    green = pitloss["green"]
    cond_loss = green - phi_draw * pitloss[f"delta_lap_{kind.lower()}"]
    results = {}
    for rho in RHOS:
        z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], N_DRAWS)
        u = norm.cdf(z)
        t_pn_lap = quantile_sum(pn["q"], u[:, 0]) + pn["later_stops"] * green
        t_so = quantile_sum(so["q"], u[:, 1]) + so["later_stops"] * green
        t_pn_all = quantile_sum(pn["q_all"], u[:, 0]) + pn["later_stops"] * green
        t_so_all = quantile_sum(so["q_all"], u[:, 1]) + so["later_stops"] * green
        gain_real = t_so - (t_pn_lap + cond_loss)   # + = pitting better, neutralisation real
        gain_false = t_so_all - (t_pn_all + green)  # call false: every lap green, green stop
        results[rho] = {
            "p_pit_better_if_real": float((gain_real > 0).mean()),
            "p_pit_better_if_false": float((gain_false > 0).mean()),
            "p_pit_better": float(p_real["p"] * (gain_real > 0).mean()
                                  + (1 - p_real["p"]) * (gain_false > 0).mean()),
            "gain_if_real_s": [float(np.quantile(gain_real, x)) for x in (0.1, 0.5, 0.9)],
            "gain_if_false_s": [float(np.quantile(gain_false, x)) for x in (0.1, 0.5, 0.9)],
        }
    # Break-even precision: act on the call (pit) when P(call real) > p*, from median gains
    # at a fixed phi (rho 0.5). Free-air time only: the track-position benefit of an SC stop
    # is not modelled, so gain_real is understated and p* is an UPPER BOUND.
    z = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], N_DRAWS)
    u = norm.cdf(z)
    t_pn_lap = quantile_sum(pn["q"], u[:, 0]) + pn["later_stops"] * green
    t_so = quantile_sum(so["q"], u[:, 1]) + so["later_stops"] * green
    t_pn_all = quantile_sum(pn["q_all"], u[:, 0]) + pn["later_stops"] * green
    t_so_all = quantile_sum(so["q_all"], u[:, 1]) + so["later_stops"] * green
    g_false = float(np.median(t_so_all - (t_pn_all + green)))
    break_even = {}
    for phi in (PHI_RANGE[0], PHI, PHI_RANGE[1]):
        g_real = float(np.median(t_so - (t_pn_lap + green - phi
                                           * pitloss[f"delta_lap_{kind.lower()}"])))
        if g_false >= 0:
            p_star = 0.0             # pitting wins even if the call is false
        elif g_real <= 0:
            p_star = None            # pitting loses even if the call is real
        else:
            p_star = -g_false / (g_real - g_false)
        break_even[phi] = {"gain_if_real_s": g_real, "gain_if_false_s": g_false,
                           "p_star": p_star}
    ps = [r["p_pit_better"] for r in results.values()]
    verdict = ("pit" if min(ps) > 0.8 else "stay out" if max(ps) < 0.2
               else "overlapping: the model cannot separate the two plans")
    h_all = np.concatenate([pn["h"], so["h"]])
    return {
        **out, "decision": verdict, "pit_lap": pit_lap, "compound_now": compound_now,
        "tyre_age_at_pit": age_at_pit, "p_call_real": p_real,
        "pit_now_plan": pn["stops"], "stay_out_plan": so["stops"],
        "by_rho": results, "break_even": break_even,
        "share_laps_h_gt_15": float((h_all > 15).mean()),
        "share_laps_h_gt_30": float((h_all > 30).mean()), "soft_bias_s": soft_bias,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clock check and recs for one race.")
    parser.add_argument("--race", required=True)
    args = parser.parse_args(argv)
    sc = pd.read_parquet(SC_PATH)
    laps = pd.read_parquet(LAPS_PATH)
    laps["rid"] = [race_id(s, e) for s, e in zip(laps["season"], laps["event"], strict=True)]
    r = laps[laps["rid"] == args.race]
    season, event = int(r["season"].iloc[0]), str(r["event"].iloc[0])
    recs = load_recs(args.race)
    print(f"{args.race}: {len(recs)} track-wide recs")
    print(recs[["t", "flag", "message", "confidence", "reason"]].to_string(index=False))
    print("\nclock check (Fast Flag official vs our sc_events):")
    print(check_clock(args.race, sc, season, event).round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
