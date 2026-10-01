"""Backtest of the live decision on holdout races. A comparison, never a claim.

What the system would have called, what each team did, what happened. The engine cannot
support "we would have done better" (traffic, track position and queue position under SC
are not modelled; Session 4), so no output makes that claim.

Per holdout race:
  call episodes   Fast Flag SC / VSC recs, a new episode after each TRACK CLEAR. Real if an
                  official neutralisation (sc_events) deploys within MATCH_WINDOW_S of the
                  call, else a false call.
  decisions       src.live.decide for every running car at each call (call carried as a
                  probability) and at each real deployment with certainty (the no-call path:
                  what happens when Fast Flag misses or is ignored).
  changed / earlier   per car, whether the recommendation at the call differs from the one
                  at the deployment (changed) or only arrives earlier.
  early-call windows  per car, the first pit opportunity after the call vs after the
                  deployment, and whether the later one still falls inside the
                  neutralisation. Uses the race's actual lap-end times after the fact, for
                  evaluation only; decisions never see them.
  false-call cost     for false calls, the free-air time cost of pitting on the call
                  (stop under green instead of the stay-out plan).

The holdout has one real neutralisation (2025 United States, VSC lap 7). Results are a
case study plus the false-call record, not a statistical test.

Usage:
    python -m src.backtest
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.live import (
    LAPS_PATH,
    PHI,
    PITLOSS_PATH,
    REACTION_S,
    SC_PATH,
    ReplayDriver,
    call_probability,
    check_clock,
    decide,
    load_recs,
    pit_loss_estimate,
    race_id,
)
from src.splits import add_split

OUT_PATH = Path("data/processed/backtest.json")
DECISIONS_PATH = Path("data/processed/backtest_decisions.parquet")
MATCH_WINDOW_S = 180.0
BREAK_EVEN_EVERY = 10
BREAK_EVEN_PATH = Path("data/processed/break_even.parquet")


def episodes(recs: pd.DataFrame) -> pd.DataFrame:
    """First SC / VSC rec after each TRACK CLEAR (or race start)."""
    rows, open_ = [], False
    for _, r in recs.iterrows():
        if r["message"] == "TRACK CLEAR":
            open_ = False
        elif r["flag"] in ("SC", "VSC") and not open_:
            rows.append({"t_call": float(r["t"]), "kind": r["flag"], "reason": r["reason"],
                         "confidence": float(r["confidence"])})
            open_ = True
    return pd.DataFrame(rows, columns=["t_call", "kind", "reason", "confidence"])


def match(eps: pd.DataFrame, neutral: pd.DataFrame) -> pd.DataFrame:
    eps = eps.copy()
    eps["real"], eps["t_deploy"], eps["actual_kind"], eps["t_end"] = False, np.nan, None, np.nan
    for i, e in eps.iterrows():
        cand = neutral[(neutral["t_deploy_s"] >= e["t_call"])
                       & (neutral["t_deploy_s"] <= e["t_call"] + MATCH_WINDOW_S)]
        if len(cand):
            n = cand.iloc[0]
            eps.loc[i, ["real", "t_deploy", "actual_kind", "t_end"]] = (
                True, n["t_deploy_s"], n["kind"], n["t_end_s"])
    eps["lead_s"] = eps["t_deploy"] - eps["t_call"]
    return eps


def next_opportunity(laps: pd.DataFrame, driver: str, t: float) -> float:
    """Session time of the car's first lap end after t + REACTION_S (pit entry is just before
    the line). Evaluation only."""
    ends = laps.loc[laps["driver"] == driver, "session_time_s"].dropna()
    after = ends[ends > t + REACTION_S]
    return float(after.min()) if len(after) else np.nan


def team_pitted(laps: pd.DataFrame, driver: str, t0: float, t1: float) -> bool:
    d = laps[(laps["driver"] == driver) & laps["is_pit_in"]]
    return bool(((d["session_time_s"] > t0) & (d["session_time_s"] <= t1)).any())


def summarise(dec: dict) -> dict:
    keys = ["driver", "t", "kind", "decision", "pit_lap", "compound_now", "tyre_age_at_pit",
            "share_laps_h_gt_15", "share_laps_h_gt_30", "soft_bias_s"]
    row = {k: dec.get(k) for k in keys}
    for rho, r in dec.get("by_rho", {}).items():
        row[f"p_pit_better_rho{rho}"] = r["p_pit_better"]
        row[f"p_if_real_rho{rho}"] = r["p_pit_better_if_real"]
        row[f"gain_if_real_p50_rho{rho}"] = r["gain_if_real_s"][1]
        row[f"gain_if_false_p50_rho{rho}"] = r["gain_if_false_s"][1]
    row["pit_now_plan"] = str(dec.get("pit_now_plan"))
    row["stay_out_plan"] = str(dec.get("stay_out_plan"))
    return row


def run_race(rid: str, laps: pd.DataFrame, sc: pd.DataFrame, pitloss_tab: pd.DataFrame,
             p_calls: dict) -> tuple[dict, list[dict]]:
    r = laps[laps["rid"] == rid].copy()
    season, event = int(r["season"].iloc[0]), str(r["event"].iloc[0])
    race_laps = int(r["race_laps"].iloc[0])
    neutral = sc[(sc["season"] == season) & (sc["event"] == event)
                 & sc["kind"].isin(["SC", "VSC"])]
    recs = load_recs(rid)
    eps = match(episodes(recs), neutral)
    clock = check_clock(rid, sc, season, event)
    driver = ReplayDriver(r, recs, neutral)
    pitloss = pit_loss_estimate(pitloss_tab, season, event, PHI)
    median_lap = float(r.loc[r["is_clean"], "lap_time_s"].median())

    rows = []
    points = [("call", e["t_call"], e["kind"], p_calls[e["kind"]], i)
              for i, e in eps.iterrows()]
    called = set(eps.loc[eps["real"], "t_deploy"])
    for _, n in neutral.iterrows():
        sure = {"p": 1.0, "q05": 1.0, "q95": 1.0, "hits": 0, "n": 0}
        points.append(("deploy", float(n["t_deploy_s"]), n["kind"], sure,
                       "called" if n["t_deploy_s"] in called else "missed"))
    for source, t, kind, p_real, tag in points:
        state = driver.at(t)
        running = state.laps.groupby("driver")["lap"].max()
        for drv in running.index:
            for bias in (0.0, 0.075):
                dec = decide(state, drv, r, kind, p_real, pitloss, race_laps, soft_bias=bias)
                rows.append({**summarise(dec), "race": rid, "source": source, "tag": str(tag),
                             "kind_called": kind})

    # early-call windows and what teams did, real episodes only (after the fact)
    windows = []
    for _, e in eps[eps["real"]].iterrows():
        for drv in r["driver"].unique():
            with_call = next_opportunity(r, drv, e["t_call"])
            no_call = next_opportunity(r, drv, e["t_deploy"])
            gained = with_call < no_call
            windows.append({
                "race": rid, "driver": drv, "t_call": e["t_call"], "t_deploy": e["t_deploy"],
                "extra_window": bool(gained),
                "no_call_window_inside_neutralisation": bool(no_call <= e["t_end"]),
                "team_pitted_during": team_pitted(r, drv, e["t_call"], e["t_end"]),
            })
    win = pd.DataFrame(windows)
    out = {
        "race": rid, "pit_loss": pitloss, "median_lap_s": median_lap,
        "clock_check": clock.round(3).to_dict(orient="records"),
        "episodes": eps.round(2).astype(object).where(eps.notna(), None).to_dict(
            orient="records"),
        "lead_laps_at_ff_median_32_6s": 32.6 / median_lap,
        "windows": win.to_dict(orient="records") if len(win) else [],
    }
    return out, rows


def break_even_map(rid: str, laps: pd.DataFrame, sc: pd.DataFrame, pitloss_tab: pd.DataFrame,
                   every: int = BREAK_EVEN_EVERY) -> list[dict]:
    """Hypothetical call at the end of every `every`-th lap, every running car, both kinds:
    the precision p* above which pitting on the call beats ignoring it. Needs no Fast Flag
    timeline, so it covers race situations the real calls never reached."""
    r = laps[laps["rid"] == rid].copy()
    season, event = int(r["season"].iloc[0]), str(r["event"].iloc[0])
    race_laps = int(r["race_laps"].iloc[0])
    driver = ReplayDriver(r, pd.DataFrame(columns=["t", "flag"]), sc.iloc[0:0])
    pitloss = pit_loss_estimate(pitloss_tab, season, event, PHI)
    rows = []
    for lap in range(every, race_laps - 4, every):
        state = driver.at_lap(lap)
        last = state.laps.groupby("driver")["lap"].max()
        for drv in last.index[last >= lap - 1]:  # every car still running (leader ends lap)
            for kind in ("SC", "VSC"):
                dec = decide(state, drv, r, kind, {"p": 0.5}, pitloss, race_laps)
                be = dec.get("break_even", {})
                row = {"race": rid, "lap": lap, "driver": drv, "kind": kind,
                       "decision": dec["decision"], "compound_now": dec.get("compound_now"),
                       "tyre_age_at_pit": dec.get("tyre_age_at_pit"),
                       "stay_out_stops": len(dec.get("stay_out_plan") or {}),
                       "laps_left": race_laps - lap}
                for phi, v in be.items():
                    row[f"p_star_phi{phi}"] = v["p_star"]
                    row[f"gain_real_phi{phi}"] = v["gain_if_real_s"]
                    row[f"gain_false_phi{phi}"] = v["gain_if_false_s"]
                rows.append(row)
        print(f"  {rid} lap {lap}", flush=True)
    return rows


def main() -> int:
    laps = add_split(pd.read_parquet(LAPS_PATH).drop(columns=["split"], errors="ignore"))
    laps["rid"] = [race_id(s, e) for s, e in zip(laps["season"], laps["event"], strict=True)]
    hold = sorted(laps.loc[laps["split"] == "holdout", "rid"].unique())
    sc = pd.read_parquet(SC_PATH)
    pitloss_tab = pd.read_parquet(PITLOSS_PATH)
    p_calls = {k: call_probability(k) for k in ("SC", "VSC")}
    if "--break-even" in sys.argv:
        be = [row for rid in hold for row in break_even_map(rid, laps, sc, pitloss_tab)]
        pd.DataFrame(be).to_parquet(BREAK_EVEN_PATH, index=False)
        print(f"wrote {BREAK_EVEN_PATH} ({len(be)} hypothetical calls)")
        return 0
    races, rows = [], []
    for rid in hold:
        try:
            res, dec = run_race(rid, laps, sc, pitloss_tab, p_calls)
        except FileNotFoundError as e:
            print(f"{rid}: no Fast Flag timeline ({e})")
            continue
        races.append(res)
        rows += dec
        print(f"{rid}: {len(res['episodes'])} call episodes, {len(dec)} decisions", flush=True)
    pd.DataFrame(rows).to_parquet(DECISIONS_PATH, index=False)
    OUT_PATH.write_text(json.dumps({"p_calls": p_calls, "races": races}, indent=2,
                                   default=str), encoding="utf-8")
    print(f"wrote {OUT_PATH} and {DECISIONS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
