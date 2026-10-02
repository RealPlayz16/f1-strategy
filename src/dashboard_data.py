"""Export a self-contained demo snapshot per race into data/demo/<race>.json.

data/processed/ and data/models/ are gitignored (they are rebuilt from FastF1 and from Fast
Flag timelines), so the dashboard reads these committed snapshots instead. Everything the
page needs is precomputed here: the running order lap by lap, the Fast Flag calls, the
official neutralisations, and the decisions already computed by src/backtest.py.

Nothing in the snapshot is recomputed at serve time, so the page cannot leak the future by
accident: each event carries only the decisions that src.live.decide made from the state as
of that moment.

Usage:
    python -m src.dashboard_data
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from src.live import ACCOUNTS_FOR, LAPS_PATH, NOT_ACCOUNTED, SC_PATH, load_recs, race_id

DECISIONS_PATH = Path("data/processed/backtest_decisions.parquet")
BACKTEST_PATH = Path("data/processed/backtest.json")
OUT_DIR = Path("data/demo")
DEMO_RACES = ["2025_United_States", "2025_Singapore"]
RHOS = ("0.0", "0.5", "0.9")


def lap_in_progress(lap_ends: pd.Series, t: float) -> int:
    """Lap the leader is on at session time t (same convention as src/safety_car.py)."""
    return int((lap_ends < t).sum()) + 1


def order_by_lap(r: pd.DataFrame) -> list[dict]:
    """Running order at the end of each lap, by session time."""
    out = []
    for lap, g in r[r["session_time_s"].notna()].groupby("lap"):
        g = g.sort_values("session_time_s")
        leader_t = float(g["session_time_s"].iloc[0])
        out.append({
            "lap": int(lap),
            "t": leader_t,
            "order": [
                {
                    "driver": str(row["driver"]),
                    "pos": i + 1,
                    "compound": str(row["compound"]),
                    "age": None if pd.isna(row["tyre_life"]) else int(row["tyre_life"]),
                    "gap_s": round(float(row["session_time_s"]) - leader_t, 2),
                    "pit": bool(row["is_pit_in"]),
                }
                for i, (_, row) in enumerate(g.iterrows())
            ],
        })
    return out


def decision_cards(dec: pd.DataFrame) -> list[dict]:
    """One card per driver: the baseline decision plus the soft-bias sensitivity."""
    cards = []
    base = dec[dec["soft_bias_s"] == 0]
    soft = dec[dec["soft_bias_s"] > 0].set_index("driver")
    for _, row in base.sort_values("driver").iterrows():
        card = {
            "driver": str(row["driver"]),
            "decision": str(row["decision"]),
            "decided": not str(row["decision"]).startswith(("no ", "race")),
            "compound_now": None if pd.isna(row["compound_now"]) else str(row["compound_now"]),
            "pit_lap": None if pd.isna(row["pit_lap"]) else int(row["pit_lap"]),
            "tyre_age": None if pd.isna(row["tyre_age_at_pit"]) else int(row["tyre_age_at_pit"]),
            "pit_now_plan": str(row["pit_now_plan"]),
            "stay_out_plan": str(row["stay_out_plan"]),
            "share_h_gt_30": None if pd.isna(row["share_laps_h_gt_30"])
            else round(float(row["share_laps_h_gt_30"]), 3),
        }
        if card["decided"]:
            card["p"] = {rho: round(float(row[f"p_pit_better_rho{rho}"]), 3) for rho in RHOS}
            card["gain_if_real"] = round(float(row["gain_if_real_p50_rho0.5"]), 2)
            card["gain_if_false"] = round(float(row["gain_if_false_p50_rho0.5"]), 2)
            if row["driver"] in soft.index:
                s = soft.loc[row["driver"]]
                card["p_soft_bias"] = {rho: round(float(s[f"p_pit_better_rho{rho}"]), 3)
                                       for rho in RHOS}
        cards.append(card)
    return cards


def export(rid: str, laps: pd.DataFrame, sc: pd.DataFrame, dec: pd.DataFrame,
           p_calls: dict) -> dict:
    r = laps[laps["rid"] == rid].copy()
    season, event = int(r["season"].iloc[0]), str(r["event"].iloc[0])
    race_laps = int(r["race_laps"].iloc[0])
    lap_ends = r.groupby("lap")["session_time_s"].min().sort_values()
    neutral = sc[(sc["season"] == season) & (sc["event"] == event)
                 & sc["kind"].isin(["SC", "VSC"])]
    recs = load_recs(rid)
    d = dec[dec["race"] == rid]

    events = []
    for _, rec in recs[recs["flag"].isin(["SC", "VSC", "RED"])].iterrows():
        t = float(rec["t"])
        kind = str(rec["flag"])
        g = d[(d["source"] == "call") & (d["kind_called"] == kind)]
        events.append({
            "kind": "call", "t": t, "lap": lap_in_progress(lap_ends, t), "flag": kind,
            "label": f"Fast Flag calls {rec['message']}",
            "reason": str(rec["reason"]), "confidence": float(rec["confidence"]),
            "p_call_real": p_calls.get(kind, {}).get("p"),
            "p_call_real_range": [p_calls.get(kind, {}).get("q05"),
                                  p_calls.get(kind, {}).get("q95")],
            "decisions": decision_cards(g) if len(g) else [],
        })
    for _, n in neutral.iterrows():
        t = float(n["t_deploy_s"])
        g = d[(d["source"] == "deploy")]
        events.append({
            "kind": "deploy", "t": t, "lap": int(n["lap_deploy"]), "flag": str(n["kind"]),
            "label": f"Race control deploys {n['kind']}",
            "reason": "official neutralisation (sc_events.parquet)", "confidence": None,
            "p_call_real": 1.0, "p_call_real_range": None,
            "lap_end": None if pd.isna(n["lap_end"]) else int(n["lap_end"]),
            "decisions": decision_cards(g) if len(g) else [],
        })
    events.sort(key=lambda e: e["t"])

    return {
        "race": rid, "season": season, "event": event, "race_laps": race_laps,
        "median_lap_s": round(float(r.loc[r["is_clean"], "lap_time_s"].median()), 2),
        "laps": order_by_lap(r),
        "neutral": [
            {"kind": str(n["kind"]), "lap_deploy": int(n["lap_deploy"]),
             "lap_end": None if pd.isna(n["lap_end"]) else int(n["lap_end"]),
             "duration_s": round(float(n["duration_s"]), 1)}
            for _, n in neutral.iterrows()
        ],
        "events": events,
        "accounts_for": ACCOUNTS_FOR,
        "not_accounted": NOT_ACCOUNTED,
    }


def main() -> int:
    laps = pd.read_parquet(LAPS_PATH)
    laps["rid"] = [race_id(s, e) for s, e in zip(laps["season"], laps["event"], strict=True)]
    sc = pd.read_parquet(SC_PATH)
    dec = pd.read_parquet(DECISIONS_PATH)
    p_calls = json.loads(BACKTEST_PATH.read_text(encoding="utf-8"))["p_calls"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for rid in DEMO_RACES:
        snap = export(rid, laps, sc, dec, p_calls)
        path = OUT_DIR / f"{rid}.json"
        path.write_text(json.dumps(snap, separators=(",", ":")), encoding="utf-8")
        n_dec = sum(len(e["decisions"]) for e in snap["events"])
        index.append({"race": rid, "event": snap["event"], "season": snap["season"],
                      "race_laps": snap["race_laps"], "events": len(snap["events"])})
        print(f"wrote {path} ({path.stat().st_size // 1024} KB, {len(snap['laps'])} laps, "
              f"{len(snap['events'])} events, {n_dec} decision cards)")
    (OUT_DIR / "index.json").write_text(json.dumps({"races": index, "p_calls": p_calls},
                                                   indent=1), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'index.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
