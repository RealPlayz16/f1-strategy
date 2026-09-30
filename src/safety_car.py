"""Safety Car and VSC events per race, and deployment rates per track.

Events come from the FastF1 track status feed, which is on session time, the clock of
laps_clean and of Fast Flag recs. Race control message Time is wall clock and cannot be
mapped to session time without telemetry (FastF1 needs car data for t0_date).
  SC   status 4 (SCDeployed) until the next 1 (AllClear), 5 (Red) or 6 (VSC)
  VSC  status 6 (VSCDeployed) until the next 1, 4 (converted to SC) or 5; 7 (VSCEnding)
       is kept as t_ending_s
  RED  status 5
Only events between session status Started and Finished count. Laps are the leader's lap
at that time. Race control deployment messages are counted per race as a cross-check.

Rates use train races only. A lap-1 deployment is a start hazard, not a per-lap one, so it
is counted apart (p_sc_lap1, share of train races with an SC on lap 1). Per-lap rates per
track come from 1 to 3 races, so they are shrunk toward the global train rate with a prior
worth PRIOR_LAPS laps (about one race distance):
    p = (deployments + PRIOR_LAPS * p_global) / (laps + PRIOR_LAPS)
Raw rates are kept next to them.

Usage:
    python -m src.safety_car
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.splits import load_splits, split_of

RAW_DIR = Path("data/processed/raw")
EVENTS_PATH = Path("data/processed/sc_events.parquet")
RATES_PATH = Path("data/processed/sc_rates.parquet")

PRIOR_LAPS = 60.0

START = {"4": "SC", "6": "VSC", "5": "RED"}
ENDS = {"SC": {"1", "5", "6"}, "VSC": {"1", "4", "5"}, "RED": {"1", "4", "6"}}
RCM_DEPLOY = {"SAFETY CAR DEPLOYED": "SC", "VIRTUAL SAFETY CAR DEPLOYED": "VSC"}


def leader_lap_at(laps: pd.DataFrame, t: float) -> int:
    """Leader's lap number at session time t: 1 + leader lap completions before t."""
    ends = laps.groupby("LapNumber")["Time"].min().sort_values()
    return int((ends < t).sum()) + 1


def race_window(session_status: pd.DataFrame) -> tuple[float, float]:
    s = session_status.set_index("Status")["Time"]
    return float(s.get("Started", 0.0)), float(s.get("Finished", np.inf))


def race_events(track_status: pd.DataFrame, laps: pd.DataFrame, window: tuple[float, float]):
    """SC / VSC / RED episodes from the track status feed."""
    ts = track_status.sort_values("Time").reset_index(drop=True)
    ts["Status"] = ts["Status"].astype(str)
    t_start, t_finish = window
    events = []
    prev = None
    for i, row in ts.iterrows():
        status, t = row["Status"], float(row["Time"])
        if status in START and status != prev and t_start <= t <= t_finish:
            kind = START[status]
            later = ts.iloc[i + 1:]
            end = later[later["Status"].isin(ENDS[kind])].head(1)
            t_end = float(end["Time"].iloc[0]) if len(end) else t_finish
            ended_by = end["Message"].iloc[0] if len(end) else "Finished"
            ending = later[(later["Status"] == "7") & (later["Time"] < t_end)].head(1)
            lap_deploy, lap_end = leader_lap_at(laps, t), leader_lap_at(laps, t_end)
            events.append(
                {
                    "kind": kind,
                    "lap_deploy": lap_deploy,
                    "t_deploy_s": t,
                    "lap_end": lap_end,
                    "t_end_s": t_end,
                    "t_ending_s": float(ending["Time"].iloc[0]) if len(ending) else np.nan,
                    "ended_by": str(ended_by),
                    "duration_s": t_end - t,
                    "duration_laps": lap_end - lap_deploy,
                }
            )
        prev = status
    return events


def build_events(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All races: (events, one row per race with race_laps, split, rcm cross-check counts)."""
    splits = load_splits()
    events, races = [], []
    for p in sorted(raw_dir.iterdir()):
        if not p.is_dir():
            continue
        meta = pd.read_parquet(p / "meta.parquet").iloc[0]
        laps = pd.read_parquet(p / "laps.parquet")
        rcm = pd.read_parquet(p / "rcm.parquet")
        season, event = int(meta["season"]), str(meta["event"])
        rcm_kinds = rcm["Message"].map(RCM_DEPLOY).value_counts()
        race = {
            "season": season, "round": int(meta["round"]), "event": event,
            "split": split_of(season, event, splits), "race_laps": int(meta["race_laps"]),
        }
        races.append(
            {**race, "rcm_sc": int(rcm_kinds.get("SC", 0)), "rcm_vsc": int(rcm_kinds.get("VSC", 0))}
        )
        window = race_window(pd.read_parquet(p / "session_status.parquet"))
        track_status = pd.read_parquet(p / "track_status.parquet")
        for ev in race_events(track_status, laps, window):
            events.append({**race, **ev})
    return pd.DataFrame(events), pd.DataFrame(races)


def rates(events: pd.DataFrame, races: pd.DataFrame, prior_laps: float = PRIOR_LAPS):
    """Per-track per-lap deployment rates from train races, shrunk to the global rate."""
    races = races[races["split"] == "train"]
    ev = events[events["split"] == "train"]
    lap1 = ev["lap_deploy"] <= 1
    per_lap = ev[~lap1]

    counts = (
        per_lap.groupby(["event", "kind"]).size().unstack("kind")
        .reindex(columns=["SC", "VSC"]).fillna(0)
    )
    out = races.groupby("event").agg(n_races=("season", "size"), race_laps=("race_laps", "sum"))
    out = out.join(counts).fillna(0)
    out = out.rename(columns={"SC": "sc_deployments", "VSC": "vsc_deployments"})
    laps_at_risk = out["race_laps"] - out["n_races"]  # lap 1 counted apart

    lap1_races = ev[lap1 & (ev["kind"] == "SC")][["season", "round"]].drop_duplicates()
    for kind in ("sc", "vsc"):
        n = out[f"{kind}_deployments"]
        p_global = n.sum() / laps_at_risk.sum()
        out[f"p_{kind}_per_lap_raw"] = n / laps_at_risk
        out[f"p_{kind}_per_lap"] = (n + prior_laps * p_global) / (laps_at_risk + prior_laps)
    out["p_sc_lap1"] = len(lap1_races) / len(races)
    cols = [
        "n_races", "race_laps", "sc_deployments", "vsc_deployments", "p_sc_per_lap",
        "p_vsc_per_lap", "p_sc_per_lap_raw", "p_vsc_per_lap_raw", "p_sc_lap1",
    ]
    return out[cols].reset_index()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SC / VSC events and rates.")
    parser.parse_args(argv)

    events, races = build_events()
    table = rates(events, races)
    events.to_parquet(EVENTS_PATH, index=False)
    table.to_parquet(RATES_PATH, index=False)

    pd.set_option("display.width", 200)
    print(f"races: {len(races)}  events: {events['kind'].value_counts().to_dict()}")
    ours = events.groupby(["season", "event", "kind"]).size().unstack("kind")
    check = races.set_index(["season", "event"])[["rcm_sc", "rcm_vsc"]]
    check = check.join(ours.reindex(columns=["SC", "VSC"])).fillna(0)
    diff = check[(check["rcm_sc"] != check["SC"]) | (check["rcm_vsc"] != check["VSC"])]
    print(f"races where track status and race control counts differ: {len(diff)}")
    if len(diff):
        print(diff.astype(int).to_string())
    print("\nevents:")
    cols = ["season", "event", "split", "kind", "lap_deploy", "lap_end", "ended_by",
            "duration_laps", "duration_s"]
    print(events[cols].round(1).to_string(index=False))
    train = events[(events["split"] == "train") & (events["kind"] != "RED")]
    print("\ntrain durations (median):")
    print(train.groupby("kind")[["duration_laps", "duration_s"]].median().round(1).to_string())
    print("\nrates (train only):")
    print(table.round(4).to_string(index=False))
    print(f"\nwrote {EVENTS_PATH} and {RATES_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
