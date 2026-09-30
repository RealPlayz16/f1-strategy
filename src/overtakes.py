"""On-track passes and close battles, from lap-end order.

A passes B on lap k when A crossed the line behind B at the end of lap k-1 and ahead of
B at the end of lap k. Order is lap-end session time within the same lap number, so a
leader lapping a backmarker is not a pass.

A car is eligible on lap k when it completed laps k-1 and k, has no in-lap on k-1 or k and
no out-lap on k, and lap k is not under SC, VSC or red. k >= 2. Cars that are pitting are
removed before ranking, so passing a car in the pit lane is not counted.

Features, overtaker A vs overtaken (or car ahead) B:
  gap_before_s    B ahead of A at the end of lap k-1, seconds
  pace_delta_s    B recent pace - A recent pace; recent pace is the median of the last
                  PACE_LAPS green, non-pit laps up to k-1. Positive: A was faster
  tyre_age_delta  B tyre life - A tyre life on lap k. Positive: A on fresher tyres
  compound_pair   "A_COMPOUND-B_COMPOUND"
  drs_likely      gap_before_s < 1.0 (DRS enabled is not checked)

Outputs (all races, split column; fit on train only):
  overtakes.parquet  every pass
  battles.parquet    every car within BATTLE_GAP_S of the eligible car directly ahead at
                     the end of lap k-1, with passed = True if it was ahead at the end of k.
                     Includes the attempts that failed, which a pass model needs.

Usage:
    python -m src.overtakes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LAPS_PATH = Path("data/processed/laps_clean.parquet")
OVERTAKES_PATH = Path("data/processed/overtakes.parquet")
BATTLES_PATH = Path("data/processed/battles.parquet")

PACE_LAPS = 3
DRS_GAP_S = 1.0
BATTLE_GAP_S = 2.0

KEYS = ["season", "round", "event", "split"]


def recent_pace(race: pd.DataFrame) -> pd.Series:
    """Per (driver, lap): median of the last PACE_LAPS green, non-pit lap times up to lap."""
    ok = (
        race["lap_time_s"].notna() & ~race["is_pit_in"] & ~race["is_pit_out"]
        & ~race["is_sc"] & ~race["is_vsc"] & ~race["is_red"] & ~race["is_lap1"]
    )
    lt = race["lap_time_s"].where(ok)
    pace = lt.groupby(race["driver"]).transform(
        lambda s: s.rolling(PACE_LAPS, min_periods=1).median().ffill()
    )
    return pd.Series(pace.to_numpy(), index=pd.MultiIndex.from_frame(race[["driver", "lap"]]))


def eligible_pairs(race: pd.DataFrame) -> pd.DataFrame:
    """One row per (driver, lap k) eligible for pass detection, with lap k-1 values."""
    race = race.sort_values(["driver", "lap"])
    prev = race.groupby("driver").shift(1)
    flagged = race["is_sc"] | race["is_vsc"] | race["is_red"]
    ok = (
        (race["lap"] >= 2)
        & (prev["lap"] == race["lap"] - 1)
        & race["session_time_s"].notna()
        & prev["session_time_s"].notna()
        & ~race["is_pit_in"]
        & ~race["is_pit_out"]
        & ~prev["is_pit_in"].astype("boolean").fillna(True)
        & ~flagged
    )
    pace = recent_pace(race)
    out = race.loc[ok, KEYS + ["driver", "lap", "compound", "tyre_life", "session_time_s"]].copy()
    out["t_prev"] = prev.loc[ok, "session_time_s"]
    out["pace_prev"] = pace.reindex(
        pd.MultiIndex.from_arrays([out["driver"], out["lap"] - 1])
    ).to_numpy()
    return out.rename(columns={"session_time_s": "t_cur"})


def pair_rows(a: pd.Series, b: pd.Series) -> dict:
    gap = float(a["t_prev"] - b["t_prev"])
    return {
        **{k: a[k] for k in KEYS},
        "lap": int(a["lap"]),
        "gap_before_s": gap,
        "pace_delta_s": float(b["pace_prev"] - a["pace_prev"]),
        "tyre_age_delta": float(b["tyre_life"] - a["tyre_life"]),
        "compound_pair": f"{a['compound']}-{b['compound']}",
        "drs_likely": gap < DRS_GAP_S,
    }


def race_passes(race: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    elig = eligible_pairs(race)
    passes, battles = [], []
    for _, lap in elig.groupby("lap"):
        lap = lap.sort_values("t_prev").reset_index(drop=True)
        t_prev, t_cur = lap["t_prev"].to_numpy(), lap["t_cur"].to_numpy()
        # i behind j before (t_prev larger) and ahead after (t_cur smaller)
        behind = t_prev[:, None] > t_prev[None, :]
        ahead = t_cur[:, None] < t_cur[None, :]
        for i, j in zip(*np.nonzero(behind & ahead), strict=True):
            a, b = lap.iloc[i], lap.iloc[j]
            passes.append({**pair_rows(a, b), "overtaker": a["driver"], "overtaken": b["driver"]})
        for i in range(1, len(lap)):
            a, b = lap.iloc[i], lap.iloc[i - 1]
            if a["t_prev"] - b["t_prev"] < BATTLE_GAP_S:
                battles.append(
                    {**pair_rows(a, b), "driver": a["driver"], "ahead": b["driver"],
                     "passed": bool(a["t_cur"] < b["t_cur"])}
                )
    return passes, battles


def build(laps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    passes, battles = [], []
    for _, race in laps.groupby(["season", "round"], sort=True):
        p, b = race_passes(race)
        passes += p
        battles += b
    over_cols = KEYS + ["lap", "overtaker", "overtaken", "pace_delta_s", "tyre_age_delta",
                        "compound_pair", "gap_before_s", "drs_likely"]
    battle_cols = KEYS + ["lap", "driver", "ahead", "gap_before_s", "pace_delta_s",
                          "tyre_age_delta", "compound_pair", "drs_likely", "passed"]
    return pd.DataFrame(passes, columns=over_cols), pd.DataFrame(battles, columns=battle_cols)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract passes and battles.")
    parser.add_argument("--laps", default=str(LAPS_PATH))
    args = parser.parse_args(argv)

    laps = pd.read_parquet(args.laps)
    over, battles = build(laps)
    over.to_parquet(OVERTAKES_PATH, index=False)
    battles.to_parquet(BATTLES_PATH, index=False)

    pd.set_option("display.width", 200)
    print(f"passes: {len(over)}  battles (gap < {BATTLE_GAP_S} s): {len(battles)}")
    print("\npasses per race:")
    print(over.groupby(["season", "event"]).size().to_string())
    print("\npasses, median features:")
    print(over[["gap_before_s", "pace_delta_s", "tyre_age_delta"]].median().round(3).to_string())
    print(f"share drs_likely: {over['drs_likely'].mean():.2f}")
    print("\nbattle pass rate by gap bin:")
    bins = pd.cut(battles["gap_before_s"], [0, 0.5, 1.0, 1.5, 2.0])
    print(battles.groupby(bins, observed=True)["passed"].agg(["mean", "size"]).round(3))
    print(f"\nwrote {OVERTAKES_PATH} and {BATTLES_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
