"""Traffic state per lap and the dirty air cost. Train races only for any fit.

Free-air pace. A follower's lap time is understated exactly when it follows (in close battles
the observed recent-lap pace delta is 0.22 s against 0.62 s in free air), so the pass model
and the dirty air curve both use free-air pace:
    free_pace(lap) = stint baseline + deg[race, compound] * tyre_life
The stint baseline is the median of (lap_time_fc_s - deg * tyre_life) over that driver's
clean laps in the stint with more than FREE_AIR_GAP_S to the car ahead at the end of the
previous lap. deg is the joint per-race slope from src/degradation.py (compound median
where missing). Stints with no free-air lap have no free pace: about half the close battles.

Dirty air cost. dev = lap_time_fc_s - free_pace, by gap to the car ahead at the end of the
previous lap (predetermined), on laps where the same car was still directly ahead at the end
of the lap (no pass either way). Two populations:
  not faster than the car ahead in free air: dev is the aero loss only
  faster: aero loss plus being held up behind a slower car
The race engine produces the held-up part itself (a car that cannot pass cannot lap faster
than the car ahead), so it must take the aero-only curve, not the pooled one, or it counts
being held up twice.

Usage:
    python -m src.traffic
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.splits import add_split

LAPS_PATH = Path("data/processed/laps_fuel_corrected.parquet")
SLOPES_PATH = Path("data/processed/degradation_slopes.parquet")
STATE_PATH = Path("data/processed/traffic_laps.parquet")
DIRTY_AIR_PATH = Path("data/processed/dirty_air.parquet")

SEED = 42
FREE_AIR_GAP_S = 3.0
GAP_BINS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf]
GAP_LABELS = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", ">=3"]
N_BOOT = 200

# Aero penalty for the race engine. PARAMETERISED, not fitted: aero loss and being held up
# do not separate cleanly. Splitting followers by free-air pace delta (an estimate) leaves a
# selection offset of -0.085 / +0.167 s at 2-3 s, where both effects should be about 0, as
# large as the effect itself. Aero-only median at 0-0.5 s: 0.35 s, race bootstrap 0.24 to
# 0.66, n = 128; about 0 beyond 0.5 s. The engine adds being held up mechanically.
DIRTY_AIR_D0_S = 0.35
DIRTY_AIR_D0_RANGE_S = (0.24, 0.66)  # Monte Carlo sweep
DIRTY_AIR_FULL_GAP_S = 0.5
DIRTY_AIR_ZERO_GAP_S = 1.0


def dirty_air_penalty(gap_s: np.ndarray | float, d0: float = DIRTY_AIR_D0_S) -> np.ndarray:
    """Aero lap time penalty for a follower gap_s behind: d0 up to 0.5 s, linear to 0 at
    1.0 s, 0 beyond."""
    gap = np.asarray(gap_s, dtype=float)
    span = DIRTY_AIR_ZERO_GAP_S - DIRTY_AIR_FULL_GAP_S
    frac = (DIRTY_AIR_ZERO_GAP_S - gap) / span
    return d0 * np.where(gap <= DIRTY_AIR_FULL_GAP_S, 1.0, np.where(gap >= DIRTY_AIR_ZERO_GAP_S,
                                                                     0.0, frac))


def load_train() -> pd.DataFrame:
    laps = add_split(pd.read_parquet(LAPS_PATH).drop(columns=["split"], errors="ignore"))
    laps = laps[laps["split"] == "train"].copy()
    laps["race"] = laps["season"].astype(str) + "_" + laps["round"].astype(str).str.zfill(2)
    return laps


def lap_state(laps: pd.DataFrame, slopes: pd.DataFrame) -> pd.DataFrame:
    """Per (race, driver, lap): gap and car ahead at the end of the previous lap, car ahead at
    the end of this lap, free-air pace of this car and of the car ahead."""
    df = laps.sort_values(["race", "lap", "session_time_s"]).copy()
    df["ahead_now"] = df.groupby(["race", "lap"])["driver"].shift(1)
    df = df.sort_values(["race", "driver", "lap"])
    g = df.groupby(["race", "driver"])
    df["gap_prev"] = g["gap_ahead_s"].shift(1)
    df["ahead_prev"] = g["ahead_now"].shift(1)
    prev_ok = g["lap"].diff() == 1
    # Leading the lap order: no car ahead, so clear air (gap_ahead_s is NaN for the leader)
    df.loc[prev_ok & df["ahead_prev"].isna(), "gap_prev"] = np.inf
    df.loc[~prev_ok, ["gap_prev", "ahead_prev"]] = np.nan

    deg = slopes.set_index(["race", "compound"])["slope_B"]
    fallback = slopes.groupby("compound")["slope_B"].median()
    keys = pd.MultiIndex.from_arrays([df["race"], df["compound"]])
    df["deg"] = deg.reindex(keys).to_numpy()
    df["deg"] = df["deg"].fillna(df["compound"].map(fallback))
    df["age_adj"] = df["lap_time_fc_s"] - df["deg"] * df["tyre_life"]

    free = df["is_clean"] & (df["gap_prev"] > FREE_AIR_GAP_S)
    base = df[free].groupby(["race", "driver", "stint"])["age_adj"].median().rename("free_base")
    df = df.join(base, on=["race", "driver", "stint"])
    df["free_pace"] = df["free_base"] + df["deg"] * df["tyre_life"]

    fp = df.set_index(["race", "driver", "lap"])["free_pace"]
    idx = pd.MultiIndex.from_arrays([df["race"], df["ahead_prev"].fillna(""), df["lap"]])
    df["free_pace_ahead"] = fp.reindex(idx).to_numpy()
    df["free_delta"] = df["free_pace_ahead"] - df["free_pace"]  # + = faster than car ahead
    df["dev"] = df["lap_time_fc_s"] - df["free_pace"]
    return df


def dirty_air(state: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Median dev by gap bin for the aero-only, held-up and pooled populations, with a
    race-cluster bootstrap 90% interval."""
    d = state[state["is_clean"] & state["dev"].notna() & state["gap_prev"].notna()]
    same = d["ahead_now"] == d["ahead_prev"]
    d = d[same | (d["gap_prev"] >= FREE_AIR_GAP_S)].copy()
    d["gap_bin"] = pd.cut(d["gap_prev"], GAP_BINS, right=False, labels=GAP_LABELS)
    free_air = d["gap_prev"] >= FREE_AIR_GAP_S
    groups = {
        "aero_only": d[(d["free_delta"] <= 0) | free_air],
        "held_up": d[(d["free_delta"] > 0) | free_air],
        "pooled": d,
    }
    rng = np.random.default_rng(SEED)
    rows = []
    for name, sub in groups.items():
        med = sub.groupby("gap_bin", observed=True)["dev"].median()
        n = sub.groupby("gap_bin", observed=True).size()
        races = sub["race"].unique()
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(races, len(races))
            s = pd.concat([sub[sub["race"] == r] for r in pick])
            boots.append(s.groupby("gap_bin", observed=True)["dev"].median())
        bt = pd.concat(boots, axis=1)
        for b in med.index:
            rows.append(
                {
                    "population": name, "gap_bin": str(b), "median_s": float(med[b]),
                    "q05_s": float(bt.loc[b].quantile(0.05)),
                    "q95_s": float(bt.loc[b].quantile(0.95)), "n_laps": int(n[b]),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    laps = load_train()
    slopes = pd.read_parquet(SLOPES_PATH)
    state = lap_state(laps, slopes)
    state.to_parquet(STATE_PATH, index=False)

    curve = dirty_air(state)
    curve.to_parquet(DIRTY_AIR_PATH, index=False)
    pd.set_option("display.width", 200)
    print(f"train races: {laps['race'].nunique()}  laps with free-air pace: "
          f"{state['free_pace'].notna().mean():.1%}")
    print("\nlap time minus own free-air pace (s), by gap to car ahead at end of previous lap:")
    print(curve.pivot(index="gap_bin", columns="population",
                      values=["median_s", "q05_s", "q95_s", "n_laps"])
          .reindex(GAP_LABELS).round(3).to_string())
    print(f"\nwrote {STATE_PATH} and {DIRTY_AIR_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
