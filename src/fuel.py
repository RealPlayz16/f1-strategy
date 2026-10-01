"""Race-progress correction: fuel burn plus track evolution, per share of race distance.

    lap_time_fc_s = lap_time_s - K * laps_remaining / race_laps

K is the lap time cost of a full race's worth of remaining progress (about 3.3 s). It is NOT a
fuel coefficient. Fuel burn and track evolution (rubber, the race-long temperature trend) both
change with race lap, so lap timing cannot separate them. For lap time prediction the combined
effect is what we want; call it race progress.

Model, fitted on train clean laps (holdout never touches it):
    lap_time_s = a[race, driver] + c[race, compound] + d[race, compound] * tyre_life
                 + K * laps_remaining / race_laps
K is identified across stints: the same tyre age falls at different race progress in a
driver's first and later stints, and at different laps for drivers on different strategies.
FIT_EXCLUDE races stay in every other use (tyre model included); they are dropped from the K
fit only. 2024 Saudi: SC on lap 7 and almost everyone ran one stint after it, so there is no
between-stint variation (per-race estimate 0.118 s/lap, noise).

Rejected spec, do not reintroduce: an intercept per (race, driver, stint, compound) with a
laps_remaining slope. Within a stint tyre_life rises by one as laps_remaining falls by one
(correlation exactly -1), so the slope is fuel minus degradation. It gave 0.0043 s/lap and
-0.003 / 0.007 / 0.007 by season.

Checks, re-run on every fit (the race set changes the evidence):
- Gate: implied per-lap effect K / race_laps at the median train race in GATE_PER_LAP
  (physical fuel burn is about 0.03 to 0.06 s per lap). Fails -> no output written.
- Scaling: fuel burned per lap scales with lap length, so a fuel-dominated effect scales with
  1 / race_laps (race distance is fixed at about 305 km). Per-race per-lap slopes must
  correlate positively with 1 / race_laps (0.58 in Session 2). This is evidence the effect is
  mostly distance driven, consistent with fuel dominating. It does not show track evolution
  is zero.
- K by season (regulation driven, should be stable) and the free-air sensitivity (laps with
  gap_ahead_s > 2 s; traffic is more common early and inflates the slope slightly).

Usage:
    python -m src.fuel
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.splits import add_split

LAPS_PATH = Path("data/processed/laps_clean.parquet")
OUT_PATH = Path("data/processed/laps_fuel_corrected.parquet")
FIT_PATH = Path("data/processed/fuel_fit.json")

FIT_EXCLUDE = {(2024, "Saudi Arabian Grand Prix")}
GATE_PER_LAP = (0.03, 0.06)
FREE_AIR_GAP_S = 2.0


def load_train(laps: pd.DataFrame) -> pd.DataFrame:
    """Clean train laps, split re-derived from config/races.yaml."""
    laps = add_split(laps.drop(columns=["split"], errors="ignore"))
    c = laps[laps["is_clean"] & (laps["split"] == "train")].copy()
    c["race_laps"] = c["laps_remaining"] + c["lap"]
    c["race_laps"] = c.groupby(["season", "round"])["race_laps"].transform("max")
    return c.reset_index(drop=True)


def design(c: pd.DataFrame, progress: str) -> tuple[pd.DataFrame, pd.Series]:
    """Within race x driver transformed design: progress term, race x compound offset and
    race x compound tyre_life slope."""
    rc = c["season"].astype(str) + "_" + c["round"].astype(str) + "_" + c["compound"]
    rd = c["season"].astype(str) + "_" + c["round"].astype(str) + "_" + c["driver"]
    cols = {"progress": c[progress].astype(float)}
    for key, idx in rc.groupby(rc).groups.items():
        ind = pd.Series(0.0, index=c.index)
        ind[idx] = 1.0
        cols[f"off_{key}"] = ind
        cols[f"slope_{key}"] = ind * c["tyre_life"]
    x = pd.DataFrame(cols)
    xw = x - x.groupby(rd).transform("mean")
    yw = c["lap_time_s"] - c.groupby(rd)["lap_time_s"].transform("mean")
    return xw.loc[:, xw.abs().sum() > 1e-9], yw


def fit_progress(c: pd.DataFrame, progress: str = "frac_remaining") -> float:
    xw, yw = design(c.reset_index(drop=True), progress)
    beta, *_ = np.linalg.lstsq(xw.to_numpy(), yw.to_numpy(), rcond=None)
    return float(beta[0])


def scaling_check(c: pd.DataFrame) -> pd.DataFrame:
    """Per-race per-lap slope on laps_remaining, for the 1 / race_laps scaling test."""
    rows = []
    for (season, _rnd), g in c.groupby(["season", "round"]):
        rows.append(
            {
                "season": season,
                "event": g["event"].iloc[0],
                "race_laps": int(g["race_laps"].iloc[0]),
                "per_lap_s": fit_progress(g, "laps_remaining"),
            }
        )
    return pd.DataFrame(rows)


def fit(laps: pd.DataFrame) -> dict:
    c = load_train(laps)
    c["frac_remaining"] = c["laps_remaining"] / c["race_laps"]
    keep = ~pd.Series(list(zip(c["season"], c["event"], strict=True))).isin(FIT_EXCLUDE)
    fit_set = c[keep.to_numpy()]

    k = fit_progress(fit_set)
    by_season = {int(s): fit_progress(g) for s, g in fit_set.groupby("season")}
    free = fit_set[fit_set["gap_ahead_s"].isna() | (fit_set["gap_ahead_s"] > FREE_AIR_GAP_S)]
    per_race = scaling_check(c)
    in_fit = ~pd.Series(list(zip(per_race["season"], per_race["event"], strict=True))).isin(
        FIT_EXCLUDE
    )
    scaling_r = float(
        np.corrcoef(per_race.loc[in_fit.to_numpy(), "per_lap_s"],
                    1 / per_race.loc[in_fit.to_numpy(), "race_laps"])[0, 1]
    )
    median_laps = float(fit_set.groupby(["season", "round"])["race_laps"].first().median())
    return {
        "K_s": k,
        "K_by_season_s": by_season,
        "K_free_air_s": fit_progress(free),
        "per_lap_free_air_s": fit_progress(free, "laps_remaining"),
        "per_lap_constant_s": fit_progress(fit_set, "laps_remaining"),
        "median_race_laps": median_laps,
        "per_lap_at_median_race_s": k / median_laps,
        "scaling_corr": scaling_r,
        "n_laps": int(len(fit_set)),
        "n_races": int(fit_set.groupby(["season", "round"]).ngroups),
        "fit_exclude": sorted(f"{s} {e}" for s, e in FIT_EXCLUDE),
        "per_race": per_race.round(4).to_dict(orient="records"),
    }


def gate(result: dict) -> list[str]:
    errors = []
    lo, hi = GATE_PER_LAP
    if not lo <= result["per_lap_at_median_race_s"] <= hi:
        errors.append(
            f"per-lap effect {result['per_lap_at_median_race_s']:.4f} outside {GATE_PER_LAP}"
        )
    if not result["scaling_corr"] > 0:
        errors.append(f"per-race slopes do not scale with 1/race_laps: r={result['scaling_corr']}")
    return errors


def correct(laps: pd.DataFrame, k: float) -> pd.DataFrame:
    """All laps, train and holdout. race_laps is known before the race."""
    out = laps.copy()
    race_laps = (out["laps_remaining"] + out["lap"]).groupby(
        [out["season"], out["round"]]
    ).transform("max")
    out["race_laps"] = race_laps
    out["progress_s"] = k * out["laps_remaining"] / race_laps
    out["lap_time_fc_s"] = out["lap_time_s"] - out["progress_s"]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit the race-progress (fuel + track) effect.")
    parser.add_argument("--laps", default=str(LAPS_PATH))
    args = parser.parse_args(argv)

    laps = pd.read_parquet(args.laps)
    result = fit(laps)

    pr = pd.DataFrame(result["per_race"]).sort_values("race_laps")
    pd.set_option("display.width", 200)
    print(f"fit: {result['n_races']} train races, {result['n_laps']} clean laps, "
          f"excluded from K fit: {result['fit_exclude']}")
    print(f"K = {result['K_s']:.3f} s per race of progress (fuel + track evolution)")
    print("K by season:", {s: round(v, 3) for s, v in result["K_by_season_s"].items()})
    print(f"per lap at median race ({result['median_race_laps']:.0f} laps): "
          f"{result['per_lap_at_median_race_s']:.4f} s")
    print(f"constant per-lap form: {result['per_lap_constant_s']:.4f} s/lap, free air "
          f"{result['per_lap_free_air_s']:.4f}; per-distance free air K "
          f"{result['K_free_air_s']:.3f}")
    print(f"scaling check: corr(per-race slope, 1/race_laps) = {result['scaling_corr']:.3f}")
    print(f"per-race per-lap slope: median {pr['per_lap_s'].median():.4f}, "
          f"range {pr['per_lap_s'].min():.4f} to {pr['per_lap_s'].max():.4f}")
    print(pr.to_string(index=False))

    errors = gate(result)
    if errors:
        print("\nGATE FAILED, nothing written:\n  " + "\n  ".join(errors))
        return 1

    correct(laps, result["K_s"]).to_parquet(OUT_PATH, index=False)
    FIT_PATH.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(f"\ngate passed. wrote {OUT_PATH} and {FIT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
