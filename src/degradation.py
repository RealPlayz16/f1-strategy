"""Within-race tyre degradation per compound, and the early-soft check. Train races only.

Everything is estimated within a (driver, stint): the stint intercept absorbs pace, car and
compound level, so only the shape with tyre age is left. Never pooled across races.

Degradation slope per (race, compound), two estimates:
  A  within-stint slope of lap_time_fc_s on tyre_life (pooled race-progress correction).
     Any error in K / race_laps for that race moves this slope one for one.
  B  joint per-race fit: race x driver intercepts, compound offsets and slopes, and the race's
     own progress slope, identified across stints (the fuel.py structure, one race at a time).
Curves: tyre_life bins as dummies within stint, relative to REF_BIN, per (race, compound);
then the median across races.

Early-soft check (Session 1 flagged soft laps at tyre age 2 to 4 above trend): within-stint
linear trend fitted on ages 5 to 15, mean residual at ages 2 to 4. A linear progress
correction cannot change curvature inside a stint, so the questions are whether the bump is
in stints that start the race or in all stints, and whether gap_ahead_s explains it.

Usage:
    python -m src.degradation
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.fuel import design  # noqa: E402
from src.splits import add_split  # noqa: E402

LAPS_PATH = Path("data/processed/laps_fuel_corrected.parquet")
SLOPES_PATH = Path("data/processed/degradation_slopes.parquet")
CURVES_PATH = Path("data/processed/degradation_curves.parquet")
FIG_PATH = Path("figures/degradation_within_race.png")

BINS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 41]  # left edges; last bin 32-40
REF_BIN = 5
MIN_LAPS_BIN = 8
MIN_RACES = 3
EARLY = (2, 4)
TREND = (5, 15)
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
SERIES = {"SOFT": "#2a78d6", "MEDIUM": "#eb6834", "HARD": "#1baf7a"}


def load() -> pd.DataFrame:
    laps = add_split(pd.read_parquet(LAPS_PATH).drop(columns=["split"], errors="ignore"))
    c = laps[laps["is_clean"] & (laps["split"] == "train")].copy()
    c["race"] = c["season"].astype(str) + "_" + c["round"].astype(str).str.zfill(2)
    c["ds"] = c["race"] + "_" + c["driver"] + "_" + c["stint"].astype(str)
    c["frac_remaining"] = c["laps_remaining"] / c["race_laps"]
    return c.reset_index(drop=True)


def within(df: pd.DataFrame, cols: list[str], group: str = "ds") -> pd.DataFrame:
    return df[cols] - df.groupby(group)[cols].transform("mean")


def ols(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    keep = x.columns[x.abs().sum() > 1e-9]
    beta, *_ = np.linalg.lstsq(x[keep].to_numpy(), y.to_numpy(), rcond=None)
    return pd.Series(beta, index=keep)


def slopes(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for race, g in c.groupby("race"):
        g = g.reset_index(drop=True)
        xw, yw = design(g, "laps_remaining")
        joint = ols(xw, yw)
        for comp, h in g.groupby("compound"):
            if len(h) < 30:
                continue
            w = within(h, ["tyre_life", "lap_time_fc_s"])
            a = (w["tyre_life"] * w["lap_time_fc_s"]).sum() / (w["tyre_life"] ** 2).sum()
            key = [k for k in joint.index if k.startswith("slope_") and k.endswith(comp)]
            rows.append(
                {
                    "race": race, "event": h["event"].iloc[0], "compound": comp,
                    "n_laps": len(h), "slope_A": float(a),
                    "slope_B": float(joint[key[0]]) if key else np.nan,
                    "race_progress_per_lap": float(joint.get("progress", np.nan)),
                }
            )
    return pd.DataFrame(rows)


def curves(c: pd.DataFrame) -> pd.DataFrame:
    c = c[c["tyre_life"] < BINS[-1]].copy()
    c["bin"] = pd.cut(c["tyre_life"], BINS, right=False, labels=BINS[:-1]).astype(int)
    rows = []
    for (race, comp), g in c.groupby(["race", "compound"]):
        counts = g["bin"].value_counts()
        g = g[g["bin"].isin(counts[counts >= MIN_LAPS_BIN].index)]
        if REF_BIN not in set(g["bin"]) or g["bin"].nunique() < 2:
            continue
        dummies = pd.get_dummies(g["bin"], prefix="b").astype(float)
        dummies = dummies.drop(columns=f"b_{REF_BIN}")
        g = pd.concat([g, dummies], axis=1)
        cols = list(dummies.columns)
        w = within(g, cols + ["lap_time_fc_s"])
        beta = ols(w[cols], w["lap_time_fc_s"])
        rows.append({"race": race, "compound": comp, "bin": REF_BIN, "delta_s": 0.0})
        rows += [
            {"race": race, "compound": comp, "bin": int(k[2:]), "delta_s": float(v)}
            for k, v in beta.items()
        ]
    per_race = pd.DataFrame(rows)
    agg = per_race.groupby(["compound", "bin"])["delta_s"].agg(
        median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75),
        n_races="size",
    ).reset_index()
    return agg[agg["n_races"] >= MIN_RACES]


def early_bump(c: pd.DataFrame, compound: str, traffic: bool) -> dict:
    """Mean residual at ages EARLY vs the within-stint linear trend on ages TREND."""
    g = c[(c["compound"] == compound) & c["tyre_life"].between(EARLY[0], TREND[1])].copy()
    g["early"] = g["tyre_life"].between(*EARLY).astype(float)
    g["age_trend"] = g["tyre_life"] * (1 - g["early"])  # slope fitted on trend ages only
    g["gap_lt1"] = (g["gap_ahead_s"] < 1.0).astype(float)
    g["gap_1_2"] = g["gap_ahead_s"].between(1.0, 2.0).astype(float)
    out = {}
    for label, sub in (("race start stints", g[g["stint"] == 1]),
                       ("later stints", g[g["stint"] > 1])):
        sub = sub[sub.groupby("ds")["early"].transform("max") > 0]  # stint has early laps
        sub = sub[sub.groupby("ds")["early"].transform("min") == 0]  # and trend laps
        cols = ["early", "age_trend"] + (["gap_lt1", "gap_1_2"] if traffic else [])
        if sub["ds"].nunique() < 5:
            out[label] = {"bump_s": np.nan, "n_stints": int(sub["ds"].nunique())}
            continue
        w = within(sub, cols + ["lap_time_fc_s"])
        beta = ols(w[cols], w["lap_time_fc_s"])
        out[label] = {
            "bump_s": float(beta.get("early", np.nan) + 0.0),
            "n_stints": int(sub["ds"].nunique()),
            "share_gap_lt1_early": float(sub.loc[sub["early"] == 1, "gap_lt1"].mean()),
            "share_gap_lt1_trend": float(sub.loc[sub["early"] == 0, "gap_lt1"].mean()),
            **({"gap_lt1_s": float(beta.get("gap_lt1", np.nan))} if traffic else {}),
        }
    return out


def plot(cur: pd.DataFrame) -> None:
    FIG_PATH.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150, facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.tick_params(colors="#52514e", length=0)
    for comp in COMPOUNDS:
        g = cur[cur["compound"] == comp].sort_values("bin")
        x = g["bin"] + 1  # bin centre
        ax.fill_between(x, g["q25"], g["q75"], color=SERIES[comp], alpha=0.12, linewidth=0)
        ax.plot(x, g["median"], color=SERIES[comp], linewidth=2, label=comp.title())
        ax.annotate(comp.title(), (x.iloc[-1], g["median"].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color="#0b0b0b")
    ax.set_xlabel("Tyre age (laps, 3-lap bins)", color="#52514e")
    ax.set_ylabel(f"Lap time vs age {REF_BIN}-{REF_BIN + 2} in same stint (s)", color="#52514e")
    ax.set_title("Degradation within stint, race-progress corrected", loc="left",
                 fontsize=12, fontweight="bold", color="#0b0b0b", pad=22)
    ax.text(0, 1.02, f"Median across train races, band = middle half of races, bins with "
            f"{MIN_RACES}+ races", transform=ax.transAxes, fontsize=8.5, color="#52514e")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_PATH, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> int:
    c = load()
    pd.set_option("display.width", 200)
    print(f"train races: {c['race'].nunique()}  clean laps: {len(c)}")

    sl = slopes(c)
    sl.to_parquet(SLOPES_PATH, index=False)
    print("\ndegradation slope per (race, compound), s per lap of tyre age:")
    summ = sl.groupby("compound")[["slope_A", "slope_B"]].describe(percentiles=[0.25, 0.5, 0.75])
    print(summ.loc[:, (slice(None), ["count", "25%", "50%", "75%"])].round(4).to_string())
    print(f"corr(slope_A, slope_B) across race x compound: "
          f"{sl[['slope_A', 'slope_B']].corr().iloc[0, 1]:.3f}")
    print(f"median |A - B|: {(sl['slope_A'] - sl['slope_B']).abs().median():.4f}")

    cur = curves(c)
    cur.to_parquet(CURVES_PATH, index=False)
    print("\nwithin-stint curve, median across races (s vs age 5-7):")
    print(cur.pivot(index="bin", columns="compound", values="median").round(3).to_string())
    plot(cur)

    print("\nearly-age bump: mean lap time at ages 2-4 above the within-stint trend on 5-15")
    for comp in COMPOUNDS:
        for traffic in (False, True):
            res = early_bump(c, comp, traffic)
            tag = "with gap_ahead controls" if traffic else "no traffic control"
            for label, r in res.items():
                print(f"  {comp:6s} {label:18s} {tag:24s} " + "  ".join(
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()
                ))
    print(f"\nwrote {SLOPES_PATH}, {CURVES_PATH}, {FIG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
