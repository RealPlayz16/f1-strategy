"""Session 1 sanity plots, written to figures/ (gitignored). Train races only.

1. figures/laptime_vs_tyre_age.png: median clean lap minus that race's median clean lap, by
   tyre age and compound. Fuel-adjusted at FUEL_S_PER_LAP per lap remaining, an assumption
   close to the Session 1 regression estimate (0.051); Session 2 fits fuel properly. Without
   it, fuel burn hides tyre degradation.
2. figures/pitloss_by_race.png: measured green pit loss and SC pit loss at PHI, with the
   PHI_RANGE sweep as a bar.

Usage:
    python -m src.plots
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.pitloss import PHI_RANGE  # noqa: E402

LAPS_PATH = Path("data/processed/laps_clean.parquet")
BYTRACK_PATH = Path("data/processed/pitloss_by_track.parquet")
FIG_DIR = Path("figures")

FUEL_S_PER_LAP = 0.05
MIN_LAPS_PER_BIN = 30
MAX_TYRE_AGE = 40

# Reference categorical palette, light mode, slots 1-3 (validated all-pairs)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def laptime_vs_age(laps: pd.DataFrame, out: Path) -> None:
    c = laps[laps["is_clean"] & (laps["split"] == "train")].copy()
    race_med = c.groupby(["season", "round"])["lap_time_s"].transform("median")
    c["delta"] = c["lap_time_s"] - race_med - FUEL_S_PER_LAP * c["laps_remaining"]
    c["delta"] -= c.groupby(["season", "round"])["delta"].transform("median")
    c = c[c["tyre_life"] <= MAX_TYRE_AGE]

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150, facecolor=SURFACE)
    style(ax)
    for color, comp in zip(SERIES, ["SOFT", "MEDIUM", "HARD"], strict=True):
        g = c[c["compound"] == comp].groupby("tyre_life")["delta"].agg(["median", "size"])
        g = g[g["size"] >= MIN_LAPS_PER_BIN]
        ax.plot(g.index, g["median"], color=color, linewidth=2, label=comp.title())
        ax.annotate(comp.title(), (g.index[-1], g["median"].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=TEXT, fontsize=9)
    ax.set_xlabel("Tyre age (laps)", color=TEXT_2)
    ax.set_ylabel("Lap time vs race median (s)", color=TEXT_2)
    ax.set_title(
        "Clean laps get slower with tyre age",
        loc="left", color=TEXT, fontsize=12, fontweight="bold", pad=22,
    )
    ax.text(0, 1.02, f"Train races, fuel-adjusted at {FUEL_S_PER_LAP} s per lap remaining "
            f"(assumed), bins with {MIN_LAPS_PER_BIN}+ laps",
            transform=ax.transAxes, color=TEXT_2, fontsize=8.5)
    ax.legend(frameon=False, labelcolor=TEXT, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def pitloss_by_race(tracks: pd.DataFrame, out: Path) -> None:
    t = tracks.sort_values("green_s").reset_index(drop=True)
    labels = [f"{e.replace(' Grand Prix', '')} {s}" for s, e in zip(t["season"], t["event"],
                                                                   strict=True)]
    lo, hi = PHI_RANGE
    sc_at = {p: t["green_s"] - p * (t["lap_sc_s"] - t["lap_green_s"]) for p in (lo, hi)}
    y = range(len(t))

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=150, facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.hlines(y, sc_at[hi], sc_at[lo], color=SERIES[1], linewidth=2, alpha=0.45,
              label=f"SC, phi {lo} to {hi}")
    ax.plot(t["sc_s"], y, "o", color=SERIES[1], markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label=f"SC, phi {t['phi'].iloc[0]} (stated)")
    ax.plot(t["green_s"], y, "o", color=SERIES[0], markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label="Green (measured)")
    ax.set_yticks(list(y), labels, color=TEXT, fontsize=8.5)
    ax.set_xlabel("Pit loss (s)", color=TEXT_2)
    ax.set_title("Pit loss by race: green measured, SC from stated phi",
                 loc="left", color=TEXT, fontsize=12, fontweight="bold")
    ax.legend(frameon=False, labelcolor=TEXT, loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    FIG_DIR.mkdir(exist_ok=True)
    laps = pd.read_parquet(LAPS_PATH)
    tracks = pd.read_parquet(BYTRACK_PATH)
    laptime_vs_age(laps, FIG_DIR / "laptime_vs_tyre_age.png")
    pitloss_by_race(tracks, FIG_DIR / "pitloss_by_race.png")
    print(f"wrote {FIG_DIR / 'laptime_vs_tyre_age.png'} and {FIG_DIR / 'pitloss_by_race.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
