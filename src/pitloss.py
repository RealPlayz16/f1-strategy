"""Pit loss per stop and per track, measured from pit lane transit.

    pit_loss = transit - phi * L

transit  PitOutTime(out-lap) - PitInTime(in-lap). Measured directly.
phi      Fraction of a lap, in time, of the track section spanned by the transit window.
         Fitted per track from green stops, where L is the driver's own pace:
             in_lap + out_lap = (2 - phi) * L + transit
L        Lap time the car would have run had it stayed out, under the conditions in force.
         Green: the driver's clean pace either side of the stop. SC / VSC: the median lap of
         cars that did not pit on those laps, or when too few stayed out, the driver's green
         pace times the median SC / VSC ratio from stops where the reference was trusted.

An error in L moves the loss by phi * error, not 2 * error as in lap-time subtraction.

Usage:
    python -m src.pitloss
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LAPS_PATH = Path("data/processed/laps_clean.parquet")
STOPS_PATH = Path("data/processed/pit_stops.parquet")
BYTRACK_PATH = Path("data/processed/pitloss_by_track.parquet")
PHI_PATH = Path("data/processed/pit_phi.parquet")

REF_LAPS = 3         # clean laps each side of a stop used for the driver's pace
MIN_STAYOUT = 5      # cars not pitting on a lap for an SC / VSC stay-out median to be trusted
MIN_SEASON_FIT = 5   # green stops for a season's phi to count toward phi_season_spread

STOP_COLS = [
    "season", "round", "event", "driver", "lap", "compound_in", "compound_out", "condition",
    "transit_s", "ref_lap_s", "lap_sum_s", "pace_trusted", "phi", "pit_loss_s",
]


def classify_condition(in_row: pd.Series, out_row: pd.Series) -> str:
    if bool(in_row["is_sc"]) or bool(out_row["is_sc"]):
        return "sc"
    if bool(in_row["is_vsc"]) or bool(out_row["is_vsc"]):
        return "vsc"
    return "green"


def tyres_changed(in_row: pd.Series, out_row: pd.Series) -> bool:
    """False for a pit lane passage that kept the same set: a drive-through or stop-go penalty,
    or the field following the Safety Car through the pit lane (2023 Austria lap 2).
    FastF1 starts a new stint on any pit lane passage, so Stint alone does not show a stop."""
    same_set = out_row["tyre_life"] == in_row["tyre_life"] + 1
    return str(in_row["compound"]) != str(out_row["compound"]) or not bool(same_set)


def driver_pace(d: pd.DataFrame, in_row: pd.Series, out_row: pd.Series) -> tuple[float, bool]:
    """Driver's clean pace around a stop, from the old and the new stint.

    Mean of the median of the last REF_LAPS clean laps before the in-lap and the first
    REF_LAPS clean laps after the out-lap. Trusted only when both sides exist.
    """
    clean = d[d["is_clean"] & d["lap_time_s"].notna()]
    before = clean[(clean["lap"] < in_row["lap"]) & (clean["stint"] == in_row["stint"])]
    after = clean[(clean["lap"] > out_row["lap"]) & (clean["stint"] == out_row["stint"])]
    sides = [
        s.median()
        for s in (before["lap_time_s"].tail(REF_LAPS), after["lap_time_s"].head(REF_LAPS))
        if len(s)
    ]
    if not sides:
        return np.nan, False
    return float(np.mean(sides)), len(sides) == 2


def stayout_pace(race: pd.DataFrame, in_lap: int) -> tuple[float, bool]:
    """Median lap time of cars that did not pit, averaged over the in-lap and the out-lap."""
    running = race[~race["is_pit_in"] & ~race["is_pit_out"] & race["lap_time_s"].notna()]
    meds, trusted = [], True
    for lap in (in_lap, in_lap + 1):
        s = running.loc[running["lap"] == lap, "lap_time_s"]
        trusted &= len(s) >= MIN_STAYOUT
        meds.append(s.median() if len(s) else np.nan)
    return float(np.mean(meds)), trusted


def stops_for_race(race: pd.DataFrame) -> pd.DataFrame:
    """Every completed stop in one race, with transit, lap sum and stay-out reference.

    Skipped: pit lane passages without a tyre change and stops touching a red flag (transit
    includes the stoppage). lap_sum_s is NaN where FastF1 has no in-lap or out-lap time,
    common on SC laps; the stop still has a transit and is kept.
    """
    rows = []
    for driver, d in race.groupby("driver", sort=False):
        d = d.sort_values("lap")
        by_lap = d.set_index("lap")
        for _, in_row in d[d["is_pit_in"]].iterrows():
            in_lap = int(in_row["lap"])
            if in_lap + 1 not in by_lap.index:
                continue
            out_row = by_lap.loc[in_lap + 1].copy()
            out_row["lap"] = in_lap + 1
            if not bool(out_row["is_pit_out"]):
                continue
            if not tyres_changed(in_row, out_row):
                continue
            if bool(in_row["is_red"]) or bool(out_row["is_red"]):
                continue
            transit = float(out_row["pit_out_time_s"]) - float(in_row["pit_in_time_s"])
            if not np.isfinite(transit):
                continue
            lap_sum = float(in_row["lap_time_s"]) + float(out_row["lap_time_s"])

            condition = classify_condition(in_row, out_row)
            own, own_trusted = driver_pace(d, in_row, out_row)
            if condition == "green":
                ref, trusted = own, own_trusted
            else:
                ref, trusted = stayout_pace(race, in_lap)
                if not trusted:
                    ref = np.nan  # filled from the condition ratio in build_stops

            rows.append(
                {
                    "season": int(in_row["season"]),
                    "round": int(in_row["round"]),
                    "event": str(in_row["event"]),
                    "driver": driver,
                    "lap": in_lap,
                    "compound_in": str(in_row["compound"]),
                    "compound_out": str(out_row["compound"]),
                    "condition": condition,
                    "transit_s": transit,
                    "ref_lap_s": ref,
                    "lap_sum_s": lap_sum,
                    "pace_trusted": bool(trusted),
                    "driver_pace_s": own,
                }
            )
    return pd.DataFrame(rows)


def fill_untrusted_refs(stops: pd.DataFrame) -> pd.DataFrame:
    """SC / VSC stops with no stay-out reference: driver's green pace times the median
    SC / VSC ratio measured on stops where the reference was trusted."""
    stops = stops.copy()
    for cond in ("sc", "vsc"):
        is_cond = stops["condition"] == cond
        ok = is_cond & stops["pace_trusted"]
        ratio = (stops.loc[ok, "ref_lap_s"] / stops.loc[ok, "driver_pace_s"]).median()
        fill = is_cond & ~stops["pace_trusted"]
        stops.loc[fill, "ref_lap_s"] = stops.loc[fill, "driver_pace_s"] * ratio
    return stops


def fit_phi(stops: pd.DataFrame) -> pd.DataFrame:
    """Per-track phi from trusted green stops: phi = 2 - (lap_sum - transit) / L."""
    g = stops[
        (stops["condition"] == "green")
        & stops["pace_trusted"]
        & stops["ref_lap_s"].notna()
        & stops["lap_sum_s"].notna()
    ].copy()
    g["phi_stop"] = 2.0 - (g["lap_sum_s"] - g["transit_s"]) / g["ref_lap_s"]

    def agg(t: pd.DataFrame) -> pd.Series:
        seasons = t.groupby("season")["phi_stop"].agg(["median", "size"])
        seasons = seasons[seasons["size"] >= MIN_SEASON_FIT]["median"]
        spread = seasons.max() - seasons.min() if len(seasons) >= 2 else np.nan
        return pd.Series(
            {"phi": t["phi_stop"].median(), "n_fit": int(len(t)), "phi_season_spread": spread}
        )

    return g.groupby("event", sort=True).apply(agg, include_groups=False).reset_index()


def build_stops(laps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [stops_for_race(d) for _, d in laps.groupby(["season", "round"], sort=True)]
    frames = [f for f in frames if len(f)]
    if not frames:
        raise SystemExit("no pit stops found")
    stops = fill_untrusted_refs(pd.concat(frames, ignore_index=True))
    # In the pit lane longer than a car staying out takes to complete a lap: the car went to
    # the garage and came back (2023 Japan, PER lap 13, 41 minutes). Not a pit stop.
    stops = stops[~(stops["transit_s"] > stops["ref_lap_s"])].reset_index(drop=True)
    phi = fit_phi(stops)
    stops = stops.merge(phi[["event", "phi"]], on="event", how="left")
    stops["pit_loss_s"] = stops["transit_s"] - stops["phi"] * stops["ref_lap_s"]
    return stops[STOP_COLS], phi


def by_track(stops: pd.DataFrame) -> pd.DataFrame:
    def agg(g: pd.DataFrame) -> pd.Series:
        cond = g["condition"]
        return pd.Series(
            {
                "green_s": g.loc[cond == "green", "pit_loss_s"].median(),
                "sc_s": g.loc[cond == "sc", "pit_loss_s"].median(),
                "vsc_s": g.loc[cond == "vsc", "pit_loss_s"].median(),
                "transit_s": g.loc[cond == "green", "transit_s"].median(),
                "phi": g["phi"].iloc[0],
                "n_stops": int(len(g)),
                "n_green": int((cond == "green").sum()),
                "n_sc": int((cond == "sc").sum()),
                "n_vsc": int((cond == "vsc").sum()),
            }
        )

    return stops.groupby("event", sort=True).apply(agg, include_groups=False).reset_index()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute pit loss from pit lane transit.")
    parser.add_argument("--laps", default=str(LAPS_PATH))
    args = parser.parse_args(argv)

    laps = pd.read_parquet(args.laps)
    stops, phi = build_stops(laps)
    tracks = by_track(stops)

    stops.to_parquet(STOPS_PATH, index=False)
    phi.to_parquet(PHI_PATH, index=False)
    tracks.to_parquet(BYTRACK_PATH, index=False)

    print(f"stops: {len(stops)}  (trusted reference: {int(stops['pace_trusted'].sum())})")
    print("\nmedian by condition:")
    cols = ["transit_s", "ref_lap_s", "pit_loss_s"]
    print(stops.groupby("condition")[cols].median().round(2).to_string())
    print("\nphi by track:")
    print(phi.round(3).to_string(index=False))
    print("\npit loss by track:")
    print(tracks.round(2).to_string(index=False))
    print(f"\nwrote {STOPS_PATH}, {PHI_PATH}, {BYTRACK_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
