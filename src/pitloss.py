"""Pit loss per stop and per (season, event). Train races only.

Green loss is measured:

    green_loss = in_lap + out_lap - 2 * L_green

with L_green the driver's own clean pace either side of the stop. SC and VSC loss use a
stated phi, not a fitted one:

    loss_cond = green_loss - PHI * (L_cond - L_green)

phi is the fraction of a lap, in time, of the track section the pit lane bypasses. Under SC
or VSC a car staying out takes longer over that section, so the stop costs less.
PHI = 0.08 central; the Monte Carlo sweeps PHI_RANGE. L_cond = L_green * r_cond, where
r_cond is the median ratio of lap time on laps run entirely under that condition to green
pace (train races).

Why phi is not fitted (Session 1, see HANDOFF.md):
- From pit lane transit: FastF1 PitInTime fires inside the pit lane, about 1.8 s before the
  timing line, not at the pit entry line. The [PitInTime, PitOutTime] window covers 0 to 3 s
  of track and phi fitted on it is about 0 at 12 of 13 tracks.
- From the measured SC discount: SC pit loss is not identifiable from lap times with this
  sample. On laps where the SC or VSC is deployed or ends, stay-out lap times spread 22 to
  31 s across the field, with rank correlation about 0.97 with running position: a car's lap
  time depends on where it was when the flag changed, so the stay-out median is not any
  one car's counterfactual. Implied phi ran 0.035 to 1.459 across tracks. Only 3 train stops
  have both laps fully under SC with lap times and a stay-out sample.
Session 4 validates PHI against observed position loss when stopping under SC, not lap-time
seconds.

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

PHI = 0.08
PHI_RANGE = (0.05, 0.12)
REF_LAPS = 3         # clean laps each side of a stop used for the driver's pace
MIN_STAYOUT = 5      # cars not pitting on a lap for an SC / VSC stay-out median
FULL_STATUS = {"sc": "4", "vsc": "6"}  # track status of a lap run entirely under the condition

STOP_COLS = [
    "season", "round", "event", "driver", "lap", "compound_in", "compound_out", "condition",
    "transit_s", "lap_sum_s", "driver_pace_s", "ref_lap_s", "pace_trusted",
    "measured_loss_s", "phi", "pit_loss_s",
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


def stayout_pace(race: pd.DataFrame, in_lap: int) -> float:
    """Median lap of cars not pitting, over the in-lap and the out-lap. NaN when fewer than
    MIN_STAYOUT stayed out on either lap. Diagnostic only: confounded by running position."""
    running = race[~race["is_pit_in"] & ~race["is_pit_out"] & race["lap_time_s"].notna()]
    meds = []
    for lap in (in_lap, in_lap + 1):
        s = running.loc[running["lap"] == lap, "lap_time_s"]
        if len(s) < MIN_STAYOUT:
            return np.nan
        meds.append(s.median())
    return float(np.mean(meds))


def stops_for_race(race: pd.DataFrame) -> pd.DataFrame:
    """Every completed stop in one race.

    Skipped: pit lane passages without a tyre change, stops touching a red flag (transit
    includes the stoppage) and garage visits (transit longer than a green lap, e.g.
    2023 Japan PER lap 13, 41 minutes). lap_sum_s is NaN where FastF1 has no in-lap or
    out-lap time, common on SC laps; the stop is kept.
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
            pace, trusted = driver_pace(d, in_row, out_row)
            if transit > pace:
                continue

            condition = classify_condition(in_row, out_row)
            lap_sum = float(in_row["lap_time_s"]) + float(out_row["lap_time_s"])
            ref = pace if condition == "green" else stayout_pace(race, in_lap)
            measured = lap_sum - 2 * pace if condition == "green" and trusted else np.nan
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
                    "lap_sum_s": lap_sum,
                    "driver_pace_s": pace,
                    "ref_lap_s": ref,
                    "pace_trusted": bool(trusted),
                    "measured_loss_s": measured,
                }
            )
    return pd.DataFrame(rows)


def condition_ratios(laps: pd.DataFrame) -> dict[str, float]:
    """Median over races of (stay-out lap on laps entirely under the condition) / (median clean
    lap of that race). Full-condition laps avoid the deploy / ending position confound."""
    out = {}
    running = laps[~laps["is_pit_in"] & ~laps["is_pit_out"] & laps["lap_time_s"].notna()]
    green = laps[laps["is_clean"]].groupby(["season", "round"])["lap_time_s"].median()
    for cond, status in FULL_STATUS.items():
        full = running[running["track_status"] == status]
        ratio = full.groupby(["season", "round"])["lap_time_s"].median() / green
        out[cond] = float(ratio.dropna().median())
    return out


def build_stops(laps: pd.DataFrame) -> pd.DataFrame:
    frames = [stops_for_race(d) for _, d in laps.groupby(["season", "round"], sort=True)]
    frames = [f for f in frames if len(f)]
    if not frames:
        raise SystemExit("no pit stops found")
    return pd.concat(frames, ignore_index=True)


def by_track(stops: pd.DataFrame, ratios: dict[str, float], phi: float = PHI) -> pd.DataFrame:
    """Per (season, event): measured green loss, and SC / VSC loss from phi."""

    def agg(g: pd.DataFrame) -> pd.Series:
        cond = g["condition"]
        green = g[(cond == "green") & g["measured_loss_s"].notna()]
        lap_green = green["driver_pace_s"].median()
        row = {
            "green_s": green["measured_loss_s"].median(),
            "transit_s": green["transit_s"].median(),
            "lap_green_s": lap_green,
        }
        for c in ("sc", "vsc"):
            row[f"lap_{c}_s"] = lap_green * ratios[c]
        for c in ("sc", "vsc"):
            row[f"{c}_s"] = row["green_s"] - phi * (row[f"lap_{c}_s"] - lap_green)
        row.update(
            {
                "phi": phi,
                "n_stops": int(len(g)),
                "n_green": int(len(green)),
                "n_sc": int((cond == "sc").sum()),
                "n_vsc": int((cond == "vsc").sum()),
            }
        )
        return pd.Series(row)

    out = stops.groupby(["season", "event"], sort=True).apply(agg, include_groups=False)
    return out.reset_index()


def apply_losses(stops: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    """Green stops keep their measured loss; SC / VSC stops take the track value."""
    stops = stops.merge(
        tracks[["season", "event", "phi", "sc_s", "vsc_s"]], on=["season", "event"], how="left"
    )
    stops["pit_loss_s"] = np.select(
        [stops["condition"] == "sc", stops["condition"] == "vsc"],
        [stops["sc_s"], stops["vsc_s"]],
        stops["measured_loss_s"],
    )
    return stops[STOP_COLS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute pit loss (train races only).")
    parser.add_argument("--laps", default=str(LAPS_PATH))
    parser.add_argument("--phi", type=float, default=PHI)
    args = parser.parse_args(argv)

    laps = pd.read_parquet(args.laps)
    laps = laps[laps["split"] == "train"]
    ratios = condition_ratios(laps)
    stops = build_stops(laps)
    tracks = by_track(stops, ratios, args.phi)
    stops = apply_losses(stops, tracks)

    stops.to_parquet(STOPS_PATH, index=False)
    tracks.to_parquet(BYTRACK_PATH, index=False)

    print(f"train races: {laps.groupby(['season', 'round']).ngroups}  stops: {len(stops)}")
    print(f"pace ratio on full-condition laps: SC {ratios['sc']:.3f}  VSC {ratios['vsc']:.3f}")
    print(f"phi {args.phi} (Monte Carlo range {PHI_RANGE[0]} to {PHI_RANGE[1]})")
    print("\npit loss by (season, event):")
    cols = ["season", "event", "green_s", "sc_s", "vsc_s", "transit_s", "lap_green_s", "n_green"]
    print(tracks[cols].round(2).to_string(index=False))
    print(f"\nwrote {STOPS_PATH} and {BYTRACK_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
