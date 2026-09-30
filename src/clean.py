"""Build laps_clean.parquet from ingested raw race data.

Usage:
    python -m src.clean
    python -m src.clean --split train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.splits import load_splits, split_of

RAW_DIR = Path("data/processed/raw")
OUT_PATH = Path("data/processed/laps_clean.parquet")
OUTLIER_FACTOR = 1.07


def parse_track_status(series: pd.Series) -> pd.DataFrame:
    """TrackStatus is a concatenation of single-digit codes, e.g. '14' = green + SC.

    Codes: 1 green, 2 yellow, 4 SC, 5 red, 6 VSC, 7 VSC ending.
    """
    s = series.fillna("").astype(str)
    return pd.DataFrame(
        {
            "is_yellow": s.str.contains("2"),
            "is_sc": s.str.contains("4"),
            "is_red": s.str.contains("5"),
            "is_vsc": s.str.contains("6") | s.str.contains("7"),
        },
        index=series.index,
    )


def add_gap_ahead(df: pd.DataFrame) -> pd.DataFrame:
    """Gap to the car ahead, from session time at lap end, within the same lap number."""
    df = df.copy()
    df["gap_ahead_s"] = np.nan
    valid = df["session_time_s"].notna()
    sub = df.loc[valid].sort_values(["lap", "session_time_s"])
    gaps = sub.groupby("lap", sort=False)["session_time_s"].diff()
    df.loc[gaps.index, "gap_ahead_s"] = gaps.values
    return df


def merge_weather(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Nearest weather sample at or before each lap end."""
    laps = laps.copy()
    if weather is None or weather.empty or "Time" not in weather.columns:
        laps["track_temp"] = np.nan
        laps["air_temp"] = np.nan
        return laps

    w = weather[["Time", "TrackTemp", "AirTemp"]].dropna(subset=["Time"]).copy()
    w = w.rename(
        columns={"Time": "session_time_s", "TrackTemp": "track_temp", "AirTemp": "air_temp"}
    )
    w = w.sort_values("session_time_s")

    has_time = laps["session_time_s"].notna()
    left = laps.loc[has_time].sort_values("session_time_s")
    merged = pd.merge_asof(left, w, on="session_time_s", direction="nearest")
    merged.index = left.index

    laps["track_temp"] = np.nan
    laps["air_temp"] = np.nan
    laps.loc[merged.index, "track_temp"] = merged["track_temp"].values
    laps.loc[merged.index, "air_temp"] = merged["air_temp"].values
    return laps


def flag_outliers(df: pd.DataFrame) -> pd.Series:
    """More than 107% of that driver's stint median, using representative laps only."""
    eligible = (
        df["lap_time_s"].notna()
        & ~df["is_pit_in"]
        & ~df["is_pit_out"]
        & ~df["is_lap1"]
        & ~df["is_sc"]
        & ~df["is_vsc"]
        & ~df["is_red"]
    )
    med = (
        df.loc[eligible]
        .groupby(["driver", "stint"])["lap_time_s"]
        .median()
        .rename("stint_median_s")
    )
    joined = df.join(med, on=["driver", "stint"])
    return (
        joined["lap_time_s"].notna()
        & joined["stint_median_s"].notna()
        & (joined["lap_time_s"] > OUTLIER_FACTOR * joined["stint_median_s"])
    )


def clean_race(race_path: Path) -> pd.DataFrame:
    laps = pd.read_parquet(race_path / "laps.parquet")
    weather = pd.read_parquet(race_path / "weather.parquet")
    meta = pd.read_parquet(race_path / "meta.parquet").iloc[0]

    df = pd.DataFrame(
        {
            "season": laps["season"].astype(int),
            "round": laps["round"].astype(int),
            "event": laps["event"].astype(str),
            "driver": laps["Driver"].astype(str),
            "team": laps["Team"].astype(str),
            "lap": laps["LapNumber"].astype(int),
            "lap_time_s": laps["LapTime"].astype(float),
            "sector1_s": laps["Sector1Time"].astype(float),
            "sector2_s": laps["Sector2Time"].astype(float),
            "sector3_s": laps["Sector3Time"].astype(float),
            "compound": laps["Compound"].astype(str).str.upper(),
            "tyre_life": laps["TyreLife"].astype(float),
            "stint": laps["Stint"].astype(float),
            "fresh_tyre": laps["FreshTyre"].fillna(False).astype(bool),
            "position": laps["Position"].astype(float),
            "session_time_s": laps["Time"].astype(float),
            "pit_in_time_s": laps["PitInTime"].astype(float),
            "pit_out_time_s": laps["PitOutTime"].astype(float),
            "track_status": laps["TrackStatus"].astype(str),
        }
    )

    status = parse_track_status(laps["TrackStatus"])
    df = pd.concat([df, status], axis=1)

    df["is_pit_in"] = laps["PitInTime"].notna().values
    df["is_pit_out"] = laps["PitOutTime"].notna().values
    df["is_lap1"] = df["lap"] == 1
    df["is_deleted"] = laps["Deleted"].fillna(False).astype(bool).values
    df["is_accurate"] = laps["IsAccurate"].fillna(False).astype(bool).values

    df = df.sort_values(["driver", "lap"]).reset_index(drop=True)
    df = add_gap_ahead(df)
    df = merge_weather(df, weather)

    df["is_outlier"] = flag_outliers(df)
    df["is_clean"] = (
        df["lap_time_s"].notna()
        & ~df["is_pit_in"]
        & ~df["is_pit_out"]
        & ~df["is_lap1"]
        & ~df["is_sc"]
        & ~df["is_vsc"]
        & ~df["is_red"]
        & ~df["is_deleted"]
        & df["is_accurate"]
        & ~df["is_outlier"]
    )

    df["laps_remaining"] = int(meta["race_laps"]) - df["lap"]
    df["split"] = split_of(int(meta["season"]), str(meta["event"]))

    cols = [
        "season", "round", "event", "split", "driver", "team", "lap", "lap_time_s",
        "sector1_s", "sector2_s", "sector3_s", "compound", "tyre_life", "stint",
        "fresh_tyre", "position", "session_time_s", "pit_in_time_s", "pit_out_time_s",
        "gap_ahead_s", "track_status",
        "is_pit_in", "is_pit_out", "is_sc", "is_vsc", "is_yellow", "is_red",
        "is_lap1", "is_deleted", "is_accurate", "is_outlier", "is_clean",
        "track_temp", "air_temp", "laps_remaining",
    ]
    return df[cols]


def build(split: str | None = None) -> pd.DataFrame:
    frames = []
    splits = load_splits()
    for race_path in sorted(RAW_DIR.iterdir()):
        if not race_path.is_dir():
            continue
        meta = pd.read_parquet(race_path / "meta.parquet").iloc[0]
        if split and split_of(int(meta["season"]), str(meta["event"]), splits) != split:
            continue
        print(f"cleaning {race_path.name} ...", flush=True)
        frames.append(clean_race(race_path))
    if not frames:
        raise SystemExit("no races found, run python -m src.ingest first")
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean ingested laps.")
    parser.add_argument("--split", default=None, choices=["train", "holdout"])
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args(argv)

    df = build(args.split)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    n = len(df)
    print(f"\nrows: {n}")
    print(f"clean rows: {int(df['is_clean'].sum())} ({df['is_clean'].mean():.1%})")
    print(f"races: {df.groupby(['season', 'round']).ngroups}")
    # Within race: pooling lap times across tracks mixes circuits of different length, so the
    # pooled compound median depends on which tracks ran which compound. Still not a pace
    # check: no fuel correction, and HARD mostly runs late on low fuel (Session 2 separates it).
    per_race = (
        df[df["is_clean"]]
        .groupby(["season", "round", "compound"])["lap_time_s"]
        .median()
        .unstack("compound")
    )
    delta = per_race.sub(per_race["MEDIUM"], axis=0)
    print("\nclean lap vs same race MEDIUM, median over races (s, not fuel corrected):")
    print(delta.median().round(3).to_string())
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())