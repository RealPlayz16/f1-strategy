"""Load FastF1 race sessions and persist raw laps, weather and race control messages.

Usage:
    python -m src.ingest --set train
    python -m src.ingest --set all --force
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import fastf1
import pandas as pd
import yaml

RAW_DIR = Path("data/processed/raw")
CACHE_DIR = Path("data/cache")
CONFIG = Path("config/races.yaml")
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}

fastf1.set_log_level("WARNING")


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def td_to_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """Convert every timedelta column to float seconds, keeping column names."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_timedelta64_dtype(out[col]):
            out[col] = out[col].dt.total_seconds()
    return out


def load_config(path: Path = CONFIG) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def race_dir(season: int, rnd: int, event: str) -> Path:
    return RAW_DIR / f"{season}_{rnd:02d}_{slugify(event)}"


def ingest_race(season: int, event: str, split: str, force: bool = False) -> dict:
    """Ingest one race. Returns a manifest row."""
    row = {
        "season": season,
        "event_query": event,
        "split": split,
        "event": None,
        "round": None,
        "n_laps": 0,
        "compounds": "",
        "status": "",
        "reason": "",
    }

    try:
        session = fastf1.get_session(season, event, "R")
        session.load(laps=True, weather=True, messages=True, telemetry=False)
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["reason"] = f"{type(exc).__name__}: {exc}"
        return row

    event_name = str(session.event["EventName"])
    rnd = int(session.event["RoundNumber"])
    row["event"] = event_name
    row["round"] = rnd

    laps = session.laps
    if laps is None or len(laps) == 0:
        row["status"] = "excluded"
        row["reason"] = "no lap data"
        return row

    compounds = set(laps["Compound"].dropna().str.upper().unique())
    row["compounds"] = "|".join(sorted(compounds))
    row["n_laps"] = int(len(laps))

    if compounds & WET_COMPOUNDS:
        row["status"] = "excluded"
        row["reason"] = f"wet compounds: {sorted(compounds & WET_COMPOUNDS)}"
        return row

    out = race_dir(season, rnd, event_name)
    if out.exists() and not force:
        row["status"] = "cached"
        return row
    out.mkdir(parents=True, exist_ok=True)

    laps_df = td_to_seconds(pd.DataFrame(laps))
    laps_df["season"] = season
    laps_df["round"] = rnd
    laps_df["event"] = event_name
    laps_df["TrackStatus"] = laps_df["TrackStatus"].astype("string")
    laps_df.to_parquet(out / "laps.parquet", index=False)

    weather = session.weather_data
    weather_df = td_to_seconds(pd.DataFrame(weather)) if weather is not None else pd.DataFrame()
    weather_df.to_parquet(out / "weather.parquet", index=False)

    rcm = session.race_control_messages
    rcm_df = td_to_seconds(pd.DataFrame(rcm)) if rcm is not None else pd.DataFrame()
    for col in ("Message", "Category", "Flag", "Scope", "Status", "RacingNumber"):
        if col in rcm_df.columns:
            rcm_df[col] = rcm_df[col].astype("string")
    rcm_df.to_parquet(out / "rcm.parquet", index=False)

    # Race control message Time is wall clock, and without telemetry FastF1 has no t0_date to
    # map it to session time. Track status and session status carry session time directly.
    for name, frame in (("track_status", session.track_status),
                        ("session_status", session.session_status)):
        df = td_to_seconds(pd.DataFrame(frame)) if frame is not None else pd.DataFrame()
        for col in ("Status", "Message"):
            if col in df.columns:
                df[col] = df[col].astype("string")
        df.to_parquet(out / f"{name}.parquet", index=False)

    # total race laps, useful for sc_rates and laps_remaining
    meta = pd.DataFrame(
        [
            {
                "season": season,
                "round": rnd,
                "event": event_name,
                "split": split,
                "race_laps": int(laps_df["LapNumber"].max()),
                "n_drivers": int(laps_df["Driver"].nunique()),
                "date": pd.to_datetime(session.event["EventDate"]),
            }
        ]
    )
    meta.to_parquet(out / "meta.parquet", index=False)

    row["status"] = "ok"
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest FastF1 race sessions.")
    parser.add_argument("--set", dest="split", default="all", choices=["train", "holdout", "all"])
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args(argv)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    cfg = load_config(Path(args.config))
    splits = ["train", "holdout"] if args.split == "all" else [args.split]

    rows = []
    for split in splits:
        for entry in cfg.get(split, []):
            season, event = int(entry["season"]), str(entry["event"])
            print(f"[{split}] {season} {event} ...", flush=True)
            row = ingest_race(season, event, split, force=args.force)
            rows.append(row)
            print(f"    -> {row['status']} {row['reason']}".rstrip(), flush=True)

    manifest = pd.DataFrame(rows)
    manifest_path = Path("data/processed/ingest_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    counts = manifest["status"].value_counts().to_dict()
    print("\nsummary:", counts)
    excluded = manifest[manifest["status"].isin(["excluded", "error"])]
    if len(excluded):
        print("\nexcluded / errored:")
        for _, r in excluded.iterrows():
            print(f"  {r['season']} {r['event_query']}: {r['reason']}")
    print(f"\nmanifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())