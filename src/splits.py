"""Train / holdout split per race, from config/races.yaml.

The config is the single source of truth. meta.parquet also stores a split, written at ingest
time, which goes stale when a race moves between sets (2025 Dutch moved to train in Session 1).

Usage:
    python -m src.splits
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

CONFIG = Path("config/races.yaml")


def load_splits(path: Path = CONFIG) -> dict[tuple[int, str], str]:
    """(season, event query as written in the config) -> split."""
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return {
        (int(e["season"]), str(e["event"])): split
        for split in ("train", "holdout")
        for e in cfg.get(split, [])
    }


def split_of(season: int, event: str, splits: dict[tuple[int, str], str] | None = None) -> str:
    """Split for a FastF1 event name, e.g. (2025, 'Dutch Grand Prix') -> 'train'."""
    splits = load_splits() if splits is None else splits
    hits = {s for (yr, q), s in splits.items() if yr == int(season) and event.startswith(q)}
    if len(hits) != 1:
        raise KeyError(f"no unique split for {season} {event}: {hits}")
    return hits.pop()


def add_split(df: pd.DataFrame) -> pd.DataFrame:
    """Add a split column from season and event."""
    splits = load_splits()
    keys = df[["season", "event"]].drop_duplicates()
    pairs = zip(keys["season"], keys["event"], strict=True)
    keys["split"] = [split_of(s, e, splits) for s, e in pairs]
    return df.merge(keys, on=["season", "event"], how="left")


def main() -> int:
    for (season, event), split in sorted(load_splits().items(), key=lambda kv: (kv[1], kv[0])):
        print(f"{split:8} {season} {event}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
