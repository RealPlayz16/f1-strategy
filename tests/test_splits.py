import pandas as pd
import pytest

from src.splits import add_split, load_splits, split_of

# Fast Flag's 2025 TRAINING races, from ~/fast-flag/data/models/risk_meta.json key "races".
# Any of these in our holdout would leak its models into the decision path. Hardcoded rather
# than read from that repo so the test runs in CI, which has no checkout of it. Re-check the
# source file when adding holdout races: it can change upstream without warning.
FAST_FLAG_TRAINING_2025 = {"Australian", "Azerbaijan", "Belgian", "British", "Dutch", "Miami"}


def test_holdout_excludes_every_fast_flag_training_race():
    splits = load_splits()
    holdout = {k for k, v in splits.items() if v == "holdout"}
    leaked = {e for season, e in holdout if season == 2025 and e in FAST_FLAG_TRAINING_2025}
    assert not leaked, f"Fast Flag training races in our holdout: {sorted(leaked)}"
    assert split_of(2025, "Dutch Grand Prix", splits) == "train"
    assert split_of(2025, "United States Grand Prix", splits) == "holdout"


def test_split_sizes_match_the_session_8_split():
    splits = load_splits()
    holdout = {k for k, v in splits.items() if v == "holdout"}
    train = {k for k, v in splits.items() if v == "train"}
    assert (len(train), len(holdout)) == (30, 9)


def test_split_of_unknown_race_raises():
    with pytest.raises(KeyError):
        split_of(2023, "Monaco Grand Prix")


def test_add_split_column():
    df = pd.DataFrame(
        {"season": [2024, 2025], "event": ["Italian Grand Prix", "Japanese Grand Prix"]}
    )
    assert add_split(df)["split"].tolist() == ["train", "holdout"]
