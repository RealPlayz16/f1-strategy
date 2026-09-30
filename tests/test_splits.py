import pandas as pd
import pytest

from src.splits import add_split, load_splits, split_of


def test_holdout_is_four_races_outside_fast_flag_training():
    splits = load_splits()
    holdout = {k for k, v in splits.items() if v == "holdout"}
    assert len(holdout) == 4
    # Fast Flag trained on 2025 Dutch: it must not be in our holdout
    assert (2025, "Dutch") not in holdout
    assert split_of(2025, "Dutch Grand Prix", splits) == "train"
    assert split_of(2025, "United States Grand Prix", splits) == "holdout"


def test_split_of_unknown_race_raises():
    with pytest.raises(KeyError):
        split_of(2023, "Monaco Grand Prix")


def test_add_split_column():
    df = pd.DataFrame(
        {"season": [2024, 2025], "event": ["Italian Grand Prix", "Japanese Grand Prix"]}
    )
    assert add_split(df)["split"].tolist() == ["train", "holdout"]
