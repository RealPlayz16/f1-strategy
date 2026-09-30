import pandas as pd

from src.clean import add_gap_ahead, flag_outliers, parse_track_status


def test_parse_track_status_multi_code():
    s = pd.Series(["1", "14", "6", "25", "", None])
    out = parse_track_status(s)
    assert out["is_sc"].tolist() == [False, True, False, False, False, False]
    assert out["is_vsc"].tolist() == [False, False, True, False, False, False]
    assert out["is_yellow"].tolist() == [False, False, False, True, False, False]
    assert out["is_red"].tolist() == [False, False, False, True, False, False]


def test_gap_ahead_within_lap():
    df = pd.DataFrame(
        {
            "driver": ["VER", "HAM", "LEC", "VER"],
            "lap": [5, 5, 5, 6],
            "session_time_s": [100.0, 101.5, 104.0, 190.0],
        }
    )
    out = add_gap_ahead(df).sort_values(["lap", "session_time_s"])
    assert pd.isna(out.iloc[0]["gap_ahead_s"])
    assert out.iloc[1]["gap_ahead_s"] == 1.5
    assert round(out.iloc[2]["gap_ahead_s"], 3) == 2.5
    assert pd.isna(out.iloc[3]["gap_ahead_s"])


def test_flag_outliers_uses_stint_median():
    df = pd.DataFrame(
        {
            "driver": ["VER"] * 5,
            "stint": [1.0] * 5,
            "lap": [2, 3, 4, 5, 6],
            "lap_time_s": [90.0, 90.0, 90.0, 90.0, 120.0],
            "is_pit_in": [False] * 5,
            "is_pit_out": [False] * 5,
            "is_lap1": [False] * 5,
            "is_sc": [False] * 5,
            "is_vsc": [False] * 5,
            "is_red": [False] * 5,
        }
    )
    out = flag_outliers(df)
    assert out.tolist() == [False, False, False, False, True]