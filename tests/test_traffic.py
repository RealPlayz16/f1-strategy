import numpy as np
import pandas as pd

from src.overtake_model import calibration_table, features
from src.traffic import dirty_air_penalty, lap_state


def test_dirty_air_penalty_shape():
    p = dirty_air_penalty([0.2, 0.5, 0.75, 1.0, 2.0], d0=0.4)
    assert np.allclose(p, [0.4, 0.4, 0.2, 0.0, 0.0])


def _laps():
    """VER leads in free air; HAM runs 3.5 s back until lap 6, then 0.4 s back (following)."""
    rows = []
    for drv, base, gaps in (("VER", 90.0, [np.nan] * 10),
                            ("HAM", 90.2, [3.5] * 5 + [0.4] * 5)):
        t = 0.0 if drv == "VER" else 3.5
        for lap in range(1, 11):
            follow = drv == "HAM" and lap >= 7
            lt = base + 0.05 * lap + (0.5 if follow else 0.0)
            t += lt
            rows.append(
                {"race": "2024_01", "driver": drv, "lap": lap, "stint": 1.0,
                 "compound": "HARD", "tyre_life": float(lap), "lap_time_fc_s": lt,
                 "session_time_s": t, "gap_ahead_s": gaps[lap - 1], "is_clean": lap > 1}
            )
    return pd.DataFrame(rows)


def test_free_pace_from_clear_air_and_dev_while_following():
    slopes = pd.DataFrame({"race": ["2024_01"], "compound": ["HARD"], "slope_B": [0.05]})
    s = lap_state(_laps(), slopes).set_index(["driver", "lap"])
    # HAM free pace from laps 3-6 (gap > 3 s at end of the previous lap): 90.2 + 0.05 * lap
    assert np.isclose(s.loc[("HAM", 8), "free_pace"], 90.2 + 0.4)
    assert np.isclose(s.loc[("HAM", 8), "dev"], 0.5)
    assert s.loc[("HAM", 8), "ahead_prev"] == "VER"
    assert np.isclose(s.loc[("HAM", 8), "free_delta"], -0.2)  # HAM slower in free air


def test_features_compound_step_and_gap_bins():
    b = pd.DataFrame(
        {"gap_before_s": [0.3, 1.7], "free_delta": [0.4, np.nan], "tyre_age_delta": [5, 0],
         "compound_pair": ["SOFT-HARD", "HARD-SOFT"], "laps_in_battle": [1, 4],
         "drs_available": [1.0, 0.0], "event": ["X", "Y"]}
    )
    x = features(b, ["X"])
    assert list(x["compound_step"]) == [2, -2]  # overtaker softer is positive
    assert list(x["free_missing"]) == [0.0, 1.0]
    assert x.filter(like="gap_").sum(axis=1).tolist() == [1.0, 1.0]
    assert x["event_X"].tolist() == [1.0, 0.0]


def test_calibration_table_buckets():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.01, 0.01, 0.6, 0.8])
    t = calibration_table(y, p)
    assert t["observed_rate"].iloc[0] == 0.0 and t["observed_rate"].iloc[-1] == 1.0
