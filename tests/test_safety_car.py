import numpy as np
import pandas as pd

from src.safety_car import leader_lap_at, race_events, rates

# 10 laps of 100 s for two cars, leader 1 s ahead
LAPS = pd.DataFrame(
    {
        "LapNumber": [lap for lap in range(1, 11) for _ in range(2)],
        "Time": [100.0 * lap + off for lap in range(1, 11) for off in (0.0, 1.0)],
    }
)
WINDOW = (0.0, 1000.0)


def _status(rows):
    return pd.DataFrame(rows, columns=["Time", "Status", "Message"])


def test_leader_lap_at():
    assert leader_lap_at(LAPS, 50.0) == 1
    assert leader_lap_at(LAPS, 150.0) == 2
    assert leader_lap_at(LAPS, 950.0) == 10


def test_sc_episode():
    ts = _status([(0.0, "1", "AllClear"), (250.0, "4", "SCDeployed"), (520.0, "1", "AllClear")])
    (ev,) = race_events(ts, LAPS, WINDOW)
    assert ev["kind"] == "SC"
    assert (ev["lap_deploy"], ev["lap_end"]) == (3, 6)
    assert np.isclose(ev["duration_s"], 270.0)


def test_vsc_converted_to_sc():
    ts = _status(
        [(0.0, "1", "AllClear"), (250.0, "6", "VSCDeployed"), (300.0, "4", "SCDeployed"),
         (600.0, "1", "AllClear")]
    )
    vsc, sc = race_events(ts, LAPS, WINDOW)
    assert vsc["kind"] == "VSC" and vsc["ended_by"] == "SCDeployed"
    assert np.isclose(vsc["t_end_s"], 300.0)
    assert sc["kind"] == "SC"


def test_vsc_ending_kept_and_yellow_ignored():
    ts = _status(
        [(0.0, "1", "AllClear"), (150.0, "2", "Yellow"), (250.0, "6", "VSCDeployed"),
         (330.0, "7", "VSCEnding"), (345.0, "1", "AllClear")]
    )
    (ev,) = race_events(ts, LAPS, WINDOW)
    assert ev["kind"] == "VSC"
    assert np.isclose(ev["t_ending_s"], 330.0)
    assert np.isclose(ev["t_end_s"], 345.0)


def test_events_outside_race_window_ignored():
    ts = _status([(0.0, "4", "SCDeployed"), (50.0, "1", "AllClear")])
    assert race_events(ts, LAPS, (100.0, 1000.0)) == []


def test_rates_train_only_lap1_apart_and_shrunk():
    races = pd.DataFrame(
        {
            "season": [2023, 2024, 2025], "round": [1, 1, 1], "event": ["A", "B", "A"],
            "split": ["train", "train", "holdout"], "race_laps": [51, 51, 51],
        }
    )
    events = pd.DataFrame(
        {
            "season": [2023, 2023, 2025], "round": [1, 1, 1], "event": ["A", "A", "A"],
            "split": ["train", "train", "holdout"], "kind": ["SC", "SC", "SC"],
            "lap_deploy": [1, 20, 20],
        }
    )
    out = rates(events, races, prior_laps=50.0).set_index("event")
    assert out.loc["A", "sc_deployments"] == 1  # lap 1 counted apart, holdout excluded
    assert np.isclose(out.loc["A", "p_sc_lap1"], 0.5)
    p_global = 1 / 100
    assert np.isclose(out.loc["A", "p_sc_per_lap_raw"], 1 / 50)
    assert np.isclose(out.loc["A", "p_sc_per_lap"], (1 + 50 * p_global) / (50 + 50))
    assert np.isclose(out.loc["B", "p_sc_per_lap"], (0 + 50 * p_global) / (50 + 50))
