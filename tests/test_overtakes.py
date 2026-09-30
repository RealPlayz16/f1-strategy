import numpy as np
import pandas as pd

from src.overtakes import build


def _race(times: dict[str, list[float]], *, pit_in=(), sc_laps=(), tyre=None) -> pd.DataFrame:
    """times[driver] = lap-end session times for laps 1..n. pit_in: (driver, lap) in-laps."""
    rows = []
    for drv, ends in times.items():
        for i, t in enumerate(ends):
            lap = i + 1
            prev_t = ends[i - 1] if i else 0.0
            rows.append(
                {
                    "season": 2024, "round": 1, "event": "Test Grand Prix", "split": "train",
                    "driver": drv, "lap": lap, "lap_time_s": t - prev_t, "session_time_s": t,
                    "compound": "SOFT", "tyre_life": float(tyre[drv] + lap if tyre else lap),
                    "is_pit_in": (drv, lap) in pit_in, "is_pit_out": (drv, lap - 1) in pit_in,
                    "is_sc": lap in sc_laps, "is_vsc": False, "is_red": False,
                    "is_lap1": lap == 1,
                }
            )
    return pd.DataFrame(rows)


def test_pass_detected_with_features():
    # HAM 0.4 s behind VER after lap 2, ahead after lap 3
    laps = _race(
        {"VER": [100.0, 190.0, 280.0, 370.0], "HAM": [100.5, 190.4, 279.9, 369.8]},
        tyre={"VER": 10, "HAM": 0},
    )
    over, battles = build(laps)
    assert len(over) == 1
    row = over.iloc[0]
    assert (row["overtaker"], row["overtaken"], row["lap"]) == ("HAM", "VER", 3)
    assert np.isclose(row["gap_before_s"], 0.4)
    assert row["drs_likely"]
    assert row["tyre_age_delta"] == 10
    assert row["compound_pair"] == "SOFT-SOFT"
    assert battles["passed"].sum() == 1


def test_car_pitting_is_not_passed():
    # VER pits on lap 3 and drops behind HAM: not an on-track pass
    laps = _race(
        {"VER": [100.0, 190.0, 300.0, 390.0], "HAM": [100.5, 190.4, 280.4, 370.4]},
        pit_in={("VER", 3)},
    )
    over, _ = build(laps)
    assert len(over) == 0


def test_sc_lap_ignored():
    laps = _race(
        {"VER": [100.0, 190.0, 280.0, 370.0], "HAM": [100.5, 190.4, 279.9, 369.8]},
        sc_laps={3},
    )
    over, _ = build(laps)
    assert len(over) == 0


def test_lapped_car_not_a_pass():
    # Leader VER laps SAR: compared lap by lap they never swap
    laps = _race({"VER": [90.0, 180.0, 270.0], "SAR": [100.0, 200.0, 300.0]})
    over, _ = build(laps)
    assert len(over) == 0


def test_failed_attempt_recorded_as_battle():
    laps = _race({"VER": [100.0, 190.0, 280.0], "HAM": [100.8, 190.6, 280.5]})
    over, battles = build(laps)
    assert len(over) == 0
    assert len(battles) == 2 and not battles["passed"].any()
