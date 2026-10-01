import numpy as np

from src.rules import Stint, can_overtake, drs_enabled, pit_loss_s, strategy_violations


def test_legal_two_compound_plan():
    assert strategy_violations([Stint("MEDIUM", 25), Stint("HARD", 32)], 57) == []


def test_single_compound_dry_race_illegal_wet_waives_it():
    assert strategy_violations([Stint("MEDIUM", 20), Stint("MEDIUM", 37)], 57)
    assert strategy_violations([Stint("INTERMEDIATE", 20), Stint("INTERMEDIATE", 37)], 57) == []


def test_stint_caps_and_lap_total():
    problems = strategy_violations([Stint("SOFT", 40), Stint("HARD", 10)], 57)
    assert any("exceeds 28" in p for p in problems)
    assert any("cover 50 laps" in p for p in problems)
    assert strategy_violations([Stint("SOFT", 20), Stint("HARD", 37)], 57, tyre_limit_laps=25)


def test_overtaking_and_drs():
    assert can_overtake("1") and can_overtake("12")
    assert not can_overtake("14") and not can_overtake("6") and not can_overtake("67")
    assert not drs_enabled(2, None, 0.5)
    assert drs_enabled(3, None, 0.5)
    assert not drs_enabled(20, 1, 0.5) and drs_enabled(20, 2, 0.5)
    assert not drs_enabled(20, None, 1.2)


def test_pit_loss_by_condition():
    row = {"green_s": 22.0, "lap_green_s": 90.0, "lap_sc_s": 128.0, "lap_vsc_s": 120.0}
    assert pit_loss_s(row, "green", 0.08) == 22.0
    assert np.isclose(pit_loss_s(row, "sc", 0.08), 22.0 - 0.08 * 38.0)
    assert pit_loss_s(row, "red", 0.08) == 0.0
