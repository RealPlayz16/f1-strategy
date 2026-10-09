import json

import numpy as np
import pandas as pd

from src.optimizer import (
    BIG,
    COMPOUNDS,
    MAX_AGE,
    MAX_STOPS,
    car_state,
    plan_time,
    solve,
)
from src.rules import MAX_STINT_LAPS, strategy_violations
from src.rules import Stint as RuleStint


def flat_grid(n_h: int, lap_s: float = 90.0, caps: bool = True) -> np.ndarray:
    """lap_s everywhere a stint cap allows, BIG elsewhere."""
    lt = np.full((n_h + 1, len(COMPOUNDS), MAX_AGE + 1, MAX_STOPS + 1), BIG)
    for h in range(1, n_h + 1):
        for ci, comp in enumerate(COMPOUNDS):
            top = MAX_STINT_LAPS[comp] if caps else MAX_AGE
            lt[h, ci, : top + 1, :] = lap_s
    return lt


def test_two_compound_rule_forces_exactly_one_stop_when_a_stop_is_free():
    lt = flat_grid(20)
    total, stops = solve(lt, 10, 30, "MEDIUM", 5.0, 0, {"MEDIUM"}, np.zeros(21))
    assert len(stops) == 1
    assert set(stops.values()) <= set(COMPOUNDS) - {"MEDIUM"}
    assert total == 20 * 90.0


def test_no_stop_when_the_rule_is_already_satisfied():
    lt = flat_grid(20)
    total, stops = solve(lt, 10, 30, "MEDIUM", 5.0, 0, {"MEDIUM", "HARD"}, np.zeros(21))
    assert stops == {}
    assert total == 20 * 90.0


def test_pit_loss_is_paid_exactly_once():
    lt = flat_grid(20)
    total, _ = solve(lt, 10, 30, "MEDIUM", 5.0, 0, {"MEDIUM"}, np.full(21, 25.0))
    assert total == 20 * 90.0 + 25.0


def test_a_cheaper_compound_is_preferred_when_it_is_legal():
    lt = flat_grid(20)
    lt[:, COMPOUNDS.index("HARD"), :, :] = np.where(
        lt[:, COMPOUNDS.index("HARD"), :, :] >= BIG, BIG, 89.0)
    _, stops = solve(lt, 10, 30, "MEDIUM", 5.0, 0, {"MEDIUM"}, np.zeros(21))
    assert list(stops.values()) == ["HARD"]


def test_stint_cap_forces_more_than_one_stop():
    # 60 laps left and only SOFT available to run on: cap 28, so one stint cannot cover it.
    n_h = 60
    lt = np.full((n_h + 1, len(COMPOUNDS), MAX_AGE + 1, MAX_STOPS + 1), BIG)
    si = COMPOUNDS.index("SOFT")
    for h in range(1, n_h + 1):
        lt[h, si, : MAX_STINT_LAPS["SOFT"] + 1, :] = 90.0
    total, stops = solve(lt, 10, 70, "SOFT", 1.0, 0, {"SOFT", "HARD"}, np.zeros(n_h + 1))
    assert np.isfinite(total)
    assert len(stops) >= 2
    assert all(c == "SOFT" for c in stops.values())


def test_solve_returns_nan_when_no_legal_plan_exists():
    lt = np.full((21, len(COMPOUNDS), MAX_AGE + 1, MAX_STOPS + 1), BIG)
    total, stops = solve(lt, 10, 30, "MEDIUM", 5.0, 0, {"MEDIUM"}, np.zeros(21))
    assert np.isnan(total)
    assert stops == {}


def test_optimal_plan_is_legal_under_rules_strategy_violations():
    """The DP's own constraints must agree with src/rules.py on a concrete plan."""
    lt = flat_grid(50)
    decision_lap, race_laps = 10, 60
    # age_now == decision_lap, so the current stint began on lap 1 and the whole race can be
    # rebuilt from the DP's stops alone. With an age offset the stint starts at
    # decision_lap - age_now and reconstructing from lap 1 would double-count those laps.
    _, stops = solve(lt, decision_lap, race_laps, "MEDIUM", float(decision_lap), 0,
                     {"MEDIUM"}, np.zeros(51))
    stints, prev, comp = [], 0, "MEDIUM"
    for lap in sorted(stops):
        stints.append(RuleStint(comp, lap - prev))
        prev, comp = lap, stops[lap]
    stints.append(RuleStint(comp, race_laps - prev))
    assert sum(s.laps for s in stints) == race_laps
    assert all(s.laps <= MAX_STINT_LAPS[s.compound] for s in stints)
    assert strategy_violations(stints, race_laps) == []


def test_plan_time_matches_a_hand_computed_plan():
    lt = flat_grid(20)
    t = plan_time(lt, 10, 30, "MEDIUM", 5.0, {15: "HARD"}, np.full(21, 25.0))
    assert t == 20 * 90.0 + 25.0


def test_car_state_reads_the_actual_stops_after_the_decision_lap():
    mine = pd.DataFrame({
        "lap": [9, 10, 11, 12, 13],
        "compound": ["MEDIUM", "MEDIUM", "MEDIUM", "HARD", "HARD"],
        "tyre_life": [9.0, 10.0, 11.0, 1.0, 2.0],
        "is_pit_in": [False, False, True, False, False],
        "team": ["T"] * 5,
    })
    st = car_state(mine, 10)
    assert st["compound_now"] == "MEDIUM"
    assert st["age_now"] == 10.0
    assert st["used"] == {"MEDIUM"}
    assert st["actual_stops"] == {11: "HARD"}
    assert json.loads(json.dumps(st["actual_stops"])) == {"11": "HARD"}


# ---------- per-set lap limit (2023 Qatar had one; 2025 Qatar is identified from data) ----------

def test_detect_tyre_limit_fires_only_on_an_equal_cap_below_the_global_caps():
    from src.rules import detect_tyre_limit
    # regulatory signature: every compound lands on the same number, below the smallest cap
    assert detect_tyre_limit({"HARD": 25, "MEDIUM": 25}) == 25
    assert detect_tyre_limit({"HARD": 25, "MEDIUM": 25, "SOFT": 25}) == 25
    # degradation signature: compounds differ
    assert detect_tyre_limit({"HARD": 32, "MEDIUM": 33, "SOFT": 28}) is None
    assert detect_tyre_limit({"HARD": 38, "MEDIUM": 40}) is None
    # equal but not below the smallest global cap, so not a limit
    assert detect_tyre_limit({"HARD": 28, "MEDIUM": 28}) is None
    # a single compound cannot identify anything
    assert detect_tyre_limit({"HARD": 25}) is None


def test_stint_caps_tighten_to_the_limit():
    from src.optimizer import stint_caps
    assert stint_caps(None) == MAX_STINT_LAPS
    assert stint_caps(25) == {"SOFT": 25, "MEDIUM": 25, "HARD": 25}
    assert stint_caps(60) == MAX_STINT_LAPS      # a limit above every cap changes nothing


def test_a_limited_set_race_produces_legal_plans():
    """The gap closed in Session 9: without the limit the DP proposes a 46-lap MEDIUM stint at
    a race where no set ran past 25, and rules.strategy_violations rejects it."""
    limit, decision_lap, race_laps = 25, 10, 57
    n_h = race_laps - decision_lap
    lt = np.full((n_h + 1, len(COMPOUNDS), MAX_AGE + 1, MAX_STOPS + 1), BIG)
    for h in range(1, n_h + 1):
        for ci, comp in enumerate(COMPOUNDS):
            lt[h, ci, : min(MAX_STINT_LAPS[comp], limit) + 1, :] = 90.0
    total, stops = solve(lt, decision_lap, race_laps, "MEDIUM", float(decision_lap), 0,
                         {"MEDIUM"}, np.zeros(n_h + 1), tyre_limit=limit)
    assert np.isfinite(total)
    stints, prev, comp = [], 0, "MEDIUM"
    for lap in sorted(stops):
        stints.append(RuleStint(comp, lap - prev))
        prev, comp = lap, stops[lap]
    stints.append(RuleStint(comp, race_laps - prev))
    assert all(s.laps <= limit for s in stints), stints
    assert strategy_violations(stints, race_laps, tyre_limit_laps=limit) == []


def test_without_the_limit_the_same_race_would_be_illegal():
    """Shows the gap was real, not hypothetical: correctness was being supplied by luck."""
    limit, decision_lap, race_laps = 25, 10, 57
    n_h = race_laps - decision_lap
    lt = np.full((n_h + 1, len(COMPOUNDS), MAX_AGE + 1, MAX_STOPS + 1), BIG)
    for h in range(1, n_h + 1):
        for ci, comp in enumerate(COMPOUNDS):
            lt[h, ci, : MAX_STINT_LAPS[comp] + 1, :] = 90.0
    _, stops = solve(lt, decision_lap, race_laps, "MEDIUM", float(decision_lap), 0,
                     {"MEDIUM"}, np.zeros(n_h + 1), tyre_limit=None)
    stints, prev, comp = [], 0, "MEDIUM"
    for lap in sorted(stops):
        stints.append(RuleStint(comp, lap - prev))
        prev, comp = lap, stops[lap]
    stints.append(RuleStint(comp, race_laps - prev))
    assert strategy_violations(stints, race_laps, tyre_limit_laps=limit) != []
