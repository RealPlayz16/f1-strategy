import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.live import RaceState, ReplayDriver, decide, plan_rows, quantile_sum


def _race():
    laps = pd.DataFrame(
        [{"driver": d, "lap": lap, "session_time_s": 100.0 * lap + off, "lap_time_s": 100.0}
         for lap in range(1, 11) for d, off in (("VER", 0.0), ("HAM", 1.5))]
    )
    recs = pd.DataFrame({"t": [350.0, 720.0], "flag": ["VSC", "SC"]})
    sc = pd.DataFrame(
        {"kind": ["VSC"], "t_deploy_s": [380.0], "t_end_s": [480.0], "lap_end": [5],
         "duration_s": [100.0], "duration_laps": [1], "t_ending_s": [470.0],
         "ended_by": ["AllClear"]}
    )
    return laps, recs, sc


def _state_frames(state):
    return state.laps.reset_index(drop=True), state.recs.reset_index(drop=True), \
        state.neutral.reset_index(drop=True)


def test_state_contains_nothing_after_t():
    d = ReplayDriver(*_race())
    s = d.at(401.0)
    assert s.laps["session_time_s"].max() <= 401.0
    assert s.recs["t"].max() <= 401.0
    assert (s.neutral["t_deploy_s"] <= 401.0).all()
    # VSC deployed at 380 is still running at 401: its end must not be visible
    assert s.neutral["t_end_s"].isna().all() and s.neutral["lap_end"].isna().all()


def test_future_changes_do_not_change_the_state():
    laps, recs, sc = _race()
    before = _state_frames(ReplayDriver(laps, recs, sc).at(401.0))
    laps2, recs2, sc2 = laps.copy(), recs.copy(), sc.copy()
    laps2.loc[laps2["session_time_s"] > 401.0, "lap_time_s"] = 999.0
    recs2.loc[recs2["t"] > 401.0, "flag"] = "RED"
    sc2["t_end_s"], sc2["lap_end"] = 900.0, 9
    after = _state_frames(ReplayDriver(laps2, recs2, sc2).at(401.0))
    for a, b in zip(before, after, strict=True):
        pdt.assert_frame_equal(a, b)


def test_state_at_lap_uses_leader_lap_end():
    d = ReplayDriver(*_race())
    s = d.at_lap(4)
    assert s.t == 400.0
    assert set(s.laps["lap"]) <= {1, 2, 3, 4}
    assert ((s.laps["lap"] == 4) & (s.laps["driver"] == "HAM")).sum() == 0  # HAM ends at 401.5


def test_pit_now_stop_is_applied_before_the_first_scored_lap():
    rows = plan_rows({}, range(8, 11), {7: "HARD", 9: "SOFT"}, "MEDIUM", 6.0, anchor_lap=6)
    assert [r["compound_f"] for r in rows] == ["HARD", "HARD", "SOFT"]
    assert [r["tyre_life_f"] for r in rows] == [1.0, 2.0, 1.0]
    assert [r["stints_ahead"] for r in rows] == [1, 1, 2]
    stay = plan_rows({}, range(8, 11), {}, "MEDIUM", 6.0, anchor_lap=6)
    assert [r["tyre_life_f"] for r in stay] == [7.0, 8.0, 9.0]


def test_undecided_rows_still_carry_the_sensitivity_field():
    """An early return is still a row of that soft-bias run; without the field the
    undecided cars disappear from any filter on it (dashboard export bug)."""
    state = RaceState(t=100.0, laps=pd.DataFrame(columns=["driver", "lap"]),
                      recs=pd.DataFrame(), neutral=pd.DataFrame())
    pitloss = {"green": 22.0, "sc": 18.0, "vsc": 19.0, "delta_lap_sc": 40.0,
               "delta_lap_vsc": 30.0, "source": "test"}
    dec = decide(state, "VER", pd.DataFrame(), "SC", {"p": 0.5}, pitloss, 56, soft_bias=0.075)
    assert dec["decision"] == "no data"
    assert dec["soft_bias_s"] == 0.075


def test_quantile_sum_comonotonic():
    q = np.array([[89.0, 90.0, 91.0], [89.5, 90.0, 91.5]])
    u = np.array([0.1, 0.5, 0.9])
    total = quantile_sum(q, u)
    assert np.allclose(total, [178.5, 180.0, 182.5])
