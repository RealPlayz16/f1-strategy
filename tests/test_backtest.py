import pandas as pd

from src.backtest import episodes, match


def test_episodes_split_on_track_clear():
    recs = pd.DataFrame(
        {"t": [100.0, 110.0, 200.0, 300.0, 320.0],
         "flag": ["VSC", "SC", "CLEAR", "SC", "SC"],
         "message": ["VSC", "SC", "TRACK CLEAR", "SC", "SC"],
         "reason": ["a", "b", "", "c", "d"], "confidence": [0.7, 0.8, 1.0, 0.7, 0.9]}
    )
    e = episodes(recs)
    assert list(e["t_call"]) == [100.0, 300.0]
    assert list(e["kind"]) == ["VSC", "SC"]


def test_match_real_and_false_calls():
    eps = pd.DataFrame({"t_call": [100.0, 1000.0], "kind": ["VSC", "SC"],
                        "reason": ["", ""], "confidence": [0.7, 0.7]})
    neutral = pd.DataFrame({"kind": ["VSC"], "t_deploy_s": [130.0], "t_end_s": [230.0]})
    m = match(eps, neutral)
    assert list(m["real"]) == [True, False]
    assert m.loc[0, "lead_s"] == 30.0
