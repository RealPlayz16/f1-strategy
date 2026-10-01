import numpy as np
import pandas as pd
import torch

from src.tyre import REF_LAPS, Encoder, build_frame, pinball


def _laps():
    rows = []
    for drv, pace in (("VER", 0.0), ("HAM", 0.5)):
        for lap in range(1, 21):
            rows.append(
                {
                    "season": 2024, "round": 1, "event": "Italian Grand Prix", "driver": drv,
                    "team": "T", "lap": lap, "is_clean": lap > 1, "compound": "HARD",
                    "tyre_life": float(lap), "stint": 1.0, "fresh_tyre": True,
                    "track_temp": 40.0, "air_temp": 25.0,
                    "lap_time_fc_s": 80.0 + pace + 0.05 * lap,
                }
            )
    return pd.DataFrame(rows)


def test_reference_window_is_causal():
    df = build_frame(_laps())
    lo, hi = REF_LAPS
    assert df["lap"].min() == hi + 1  # only laps after the window are modelled
    window = _laps().query("@lo <= lap <= @hi")
    ref = window["lap_time_fc_s"].median()
    assert np.allclose(df["race_ref_s"], ref)
    ham = window[window["driver"] == "HAM"]["lap_time_fc_s"].median() - ref
    assert np.allclose(df.loc[df["driver"] == "HAM", "driver_pace_s"], ham)


def test_holdout_never_in_frame():
    laps = _laps()
    laps["season"], laps["event"] = 2025, "Japanese Grand Prix"
    assert build_frame(laps).empty


def test_pinball_is_quantile_loss():
    y = torch.tensor([1.0])
    pred = torch.tensor([[0.0, 1.0, 2.0]])  # p10 below by 1, p50 exact, p90 above by 1
    expected = (0.1 * 1 + 0 + (1 - 0.9) * 1) / 3
    assert np.isclose(pinball(pred, y).item(), expected)


def test_unknown_category_maps_to_zero():
    df = build_frame(_laps())
    enc = Encoder(df)
    other = df.head(1).assign(event="Unseen Grand Prix")
    _, cat = enc(other)
    assert cat[0, 1].item() == 0
