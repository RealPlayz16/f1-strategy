import numpy as np
import pandas as pd
import torch

from src.tyre import (
    CATS,
    HORIZONS,
    NUMS,
    SPREAD_FEATURES,
    WINDOW_START,
    Encoder,
    build_frame,
    compute_anchors,
    pinball,
)


def _laps(n=40):
    rows = []
    for drv, pace in (("VER", 0.0), ("HAM", 0.5), ("LEC", 0.2)):
        for lap in range(1, n + 1):
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


def test_anchor_skips_race_start_and_is_causal():
    laps = _laps()
    a = compute_anchors(laps[laps["is_clean"]])
    assert a["lap"].min() >= WINDOW_START + 1  # needs 2 driver laps from lap 5 on
    before = a[(a["driver"] == "HAM") & (a["lap"] == 10)]["anchor_s"].iloc[0]
    changed = laps.copy()
    changed.loc[changed["lap"] > 10, "lap_time_fc_s"] += 5.0  # the future must not matter
    after = compute_anchors(changed[changed["is_clean"]])
    after = after[(after["driver"] == "HAM") & (after["lap"] == 10)]["anchor_s"].iloc[0]
    assert np.isclose(before, after)


def test_anchor_carries_driver_gap():
    a = compute_anchors(_laps()[lambda d: d["is_clean"]]).set_index(["driver", "lap"])
    # HAM is 0.3 s slower than the per-lap field median (LEC) on every lap
    assert np.isclose(a.loc[("HAM", 20), "anchor_s"] - a.loc[("HAM", 20), "field_ref_s"], 0.3)


def test_frame_targets_are_future_laps():
    df = build_frame(_laps())
    assert set(df["h"]) <= set(HORIZONS)
    assert (df["lap_f"] == df["lap"] + df["h"]).all()
    row = df[(df["driver"] == "VER") & (df["lap"] == 10) & (df["h"] == 5)].iloc[0]
    assert np.isclose(row["y"], (80.0 + 0.05 * 15) - row["anchor_s"])


def test_holdout_never_in_frame():
    laps = _laps()
    laps["season"], laps["event"] = 2025, "Japanese Grand Prix"
    assert build_frame(laps).empty


def test_pinball_is_quantile_loss():
    y = torch.tensor([1.0])
    pred = torch.tensor([[0.0, 2.0]])  # p10 below by 1, p90 above by 1
    expected = (0.1 * 1 + (1 - 0.9) * 1) / 2
    assert np.isclose(pinball(pred, y).item(), expected)


def test_unknown_category_maps_to_zero_and_state_roundtrips():
    df = build_frame(_laps())
    enc = Encoder(df, CATS, NUMS)
    _, cat = enc(df.head(1).assign(event="Unseen Grand Prix"))
    assert cat[0, CATS.index("event")].item() == 0
    back = Encoder.from_state(enc.state())
    a, b = enc(df.head(3)), back(df.head(3))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_spread_features_cannot_identify_a_race():
    cats, nums = SPREAD_FEATURES["tyre"]
    for col in ("event", "driver", "team", "track_temp", "air_temp"):
        assert col not in cats + nums
