import numpy as np
import pandas as pd

from src.fuel import correct, fit_progress, gate

K_TRUE = 3.0
DEG = 0.05  # s per lap of tyre age
RACE_LAPS = 60


def _race(season=2024, rnd=1, n_drivers=8, rng=None):
    """Two-stint races with varied pit laps. lap = base + K * frac_remaining + DEG * age."""
    rng = rng or np.random.default_rng(42)
    rows = []
    for i in range(n_drivers):
        pit = 18 + 3 * i
        base = 90.0 + 0.1 * i
        for lap in range(2, RACE_LAPS + 1):
            stint = 1 if lap <= pit else 2
            age = lap if stint == 1 else lap - pit
            compound = "MEDIUM" if stint == 1 else "HARD"
            remaining = RACE_LAPS - lap
            lt = base + K_TRUE * remaining / RACE_LAPS + DEG * age + (0.3 if stint == 2 else 0)
            rows.append(
                {
                    "season": season, "round": rnd, "event": "Test Grand Prix",
                    "driver": f"D{i}", "lap": lap, "stint": float(stint),
                    "compound": compound, "tyre_life": float(age),
                    "laps_remaining": remaining, "race_laps": RACE_LAPS,
                    "frac_remaining": remaining / RACE_LAPS,
                    "lap_time_s": lt + rng.normal(0, 0.05),
                }
            )
    return pd.DataFrame(rows)


def test_progress_recovered_across_stints():
    c = _race()
    assert np.isclose(fit_progress(c), K_TRUE, atol=0.1)


def test_stint_intercept_spec_estimates_fuel_minus_degradation():
    """The rejected spec: within a stint tyre_life and laps_remaining are collinear."""
    c = _race()
    g = c.groupby(["driver", "stint"])
    lr = c["laps_remaining"] - g["laps_remaining"].transform("mean")
    lt = c["lap_time_s"] - g["lap_time_s"].transform("mean")
    tl = c["tyre_life"] - g["tyre_life"].transform("mean")
    assert np.isclose(np.corrcoef(lr, tl)[0, 1], -1.0)
    slope = (lr * lt).sum() / (lr**2).sum()
    assert np.isclose(slope, K_TRUE / RACE_LAPS - DEG, atol=0.005)


def test_correct_uses_race_laps_only():
    laps = pd.DataFrame(
        {"season": [2025] * 3, "round": [1] * 3, "lap": [1, 2, 3],
         "laps_remaining": [2, 1, 0], "lap_time_s": [91.0, 90.5, 90.0]}
    )
    out = correct(laps, k=3.0)
    assert np.allclose(out["lap_time_fc_s"], [91.0 - 2.0, 90.5 - 1.0, 90.0])


def test_gate():
    ok = {"per_lap_at_median_race_s": 0.05, "scaling_corr": 0.5}
    assert gate(ok) == []
    assert gate({**ok, "per_lap_at_median_race_s": 0.004})
    assert gate({**ok, "scaling_corr": -0.1})
