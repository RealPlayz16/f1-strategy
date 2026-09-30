import numpy as np
import pandas as pd

from src.pitloss import MIN_SEASON_FIT, build_stops, by_track, classify_condition, fit_phi

PHI = 0.1
GREEN = 90.0
SC = 126.0
TRANSIT = 25.0
DRIVERS = ["VER", "HAM", "LEC", "NOR", "RUS", "PIA"]


def _driver(drv, *, stop_lap=None, sc_laps=(), season=2024, rnd=1, n=10, transit=TRANSIT):
    """One driver's race. A stop obeys in_lap + out_lap = (2 - PHI) * pace + transit."""
    rows, t, stint, tyre_life = [], 0.0, 1.0, 0.0
    for lap in range(1, n + 1):
        sc = lap in sc_laps
        pace = SC if sc else GREEN
        is_in = stop_lap is not None and lap == stop_lap
        is_out = stop_lap is not None and lap == stop_lap + 1
        lap_time = ((2 - PHI) * pace + transit) / 2 if (is_in or is_out) else pace
        if is_out:
            stint, tyre_life = 2.0, 0.0
        tyre_life += 1
        t += lap_time
        rows.append(
            {
                "season": season, "round": rnd, "event": "Test Grand Prix", "driver": drv,
                "lap": lap, "lap_time_s": lap_time, "compound": "HARD" if stint == 2 else "SOFT",
                "stint": stint, "tyre_life": tyre_life, "is_pit_in": is_in,
                "is_pit_out": is_out, "is_sc": sc, "is_vsc": False, "is_red": False,
                "is_clean": not (is_in or is_out or sc or lap == 1),
                "pit_in_time_s": t - 2.0 if is_in else np.nan, "pit_out_time_s": np.nan,
            }
        )
    df = pd.DataFrame(rows)
    if stop_lap is not None:
        pit_in = df.loc[df["lap"] == stop_lap, "pit_in_time_s"].iloc[0]
        df.loc[df["lap"] == stop_lap + 1, "pit_out_time_s"] = pit_in + transit
    return df


def _green_race(season=2024, rnd=1):
    """Every driver makes one green stop, on a different lap."""
    return pd.concat(
        [_driver(d, stop_lap=4 + i % 3, season=season, rnd=rnd) for i, d in enumerate(DRIVERS)],
        ignore_index=True,
    )


def _sc_race(season=2025, rnd=1):
    """SC on laps 4-7; only VER pits, five cars stay out."""
    frames = [_driver("VER", stop_lap=5, sc_laps=range(4, 8), season=season, rnd=rnd)]
    frames += [_driver(d, sc_laps=range(4, 8), season=season, rnd=rnd) for d in DRIVERS[1:]]
    return pd.concat(frames, ignore_index=True)


def _mass_pit_race(season=2025, rnd=2):
    """SC on laps 2-4 and the whole field pits on lap 2: no stay-out reference."""
    return pd.concat(
        [_driver(d, stop_lap=2, sc_laps=range(2, 5), season=season, rnd=rnd) for d in DRIVERS],
        ignore_index=True,
    )


def test_phi_recovered_from_green_stops():
    stops, phi = build_stops(_green_race())
    assert len(stops) == len(DRIVERS)
    assert np.isclose(phi["phi"].iloc[0], PHI)
    assert stops["pace_trusted"].all()
    assert np.allclose(stops["ref_lap_s"], GREEN)
    # transit - phi * L equals lap-sum subtraction when L is known
    assert np.allclose(stops["pit_loss_s"], TRANSIT - PHI * GREEN)
    assert np.allclose(stops["pit_loss_s"], stops["lap_sum_s"] - 2 * GREEN)


def test_sc_discount_falls_out_of_phi():
    stops, _ = build_stops(pd.concat([_green_race(), _sc_race()], ignore_index=True))
    sc = stops[stops["condition"] == "sc"].iloc[0]
    assert sc["pace_trusted"]
    assert np.isclose(sc["ref_lap_s"], SC)
    assert np.isclose(sc["pit_loss_s"], TRANSIT - PHI * SC)
    tracks = by_track(stops)
    assert tracks["sc_s"].iloc[0] < tracks["green_s"].iloc[0]


def test_mass_pit_uses_condition_ratio():
    laps = pd.concat([_green_race(), _sc_race(), _mass_pit_race()], ignore_index=True)
    stops, _ = build_stops(laps)
    mass = stops[(stops["round"] == 2) & (stops["season"] == 2025)]
    assert len(mass) == len(DRIVERS)
    assert not mass["pace_trusted"].any()
    # ratio SC / green = 1.4 from the trusted SC stop, applied to each driver's green pace
    assert np.allclose(mass["ref_lap_s"], GREEN * SC / GREEN)
    assert np.allclose(mass["pit_loss_s"], TRANSIT - PHI * SC)


def test_pit_lane_passage_without_tyre_change_skipped():
    """Field follows the SC through the pit lane: new FastF1 stint, same tyres."""
    passage = _mass_pit_race()
    out_lap = passage["lap"] == 3
    passage.loc[out_lap, "compound"] = "SOFT"
    passage.loc[passage["lap"] >= 3, "tyre_life"] += 2  # set continues: 2 -> 3 -> ...
    stops, _ = build_stops(pd.concat([_green_race(), passage], ignore_index=True))
    assert not ((stops["season"] == 2025) & (stops["round"] == 2)).any()


def test_red_flag_stop_skipped():
    df = _driver("VER", stop_lap=5)
    df.loc[df["lap"] == 5, "is_red"] = True
    other = _green_race(rnd=2)
    stops, _ = build_stops(pd.concat([df, other], ignore_index=True))
    assert not ((stops["round"] == 1) & (stops["driver"] == "VER")).any()


def test_stop_without_lap_time_kept_but_not_fitted():
    """FastF1 often has no lap time on SC laps; the transit still gives the loss."""
    df = _driver("VER", stop_lap=5, rnd=2)
    df.loc[df["lap"] == 6, "lap_time_s"] = np.nan
    stops, phi = build_stops(pd.concat([_green_race(), df], ignore_index=True))
    kept = stops[stops["round"] == 2].iloc[0]
    assert np.isnan(kept["lap_sum_s"])
    assert np.isclose(kept["pit_loss_s"], TRANSIT - PHI * GREEN)
    assert phi["n_fit"].iloc[0] == len(DRIVERS)


def test_garage_visit_dropped():
    """Car goes to the garage and rejoins much later: longer than a lap, not a stop."""
    df = _driver("VER", stop_lap=5, transit=2485.0, rnd=2)
    df.loc[df["lap"] == 6, "lap_time_s"] = np.nan
    stops, _ = build_stops(pd.concat([_green_race(), df], ignore_index=True))
    assert not (stops["round"] == 2).any()


def test_phi_season_spread():
    rows = []
    for season, phi in ((2023, 0.10), (2024, 0.13), (2025, 0.11)):
        for _ in range(MIN_SEASON_FIT):
            rows.append(
                {
                    "season": season, "event": "Test Grand Prix", "condition": "green",
                    "pace_trusted": True, "ref_lap_s": GREEN, "transit_s": TRANSIT,
                    "lap_sum_s": (2 - phi) * GREEN + TRANSIT,
                }
            )
    # a thin season does not count toward the spread
    rows.append({**rows[0], "season": 2022, "lap_sum_s": (2 - 0.5) * GREEN + TRANSIT})
    out = fit_phi(pd.DataFrame(rows))
    assert np.isclose(out["phi_season_spread"].iloc[0], 0.03)
    assert out["n_fit"].iloc[0] == 3 * MIN_SEASON_FIT + 1


def test_sc_condition_wins_over_vsc():
    row_sc = pd.Series({"is_sc": True, "is_vsc": True})
    row_plain = pd.Series({"is_sc": False, "is_vsc": False})
    assert classify_condition(row_sc, row_plain) == "sc"
    assert classify_condition(row_plain, pd.Series({"is_sc": False, "is_vsc": True})) == "vsc"
    assert classify_condition(row_plain, row_plain) == "green"
