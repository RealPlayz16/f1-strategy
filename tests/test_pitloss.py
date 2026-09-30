import numpy as np
import pandas as pd

from src.pitloss import (
    PHI,
    apply_losses,
    build_stops,
    by_track,
    classify_condition,
    condition_ratios,
)

TRUE_PHI = 0.1
GREEN = 90.0
SC = 126.0
TRANSIT = 25.0
DRIVERS = ["VER", "HAM", "LEC", "NOR", "RUS", "PIA"]
RATIOS = {"sc": 1.4, "vsc": 1.3}


def _driver(drv, *, stop_lap=None, sc_laps=(), season=2024, rnd=1, n=10, transit=TRANSIT):
    """One driver's race. A stop obeys in_lap + out_lap = (2 - TRUE_PHI) * pace + transit."""
    rows, t, stint, tyre_life = [], 0.0, 1.0, 0.0
    for lap in range(1, n + 1):
        sc = lap in sc_laps
        pace = SC if sc else GREEN
        is_in = stop_lap is not None and lap == stop_lap
        is_out = stop_lap is not None and lap == stop_lap + 1
        lap_time = ((2 - TRUE_PHI) * pace + transit) / 2 if (is_in or is_out) else pace
        if is_out:
            stint, tyre_life = 2.0, 0.0
        tyre_life += 1
        t += lap_time
        rows.append(
            {
                "season": season, "round": rnd, "event": "Test Grand Prix", "split": "train",
                "driver": drv, "lap": lap, "lap_time_s": lap_time,
                "compound": "HARD" if stint == 2 else "SOFT", "stint": stint,
                "tyre_life": tyre_life, "is_pit_in": is_in, "is_pit_out": is_out,
                "is_sc": sc, "is_vsc": False, "is_red": False, "track_status": "4" if sc else "1",
                "is_clean": not (is_in or is_out or sc or lap == 1),
                "pit_in_time_s": t - 2.0 if is_in else np.nan, "pit_out_time_s": np.nan,
            }
        )
    df = pd.DataFrame(rows)
    if stop_lap is not None:
        pit_in = df.loc[df["lap"] == stop_lap, "pit_in_time_s"].iloc[0]
        df.loc[df["lap"] == stop_lap + 1, "pit_out_time_s"] = pit_in + transit
    return df


def _green_race(season=2024, rnd=1, transit=TRANSIT):
    """Every driver makes one green stop, on a different lap."""
    return pd.concat(
        [
            _driver(d, stop_lap=4 + i % 3, season=season, rnd=rnd, transit=transit)
            for i, d in enumerate(DRIVERS)
        ],
        ignore_index=True,
    )


def _sc_race(season=2024, rnd=2):
    """SC on laps 4-7; only VER pits, five cars stay out."""
    frames = [_driver("VER", stop_lap=5, sc_laps=range(4, 8), season=season, rnd=rnd)]
    frames += [_driver(d, sc_laps=range(4, 8), season=season, rnd=rnd) for d in DRIVERS[1:]]
    return pd.concat(frames, ignore_index=True)


def _losses(laps, ratios=RATIOS):
    stops = build_stops(laps)
    tracks = by_track(stops, ratios)
    return apply_losses(stops, tracks), tracks


def test_green_loss_is_measured_lap_sum_loss():
    stops, tracks = _losses(_green_race())
    assert len(stops) == len(DRIVERS)
    assert stops["pace_trusted"].all()
    expected = (2 - TRUE_PHI) * GREEN + TRANSIT - 2 * GREEN
    assert np.allclose(stops["pit_loss_s"], expected)
    assert np.isclose(tracks["green_s"].iloc[0], expected)


def test_sc_loss_uses_stated_phi_and_track_value():
    stops, tracks = _losses(pd.concat([_green_race(), _sc_race()], ignore_index=True))
    green = tracks["green_s"].iloc[0]
    assert np.isclose(tracks["sc_s"].iloc[0], green - PHI * (GREEN * 1.4 - GREEN))
    assert np.isclose(tracks["vsc_s"].iloc[0], green - PHI * (GREEN * 1.3 - GREEN))
    sc = stops[stops["condition"] == "sc"].iloc[0]
    assert np.isnan(sc["measured_loss_s"])  # not identifiable per stop
    assert np.isclose(sc["pit_loss_s"], tracks["sc_s"].iloc[0])
    assert np.isclose(sc["ref_lap_s"], SC)  # stay-out median kept as a diagnostic


def test_phi_is_a_parameter():
    stops = build_stops(_green_race())
    lo, hi = by_track(stops, RATIOS, phi=0.05), by_track(stops, RATIOS, phi=0.12)
    assert (hi["sc_s"] < lo["sc_s"]).all()
    assert np.allclose(hi["green_s"], lo["green_s"])


def test_keyed_by_season():
    laps = pd.concat([_green_race(2024), _green_race(2025, transit=21.0)], ignore_index=True)
    _, tracks = _losses(laps)
    assert tracks[["season", "event"]].duplicated().sum() == 0
    assert len(tracks) == 2
    assert tracks.set_index("season")["green_s"].diff().iloc[-1] < 0


def test_condition_ratio_from_full_condition_laps():
    laps = pd.concat([_green_race(), _sc_race()], ignore_index=True)
    ratios = condition_ratios(laps)
    assert np.isclose(ratios["sc"], SC / GREEN)


def test_pit_lane_passage_without_tyre_change_skipped():
    """Field follows the SC through the pit lane: new FastF1 stint, same tyres."""
    passage = _driver("VER", stop_lap=2, rnd=2)
    passage.loc[passage["lap"] == 3, "compound"] = "SOFT"
    passage.loc[passage["lap"] >= 3, "tyre_life"] += 2  # set continues: 2 -> 3 -> ...
    stops = build_stops(pd.concat([_green_race(), passage], ignore_index=True))
    assert not (stops["round"] == 2).any()


def test_red_flag_stop_skipped():
    df = _driver("VER", stop_lap=5, rnd=2)
    df.loc[df["lap"] == 5, "is_red"] = True
    stops = build_stops(pd.concat([_green_race(), df], ignore_index=True))
    assert not (stops["round"] == 2).any()


def test_stop_without_lap_time_kept_but_not_measured():
    """FastF1 often has no lap time on SC laps; the stop is still a stop."""
    df = _driver("VER", stop_lap=5, rnd=2)
    df.loc[df["lap"] == 6, "lap_time_s"] = np.nan
    stops = build_stops(pd.concat([_green_race(), df], ignore_index=True))
    kept = stops[stops["round"] == 2].iloc[0]
    assert np.isnan(kept["lap_sum_s"])
    assert np.isnan(kept["measured_loss_s"])


def test_garage_visit_dropped():
    """Car goes to the garage and rejoins much later: longer than a lap, not a stop."""
    df = _driver("VER", stop_lap=5, transit=2485.0, rnd=2)
    df.loc[df["lap"] == 6, "lap_time_s"] = np.nan
    stops = build_stops(pd.concat([_green_race(), df], ignore_index=True))
    assert not (stops["round"] == 2).any()


def test_sc_condition_wins_over_vsc():
    row_sc = pd.Series({"is_sc": True, "is_vsc": True})
    row_plain = pd.Series({"is_sc": False, "is_vsc": False})
    assert classify_condition(row_sc, row_plain) == "sc"
    assert classify_condition(row_plain, pd.Series({"is_sc": False, "is_vsc": True})) == "vsc"
    assert classify_condition(row_plain, row_plain) == "green"
