"""Lap-by-lap 20-car race engine.

GOVERNING RISK (Session 4): the optimizer built on this engine searches for the strategy that
maximises the modelled outcome, so it selects for our known biases: softs (tyre model soft
p50 understated by 0.075 s/lap), overtake-dependent strategies (pass model 15% high overall,
about 70% high at unseen tracks) and long stints (h = 30 tyre intervals are a floor). These
do not average out. Every strategy claim must survive the joint bias sweep and the
comparison with what teams actually ran (see HANDOFF.md).

FOLLOWING MODEL (Session 7). What holds a car back is a pace constraint, not a position, so
there is no minimum gap anywhere in this engine. For car d with the nearest non-pitting car a
ahead and gap g at the end of the previous lap:

  unconstrained   L0_d = free pace + lap noise (+ soft bias) + dirty_air_penalty(g, d0)
  held iff        t_d + L0_d < arrival of a   (it would finish ahead of a car it did not pass)
  held            L_d = L_a + dirty_air_penalty(g, d0),  L_a = a's final lap time this lap

Held cars therefore end the lap g + aero(g) behind, and L_d > L0_d always (held implies
L0_d < L_a - g < L_a), so the rule can only ever slow a car down. A car settles where its
pace in hand is cancelled by the aero penalty, aero(g*) = delta; a car with more than d0 in
hand has no equilibrium and runs right up behind. Trains form because L_a already carries
a's own held time. Earlier versions set the gap as a position rule and failed validation
twice (sampled 0.94 s floor, then a 0.2 s constant); see HANDOFF.md.

  dirty_air_penalty is the AERO-ONLY curve (src/traffic.py). Being held up is produced here,
  mechanically, and must never be taken from the pooled curve as well.

Each lap, every car, front to back in the order at the end of the previous lap:
  lap time = free-air pace + lap noise (+ soft bias on softs) (+ pit loss on an in-lap)
  + dirty_air_penalty(gap to the reference car, d0)
  within 2 s: draw a pass, p = pass_scale * pass model               (src/overtake_model.py)
    pass:    the PASSED car is pushed to the passer's arrival + PASS_MARGIN_S. The passer
             keeps its own lap time and gains nothing it did not run (ASSUMPTION about who
             pays the swap; the pass model is already 15% high, so the bias is taken in the
             direction that does not compound it). The passer is then re-tested against the
             car that was ahead of the one it passed, so it cannot clear a third car with
             no draw.
    no pass: the held rule above
  a car pitting on this lap is unconstrained (the stop reorders); a car behind it references
  the nearest non-pitting car ahead, as battles.parquet does.
SC / VSC laps: every car runs the race's actual median lap time for that lap, no passing; at
the end of an SC the field closes up to SC_RESTART_GAP_S per position. That gap is MEASURED:
median gap to the car ahead on the last lap of each train SC, 0.41 s over 11 events and 173
gaps (per-race 0.28 to 0.54). It is a starting gap on a neutral lap, and the validation curve
excludes neutral laps, so it does not feed its own target.
Lap 1 comes from data (start not modelled). Backmarkers are not modelled (blue flags).

VALIDATION TARGET (changed in Session 7). The primary target is now the BATTLE-LAP
DISTRIBUTION by gap bin: car-laps per race spent within BATTLE_GAP_S of the eligible car
directly ahead, 415 per race on train races, split 63.2 / 162.5 / 102.3 / 87.0 over
0-0.5 / 0.5-1 / 1-1.5 / 1.5-2 s. It is measured from battles.parquet and NOTHING in the
following model was fitted to it: d0 and the aero shape come from conditional mean lap times,
free-air pace from clear-air laps, the pass model from pass outcomes. It is also a
distribution over four bins rather than five conditional means, so it constrains where cars
spend their time, which is what drives pass counts and traffic cost.
  Measured exactly as overtakes.py measured it (see eligible_sim): ineligible cars removed
  from the ordering, lap 1 and neutral laps on lap k dropped, the in-lap and out-lap dropped,
  and a lap k-1 under a neutralisation KEPT, so restart laps count on both sides.
  passes_sim is likewise scored as overtakes.py scores it (any swap in lap-end order between
  eligible cars), not as the count of drawn passes; passes_drawn is kept beside it.

  DEMOTED to a sanity check in the same session: the following curve (lap time minus free-air
  pace by gap bin, target 0.80 / 0.33 / 0.21 / 0.14 / 0.04). Under the pace rule an unheld
  car's dev is exactly dirty_air_penalty(gap), which is zero past 1.0 s by construction, so
  the 1.5-2 and 2-3 bins read 0.000 and CANNOT FAIL. A near-match there is worth nothing.
  The bins that can still move take d0, the aero shape and free-air pace from the same
  following laps that produce the target, so none of the curve is out of sample. Keep
  printing it, do not treat it as a test.

Pace: a function (driver, lap, compound, tyre_life) -> free-air lap time in absolute seconds.
The validation replay uses each car's measured free-air pace (oracle, actual strategy),
which tests the engine mechanics apart from tyre model error. Lap noise is sampled from
free-air laps' deviation from free-air pace (train, heavy tailed: sd 0.52, robust 0.27).

Usage:
    python -m src.engine            (validation replay on train races)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.overtake_model import features as pass_features
from src.rules import drs_enabled
from src.traffic import DIRTY_AIR_D0_S, dirty_air_penalty

STATE_PATH = Path("data/processed/traffic_laps.parquet")
PITLOSS_PATH = Path("data/processed/pitloss_by_track.parquet")
SC_PATH = Path("data/processed/sc_events.parquet")
PASS_MODEL_PATH = Path("data/models/pass_model.json")
REPLAY_PATH = Path("data/processed/engine_replay.parquet")
VALIDATION_PATH = Path("data/processed/engine_validation.json")
FOLLOWING_PATH = Path("data/processed/engine_following.parquet")
BATTLES_SIM_PATH = Path("data/processed/engine_battles.parquet")
PAIRS_SIM_PATH = Path("data/processed/engine_pairs.parquet")
PASS_OOF_PATH = Path("data/processed/pass_oof.parquet")

SEED = 42
BATTLE_GAP_S = 2.0
# Measured: median gap to the car ahead on the last lap of each train SC (11 events, 173
# gaps, per-race medians 0.28 to 0.54). Replaces a stated 1.0 s. Neutral-lap population,
# which the green-flag following curve excludes.
SC_RESTART_GAP_S = 0.41
PASS_MARGIN_S = 0.1
FALLBACK_QUANTILE = 0.25
# PRIMARY VALIDATION TARGET (Session 7): battle laps per race by gap bin, measured from
# battles.parquet. Nothing in the following model was fitted to this distribution, and it
# is a distribution rather than five conditional means. See the docstring.
BATTLE_BINS = [0.0, 0.5, 1.0, 1.5, 2.0]
BATTLE_LABELS = ["0-0.5", "0.5-1", "1-1.5", "1.5-2"]
# SANITY CHECK ONLY, demoted in Session 7: the pace rule makes an unheld car's dev exactly
# aero(gap), which is zero past 1.0 s by construction, so every bin past 1.0 s reads 0.000
# and cannot fail. The remaining bins take d0, the aero shape and free-air pace from the
# same following laps that produce the target.
FOLLOW_TARGET = {"0-0.5": 0.80, "0.5-1": 0.33, "1-1.5": 0.21, "1.5-2": 0.14, "2-3": 0.04}
GAP_BINS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
N_REPLAYS = 20


@dataclass
class Params:
    phi: float = 0.08
    d0: float = DIRTY_AIR_D0_S
    soft_bias: float = 0.0
    pass_scale: float = 1.0
    restart_gap: float = SC_RESTART_GAP_S


@dataclass
class Race:
    """Everything the engine needs for one race."""
    season: int
    round: int
    event: str
    race_laps: int
    t_lap1: dict[str, float]               # session time at end of lap 1
    last_lap: dict[str, int]               # laps each car completes (retired or lapped)
    pits: dict[str, dict[int, str]]        # in-lap -> compound fitted
    start_compound: dict[str, str]
    start_age: dict[str, float]
    neutral: dict[int, tuple[str, float]]  # lap -> (SC / VSC, lap time)
    sc_end_laps: set[int]
    pit_loss: dict[str, float]             # green / sc / vsc
    covered: dict[tuple[str, int], bool] = field(default_factory=dict)


# ---------- shared inputs ----------

class PassModel:
    """Logistic pass model from data/models/pass_model.json."""

    def __init__(self, path: Path = PASS_MODEL_PATH):
        m = json.loads(path.read_text(encoding="utf-8"))
        self.columns, self.coef = m["columns"], np.array(m["coef"])
        self.mean, self.std, self.events = pd.Series(m["mean"]), pd.Series(m["std"]), m["events"]

    def predict(self, b: pd.DataFrame) -> np.ndarray:
        x = pass_features(b, self.events)[self.columns]
        x[self.mean.index] = (x[self.mean.index] - self.mean) / self.std
        return 1.0 / (1.0 + np.exp(-(x.to_numpy() @ self.coef)))


def lap_noise(state: pd.DataFrame) -> np.ndarray:
    free = state["is_clean"] & (state["gap_prev"] >= 3.0) & state["dev"].notna()
    return state.loc[free, "dev"].to_numpy()


def oracle_pace(state: pd.DataFrame) -> pd.DataFrame:
    """Free-air lap time per (race, driver, lap) on the actual strategy, absolute seconds.
    Stints with no free-air lap fall back to the FALLBACK_QUANTILE of their age-adjusted
    clean laps (flagged covered = False)."""
    s = state.copy()
    clean = s[s["is_clean"]]
    fb = clean.groupby(["race", "driver", "stint"])["age_adj"].quantile(
        FALLBACK_QUANTILE).rename("fb_base")
    # Stints with no clean lap at all (e.g. lap-1 SC stints): the driver's clean laps in the
    # race, then the field's
    fb_drv = clean.groupby(["race", "driver"])["age_adj"].quantile(FALLBACK_QUANTILE).rename(
        "fb_driver")
    fb_race = clean.groupby("race")["age_adj"].quantile(FALLBACK_QUANTILE).rename("fb_race")
    s = s.join(fb, on=["race", "driver", "stint"]).join(fb_drv, on=["race", "driver"])
    s = s.join(fb_race, on="race")
    s["covered"] = s["free_pace"].notna()
    base = s["fb_base"].fillna(s["fb_driver"]).fillna(s["fb_race"])
    fc = s["free_pace"].fillna(base + s["deg"] * s["tyre_life"])
    s["pace_s"] = fc + s["progress_s"]
    return s


def build_race(state: pd.DataFrame, sc: pd.DataFrame, pitloss: pd.DataFrame,
               season: int, rnd: int, phi: float) -> Race:
    r = state[(state["season"] == season) & (state["round"] == rnd)].sort_values(["driver", "lap"])
    event = str(r["event"].iloc[0])
    race_laps = int(r["race_laps"].iloc[0])
    lap1 = r[r["lap"] == 1].dropna(subset=["session_time_s"])
    t_lap1 = dict(zip(lap1["driver"], lap1["session_time_s"], strict=True))
    last = r.groupby("driver")["lap"].max().to_dict()
    pits, start_c, start_a = {}, {}, {}
    for drv, d in r.groupby("driver"):
        d = d.set_index("lap")
        start_c[drv] = str(d["compound"].iloc[0])
        start_a[drv] = float(d["tyre_life"].iloc[0])
        stops = {}
        for lap in d.index[d["is_pit_in"]]:
            if lap + 1 in d.index:
                stops[int(lap)] = str(d.loc[lap + 1, "compound"])
        pits[drv] = stops
    neutral = {}
    ev = sc[(sc["season"] == season) & (sc["round"] == rnd) & sc["kind"].isin(["SC", "VSC"])]
    running = r[~r["is_pit_in"] & ~r["is_pit_out"] & r["lap_time_s"].notna()]
    med = running.groupby("lap")["lap_time_s"].median()
    sc_end = set()
    for _, e in ev.iterrows():
        for lap in range(int(e["lap_deploy"]), int(e["lap_end"]) + 1):
            if lap in med.index and lap > 1:
                neutral[lap] = (e["kind"], float(med[lap]))
        if e["kind"] == "SC":
            sc_end.add(int(e["lap_end"]))
    row = pitloss[(pitloss["season"] == season) & (pitloss["event"] == event)].iloc[0]
    loss = {"green": float(row["green_s"])}
    for c in ("sc", "vsc"):
        loss[c] = float(row["green_s"] - phi * (row[f"lap_{c}_s"] - row["lap_green_s"]))
    covered = {(d, int(lap)): bool(c)
               for d, lap, c in zip(r["driver"], r["lap"], r["covered"], strict=True)}
    return Race(season, rnd, event, race_laps, t_lap1, last, pits, start_c, start_a, neutral,
                sc_end, loss, covered)


# ---------- simulation ----------

def simulate(race: Race, pace, params: Params, pass_model: PassModel,
             noise: np.ndarray, rng: np.random.Generator,
             strategies: dict[str, dict[int, str]] | None = None,
             pair_log: list | None = None) -> pd.DataFrame:
    """One race. pace(driver, lap, compound, tyre_life) -> free-air lap time.
    strategies overrides race.pits for any driver given. Returns one row per car-lap.
    pair_log, if given, collects one row per battle pair drawn against (gap, free-air
    pace delta, predicted pass probability), for the population comparison against
    battles.parquet. Diagnostic only; it does not affect the simulation."""
    pits = {**race.pits, **(strategies or {})}
    t = dict(race.t_lap1)
    compound = {d: race.start_compound[d] for d in t}
    age = {d: race.start_age[d] for d in t}
    battle = {}
    rows = []
    prev_pitting: set[str] = set()
    for lap in range(2, race.race_laps + 1):
        active = [d for d in sorted(t, key=t.get) if race.last_lap.get(d, 0) >= lap]
        if not active:
            break
        for d in active:
            age[d] += 1
        neutral = race.neutral.get(lap)
        pitting = {d for d in active if lap in pits.get(d, {})}

        # base lap time: own pace, before dirty air and before the held rule
        base, free = {}, {}
        for d in active:
            free[d] = pace(d, lap, compound[d], age[d])
            if not neutral and not np.isfinite(free[d]):
                raise ValueError(f"no pace for {d} lap {lap} in {race.season} {race.event}")
            if neutral:
                lt = neutral[1]
            else:
                lt = free[d] + rng.choice(noise)
                if compound[d] == "SOFT":
                    lt += params.soft_bias
            if d in pitting:
                lt += race.pit_loss["green" if not neutral else neutral[0].lower()]
            base[d] = lt

        passes = set()
        arr, lt_final = {}, {}
        if neutral:
            for d in active:
                arr[d], lt_final[d] = t[d] + base[d], base[d]
            if lap in race.sc_end_laps:
                order = sorted(active, key=arr.get)
                for k, d in enumerate(order[1:], start=1):
                    arr[d] = arr[order[0]] + k * params.restart_gap
                    lt_final[d] = arr[d] - t[d]
        else:
            # reference car: the nearest car ahead that is not pitting this lap, as
            # battles.parquet defines "the eligible car directly ahead". A pitting car is
            # unconstrained and is not a reference (the stop reorders).
            ref, prev_ok = {}, None
            for d in active:
                if d in pitting:
                    continue
                ref[d] = prev_ok
                prev_ok = d

            pairs = []
            for d in active:
                a = ref.get(d)
                if a is None:
                    continue
                gap = t[d] - t[a]
                base[d] += float(dirty_air_penalty(gap, params.d0))
                if gap < BATTLE_GAP_S:
                    battle[(d, a)] = battle.get((d, a), 0) + 1
                    pairs.append((d, a, gap))
                else:
                    battle.pop((d, a), None)
            if pairs:
                b = pd.DataFrame(
                    {
                        "gap_before_s": [g for _, _, g in pairs],
                        "free_delta": [free[a] - free[d] for d, a, _ in pairs],
                        "tyre_age_delta": [age[a] - age[d] for d, a, _ in pairs],
                        "compound_pair": [f"{compound[d]}-{compound[a]}" for d, a, _ in pairs],
                        "laps_in_battle": [battle[(d, a)] for d, a, _ in pairs],
                        "drs_available": [float(drs_enabled(lap, None, g)) for _, _, g in pairs],
                        "event": race.event,
                    }
                )
                p_pass = params.pass_scale * pass_model.predict(b)
                draws = rng.random(len(pairs)) < p_pass
                passes = {d for (d, _, _), ok in zip(pairs, draws, strict=True) if ok}
                if pair_log is not None:
                    for (dd, aa, gg), pp, ok in zip(pairs, p_pass, draws, strict=True):
                        pair_log.append(
                            {"lap": lap, "gap": gg, "free_delta": free[aa] - free[dd],
                             "p_pass": float(pp), "passed": bool(ok),
                             "elig": dd not in prev_pitting and aa not in prev_pitting}
                        )

            # Front to back, so every reference lap time is already final. A car cannot
            # finish ahead of ANY car that was ahead of it and that it did not pass, so the
            # binding car is the resolved car with the latest arrival, not always the nearest
            # one: a car pushed back by a pass can end up blocking the cars behind it.
            cmax = []
            for d in active:
                a = ref.get(d)
                if a is None:
                    arr[d], lt_final[d] = t[d] + base[d], base[d]
                    if d not in pitting:
                        cmax.append(d)
                    continue
                arr[d], lt_final[d] = t[d] + base[d], base[d]
                block = cmax[-2] if (d in passes and len(cmax) >= 2) else cmax[-1]
                if block is not None and arr[d] < arr[block]:
                    # held: cannot use its pace advantage, so it runs the blocking car's lap
                    # time plus the aero penalty. Ends the lap gap + aero(gap) behind.
                    lt_final[d] = lt_final[block] + float(
                        dirty_air_penalty(t[d] - t[block], params.d0))
                    arr[d] = t[d] + lt_final[d]
                if d in passes and arr[a] < arr[d] + PASS_MARGIN_S:
                    # the passer keeps its own lap time; the passed car pays the swap, so it
                    # must end the lap at least PASS_MARGIN_S behind the passer
                    arr[a] = arr[d] + PASS_MARGIN_S
                    lt_final[a] = arr[a] - t[a]
                    if arr[a] > arr[cmax[-1]]:
                        cmax[-1] = a
                cmax.append(d if arr[d] >= arr[cmax[-1]] else cmax[-1])

        for d in active:
            rows.append(
                {"driver": d, "lap": lap, "t": arr[d], "lap_time": lt_final[d],
                 "free_pace": free[d], "gap_prev": np.nan, "pit": d in pitting,
                 "neutral": bool(neutral), "passed": d in passes, "compound": compound[d]}
            )
            t[d] = arr[d]
            if d in pitting:
                compound[d], age[d] = pits[d][lap], 0.0
        prev_pitting = pitting
    out = pd.DataFrame(rows)
    return add_order(out)


def add_order(sim: pd.DataFrame) -> pd.DataFrame:
    """Car ahead and gap at the end of each lap, and at the end of the previous lap."""
    sim = sim.sort_values(["lap", "t"]).copy()
    sim["ahead_now"] = sim.groupby("lap")["driver"].shift(1)
    sim["gap_now"] = sim["t"] - sim.groupby("lap")["t"].shift(1)
    prev = sim[["driver", "lap", "ahead_now", "gap_now"]].copy()
    prev["lap"] += 1
    prev = prev.rename(columns={"ahead_now": "ahead_prev", "gap_now": "gap_prev"})
    sim = sim.drop(columns=["gap_prev"]).merge(prev, on=["driver", "lap"], how="left")
    sim.loc[sim["ahead_prev"].isna() & sim["lap"].gt(2), "gap_prev"] = np.inf
    return sim


# ---------- validation ----------

def following_curve(sim: pd.DataFrame, race: Race) -> pd.DataFrame:
    """Same statistic as Session 3's pooled curve, on simulated laps, covered cars only."""
    s = sim[~sim["pit"] & ~sim["neutral"] & sim["gap_prev"].notna()].copy()
    pit_laps = set(zip(sim.loc[sim["pit"], "driver"], sim.loc[sim["pit"], "lap"], strict=True))
    keep = np.array([(a, lap) not in pit_laps
                     for a, lap in zip(s["ahead_prev"], s["lap"], strict=True)], dtype=bool)
    s = s[keep]
    keep = np.array([race.covered.get((d, lap), False)
                     for d, lap in zip(s["driver"], s["lap"], strict=True)], dtype=bool)
    s = s[keep]
    s = s[(s["ahead_now"] == s["ahead_prev"]) | (s["gap_prev"] >= 3.0)]
    s["dev"] = s["lap_time"] - s["free_pace"]
    return s[["gap_prev", "dev"]]


def eligible_sim(sim: pd.DataFrame) -> pd.DataFrame:
    """Reproduce overtakes.eligible_pairs on simulated laps, so the battle-lap count and the
    pass count are measured exactly as battles.parquet and overtakes.parquet were.
    One row per (driver, lap k) with cumulative time at the end of k-1 and of k. Drops lap 1,
    laps under SC / VSC / red, the in-lap and the out-lap. Note that a lap k-1 under a
    neutralisation is NOT dropped (overtakes.py flags lap k only), so restart laps count."""
    s = sim.sort_values(["driver", "lap"]).copy()
    g = s.groupby("driver")
    s["t_prev"] = g["t"].shift(1)
    s["lap_prev"] = g["lap"].shift(1)
    s["pit_prev"] = g["pit"].shift(1)
    ok = (
        (s["lap"] >= 2)
        & (s["lap_prev"] == s["lap"] - 1)
        & s["t_prev"].notna()
        & ~s["neutral"]
        & ~s["pit"]
        & ~s["pit_prev"].astype("boolean").fillna(True)
    )
    return s.loc[ok, ["driver", "lap", "t_prev", "t"]]


def battle_gaps(sim: pd.DataFrame) -> np.ndarray:
    """Gap to the eligible car directly ahead at the end of lap k-1, under BATTLE_GAP_S.
    Ineligible cars are removed from the ordering first, as overtakes.py does."""
    out = []
    for _, d in eligible_sim(sim).groupby("lap"):
        gaps = np.diff(np.sort(d["t_prev"].to_numpy()))
        out.append(gaps[gaps < BATTLE_GAP_S])
    return np.concatenate(out) if out else np.array([])


def scored_passes(sim: pd.DataFrame) -> int:
    """Passes as overtakes.py scores them: any swap in lap-end order between eligible cars,
    not only adjacent pairs and not only drawn ones. This is what 35.1 per race counts."""
    n = 0
    for _, d in eligible_sim(sim).groupby("lap"):
        tp, tc = d["t_prev"].to_numpy(), d["t"].to_numpy()
        n += int(((tp[:, None] > tp[None, :]) & (tc[:, None] < tc[None, :])).sum())
    return n


def battle_target(battles: pd.DataFrame, n_races: int) -> pd.Series:
    """Actual battle laps per race by gap bin, train only. Measured, never fitted."""
    b = battles[battles["split"] == "train"]
    binned = pd.cut(b["gap_before_s"], BATTLE_BINS, right=False, labels=BATTLE_LABELS)
    return (b.groupby(binned, observed=True).size() / n_races).rename("target")


def finishing(sim: pd.DataFrame) -> pd.DataFrame:
    last = sim.sort_values("lap").groupby("driver").tail(1)
    return last.sort_values(["lap", "t"], ascending=[False, True]).reset_index(drop=True)


def actual_finishing(state: pd.DataFrame, season: int, rnd: int) -> pd.DataFrame:
    r = state[(state["season"] == season) & (state["round"] == rnd)]
    last = r.dropna(subset=["session_time_s"]).sort_values("lap").groupby("driver").tail(1)
    return last.sort_values(["lap", "session_time_s"], ascending=[False, True]).reset_index(
        drop=True)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    params = Params()
    tag = ""
    if "--restart-gap" in argv:                 # isolation run, see HANDOFF Session 7
        params.restart_gap = float(argv[argv.index("--restart-gap") + 1])
        tag = f"_restart{params.restart_gap:g}"
        print(f"restart gap override: {params.restart_gap} s")

    state = oracle_pace(pd.read_parquet(STATE_PATH))
    sc = pd.read_parquet(SC_PATH)
    pitloss = pd.read_parquet(PITLOSS_PATH)
    pm, noise = PassModel(), lap_noise(state)
    actual_passes = pd.read_parquet("data/processed/overtakes.parquet")
    battles = pd.read_parquet("data/processed/battles.parquet")
    rng = np.random.default_rng(SEED)

    pace_tab = state.set_index(["season", "round", "driver", "lap"])["pace_s"]
    curves, fin_rows, bgaps, plogs = [], [], [], []
    races = state[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    n_races = len(races)
    for season, rnd in races.itertuples(index=False):
        race = build_race(state, sc, pitloss, season, rnd, params.phi)
        tab = pace_tab.loc[(season, rnd)]

        def pace(d, lap, _c, _a, tab=tab):
            return float(tab.get((d, lap), np.nan))

        act = actual_finishing(state, season, rnd)
        n_act = int(((actual_passes["season"] == season)
                     & (actual_passes["round"] == rnd)).sum())
        for i in range(N_REPLAYS):
            plog = [] if i < 5 else None
            sim = simulate(race, pace, params, pm, noise, rng, pair_log=plog)
            sim = sim[np.isfinite(sim["t"])]
            if i < 5:
                c = following_curve(sim, race)
                c["season"], c["event"], c["replay"] = season, race.event, i
                curves.append(c)
            if plog is not None:
                pl = pd.DataFrame(plog)
                pl["season"], pl["event"], pl["replay"] = season, race.event, i
                plogs.append(pl)
            bg = battle_gaps(sim)
            bgaps.append(pd.DataFrame({"gap": bg, "season": season, "event": race.event,
                                       "replay": i}))
            fin = finishing(sim)
            pos_sim = {d: k for k, d in enumerate(fin["driver"])}
            common = [d for d in act["driver"] if d in pos_sim]
            ranks_a = np.arange(len(common))
            ranks_s = np.array([pos_sim[d] for d in common])
            winner_t = fin["t"].iloc[0]
            gaps_s = fin.set_index("driver")["t"] - winner_t
            gaps_a = act.set_index("driver")["session_time_s"] - act["session_time_s"].iloc[0]
            same_lap = [d for d in common if fin.set_index("driver").loc[d, "lap"]
                        == race.race_laps and act.set_index("driver").loc[d, "lap"]
                        == race.race_laps]
            fin_rows.append(
                {
                    "season": season, "round": rnd, "event": race.event, "replay": i,
                    "spearman": float(pd.Series(ranks_a).corr(pd.Series(ranks_s),
                                                              method="spearman")),
                    "mean_abs_pos_err": float(np.abs(ranks_a - np.argsort(np.argsort(
                        ranks_s))).mean()),
                    "winner_correct": bool(fin["driver"].iloc[0] == act["driver"].iloc[0]),
                    "median_abs_gap_err_s": float(np.median(np.abs(
                        gaps_s[same_lap] - gaps_a[same_lap]))) if same_lap else np.nan,
                    "passes_drawn": int(sim["passed"].sum()),
                    "passes_sim": scored_passes(sim), "passes_actual": n_act,
                    "battle_laps_scored": int(len(bg)),
                    "battle_laps": int(((sim["gap_prev"] < BATTLE_GAP_S) & ~sim["neutral"]
                                         & ~sim["pit"]).sum()),
                }
            )
        print(f"{season} {race.event}: done", flush=True)

    cur = pd.concat(curves)
    cur["bin"] = pd.cut(cur["gap_prev"], GAP_BINS + [np.inf], right=False,
                        labels=list(FOLLOW_TARGET) + [">=3"])
    curve = cur.groupby("bin", observed=True)["dev"].agg(["median", "size"])
    curve["target"] = pd.Series(FOLLOW_TARGET)
    # Shape, not just the level: the 0-0.5 median is a prediction about which cars end up
    # there (dev = their pace in hand, for cars with more than d0 in hand and so no
    # equilibrium), not a fitted quantity. A right-skewed spread is the model working; a
    # spike at d0 would mean the median is right by accident.
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    shape = cur.groupby("bin", observed=True)["dev"].describe(percentiles=qs)
    cur.to_parquet(FOLLOWING_PATH.with_name(f"{FOLLOWING_PATH.stem}{tag}.parquet"), index=False)
    fin = pd.DataFrame(fin_rows)
    fin.to_parquet(REPLAY_PATH.with_name(f"{REPLAY_PATH.stem}{tag}.parquet"), index=False)
    per_race = fin.groupby(["season", "event"]).agg(
        spearman=("spearman", "mean"), pos_err=("mean_abs_pos_err", "mean"),
        winner_correct=("winner_correct", "mean"), gap_err_s=("median_abs_gap_err_s", "median"),
        passes_sim=("passes_sim", "mean"), passes_drawn=("passes_drawn", "mean"),
        passes_actual=("passes_actual", "first"),
        battle_laps=("battle_laps_scored", "mean"),
    )
    bg = pd.concat(bgaps)
    bg["bin"] = pd.cut(bg["gap"], BATTLE_BINS, right=False, labels=BATTLE_LABELS)
    per_replay = bg.groupby(["season", "event", "replay", "bin"], observed=True).size()
    dist = per_replay.groupby("bin", observed=True).mean().rename("sim_per_race")
    dist = pd.concat([dist, battle_target(battles, n_races)], axis=1)
    dist["ratio"] = (dist["sim_per_race"] / dist["target"]).round(2)
    bg.to_parquet(BATTLES_SIM_PATH.with_name(f"{BATTLES_SIM_PATH.stem}{tag}.parquet"),
                  index=False)

    res = {"restart_gap_s": params.restart_gap,
           "battle_distribution": dist.reset_index().round(3).astype(str).to_dict(
               orient="records"),
           "following_curve": curve.reset_index().astype(str).to_dict(orient="records"),
           "following_shape": shape.reset_index().round(3).astype(str).to_dict(
               orient="records"),
           "per_race": per_race.reset_index().round(3).astype(str).to_dict(orient="records")}
    VALIDATION_PATH.with_name(f"{VALIDATION_PATH.stem}{tag}.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")

    pl = pd.concat(plogs)
    pl = pl[pl["elig"]]
    pl["bin"] = pd.cut(pl["gap"], BATTLE_BINS, right=False, labels=BATTLE_LABELS)
    pl = pl.dropna(subset=["bin"])
    pop = pl.groupby("bin", observed=True).agg(
        n=("free_delta", "size"),
        delta_p25=("free_delta", lambda x: x.quantile(0.25)),
        delta_med=("free_delta", "median"),
        delta_p75=("free_delta", lambda x: x.quantile(0.75)),
        share_not_faster=("free_delta", lambda x: float((x <= 0).mean())),
        mean_p=("p_pass", "mean"),
    )
    pop["laps_per_race"] = (pop["n"] / 5 / n_races).round(1)
    act = battles[battles["split"] == "train"].copy()
    act["bin"] = pd.cut(act["gap_before_s"], BATTLE_BINS, right=False, labels=BATTLE_LABELS)
    act = act.dropna(subset=["bin"])
    act_rate = act.groupby("bin", observed=True)["passed"].mean().rename("actual_rate")
    # The pass model has a covered branch (free-air pace known) and a missing-flag branch.
    # The engine is ALWAYS on the covered branch: it always has a pace estimate. Reality is
    # covered for about half of real battles, and the two populations pass at very different
    # rates, so the covered rate is reported beside the blended one to size that mismatch.
    oof = pd.read_parquet(PASS_OOF_PATH)
    oof = oof[oof["split"] == "train"].copy()
    oof["bin"] = pd.cut(oof["gap_before_s"], BATTLE_BINS, right=False, labels=BATTLE_LABELS)
    oof = oof.dropna(subset=["bin"])
    grp = oof.groupby("bin", observed=True)
    cov_rate = grp.apply(
        lambda d: d.loc[d["free_delta"].notna(), "passed"].mean(), include_groups=False
    ).rename("covered_rate")
    coverage = grp["free_delta"].apply(lambda x: float(x.notna().mean())).rename("coverage")
    pop = pop.join(act_rate).join(cov_rate).join(coverage)
    pop["vs_blend"] = (pop["mean_p"] / pop["actual_rate"]).round(2)
    pop["vs_covered"] = (pop["mean_p"] / pop["covered_rate"]).round(2)
    pop["branch_mismatch"] = (pop["covered_rate"] / pop["actual_rate"]).round(2)
    res["pair_population"] = pop.reset_index().round(3).astype(str).to_dict(orient="records")
    pl.to_parquet(PAIRS_SIM_PATH.with_name(f"{PAIRS_SIM_PATH.stem}{tag}.parquet"), index=False)
    VALIDATION_PATH.with_name(f"{VALIDATION_PATH.stem}{tag}.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")

    pd.set_option("display.width", 220)
    print("\nPAIR POPULATION: simulated battle pairs vs battles.parquet, by gap bin")
    print(pop.round(3).to_string())
    print("  vs_blend splits into branch_mismatch (the engine is always on the covered")
    print("  branch) times vs_covered (how much the simulated pair population differs).")
    # Decomposition of the pass miss: how much is too many battle laps, and how much is the
    # wrong population within those laps.
    sim_p = (pop["laps_per_race"] * pop["mean_p"]).sum()
    counterfactual = (pop["laps_per_race"] * pop["actual_rate"]).sum()
    branch = (pop["laps_per_race"] * pop["covered_rate"]).sum()
    act_laps = (act.groupby("bin", observed=True).size() / n_races)
    actual_p = (act_laps * pop["actual_rate"]).sum()
    print("\n  pass decomposition (adjacent pairs, drawn):")
    print(f"    actual                      {actual_p:5.1f}")
    print(f"    sim laps, blended rate      {counterfactual:5.1f}   too many battle laps")
    print(f"    sim laps, covered rate      {branch:5.1f}   + always on the covered branch")
    print(f"    sim laps, sim rate          {sim_p:5.1f}   + simulated pair population")

    print("\nPRIMARY TARGET: battle laps per race by gap bin (out of sample):")
    print(dist.round(2).to_string())
    print(f"  total  sim {dist['sim_per_race'].sum():.1f}  "
          f"actual {dist['target'].sum():.1f}")
    print("\nSANITY CHECK ONLY (demoted, see module docstring): following curve")
    print(curve.round(3).to_string())
    print("\ndev distribution by bin (shape test, not just the level):")
    print(shape.round(3).to_string())
    print("\nreplay of actual strategies, mean over replays:")
    print(per_race.round(3).to_string())
    print("\nall races:", per_race[["spearman", "pos_err", "winner_correct", "gap_err_s",
                                    "passes_sim", "passes_drawn", "passes_actual",
                                    "battle_laps"]].mean().round(3).to_dict())
    print(f"\nwrote {FOLLOWING_PATH} and {VALIDATION_PATH} (tag {tag!r})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
