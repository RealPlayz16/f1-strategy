"""Lap-by-lap 20-car race engine.

GOVERNING RISK (Session 4): the optimizer built on this engine searches for the strategy that
maximises the modelled outcome, so it selects for our known biases: softs (tyre model soft
p50 understated by 0.075 s/lap), overtake-dependent strategies (pass model 15% high overall,
about 70% high at unseen tracks) and long stints (h = 30 tyre intervals are a floor). These
do not average out. Every strategy claim must survive the joint bias sweep and the
comparison with what teams actually ran (see HANDOFF.md).

Each lap, for every car, front to back in the order at the end of the previous lap:
  lap time = free-air pace + lap noise (+ soft bias on softs) (+ pit loss on an in-lap)
  within 1 s of the car ahead: + dirty_air_penalty(gap, d0)          (src/traffic.py)
  within 2 s: draw a pass, p = pass_scale * pass model               (src/overtake_model.py)
    pass:    finishes the lap ahead of that car
    no pass: cannot finish ahead; sits a held gap behind (sampled from train battles that
             lasted 3+ laps without a pass, median 0.94 s). Being held up comes from here,
             never from the dirty air curve.
  a car pitting on this lap, or ahead of a car pitting, is not constrained (stops reorder)
SC / VSC laps: every car runs the race's actual median lap time for that lap, no passing;
at the end of an SC the field closes up to SC_RESTART_GAP_S per position.
Lap 1 comes from data (start not modelled). Backmarkers are not modelled (blue flags).

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
BATTLES_PATH = Path("data/processed/battles.parquet")
PITLOSS_PATH = Path("data/processed/pitloss_by_track.parquet")
SC_PATH = Path("data/processed/sc_events.parquet")
PASS_MODEL_PATH = Path("data/models/pass_model.json")
REPLAY_PATH = Path("data/processed/engine_replay.parquet")
VALIDATION_PATH = Path("data/processed/engine_validation.json")

SEED = 42
BATTLE_GAP_S = 2.0
SC_RESTART_GAP_S = 1.0
PASS_MARGIN_S = 0.1
HELD_MIN_BATTLE_LAPS = 3
FALLBACK_QUANTILE = 0.25
FOLLOW_TARGET = {"0-0.5": 0.80, "0.5-1": 0.33, "1-1.5": 0.21, "1.5-2": 0.14, "2-3": 0.04}
GAP_BINS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
N_REPLAYS = 20


@dataclass
class Params:
    phi: float = 0.08
    d0: float = DIRTY_AIR_D0_S
    soft_bias: float = 0.0
    pass_scale: float = 1.0


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


def held_gaps() -> np.ndarray:
    b = pd.read_parquet(BATTLES_PATH)
    b = b[b["split"] == "train"].sort_values(["season", "round", "driver", "lap"])
    pair = b["driver"] + ">" + b["ahead"]
    new = b.groupby(["season", "round", pair])["lap"].diff() != 1
    run = new.groupby([b["season"], b["round"], pair]).cumsum()
    lib = b.groupby([b["season"], b["round"], pair, run]).cumcount() + 1
    return b.loc[(lib >= HELD_MIN_BATTLE_LAPS) & ~b["passed"], "gap_before_s"].to_numpy()


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

def simulate(race: Race, pace, params: Params, pass_model: PassModel, held: np.ndarray,
             noise: np.ndarray, rng: np.random.Generator,
             strategies: dict[str, dict[int, str]] | None = None) -> pd.DataFrame:
    """One race. pace(driver, lap, compound, tyre_life) -> free-air lap time.
    strategies overrides race.pits for any driver given. Returns one row per car-lap."""
    pits = {**race.pits, **(strategies or {})}
    t = dict(race.t_lap1)
    compound = {d: race.start_compound[d] for d in t}
    age = {d: race.start_age[d] for d in t}
    battle = {}
    rows = []
    for lap in range(2, race.race_laps + 1):
        active = [d for d in sorted(t, key=t.get) if race.last_lap.get(d, 0) >= lap]
        if not active:
            break
        for d in active:
            age[d] += 1
        neutral = race.neutral.get(lap)
        pitting = {d for d in active if lap in pits.get(d, {})}
        cand, free = {}, {}
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
            cand[d] = t[d] + lt

        passes = set()
        if not neutral:
            pairs = []
            for i in range(1, len(active)):
                d, a = active[i], active[i - 1]
                gap = t[d] - t[a]
                if d in pitting or a in pitting:
                    battle.pop((d, a), None)
                    continue
                cand[d] += float(dirty_air_penalty(gap, params.d0))
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
                p = params.pass_scale * pass_model.predict(b)
                draws = rng.random(len(pairs)) < p
                for (d, a, _), ok in zip(pairs, draws, strict=True):
                    if ok:
                        passes.add(d)
                        cand[d] = min(cand[d], cand[a] - PASS_MARGIN_S)
            # resolve front to back so a held car is placed behind the car's final time
            for i in range(1, len(active)):
                d, a = active[i], active[i - 1]
                if d in pitting or a in pitting or d in passes:
                    continue
                if t[d] - t[a] < BATTLE_GAP_S or cand[d] < cand[a]:
                    cand[d] = max(cand[d], cand[a] + rng.choice(held))
        elif lap in race.sc_end_laps:
            order = sorted(active, key=cand.get)
            for k, d in enumerate(order[1:], start=1):
                cand[d] = cand[order[0]] + k * SC_RESTART_GAP_S

        for d in active:
            rows.append(
                {"driver": d, "lap": lap, "t": cand[d], "lap_time": cand[d] - t[d],
                 "free_pace": free[d], "gap_prev": np.nan, "pit": d in pitting,
                 "neutral": bool(neutral), "passed": d in passes, "compound": compound[d]}
            )
            t[d] = cand[d]
            if d in pitting:
                compound[d], age[d] = pits[d][lap], 0.0
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


def finishing(sim: pd.DataFrame) -> pd.DataFrame:
    last = sim.sort_values("lap").groupby("driver").tail(1)
    return last.sort_values(["lap", "t"], ascending=[False, True]).reset_index(drop=True)


def actual_finishing(state: pd.DataFrame, season: int, rnd: int) -> pd.DataFrame:
    r = state[(state["season"] == season) & (state["round"] == rnd)]
    last = r.dropna(subset=["session_time_s"]).sort_values("lap").groupby("driver").tail(1)
    return last.sort_values(["lap", "session_time_s"], ascending=[False, True]).reset_index(
        drop=True)


def main() -> int:
    state = oracle_pace(pd.read_parquet(STATE_PATH))
    sc = pd.read_parquet(SC_PATH)
    pitloss = pd.read_parquet(PITLOSS_PATH)
    pm, held, noise = PassModel(), held_gaps(), lap_noise(state)
    actual_passes = pd.read_parquet("data/processed/overtakes.parquet")
    rng = np.random.default_rng(SEED)
    params = Params()

    pace_tab = state.set_index(["season", "round", "driver", "lap"])["pace_s"]
    curves, fin_rows = [], []
    races = state[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    for season, rnd in races.itertuples(index=False):
        race = build_race(state, sc, pitloss, season, rnd, params.phi)
        tab = pace_tab.loc[(season, rnd)]

        def pace(d, lap, _c, _a, tab=tab):
            return float(tab.get((d, lap), np.nan))

        act = actual_finishing(state, season, rnd)
        n_act = int(((actual_passes["season"] == season)
                     & (actual_passes["round"] == rnd)).sum())
        for i in range(N_REPLAYS):
            sim = simulate(race, pace, params, pm, held, noise, rng)
            sim = sim[np.isfinite(sim["t"])]
            if i < 5:
                curves.append(following_curve(sim, race))
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
                    "passes_sim": int(sim["passed"].sum()), "passes_actual": n_act,
                }
            )
        print(f"{season} {race.event}: done", flush=True)

    cur = pd.concat(curves)
    cur["bin"] = pd.cut(cur["gap_prev"], GAP_BINS + [np.inf], right=False,
                        labels=list(FOLLOW_TARGET) + [">=3"])
    curve = cur.groupby("bin", observed=True)["dev"].agg(["median", "size"])
    curve["target"] = pd.Series(FOLLOW_TARGET)
    fin = pd.DataFrame(fin_rows)
    fin.to_parquet(REPLAY_PATH, index=False)
    per_race = fin.groupby(["season", "event"]).agg(
        spearman=("spearman", "mean"), pos_err=("mean_abs_pos_err", "mean"),
        winner_correct=("winner_correct", "mean"), gap_err_s=("median_abs_gap_err_s", "median"),
        passes_sim=("passes_sim", "mean"), passes_actual=("passes_actual", "first"),
    )
    res = {"following_curve": curve.reset_index().astype(str).to_dict(orient="records"),
           "per_race": per_race.reset_index().round(3).astype(str).to_dict(orient="records")}
    VALIDATION_PATH.write_text(json.dumps(res, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("\nfollowing curve, simulated (oracle pace) vs Session 3 data target:")
    print(curve.round(3).to_string())
    print("\nreplay of actual strategies, mean over replays:")
    print(per_race.round(3).to_string())
    print("\nall races:", per_race[["spearman", "pos_err", "winner_correct", "gap_err_s",
                                    "passes_sim", "passes_actual"]].mean().round(3).to_dict())
    print(f"\nwrote {REPLAY_PATH} and {VALIDATION_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
