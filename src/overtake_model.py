"""Pass probability per lap for a car within 2 s of the car ahead. Train races only.

P(pass on lap k | state at end of lap k-1), logistic regression:
  gap bins (flexible in gap), DRS available (gap < 1 s, lap >= 3, not within 2 laps of an SC
  restart), free-air pace delta (src/traffic.py) with a missing flag, tyre age delta,
  compound step (overtaker softer = positive), track (event dummies, L2 shrunk toward the
  global rate: 1 to 3 races per track), log of laps already in this battle.

Identification (Session 3, see HANDOFF.md):
- Observed recent-lap pace delta is censored by dirty air (0.22 s vs 0.62 s free air within
  0.5 s). The model uses free-air pace delta, which the race engine knows. It exists for 52%
  of battles (70% of passes); the rest are stints with no clear-air lap, skewed to stuck
  pairs, and get the missing flag. Calibration is reported separately on covered rows,
  which are what the engine will query.
- Free-air pace delta and tyre age delta correlate 0.56: predictions are fine, separate
  coefficients are not interpretable.
- Laps in battle (pass rate 20% -> 5%) stands in for unobserved pair difficulty; the
  engine applies the average decay to every pair.
- A pass and re-pass within one lap is invisible (lap-end order).

Validation: GroupKFold by race, never row splits. Calibration buckets are the main output.
No post-processing of probabilities.

Usage:
    python -m src.overtake_model
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

BATTLES_PATH = Path("data/processed/battles.parquet")
STATE_PATH = Path("data/processed/traffic_laps.parquet")
SC_PATH = Path("data/processed/sc_events.parquet")
OOF_PATH = Path("data/processed/pass_oof.parquet")
REPORT_PATH = Path("data/processed/pass_report.json")
MODEL_PATH = Path("data/models/pass_model.json")

SEED = 42
N_FOLDS = 5
C = 1.0
GAP_EDGES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
PROB_BUCKETS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
HARDNESS = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
NUMERIC = ["free_delta_f", "free_missing", "tyre_age_delta", "compound_step",
           "log_laps_in_battle", "drs_available"]


def load() -> pd.DataFrame:
    b = pd.read_parquet(BATTLES_PATH)
    b = b[b["split"] == "train"].copy()
    b["race"] = b["season"].astype(str) + "_" + b["round"].astype(str).str.zfill(2)
    # CAUSAL free pace (Session 9). free_pace is the median over the WHOLE stint's clear-air
    # laps, so fitting on it estimates the covered branch from a population defined with
    # information the engine does not have at decision time, and the engine then always takes
    # that branch. Both the FLAG (is it available) and the VALUE must be causal, or a
    # causally-covered row still carries a delta computed from future laps.
    # Session 8 measured the cost of the old version: the covered branch, fitted where the
    # whole-stint-covered rate is 45.4% at 0-0.5 s, was standing in for a causally-covered
    # population that passes at 53.7%, while the missing branch at 19.1% stood in for one at
    # 27.4%. Set PASS_WHOLE_STINT_PACE=1 to refit the old way for comparison.
    col = "free_pace" if os.environ.get("PASS_WHOLE_STINT_PACE") == "1" else "free_pace_causal"
    fp = pd.read_parquet(STATE_PATH).set_index(["race", "driver", "lap"])[col]
    me = fp.reindex(pd.MultiIndex.from_arrays([b["race"], b["driver"], b["lap"]])).to_numpy()
    them = fp.reindex(pd.MultiIndex.from_arrays([b["race"], b["ahead"], b["lap"]])).to_numpy()
    b["free_delta"] = them - me  # + = follower faster in free air
    b.attrs["pace_column"] = col

    sc = pd.read_parquet(SC_PATH)
    sc = sc[sc["kind"] == "SC"]
    restart = {(s, r): set() for s, r in zip(sc["season"], sc["round"], strict=True)}
    for s, r, end in zip(sc["season"], sc["round"], sc["lap_end"], strict=True):
        restart[(s, r)].update({end, end + 1, end + 2})
    near_restart = [lap in restart.get((s, r), set())
                    for s, r, lap in zip(b["season"], b["round"], b["lap"], strict=True)]
    b["drs_available"] = ((b["gap_before_s"] < 1.0) & (b["lap"] >= 3)
                          & ~np.array(near_restart)).astype(float)

    pair = b["driver"] + ">" + b["ahead"]
    b = b.assign(pair=pair).sort_values(["race", "pair", "lap"])
    new_run = b.groupby(["race", "pair"])["lap"].diff() != 1
    run = new_run.groupby([b["race"], b["pair"]]).cumsum()
    b["laps_in_battle"] = b.groupby([b["race"], b["pair"], run]).cumcount() + 1
    return b.sort_index()


def features(b: pd.DataFrame, events: list[str]) -> pd.DataFrame:
    x = pd.DataFrame(index=b.index)
    bins = pd.cut(b["gap_before_s"], GAP_EDGES, right=False)
    for i, interval in enumerate(bins.cat.categories):
        x[f"gap_{i}"] = (bins == interval).astype(float)  # full set, no intercept below
    x["free_missing"] = b["free_delta"].isna().astype(float)
    x["free_delta_f"] = b["free_delta"].fillna(0.0)
    x["tyre_age_delta"] = b["tyre_age_delta"]
    pair = b["compound_pair"].str.split("-", expand=True)
    x["compound_step"] = pair[1].map(HARDNESS) - pair[0].map(HARDNESS)
    x["log_laps_in_battle"] = np.log(b["laps_in_battle"])
    x["drs_available"] = b["drs_available"]
    for e in events:
        x[f"event_{e}"] = (b["event"] == e).astype(float)
    return x


class PassModel:
    def __init__(self, train: pd.DataFrame):
        self.events = sorted(train["event"].unique())
        x = features(train, self.events)
        self.columns = list(x.columns)
        self.mean = x[NUMERIC].mean()
        self.std = x[NUMERIC].std().replace(0, 1.0)
        self.lr = LogisticRegression(C=C, fit_intercept=False, max_iter=2000)
        self.lr.fit(self._scale(x), train["passed"].astype(int))

    def _scale(self, x: pd.DataFrame) -> np.ndarray:
        x = x.copy()
        x[NUMERIC] = (x[NUMERIC] - self.mean) / self.std
        return x.to_numpy()

    def predict(self, b: pd.DataFrame) -> np.ndarray:
        """Unseen tracks get no event term: the global rate."""
        return self.lr.predict_proba(self._scale(features(b, self.events)))[:, 1]

    def to_json(self) -> dict:
        return {"columns": self.columns, "coef": self.lr.coef_[0].tolist(),
                "mean": self.mean.to_dict(), "std": self.std.to_dict(), "events": self.events,
                "gap_edges": GAP_EDGES}


def calibration_table(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    bucket = pd.cut(p, PROB_BUCKETS, right=False)
    t = pd.DataFrame({"y": y, "p": p, "bucket": bucket})
    return t.groupby("bucket", observed=True).agg(
        mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size")
    )


def gap_table(b: pd.DataFrame) -> pd.DataFrame:
    g = pd.cut(b["gap_before_s"], [0, 0.5, 1.0, 1.5, 2.0], right=False)
    return b.groupby(g, observed=True).agg(
        observed=("passed", "mean"), predicted=("p_pass", "mean"), n=("passed", "size")
    )


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    return {"auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p)), "base_rate": float(y.mean()), "n": int(len(y))}


def main() -> int:
    b = load()
    oof = b.copy()
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(b, groups=b["race"]):
        model = PassModel(b.iloc[tr])
        oof.loc[b.index[te], "p_pass"] = model.predict(b.iloc[te])
    oof.to_parquet(OOF_PATH, index=False)

    y, p = oof["passed"].astype(int).to_numpy(), oof["p_pass"].to_numpy()
    multi = (oof.groupby("event")["race"].transform("nunique") > 1).to_numpy()
    covered = oof["free_delta"].notna().to_numpy()
    # Reference: gap bins only, same folds
    ref = np.zeros(len(b))
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(b, groups=b["race"]):
        g = pd.cut(b["gap_before_s"], GAP_EDGES, right=False)
        rates = b.iloc[tr].groupby(g.iloc[tr], observed=False)["passed"].mean()
        ref[te] = g.iloc[te].map(rates).astype(float).to_numpy()

    res = {
        "all": scores(y, p), "covered": scores(y[covered], p[covered]),
        "gap_only_reference": scores(y, ref),
        # Single-race events are never seen in their training fold (no event term)
        "seen_track": {"predicted": float(p[multi].mean()), "observed": float(y[multi].mean())},
        "unseen_track": {"predicted": float(p[~multi].mean()),
                         "observed": float(y[~multi].mean())},
        "calibration_all": calibration_table(y, p).round(4).reset_index().astype(str)
        .to_dict(orient="records"),
    }
    REPORT_PATH.write_text(json.dumps(res, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(f"train battles: {len(b)}  passes: {int(y.sum())}  races: {b['race'].nunique()}  "
          f"free-air pace covered: {covered.mean():.1%}")
    print("\nheld-out calibration, all battles:")
    print(calibration_table(y, p).round(3).to_string())
    print("\nheld-out calibration, covered rows (what the engine queries):")
    print(calibration_table(y[covered], p[covered]).round(3).to_string())
    print("\nbase rates by gap, held-out:")
    print(gap_table(oof).round(3).to_string())
    print("\ndiscrimination (held-out):")
    print(pd.DataFrame({k: res[k] for k in ("all", "covered", "gap_only_reference")})
          .round(4).to_string())

    print(f"\nmean predicted vs observed: seen tracks {res['seen_track']['predicted']:.4f} / "
          f"{res['seen_track']['observed']:.4f}, unseen tracks "
          f"{res['unseen_track']['predicted']:.4f} / {res['unseen_track']['observed']:.4f}")

    final = PassModel(b)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(final.to_json(), indent=2), encoding="utf-8")
    coef = pd.Series(final.lr.coef_[0], index=final.columns)
    print("\nfinal coefficients (numeric terms standardised):")
    print(coef[[c for c in coef.index if not c.startswith("event_")]].round(3).to_string())
    print(f"\nwrote {OOF_PATH}, {REPORT_PATH}, {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
