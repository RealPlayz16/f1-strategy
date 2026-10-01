"""Tyre lap time model with uncertainty, re-anchored on the live race state.

At anchor lap t the model sees only laps <= t and predicts the lap time h laps ahead:

    y = lap_time_fc_s(t + h) - anchor_s(t)

anchor_s(t) = field_ref(t) + the driver's median gap to the per-lap field median, over the
driver's clean laps in the window, current stint only (at least MIN_DRIVER_LAPS).
field_ref(t) = median fuel-corrected clean lap of the field over laps max(WINDOW_START, t-W+1)
to t. The window never includes laps 2-4: the race-start part of the early-stint effect would
bias baselines by start compound (src/degradation.py).

Why re-anchored (Session 2): a fixed reference from early laps left a per-race offset with sd
0.60 s on unseen races, while the quantile net learned each training race's level from event
and temperatures (in-sample per-race sd 0.043 s), so held-out intervals missed 27 / 22% at
p10 / p90. Anchoring on recent laps removes most of the race-level offset; the remaining
horizon-dependent uncertainty is what the quantiles have to carry.

Interface change from the original brief (predict_laptime(features) -> (p10, p50, p90)):
predictions are now horizon dependent and need the anchor. predict_laptime takes anchor_s and
h with the tyre plan for lap t + h and returns absolute seconds:
    anchor_s + predicted delta + K * laps_remaining(t + h) / race_laps
compute_anchors(laps) builds anchors from laps up to t. Session 4's optimizer needs h up to a
full stint (30).

Features: h; tyre now (tyre_life_t, compound_t); tyre at t + h (tyre_life_f, compound_f,
fresh_tyre_f, stints_ahead, so a planned stop is inside the horizon); track_temp and air_temp
at t (future temperatures are not known live); event, driver, team.

Two stages:
  p50       LightGBM median on all features.
  p10, p90  p50 + offsets from a PyTorch quantile network (pinball, seed 42) trained on
            LightGBM residuals from races left out of an inner GroupKFold, so the offsets
            carry the error the median model makes on a race it has not seen.
The spread net only gets features that cannot identify a race (SPREAD_FEATURES["tyre"]).
Given event and temperatures it learns each training race's residual offset and the
quantiles shrink back to within-race noise: the failure of the first model (in-sample
per-race residual sd 0.043 s vs 0.60 s held out). An earlier single network for all three
quantiles also had a location bias growing with h (p50 0.036 to 0.104 s slow), while the
LightGBM median was unbiased.

GroupKFold by race over train races only; the holdout is never loaded. No clamping, sorting
or other post-processing; quantile crossings are counted and reported.

Usage:
    python -m src.tyre
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch import nn

from src.splits import add_split

LAPS_PATH = Path("data/processed/laps_fuel_corrected.parquet")
FIT_PATH = Path("data/processed/fuel_fit.json")
MODEL_DIR = Path("data/models")
OOF_PATH = Path("data/processed/tyre_oof.parquet")
REPORT_PATH = Path("data/processed/tyre_report.json")

SEED = 42
WINDOW = 5
WINDOW_START = 5          # skip laps 2-4 (race-start part of the early-stint effect)
MIN_DRIVER_LAPS = 2
MIN_FIELD_LAPS = 5
HORIZONS = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30]
REPORT_H = [1, 5, 15, 30]
SPREAD_QUANTILES = (0.1, 0.9)
N_FOLDS = 5
CATS = ["compound_t", "compound_f", "event", "driver", "team"]
NUMS = ["h", "tyre_life_t", "tyre_life_f", "fresh_tyre_f", "stints_ahead", "track_temp",
        "air_temp"]
FEATURES = NUMS + CATS
# Spread net features. "tyre" cannot identify a race, so the net cannot learn per-race
# residual offsets and its quantiles keep the between-race variance. "all" is the
# race-identifying comparison only.
SPREAD_FEATURES = {
    "tyre": (["compound_t", "compound_f"],
             ["h", "tyre_life_t", "tyre_life_f", "fresh_tyre_f", "stints_ahead"]),
    "all": (CATS, NUMS),
}
EPOCHS = 30
BATCH = 1024
CAT_DROPOUT = 0.1
LGB_PARAMS = {
    "objective": "quantile", "alpha": 0.5, "learning_rate": 0.05, "num_leaves": 31,
    "min_data_in_leaf": 50, "feature_fraction": 0.9, "bagging_fraction": 0.8,
    "bagging_freq": 1, "seed": SEED, "verbose": -1, "deterministic": True,
}
LGB_ROUNDS = 400


# ---------- anchors and training frame ----------

def compute_anchors(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per (race, driver, lap t) with anchor_s, using only laps <= t.

    laps: clean laps of one or more races with lap_time_fc_s. Anchors exist where the
    driver has MIN_DRIVER_LAPS clean laps in the window within the current stint."""
    c = laps[laps["lap"] >= WINDOW_START].copy()
    keys = ["season", "round"]
    lap_med = c.groupby(keys + ["lap"])["lap_time_fc_s"].median().rename("lap_med")
    c = c.join(lap_med, on=keys + ["lap"])
    c["gap_to_field"] = c["lap_time_fc_s"] - c["lap_med"]
    rows = []
    for (season, rnd), race in c.groupby(keys, sort=True):
        for t in sorted(race["lap"].unique()):
            lo = max(WINDOW_START, t - WINDOW + 1)
            field = race[race["lap"].between(lo, t)]
            if len(field) < MIN_FIELD_LAPS:
                continue
            field_ref = float(field["lap_time_fc_s"].median())
            now = race[race["lap"] == t]
            for _, row in now.iterrows():
                mine = field[(field["driver"] == row["driver"]) & (field["stint"] == row["stint"])]
                if len(mine) < MIN_DRIVER_LAPS:
                    continue
                rows.append(
                    {
                        "season": season, "round": rnd, "driver": row["driver"], "lap": t,
                        "field_ref_s": field_ref,
                        "anchor_s": field_ref + float(mine["gap_to_field"].median()),
                        "tyre_life_t": row["tyre_life"], "compound_t": row["compound"],
                        "stint_t": row["stint"], "track_temp": row["track_temp"],
                        "air_temp": row["air_temp"],
                    }
                )
    return pd.DataFrame(rows)


def build_frame(laps: pd.DataFrame) -> pd.DataFrame:
    """(anchor, horizon) rows for train races: features at t, tyre plan and target at t + h."""
    laps = add_split(laps.drop(columns=["split"], errors="ignore"))
    c = laps[laps["is_clean"] & (laps["split"] == "train")].copy()
    if c.empty:
        return pd.DataFrame(columns=FEATURES + ["y", "race", "h"])
    anchors = compute_anchors(c)
    future = c[["season", "round", "event", "driver", "team", "lap", "lap_time_fc_s",
                "tyre_life", "compound", "fresh_tyre", "stint", "split"]].rename(
        columns={"lap": "lap_f", "tyre_life": "tyre_life_f", "compound": "compound_f",
                 "fresh_tyre": "fresh_tyre_f", "stint": "stint_f"}
    )
    frames = []
    for h in HORIZONS:
        a = anchors.assign(h=h, lap_f=anchors["lap"] + h)
        frames.append(a.merge(future, on=["season", "round", "driver", "lap_f"]))
    df = pd.concat(frames, ignore_index=True)
    df["fresh_tyre_f"] = df["fresh_tyre_f"].astype(float)
    df["stints_ahead"] = df["stint_f"] - df["stint_t"]
    df["race"] = df["season"].astype(str) + "_" + df["round"].astype(str).str.zfill(2)
    df["y"] = df["lap_time_fc_s"] - df["anchor_s"]
    return df.reset_index(drop=True)


# ---------- LightGBM median ----------

def lgb_frame(df: pd.DataFrame, cats: dict[str, list[str]]) -> pd.DataFrame:
    x = df[FEATURES].copy()
    for col in CATS:
        x[col] = pd.Categorical(x[col], categories=cats[col])
    return x


def fit_lgb(df: pd.DataFrame, cats: dict[str, list[str]]) -> lgb.Booster:
    data = lgb.Dataset(lgb_frame(df, cats), df["y"], categorical_feature=CATS)
    return lgb.train(LGB_PARAMS, data, num_boost_round=LGB_ROUNDS)


# ---------- spread network: p10 / p90 offsets around the LightGBM median ----------

class Encoder:
    """Category -> index (0 = unknown) and numeric standardisation, fitted on training rows."""

    def __init__(self, df: pd.DataFrame | None, cats: list[str], nums: list[str]):
        self.cats, self.nums = cats, nums
        if df is None:  # filled by from_state
            return
        self.vocab = {c: {v: i + 1 for i, v in enumerate(sorted(df[c].unique()))} for c in cats}
        num = df[nums]
        self.mean = num.mean()
        self.std = num.std().replace(0, 1.0)

    def state(self) -> dict:
        return {"cats": self.cats, "nums": self.nums, "vocab": self.vocab,
                "mean": self.mean.to_dict(), "std": self.std.to_dict()}

    @classmethod
    def from_state(cls, s: dict) -> Encoder:
        enc = cls(None, s["cats"], s["nums"])
        enc.vocab, enc.mean, enc.std = s["vocab"], pd.Series(s["mean"]), pd.Series(s["std"])
        return enc

    def __call__(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        num = (df[self.nums] - self.mean) / self.std
        missing = num.isna().astype(float).add_suffix("_missing")
        num = pd.concat([num.fillna(0.0), missing], axis=1).to_numpy(np.float32)
        cat = np.stack(
            [df[c].map(self.vocab[c]).fillna(0).astype(int).to_numpy() for c in self.cats],
            axis=1,
        )
        return torch.from_numpy(num), torch.from_numpy(cat)


class QuantileNet(nn.Module):
    def __init__(self, n_num: int, vocab_sizes: list[int], n_q: int):
        super().__init__()
        dims = [min(16, (n + 1) // 2 + 1) for n in vocab_sizes]
        self.emb = nn.ModuleList(
            nn.Embedding(n + 1, d) for n, d in zip(vocab_sizes, dims, strict=True)
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_num + sum(dims), 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_q),
        )

    def forward(self, num: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        e = [emb(cat[:, i]) for i, emb in enumerate(self.emb)]
        return self.mlp(torch.cat([num, *e], dim=1))


def pinball(pred: torch.Tensor, y: torch.Tensor, quantiles=SPREAD_QUANTILES) -> torch.Tensor:
    q = torch.tensor(quantiles, dtype=pred.dtype)
    err = y[:, None] - pred
    return torch.maximum(q * err, (q - 1) * err).mean()


def fit_net(df: pd.DataFrame, target: str, cats: list[str], nums: list[str]):
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    enc = Encoder(df, cats, nums)
    num, cat = enc(df)
    y = torch.tensor(df[target].to_numpy(np.float32))
    net = QuantileNet(num.shape[1], [len(enc.vocab[c]) for c in cats], len(SPREAD_QUANTILES))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = len(y)
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            c = cat[idx].clone()
            drop = torch.rand(c.shape, generator=gen) < CAT_DROPOUT  # trains the unknown index
            c[drop] = 0
            loss = pinball(net(num[idx], c), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net, enc


def predict_net(net: QuantileNet, enc: Encoder, df: pd.DataFrame) -> np.ndarray:
    net.eval()
    with torch.no_grad():
        num, cat = enc(df)
        return net(num, cat).numpy()


def out_of_race_residuals(train: pd.DataFrame, cats: dict[str, list[str]]) -> pd.Series:
    """LightGBM residuals on races left out of an inner GroupKFold: what the median model's
    error looks like on a race it has not seen."""
    resid = pd.Series(np.nan, index=train.index)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(train, groups=train["race"]):
        booster = fit_lgb(train.iloc[tr], cats)
        te_idx = train.index[te]
        resid.loc[te_idx] = train.loc[te_idx, "y"] - booster.predict(
            lgb_frame(train.loc[te_idx], cats)
        )
    return resid


def fit_two_stage(train: pd.DataFrame, cats: dict[str, list[str]], spread: str = "tyre"):
    """LightGBM median on all rows; spread net on out-of-race residuals."""
    booster = fit_lgb(train, cats)
    train = train.assign(resid=out_of_race_residuals(train, cats))
    s_cats, s_nums = SPREAD_FEATURES[spread]
    net, enc = fit_net(train, "resid", s_cats, s_nums)
    return booster, net, enc


def predict_two_stage(booster, net, enc, df, cats) -> np.ndarray:
    p50 = booster.predict(lgb_frame(df, cats))
    off = predict_net(net, enc, df)
    return np.column_stack([p50 + off[:, 0], p50, p50 + off[:, 1]])


# ---------- cross-validation and report ----------

def cross_validate(df: pd.DataFrame, spread: str = "tyre") -> pd.DataFrame:
    cats = {c: sorted(df[c].unique()) for c in CATS}
    oof = df[["race", "season", "event", "driver", "compound_t", "compound_f", "h", "lap",
              "y"]].copy()
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=N_FOLDS).split(df, groups=df["race"])):
        train, test = df.iloc[tr], df.iloc[te]
        booster, net, enc = fit_two_stage(train, cats, spread)
        q = predict_two_stage(booster, net, enc, test, cats)
        for j, name in enumerate(["p10", "p50", "p90"]):
            oof.loc[test.index, name] = q[:, j]
        oof.loc[test.index, "fold"] = fold
        print(f"fold {fold}: {test['race'].nunique()} races, {len(test)} rows", flush=True)
    return oof


def calibration(o: pd.DataFrame) -> dict:
    resid = o["y"] - o["p50"]
    inner = resid - resid.groupby(o["race"]).transform("median")
    return {
        "n": int(len(o)),
        "frac_below_p10": float((o["y"] < o["p10"]).mean()),
        "frac_above_p90": float((o["y"] > o["p90"]).mean()),
        "median_width_s": float((o["p90"] - o["p10"]).median()),
        "median_resid_s": float(resid.median()),
        "mae_p50": float(resid.abs().mean()),
        "race_offset_sd": float(resid.groupby(o["race"]).median().std()),
        "below_offset_removed": float((inner < o["p10"] - o["p50"]).mean()),
        "above_offset_removed": float((inner > o["p90"] - o["p50"]).mean()),
    }


def report(oof: pd.DataFrame) -> dict:
    out = {"all": calibration(oof)}
    out["crossing"] = float(((oof["p10"] > oof["p50"]) | (oof["p50"] > oof["p90"])).mean())
    out["by_h"] = {int(h): calibration(oof[oof["h"] == h]) for h in REPORT_H}
    out["by_compound_f"] = {c: calibration(g) for c, g in oof.groupby("compound_f")}
    out["by_compound_f_h"] = {
        f"{c} h={h}": calibration(g)
        for (c, h), g in oof[oof["h"].isin(REPORT_H)].groupby(["compound_f", "h"])
    }
    out["by_fold"] = {int(f): calibration(g) for f, g in oof.groupby("fold")}
    return out


# ---------- interface ----------

_FINAL: dict = {}


def load_final() -> dict:
    if not _FINAL:
        s = torch.load(MODEL_DIR / "tyre_model.pt", weights_only=False)
        enc = Encoder.from_state(s["encoder"])
        net = QuantileNet(s["n_num"], s["vocab_sizes"], len(SPREAD_QUANTILES))
        net.load_state_dict(s["state_dict"])
        _FINAL.update({"net": net, "encoder": enc, "cats": s["cats"], "K_s": s["K_s"],
                       "booster": lgb.Booster(model_file=str(MODEL_DIR / "tyre_lgb.txt"))})
    return _FINAL


def predict_laptime(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """features: FEATURES plus anchor_s (from compute_anchors at lap t), laps_remaining at
    t + h and race_laps. Returns absolute lap time p10, p50, p90 in seconds for lap t + h."""
    s = load_final()
    q = predict_two_stage(s["booster"], s["net"], s["encoder"], features, s["cats"])
    base = features["anchor_s"].to_numpy() + s["K_s"] * (
        features["laps_remaining"].to_numpy() / features["race_laps"].to_numpy()
    )
    return base + q[:, 0], base + q[:, 1], base + q[:, 2]


def print_report(res: dict) -> None:
    pd.set_option("display.width", 220)
    keys = ["frac_below_p10", "frac_above_p90", "median_width_s", "median_resid_s", "mae_p50",
            "race_offset_sd", "below_offset_removed", "above_offset_removed", "n"]
    by_h = pd.DataFrame({"all": res["all"], **{f"h={h}": v for h, v in res["by_h"].items()}})
    print(by_h.loc[keys].round(3).to_string())
    print("\nby future compound:")
    print(pd.DataFrame(res["by_compound_f"]).loc[keys].round(3).to_string())
    print("\nby future compound and horizon:")
    t = pd.DataFrame(res["by_compound_f_h"]).T[["frac_below_p10", "frac_above_p90",
                                                "median_resid_s", "n"]]
    print(t.round(3).to_string())
    print(f"\nquantile crossing: {res['crossing']:.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tyre model: CV report and final fit.")
    parser.add_argument("--no-final", action="store_true", help="skip the final fit")
    parser.add_argument("--final-only", action="store_true", help="skip cross-validation")
    parser.add_argument("--spread", default="tyre", choices=sorted(SPREAD_FEATURES),
                        help="spread net features; 'all' is the race-identifying comparison")
    args = parser.parse_args(argv)
    np.random.seed(SEED)

    df = build_frame(pd.read_parquet(LAPS_PATH))
    print(f"train races: {df['race'].nunique()}  rows: {len(df)}  "
          f"holdout rows: {int((df['split'] != 'train').sum())}  spread: {args.spread}")

    if not args.final_only:
        oof = cross_validate(df, args.spread)
        tag = "" if args.spread == "tyre" else f"_{args.spread}"
        oof.to_parquet(OOF_PATH.with_name(f"tyre_oof{tag}.parquet"), index=False)
        res = report(oof)
        REPORT_PATH.with_name(f"tyre_report{tag}.json").write_text(
            json.dumps(res, indent=2), encoding="utf-8"
        )
        print_report(res)

    if not args.no_final:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cats = {c: sorted(df[c].unique()) for c in CATS}
        booster, net, enc = fit_two_stage(df, cats, args.spread)
        booster.save_model(str(MODEL_DIR / "tyre_lgb.txt"))
        k = json.loads(FIT_PATH.read_text(encoding="utf-8"))["K_s"]
        # Plain state, not pickled objects: classes defined under python -m live in __main__
        torch.save(
            {"state_dict": net.state_dict(), "n_num": net.mlp[0].in_features - sum(
                e.embedding_dim for e in net.emb),
             "vocab_sizes": [len(enc.vocab[c]) for c in enc.cats], "encoder": enc.state(),
             "cats": cats, "K_s": k},
            MODEL_DIR / "tyre_model.pt",
        )
        print(f"wrote {MODEL_DIR / 'tyre_lgb.txt'} and {MODEL_DIR / 'tyre_model.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
