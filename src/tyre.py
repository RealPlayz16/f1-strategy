"""Tyre lap time model with uncertainty: LightGBM median and a PyTorch quantile network.

Target, per clean lap after the reference window, on race-progress corrected time:
    y = lap_time_fc_s - race_ref_s
race_ref_s is the median fuel-corrected clean lap of the whole field on laps REF_LAPS. Absolute
lap time across races is not identified for a track the fold has never seen (5 of 23 train
events appear once), so the model predicts lap time relative to the race's own level, which is
known by the end of the reference window in live use.

Features: tyre_life, compound, track_temp, air_temp, event, driver, team, stint, fresh_tyre,
driver_pace_s. driver_pace_s is the driver's median fuel-corrected clean lap on laps REF_LAPS
minus race_ref_s: causal, never computed from the laps being predicted. NaN when the driver has
no clean lap in the window.

Evaluation: GroupKFold by race over train races only; the holdout is never loaded. Laps within
a stint are nearly identical, so random row splits would leak.

Interface: predict_laptime(features) -> (p10, p50, p90) in absolute seconds, from the quantile
network, adding back race_ref_s and the race-progress term. No clamping, sorting or other
post-processing; quantile crossings are counted and reported, not fixed.

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

SEED = 42
REF_LAPS = (2, 10)
QUANTILES = (0.1, 0.5, 0.9)
N_FOLDS = 5
CATS = ["compound", "event", "driver", "team"]
NUMS = ["tyre_life", "track_temp", "air_temp", "stint", "fresh_tyre", "driver_pace_s"]
FEATURES = NUMS + CATS
EPOCHS = 60
BATCH = 512
CAT_DROPOUT = 0.1
LGB_PARAMS = {
    "objective": "quantile", "alpha": 0.5, "learning_rate": 0.05, "num_leaves": 31,
    "min_data_in_leaf": 50, "feature_fraction": 0.9, "bagging_fraction": 0.8,
    "bagging_freq": 1, "seed": SEED, "verbose": -1, "deterministic": True,
}
LGB_ROUNDS = 400


def build_frame(laps: pd.DataFrame) -> pd.DataFrame:
    """Train clean laps after the reference window, with race_ref_s, driver_pace_s and y."""
    laps = add_split(laps.drop(columns=["split"], errors="ignore"))
    c = laps[laps["is_clean"] & (laps["split"] == "train")].copy()
    c["race"] = c["season"].astype(str) + "_" + c["round"].astype(str).str.zfill(2)
    lo, hi = REF_LAPS
    window = c[c["lap"].between(lo, hi)]
    ref = window.groupby("race")["lap_time_fc_s"].median().rename("race_ref_s")
    drv = window.groupby(["race", "driver"])["lap_time_fc_s"].median().rename("drv_ref")
    c = c.join(ref, on="race").join(drv, on=["race", "driver"])
    c["driver_pace_s"] = c["drv_ref"] - c["race_ref_s"]
    c = c[c["lap"] > hi].copy()
    c["fresh_tyre"] = c["fresh_tyre"].astype(float)
    c["y"] = c["lap_time_fc_s"] - c["race_ref_s"]
    return c.reset_index(drop=True)


# ---------- LightGBM median ----------

def lgb_frame(df: pd.DataFrame, cats: dict[str, list[str]]) -> pd.DataFrame:
    x = df[FEATURES].copy()
    for col in CATS:
        x[col] = pd.Categorical(x[col], categories=cats[col])
    return x


def fit_lgb(df: pd.DataFrame, cats: dict[str, list[str]]) -> lgb.Booster:
    data = lgb.Dataset(lgb_frame(df, cats), df["y"], categorical_feature=CATS)
    return lgb.train(LGB_PARAMS, data, num_boost_round=LGB_ROUNDS)


# ---------- quantile network ----------

class Encoder:
    """Category -> index (0 = unknown) and numeric standardisation, fitted on training rows."""

    def __init__(self, df: pd.DataFrame):
        self.vocab = {c: {v: i + 1 for i, v in enumerate(sorted(df[c].unique()))} for c in CATS}
        num = df[NUMS]
        self.mean = num.mean()
        self.std = num.std().replace(0, 1.0)

    def __call__(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        num = (df[NUMS] - self.mean) / self.std
        missing = num.isna().astype(float).add_suffix("_missing")
        num = pd.concat([num.fillna(0.0), missing], axis=1).to_numpy(np.float32)
        cat = np.stack(
            [df[c].map(self.vocab[c]).fillna(0).astype(int).to_numpy() for c in CATS], axis=1
        )
        return torch.from_numpy(num), torch.from_numpy(cat)


class QuantileNet(nn.Module):
    def __init__(self, n_num: int, vocab_sizes: list[int], n_q: int = len(QUANTILES)):
        super().__init__()
        dims = [min(16, (n + 1) // 2 + 1) for n in vocab_sizes]
        self.emb = nn.ModuleList(nn.Embedding(n + 1, d) for n, d in zip(vocab_sizes, dims,
                                                                         strict=True))
        self.mlp = nn.Sequential(
            nn.Linear(n_num + sum(dims), 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_q),
        )

    def forward(self, num: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        e = [emb(cat[:, i]) for i, emb in enumerate(self.emb)]
        return self.mlp(torch.cat([num, *e], dim=1))


def pinball(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    q = torch.tensor(QUANTILES, dtype=pred.dtype)
    err = y[:, None] - pred
    return torch.maximum(q * err, (q - 1) * err).mean()


def fit_net(df: pd.DataFrame) -> tuple[QuantileNet, Encoder]:
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    enc = Encoder(df)
    num, cat = enc(df)
    y = torch.tensor(df["y"].to_numpy(np.float32))
    net = QuantileNet(num.shape[1], [len(enc.vocab[c]) for c in CATS])
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


# ---------- cross-validation ----------

def cross_validate(df: pd.DataFrame) -> pd.DataFrame:
    cats = {c: sorted(df[c].unique()) for c in CATS}
    oof = df[["race", "season", "event", "driver", "compound", "tyre_life", "lap", "y"]].copy()
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=N_FOLDS).split(df, groups=df["race"])):
        train, test = df.iloc[tr], df.iloc[te]
        booster = fit_lgb(train, cats)
        oof.loc[test.index, "lgb_p50"] = booster.predict(lgb_frame(test, cats))
        net, enc = fit_net(train)
        q = predict_net(net, enc, test)
        for j, name in enumerate(["p10", "p50", "p90"]):
            oof.loc[test.index, name] = q[:, j]
        oof.loc[test.index, "fold"] = fold
        unseen = ~test["event"].isin(train["event"].unique())
        oof.loc[test.index, "event_unseen"] = unseen.to_numpy()
        print(f"fold {fold}: {test['race'].nunique()} races, {len(test)} laps", flush=True)
    return oof


def report(oof: pd.DataFrame) -> dict:
    below = (oof["y"] < oof["p10"]).mean()
    above = (oof["y"] > oof["p90"]).mean()
    crossing = ((oof["p10"] > oof["p50"]) | (oof["p50"] > oof["p90"])).mean()
    out = {
        "mae_lgb_p50": float((oof["y"] - oof["lgb_p50"]).abs().mean()),
        "mae_net_p50": float((oof["y"] - oof["p50"]).abs().mean()),
        "mae_baseline_zero": float(oof["y"].abs().mean()),
        "frac_below_p10": float(below),
        "frac_above_p90": float(above),
        "quantile_crossing": float(crossing),
        "n_laps": int(len(oof)),
    }
    by = oof.assign(below=oof["y"] < oof["p10"], above=oof["y"] > oof["p90"],
                    ae_lgb=(oof["y"] - oof["lgb_p50"]).abs())
    out["by_fold"] = by.groupby("fold")[["below", "above", "ae_lgb"]].mean().round(4).to_dict()
    out["by_compound"] = by.groupby("compound")[["below", "above", "ae_lgb"]].mean().round(
        4).to_dict()
    out["by_event_unseen"] = by.groupby("event_unseen")[["below", "above", "ae_lgb"]].mean(
    ).round(4).to_dict()
    return out


# ---------- interface ----------

_FINAL: dict = {}


def load_final() -> dict:
    if not _FINAL:
        state = torch.load(MODEL_DIR / "tyre_net.pt", weights_only=False)
        _FINAL.update(state)
    return _FINAL


def predict_laptime(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """features: FEATURES plus race_ref_s, laps_remaining, race_laps. Returns absolute lap time
    p10, p50, p90 in seconds: race_ref_s + predicted delta + K * laps_remaining / race_laps."""
    state = load_final()
    q = predict_net(state["net"], state["encoder"], features)
    base = features["race_ref_s"].to_numpy() + state["K_s"] * (
        features["laps_remaining"].to_numpy() / features["race_laps"].to_numpy()
    )
    return base + q[:, 0], base + q[:, 1], base + q[:, 2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tyre model: CV report and final fit.")
    parser.add_argument("--no-final", action="store_true", help="skip the final fit")
    args = parser.parse_args(argv)
    np.random.seed(SEED)

    laps = pd.read_parquet(LAPS_PATH)
    df = build_frame(laps)
    print(f"train races: {df['race'].nunique()}  laps after window: {len(df)}  "
          f"holdout rows: {int((df['split'] != 'train').sum())}")

    oof = cross_validate(df)
    oof.to_parquet(OOF_PATH, index=False)
    res = report(oof)
    print(json.dumps(res, indent=2))

    if not args.no_final:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cats = {c: sorted(df[c].unique()) for c in CATS}
        fit_lgb(df, cats).save_model(str(MODEL_DIR / "tyre_lgb.txt"))
        net, enc = fit_net(df)
        k = json.loads(FIT_PATH.read_text(encoding="utf-8"))["K_s"]
        torch.save({"net": net, "encoder": enc, "K_s": k}, MODEL_DIR / "tyre_net.pt")
        print(f"wrote {MODEL_DIR / 'tyre_lgb.txt'} and {MODEL_DIR / 'tyre_net.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
