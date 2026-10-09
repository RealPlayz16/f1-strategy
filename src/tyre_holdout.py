"""Score the FROZEN tyre model on the holdout. One terminal pass, no fitting, ever.

THE RULING (Session 10, user decision, recorded here so the reasoning survives). A single
terminal evaluation of a frozen model on the holdout is NOT contamination. The rule exists to
stop the holdout influencing fits, and one scoring pass with no tuning afterward does not.
What would be contamination is what this module cannot do: refit, reseed, change a threshold
or a feature because of what the number came back as, or tune anything downstream of it.

Consequently this module is deliberately separate from src/tyre.py. There is no code path
here that fits anything: it loads the saved artefacts through tyre.load_final, and it records
the SHA-256 of both model files in its own output so "frozen" is verifiable afterwards rather
than asserted. If the artefacts are missing it refuses rather than falling back to a fit.

Scored exactly as the train CV figures were scored, through tyre.calibration on the same y
relative to the anchor, so the two are comparable: 10.7% below p10 and 10.2% above p90,
median residual near zero, MAE 0.506 on 29 train races under GroupKFold.

The frozen LightGBM median takes event, driver and team, and every holdout track is unseen by
it, so those rows go down its missing branch. That is the honest behaviour of this model in
deployment and is not corrected for. The spread net never sees event, driver or team by design
(it would learn per-race offsets, see tyre.SPREAD_FEATURES), so interval WIDTHS are unaffected
by unseen tracks and only the median is.

Usage:
    python -m src.tyre_holdout
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from src.splits import load_splits
from src.tyre import (
    LAPS_PATH,
    MODEL_DIR,
    REPORT_H,
    build_frame,
    calibration,
    load_final,
    predict_two_stage,
)

OUT_PATH = Path("data/processed/tyre_holdout_scores.parquet")
REPORT_PATH = Path("data/processed/tyre_holdout_report.json")
MODEL_FILES = ("tyre_lgb.txt", "tyre_model.pt")

# The train CV figures this is compared against, from data/processed/tyre_report.json.
CV_REFERENCE = {"frac_below_p10": 0.107, "frac_above_p90": 0.102, "mae_p50": 0.506,
                "median_resid_s": 0.009, "race_offset_sd": 0.168}


def model_fingerprint() -> dict[str, str]:
    """SHA-256 of each frozen artefact, so the report is tied to the model that produced it."""
    out = {}
    for name in MODEL_FILES:
        path = MODEL_DIR / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. This module scores a FROZEN model and must not fit one; "
                f"run 'python -m src.tyre --final-only' deliberately if a refit is intended.")
        out[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return out


def train_events() -> set[str]:
    """Event names that appear in the TRAIN split, so the frozen LightGBM median has seen the
    track. Read from config/races.yaml, not from holdout data."""
    return {e for (_, e), sp in load_splits().items() if sp == "train"}


def score() -> pd.DataFrame:
    """Holdout rows with p10/p50/p90 from the frozen model. Fits nothing."""
    df = build_frame(pd.read_parquet(LAPS_PATH), split="holdout")
    if df.empty:
        raise SystemExit("no holdout rows: check config/races.yaml and the cleaned laps")
    s = load_final()
    q = predict_two_stage(s["booster"], s["net"], s["encoder"], df, s["cats"])
    out = df[["race", "season", "event", "driver", "compound_t", "compound_f", "h", "lap",
              "y"]].copy()
    out["p10"], out["p50"], out["p90"] = q[:, 0], q[:, 1], q[:, 2]
    # Decided and written BEFORE the run, not chosen after seeing the numbers: the stated
    # reason for expecting degradation is unseen tracks, so the seen/unseen cut is the test of
    # that reason. 3 of the 9 holdout races are at tracks the train split contains.
    seen = train_events()
    out["track_seen"] = [any(str(e).startswith(q) for q in seen) for e in out["event"]]
    return out


def main(argv: list[str] | None = None) -> int:
    _ = argv
    fingerprint = model_fingerprint()
    o = score()
    res: dict = {
        "ruling": "one terminal scoring pass of a frozen model; no fit, no tuning afterward",
        "model_sha256_16": fingerprint,
        "cv_reference": CV_REFERENCE,
        "all": calibration(o),
        "by_h": {int(h): calibration(o[o["h"] == h]) for h in REPORT_H},
        "by_compound_f": {c: calibration(g) for c, g in o.groupby("compound_f")},
        "by_compound_f_h": {
            f"{c} h={h}": calibration(g)
            for (c, h), g in o[o["h"].isin(REPORT_H)].groupby(["compound_f", "h"])
        },
        "by_race": {str(r): calibration(g) for r, g in o.groupby("race")},
        "by_track_seen": {("seen" if k else "unseen"): calibration(g)
                          for k, g in o.groupby("track_seen")},
        "crossing": float(((o["p10"] > o["p50"]) | (o["p50"] > o["p90"])).mean()),
    }
    o.to_parquet(OUT_PATH, index=False)
    REPORT_PATH.write_text(json.dumps(res, indent=2), encoding="utf-8")

    pd.set_option("display.width", 220)
    print("FROZEN MODEL SCORED ON THE HOLDOUT, ONE PASS. No fit, no tuning afterward.")
    print(f"  artefacts {fingerprint}")
    print(f"  {o['race'].nunique()} holdout races, {len(o)} rows, "
          f"{o['event'].nunique()} distinct events")
    print(f"  quantile crossing {res['crossing']:.4f}")

    def table(d: dict, label: str) -> None:
        t = pd.DataFrame(d).T[["n", "frac_below_p10", "frac_above_p90", "median_width_s",
                               "median_resid_s", "mae_p50", "race_offset_sd"]]
        print(f"\n{label}")
        print(t.round(3).to_string())

    print("\nTRAIN CV REFERENCE: below p10 10.7%, above p90 10.2%, median resid 0.009, "
          "MAE 0.506, race offset sd 0.168")
    table({"holdout": res["all"]}, "overall")
    table(res["by_h"], "by horizon")
    table(res["by_compound_f"], "by future compound")
    table(res["by_compound_f_h"], "by future compound and horizon")
    table(res["by_track_seen"],
          "by whether the train split contains that track (cut fixed before the run)")
    table(res["by_race"], "by race (9 races, so a single bad race is visible)")
    print(f"\nwrote {OUT_PATH} and {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
