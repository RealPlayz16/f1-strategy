"""Session 5 results report: docs/session5_results.md from the backtest outputs.

Usage:
    python -m src.live_report
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from src.live import ACCOUNTS_FOR, NOT_ACCOUNTED

BACKTEST = Path("data/processed/backtest.json")
DECISIONS = Path("data/processed/backtest_decisions.parquet")
BREAK_EVEN = Path("data/processed/break_even.parquet")
OUT = Path("docs/session5_results.md")
PHIS = ("0.05", "0.08", "0.12")
TRIGGER_P = 0.5  # "would trigger a pit": P(pit better) >= 0.5 at rho 0.5

FRAMING = (
    "This is a working live decision system, demonstrated end to end, with honest "
    "uncertainty. It is not a validated strategy system. The holdout contains one real "
    "neutralisation (2025 United States, VSC lap 7), and one real neutralisation cannot "
    "validate anything."
)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(
            f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def break_even_section(be: pd.DataFrame, p_calls: dict) -> list[str]:
    out = ["## Break-even precision (headline)", "",
           "For a hypothetical Fast Flag call at the end of every 10th lap, for every running "
           "car, p* is the call precision above which pitting on the call beats ignoring it. "
           "Free-air time only: the track-position benefit of stopping under a neutralisation "
           "is not modelled, so the gain from a real call is understated and **every p* here "
           "is an upper bound** on the precision really needed.", ""]
    decided = ~be["decision"].astype(str).str.startswith(("no ", "race"))
    out.append(f"Hypothetical calls: {len(be)}; with a decision: {int(decided.sum())} "
               f"(the rest have no tyre anchor yet or the race is ending).")
    out.append("")
    rows = []
    for kind in ("SC", "VSC"):
        k = be[decided & (be["kind"] == kind)]
        prec = p_calls[kind]["p"]
        for phi in PHIS:
            col = f"p_star_phi{phi}"
            p = k[col]
            never = p.isna()
            anyway = p == 0
            interior = p[(p > 0) & p.notna()]
            beats = ((p.notna()) & (p < prec)).mean()
            rows.append({
                "kind": kind, "phi": phi, "n": len(k),
                "pit anyway (p*=0)": f"{anyway.mean():.0%}",
                "never pit (no p*)": f"{never.mean():.0%}",
                "p* median (when 0<p*<inf)": (f"{interior.median():.2f}" if len(interior)
                                               else "n/a"),
                "p* 10-90%": (f"{interior.quantile(0.1):.2f}-{interior.quantile(0.9):.2f}"
                              if len(interior) else "n/a"),
                f"acting beats ignoring at FF precision {prec:.2f}": f"{beats:.0%}",
            })
    out += [md_table(pd.DataFrame(rows)), ""]
    k = be[decided & (be["kind"] == "SC")]
    if len(k):
        need = k[k["stay_out_stops"] > 0]
        free = k[k["stay_out_stops"] == 0]
        out.append("Split at phi 0.08, SC calls: cars whose stay-out plan still needs a stop "
                   f"(n={len(need)}): p* median "
                   f"{need['p_star_phi0.08'].median():.2f}, never-pit "
                   f"{need['p_star_phi0.08'].isna().mean():.0%}; cars that can run to the flag "
                   f"(n={len(free)}): never-pit {free['p_star_phi0.08'].isna().mean():.0%}.")
        out.append("")
    return out


def false_call_section(dec: pd.DataFrame, races: list[dict]) -> list[str]:
    out = ["## False calls", ""]
    rows = []
    for race in races:
        for i, e in enumerate(race["episodes"]):
            rows.append({"race": race["race"], "episode": i, "kind called": e["kind"],
                         "real": e["real"], "actual": e.get("actual_kind"),
                         "lead_s": e.get("lead_s")})
    eps = pd.DataFrame(rows)
    if eps.empty:
        return out + ["Fast Flag made no SC / VSC calls in the built holdout races.", ""]
    out += [md_table(eps.fillna("")), ""]
    false_eps = eps[~eps["real"].astype(bool)]
    if false_eps.empty:
        out.append("No false calls in the built holdout races, so no realised false-call "
                   "cost; the break-even table above is the general answer.")
        return out + [""]
    d = dec[(dec["source"] == "call") & (dec["p50_bias_s"] == 0)]
    for _, e in false_eps.iterrows():
        g = d[(d["race"] == e["race"]) & (d["tag"] == str(e["episode"]))]
        decided = g[~g["decision"].astype(str).str.startswith(("no ", "race"))]
        trig = decided[decided["p_pit_better_rho0.5"] >= TRIGGER_P]
        cost = trig["gain_if_false_p50_rho0.5"]
        out.append(f"- {e['race']} episode {e['episode']}: {len(decided)} cars with a decision, "
                   f"{len(trig)} would have pitted (P >= {TRIGGER_P} at rho 0.5); median "
                   f"free-air time change from pitting on the false call: "
                   f"{cost.median() if len(cost) else float('nan'):+.1f} s (positive = a gain: "
                   f"those cars were in their pit window anyway).")
    return out + [""]


def real_event_section(dec: pd.DataFrame, races: list[dict]) -> list[str]:
    out = ["## The one real neutralisation: 2025 United States, VSC lap 7", ""]
    us = next((r for r in races if r["race"] == "2025_United_States"), None)
    if us is None:
        return out + ["Not built.", ""]
    e = us["episodes"][0] if us["episodes"] else None
    if e:
        out.append(f"- Fast Flag called **{e['kind']}** at t = {e['t_call']:.1f} s "
                   f"(\"{e['reason']}\"); race control deployed a **{e['actual_kind']}** at "
                   f"{e['t_deploy']:.1f} s. Lead {e['lead_s']:.1f} s = "
                   f"{e['lead_s'] / us['median_lap_s']:.2f} laps. Right event, wrong kind.")
    clock = us["clock_check"]
    if clock:
        out.append(f"- Clock check: Fast Flag's official {clock[0]['kind']} vs our "
                   f"sc_events differ by {clock[0]['diff_s']:.2f} s.")
    d = dec[(dec["race"] == "2025_United_States") & (dec["p50_bias_s"] == 0)]
    at_call = d[d["source"] == "call"]["decision"].value_counts().to_dict()
    at_dep = d[d["source"] == "deploy"]["decision"].value_counts().to_dict()
    out.append(f"- Decisions at the call: {at_call}. The call came at the end of lap 5; the "
               "tyre model anchors on at least 2 clean laps from lap 5 on (laps 2-4 are "
               "skipped, early-stint effect), so **the system was blind when the call "
               "arrived**. The lead time bought nothing on the one real event.")
    out.append(f"- Decisions at the deployment (no-call path, P = 1): {at_dep}.")
    dd = d[(d["source"] == "deploy")
           & ~d["decision"].astype(str).str.startswith(("no ", "race"))]
    if len(dd):
        tab = dd[["driver", "compound_now", "p_pit_better_rho0.0", "p_pit_better_rho0.5",
                  "p_pit_better_rho0.9", "gain_if_real_p50_rho0.5", "share_laps_h_gt_30",
                  "pit_now_plan", "stay_out_plan"]].sort_values("driver")
        out += ["", md_table(tab.round(3)), ""]
        soft = dec[(dec["race"] == "2025_United_States") & (dec["source"] == "deploy")
                   & (dec["p50_bias_s"] > 0)]
        flips = 0
        for drv, g in dd.groupby("driver"):
            s = soft[soft["driver"] == drv]
            if len(s) and (s["p_pit_better_rho0.5"].iloc[0] >= 0.5) != (
                    g["p_pit_better_rho0.5"].iloc[0] >= 0.5):
                flips += 1
        out.append(f"- Long-horizon p50 + 0.09 s on MEDIUM and SOFT beyond h = 15 "
                   f"(Session 8 spec, replaces the dead soft + 0.075): "
                   f"{flips} of {len(dd)} cars change side "
                   "of P = 0.5 at rho 0.5.")
        out.append(f"- Share of plan laps beyond h = 30 (tyre intervals a floor there): "
                   f"{dd['share_laps_h_gt_30'].median():.0%}. The probabilities above are "
                   "overconfident by that much.")
    w = pd.DataFrame(us["windows"])
    if len(w):
        extra = int(w["extra_window"].sum())
        changed = int((w["extra_window"] & ~w["no_call_window_inside_neutralisation"]).sum())
        out.append(f"- Early call windows: {extra} of {len(w)} cars got an extra pit window "
                   f"from the call; for {changed} of them the no-call window fell outside the "
                   "neutralisation. **The call confirmed earlier; it changed no car's options.** "
                   f"Teams that pitted during the VSC: {int(w['team_pitted_during'].sum())}.")
    out.append("- Synthetic no-call path (**SYNTHETIC**): the deployment decisions above are "
               "what the system does when Fast Flag misses or is ignored. The holdout has no "
               "real missed neutralisation.")
    return out + [""]


def lead_section(races: list[dict]) -> list[str]:
    out = ["## Lead time in laps", "",
           "Fast Flag's median lead over race control is 32.6 s (its out-of-sample "
           "scorecard). In laps of advance notice per holdout track:", ""]
    rows = [{"race": r["race"], "median lap (s)": round(r["median_lap_s"], 1),
             "32.6 s in laps": round(r["lead_laps_at_ff_median_32_6s"], 2)} for r in races]
    out += [md_table(pd.DataFrame(rows)), "",
            "A third of a lap. An SC (median 3 laps) outlasts that easily, so an early SC call "
            "mostly confirms a decision earlier. A VSC (median 96 s, about one lap) is where a "
            "third of a lap can decide whether a car's next pit window is still inside the "
            "neutralisation; in the one real VSC it did not (the VSC ran 213 s).", ""]
    return out


def main() -> int:
    bt = json.loads(BACKTEST.read_text(encoding="utf-8"))
    dec = pd.read_parquet(DECISIONS) if DECISIONS.exists() else pd.DataFrame()
    be = pd.read_parquet(BREAK_EVEN) if BREAK_EVEN.exists() else pd.DataFrame()
    p_calls = bt["p_calls"]
    lines = ["# Session 5 results: live Fast Flag decision system", "", f"**{FRAMING}**", "",
             "Every decision output carries these lists:", "", "Accounts for:"]
    lines += [f"- {x}" for x in ACCOUNTS_FOR] + ["", "Does not account for:"]
    lines += [f"- {x}" for x in NOT_ACCOUNTED] + [""]
    lines += [f"Fast Flag call precision (its out-of-sample scorecard): SC "
              f"{p_calls['SC']['hits']}/{p_calls['SC']['n']}, P = {p_calls['SC']['p']:.2f} "
              f"(90% {p_calls['SC']['q05']:.2f}-{p_calls['SC']['q95']:.2f}); VSC "
              f"{p_calls['VSC']['hits']}/{p_calls['VSC']['n']}, P = {p_calls['VSC']['p']:.2f} "
              f"(90% {p_calls['VSC']['q05']:.2f}-{p_calls['VSC']['q95']:.2f}).", ""]
    built = [r["race"] for r in bt["races"]]
    n_hold = len(sorted(be["race"].unique())) if len(be) else len(built)
    lines += [
        f"Holdout races with a Fast Flag timeline: {', '.join(built)}.",
        "",
        "### Two different sample sizes in this report, do not mix them",
        "",
        f"The holdout is **{n_hold} races**, and the hypothetical-call analysis below uses all "
        f"of them. The **real-call** evidence does not: a Fast Flag recommendation exists only "
        f"for a race Fast Flag has built a timeline for, and it has built "
        f"**{len(built)}**. Expanding the race set in Session 8 scaled the hypothetical sample "
        f"and moved the real-call sample not at all.",
        "",
        "**The real-call path is unchanged at n = 1**: one real neutralisation, 2025 United "
        "States, VSC lap 7. A larger break-even sample is more coverage of race *situations*, "
        "not more validation of the live decision system. The ceiling is set by an upstream "
        "dependency this project does not control, so no amount of work here raises it. See "
        "\"VALIDATION CEILING\" in HANDOFF.md.",
        "",
    ]
    if len(be):
        lines += break_even_section(be, p_calls)
    lines += false_call_section(dec, bt["races"])
    lines += real_event_section(dec, bt["races"])
    lines += lead_section(bt["races"])
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
