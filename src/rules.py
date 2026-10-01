"""Race rules that constrain the Session 4 optimizer and race engine.

Simple and explicit on purpose: these bound the DP optimizer's search space.

Sources: FIA Formula One Sporting Regulations 2023-2025. Article numbers are not verified
here; each rule names its topic so it can be checked against the current text.
  Compound rule (tyres): in a dry race each car must use at least two different dry
    compounds. Waived when intermediate or wet tyres are used.
  Safety Car / VSC: no overtaking while the SC or VSC is deployed (track status 4, 6, 7).
    Under SC lapped cars are normally allowed to unlap before the restart (race director's
    call); modelled as a switch.
  DRS: enabled from the third racing lap and two laps after a Safety Car restart, only within
    1 s of the car ahead at the detection point.
  Red flag (suspension): cars may change tyres in the pit lane during the suspension, so a
    stop there costs no pit loss.

Data-derived, not regulation:
  MAX_STINT_LAPS: longest stint per compound that ended in a pit stop on train races
    (stints run to the flag are not counted, so this is a floor on viability).
  A race may impose a per-set lap limit (2023 Qatar); pass it as tyre_limit_laps.

Usage:
    python -m src.rules
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
WET_COMPOUNDS = ("INTERMEDIATE", "WET")
MAX_STINT_LAPS = {"SOFT": 28, "MEDIUM": 46, "HARD": 53}
MIN_STINT_LAPS = 1
NO_OVERTAKE_STATUS = ("4", "5", "6", "7")  # SC, red, VSC, VSC ending
DRS_FIRST_LAP = 3
DRS_LAPS_AFTER_RESTART = 2
DRS_GAP_S = 1.0
UNLAP_UNDER_SC = True


@dataclass(frozen=True)
class Stint:
    compound: str
    laps: int


def strategy_violations(stints: list[Stint], race_laps: int,
                        tyre_limit_laps: int | None = None) -> list[str]:
    """Empty list if the plan is legal and viable, otherwise one message per problem."""
    problems = []
    total = sum(s.laps for s in stints)
    if total != race_laps:
        problems.append(f"stints cover {total} laps, race has {race_laps}")
    compounds = [s.compound for s in stints]
    unknown = [c for c in compounds if c not in DRY_COMPOUNDS + WET_COMPOUNDS]
    if unknown:
        problems.append(f"unknown compounds {unknown}")
    wet = any(c in WET_COMPOUNDS for c in compounds)
    if not wet and len({c for c in compounds if c in DRY_COMPOUNDS}) < 2:
        problems.append("dry race needs at least two different dry compounds")
    for i, s in enumerate(stints):
        if s.laps < MIN_STINT_LAPS:
            problems.append(f"stint {i + 1} shorter than {MIN_STINT_LAPS} lap")
        cap = MAX_STINT_LAPS.get(s.compound)
        if cap is not None and s.laps > cap:
            problems.append(f"stint {i + 1} {s.compound} {s.laps} laps exceeds {cap}")
        if tyre_limit_laps is not None and s.laps > tyre_limit_laps:
            problems.append(f"stint {i + 1} exceeds the race tyre limit of {tyre_limit_laps}")
    return problems


def can_overtake(track_status: str) -> bool:
    """Track status is a string of single-digit codes, never an int (see clean.py)."""
    return not any(code in str(track_status) for code in NO_OVERTAKE_STATUS)


def drs_enabled(lap: int, laps_since_restart: int | None, gap_s: float) -> bool:
    """laps_since_restart: racing laps completed since the last SC restart, None if none."""
    if lap < DRS_FIRST_LAP or gap_s >= DRS_GAP_S:
        return False
    return laps_since_restart is None or laps_since_restart >= DRS_LAPS_AFTER_RESTART


def pit_loss_s(track: dict, condition: str, phi: float) -> float:
    """Pit loss for one (season, event) row of pitloss_by_track.parquet.
    condition: green, sc, vsc or red (tyres changed during a suspension cost nothing)."""
    if condition == "red":
        return 0.0
    if condition == "green":
        return float(track["green_s"])
    lap_cond = track[f"lap_{condition}_s"]
    return float(track["green_s"] - phi * (lap_cond - track["lap_green_s"]))


def main() -> int:
    plan = [Stint("MEDIUM", 25), Stint("HARD", 32)]
    print("example plan:", plan, "->", strategy_violations(plan, 57) or "legal")
    bad = [Stint("SOFT", 30), Stint("SOFT", 27)]
    print("example plan:", bad, "->", strategy_violations(bad, 57))
    return 0


if __name__ == "__main__":
    sys.exit(main())
