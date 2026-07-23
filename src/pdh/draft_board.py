"""
Draft board with VOR (value over replacement) using roster settings.
"""

from __future__ import annotations
import json
import pandas as pd
from typing import Dict, List

FLEX_MAP = {"FM_FLEX": {"F", "M"}, "MD_FLEX": {"M", "D"}}


def roster_requirements(roster_positions: List[str]) -> Dict[str, int]:
    base = {"F": 0, "M": 0, "D": 0, "GK": 0, "FM_FLEX": 0, "MD_FLEX": 0, "BN": 0}
    for p in roster_positions:
        base[p] = base.get(p, 0) + 1
    return base


def replacement_ranks(req: Dict[str, int], teams: int) -> Dict[str, int]:
    # Pure starters at each position (not using flex yet)
    return {
        "F": req.get("F", 0) * teams,
        "M": req.get("M", 0) * teams,
        "D": req.get("D", 0) * teams,
        "GK": req.get("GK", 0) * teams,
    }


def effective_replacement_ranks_with_flex(
    df, pos_col, proj_col, roster_positions, teams=8
):
    """
    Estimate replacement ranks including FLEX slots by simulating how global pools would be filled.
    FM_FLEX allows F or M. MD_FLEX allows M or D.
    We fill base position counts, then for each team’s flex slot, allocate it to the position
    whose next-available player has the higher projection.
    """
    # Base counts per team
    base = {"F": 0, "M": 0, "D": 0, "GK": 0}
    flex_counts = {"FM_FLEX": 0, "MD_FLEX": 0}
    for p in roster_positions:
        if p in base:
            base[p] += 1
        elif p in flex_counts:
            flex_counts[p] += 1

    # Global totals before flex
    counts = {k: base[k] * teams for k in base}

    # Sorted projection pools per position
    pools = {}
    for p in base:
        vals = (
            df[df[pos_col] == p]
            .sort_values(proj_col, ascending=False)[proj_col]
            .to_list()
        )
        pools[p] = vals

    def next_score(p, idx):  # safe access
        arr = pools[p]
        return arr[idx] if idx < len(arr) else float("-inf")

    # Allocate FM_FLEX (F or M) and MD_FLEX (M or D) per team
    for _ in range(flex_counts.get("FM_FLEX", 0) * teams):
        f_score = next_score("F", counts["F"])
        m_score = next_score("M", counts["M"])
        if f_score >= m_score:
            counts["F"] += 1
        else:
            counts["M"] += 1

    for _ in range(flex_counts.get("MD_FLEX", 0) * teams):
        m_score = next_score("M", counts["M"])
        d_score = next_score("D", counts["D"])
        if m_score >= d_score:
            counts["M"] += 1
        else:
            counts["D"] += 1

    return counts  # replacement ranks including flex


def vor_from_ranks(
    df: pd.DataFrame,
    repl_ranks: Dict[str, int],
    pos_col: str = "pos",
    proj_col: str = "proj_points",
) -> pd.DataFrame:
    """
    VOR = player's projected points minus the projected points of the
    replacement-level player at that position, given pre-computed
    replacement ranks per position (see replacement_ranks for the pure-
    position case, effective_replacement_ranks_with_flex for the flex-aware
    case - both a rank K means "the Kth-best player at that position is
    replacement level").
    """
    rows = []
    for pos, k in repl_ranks.items():
        sub = (
            df[df[pos_col] == pos]
            .sort_values(proj_col, ascending=False)
            .reset_index(drop=True)
        )
        if len(sub) == 0:
            continue
        k = max(1, int(k))
        repl_points = sub.iloc[min(len(sub) - 1, k - 1)][proj_col]
        sub = sub.copy()
        sub["replacement_at_rank"] = k
        sub["replacement_points"] = repl_points
        sub["VOR"] = sub[proj_col] - repl_points
        rows.append(sub)
    if not rows:
        return pd.DataFrame(
            columns=[
                pos_col,
                proj_col,
                "replacement_at_rank",
                "replacement_points",
                "VOR",
            ]
        )
    return pd.concat(rows, ignore_index=True).sort_values("VOR", ascending=False)


def compute_vor(
    df: pd.DataFrame,
    pos_col="pos",
    proj_col="proj_points",
    req: Dict[str, int] = None,
    teams=8,
) -> pd.DataFrame:
    """
    VOR for the pure-position case (flex slots ignored, to keep replacement
    ranks conservative) - see vor_from_ranks for the underlying computation.
    """
    if req is None:
        raise ValueError("Provide roster requirements dict")
    repl = replacement_ranks(req, teams=teams)
    return vor_from_ranks(df, repl, pos_col=pos_col, proj_col=proj_col)
