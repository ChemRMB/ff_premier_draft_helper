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


def snake_pick_sequence(teams: int, rounds: int) -> List[int]:
    """Full pick order for a snake draft, as team slot numbers (1..teams),
    one entry per overall pick. Round 1 is 1..teams, round 2 reverses, etc."""
    seq: List[int] = []
    for r in range(1, rounds + 1):
        order = list(range(1, teams + 1)) if r % 2 == 1 else list(range(teams, 0, -1))
        seq.extend(order)
    return seq


def picks_until_next_turn(
    slot_sequence: List[int], n_picks_made: int, my_slot: int
) -> List[int]:
    """
    Team slots that will pick between now (`n_picks_made` picks already made)
    and this team's (`my_slot`) next turn - i.e. every opponent pick you have
    to sit through before you're on the clock again. Empty once the mock/live
    draft is past `my_slot`'s last pick.
    """
    out: List[int] = []
    for s in slot_sequence[n_picks_made:]:
        if s == my_slot:
            break
        out.append(s)
    return out


def team_position_needs(
    team_picks_pos: List[str], roster_positions: List[str]
) -> Dict[str, bool]:
    """
    Which of F/M/D/GK a team still has an unfilled *required* (non-flex,
    non-bench) slot for, given the positions they've drafted so far
    (`team_picks_pos`, e.g. picks_df[picks_df.roster_id==X]["pos"].tolist()).
    Conservative like replacement_ranks/compute_vor: flex slots are ignored,
    so a team is only "still needing" a position while it hasn't filled that
    position's pure starting slots yet - once it has, this stops counting
    them as in the market for it (they might still take one for a flex/bench
    spot, but that's much less predictable, so it's excluded from the
    heuristic rather than guessed at).
    """
    req = roster_requirements(roster_positions)
    base_need = {pos: req.get(pos, 0) for pos in ("F", "M", "D", "GK")}
    have: Dict[str, int] = {}
    for p in team_picks_pos:
        have[p] = have.get(p, 0) + 1
    return {pos: have.get(pos, 0) < n for pos, n in base_need.items()}


def positional_scarcity(
    board: pd.DataFrame,
    picks_df: pd.DataFrame,
    roster_positions: List[str],
    teams: int,
    rounds: int,
    my_slot: int,
    n_picks_made: int,
    slot_to_roster_id: Dict[int, int],
    pos_col: str = "pos",
    vor_col: str = "VOR",
) -> pd.DataFrame:
    """
    Per-position "run risk" heuristic for a live snake draft: given the exact
    picks that will happen before your next turn (from the snake order), how
    many of those picking teams still need that position (see
    team_position_needs) versus how many "quality" (VOR > 0, i.e. still
    above replacement) players are left at it on `board`. A `scarcity_ratio`
    > 1 means more hungry teams than quality players remain at that position
    before you pick again - i.e. a real risk the position "runs out" on you,
    worth weighing against pure VOR when deciding what to take right now.

    This is a heuristic, not a prediction: it assumes a team "in need" of a
    position might take one, not that it definitely will. `picks_df` needs
    `roster_id` and `pos` columns (see pdh.webapp.data.load_live_draft_picks).
    """
    seq = snake_pick_sequence(teams, rounds)
    upcoming_slots = picks_until_next_turn(seq, n_picks_made, my_slot)
    upcoming_roster_ids = [slot_to_roster_id[s] for s in upcoming_slots if s in slot_to_roster_id]

    rows = []
    for pos in ("F", "M", "D", "GK"):
        remaining_quality = int(
            ((board[pos_col] == pos) & (board[vor_col] > 0)).sum()
        )
        demand = 0
        for rid in upcoming_roster_ids:
            team_pos = (
                picks_df.loc[picks_df["roster_id"] == rid, "pos"].tolist()
                if not picks_df.empty
                else []
            )
            if team_position_needs(team_pos, roster_positions).get(pos, False):
                demand += 1
        ratio = demand / max(1, remaining_quality)
        rows.append(
            {
                "pos": pos,
                "remaining_quality": remaining_quality,
                "picks_before_your_turn": len(upcoming_roster_ids),
                "teams_needing": demand,
                "scarcity_ratio": ratio,
            }
        )
    return pd.DataFrame(rows).sort_values("scarcity_ratio", ascending=False).reset_index(
        drop=True
    )


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
