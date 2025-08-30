"""
Normalize and engineer features for Sleeper-like scoring from FBref match logs.
"""

from __future__ import annotations
import pandas as pd, numpy as np

CANDIDATES = {
    # attacking
    "goals": ["summary_Performance_Gls"],
    "assists": ["summary_Performance_Ast", "passing_Ast"],
    "shots_on_target": ["summary_Performance_SoT"],
    "shots_total": ["summary_Performance_Sh"],
    "key_passes": ["passing_KP"],
    # defensive / possession
    "tackles_won": ["defense_Tackles_TklW", "misc_Performance_TklW"],
    "interceptions": ["defense_Int", "summary_Performance_Int", "misc_Performance_Int"],
    "blocks_shots": ["defense_Blocks_Blocks"],
    "aerials_won": ["misc_Aerial Duels_Won"],
    "dribbles_completed": ["possession_Take-Ons_Succ"],
    "crosses_completed": ["passing_types_Pass Types_Crs"],
    "dispossessed": ["possession_Carries_Dis"],
    "clearances": ["defense_Clr"],
    # cards
    "yellow_cards": ["summary_Performance_CrdY", "misc_Performance_CrdY"],
    "red_cards": ["summary_Performance_CrdR", "misc_Performance_CrdR"],
    "second_yellow": ["misc_Performance_2CrdY"],
    # GK
    "saves": ["keepers_Shot Stopping_Saves"],
    "goals_conceded": ["keepers_Shot Stopping_GA"],
}

# MINUTES: prefer outfield minutes first, then fallback to keepers (row-wise!)
MINUTE_CANDS = [
    "summary_min",
    "passing_min",
    "defense_min",
    "possession_min",
    "misc_min",
    "keepers_min",
]

META_CANDS = {
    "season": ["season"],
    "date": ["date"],
    "team": ["team", "squad"],
    "opponent": ["opponent"],
    "position_raw": [
        "summary_pos",
        "passing_pos",
        "defense_pos",
        "possession_pos",
        "misc_pos",
    ],
    "player": ["player"],
}


def _copy_first_present(
    src: pd.DataFrame, dst: pd.DataFrame, new: str, options: list[str]
):
    for c in options:
        if c in src.columns:
            dst[new] = src[c]
            return


def _coalesce_rowwise(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    avail = [c for c in cols if c in df.columns]
    if not avail:
        return pd.Series(np.nan, index=df.index)
    return df[avail].bfill(axis=1).iloc[:, 0]


def normalize_player_matches(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # metadata
    for k, opts in META_CANDS.items():
        if k not in out.columns:
            _copy_first_present(out, out, k, opts)

    # stats -> canonical
    for canon, opts in CANDIDATES.items():
        if canon not in out.columns:
            _copy_first_present(out, out, canon, opts)

    # minutes (row-wise coalesce!)
    out["minutes"] = _coalesce_rowwise(out, MINUTE_CANDS)

    # penalties missed (optional convenience)
    if (
        "summary_Performance_PKatt" in out.columns
        and "summary_Performance_PK" in out.columns
    ):
        out["penalties_missed"] = (
            out["summary_Performance_PKatt"] - out["summary_Performance_PK"]
        )

    # GK clean sheet
    if "goals_conceded" in out.columns:
        out["clean_sheet"] = (
            (out["goals_conceded"] == 0) & (out["minutes"] >= 60)
        ).astype(int)

    # per-90 for present stats
    m = out["minutes"].replace(0, np.nan)
    for c in [
        "goals",
        "assists",
        "shots_on_target",
        "shots_total",
        "key_passes",
        "tackles_won",
        "interceptions",
        "blocks_shots",
        "aerials_won",
        "dribbles_completed",
        "crosses_completed",
        "dispossessed",
        "clearances",
        "saves",
    ]:
        if c in out.columns:
            out[c + "_p90"] = out[c] / (m / 90.0)
    out["minutes"] = out["minutes"].fillna(0)

    return out
