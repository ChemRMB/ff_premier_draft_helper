"""
soccerdata (FBref) accessors for team + player stats (last 3 seasons).
See: https://soccerdata.readthedocs.io/
"""

from __future__ import annotations
import pandas as pd
from typing import Iterable, List
import soccerdata as sd

FBREF_LEAGUE = "ENG-Premier League"


def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    # If MultiIndex columns (e.g., ('Performance','Gls')), flatten to 'Performance_Gls'
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join([str(x) for x in tup if str(x) != ""]).strip("_")
            for tup in df.columns
        ]
    return df


def player_match_stats(seasons: Iterable[int]) -> pd.DataFrame:
    """
    Return player-match stats by merging summary, passing, passing_types, defense, possession, and misc tables.
    Seasons are end years (e.g., 2023 for 2022/23).
    """
    fb = sd.FBref(leagues=FBREF_LEAGUE, seasons=list(seasons))

    def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
        # If MultiIndex columns (e.g., ('Performance','Gls')), flatten to 'Performance_Gls'
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = [
                "_".join([str(x) for x in tup if str(x) != ""]).strip("_")
                for tup in df.columns
            ]
        return df

    def _read(stat_type: str, name: str) -> tuple[pd.DataFrame, set[str]]:
        # Bring index levels back as columns, then flatten column names
        dfi = fb.read_player_match_stats(stat_type=stat_type)
        if dfi.index.names is not None:
            dfi = dfi.reset_index()
        dfi = _flatten_cols(dfi)

        # Candidate ID columns that often exist across tables
        candidates = [
            "league",
            "competition",
            "season",
            "comp_season",
            "game",
            "game_id",
            "date",
            "team",
            "squad",
            "opponent",
            "player",
            "player_id",
        ]
        keys = [c for c in candidates if c in dfi.columns]

        # Prefix only metric (non-key) columns to avoid name clashes across tables
        metric_cols = [c for c in dfi.columns if c not in keys]
        dfi = dfi[keys + metric_cols]
        dfi = dfi.rename(columns={c: f"{name}_{c}" for c in metric_cols})
        return dfi, set(keys)

    summary, ksum = _read("summary", "summary")
    passing, kpas = _read("passing", "passing")
    ptypes, kpt = _read("passing_types", "passing_types")
    defense, kdef = _read("defense", "defense")
    possession, kpos = _read("possession", "possession")
    misc, kmisc = _read("misc", "misc")
    keepers, kkeep = _read("keepers", "keepers")

    # Compute intersection of keys present in ALL tables
    # common_keys = list(ksum & kpas & kpt & kdef & kpos & kmisc)
    common_keys = list(ksum & kpas & kpt & kdef & kpos & kmisc & kkeep)
    if not common_keys:
        raise ValueError(
            "No common join keys found across FBref tables. Inspect raw frames to see available ID columns."
        )

    # Merge on shared keys
    df = (
        summary.merge(passing, on=common_keys, how="outer")
        .merge(ptypes, on=common_keys, how="outer")
        .merge(defense, on=common_keys, how="outer")
        .merge(possession, on=common_keys, how="outer")
        .merge(misc, on=common_keys, how="outer")
        .merge(keepers, on=common_keys, how="outer")
    )

    return df


def team_season_stats(seasons: Iterable[int]) -> pd.DataFrame:
    fb = sd.FBref(leagues=FBREF_LEAGUE, seasons=list(seasons))
    team_std = fb.read_team_season_stats(stat_type="standard")
    if team_std.index.names is not None:
        team_std = team_std.reset_index()
    team_std = flatten_cols(team_std)
    team_std["season"] = team_std["season"].astype(str)
    return team_std


def schedule(seasons: Iterable[int], leagues=FBREF_LEAGUE) -> pd.DataFrame:
    fb = sd.FBref(leagues=leagues, seasons=list(seasons))
    return fb.read_schedule()
