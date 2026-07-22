"""
soccerdata (Understat) accessor for player-level expected-goals/assists stats.
See: https://soccerdata.readthedocs.io/

Understat exposes xg, xa, np_xg (non-penalty xG), xg_chain and xg_buildup at
player-season and player-match granularity, plus per-shot xG via shot events -
metrics FBref (src/pdh/fbref.py) doesn't provide, and which are more
predictive of future scoring than raw goals/assists in small samples.

Unlike FBref, soccerdata's Understat reader is a plain requests-based scraper
(`BaseRequestsReader`) - no SeleniumBase/browser workaround needed, so this
module is much simpler than fbref.py. It is NOT trouble-free though: repeated
automated requests to understat.com in a short window have been observed to
get a stub homepage back (HTTP 200, ~13KB, no embedded season data) instead of
the real page, causing `read_seasons()`/`read_player_season_stats()` etc. to
raise `KeyError` ('statData' or 'stat'). This isn't a soccerdata version bug -
it reproduces on soccerdata 1.9.0 with a from-scratch cache. Treat a failure
here as "retry later" (soccerdata caches successful per-page responses on
disk, so a working run won't need to re-fetch), not as "Understat has no
data".
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOCCERDATA_BASE_DIR = ROOT / "data" / "cache" / "soccerdata"
os.environ.setdefault("SOCCERDATA_DIR", str(SOCCERDATA_BASE_DIR))

import soccerdata as sd

UNDERSTAT_LEAGUE = "ENG-Premier League"


def season_to_understat_str(end_year: int) -> str:
    """
    Convert an FBref-style end-year season (e.g. 2025 for the 2024/25 season -
    the convention used in config/seasons.yaml and src/pdh/fbref.py) to
    Understat's "YYYY/YYYY" season string (e.g. "2024/2025").
    """
    return f"{end_year - 1}/{end_year}"


def _reader(seasons: Iterable[int], leagues: str = UNDERSTAT_LEAGUE) -> sd.Understat:
    season_strs = [season_to_understat_str(s) for s in seasons]
    return sd.Understat(leagues=leagues, seasons=season_strs)


def _reset_index(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index() if df.index.names and df.index.names[0] is not None else df


def player_season_stats(seasons: Iterable[int]) -> pd.DataFrame:
    """
    Player-season xG/xA stats: goals, xg, np_goals, np_xg, assists, xa, shots,
    key_passes, xg_chain, xg_buildup, minutes, matches, per player/team/season.
    `seasons` are FBref-style end years.

    Note: Understat only lists a season once it has started - querying a
    season before kickoff returns an empty frame, not an error.
    """
    return _reset_index(_reader(seasons).read_player_season_stats())


def player_match_stats(seasons: Iterable[int]) -> pd.DataFrame:
    """Player-match xG/xA stats (one row per player per match)."""
    return _reset_index(_reader(seasons).read_player_match_stats())


def shot_events(seasons: Iterable[int]) -> pd.DataFrame:
    """
    Shot-level events with per-shot xG - the finest granularity Understat
    offers, useful for e.g. penalty/set-piece xG splits.
    """
    return _reset_index(_reader(seasons).read_shot_events())
