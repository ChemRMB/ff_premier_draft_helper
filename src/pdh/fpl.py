"""
FPL public endpoints: squads, fixtures, per-player history.
Docs (community):
- bootstrap-static, fixtures, element-summary endpoints.
"""

from __future__ import annotations
import requests
import pandas as pd
from typing import Optional, Dict, Any

BASE = "https://fantasy.premierleague.com/api"


def _get(url: str) -> Any:
    r = requests.get(
        url, headers={"User-Agent": "Mozilla/5.0 (compatible; PremierDraftHelper/1.0)"}
    )
    r.raise_for_status()
    return r.json()


def get_bootstrap() -> Dict[str, Any]:
    """Teams, players ('elements'), positions ('element_types'), and gameweeks ('events')."""
    return _get(f"{BASE}/bootstrap-static/")


def get_fixtures(event: Optional[int] = None) -> pd.DataFrame:
    """All fixtures, or a single GW if event provided. Includes 'event' (GW), 'team_h', 'team_a', difficulty, and per-fixture stats refs."""
    if event is None:
        data = _get(f"{BASE}/fixtures/")
    else:
        data = _get(f"{BASE}/fixtures/?event={event}")
    return pd.DataFrame(data)


def get_element_summary(player_id: int) -> Dict[str, Any]:
    """Per-player history (match by match) and season summary."""
    return _get(f"{BASE}/element-summary/{player_id}/")


def current_squads_df() -> pd.DataFrame:
    """Return current season player list with team, position, status, and chance_of_playing flags."""
    boot = get_bootstrap()
    elements = pd.DataFrame(boot["elements"])
    teams = pd.DataFrame(boot["teams"])[
        [
            "id",
            "name",
            "short_name",
            "strength",
            "strength_attack_home",
            "strength_attack_away",
            "strength_defence_home",
            "strength_defence_away",
        ]
    ]
    positions = pd.DataFrame(boot["element_types"])[["id", "plural_name_short"]].rename(
        columns={"id": "element_type", "plural_name_short": "pos"}
    )
    df = elements.merge(positions, on="element_type", how="left").merge(
        teams, left_on="team", right_on="id", how="left", suffixes=("", "_team")
    )
    keep = [
        "id",
        "web_name",
        "first_name",
        "second_name",
        "now_cost",
        "status",
        "chance_of_playing_next_round",
        "chance_of_playing_this_round",
        "pos",
        "name",
        "short_name",
    ]
    return df[keep].rename(
        columns={"id": "element_id", "name": "team_name", "short_name": "team_code"}
    )


def fixtures_df() -> pd.DataFrame:
    """All fixtures for the season, with GW numbers and difficulties."""
    fix = get_fixtures()
    cols = [
        "id",
        "event",
        "kickoff_time",
        "team_h",
        "team_a",
        "team_h_score",
        "team_a_score",
        "team_h_difficulty",
        "team_a_difficulty",
        "finished",
        "started",
        "minutes",
    ]
    return fix[cols]
