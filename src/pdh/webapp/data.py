"""
Data-loading layer for the Streamlit app: wraps existing pdh modules and
the current season's data/season_<tag>/game_weeks/gwN/ outputs (tag from
config/seasons.yaml, see pdh.seasons) rather than recomputing anything - run
scripts/make_recommendations.py separately to (re)generate projections.
"""

from __future__ import annotations
import json
import re
from pathlib import Path

import pandas as pd

from pdh import fpl
from pdh.projections import to_canon_pos
from pdh.seasons import current_season_dir
from pdh.sleeper import (
    get_league_users,
    get_sleeper_rosters,
    get_draft_picks,
    draft_picks_df,
    normalize_squad_df,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "config"
SEASON_DIR = current_season_dir()
GAME_WEEKS_DIR = SEASON_DIR / "game_weeks"
SQUADS_PATH = SEASON_DIR / "current_squads.csv"


def load_league_config() -> tuple[dict, dict]:
    """config/sleeper_setup.json, config/sleeper_draft.json - refresh via
    scripts/refresh_league_config.py once the draft order/roster settings
    change (e.g. right after the commissioner randomizes draft order)."""
    setup = json.loads((CONFIG_DIR / "sleeper_setup.json").read_text())
    draft = json.loads((CONFIG_DIR / "sleeper_draft.json").read_text())
    return setup, draft


def load_league_teams() -> pd.DataFrame:
    """One row per Sleeper team: roster_id, owner_id, display_name, and the
    custom `team_name` set in the Sleeper app (falls back to display_name)."""
    users = pd.DataFrame(get_league_users())
    users["team_name"] = users["metadata"].apply(
        lambda m: (m or {}).get("team_name") if isinstance(m, dict) else None
    )
    users["team_name"] = users["team_name"].fillna(users["display_name"])
    rosters = get_sleeper_rosters()
    teams = rosters[["roster_id", "owner_id"]].merge(
        users[["user_id", "display_name", "team_name"]],
        left_on="owner_id",
        right_on="user_id",
        how="left",
    )
    return teams.sort_values("roster_id").reset_index(drop=True)


def load_squads_normalized() -> pd.DataFrame:
    squads = pd.read_csv(SQUADS_PATH)
    return normalize_squad_df(squads)


def load_live_draft_picks() -> pd.DataFrame:
    """
    Live picks so far, one row per pick, joined with the drafting team's
    name (see load_league_teams), matched FPL web_name (see
    pdh.sleeper.draft_picks_df), and a canonical `pos` (F/M/D/GK, from
    current_squads.csv - needed for positional_scarcity's team-needs check).
    Empty DataFrame before the draft starts.
    """
    _setup, draft = load_league_config()
    draft_id = draft.get("draft_id")
    if not draft_id:
        return pd.DataFrame()
    picks = get_draft_picks(draft_id)
    if not picks:
        return pd.DataFrame()
    squads_n = load_squads_normalized()
    picks_df = draft_picks_df(picks, squads_n)
    teams = load_league_teams()
    picks_df = picks_df.merge(
        teams[["roster_id", "team_name", "display_name"]], on="roster_id", how="left"
    )
    squads_pos = squads_n[["web_name", "pos"]].drop_duplicates(subset=["web_name"]).copy()
    squads_pos["pos"] = squads_pos["pos"].apply(to_canon_pos)
    picks_df = picks_df.merge(squads_pos, on="web_name", how="left")
    return picks_df.sort_values("pick_no").reset_index(drop=True)


def my_draft_slot(draft: dict, my_team_id: str) -> int | None:
    """This team's 1-indexed snake-draft slot, from draft_order (a {user_id:
    slot} dict Sleeper only populates once the commissioner starts the
    draft) - None beforehand."""
    draft_order = draft.get("draft_order")
    if not draft_order:
        return None
    slot = draft_order.get(my_team_id)
    return int(slot) if slot is not None else None


def slot_to_roster_id_map(draft: dict, teams_df: pd.DataFrame) -> dict[int, int]:
    """{draft slot: roster_id}, via draft_order ({user_id: slot}) and
    load_league_teams's user_id<->roster_id mapping. Empty before draft_order
    is set."""
    draft_order = draft.get("draft_order")
    if not draft_order:
        return {}
    user_to_roster = dict(zip(teams_df["user_id"], teams_df["roster_id"]))
    return {
        int(slot): user_to_roster[user_id]
        for user_id, slot in draft_order.items()
        if user_id in user_to_roster
    }


def available_gameweeks() -> list[str]:
    """gwN folders only (the game_weeks dir may also hold ad-hoc
    experimental gwN_suffix folders from earlier testing that aren't real
    per-gameweek outputs - excluded here)."""
    if not GAME_WEEKS_DIR.exists():
        return []
    names = [
        p.name
        for p in GAME_WEEKS_DIR.iterdir()
        if p.is_dir() and re.fullmatch(r"gw\d+", p.name)
    ]
    return sorted(names, key=lambda s: int(s[2:]))


def load_draft_board(gw_tag: str) -> pd.DataFrame:
    """draft_board_flexaware.csv for a gameweek (VOR-ranked, position/team
    aware) - see scripts/make_recommendations.py section 8. Empty DataFrame
    if that gameweek hasn't been computed yet."""
    path = GAME_WEEKS_DIR / gw_tag / "draft_board_flexaware.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def current_fpl_gameweek_tag() -> str:
    """The gwN folder name matching pdh.fpl.current_gameweek(), for
    defaulting the app to "this week's" board without asking the user."""
    return f"gw{fpl.current_gameweek()}"


def best_available(board: pd.DataFrame, drafted_web_names: set[str]) -> pd.DataFrame:
    """`board` (see load_draft_board) with already-drafted players removed,
    ranked by VOR. Filtering is done live against `drafted_web_names` (from
    load_live_draft_picks) rather than relying on taken.csv being fresh, so
    the app stays accurate even between script reruns."""
    if board.empty:
        return board
    lowered = {n.lower() for n in drafted_web_names}
    out = board[~board["web_name"].str.lower().isin(lowered)].copy()
    return out.sort_values("VOR", ascending=False).reset_index(drop=True)
