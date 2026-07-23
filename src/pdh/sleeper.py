import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from pdh.namelink import apply_curated_overrides, canonical_name, link_names

# https://docs.sleeper.com/
# NOTE: docs.sleeper.com only documents NFL. Club-soccer usage below
# (sport="clubsoccer:epl") works empirically but is otherwise undocumented -
# endpoint semantics (e.g. week/round numbering) should not be assumed to match
# the NFL docs without verifying against real responses.

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
SLEEPER_SETUP_PATH = CONFIG_DIR / "sleeper_setup.json"
SLEEPER_DRAFT_PATH = CONFIG_DIR / "sleeper_draft.json"
SLEEPER_NAME_CURATED_PATH = CONFIG_DIR / "name_link_sleeper_curated.csv"
SLEEPER_CACHE_DIR = ROOT / "data" / "cache" / "sleeper"

# Seed ids used only to bootstrap refresh_league_config() before
# config/sleeper_setup.json exists/matches the current league. Once refreshed,
# LEAGUE_ID/MY_TEAM_ID below are read from that file instead, so this pair only
# needs updating again if we ever join a genuinely different league.
_SEED_LEAGUE_ID = "1385257445575639040"
_SEED_MY_TEAM_ID = "1259256320754724864"


def _load_league_ids() -> tuple[str, str]:
    if SLEEPER_SETUP_PATH.exists():
        try:
            cfg = json.loads(SLEEPER_SETUP_PATH.read_text())
            return (
                cfg.get("league_id") or _SEED_LEAGUE_ID,
                cfg.get("my_team_id") or _SEED_MY_TEAM_ID,
            )
        except (json.JSONDecodeError, OSError):
            pass
    return _SEED_LEAGUE_ID, _SEED_MY_TEAM_ID


LEAGUE_ID, MY_TEAM_ID = _load_league_ids()


def get_league_info(league_id: str = LEAGUE_ID) -> dict:
    """Fetch league settings/scoring from the Sleeper API."""
    url = f"https://api.sleeper.app/v1/league/{league_id}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def get_league_users(league_id: str = LEAGUE_ID) -> list[dict]:
    """Fetch league members (owners) from the Sleeper API."""
    url = f"https://api.sleeper.app/v1/league/{league_id}/users"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def get_league_drafts(league_id: str = LEAGUE_ID) -> list[dict]:
    """Fetch all drafts associated with a league from the Sleeper API."""
    url = f"https://api.sleeper.app/v1/league/{league_id}/drafts"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def get_draft_picks(draft_id: str) -> list[dict]:
    """
    Fetch all picks made so far in a draft from the Sleeper API. Empty list
    before the draft starts. Each pick has `player_id`/`picked_by`/`round`/
    `pick_no` and a `metadata` dict which - per the NFL-documented shape,
    unverified but observed consistent for club soccer - includes that
    pick's `first_name`/`last_name`/`team` directly, so no separate
    get_sleeper_players() lookup is needed to identify who was picked.
    """
    url = f"https://api.sleeper.app/v1/draft/{draft_id}/picks"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def draft_picks_df(
    picks: list[dict], squad_df_normalized: pd.DataFrame, cutoff: float = 0.90
) -> pd.DataFrame:
    """
    Convert live Sleeper draft picks (see get_draft_picks) into a DataFrame -
    one row per pick, preserving `picked_by`/`roster_id`/`round`/`pick_no` -
    with a matched FPL `web_name` column (NaN if unmatched). Used by the
    Streamlit "live draft" view to show who each team has picked so far, and
    by `draft_picks_to_web_names` for the flat taken-player list.

    Matches the same way as get_sleeper_name_to_web_name (full name -> FPL's
    often-abbreviated web_name), with the same curated-override file
    (config/name_link_sleeper_curated.csv), but keeps every pick row instead
    of dropping unmatched ones - unmatched rows still carry real
    picked_by/round info the caller may want to show (e.g. "opponent picked
    someone we couldn't identify").
    """
    if not picks:
        return pd.DataFrame(
            columns=[
                "player_id",
                "picked_by",
                "roster_id",
                "round",
                "pick_no",
                "full_name",
                "team_abbr",
                "web_name",
                "match_method",
            ]
        )

    rows = []
    for p in picks:
        meta = p.get("metadata") or {}
        first = meta.get("first_name", "")
        last = meta.get("last_name", "")
        rows.append(
            {
                "player_id": p.get("player_id"),
                "picked_by": p.get("picked_by"),
                "roster_id": p.get("roster_id"),
                "round": p.get("round"),
                "pick_no": p.get("pick_no"),
                "full_name": f"{first} {last}".strip(),
                "team_abbr": meta.get("team", ""),
                "first_name": first,
                "last_name": last,
            }
        )
    picks_df = pd.DataFrame(rows)

    tgt = squad_df_normalized.copy()
    tgt["player_full"] = (
        tgt["first_name"].astype(str).str.strip()
        + " "
        + tgt["second_name"].astype(str).str.strip()
    ).str.replace(r"\s+", " ", regex=True)

    linked = link_names(
        source=picks_df,
        target=tgt,
        source_name_col="full_name",
        target_name_col="player_full",
        source_team_col="team_abbr" if "team_abbr" in picks_df.columns else None,
        target_team_col="team_code" if "team_code" in tgt.columns else None,
        cutoff=cutoff,
    )
    full_to_web = dict(zip(tgt["player_full"], tgt["web_name"]))
    linked["web_name"] = linked["matched_name"].map(full_to_web)

    linked, _n_overrides = apply_curated_overrides(
        linked, SLEEPER_NAME_CURATED_PATH, key_col="full_name", match_col="web_name"
    )
    return linked.drop(columns=["matched_name"], errors="ignore")


def draft_picks_to_web_names(
    picks: list[dict], squad_df_normalized: pd.DataFrame, cutoff: float = 0.90
) -> list[str]:
    """
    Flat list of FPL `web_name` values already drafted (unmatched picks
    dropped) - see draft_picks_df for the full per-pick DataFrame. Used to
    auto-populate taken.csv during a live draft.
    """
    df = draft_picks_df(picks, squad_df_normalized, cutoff=cutoff)
    return df.loc[df["web_name"].notna(), "web_name"].tolist()


def get_player_week_stats(season: int, week: int, sport: str = "clubsoccer:epl") -> pd.DataFrame:
    """
    Fetch Sleeper's own raw per-player stats and precomputed standard fantasy
    points (`pts_std`) for one completed gameweek - the same numbers the
    Sleeper app shows for "how each player scored". Undocumented endpoint
    (docs.sleeper.com is NFL-only) but confirmed working for club soccer;
    data is sourced from Opta per the `company` field on the season-level
    endpoint. Keyed by Sleeper `player_id` - no name-matching needed, unlike
    FBref/Understat.

    Ground truth for backtesting src/pdh/projections.py: compare `pts_std`
    here against `proj_points` for the same week to check model accuracy.
    """
    url = f"https://api.sleeper.app/v1/stats/{sport}/regular/{season}/{week}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return pd.DataFrame(data).T.reset_index(names="player_id")


def get_player_season_stats(season: int, sport: str = "clubsoccer:epl") -> pd.DataFrame:
    """
    Fetch Sleeper's own raw per-player season totals and precomputed standard
    fantasy points (`pts_std`)/rank (`rank_std`) - same source/caveats as
    get_player_week_stats. Returns one row per player with `stats` flattened
    into columns.
    """
    url = f"https://api.sleeper.app/stats/{sport}/{season}"
    response = requests.get(url, params={"season_type": "regular"})
    response.raise_for_status()
    rows = response.json()
    out = pd.json_normalize(rows, sep="_")
    return out


def refresh_league_config(
    league_id: str,
    my_team_id: str,
    setup_path: Path = SLEEPER_SETUP_PATH,
    draft_path: Path = SLEEPER_DRAFT_PATH,
) -> tuple[dict, dict]:
    """
    Refresh config/sleeper_setup.json and config/sleeper_draft.json from the live
    Sleeper API, so downstream code (VOR, scoring, snake-draft planner) never
    relies on stale/hand-edited league settings (team count, roster_positions,
    scoring, draft order).

    Safe to re-run. The league sits in "pre_draft" status until the draft
    happens, at which point roster_positions/draft_order/scoring can change -
    re-run this after the draft to pick that up.
    """
    league = get_league_info(league_id)

    users = get_league_users(league_id)
    user_ids = {u["user_id"] for u in users}
    if my_team_id not in user_ids:
        raise ValueError(
            f"my_team_id {my_team_id!r} is not a member of league {league_id!r}. "
            f"League members: {sorted(user_ids)}"
        )

    setup = dict(league)
    setup["my_team_id"] = my_team_id
    setup_path.parent.mkdir(parents=True, exist_ok=True)
    setup_path.write_text(json.dumps(setup, indent=4) + "\n")

    drafts = get_league_drafts(league_id)
    draft = next(
        (d for d in drafts if d.get("draft_id") == league.get("draft_id")),
        drafts[0] if drafts else None,
    )
    if draft is None:
        raise ValueError(f"No drafts found for league {league_id!r}")
    draft_path.parent.mkdir(parents=True, exist_ok=True)
    draft_path.write_text(json.dumps(draft, indent=4) + "\n")

    print(
        f"Refreshed {setup_path.name}/{draft_path.name} for '{league.get('name')}' "
        f"(status={league.get('status')}, total_rosters={league.get('total_rosters')})"
    )
    if draft.get("draft_order") is None:
        print(
            "Note: draft_order is not set yet (league is still pre-draft) - "
            "re-run this once the commissioner randomizes the draft order."
        )
    return setup, draft


def get_sleeper_rosters(league_id=LEAGUE_ID):
    """Fetch rosters from Sleeper API and return as DataFrame."""
    url = f"https://api.sleeper.app/v1/league/{league_id}/rosters"
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    rosters = response.json()
    return pd.DataFrame(rosters)


def get_sleeper_players(
    league: str = "clubsoccer:epl",
    max_age_days: int = 7,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch EPL players from Sleeper API and return as DataFrame.

    Cached on disk for up to `max_age_days` (default weekly) so we don't spam
    this ~5MB endpoint on every call.
    """
    cache_path = SLEEPER_CACHE_DIR / f"players_{league.replace(':', '_')}.json"

    if not force_refresh and cache_path.exists():
        age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if age < timedelta(days=max_age_days):
            players = json.loads(cache_path.read_text())
            return pd.DataFrame(players).T

    url = f"https://api.sleeper.app/v1/players/{league}"
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    players = response.json()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(players))

    return pd.DataFrame(players).T


def get_players_by_team(rosters_df, players_df, my_team_id=MY_TEAM_ID):
    """Map players to their respective teams based on rosters."""
    team_players = {}
    for _, row in rosters_df[rosters_df["owner_id"] != my_team_id].iterrows():
        team_id = row["owner_id"]
        player_ids = row["players"]
        team_players[team_id] = players_df[players_df["player_id"].isin(player_ids)]
    return team_players


def get_my_team_players(rosters_df, players_df, my_team_id=MY_TEAM_ID):
    """Get players for my own team based on rosters."""
    my_team_row = rosters_df[rosters_df["owner_id"] == my_team_id]
    if my_team_row.empty:
        raise ValueError(f"No roster found for team ID {my_team_id}")
    player_ids = my_team_row.iloc[0]["players"]
    my_players = players_df[players_df["player_id"].isin(player_ids)]
    if my_players.empty:
        player_ids = [int(pid) for pid in player_ids]
        my_players = players_df[players_df["player_id"].isin(player_ids)]
    return my_players


def get_positions(players_df):
    positions = players_df["fantasy_positions"].apply(
        lambda x: x[0] if isinstance(x, list) else []
    )
    return positions


def normalize_squad_df(squad_df):
    """Ensure normalized columns on squad_df."""
    squad_df = squad_df.copy()
    if "web_name_n" not in squad_df.columns:
        squad_df["web_name_n"] = squad_df["web_name"].map(canonical_name)
    if "first_name_n" not in squad_df.columns:
        squad_df["first_name_n"] = squad_df["first_name"].map(canonical_name)
    if "second_name" in squad_df.columns and "second_name_n" not in squad_df.columns:
        squad_df["second_name_n"] = squad_df["second_name"].map(canonical_name)
    elif "second_name_n" not in squad_df.columns:
        squad_df["second_name_n"] = pd.Series(
            [""] * len(squad_df), index=squad_df.index
        )
    # normalized team code for reliable team filtering
    if "team_code" in squad_df.columns and "team_code_n" not in squad_df.columns:
        squad_df["team_code_n"] = squad_df["team_code"].map(canonical_name)
    elif "team_code_n" not in squad_df.columns:
        squad_df["team_code_n"] = pd.Series([""] * len(squad_df), index=squad_df.index)
    return squad_df


def get_sleeper_name_to_web_name(df_team_players, squad_df_normalized, cutoff: float = 0.90):
    """
    Match each Sleeper roster row to its FPL `web_name`.

    FPL's `web_name` is often an abbreviated display name (e.g. "M.Salah", or
    just a surname), not a full name - matching against it directly fails even
    for exact matches. So we match on FPL's unabbreviated `first_name`+
    `second_name` (same approach as the proven FPL<->FBref linker in
    scripts/make_recommendations.py, generalized in src/pdh/namelink.py) and
    only resolve to `web_name` once a match is found. Curated overrides (if
    any) live in config/name_link_sleeper_curated.csv with columns
    `full_name`, `matched_name` (the desired web_name).
    """
    if df_team_players.empty:
        return []

    src = df_team_players.copy()
    if "full_name" not in src.columns or src["full_name"].isna().all():
        src["full_name"] = (
            src.get("first_name", pd.Series(dtype=str)).fillna("")
            + " "
            + src.get("last_name", pd.Series(dtype=str)).fillna("")
        ).str.strip()

    tgt = squad_df_normalized.copy()
    tgt["player_full"] = (
        tgt["first_name"].astype(str).str.strip()
        + " "
        + tgt["second_name"].astype(str).str.strip()
    ).str.replace(r"\s+", " ", regex=True)

    linked = link_names(
        source=src,
        target=tgt,
        source_name_col="full_name",
        target_name_col="player_full",
        source_team_col="team_abbr" if "team_abbr" in src.columns else None,
        target_team_col="team_code" if "team_code" in tgt.columns else None,
        cutoff=cutoff,
    )
    full_to_web = dict(zip(tgt["player_full"], tgt["web_name"]))
    linked["matched_name"] = linked["matched_name"].map(full_to_web)

    linked, _n_overrides = apply_curated_overrides(
        linked, SLEEPER_NAME_CURATED_PATH, key_col="full_name", match_col="matched_name"
    )

    for _, row in linked.iterrows():
        if pd.notna(row["matched_name"]):
            print(
                f"Found {row.get('last_name','')} -> {row['matched_name']} "
                f"({row['match_method']})"
            )
        else:
            print(
                f"Did NOT find {row.get('last_name','')} (first='{row.get('first_name','')}')"
            )

    return linked.loc[linked["matched_name"].notna(), "matched_name"].tolist()


def team_players_to_web_names(team_players, squad_df_normalized):
    """Convert a dict of team_id -> players_df to a list of web_names."""
    lst_team_web_names = []
    for team_id, df_players in team_players.items():
        print(f"Processing team {team_id} with {len(df_players)} players")
        lst_web_names = get_sleeper_name_to_web_name(df_players, squad_df_normalized)
        lst_team_web_names += lst_web_names
    return lst_team_web_names


def write_taken_to_csv(taken_list, filepath):
    """Write the list of taken players to a CSV file."""
    df_taken = pd.DataFrame(taken_list, columns=["web_name"])
    df_taken.to_csv(filepath, index=False)
    print(f"Wrote {len(taken_list)} taken players to {filepath}")
