import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import re, unicodedata


# https://docs.sleeper.com/
CURRENT_WEEK = datetime.now().isocalendar()[1]
LEAGUE_ID = "1259142774104526848"
MY_TEAM_ID = "1259256320754724864"


def get_sleeper_rosters(league_id=LEAGUE_ID):
    """Fetch rosters from Sleeper API and return as DataFrame."""
    url = f"https://api.sleeper.app/v1/league/{league_id}/rosters"
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    rosters = response.json()
    return pd.DataFrame(rosters)


def get_sleeper_players(leage: str = "clubsoccer:epl"):
    """Fetch EPL players from Sleeper API and return as DataFrame."""
    url = f"https://api.sleeper.app/v1/players/{leage}"
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    players = response.json()
    return pd.DataFrame(players).T


def get_players_by_team(rosters_df, players_df):
    """Map players to their respective teams based on rosters."""
    team_players = {}
    for _, row in rosters_df[rosters_df["owner_id"] != MY_TEAM_ID].iterrows():
        team_id = row["owner_id"]
        player_ids = row["players"]
        team_players[team_id] = players_df[players_df["player_id"].isin(player_ids)]
    return team_players


def get_my_team_players(rosters_df, players_df):
    """Get players for my own team based on rosters."""
    my_team_row = rosters_df[rosters_df["owner_id"] == MY_TEAM_ID]
    print("my_team_row:")
    print(my_team_row)
    if my_team_row.empty:
        raise ValueError(f"No roster found for team ID {MY_TEAM_ID}")
    player_ids = my_team_row.iloc[0]["players"]
    print("player_ids:", player_ids)
    my_players = players_df[players_df["player_id"].isin(player_ids)]
    if my_players.empty:
        player_ids = [int(pid) for pid in player_ids]
        print("Converted player_ids to int:", player_ids)
        my_players = players_df[players_df["player_id"].isin(player_ids)]
    return my_players


def get_positions(players_df):
    positions = players_df["fantasy_positions"].apply(
        lambda x: x[0] if isinstance(x, list) else []
    )
    return positions


def norm(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[\"'`]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_squad_df(squad_df):
    """Ensure normalized columns on squad_df."""
    squad_df = squad_df.copy()
    if "web_name_n" not in squad_df.columns:
        squad_df["web_name_n"] = squad_df["web_name"].map(norm)
    if "first_name_n" not in squad_df.columns:
        squad_df["first_name_n"] = squad_df["first_name"].map(norm)
    if "second_name" in squad_df.columns and "second_name_n" not in squad_df.columns:
        squad_df["second_name_n"] = squad_df["second_name"].map(norm)
    elif "second_name_n" not in squad_df.columns:
        squad_df["second_name_n"] = pd.Series(
            [""] * len(squad_df), index=squad_df.index
        )
    # normalized team code for reliable team filtering
    if "team_code" in squad_df.columns and "team_code_n" not in squad_df.columns:
        squad_df["team_code_n"] = squad_df["team_code"].map(norm)
    elif "team_code_n" not in squad_df.columns:
        squad_df["team_code_n"] = pd.Series([""] * len(squad_df), index=squad_df.index)
    return squad_df


def get_sleeper_name_to_web_name(df_team_players, squad_df_normalized):

    lst_taken = []
    for _, row in df_team_players.iterrows():
        last_n = norm(row.get("last_name", ""))
        first_n = norm(row.get("first_name", ""))
        full_n = norm(row.get("full_name", ""))
        team_n = norm(row.get("team_abbr", ""))
        # restrict to same team if possible
        if team_n:
            df_team = squad_df_normalized[squad_df_normalized["team_code_n"] == team_n]
            if df_team.empty:
                # fallback to whole dataset if team not found
                df_team = squad_df_normalized
                team_note = "(team not found, searching all teams)"
            else:
                team_note = f"(team match {team_n})"
        else:
            df_team = squad_df_normalized
            team_note = "(no team_abbr, searching all teams)"

        found = None
        reason = None

        # 1) exact web_name match (preferred)
        m = df_team[df_team["web_name_n"] == last_n]
        if not m.empty:
            found = m["web_name"].iloc[0]
            reason = "web_name exact match"
        else:
            # 2) web_name contains last name
            if last_n:
                m = df_team[
                    df_team["web_name_n"].str.contains(re.escape(last_n), na=False)
                ]
                if not m.empty:
                    found = m["web_name"].iloc[0]
                    reason = "web_name contains last name"
            # 3) first_name contains player's first name / nickname
            if not found and first_n:
                m = df_team[
                    df_team["first_name_n"].str.contains(re.escape(first_n), na=False)
                ]
                if not m.empty:
                    found = m["web_name"].iloc[0]
                    reason = "first_name contains first/nickname"
            # 4) try tokens from full_name against web_name
            if not found and full_n:
                for token in full_n.split():
                    if not token:
                        continue
                    m = df_team[
                        df_team["web_name_n"].str.contains(re.escape(token), na=False)
                    ]
                    if not m.empty:
                        found = m["web_name"].iloc[0]
                        reason = f"token '{token}' in web_name"
                        break

        if found:
            print(f"Found {row.get('last_name','')} -> {found} {team_note} ({reason})")
            lst_taken.append(found)

        else:
            print(
                f"Did NOT find {row.get('last_name','')} (first='{row.get('first_name','')}')"
            )
    return lst_taken


def team_players_to_web_names(team_players, squad_df_normalized):
    """Convert a dict of team_id -> players_df to a list of web_names."""
    lst_team_web_names = []
    for team_id, df_players in team_players.items():
        print(f"Processing team {team_id} with {len(df_players)} players")
        lst_web_names = get_sleeper_name_to_web_name(df_players, squad_df_normalized)
        lst_team_web_names += lst_web_names
    return lst_team_web_names


# def sleeper_name_to_web_name(team_players, squad_df):
#     """Create a mapping from Sleeper player IDs to web_name.
#     ['last_name', 'first_name', 'search_last_name', 'full_name', 'search_full_name', 'team_abbr']
#     """

#     # ensure normalized columns on squad_df
#     squad_df = normalize_squad_df(squad_df)

#     lst_taken = []
#     for owner_id, df_team_players in team_players.items():
#         print("--" * 20)
#         print(f"Owner {owner_id} has {len(df_team_players)} players")
#         for _, row in df_team_players.iterrows():
#             last_n = norm(row.get("last_name", ""))
#             first_n = norm(row.get("first_name", ""))
#             full_n = norm(row.get("full_name", ""))
#             team_n = norm(row.get("team_abbr", ""))
#             # restrict to same team if possible
#             if team_n:
#                 df_team = squad_df[squad_df["team_code_n"] == team_n]
#                 if df_team.empty:
#                     # fallback to whole dataset if team not found
#                     df_team = squad_df
#                     team_note = "(team not found, searching all teams)"
#                 else:
#                     team_note = f"(team match {team_n})"
#             else:
#                 df_team = squad_df
#                 team_note = "(no team_abbr, searching all teams)"

#             found = None
#             reason = None

#             # 1) exact web_name match (preferred)
#             m = df_team[df_team["web_name_n"] == last_n]
#             if not m.empty:
#                 found = m["web_name"].iloc[0]
#                 reason = "web_name exact match"
#             else:
#                 # 2) web_name contains last name
#                 if last_n:
#                     m = df_team[
#                         df_team["web_name_n"].str.contains(re.escape(last_n), na=False)
#                     ]
#                     if not m.empty:
#                         found = m["web_name"].iloc[0]
#                         reason = "web_name contains last name"
#                 # 3) first_name contains player's first name / nickname
#                 if not found and first_n:
#                     m = df_team[
#                         df_team["first_name_n"].str.contains(
#                             re.escape(first_n), na=False
#                         )
#                     ]
#                     if not m.empty:
#                         found = m["web_name"].iloc[0]
#                         reason = "first_name contains first/nickname"
#                 # 4) try tokens from full_name against web_name
#                 if not found and full_n:
#                     for token in full_n.split():
#                         if not token:
#                             continue
#                         m = df_team[
#                             df_team["web_name_n"].str.contains(
#                                 re.escape(token), na=False
#                             )
#                         ]
#                         if not m.empty:
#                             found = m["web_name"].iloc[0]
#                             reason = f"token '{token}' in web_name"
#                             break

#             if found:
#                 print(
#                     f"Found {row.get('last_name','')} -> {found} {team_note} ({reason})"
#                 )
#                 lst_taken.append(found)

#             else:
#                 print(
#                     f"Did NOT find {row.get('last_name','')} (first='{row.get('first_name','')}')"
#                 )
#     return lst_taken


def write_taken_to_csv(taken_list, filepath):
    """Write the list of taken players to a CSV file."""
    df_taken = pd.DataFrame(taken_list, columns=["web_name"])
    df_taken.to_csv(filepath, index=False)
    print(f"Wrote {len(taken_list)} taken players to {filepath}")


# update sleeper_setup

# league_id_info = requests.get(f"https://api.sleeper.app/v1/league/{league_id}").json()
# print("League ID:", league_id)
# print(league_id_info)

# # get sleeper league users
# league_users = requests.get(
#     f"https://api.sleeper.app/v1/league/{league_id}/users"
# ).json()
# league_users_df = pd.DataFrame(league_users).T.reset_index()
# print(league_users_df.head())

# get league matchups
# print(current_week)
# league_matchups = requests.get(
#     f"https://api.sleeper.app/v1/league/{league_id}/matchups/{current_week}"
# ).json()
# print(league_matchups)

# get transactions
# league_transactions = requests.get(
#     f"https://api.sleeper.app/v1/league/{league_id}/transactions{current_week}"
# ).json()
# print("Transactions for week", current_week, ":", league_transactions)

# get drafts
# league_drafts = requests.get(
#     f"https://api.sleeper.app/v1/league/{league_id}/drafts"
# ).json()[0]
# print("Drafts:")
# print(league_drafts)
# print("draft order:", league_drafts.get("draft_order", {}))

# get draft picks
# league_draft_picks = requests.get(
#     f"https://api.sleeper.app/v1/draft/1259142777103470592/picks"
# ).json()
# print(league_draft_picks)

# # get rosters from sleeper
# print("Fetching Sleeper rosters...")
# sleeper_rosters = requests.get(
#     f"https://api.sleeper.app/v1/league/{league_id}/rosters"
# ).json()
# sleeper_rosters_df = pd.DataFrame(sleeper_rosters)  # .T.reset_index()
# print(sleeper_rosters_df.head())
