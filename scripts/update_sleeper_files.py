import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

# https://docs.sleeper.com/
current_week = datetime.now().isocalendar()[1]
league_id = "1259142774104526848"


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
league_draft_picks = requests.get(
    f"https://api.sleeper.app/v1/draft/1259142777103470592/picks"
).json()
print(league_draft_picks)

# # get rosters from sleeper
# print("Fetching Sleeper rosters...")
# sleeper_rosters = requests.get(
#     f"https://api.sleeper.app/v1/league/{league_id}/rosters"
# ).json()
# sleeper_rosters_df = pd.DataFrame(sleeper_rosters)  # .T.reset_index()
# print(sleeper_rosters_df.head())
