"""
Pull last 3 seasons of team & player data via soccerdata (FBref) and write CSVs.
"""

import pandas as pd, yaml
from pathlib import Path
from src.pdh.fbref import player_match_stats, team_season_stats
from src.pdh.normalize import normalize_player_matches
from datetime import datetime
import yaml

date_now = datetime.now().isoformat()
# format to YYYYMMDD
date_now = date_now.split("T")[0].replace("-", "")

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config/seasons.yaml", "r"))
seasons = cfg["historical_seasons"]

# print("Seasons:", seasons)
print(f"Fetching FBref player match stats for seasons: {seasons} ...")
pm = player_match_stats(seasons)
# Normalize player match stats, filling in missing columns
# pm = pd.read_csv(ROOT / "data/outputs/players_historical_{}.csv")
pm = normalize_player_matches(pm)
outdir = ROOT / "data/historical"
outdir.mkdir(parents=True, exist_ok=True)
pm.to_csv(outdir / f"players_historical_{date_now}.csv", index=False)
print("Wrote:", outdir / f"players_historical_{date_now}.csv")

# print("Fetching team season stats ...")
# ts = team_season_stats(seasons)  # fails on season 2025 because it is not available yet
# ts.to_csv(outdir / f"teams_season_stats_{date_now}.csv", index=False)
# print("Wrote:", outdir / f"teams_season_stats_{date_now}.csv")
