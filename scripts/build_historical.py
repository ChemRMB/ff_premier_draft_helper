"""
Pull last 3 seasons of team & player data via soccerdata (FBref) and write CSVs.
"""

import pandas as pd, yaml
from pathlib import Path
from src.pdh.fbref import player_match_stats, team_season_stats
from src.pdh.normalize import normalize_player_matches

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config/seasons.yaml", "r"))
seasons = cfg["historical_seasons"]

print(f"Fetching FBref player match stats for seasons: {seasons} ...")
# pm = player_match_stats(seasons)
# Normalize player match stats, filling in missing columns
pm = pd.read_csv(ROOT / "data/outputs/players_historical.csv")
pm = normalize_player_matches(pm)
outdir = ROOT / "data/outputs"
outdir.mkdir(parents=True, exist_ok=True)
pm.to_csv(outdir / "players_historical.csv", index=False)
print("Wrote:", outdir / "players_historical.csv")

# print("Fetching team season stats ...")
# ts = team_season_stats(seasons)
# ts.to_csv(outdir / "teams_season_stats.csv", index=False)
# print("Wrote:", outdir / "teams_season_stats.csv")
