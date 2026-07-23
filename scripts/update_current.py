"""
Fetch current squads and fixtures from FPL
"""

import pandas as pd, yaml, json
from pathlib import Path
from pdh import fpl
from pdh.fbref import schedule, flatten_cols
from pdh.sleeper import (
    get_sleeper_rosters,
    get_sleeper_players,
    get_players_by_team,
    normalize_squad_df,
    team_players_to_web_names,
    write_taken_to_csv,
)


ROOT = Path(__file__).resolve().parents[1]
outdir = ROOT / "data/season_2526"
outdir.mkdir(parents=True, exist_ok=True)

cfg_seasons = yaml.safe_load(open(ROOT / "config/seasons.yaml", "r"))
current_season_end_year = cfg_seasons["current_season_end_year"]

print("Fetching current squads (FPL bootstrap)...")
df_squads = fpl.current_squads_df()
df_squads_normalized = normalize_squad_df(df_squads)
df_squads_normalized.to_csv(outdir / "current_squads.csv", index=False)
print("Wrote:", outdir / "current_squads.csv")
print(df_squads_normalized.team_name.value_counts().sort_index())
print("-" * 20)
print(df_squads_normalized.team_name.nunique(), "teams in current squads")

print("Fetching full fixtures...")
fixtures_fbref = schedule(
    seasons=[current_season_end_year],
    leagues="ENG-Premier League",
    use_safe=True,
    cache_dir=ROOT / "data" / "cache" / "fbref",
)
if fixtures_fbref.index.names is not None:
    fixtures_fbref = fixtures_fbref.reset_index()
fixtures_fbref = flatten_cols(fixtures_fbref)
fixtures_fpl = fpl.fixtures_df()
fixtures = fixtures_fbref.join(fixtures_fpl)  # index based join


fixtures.to_csv(outdir / "fixtures.csv", index=False)
print("Wrote:", outdir / "fixtures.csv")

print("Fetching Sleeper stuff...")
sleeper_roster_df = get_sleeper_rosters()
players_df = get_sleeper_players()
players_df.to_csv(outdir / "sleeper_players.csv", index=False)
players_by_team = get_players_by_team(sleeper_roster_df, players_df)
lst_taken = team_players_to_web_names(players_by_team, df_squads_normalized)
write_taken_to_csv(lst_taken, outdir / "players_taken.csv")
print("Wrote:", outdir / "players_taken.csv")
