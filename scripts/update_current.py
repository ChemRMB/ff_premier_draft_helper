"""
Fetch current squads and fixtures from FPL
"""

import pandas as pd, yaml, json
from pathlib import Path
from pdh import fpl
from pdh.fbref import schedule, flatten_cols
from pdh.seasons import current_season_dir, load_seasons_config
from pdh.sleeper import (
    get_sleeper_rosters,
    get_sleeper_players,
    get_players_by_team,
    normalize_squad_df,
    team_players_to_web_names,
    write_taken_to_csv,
)


ROOT = Path(__file__).resolve().parents[1]
outdir = current_season_dir()
outdir.mkdir(parents=True, exist_ok=True)

cfg_seasons = load_seasons_config()
current_season_end_year = cfg_seasons["current_season_end_year"]

print("Fetching current squads (FPL bootstrap)...")
df_squads = fpl.current_squads_df()
df_squads_normalized = normalize_squad_df(df_squads)
df_squads_normalized.to_csv(outdir / "current_squads.csv", index=False)
print("Wrote:", outdir / "current_squads.csv")
print(df_squads_normalized.team_name.value_counts().sort_index())
print("-" * 20)
print(df_squads_normalized.team_name.nunique(), "teams in current squads")

print("Fetching full fixtures (FPL official API)...")
# FPL's official fixtures endpoint is the primary source: reliable (official
# API), already carries every column the projection pipeline needs (event =
# gameweek, team_h/team_a, *_difficulty, finished, started, kickoff_time -
# see fpl.fixtures_df / project_players), and is populated before the season
# starts, so this works during pre-season / pre-draft. FBref's schedule
# (slower, browser-scraped, and absent for a season that hasn't kicked off)
# is only a fallback for the rare case FPL returns nothing.
fixtures = fpl.fixtures_df()
if fixtures.empty:
    print("  [info] FPL fixtures empty; falling back to FBref schedule...")
    fixtures_fbref = schedule(
        seasons=[current_season_end_year],
        leagues="ENG-Premier League",
        use_safe=True,
        cache_dir=ROOT / "data" / "cache" / "fbref",
    )
    if fixtures_fbref.index.names is not None:
        fixtures_fbref = fixtures_fbref.reset_index()
    fixtures = flatten_cols(fixtures_fbref)

fixtures.to_csv(outdir / "fixtures.csv", index=False)
print("Wrote:", outdir / "fixtures.csv", f"({len(fixtures)} fixtures)")

print("Fetching Sleeper stuff...")
sleeper_roster_df = get_sleeper_rosters()
players_df = get_sleeper_players()
players_df.to_csv(outdir / "sleeper_players.csv", index=False)
players_by_team = get_players_by_team(sleeper_roster_df, players_df)
lst_taken = team_players_to_web_names(players_by_team, df_squads_normalized)
write_taken_to_csv(lst_taken, outdir / "players_taken.csv")
print("Wrote:", outdir / "players_taken.csv")
