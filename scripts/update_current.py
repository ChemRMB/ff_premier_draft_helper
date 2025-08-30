"""
Fetch current squads and fixtures from FPL
"""

import pandas as pd, yaml, json
from pathlib import Path
from src.pdh import fpl
from src.pdh.fbref import schedule, flatten_cols

ROOT = Path(__file__).resolve().parents[1]
outdir = ROOT / "data/outputs"
outdir.mkdir(parents=True, exist_ok=True)

print("Fetching current squads (FPL bootstrap)...")
squads = fpl.current_squads_df()
# squads.to_csv(outdir / "current_squads.csv", index=False)
print("Wrote:", outdir / "current_squads.csv")
print(squads.team_name.value_counts().sort_index())
print("-" * 20)
print(squads.team_name.nunique(), "teams in current squads")

print("Fetching full fixtures...")
fixtures_fbref = schedule(seasons=[2025], leagues="ENG-Premier League")
if fixtures_fbref.index.names is not None:
    fixtures_fbref = fixtures_fbref.reset_index()
fixtures_fbref = flatten_cols(fixtures_fbref)
fixtures_fpl = fpl.fixtures_df()
fixtures = fixtures_fbref.join(fixtures_fpl)  # index based join

fixtures.to_csv(outdir / "fixtures.csv", index=False)
print("Wrote:", outdir / "fixtures.csv")
