"""
Refresh config/sleeper_setup.json and config/sleeper_draft.json from the live
Sleeper API. Run this now, and again after the draft happens (roster_positions,
draft_order and scoring can change once the league leaves "pre_draft" status).

Usage:
    python scripts/refresh_league_config.py [league_id] [my_team_id]
"""

import sys

from pdh.sleeper import _SEED_LEAGUE_ID, _SEED_MY_TEAM_ID, refresh_league_config

if __name__ == "__main__":
    league_id = sys.argv[1] if len(sys.argv) > 1 else _SEED_LEAGUE_ID
    my_team_id = sys.argv[2] if len(sys.argv) > 2 else _SEED_MY_TEAM_ID
    refresh_league_config(league_id, my_team_id)
