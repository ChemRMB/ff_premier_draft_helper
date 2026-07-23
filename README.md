# Premier Draft Helper (PL + Sleeper)

A Python scaffold to build a draft and weekly recommender for your Sleeper Premier League league.
It combines **FPL public API** (current squads, fixtures) with **soccerdata** (FBref for broad per-90 stats, Understat for player-level xG/xA) for historical player & team performance, and projects points using your **Sleeper** scoring and roster configuration. See `config/scraper_comparison.md` for the full source comparison and decision.

## Quick start

```bash
uv sync  # creates .venv from pyproject.toml/uv.lock

# One-time: refresh config/sleeper_setup.json + sleeper_draft.json from the live Sleeper API
uv run python scripts/refresh_league_config.py

# Fetch & cache data + build draft board for GW1 (edit arguments inside the script as needed):
uv run python scripts/build_historical.py
uv run python scripts/update_current.py
uv run python scripts/make_recommendations.py --refresh

# Live snake-draft assistant (best-available/VOR, taken players excluded in
# real time against the actual Sleeper draft, every team's picks so far):
uv run streamlit run scripts/streamlit_app.py
```

## What you get

- `src/pdh/fpl.py` – FPL endpoints: bootstrap (players/teams/events), fixtures, per-player history.
- `src/pdh/fbref.py` – soccerdata FBref access: team & player match logs per season (broad per-90 stats, no xG/xA).
- `src/pdh/understat.py` – soccerdata Understat access: player-level xG/xA/np_xG/xG-chain/xG-buildup.
- `src/pdh/sleeper.py` – Sleeper API access: rosters, players (weekly-cached), league/draft config refresh.
- `src/pdh/namelink.py` – shared canonical-name + fuzzy + curated-CSV player-name matching, used by both Sleeper<->FPL and FPL<->FBref linking.
- `src/pdh/normalize.py` – standardize player stats (per-90, tidy columns, join across tables).
- `src/pdh/scoring.py` – loads Sleeper scoring, maps stat names (editable in `config/stat_map.yaml`), computes fantasy points.
- `src/pdh/projections.py` – season-weighted per-90 rates (with Understat/FPL xG-xA blending), team attack/defense power, GW point projections.
- `src/pdh/draft_board.py` – VOR (value over replacement) ranks by position given your snake-draft roster settings.
- `src/pdh/stickiness.py` – which players are reliable/outlier enough that trade suggestions should never offer them away (Plan D).
- `src/pdh/webapp/data.py` – data-loading layer for the Streamlit app; reads existing pipeline outputs, doesn't recompute anything.
- `scripts/*.py` – runnable pipeline: `refresh_league_config.py`, `compare_scrapers.py`, `build_historical.py`, `update_current.py`, `make_recommendations.py`, `recommend_formation.py`, `weekly_lineup.py`, `suggest_name_links.py` (suggests FBref matches for unmatched FPL players to speed up curating `config/name_link_curated.csv`), `streamlit_app.py`.

## Configure

- `config/sleeper_setup.json` and `config/sleeper_draft.json` – your league settings, regenerated from the live Sleeper API via `scripts/refresh_league_config.py` (don't hand-edit; re-run the script instead, e.g. again after the draft happens).
- `config/stat_map.yaml` – edit this to align Sleeper’s stat keys with available FBref/Understat fields.
- `config/seasons.yaml` – choose historical seasons and weighting (e.g., 2025: 1.5, 2024: 1.25, 2023: 1.0).

## Data outputs

- `data/historical/players_historical_*.csv` / `teams_season_stats.csv` – player-match rows and team season stats (from `build_historical.py`).
- `data/season_2526/current_squads.csv`, `fixtures.csv`, `players_taken.csv` – current-season snapshot (from `update_current.py`).
- `data/season_2526/game_weeks/gwN/` – per-gameweek outputs from `make_recommendations.py`: `draft_board_flexaware.csv`/`draft_board_nonflex.csv` (ranked, with VOR), `top10.csv` (overall + per position), `players_projected_cache.csv` (cached projections - see `--refresh`), `taken.csv`/`snake_state.csv` (live draft state, auto-populated from the real Sleeper draft), `top_roster.csv`, `snake_plan.csv`.

## Licenses & ToS

Respect the terms of use for FPL, FBref, Understat, and other sites. Unofficial endpoints may change.
