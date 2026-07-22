# Premier Draft Helper (PL + Sleeper)

A Python scaffold to build a draft and weekly recommender for your Sleeper Premier League league.
It combines **FPL public API** (current squads, fixtures) with **soccerdata** (FBref for broad per-90 stats, Understat for player-level xG/xA) for historical player & team performance, and projects points using your **Sleeper** scoring and roster configuration.
FotMob was previously listed as a source here, but soccerdata removed its FotMob reader entirely as of 1.9.0 - it's not usable via this library. See `config/scraper_comparison.md` for the full source comparison and decision.

## Quick start

```bash
uv sync  # creates .venv from pyproject.toml/uv.lock

# One-time: refresh config/sleeper_setup.json + sleeper_draft.json from the live Sleeper API
uv run python scripts/refresh_league_config.py

# Fetch & cache data + build draft board for GW1 (edit arguments inside the script as needed):
uv run python scripts/build_historical.py
uv run python scripts/update_current.py
uv run python scripts/make_recommendations.py
```

## What you get

- `src/pdh/fpl.py` – FPL endpoints: bootstrap (players/teams/events), fixtures, per-player history.
- `src/pdh/fbref.py` – soccerdata FBref access: team & player match logs per season (broad per-90 stats, no xG/xA).
- `src/pdh/understat.py` – soccerdata Understat access: player-level xG/xA/np_xG/xG-chain/xG-buildup.
- `src/pdh/sleeper.py` – Sleeper API access: rosters, players (weekly-cached), league/draft config refresh.
- `src/pdh/namelink.py` – shared canonical-name + fuzzy + curated-CSV player-name matching, used by both Sleeper<->FPL and FPL<->FBref linking.
- `src/pdh/normalize.py` – standardize player stats (per-90, tidy columns, join across tables).
- `src/pdh/scoring.py` – loads Sleeper scoring, maps stat names (editable in `config/stat_map.yaml`), computes fantasy points.
- `src/pdh/recommend.py` – season-weighted player rates, fixture difficulty adjustment, GW projections.
- `src/pdh/draft_board.py` – VOR (value over replacement) ranks by position given your snake-draft roster settings.
- `scripts/*.py` – runnable pipeline: `refresh_league_config.py`, `compare_scrapers.py`, `build_historical.py`, `update_current.py`, `make_recommendations.py`, `recommend_formation.py`.

## Configure

- `config/sleeper_setup.json` and `config/sleeper_draft.json` – your league settings, regenerated from the live Sleeper API via `scripts/refresh_league_config.py` (don't hand-edit; re-run the script instead, e.g. again after the draft happens).
- `config/stat_map.yaml` – edit this to align Sleeper’s stat keys with available FBref/Understat fields.
- `config/seasons.yaml` – choose historical seasons and weighting (e.g., 2025: 1.5, 2024: 1.25, 2023: 1.0).

## Data outputs

- `data/outputs/players_historical.csv` – player-match rows (last 3 seasons) with normalized stats.
- `data/outputs/current_squads.csv` – current season players (FPL) with status/positions.
- `data/outputs/fixtures.csv` – all fixtures this season with GW (event) tags.
- `data/outputs/draft_board.csv` – ranked list for draft (with VOR).
- `data/outputs/gw_top10.csv` – weekly top 10 suggestions overall and by position.

## Licenses & ToS

Respect the terms of use for FPL, FBref, Understat, and other sites. Unofficial endpoints may change.
