# Premier Draft Helper (PL + Sleeper)

A Python scaffold to build a draft and weekly recommender for your Sleeper Premier League league.
It combines **FPL public API** (current squads, fixtures) with **soccerdata** (FBref/FotMob) for historical player & team performance, and projects points using your **Sleeper** scoring and roster configuration.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# One-time: place your Sleeper files in ./config/ (or keep the samples)
# Then fetch & cache data + build draft board for GW1 (edit arguments inside the script as needed):
python scripts/build_historical.py
python scripts/update_current.py
python scripts/make_recommendations.py
```

## What you get

- `src/pdh/fpl.py` – FPL endpoints: bootstrap (players/teams/events), fixtures, per-player history.
- `src/pdh/fbref.py` – soccerdata FBref access: team & player match logs per season.
- `src/pdh/normalize.py` – standardize player stats (per-90, tidy columns, join across tables).
- `src/pdh/scoring.py` – loads Sleeper scoring, maps stat names (editable in `config/stat_map.yaml`), computes fantasy points.
- `src/pdh/recommend.py` – season-weighted player rates, fixture difficulty adjustment, GW projections.
- `src/pdh/draft_board.py` – VOR (value over replacement) ranks by position given your snake-draft roster settings.
- `scripts/*.py` – runnable examples that write CSVs to `./data/outputs/`.

## Configure

- `config/sleeper_setup.json` and `config/sleeper_draft.json` – your league settings (samples included from your uploads).
- `config/stat_map.yaml` – edit this to align Sleeper’s stat keys with available FBref/FotMob fields.
- `config/seasons.yaml` – choose historical seasons and weighting (e.g., 2025: 1.5, 2024: 1.25, 2023: 1.0).

## Data outputs

- `data/outputs/players_historical.csv` – player-match rows (last 3 seasons) with normalized stats.
- `data/outputs/current_squads.csv` – current season players (FPL) with status/positions.
- `data/outputs/fixtures.csv` – all fixtures this season with GW (event) tags.
- `data/outputs/draft_board.csv` – ranked list for draft (with VOR).
- `data/outputs/gw_top10.csv` – weekly top 10 suggestions overall and by position.

## Licenses & ToS

Respect the terms of use for FPL, FBref, FotMob, and other sites. Unofficial endpoints may change.
