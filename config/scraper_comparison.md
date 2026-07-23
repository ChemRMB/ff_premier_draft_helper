# Scraper comparison (2026-07-23, season=2025)

Run via `python scripts/compare_scrapers.py`. See config/to_do.md
"Assess scrapers" / "Assess naming of players for different scrapers".

| Source | OK | Seconds | Player-level xG/xA | Notes |
|---|---|---|---|---|
| FBref | True | 3.9 | False | No xG/xA in the stat tables this project currently pulls from FBref. |
| Understat | False | 0.2 | None | 'stat' |
| FotMob | True | 0.0 | None | soccerdata 1.9.0 no longer exposes a FotMob reader at all (present in 1.8.7, removed by 1.9.0 - likely a takedown/ToS issue). Not usable via soccerdata regardless of xG coverage. |

## Decision

- **FBref** stays the primary source for the broad per-90 stat set (passing, defense, possession, misc) via src/pdh/fbref.py. Reliable once warm (SeleniumBase undetected-Chromium bypasses Cloudflare), but requires a real browser and is comparatively slow/heavy.
- **Understat** (src/pdh/understat.py, new) is the source for player-level xG/xA/np_xG/xG-chain/xG-buildup - FBref has none of this. It's a plain `requests`-based reader (no browser needed) and returns rich data when reachable, but has been observed to soft-block (HTTP 200, stub page, no embedded data) under repeated automated requests in a short window - treat empty results as 'retry later', not 'no data', and rely on soccerdata's per-page on-disk cache to avoid re-hitting it unnecessarily.
- **FotMob** is not usable at all: soccerdata removed the FotMob reader entirely as of 1.9.0 (present in 1.8.7, gone by 1.9.0 - likely a takedown/ToS issue). Not a source, full stop; drop it from consideration rather than treating it as a secondary team-power input.

## Appendix: Understat evidence from a successful fetch

The run above happened to hit Understat's soft-block, so the table shows a
failure. Earlier in the same investigation, before the block kicked in,
`sd.Understat(leagues="ENG-Premier League", seasons="2024/2025").read_player_season_stats()`
succeeded and returned 562 rows with these columns:

```
league_id, season_id, team_id, player_id, position, matches, minutes, goals,
xg, np_goals, np_xg, assists, xa, shots, key_passes, yellow_cards, red_cards,
xg_chain, xg_buildup
```

Sample (Arsenal, 2024/25):

| player | matches | minutes | goals | xg | assists | xa | xg_chain | xg_buildup |
|---|---|---|---|---|---|---|---|---|
| Bukayo Saka | 25 | 1763 | 6 | 8.94 | 10 | 11.58 | 16.46 | 6.02 |
| Declan Rice | 35 | 2848 | 4 | 3.63 | 7 | 9.06 | 17.24 | 10.36 |
| David Raya | 38 | 3420 | 0 | 0.0 | 0 | 0.09 | 6.11 | 6.02 |

This is real, usable player-level xG/xA data - the block is a request-pacing
problem, not a data-availability problem. **Decision stands: use Understat for
xG/xA, with caching and tolerance for occasional retries.**

## Addendum: Sleeper's own stats feed (found while working Plan B)

Sleeper exposes an undocumented (docs.sleeper.com is NFL-only) but working
stats feed, confirmed live against the completed 2024/25 season
(league `1259142774104526848`, status `complete`):

- `GET /v1/stats/clubsoccer:epl/regular/<season>/<week>` - per-player raw
  stats + precomputed standard fantasy points (`pts_std`) for one gameweek.
- `GET /stats/clubsoccer:epl/<season>?season_type=regular` (no `/v1/` prefix)
  - same, at season-cumulative granularity, plus `rank_std` and embedded
  player metadata. Data is sourced from **Opta** (`company: "opta"` field).

Both are keyed by Sleeper's own `player_id` - **no name-matching needed at
all**, unlike FBref/Understat. Verified real output for season 2025 (2024/25):
top scorers were Bruno Fernandes (615.5 pts), Erling Haaland (509.0), Antoine
Semenyo (432.0) - sane, plausible numbers. Implemented as
`src/pdh/sleeper.py::get_player_week_stats()` / `get_player_season_stats()`.

**Why this matters beyond "another source":** because it's pre-scored and
keyed by Sleeper `player_id`, it's the only source that can directly validate
the projection engine - compare `proj_points` (src/pdh/projections.py)
against real `pts_std` for a past gameweek, with zero name-linking risk in
the comparison itself. That backtest harness isn't built yet (would need the
pipeline to reconstruct "what we'd have projected using only data available
before week N," which the current single-snapshot `hist` load doesn't
support) - noted as a natural follow-up, not done in this pass. The
`/v1/league/<id>/matchups/<week>` endpoint, by contrast, returned 404 for
every one of the 38 weeks of that same completed season - it does not appear
to work for this sport/league at all, despite the league clearly having
played a full head-to-head season (win/loss record on the roster object).
Plan D's matchup-fetching should treat that endpoint as unreliable and lean on
this stats feed instead where possible.
