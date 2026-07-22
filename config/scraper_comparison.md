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
