# Plan C: snake-draft assistant hardening

## 1) VOR/replacement-rank dedup

`scripts/make_recommendations.py` used to reimplement `roster_requirements`,
`replacement_ranks`/`replacement_ranks_nonflex`, `effective_replacement_ranks_with_flex`,
and VOR computation (`build_vor`) locally, duplicating `src/pdh/draft_board.py`
almost line-for-line - `draft_board.py` was written but never actually
imported. The script now imports all of it from `pdh.draft_board`.
`draft_board.compute_vor` only handled the non-flex case (it always
recomputed replacement ranks internally), so the shared VOR math was pulled
into a new `vor_from_ranks(df, repl_ranks, ...)` that accepts pre-computed
ranks - `compute_vor` is now a thin wrapper around it for the non-flex case,
and the script calls `vor_from_ranks` directly for both the non-flex and
flex-aware boards. Verified behavior-preserving with a synthetic-data test
comparing the old and new code paths (byte-identical DataFrames).

Team count/`roster_positions` already flow from `config/sleeper_setup.json`/
`sleeper_draft.json` (refreshed live from the Sleeper API in Plan A) - no
hardcoded team count anywhere in this pipeline.

## 2) `scripts/live_draft_helper.py` - deleted

It read from `data/outputs/draft_board_flex_aware.csv`, which doesn't exist
(real outputs live under `data/season_2526/game_weeks/gwN/`) - it's been
broken since that output path changed. Its entire purpose (filter the board
by already-taken players, show top N overall + per position) is already
fully covered by `make_recommendations.py`'s `taken.csv`-filtered `top10.csv`
output (section 7), and better: that reads persistent `taken.csv`/
`taken_next.csv` state files updated by the normal workflow, rather than
requiring every taken player to be retyped as a `--taken` CLI arg on each
run. Nothing was folded in because there was nothing left to fold in.

## 3) Opponent-pick simulation (`deprecated/make_recommendations_old.py`) - not reinstated

The old script simulated a full mock draft: each round, in snake order, every
*other* team's slot just took the single best-remaining player by
VOR/proj_points from the shared pool (`pop_best_available`), with no model of
what that opponent's roster already needs (position balance, starters they've
already filled). That's a weak assumption - real drafters draft for need, not
pure best-available-VOR - and a wrong prediction here ("your target will
still be there in round 5") is worse than no prediction, since it could lead
to a bad real-time decision.

The current script deliberately dropped this (see the comment above section 9:
*"We do NOT simulate other teams' picks anymore"*) in favor of a reactive
model: given the real `taken.csv` state at this exact moment, what's the best
available pick right now, plus a ranked top-10 (overall and per position) as
backups if your intended pick gets taken before your turn. That already
solves the actual problem from `config/to_do.md` ("I need backup options...
so if an opponent chooses the player I intended, I have backup players to
choose from") without needing to model opponent behavior at all.

**Decision: leave it discarded.** If a real predictive need shows up later
(e.g. "what's the probability my round-4 target survives to my pick"), it'd
be worth building on top of real signals (opponent roster needs from
Sleeper rosters, position scarcity) rather than reviving the old
best-VOR-only simulation.

## 4) Precompute for the live draft loop

`config/to_do.md`'s complaint - "the script will sometimes take time to run"
- matters most during the live draft, where you rerun the script after every
opponent pick just to refresh "what should I take now." Profiling a real run
(`--event 11`) showed ~7.5s end-to-end, almost all of it in three
taken.csv-independent steps: FPL<->FBref name-linking (fuzzy matching ~800+
players), the Understat fetch (network, and can be slow/soft-blocked), and
team-power computation. None of that changes between one opponent's pick and
the next - only `taken.csv` does.

`make_recommendations.py` now caches the fully-projected player pool (before
the taken-filter) to `players_projected_cache.csv` in the gameweek's output
folder. A run reuses that cache unless `--refresh` is passed, skipping
straight to the taken-filter/ranking/planner steps. Measured: ~7.7s cold ->
~1.3s warm (~6x), with byte-identical `draft_board_flexaware.csv`/`top10.csv`
output verified between cold and warm runs.

**Recommended live-draft workflow:** run once with `--refresh` before the
draft (or whenever squads/hist/fixtures data changes, e.g. after
`update_current.py`), then plain reruns after each opponent pick reuse the
cache. The cache file lives under `data/` (gitignored) so it's never
committed.

## 5) taken.csv auto-populated from the live draft

Previously `taken.csv` had to be hand-edited after every opponent pick.
`src/pdh/sleeper.py::get_draft_picks`/`draft_picks_to_web_names` fetch the
real picks from `GET /v1/draft/<draft_id>/picks` and map them to FPL
`web_name`s using the same full-name matching as roster linking. Wired into
`make_recommendations.py` right before the taken-filter step: every run
merges any new live picks into `taken.csv` automatically. Falls back
silently (no-op) when there's no live draft data yet (pre-draft, or the
fetch fails) - so plain manual edits still work too, e.g. for weekly-roster
"taken" tracking outside of the actual draft event. Verified with a
synthetic-picks unit test (couldn't test against real picks - this league's
draft hasn't happened yet); confirmed the pre-draft (empty-picks) case is a
safe no-op against the live league.
