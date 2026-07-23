"""
Player "stickiness": which players are reliable enough weekly scorers that
trade suggestions (Plan D) should down-rank or exclude them - nobody
realistically trades away a player who is both a big step above replacement
level *and* consistent about it, even at a good price.

Source of truth: real per-gameweek Sleeper fantasy points (`pts_std` from
pdh.sleeper.get_player_week_stats) for the most recently completed season -
actual outcomes, not our own projections, so this can't just circularly
validate the projection engine.

Statistical design
-------------------
The prompt for this was "should we use Dunnett's T3 test": T3 (like
Games-Howell/Tamhane's T2) is an *all-pairs* unequal-variance post-hoc
procedure - built for "compare every group to every other group". That's not
this problem: we don't care whether Haaland is significantly stickier than
Salah, only whether each individual player is significantly and reliably
above a *replacement-level control group* at their position. That's the
shape of the original Dunnett test (many treatments vs one control), just
with unequal variances/sample sizes across players (some played every week,
some were rotated/transferred) - which T3's underlying machinery (robust to
heteroscedasticity) is exactly built to handle.

So: per player, a one-sided Welch's t-test (scipy.stats.ttest_ind,
equal_var=False - the same heteroscedasticity-robust approach T3 uses,
without needing T3's specific all-pairs studentized-modulus tables) against
a pooled control group of replacement-level players at the same position.
Testing hundreds of players simultaneously inflates false positives, so
p-values are Benjamini-Hochberg FDR-corrected across all players tested.

Statistical significance alone only says "this player scores more than
replacement" - it doesn't capture *reliability*, which is the actual
"untradeable" signal. So the reliability check also requires: a minimum
number of weeks played (else a 3-game hot streak looks "significant" on
paper), and a capped coefficient of variation (std/mean of weekly points,
computed over *all* season weeks including 0s for DNPs - a boom/bust or
rotation-prone player would otherwise pass despite an unreliable week-to-
week floor).

The CV cap is deliberately not the only route to "sticky", though: a
genuine superstar (Haaland, Musiala-tier) can have a high CV simply because
their game involves booms and blanks rather than plodding consistency, and
nobody trades that player away regardless. So a second, independent check
flags anyone whose season mean is a raw positive outlier at their position
(z-score based) - full stop, no CV or reliability requirement. `sticky` is
the OR of the reliability check and the outlier check.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

from pdh.sleeper import get_player_week_stats
from pdh.draft_board import roster_requirements, replacement_ranks

POS_COLS = {"F": "pos_f_min", "M": "pos_m_min", "D": "pos_d_min", "GK": "pos_gk_min"}


def _week_positions(week_df: pd.DataFrame) -> pd.Series:
    """Infer each player's scored position for one gameweek from which
    pos_{x}_min column is populated (matches the top-level `min`)."""
    pos = pd.Series(pd.NA, index=week_df.index, dtype="object")
    for label, col in POS_COLS.items():
        if col in week_df.columns:
            hit = week_df[col].notna() & (pos.isna())
            pos.loc[hit] = label
    return pos


def season_weekly_points(
    season: int, sport: str = "clubsoccer:epl", max_weeks: int = 38
) -> pd.DataFrame:
    """
    Long-format player x gameweek Sleeper points for a completed season:
    columns player_id, week, pos, pts_std. Only players who scored ANY
    points that week appear per-week (DNP weeks are absent here - see
    `season_matrix` for the zero-filled wide form stickiness needs).
    """
    rows = []
    for week in range(1, max_weeks + 1):
        wk_df = get_player_week_stats(season, week, sport=sport)
        if wk_df.empty or "pts_std" not in wk_df.columns:
            break
        wk_df = wk_df[wk_df["pts_std"].notna()].copy()
        wk_df["pos"] = _week_positions(wk_df)
        wk_df["week"] = week
        rows.append(wk_df[["player_id", "week", "pos", "pts_std"]])
    if not rows:
        return pd.DataFrame(columns=["player_id", "week", "pos", "pts_std"])
    return pd.concat(rows, ignore_index=True)


def season_matrix(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per player: their most common position that season, and a
    zero-filled `weekly_points` array covering week 1..max(week) (0 for
    weeks with no record - i.e. didn't play/wasn't in a squad that week).
    """
    if long_df.empty:
        return pd.DataFrame(columns=["player_id", "pos", "weekly_points", "weeks_played"])

    n_weeks = int(long_df["week"].max())
    out_rows = []
    for player_id, g in long_df.groupby("player_id"):
        pos = g["pos"].mode()
        pos = pos.iloc[0] if len(pos) and pd.notna(pos.iloc[0]) else None
        if pos is None:
            continue
        weekly = np.zeros(n_weeks)
        for _, r in g.iterrows():
            weekly[int(r["week"]) - 1] = r["pts_std"]
        out_rows.append(
            {
                "player_id": player_id,
                "pos": pos,
                "weekly_points": weekly,
                "weeks_played": int((weekly != 0).sum()),
            }
        )
    return pd.DataFrame(out_rows)


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction (no statsmodels dependency needed
    for this one function)."""
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out


def compute_stickiness(
    matrix: pd.DataFrame,
    roster_positions: list[str],
    teams: int,
    control_band: int = 7,
    min_weeks_played: int = 15,
    alpha: float = 0.05,
    cv_max: float = 0.75,
    outlier_z: float = 2.0,
) -> pd.DataFrame:
    """
    See module docstring for the statistical design. `matrix` is
    `season_matrix()`'s output. `roster_positions`/`teams` come from
    config/sleeper_setup.json (same replacement-rank convention as
    src/pdh/draft_board.py's VOR).

    Returns one row per eligible player (>= min_weeks_played) with:
    - `sticky_reliable`: p_value_adj < alpha (mean reliably above
      replacement) AND cv <= cv_max (consistent, not boom/bust).
    - `sticky_outlier`: mean_pts is a raw positive outlier at that position
      (z-score > outlier_z among eligible players there) - no CV
      requirement, so it still catches explosive high-ceiling stars whose
      week-to-week variance would otherwise fail the CV cap.
    - `sticky`: sticky_reliable OR sticky_outlier.
    """
    req = roster_requirements(roster_positions)
    repl_ranks = replacement_ranks(req, teams=teams)

    results = []
    for pos, rank in repl_ranks.items():
        if rank <= 0:
            continue
        pos_players = matrix[matrix["pos"] == pos].copy()
        if pos_players.empty:
            continue
        pos_players["season_total"] = pos_players["weekly_points"].apply(np.sum)
        pos_players = pos_players.sort_values(
            "season_total", ascending=False
        ).reset_index(drop=True)

        lo = max(0, rank - 1 - control_band)
        hi = min(len(pos_players), rank - 1 + control_band + 1)
        control_pool = pos_players.iloc[lo:hi]
        if control_pool.empty:
            continue
        control_weeks = np.concatenate(control_pool["weekly_points"].to_list())

        eligible = pos_players[pos_players["weeks_played"] >= min_weeks_played]
        pos_results = []
        for _, row in eligible.iterrows():
            weekly = row["weekly_points"]
            mean = float(weekly.mean())
            std = float(weekly.std(ddof=1)) if len(weekly) > 1 else 0.0
            cv = (std / mean) if mean > 0 else np.inf
            t_stat, p_val = stats.ttest_ind(
                weekly, control_weeks, equal_var=False, alternative="greater"
            )
            pos_results.append(
                {
                    "player_id": row["player_id"],
                    "pos": pos,
                    "weeks_played": row["weeks_played"],
                    "mean_pts": mean,
                    "std_pts": std,
                    "cv": cv,
                    "t_stat": t_stat,
                    "p_value": p_val,
                }
            )

        if pos_results:
            means = np.array([r["mean_pts"] for r in pos_results])
            m, s = means.mean(), means.std(ddof=1) if len(means) > 1 else 0.0
            for r in pos_results:
                z = (r["mean_pts"] - m) / s if s > 0 else 0.0
                r["z_score"] = z
                r["sticky_outlier"] = bool(z > outlier_z)
            results.extend(pos_results)

    out = pd.DataFrame(results)
    if out.empty:
        for c in ["p_value_adj", "sticky_reliable", "sticky_outlier", "sticky"]:
            out[c] = pd.Series(dtype=float if c == "p_value_adj" else bool)
        return out

    out["p_value_adj"] = _bh_fdr(out["p_value"].to_numpy())
    out["sticky_reliable"] = (out["p_value_adj"] < alpha) & (out["cv"] <= cv_max)
    out["sticky"] = out["sticky_reliable"] | out["sticky_outlier"]
    return out.sort_values(["sticky", "mean_pts"], ascending=[False, False]).reset_index(
        drop=True
    )


def vor_soft_tier(
    df: pd.DataFrame,
    pos_col: str = "pos",
    vor_col: str = "VOR",
    top_pct: float = 0.15,
) -> pd.Series:
    """
    Soft-stickiness down-rank tier: flags the top `top_pct` of players by VOR
    within each position (from src/pdh/draft_board.py's VOR - e.g. a
    draft_board_flexaware.csv from make_recommendations.py). This is meant
    to run alongside `compute_stickiness`'s `sticky` flag once real
    rosters/trade candidates exist: `sticky` -> exclude from trade-away
    suggestions entirely (statistically proven outlier or reliably above
    replacement); this tier -> down-rank but don't exclude (good projected
    value without that statistical proof, e.g. a strong player who's new to
    the league or coming back from injury with limited season-stats
    history).
    """
    cutoffs = df.groupby(pos_col)[vor_col].transform(lambda s: s.quantile(1 - top_pct))
    return df[vor_col] >= cutoffs
