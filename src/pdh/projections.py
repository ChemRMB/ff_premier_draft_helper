"""
Season-weighted per-90 fantasy-point rates, team attack/defense power model,
and gameweek point projections.

Extracted from scripts/make_recommendations.py (previously ~675 lines of
module-level script code, sections "1) Build season-weighted per90 points"
through "5b) Multi-week projection for bench planning") so the projection
engine is reusable/testable independent of the draft-board/snake-planner CLI.
This is a behavior-preserving extraction: the computations are unchanged from
the original script, only parametrized (module globals -> function arguments).
"""

from __future__ import annotations
import difflib
import numpy as np
import pandas as pd

from pdh.scoring import points_from_row
from pdh import fpl
from pdh.namelink import link_names
from pdh.normalize import blend_with_shrinkage

POS_FPL_TO_GENERAL = {"GK": "GK", "DEF": "D", "MID": "M", "FWD": "F"}
POS_ALIAS = {
    "GKP": "GK",
    "GK": "GK",
    "GOALKEEPER": "GK",
    "KEEPER": "GK",
    "DEF": "D",
    "D": "D",
    "CB": "D",
    "LB": "D",
    "RB": "D",
    "WB": "D",
    "MID": "M",
    "M": "M",
    "AM": "M",
    "CM": "M",
    "DM": "M",
    "LM": "M",
    "RM": "M",
    "FWD": "F",
    "FW": "F",
    "ST": "F",
    "CF": "F",
    "LW": "F",
    "RW": "F",
    "ATT": "F",
    "F": "F",
}
STATUS_MULT = {"a": 1.00, "d": 0.65, "i": 0.00, "s": 0.00, "u": 0.00}
POS_PRIOR = {"GK": 90.0, "D": 80.0, "M": 75.0, "F": 75.0}

# FBref "team" -> FPL "team_name" (canonical)
ALIAS_FBREF_TO_FPL = {
    "Manchester United": "Man Utd",
    "Man United": "Man Utd",
    "Manchester Utd": "Man Utd",
    "Manchester City": "Man City",
    "Tottenham Hotspur": "Spurs",
    "Tottenham": "Spurs",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton": "Wolves",
    "Nottingham Forest": "Nott'm Forest",
    "Newcastle United": "Newcastle",
    "Leeds United": "Leeds",
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Sheffield United": "Sheffield Utd",
    "Leicester City": "Leicester",
    "Luton Town": "Luton",
    "Ipswich Town": "Ipswich",
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Liverpool": "Liverpool",
    "Newcastle": "Newcastle",
    "Spurs": "Spurs",
    "Man Utd": "Man Utd",
    "Man City": "Man City",
    "West Ham": "West Ham",
    "Wolves": "Wolves",
    "Leeds": "Leeds",
    "Brighton": "Brighton",
}


def to_canon_pos(x):
    if pd.isna(x):
        return None
    t = str(x).upper().strip()
    return POS_ALIAS.get(t, t if t in ("F", "M", "D", "GK") else None)


def attack_mult(diff):  # For F & M
    d = int(diff) if pd.notna(diff) else 3
    return {2: 1.10, 3: 1.00, 4: 0.93, 5: 0.85}.get(d, 1.00)


def defend_mult(diff):  # For D & GK
    d = int(diff) if pd.notna(diff) else 3
    return {2: 1.10, 3: 1.00, 4: 0.93, 5: 0.88}.get(d, 1.00)


def coalesce_minutes_rowwise(df: pd.DataFrame) -> pd.Series:
    cands = [
        "summary_min",
        "passing_min",
        "defense_min",
        "possession_min",
        "misc_min",
        "keepers_min",
    ]
    avail = [c for c in cands if c in df.columns]
    if not avail:
        return pd.Series(np.nan, index=df.index)
    return df[avail].bfill(axis=1).iloc[:, 0]


# ---------------------------
# xG/xA blending (Understat) - Plan B
# ---------------------------
def link_understat_to_fbref_names(
    understat_df: pd.DataFrame, fbref_names: pd.DataFrame, cutoff: float = 0.90
) -> pd.DataFrame:
    """
    Link Understat's `player`/`team` to the FBref `player` names used in
    `hist`/`per_season`, via src/pdh/namelink. Both sources use full
    "First Last" names (unlike FPL's abbreviated web_name), so a direct
    canonical + team-scoped fuzzy match works without the web_name workaround
    needed for Sleeper<->FPL matching.

    `fbref_names` needs a `player` column and, ideally, a `team` column (e.g.
    `hist[["player", "team"]].drop_duplicates()`). Returns `understat_df` with
    an added `player_fbref` column (NaN where unmatched).
    """
    linked = link_names(
        source=understat_df,
        target=fbref_names,
        source_name_col="player",
        target_name_col="player",
        source_team_col="team" if "team" in understat_df.columns else None,
        target_team_col="team" if "team" in fbref_names.columns else None,
        cutoff=cutoff,
    )
    return linked.rename(columns={"matched_name": "player_fbref"})


def fpl_xg_table(
    squads_df: pd.DataFrame,
    season_end_year: int,
    min_minutes: float = 90.0,
) -> pd.DataFrame:
    """
    FPL's own current-season expected_goals_per_90/expected_assists_per_90
    (bootstrap-static `elements`, see pdh.fpl.current_squads_df), already
    keyed to FBref names via the `player_fbref` column added to `squads_df`
    by the FPL<->FBref name link in make_recommendations.py. Meant to
    supplement Understat in blend_expected_goals: FPL's public API doesn't
    soft-block the way Understat can, so it fills gaps where Understat has no
    row for a player (blocked fetch, or a name-link miss) rather than
    replacing Understat where it *does* have data.

    Only covers `season_end_year` (the current season) - bootstrap-static
    only exposes cumulative current-season totals, not a season-by-season
    history like Understat's player_season_stats. Rows with fewer than
    `min_minutes` played are dropped: FPL reports exactly 0.00 (not NaN) for
    a per-90 rate with ~no minutes played, which would otherwise look like a
    real (and misleadingly confident) zero-xG signal.
    """
    needed = {
        "player_fbref",
        "expected_goals_per_90",
        "expected_assists_per_90",
        "fpl_minutes",
    }
    if squads_df is None or squads_df.empty or not needed.issubset(squads_df.columns):
        return pd.DataFrame(columns=["player", "season", "xg_p90", "xa_p90"])

    out = squads_df[list(needed)].dropna(subset=["player_fbref"]).copy()
    out["fpl_minutes"] = pd.to_numeric(out["fpl_minutes"], errors="coerce")
    out = out[out["fpl_minutes"] >= min_minutes]
    out["xg_p90"] = pd.to_numeric(out["expected_goals_per_90"], errors="coerce")
    out["xa_p90"] = pd.to_numeric(out["expected_assists_per_90"], errors="coerce")
    out["season"] = season_end_year
    out = out.rename(columns={"player_fbref": "player"})[
        ["player", "season", "xg_p90", "xa_p90"]
    ]
    return out.groupby(["player", "season"], as_index=False).mean(numeric_only=True)


def blend_expected_goals(
    per_season: pd.DataFrame,
    hist: pd.DataFrame,
    understat_df: pd.DataFrame | None,
    k: float = 900.0,
    fpl_xg_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Blend FBref actual goals_p90/assists_p90 in `per_season` toward an
    expected-goals/assists rate using shrinkage (normalize.blend_with_shrinkage),
    trusting actuals more as a player accumulates minutes that season. Rows
    with no expected-rate match, or missing goals_p90/assists_p90 to begin
    with, are left untouched.

    `understat_df` is the raw output of `pdh.understat.player_season_stats()`
    (columns include player, team, season_id, minutes, goals, xg, assists,
    xa) and is the primary xG/xA source, covering all historical seasons.
    `fpl_xg_df` (see fpl_xg_table) supplements it for the current season with
    FPL's own expected_goals_per_90/expected_assists_per_90, filling rows
    Understat is missing (soft-blocked fetch, or a name-link miss) rather
    than overriding rows it already has.
    """
    if "goals_p90" not in per_season.columns and "assists_p90" not in per_season.columns:
        return per_season

    have_understat = understat_df is not None and not understat_df.empty
    have_fpl = fpl_xg_df is not None and not fpl_xg_df.empty
    if not have_understat and not have_fpl:
        return per_season

    if have_understat:
        fbref_names = (
            hist[["player", "team"]].drop_duplicates()
            if "team" in hist.columns
            else hist[["player"]].drop_duplicates()
        )
        linked = link_understat_to_fbref_names(understat_df, fbref_names).dropna(
            subset=["player_fbref"]
        )
    else:
        linked = pd.DataFrame()

    if not linked.empty:
        u = linked.copy()
        # Understat's season_id is the start year (e.g. 2024 for 2024/25); hist/FBref
        # use the end year (2025) - see src/pdh/understat.py::season_to_understat_str.
        u["season"] = pd.to_numeric(u["season_id"], errors="coerce") + 1
        minutes = pd.to_numeric(u["minutes"], errors="coerce")
        m90 = (minutes / 90.0).replace(0, np.nan)
        u["xg_p90"] = pd.to_numeric(u["xg"], errors="coerce") / m90
        u["xa_p90"] = pd.to_numeric(u["xa"], errors="coerce") / m90
        u = (
            u[["player_fbref", "season", "xg_p90", "xa_p90"]]
            .rename(columns={"player_fbref": "player"})
            .groupby(["player", "season"], as_index=False)
            .mean(numeric_only=True)
        )
    else:
        u = pd.DataFrame(columns=["player", "season", "xg_p90", "xa_p90"])

    if have_fpl:
        u = u.merge(fpl_xg_df, on=["player", "season"], how="outer", suffixes=("", "_fpl"))
        if "xg_p90_fpl" in u.columns:
            u["xg_p90"] = pd.to_numeric(u["xg_p90"], errors="coerce").fillna(
                pd.to_numeric(u.pop("xg_p90_fpl"), errors="coerce")
            )
        if "xa_p90_fpl" in u.columns:
            u["xa_p90"] = pd.to_numeric(u["xa_p90"], errors="coerce").fillna(
                pd.to_numeric(u.pop("xa_p90_fpl"), errors="coerce")
            )

    if u.empty:
        return per_season

    season_minutes = (
        hist.groupby(["player", "season"])["minutes"]
        .sum()
        .reset_index()
        .rename(columns={"minutes": "season_minutes"})
    )

    out = per_season.merge(u, on=["player", "season"], how="left").merge(
        season_minutes, on=["player", "season"], how="left"
    )
    if "goals_p90" in out.columns:
        out["goals_p90"] = np.where(
            out["xg_p90"].notna(),
            blend_with_shrinkage(out["goals_p90"], out["xg_p90"], out["season_minutes"], k=k),
            out["goals_p90"],
        )
    if "assists_p90" in out.columns:
        out["assists_p90"] = np.where(
            out["xa_p90"].notna(),
            blend_with_shrinkage(out["assists_p90"], out["xa_p90"], out["season_minutes"], k=k),
            out["assists_p90"],
        )
    return out.drop(columns=["xg_p90", "xa_p90", "season_minutes"], errors="ignore")


# ---------------------------
# 1) Season-weighted per-90 fantasy points, by position
# ---------------------------
def build_weighted_points90(
    hist: pd.DataFrame,
    weights: dict,
    scoring: dict,
    stat_map: dict,
    understat_df: pd.DataFrame | None = None,
    xg_shrink_k: float = 900.0,
    fpl_xg_df: pd.DataFrame | None = None,
    points90_shrink_k: float = 900.0,
) -> pd.DataFrame:
    """
    One row per player: season-weighted per-90 Sleeper fantasy points at each
    position (points90w_F/M/D/GK), plus a season-weighted historical minutes
    baseline (hist_minutes_w).

    If `understat_df` is given (see pdh.understat.player_season_stats), actual
    goals_p90/assists_p90 are first shrunk toward Understat's xG_p90/xA_p90
    (see blend_expected_goals) before being scored - more stable projections,
    especially for low-minute samples where finishing variance dominates raw
    actuals. `fpl_xg_df` (see fpl_xg_table) supplements Understat for the
    current season using FPL's own expected_goals_per_90/expected_assists_per_90,
    filling gaps rather than overriding Understat where it already has data -
    useful since Understat can soft-block after repeated automated requests.
    Omitting both (the default) reproduces the original, actuals-only
    behavior exactly.

    Separately (and always applied, not optional): each player-season's raw
    points90_{pos} is itself shrunk toward that season's across-league mean
    at that position, weighted by the player's total minutes that season
    (`normalize.blend_with_shrinkage`, `k=points90_shrink_k`). Without this,
    a handful of low-minute cameo appearances (e.g. 4 minutes with a goal)
    can produce a nonsensical 100+ points-per-90 rate that then dominates
    team-power aggregates (build_team_power's roster_att/roster_def, which
    sum the top-N players by this rate) and that player's own projection
    alike - the rate itself carries no signal about reliability at tiny
    sample sizes, only the shrinkage toward a sane prior does.
    """
    hist = hist.copy()
    if "minutes" not in hist.columns:
        hist["minutes"] = coalesce_minutes_rowwise(hist)
    per90_cols = [c for c in hist.columns if c.endswith("_p90")]

    per_season = hist.groupby(["player", "season"])[per90_cols].mean().reset_index()

    if understat_df is not None or fpl_xg_df is not None:
        per_season = blend_expected_goals(
            per_season, hist, understat_df, k=xg_shrink_k, fpl_xg_df=fpl_xg_df
        )

    mins_per_season = hist.groupby(["player", "season"])["minutes"].mean().reset_index()
    mins_tmp = mins_per_season.copy()
    mins_tmp["_w"] = mins_tmp["season"].astype(str).map(weights).fillna(1.0)
    den_m = mins_tmp.groupby("player")["_w"].sum()
    num_m = (mins_tmp["minutes"] * mins_tmp["_w"]).groupby(mins_tmp["player"]).sum()
    min_weighted = (num_m / den_m).rename("hist_minutes_w").reset_index()

    def score_row_per90(row: pd.Series, position: str) -> float:
        base = {c[:-4]: row[c] for c in per90_cols if pd.notna(row[c])}
        return points_from_row(pd.Series(base), position, scoring, stat_map)

    for pos in ["F", "M", "D", "GK"]:
        per_season[f"points90_{pos}"] = per_season.apply(
            lambda r: score_row_per90(r, pos), axis=1
        )

    season_minutes_sum = (
        hist.groupby(["player", "season"])["minutes"]
        .sum()
        .reset_index()
        .rename(columns={"minutes": "season_minutes_sum"})
    )
    per_season = per_season.merge(season_minutes_sum, on=["player", "season"], how="left")
    for pos in ["F", "M", "D", "GK"]:
        col = f"points90_{pos}"
        season_prior = per_season.groupby("season")[col].transform("mean")
        per_season[col] = blend_with_shrinkage(
            per_season[col],
            season_prior,
            per_season["season_minutes_sum"],
            k=points90_shrink_k,
        )

    agg = per_season.copy()
    agg["_w"] = agg["season"].astype(str).map(weights).fillna(1.0)
    den = agg.groupby("player")["_w"].sum()

    def wmean(col: str) -> pd.Series:
        return (agg[col] * agg["_w"]).groupby(agg["player"]).sum() / den

    weighted = pd.DataFrame(
        {
            "player": den.index,
            "points90w_F": wmean("points90_F").reindex(den.index).values,
            "points90w_M": wmean("points90_M").reindex(den.index).values,
            "points90w_D": wmean("points90_D").reindex(den.index).values,
            "points90w_GK": wmean("points90_GK").reindex(den.index).values,
        }
    ).merge(min_weighted, on="player", how="left")

    return weighted


# ---------------------------
# 3) Team power model (attack + defense)
# ---------------------------
ATT_FEATURES_CAND = [
    "Progression_PrgC",
    "Progression_PrgP",
    "Per 90 Minutes_Gls",
    "Per 90 Minutes_Ast",
    "Per 90 Minutes_G+A",
    "Per 90 Minutes_G-PK",
    "Per 90 Minutes_G+A-PK",
    "Per 90 Minutes_xG",
    "Per 90 Minutes_xAG",
    "Per 90 Minutes_xG+xAG",
    "Per 90 Minutes_npxG",
    "Per 90 Minutes_npxG+xAG",
]


def _zscore_per_season(df, cols, invert: bool = False, label="idx"):
    out = df.copy()
    if not cols:
        out[label] = 0.0
        return out[["team", "season", label]]
    for c in cols:
        mu = out[c].mean()
        sd = out[c].std(ddof=0)
        if sd == 0 or np.isnan(sd):
            out[c + "_z"] = 0.0
        else:
            z = (out[c] - mu) / sd
            out[c + "_z"] = (-z) if invert else z
    zcols = [c + "_z" for c in cols]
    out[f"{label}"] = out[zcols].mean(axis=1)
    return out[["team", "season", label]]


def _agg_team_season_def(grp: pd.DataFrame) -> pd.Series:
    mins = grp["minutes"].sum()
    ga = grp["goals_conceded"].sum(min_count=1)
    if "game_id" in grp.columns:
        n_matches = grp["game_id"].nunique()
    else:
        n_matches = len(grp)
    cs = grp["clean_sheet"].sum(min_count=1)
    ga_p90 = np.nan
    if pd.notna(ga) and mins > 0:
        ga_p90 = ga / (mins / 90.0)
    cs_rate = np.nan
    if pd.notna(cs) and n_matches > 0:
        cs_rate = cs / n_matches
    return pd.Series({"ga_p90": ga_p90, "cs_rate": cs_rate})


def _norm_mean1(s: pd.Series) -> pd.Series:
    """
    For strictly-positive metrics (e.g. roster_att/roster_def - sums of
    points90 rates, always >= 0): normalize to mean=1.0 by dividing by the
    mean. NOT valid for signed, ~zero-mean inputs like z-scores - see
    _zscore_to_mult for those (dividing a value near 0 by a mean near 0
    explodes the scale and can flip sign, producing a nonsensical negative
    "power" multiplier).
    """
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    if mu == 0 or np.isnan(mu):
        return s.fillna(0.0).apply(lambda x: 1.0)
    return (s / mu).replace([np.inf, -np.inf], 1.0).fillna(1.0)


def _zscore_to_mult(
    s: pd.Series, scale: float = 0.15, clip: tuple[float, float] = (0.7, 1.4)
) -> pd.Series:
    """
    Convert a z-score-like index (e.g. attack_idx_hist/defense_idx_hist from
    _zscore_per_season - mean ~0, can be negative) into a multiplicative
    index centered at 1.0: `1.0 + scale * z`, clipped to a modest range so
    one team's history can't dominate a blend or send projected points
    negative. `scale=0.15` means a team 1 std-dev above average gets a 15%
    boost; `clip` bounds the most extreme teams (e.g. the current-season-only
    Sunderland with no real multi-season history, z=0) to a sane range.
    """
    s = pd.to_numeric(s, errors="coerce")
    return (1.0 + scale * s).clip(*clip).fillna(1.0)


def _winsor_mean1_shrink(
    s: pd.Series,
    p_lo=0.10,
    p_hi=0.90,
    shrink=0.5,
    final_clip: tuple[float, float] | None = (0.90, 1.10),
) -> pd.Series:
    """
    Winsorize to [p_lo, p_hi] percentiles, normalize to mean=1, then shrink
    toward 1. Optionally apply a light final clip. Not currently applied in
    build_team_power (kept, as in the original script, for future tuning).
    """
    x = pd.to_numeric(s, errors="coerce").copy()
    lo = x.quantile(p_lo)
    hi = x.quantile(p_hi)
    x = x.clip(lo, hi)
    mu = x.mean()
    if not np.isfinite(mu) or mu == 0:
        x = x.fillna(1.0)
        mu = 1.0
    x = x / mu
    x = 1.0 + shrink * (x - 1.0)
    if final_clip:
        lo_c, hi_c = final_clip
        x = x.clip(lo_c, hi_c)
    return x.fillna(1.0)


def _fuzzy_pick(name: str, cands: pd.DataFrame, col: str, cutoff=0.78):
    if cands.empty:
        return np.nan, None, 0.0
    pool = cands["team_fpl_like"].tolist()
    best = difflib.get_close_matches(name, pool, n=1, cutoff=cutoff)
    if not best:
        return np.nan, None, 0.0
    best_name = best[0]
    score = difflib.SequenceMatcher(None, name, best_name).ratio()
    val = cands.loc[cands["team_fpl_like"] == best_name, col].iloc[0]
    return val, best_name, score


def build_team_power(
    team_season_stats: pd.DataFrame,
    hist: pd.DataFrame,
    squads: pd.DataFrame,
    weighted: pd.DataFrame,
    weights: dict,
    alpha_hist: float = 0.2,
    alpha_ros: float = 0.8,
    roster_top_n_att: int = 10,
    roster_top_n_def: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Blend historical team attack/defense strength (z-scored team_season_stats
    features + keeper logs in `hist`) with current-roster strength (from
    `squads` + `weighted` per-90 points) into per-team attack_power/
    defense_power multipliers, mean ~1.0 across the league.

    `squads` must already have a `player_fbref` column (see name-linking in
    scripts/make_recommendations.py) and a numeric `team` id.
    """
    ts = team_season_stats.copy()
    ts = ts.dropna(subset=["team", "season"])
    att_cols = [c for c in ATT_FEATURES_CAND if c in ts.columns]
    for c in att_cols:
        ts[c] = pd.to_numeric(ts[c], errors="coerce")

    ts_att_idx = _zscore_per_season(
        ts[["team", "season"] + att_cols], att_cols, invert=False, label="attack_idx_season"
    )
    tmpa = ts_att_idx.copy()
    tmpa["_w"] = tmpa["season"].astype(str).map(weights).fillna(1.0)
    den_att = tmpa.groupby("team")["_w"].sum()
    num_att = (tmpa["attack_idx_season"] * tmpa["_w"]).groupby(tmpa["team"]).sum()
    team_attack_hist = (num_att / den_att).rename("attack_idx_hist").reset_index()

    # Defense index from GK logs in hist (GA/90 low good; CS rate high good)
    gk_mask = pd.Series(False, index=hist.index)
    if "keepers_min" in hist.columns:
        gk_mask = gk_mask | hist["keepers_min"].notna()
    for col in ["keepers_Shot Stopping_Saves", "keepers_Shot Stopping_GA"]:
        if col in hist.columns:
            gk_mask = gk_mask | hist[col].notna()
    gk = hist[gk_mask].copy()
    gk["minutes"] = gk["minutes"].fillna(0)
    for need in ["goals_conceded", "clean_sheet"]:
        if need not in gk.columns:
            gk[need] = np.nan

    def_df = (
        gk.groupby(["team", "season"], dropna=False)
        .apply(_agg_team_season_def, include_groups=False)
        .reset_index()
    )
    for c in ["ga_p90", "cs_rate"]:
        def_df[c] = pd.to_numeric(def_df[c], errors="coerce")
    mu_ga, sd_ga = def_df["ga_p90"].mean(), def_df["ga_p90"].std(ddof=0)
    mu_cs, sd_cs = def_df["cs_rate"].mean(), def_df["cs_rate"].std(ddof=0)
    def_df["ga_p90_z"] = (
        0.0 if (sd_ga == 0 or np.isnan(sd_ga)) else (mu_ga - def_df["ga_p90"]) / sd_ga
    )
    def_df["cs_rate_z"] = (
        0.0 if (sd_cs == 0 or np.isnan(sd_cs)) else (def_df["cs_rate"] - mu_cs) / sd_cs
    )
    def_df["defense_idx_season"] = def_df[["ga_p90_z", "cs_rate_z"]].mean(axis=1)
    tmpd = def_df.copy()
    tmpd["_w"] = tmpd["season"].astype(str).map(weights).fillna(1.0)
    den_def = tmpd.groupby("team")["_w"].sum()
    num_def = (tmpd["defense_idx_season"] * tmpd["_w"]).groupby(tmpd["team"]).sum()
    team_defense_hist = (num_def / den_def).rename("defense_idx_hist").reset_index()

    fb_att = team_attack_hist.copy()
    fb_def = team_defense_hist.copy()
    fb_att["team_fpl_like"] = fb_att["team"].map(ALIAS_FBREF_TO_FPL).fillna(fb_att["team"])
    fb_def["team_fpl_like"] = fb_def["team"].map(ALIAS_FBREF_TO_FPL).fillna(fb_def["team"])

    team_names_fpl = squads[["team", "team_name"]].drop_duplicates().copy()

    fb_att["_join"] = fb_att["team_fpl_like"].str.lower()
    fb_def["_join"] = fb_def["team_fpl_like"].str.lower()
    team_names_fpl["_join"] = team_names_fpl["team_name"].str.lower()

    team_power = (
        team_names_fpl.merge(fb_att[["_join", "attack_idx_hist"]], on="_join", how="left")
        .merge(fb_def[["_join", "defense_idx_hist"]], on="_join", how="left")
        .drop(columns=["_join"])
        .drop_duplicates(subset=["team", "team_name"])
    )

    need_attack = team_power["attack_idx_hist"].isna()
    need_def = team_power["defense_idx_hist"].isna()

    if need_attack.any() or need_def.any():
        fb_att_cands = fb_att[["team_fpl_like", "attack_idx_hist"]].dropna().drop_duplicates()
        fb_def_cands = fb_def[["team_fpl_like", "defense_idx_hist"]].dropna().drop_duplicates()
        fuzzy_hits = []

        tp = team_power.copy()
        for idx, row in tp.iterrows():
            tname = str(row["team_name"])
            if need_attack.iloc[idx]:
                val, cand, sc = _fuzzy_pick(tname, fb_att_cands, "attack_idx_hist", cutoff=0.78)
                if pd.notna(val):
                    tp.at[idx, "attack_idx_hist"] = val
                    fuzzy_hits.append(f"[fuzzy-attack] '{tname}' -> '{cand}' (score={sc:.2f})")
            if need_def.iloc[idx]:
                val, cand, sc = _fuzzy_pick(tname, fb_def_cands, "defense_idx_hist", cutoff=0.78)
                if pd.notna(val):
                    tp.at[idx, "defense_idx_hist"] = val
                    fuzzy_hits.append(f"[fuzzy-defense] '{tname}' -> '{cand}' (score={sc:.2f})")
        team_power = tp

        if verbose and fuzzy_hits:
            print("\n".join(fuzzy_hits))

    # Current roster strengths (attack/defense)
    roster_tbl = squads.copy()
    roster_tbl["pos"] = roster_tbl["pos"].map(POS_FPL_TO_GENERAL).fillna(roster_tbl["pos"])
    roster_tbl["pos"] = roster_tbl["pos"].apply(to_canon_pos)
    roster_tbl = roster_tbl.merge(
        weighted[["player", "points90w_F", "points90w_M", "points90w_D", "points90w_GK"]],
        left_on="player_fbref",
        right_on="player",
        how="left",
    )
    roster_tbl["points90w_sel"] = np.where(
        roster_tbl["pos"].eq("F"),
        roster_tbl["points90w_F"],
        np.where(
            roster_tbl["pos"].eq("M"),
            roster_tbl["points90w_M"],
            np.where(
                roster_tbl["pos"].eq("D"),
                roster_tbl["points90w_D"],
                np.where(
                    roster_tbl["pos"].eq("GK"),
                    roster_tbl["points90w_GK"],
                    roster_tbl["points90w_M"],
                ),
            ),
        ),
    )
    roster_att = (
        roster_tbl[roster_tbl["pos"].isin(["F", "M"])]
        .dropna(subset=["team_name", "points90w_sel"])
        .sort_values(["team_name", "points90w_sel"], ascending=[True, False])
        .groupby("team_name")
        .head(roster_top_n_att)
        .groupby("team_name")["points90w_sel"]
        .sum()
        .rename("roster_att")
        .reset_index()
    )
    roster_tbl["points90w_def_component"] = np.where(
        roster_tbl["pos"].eq("GK"),
        roster_tbl["points90w_GK"] * 1.2,
        np.where(roster_tbl["pos"].eq("D"), roster_tbl["points90w_D"], 0.0),
    )
    roster_def = (
        roster_tbl[roster_tbl["pos"].isin(["D", "GK"])]
        .dropna(subset=["team_name", "points90w_def_component"])
        .sort_values(["team_name", "points90w_def_component"], ascending=[True, False])
        .groupby("team_name")
        .head(roster_top_n_def)
        .groupby("team_name")["points90w_def_component"]
        .sum()
        .rename("roster_def")
        .reset_index()
    )

    team_power = team_power.merge(roster_att, on="team_name", how="left").merge(
        roster_def, on="team_name", how="left"
    )

    for c in ["attack_idx_hist", "defense_idx_hist", "roster_att", "roster_def"]:
        team_power[c] = pd.to_numeric(team_power[c], errors="coerce")

    # attack_idx_hist/defense_idx_hist are z-scores (mean ~0, can be
    # negative) - _zscore_to_mult, not _norm_mean1, is the correct transform
    # (see both docstrings). roster_att/roster_def are positive sums of
    # points90 rates, where _norm_mean1 is valid.
    team_power["attack_hist_n"] = _zscore_to_mult(team_power["attack_idx_hist"])
    team_power["roster_att_n"] = _norm_mean1(team_power["roster_att"])
    team_power["defense_hist_n"] = _zscore_to_mult(team_power["defense_idx_hist"])
    team_power["roster_def_n"] = _norm_mean1(team_power["roster_def"])

    att_hist_n = team_power["attack_hist_n"].fillna(1.0)
    ros_att_n = team_power["roster_att_n"].fillna(1.0)
    def_hist_n = team_power["defense_hist_n"].fillna(1.0)
    ros_def_n = team_power["roster_def_n"].fillna(1.0)

    att_blend = alpha_hist * att_hist_n + alpha_ros * ros_att_n
    def_blend = alpha_hist * def_hist_n + alpha_ros * ros_def_n

    team_power["attack_power"] = att_blend
    team_power["defense_power"] = def_blend

    if verbose:
        for label, col in [("attack_power", "attack_power"), ("defense_power", "defense_power")]:
            x = team_power[col]
            print(
                f"[dbg] {label}: min={x.min():.3f}, p10={x.quantile(0.10):.3f}, "
                f"median={x.median():.3f}, p90={x.quantile(0.90):.3f}, max={x.max():.3f}, mean={x.mean():.3f}"
            )
        dbg_missing = team_power[
            team_power["attack_idx_hist"].isna() | team_power["defense_idx_hist"].isna()
        ]
        if not dbg_missing.empty:
            print("[debug] still missing history for:", ", ".join(dbg_missing["team_name"].tolist()))

    return team_power


# ---------------------------
# 4-5) Expected minutes + gameweek/multi-week projections
# ---------------------------
def compute_exp_minutes(row: pd.Series, chance_field: str) -> float:
    base = row.get("hist_minutes_w", np.nan)
    if pd.isna(base):
        base = POS_PRIOR.get(row.get("pos"), 75.0)
    status = str(row.get("status", "")).lower()
    sm = STATUS_MULT.get(status, 0.80 if status == "" else 0.0)
    cp = row.get(chance_field, np.nan)
    cm = float(cp) / 100.0 if pd.notna(cp) else 0.50
    est = base * sm * cm
    return float(max(0.0, min(90.0, est)))


def _select_points90(df: pd.DataFrame) -> np.ndarray:
    return np.where(
        df["pos"].eq("F"),
        df["points90w_F"],
        np.where(
            df["pos"].eq("M"),
            df["points90w_M"],
            np.where(
                df["pos"].eq("D"),
                df["points90w_D"],
                np.where(df["pos"].eq("GK"), df["points90w_GK"], df["points90w_M"]),
            ),
        ),
    )


def proj_points_row(r: pd.Series) -> float:
    pos = r["pos"]
    if pos not in ("F", "M", "D", "GK"):
        return 0.0
    points90 = float(r["points90w_sel"])
    minutes = float(r.get("exp_minutes", 0.0))
    diff = r.get("gw_difficulty", 3)
    mult_pos = attack_mult(diff) if pos in ("F", "M") else defend_mult(diff)
    mult_team = (
        float(r.get("attack_power", 1.0))
        if pos in ("F", "M")
        else float(r.get("defense_power", 1.0))
    )
    return points90 * (minutes / 90.0) * mult_pos * mult_team


def event_team_difficulty_map(fixtures: pd.DataFrame, event: int) -> dict[int, int]:
    fx = fixtures[fixtures["event"] == event]
    td = {}
    for _, r in fx.iterrows():
        td[int(r["team_h"])] = int(r["team_h_difficulty"])
        td[int(r["team_a"])] = int(r["team_a_difficulty"])
    return td


def proj_for_event(df: pd.DataFrame, fixtures: pd.DataFrame, event: int) -> pd.Series:
    td = event_team_difficulty_map(fixtures, event)
    pos_vals = df["pos"].fillna("M").to_numpy()
    diffs = df["team"].map(td).to_numpy()
    mins = df["exp_minutes"].to_numpy()
    pts90 = np.where(
        pos_vals == "F",
        df["points90w_F"].to_numpy(),
        np.where(
            pos_vals == "M",
            df["points90w_M"].to_numpy(),
            np.where(
                pos_vals == "D",
                df["points90w_D"].to_numpy(),
                np.where(
                    pos_vals == "GK",
                    df["points90w_GK"].to_numpy(),
                    df["points90w_M"].to_numpy(),
                ),
            ),
        ),
    )
    mult_pos = np.array(
        [attack_mult(d) if p in ("F", "M") else defend_mult(d) for d, p in zip(diffs, pos_vals)]
    )
    mult_team = np.where(
        np.isin(pos_vals, ["F", "M"]),
        df["attack_power"].fillna(1.0).to_numpy(),
        df["defense_power"].fillna(1.0).to_numpy(),
    )
    return pd.Series(pts90 * (mins / 90.0) * mult_pos * mult_team, index=df.index)


def project_players(
    squads: pd.DataFrame,
    weighted: pd.DataFrame,
    team_power: pd.DataFrame,
    fixtures: pd.DataFrame,
    gw: int,
    chance_field: str,
    weeks_for_bench: int = 4,
) -> pd.DataFrame:
    """
    Assemble the full projected-players frame for gameweek `gw`: merges
    `weighted` per-90 points onto `squads` (via player_fbref), merges team
    attack/defense power, computes expected minutes, this-gameweek proj_points,
    and the proj_nextN average over the following `weeks_for_bench` events.
    """
    squads = squads.copy()
    squads["pos"] = squads["pos"].map(POS_FPL_TO_GENERAL).fillna(squads["pos"])
    players = squads.merge(weighted, left_on="player_fbref", right_on="player", how="left")

    # ensure numeric team id if still missing (rare)
    if "team" not in players.columns or players["team"].isna().all():
        boot = fpl.get_bootstrap()
        teams_df = pd.DataFrame(boot["teams"])[["id", "name", "short_name"]]
        players = players.merge(
            teams_df.rename(columns={"id": "team"}), left_on="team_name", right_on="name", how="left"
        )

    gw_fix = fixtures[fixtures["event"] == gw].copy()
    team_diff = {}
    for _, r in gw_fix.iterrows():
        team_diff[int(r["team_h"])] = int(r["team_h_difficulty"])
        team_diff[int(r["team_a"])] = int(r["team_a_difficulty"])
    players["gw_difficulty"] = players["team"].map(team_diff)

    # Guard against NaN team ids: pandas merges NaN==NaN, so if several teams
    # failed id resolution (e.g. pre-season FPL API drift), every player on
    # any such team would cartesian-join every NaN-team row in team_power -
    # duplicating players and cross-contaminating their team power. Only
    # merge rows with a real team id; NaN-team players keep neutral (1.0)
    # power via the fillna below.
    tp_keyed = team_power[["team", "team_name", "attack_power", "defense_power"]].dropna(
        subset=["team"]
    )
    players = players.merge(
        tp_keyed,
        on="team",
        how="left",
        suffixes=("", "_tp"),
    )
    if "team_name_tp" in players.columns:
        players["team_name"] = players["team_name"].fillna(players["team_name_tp"])
        players = players.drop(columns=["team_name_tp"])
    players["attack_power"] = players["attack_power"].fillna(1.0)
    players["defense_power"] = players["defense_power"].fillna(1.0)

    players["pos"] = players["pos"].apply(to_canon_pos)
    players["exp_minutes"] = players.apply(lambda r: compute_exp_minutes(r, chance_field), axis=1)
    players["points90w_sel"] = _select_points90(players)
    players["proj_points"] = players.apply(proj_points_row, axis=1)

    players["proj_nextN"] = players["proj_points"].astype(float)
    future_events = [e for e in range(gw, gw + weeks_for_bench)]
    acc = pd.Series(0.0, index=players.index, dtype="float64")
    denN = 0
    for e in future_events:
        if (fixtures["event"] == e).any():
            acc = acc.add(proj_for_event(players, fixtures, e), fill_value=0.0)
            denN += 1
    if denN > 0:
        players["proj_nextN"] = acc / denN

    return players
