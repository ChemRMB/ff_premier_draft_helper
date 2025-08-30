"""
Weekly projections + draft boards (non-flex and flex-aware) + snake-draft planner
with team strength (attack vs defense) multipliers and robust name mapping.

Usage:
  python scripts/make_recommendations.py --event 1 --topn 20 --my_team_id 1259256320754724864
"""

from __future__ import annotations
import argparse, json, re, unicodedata, difflib
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from src.pdh.scoring import load_sleeper_scoring, load_stat_map, points_from_row
from src.pdh import fpl  # used to fetch FPL team ids if missing


# ---------------------------
# CLI
# ---------------------------
ap = argparse.ArgumentParser()
ap.add_argument(
    "--event",
    type=int,
    default=None,
    help="FPL gameweek (event). If omitted, uses next unplayed.",
)
ap.add_argument("--topn", type=int, default=20, help="Rows to include in top lists.")
ap.add_argument(
    "--my_team_id",
    type=str,
    default="1259256320754724864",
    help="Your Sleeper team id (falls back to draft_order if omitted).",
)
ap.add_argument(
    "--weeks_for_bench",
    type=int,
    default=4,
    help="Lookahead weeks to rate bench picks.",
)
args = ap.parse_args()


# ---------------------------
# Paths & config
# ---------------------------
ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data" / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

with open(ROOT / "config" / "seasons.yaml", "r") as fh:
    cfg_seasons = yaml.safe_load(fh)

weights = {str(k): float(v) for k, v in cfg_seasons["weights"].items()}

sleeper_setup = json.load(open(ROOT / "config" / "sleeper_setup.json"))
sleeper_draft = json.load(open(ROOT / "config" / "sleeper_draft.json"))
teams_in_league = sleeper_draft["settings"]["teams"]

# Sleeper config
scoring = load_sleeper_scoring(str(ROOT / "config" / "sleeper_setup.json"))
stat_map = load_stat_map(str(ROOT / "config" / "stat_map.yaml"))
roster_positions = sleeper_setup["roster_positions"]


# ---------------------------
# Load data produced by other scripts
# ---------------------------
hist = pd.read_csv(
    ROOT / "data" / "outputs" / "players_historical.csv"
)  # from build_historical.py
squads = pd.read_csv(
    ROOT / "data" / "outputs" / "current_squads.csv"
)  # from update_current.py
fixtures = pd.read_csv(
    ROOT / "data" / "outputs" / "fixtures.csv"
)  # from update_current.py
team_season_stats = pd.read_csv(
    ROOT / "data" / "outputs" / "teams_season_stats.csv"
)  # from build_historical.py


# ---------------------------
# Helpers
# ---------------------------
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
FLEX_MAP = {"FM_FLEX": {"F", "M"}, "MD_FLEX": {"M", "D"}}

# FBref "team" → FPL "team_name" (canonical)
ALIAS_FBREF_TO_FPL = {
    # exact long ↔ short
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
    # identities (won't hurt)
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


def replacement_ranks_nonflex(
    roster_positions: list[str], teams: int
) -> dict[str, int]:
    base = {"F": 0, "M": 0, "D": 0, "GK": 0}
    for p in roster_positions:
        if p in base:
            base[p] += 1
    return {k: base[k] * teams for k in base}


def effective_replacement_ranks_with_flex(
    df: pd.DataFrame,
    pos_col: str,
    proj_col: str,
    roster_positions: list[str],
    teams: int,
) -> dict[str, int]:
    base = {"F": 0, "M": 0, "D": 0, "GK": 0}
    flex_counts = {"FM_FLEX": 0, "MD_FLEX": 0}
    for p in roster_positions:
        if p in base:
            base[p] += 1
        elif p in flex_counts:
            flex_counts[p] += 1
    counts = {k: base[k] * teams for k in base}
    pools = {}
    for p in base:
        vals = (
            df[df[pos_col] == p]
            .sort_values(proj_col, ascending=False)[proj_col]
            .to_list()
        )
        pools[p] = vals

    def next_score(p, idx):
        arr = pools[p]
        return arr[idx] if idx < len(arr) else float("-inf")

    for _ in range(flex_counts.get("FM_FLEX", 0) * teams):
        f_score = next_score("F", counts["F"])
        m_score = next_score("M", counts["M"])
        counts["F" if f_score >= m_score else "M"] += 1
    for _ in range(flex_counts.get("MD_FLEX", 0) * teams):
        m_score = next_score("M", counts["M"])
        d_score = next_score("D", counts["D"])
        counts["M" if m_score >= d_score else "D"] += 1
    return counts


def build_vor(board: pd.DataFrame, repl_ranks: dict[str, int]) -> pd.DataFrame:
    rows = []
    for pos, k in repl_ranks.items():
        sub = (
            board[board["pos"] == pos]
            .sort_values("proj_points", ascending=False)
            .reset_index(drop=True)
        )
        if len(sub) == 0:
            continue
        k = max(1, int(k))
        repl_points = sub.iloc[min(len(sub) - 1, k - 1)]["proj_points"]
        sub = sub.copy()
        sub["replacement_at_rank"] = k
        sub["replacement_points"] = repl_points
        sub["VOR"] = sub["proj_points"] - repl_points
        rows.append(sub)
    if not rows:
        return pd.DataFrame(
            columns=[
                "web_name",
                "pos",
                "team_name",
                "proj_points",
                "replacement_at_rank",
                "replacement_points",
                "VOR",
            ]
        )
    return pd.concat(rows, ignore_index=True).sort_values("VOR", ascending=False)


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


# ----- Name mapping helpers -----
def _strip_accents(s: str) -> str:
    if s is None:
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def canonical_name(s: str) -> str:
    s = _strip_accents(str(s).lower())
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def norm_team_key(s: str) -> str:
    s = _strip_accents(str(s).lower())
    return re.sub(r"[^a-z0-9]+", "", s)


# ----helpers for build_player_link() ----
def _validate_name_link(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["player_full", "web_name", "team_name", "player_fbref"]
    # allow minimal curated files that only provide player_full->player_fbref
    if not all(c in df.columns for c in needed):
        # try to coerce if possible
        if "player_full" in df.columns and "player_fbref" in df.columns:
            for col in ["web_name", "team_name"]:
                if col not in df.columns:
                    df[col] = np.nan
        else:
            raise ValueError(
                "name_link_curated.csv must contain at least columns: player_full, player_fbref"
            )
    # standardize whitespace
    df["player_full"] = (
        df["player_full"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df["player_fbref"] = (
        df["player_fbref"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    return df[["player_full", "web_name", "team_name", "player_fbref"]]


def apply_curated_overrides(
    auto_link: pd.DataFrame, curated: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """Override auto mapping with curated 'player_fbref' where provided (by player_full)."""
    cur = _validate_name_link(curated.copy())
    merged = auto_link.merge(
        cur[["player_full", "player_fbref"]],
        on="player_full",
        how="left",
        suffixes=("", "_cur"),
    )
    overrides = merged["player_fbref_cur"].notna().sum()
    merged["player_fbref"] = merged["player_fbref_cur"].combine_first(
        merged["player_fbref"]
    )
    merged = merged.drop(columns=["player_fbref_cur"])
    return merged, int(overrides)


def load_or_build_name_link(
    squads_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    gw_folder: Path | None = None,
    cutoff: float = 0.90,  # you said you want ~0.90
) -> tuple[pd.DataFrame, dict]:
    """
    Load curated name link if available, else build automatically.
    Precedence:
      1) {gw_folder}/name_link_curated.csv (if gw_folder provided)
      2) ROOT/config/name_link_curated.csv
      3) ROOT/data/outputs/name_link_curated.csv
      4) build_player_link(...)
    Returns (link_df, meta_dict).
    """
    meta = {"source": "auto", "curated_overrides": 0, "used_paths": []}

    # Always build auto baseline (so we can fill gaps even when curated is partial)
    auto = build_player_link(squads_df, hist_df, cutoff=cutoff)

    # Candidate curated locations (later ones only applied if present)
    candidates = []
    if gw_folder is not None:
        candidates.append(gw_folder / "name_link_curated.csv")
    candidates.append(ROOT / "config" / "name_link_curated.csv")
    candidates.append(OUTDIR / "name_link_curated.csv")

    link = auto.copy()
    for p in candidates:
        if p.exists():
            try:
                cur = pd.read_csv(p)
                link, n_over = apply_curated_overrides(link, cur)
                meta["curated_overrides"] += n_over
                meta["used_paths"].append(str(p))
                meta["source"] = "auto+curated"
            except Exception as e:
                print(f"[warn] Failed to apply curated overrides from {p}: {e}")

    return link, meta


def write_name_link_artifacts(outdir_gw: Path, link: pd.DataFrame):
    """Write name_link.csv and a helper name_link_missing.csv for quick curation."""
    link.to_csv(outdir_gw / "name_link.csv", index=False)
    missing = link[link["player_fbref"].isna()][
        ["player_full", "web_name", "team_name"]
    ].sort_values(["team_name", "player_full"])
    missing.to_csv(outdir_gw / "name_link_missing.csv", index=False)


def build_player_link(
    squads_df: pd.DataFrame, hist_df: pd.DataFrame, cutoff: float = 0.94
) -> pd.DataFrame:
    """
    Return mapping from FPL player_full -> FBref player (player_fbref).
    Strategy: exact canonical match, then team-scoped fuzzy fallback (cutoff configurable).
    """
    fpl = (
        squads_df[["first_name", "second_name", "web_name", "team_name"]]
        .drop_duplicates()
        .copy()
    )
    fpl["player_full"] = (
        fpl["first_name"].astype(str).str.strip()
        + " "
        + fpl["second_name"].astype(str).str.strip()
    ).str.replace(r"\s+", " ", regex=True)
    fpl["name_key"] = fpl["player_full"].map(canonical_name)
    fpl["tkey"] = fpl["team_name"].map(norm_team_key)

    fb = hist_df[["player", "team"]].drop_duplicates().copy()
    fb["name_key"] = fb["player"].map(canonical_name)
    fb["tkey"] = fb["team"].map(norm_team_key)

    # exact canonical
    exact = fpl.merge(fb[["player", "name_key"]], on="name_key", how="left").rename(
        columns={"player": "player_fbref"}
    )

    # fuzzy fallback within same team
    pending = exact[exact["player_fbref"].isna()].copy()
    if not pending.empty:
        fb_by_team = {
            t: grp["player"].unique().tolist() for t, grp in fb.groupby("tkey")
        }
        rows = []
        for _, r in pending.iterrows():
            cand_pool = fb_by_team.get(r["tkey"], fb["player"].unique().tolist())
            best_name, best_score = None, 0.0
            target = canonical_name(r["player_full"])
            for cand in cand_pool:
                score = difflib.SequenceMatcher(
                    None, target, canonical_name(cand)
                ).ratio()
                if score > best_score:
                    best_name, best_score = cand, score
            if best_score >= cutoff:  # 0.94 as default
                rows.append(
                    {"player_full": r["player_full"], "player_fbref": best_name}
                )
        fuzzy = pd.DataFrame(rows)
        if not fuzzy.empty:
            exact = exact.merge(fuzzy, on="player_full", how="left")
            exact["player_fbref"] = exact["player_fbref_x"].fillna(
                exact["player_fbref_y"]
            )
            exact = exact.drop(columns=["player_fbref_x", "player_fbref_y"])

    link = exact[
        ["player_full", "web_name", "team_name", "player_fbref"]
    ].drop_duplicates()
    return link


def ensure_team_id(squads_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure numeric FPL team id in column 'team'. Try team_name then short_name via bootstrap.
    """
    if "team" in squads_df.columns and squads_df["team"].notna().any():
        return squads_df
    boot = fpl.get_bootstrap()
    teams_df = pd.DataFrame(boot["teams"])[["id", "name", "short_name"]].copy()
    teams_df = teams_df.rename(
        columns={
            "id": "team",
            "name": "team_name_boot",
            "short_name": "short_name_boot",
        }
    )
    out = squads_df.copy()
    if "team_name" in out.columns:
        out = out.merge(
            teams_df[["team", "team_name_boot"]],
            left_on="team_name",
            right_on="team_name_boot",
            how="left",
        )
        out = out.drop(columns=["team_name_boot"])
    if ("team" not in out.columns) or (out["team"].isna().all()):
        key_left = (
            "team_short_name"
            if "team_short_name" in out.columns
            else ("short_name" if "short_name" in out.columns else None)
        )
        if key_left is not None:
            out = out.merge(
                teams_df[["team", "short_name_boot"]],
                left_on=key_left,
                right_on="short_name_boot",
                how="left",
            )
            out = out.drop(columns=["short_name_boot"], errors="ignore")
    return out


# ------SNAKE planner helpers------
def read_snake_state(folder: Path) -> pd.DataFrame:
    """
    Load previously picked players and their assigned slots.
    Expected columns: web_name, slot_used, my_round (optional).
    Creates an empty template if not present.
    """
    path = folder / "snake_state.csv"
    if not path.exists():
        pd.DataFrame({"web_name": [], "slot_used": [], "my_round": []}).to_csv(
            path, index=False
        )
        print(f"[info] Created empty snake state file: {path}")
        return pd.DataFrame(columns=["web_name", "slot_used", "my_round"])

    df = pd.read_csv(path)
    for c in ["web_name", "slot_used", "my_round"]:
        if c not in df.columns:
            df[c] = np.nan
    df["web_name"] = df["web_name"].astype(str).str.strip()
    df["slot_used"] = df["slot_used"].astype(str).str.strip()
    df["my_round"] = pd.to_numeric(df["my_round"], errors="coerce")
    return df


def write_snake_state_next(
    folder: Path, prev: pd.DataFrame, new_picks: pd.DataFrame
) -> Path:
    """
    Write a proposed next-state file that includes previous picks + newly suggested picks.
    """
    keep_cols = ["web_name", "slot_used", "my_round"]
    prev = prev.copy()[keep_cols] if len(prev) else pd.DataFrame(columns=keep_cols)
    nxt = (
        new_picks.copy()[keep_cols]
        if len(new_picks)
        else pd.DataFrame(columns=keep_cols)
    )

    out = pd.concat([prev, nxt], ignore_index=True).drop_duplicates(
        subset=["web_name"], keep="last"
    )
    path = folder / "snake_state_next.csv"
    out.to_csv(path, index=False)
    return path


def subtract_used_slots(req: dict[str, int], used_slots: pd.Series) -> dict[str, int]:
    """
    Decrease roster requirements by counts in 'used_slots'.
    """
    r = dict(req)
    for slot, n in used_slots.value_counts().items():
        if slot in r:
            r[slot] = max(0, r[slot] - int(n))
    return r


# ---------------------------
# 1) Build season-weighted per90 points per player, per position
# ---------------------------
if "minutes" not in hist.columns:
    hist["minutes"] = coalesce_minutes_rowwise(hist)
per90_cols = [c for c in hist.columns if c.endswith("_p90")]

# collapse to player-season average per90
per_season = hist.groupby(["player", "season"])[per90_cols].mean().reset_index()

# season-weighted minutes baseline (vectorized)
mins_per_season = hist.groupby(["player", "season"])["minutes"].mean().reset_index()
mins_tmp = mins_per_season.copy()
mins_tmp["_w"] = mins_tmp["season"].astype(str).map(weights).fillna(1.0)
den_m = mins_tmp.groupby("player")["_w"].sum()
num_m = (mins_tmp["minutes"] * mins_tmp["_w"]).groupby(mins_tmp["player"]).sum()
min_weighted = (num_m / den_m).rename("hist_minutes_w").reset_index()


# per-season per90 points using Sleeper scoring + stat_map
def score_row_per90(row: pd.Series, position: str) -> float:
    base = {c[:-4]: row[c] for c in per90_cols if pd.notna(row[c])}
    return points_from_row(pd.Series(base), position, scoring, stat_map)


for pos in ["F", "M", "D", "GK"]:
    per_season[f"points90_{pos}"] = per_season.apply(
        lambda r: score_row_per90(r, pos), axis=1
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


# ---------------------------
# 2) Name mapping + team ids
# ---------------------------
# Build player_full in squads and name link to FBref
squads["player_full"] = (
    squads["first_name"].astype(str).str.strip()
    + " "
    + squads["second_name"].astype(str).str.strip()
).str.replace(r"\s+", " ", regex=True)

name_link, name_meta = load_or_build_name_link(
    squads, hist, gw_folder=None, cutoff=0.90
)
squads = squads.merge(
    name_link[["player_full", "player_fbref"]], on="player_full", how="left"
)

# Ensure numeric team id exists
squads = ensure_team_id(squads)
missing_team = squads["team"].isna().sum() if "team" in squads.columns else len(squads)
if missing_team:
    print(
        f"[warn] {missing_team} squad rows have no numeric team id; applying neutral team multipliers for those."
    )


# ---------------------------
# 3) Team power model (attack + defense)
# ---------------------------
# Attack features (team_season_stats)
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
ts = team_season_stats.copy()
ts = ts.dropna(subset=["team", "season"])
att_cols = [c for c in ATT_FEATURES_CAND if c in ts.columns]
for c in att_cols:
    ts[c] = pd.to_numeric(ts[c], errors="coerce")


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


ts_att_idx = _zscore_per_season(
    ts[["team", "season"] + att_cols], att_cols, invert=False, label="attack_idx_season"
)
# Weighted attack history
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

# --- Reconcile team names (aliases + fuzzy fallback) ---
# team_names_fpl = squads[["team", "team_name"]].drop_duplicates()


# def _keyize(s: pd.Series) -> pd.Series:
#     return s.str.upper().str.replace(r"[^A-Z0-9]+", "", regex=True)


# team_attack_hist["_name_for_key"] = team_attack_hist["team"].replace(ALIAS_FBREF_TO_FPL)
# team_defense_hist["_name_for_key"] = team_defense_hist["team"].replace(
#     ALIAS_FBREF_TO_FPL
# )
# team_attack_hist["_key"] = _keyize(team_attack_hist["_name_for_key"])
# team_defense_hist["_key"] = _keyize(team_defense_hist["_name_for_key"])
# team_names_fpl["_key"] = _keyize(team_names_fpl["team_name"])

# team_power = (
#     team_names_fpl.merge(
#         team_attack_hist[["_key", "attack_idx_hist"]], on="_key", how="left"
#     )
#     .merge(team_defense_hist[["_key", "defense_idx_hist"]], on="_key", how="left")
#     .drop(columns=["_key"])
#     .drop_duplicates(subset=["team", "team_name"])
# )
# --- Reconcile team names (canonicalize to FPL first, then join; fuzzy as backup) ---

# Canonicalize FBref names to FPL style up front
fb_att = team_attack_hist.copy()
fb_def = team_defense_hist.copy()
fb_att["team_fpl_like"] = fb_att["team"].map(ALIAS_FBREF_TO_FPL).fillna(fb_att["team"])
fb_def["team_fpl_like"] = fb_def["team"].map(ALIAS_FBREF_TO_FPL).fillna(fb_def["team"])

# Unique current FPL teams
team_names_fpl = squads[["team", "team_name"]].drop_duplicates().copy()

# 1) Direct join on canonical strings (case-insensitive safe via lower)
fb_att["_join"] = fb_att["team_fpl_like"].str.lower()
fb_def["_join"] = fb_def["team_fpl_like"].str.lower()
team_names_fpl["_join"] = team_names_fpl["team_name"].str.lower()

team_power = (
    team_names_fpl.merge(fb_att[["_join", "attack_idx_hist"]], on="_join", how="left")
    .merge(fb_def[["_join", "defense_idx_hist"]], on="_join", how="left")
    .drop(columns=["_join"])
    .drop_duplicates(subset=["team", "team_name"])
)

# 2) Fuzzy fallback for any still-missing rows (compare FPL name vs fb_att/def team_fpl_like)
need_attack = team_power["attack_idx_hist"].isna()
need_def = team_power["defense_idx_hist"].isna()

if need_attack.any() or need_def.any():
    fb_att_cands = (
        fb_att[["team_fpl_like", "attack_idx_hist"]].dropna().drop_duplicates()
    )
    fb_def_cands = (
        fb_def[["team_fpl_like", "defense_idx_hist"]].dropna().drop_duplicates()
    )
    fuzzy_hits = []

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

    tp = team_power.copy()
    for idx, row in tp.iterrows():
        tname = str(row["team_name"])
        if need_attack.iloc[idx]:
            val, cand, sc = _fuzzy_pick(
                tname, fb_att_cands, "attack_idx_hist", cutoff=0.78
            )
            if pd.notna(val):
                tp.at[idx, "attack_idx_hist"] = val
                fuzzy_hits.append(
                    f"[fuzzy-attack] '{tname}' -> '{cand}' (score={sc:.2f})"
                )
        if need_def.iloc[idx]:
            val, cand, sc = _fuzzy_pick(
                tname, fb_def_cands, "defense_idx_hist", cutoff=0.78
            )
            if pd.notna(val):
                tp.at[idx, "defense_idx_hist"] = val
                fuzzy_hits.append(
                    f"[fuzzy-defense] '{tname}' -> '{cand}' (score={sc:.2f})"
                )
    team_power = tp

    if fuzzy_hits:
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
ROSTER_TOP_N_ATT = 6
ROSTER_TOP_N_DEF = 6
roster_att = (
    roster_tbl[roster_tbl["pos"].isin(["F", "M"])]
    .dropna(subset=["team_name", "points90w_sel"])
    .sort_values(["team_name", "points90w_sel"], ascending=[True, False])
    .groupby("team_name")
    .head(ROSTER_TOP_N_ATT)
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
    .head(ROSTER_TOP_N_DEF)
    .groupby("team_name")["points90w_def_component"]
    .sum()
    .rename("roster_def")
    .reset_index()
)

team_power = team_power.merge(roster_att, on="team_name", how="left").merge(
    roster_def, on="team_name", how="left"
)


def _norm_mean1(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    if mu == 0 or np.isnan(mu):
        return s.fillna(0.0).apply(lambda x: 1.0)
    return (s / mu).replace([np.inf, -np.inf], 1.0).fillna(1.0)


ALPHA_HIST = 0.6
ALPHA_ROS = 0.4
team_power["attack_hist_n"] = _norm_mean1(team_power["attack_idx_hist"])
team_power["roster_att_n"] = _norm_mean1(team_power["roster_att"])
team_power["defense_hist_n"] = _norm_mean1(team_power["defense_idx_hist"])
team_power["roster_def_n"] = _norm_mean1(team_power["roster_def"])


def _clamp_norm(s, lo=0.85, hi=1.15):
    x = s.clip(lo, hi)
    return _norm_mean1(x)


team_power["attack_power"] = _clamp_norm(
    ALPHA_HIST * team_power["attack_hist_n"] + ALPHA_ROS * team_power["roster_att_n"]
)
team_power["defense_power"] = _clamp_norm(
    ALPHA_HIST * team_power["defense_hist_n"] + ALPHA_ROS * team_power["roster_def_n"]
)

dbg_missing = team_power[
    team_power["attack_idx_hist"].isna() | team_power["defense_idx_hist"].isna()
]
if not dbg_missing.empty:
    print(
        "[debug] still missing history for:",
        ", ".join(dbg_missing["team_name"].tolist()),
    )
# ---------------------------
# 4) Merge weighted with squads, GW setup & team power mapping
# ---------------------------
squads["pos"] = squads["pos"].map(POS_FPL_TO_GENERAL).fillna(squads["pos"])
players = squads.merge(weighted, left_on="player_fbref", right_on="player", how="left")

# choose the GW (event)
upcoming = fixtures[~fixtures["finished"]].sort_values(["event", "kickoff_time"])
if args.event is not None:
    gw = int(args.event)
else:
    gw = int(upcoming["event"].min()) if len(upcoming) else int(fixtures["event"].max())

gw_tag = f"gw{gw}"
OUTDIR_GW = OUTDIR / gw_tag
OUTDIR_GW.mkdir(parents=True, exist_ok=True)

# also write the name map so you can audit/override
name_link_gw, name_meta_gw = load_or_build_name_link(
    squads, hist, gw_folder=OUTDIR_GW, cutoff=0.90
)

# If GW curated added more overrides, update mapping on 'squads' too
if name_meta_gw["curated_overrides"] > name_meta.get("curated_overrides", 0):
    name_link = name_link_gw
    squads = squads.drop(columns=["player_fbref"], errors="ignore").merge(
        name_link[["player_full", "player_fbref"]], on="player_full", how="left"
    )

# Write link + missing helpers for this run
write_name_link_artifacts(OUTDIR_GW, name_link)

# decide which chance field to use
next_unplayed = int(upcoming["event"].min()) if len(upcoming) else None
chance_field = (
    "chance_of_playing_this_round"
    if (args.event is None or (next_unplayed is not None and gw == next_unplayed))
    else "chance_of_playing_next_round"
)

# GW difficulty map
gw_fix = fixtures[fixtures["event"] == gw].copy()
team_diff = {}
for _, r in gw_fix.iterrows():
    team_diff[int(r["team_h"])] = int(r["team_h_difficulty"])
    team_diff[int(r["team_a"])] = int(r["team_a_difficulty"])

# ensure team id if still missing (rare)
if "team" not in players.columns or players["team"].isna().all():
    boot = fpl.get_bootstrap()
    teams_df = pd.DataFrame(boot["teams"])[["id", "name", "short_name"]]
    players = players.merge(
        teams_df.rename(columns={"id": "team"}),
        left_on="team_name",
        right_on="name",
        how="left",
    )

players["gw_difficulty"] = players["team"].map(team_diff)

# Map team powers, keep a single clean team_name
players = players.merge(
    team_power[["team", "team_name", "attack_power", "defense_power"]],
    on="team",
    how="left",
    suffixes=("", "_tp"),
)
if "team_name_tp" in players.columns:
    players["team_name"] = players["team_name"].fillna(players["team_name_tp"])
    players.drop(columns=["team_name_tp"], inplace=True)
players["attack_power"] = players["attack_power"].fillna(1.0)
players["defense_power"] = players["defense_power"].fillna(1.0)

# write team power table
team_power_rank = team_power.copy()
team_power_rank["attack_rank"] = (
    team_power_rank["attack_power"].rank(ascending=False, method="min").astype(int)
)
team_power_rank["defense_rank"] = (
    team_power_rank["defense_power"].rank(ascending=False, method="min").astype(int)
)
tp_out = team_power_rank.sort_values("attack_power", ascending=False)[
    [
        "team",
        "team_name",
        "attack_idx_hist",
        "defense_idx_hist",
        "roster_att",
        "roster_def",
        "attack_hist_n",
        "roster_att_n",
        "defense_hist_n",
        "roster_def_n",
        "attack_power",
        "defense_power",
        "attack_rank",
        "defense_rank",
    ]
].copy()

# purely cosmetic for CSV readability: show 0.0 when raw history is absent (model already treats NaN as neutral)
tp_out["attack_idx_hist"] = pd.to_numeric(
    tp_out["attack_idx_hist"], errors="coerce"
).fillna(0.0)
tp_out["defense_idx_hist"] = pd.to_numeric(
    tp_out["defense_idx_hist"], errors="coerce"
).fillna(0.0)

tp_out.to_csv(OUTDIR_GW / "team_power.csv", index=False)
print(f"Wrote {OUTDIR_GW / 'team_power.csv'}")

# Normalize positions to F/M/D/GK
players["pos"] = players["pos"].apply(to_canon_pos)


# expected minutes: hist baseline × status × chance
def exp_minutes(row):
    base = row.get("hist_minutes_w", np.nan)
    if pd.isna(base):
        base = POS_PRIOR.get(row.get("pos"), 75.0)
    status = str(row.get("status", "")).lower()
    sm = STATUS_MULT.get(status, 0.80 if status == "" else 0.0)
    cp = row.get(chance_field, np.nan)
    cm = float(cp) / 100.0 if pd.notna(cp) else 1.0
    est = base * sm * cm
    return float(max(0.0, min(90.0, est)))


players["exp_minutes"] = players.apply(exp_minutes, axis=1)


# ---------------------------
# 5) Per-GW projection
# ---------------------------
players["points90w_sel"] = np.where(
    players["pos"].eq("F"),
    players["points90w_F"],
    np.where(
        players["pos"].eq("M"),
        players["points90w_M"],
        np.where(
            players["pos"].eq("D"),
            players["points90w_D"],
            np.where(
                players["pos"].eq("GK"), players["points90w_GK"], players["points90w_M"]
            ),
        ),
    ),
)


def proj_points_row(r):
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


players["proj_points"] = players.apply(proj_points_row, axis=1)


# ---------------------------
# 5b) Multi-week projection for bench planning
# ---------------------------
def event_team_difficulty_map(event: int) -> dict[int, int]:
    fx = fixtures[fixtures["event"] == event]
    td = {}
    for _, r in fx.iterrows():
        td[int(r["team_h"])] = int(r["team_h_difficulty"])
        td[int(r["team_a"])] = int(r["team_a_difficulty"])
    return td


def proj_for_event(df: pd.DataFrame, event: int) -> pd.Series:
    td = event_team_difficulty_map(event)
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
        [
            attack_mult(d) if p in ("F", "M") else defend_mult(d)
            for d, p in zip(diffs, pos_vals)
        ]
    )
    mult_team = np.where(
        np.isin(pos_vals, ["F", "M"]),
        df["attack_power"].fillna(1.0).to_numpy(),
        df["defense_power"].fillna(1.0).to_numpy(),
    )
    return pd.Series(pts90 * (mins / 90.0) * mult_pos * mult_team, index=df.index)


# initialize nextN to current
players["proj_nextN"] = players["proj_points"].astype(float)
future_events = [e for e in range(gw, gw + args.weeks_for_bench)]
acc = pd.Series(0.0, index=players.index, dtype="float64")
denN = 0
for e in future_events:
    if (fixtures["event"] == e).any():
        acc = acc.add(proj_for_event(players, e), fill_value=0.0)
        denN += 1
if denN > 0:
    players["proj_nextN"] = acc / denN


# ---------------------------
# 6) Taken list per-GW & filter
# ---------------------------
def read_taken_csv(folder: Path) -> list[str]:
    path = folder / "taken.csv"
    if not path.exists():
        pd.DataFrame({"web_name": []}).to_csv(path, index=False)
        print(f"[info] Created empty taken file: {path}")
        return []
    df = pd.read_csv(path)
    if "web_name" not in df.columns and len(df.columns) > 0:
        df = df.rename(columns={df.columns[0]: "web_name"})
    return [str(x).strip() for x in df["web_name"].dropna() if str(x).strip()]


def write_taken_next(folder: Path, current: list[str], my_new_picks: list[str]) -> Path:
    seen, merged = set(), []
    for n in current + my_new_picks:
        k = n.lower().strip()
        if k and k not in seen:
            merged.append(n.strip())
            seen.add(k)
    path = folder / "taken_next.csv"
    pd.DataFrame({"web_name": merged}).to_csv(path, index=False)
    return path


taken_names = read_taken_csv(OUTDIR_GW)
if taken_names:
    players = players[
        ~players["web_name"].str.lower().isin([n.lower() for n in taken_names])
    ]


# ---------------------------
# 7) Outputs: top10 per GW (overall + by position)
# ---------------------------
# ensure clean team_name exists
if "team_name" not in players.columns:
    for c in ("team_name_x", "team_name_y", "team_name_tp"):
        if c in players.columns:
            players["team_name"] = players[c]
            break
    if "team_name" not in players.columns:
        players["team_name"] = np.nan

top_overall = players.sort_values("proj_points", ascending=False).head(args.topn)
top_f = players[players["pos"] == "F"].nlargest(args.topn, "proj_points")
top_m = players[players["pos"] == "M"].nlargest(args.topn, "proj_points")
top_d = players[players["pos"] == "D"].nlargest(args.topn, "proj_points")
top_gk = players[players["pos"] == "GK"].nlargest(args.topn, "proj_points")

top10 = pd.concat(
    [
        top_overall.assign(bucket=f"{gw_tag} Overall"),
        top_f.assign(bucket=f"{gw_tag} Forwards"),
        top_m.assign(bucket=f"{gw_tag} Midfielders"),
        top_d.assign(bucket=f"{gw_tag} Defenders"),
        top_gk.assign(bucket=f"{gw_tag} Goalkeepers"),
    ],
    ignore_index=True,
)[["bucket", "web_name", "team_name", "pos", "proj_points"]]

top10.to_csv(OUTDIR_GW / "top10.csv", index=False)


# ---------------------------
# 8) Draft boards (non-flex and flex-aware) with VOR
# ---------------------------
board_base = (
    players[players["pos"].isin(["F", "M", "D", "GK"])][
        ["web_name", "pos", "team_name", "proj_points", "proj_nextN", "team"]
    ]
    .dropna(subset=["proj_points"])
    .copy()
)

# Non-flex
repl_nonflex = replacement_ranks_nonflex(roster_positions, teams_in_league)
draft_board_nonflex = build_vor(board_base, repl_nonflex)
draft_board_nonflex.to_csv(OUTDIR_GW / "draft_board_nonflex.csv", index=False)

# Flex-aware
repl_flex = effective_replacement_ranks_with_flex(
    board_base,
    pos_col="pos",
    proj_col="proj_points",
    roster_positions=roster_positions,
    teams=teams_in_league,
)
draft_board_flex = build_vor(board_base, repl_flex)
draft_board_flex.to_csv(OUTDIR_GW / "draft_board_flexaware.csv", index=False)

print(f"Wrote {OUTDIR_GW / 'top10.csv'}")
print(f"Wrote {OUTDIR_GW / 'draft_board_nonflex.csv'}")
print(f"Wrote {OUTDIR_GW / 'draft_board_flexaware.csv'}")


# ---------------------------
# 9) Snake draft planner (stateful)
# ---------------------------
def snake_round_order(teams: int, round_num: int) -> list[int]:
    base = list(range(1, teams + 1))
    return base if (round_num % 2 == 1) else list(reversed(base))


def roster_requirements(positions: list[str]) -> dict[str, int]:
    req = {"F": 0, "M": 0, "D": 0, "GK": 0, "FM_FLEX": 0, "MD_FLEX": 0, "BN": 0}
    for p in positions:
        req[p] = req.get(p, 0) + 1
    return req


def starters_remaining(req: dict[str, int]) -> int:
    return (
        req.get("F", 0)
        + req.get("M", 0)
        + req.get("D", 0)
        + req.get("GK", 0)
        + req.get("FM_FLEX", 0)
        + req.get("MD_FLEX", 0)
    )


def can_assign(pos: str, req: dict[str, int]) -> tuple[bool, str]:
    # Try direct positional slot
    if req.get(pos, 0) > 0:
        return True, pos
    # FLEX logic
    if pos in FLEX_MAP["FM_FLEX"] and req.get("FM_FLEX", 0) > 0:
        return True, "FM_FLEX"
    if pos in FLEX_MAP["MD_FLEX"] and req.get("MD_FLEX", 0) > 0:
        return True, "MD_FLEX"
    # Bench only after all starters filled
    if req.get("BN", 0) > 0 and starters_remaining(req) == 0:
        return True, "BN"
    return False, ""


def assign_slot(req: dict[str, int], slot: str):
    req[slot] = max(0, req.get(slot, 0) - 1)


draft_order = sleeper_draft.get("draft_order", {})
my_team_id = (
    args.my_team_id
    or sleeper_setup.get("my_team_id")
    or next(iter(draft_order.keys()), None)
)
if my_team_id in draft_order:
    my_slot = int(draft_order[my_team_id])
else:
    my_slot = teams_in_league  # fallback: last pick

num_rounds = len(roster_positions)

# Base board
board_base = (
    players[players["pos"].isin(["F", "M", "D", "GK"])][
        ["web_name", "pos", "team_name", "proj_points", "proj_nextN", "team"]
    ]
    .dropna(subset=["proj_points"])
    .copy()
)

# Replacement ranks
repl_flex = effective_replacement_ranks_with_flex(
    board_base,
    pos_col="pos",
    proj_col="proj_points",
    roster_positions=roster_positions,
    teams=teams_in_league,
)
draft_board_flex = build_vor(board_base, repl_flex)

# Merge nextN for convenience
if "proj_nextN" in board_base.columns:
    name_nextN = board_base[["web_name", "proj_nextN"]].drop_duplicates(
        subset=["web_name"]
    )
    avail = draft_board_flex.merge(name_nextN, on="web_name", how="left")
else:
    avail = draft_board_flex.copy()
if "proj_nextN" not in avail.columns:
    avail["proj_nextN"] = avail["proj_points"]
else:
    avail["proj_nextN"] = avail["proj_nextN"].fillna(avail["proj_points"])
avail = (
    avail[
        [
            "web_name",
            "pos",
            "team_name",
            "proj_points",
            "replacement_at_rank",
            "replacement_points",
            "VOR",
            "proj_nextN",
        ]
    ]
    .drop_duplicates(subset=["web_name"])
    .reset_index(drop=True)
)

# Remove globally taken names
taken_names = read_taken_csv(OUTDIR_GW)
if taken_names:
    avail = avail[
        ~avail["web_name"].str.lower().isin([n.lower() for n in taken_names])
    ].reset_index(drop=True)

# Load my current state (already picked players + their slots)
my_state = read_snake_state(OUTDIR_GW)

# Also remove my previously picked players from the pool
if len(my_state):
    avail = avail[
        ~avail["web_name"].str.lower().isin(my_state["web_name"].str.lower())
    ].reset_index(drop=True)

# Determine remaining requirements after applying my picked slots
req = roster_requirements(roster_positions)
if len(my_state) and my_state["slot_used"].notna().any():
    req = subtract_used_slots(req, my_state["slot_used"].dropna())

# If some of my prior picks are missing slot_used, try to infer slots now to keep req consistent
missing_slot_mask = my_state["slot_used"].isna()
if missing_slot_mask.any():
    # Build a quick lookup from avail+players (to get positions)
    pos_lookup = players.set_index("web_name")["pos"].to_dict()
    for i, row in my_state[missing_slot_mask].iterrows():
        wn = row["web_name"]
        pos = pos_lookup.get(wn)
        if pos:
            ok, which = can_assign(pos, req)
            if ok:
                assign_slot(req, which)
                my_state.at[i, "slot_used"] = which

# Figure out where to start (next round)
picked_rounds = my_state["my_round"].dropna()
if len(picked_rounds):
    start_round = int(picked_rounds.max()) + 1
else:
    start_round = len(my_state) + 1  # fallback if my_rounds not provided

# Build plan from start_round → num_rounds
my_picks_new = []


def pop_best_available(av: pd.DataFrame) -> pd.Series | None:
    return av.iloc[0] if len(av) else None


def remove_player(av: pd.DataFrame, name: str) -> pd.DataFrame:
    return av[av["web_name"].str.lower() != name.lower()].reset_index(drop=True)


for rnd in range(start_round, num_rounds + 1):
    order = snake_round_order(teams_in_league, rnd)
    for slot in order:
        if slot == my_slot:
            pick = None
            av = avail.copy()
            # Starters first, then bench
            if starters_remaining(req) > 0:
                for _, row in av.sort_values(
                    ["VOR", "proj_points"], ascending=False
                ).iterrows():
                    ok, which = can_assign(row["pos"], req)
                    if ok:
                        pick = row.copy()
                        pick["slot_used"] = which
                        assign_slot(req, which)
                        break
            else:
                for _, row in av.sort_values(
                    ["proj_nextN", "VOR"], ascending=False
                ).iterrows():
                    ok, which = can_assign(row["pos"], req)
                    if ok:
                        pick = row.copy()
                        pick["slot_used"] = which
                        assign_slot(req, which)
                        break
            if pick is None:
                row = pop_best_available(av)
                if row is None:
                    continue
                pick = row.copy()
                pick["slot_used"] = "BN" if starters_remaining(req) == 0 else "ANY"
                if pick["slot_used"] == "BN":
                    assign_slot(req, "BN")

            pick["my_round"] = rnd
            overall = (rnd - 1) * teams_in_league + (order.index(slot) + 1)
            pick["overall_pick"] = overall
            my_picks_new.append(pick)
            avail = remove_player(avail, pick["web_name"])
        else:
            row = pop_best_available(
                avail.sort_values(["VOR", "proj_points"], ascending=False)
            )
            if row is not None:
                avail = remove_player(avail, row["web_name"])


# Combine previously picked + new plan
def overall_from_round(rnd: float, my_slot: int, teams: int) -> float:
    if pd.isna(rnd):
        return np.nan
    rnd = int(rnd)
    return (rnd - 1) * teams + (my_slot if (rnd % 2 == 1) else (teams - my_slot + 1))


already = pd.DataFrame(
    columns=[
        "web_name",
        "pos",
        "team_name",
        "proj_points",
        "replacement_at_rank",
        "replacement_points",
        "VOR",
        "proj_nextN",
        "slot_used",
        "my_round",
        "overall_pick",
    ]
)
if len(my_state):
    # enrich my_state rows with projections so the view is informative
    enrich = players[["web_name", "pos", "team_name"]].drop_duplicates()
    already = my_state.merge(enrich, on="web_name", how="left")
    already["proj_points"] = np.nan
    already["replacement_at_rank"] = np.nan
    already["replacement_points"] = np.nan
    already["VOR"] = np.nan
    already["proj_nextN"] = np.nan
    already["overall_pick"] = already.apply(
        lambda r: overall_from_round(r["my_round"], my_slot, teams_in_league), axis=1
    )

new_df = (
    pd.DataFrame(my_picks_new)
    if len(my_picks_new)
    else pd.DataFrame(columns=already.columns)
)

plan = pd.concat(
    [already.assign(already_picked=True), new_df.assign(already_picked=False)],
    ignore_index=True,
)
plan = plan.sort_values(
    ["already_picked", "my_round", "overall_pick"], ascending=[False, True, True]
).reset_index(drop=True)

# Save plan + next state + taken_next
plan_out = OUTDIR_GW / "snake_plan.csv"
plan.to_csv(plan_out, index=False)
print(f"Wrote {plan_out}")

# Next state proposal: prior state + newly suggested picks
next_state_path = write_snake_state_next(
    OUTDIR_GW, my_state, plan[plan["already_picked"] == False]
)
print(
    f"Wrote {next_state_path}  (rename to {OUTDIR_GW/'snake_state.csv'} for the next run)"
)

# Update taken_next so you can use it for subsequent runs too
taken_all = set([n.lower() for n in read_taken_csv(OUTDIR_GW)]) | set(
    plan["web_name"].str.lower()
)
pd.DataFrame(
    {"web_name": sorted({w for w in plan["web_name"] if isinstance(w, str)})}
).to_csv(OUTDIR_GW / "taken_next.csv", index=False)
print(f"Wrote {OUTDIR_GW / 'taken_next.csv'}  (rename to taken.csv when appropriate)")
print(f"Wrote {OUTDIR_GW / 'top10.csv'}")
print(f"Wrote {OUTDIR_GW / 'draft_board_nonflex.csv'}")
print(f"Wrote {OUTDIR_GW / 'draft_board_flexaware.csv'}")

# ---- tiny mapping log ----
unmatched = name_link[name_link["player_fbref"].isna()]
matched_n = len(name_link) - len(unmatched)
total_n = len(name_link)
print(
    f"[info] Name mapping: {matched_n}/{total_n} matched; {len(unmatched)} unmatched."
)
print(
    f"[info] Curated overrides applied: {name_meta.get('curated_overrides', 0)}"
    + (
        f" (+{name_meta_gw.get('curated_overrides', 0) - name_meta.get('curated_overrides', 0)} from GW file)"
        if "name_meta_gw" in globals()
        else ""
    )
)

if name_meta.get("used_paths"):
    print("[info] Curated sources:", " | ".join(name_meta["used_paths"]))
if "name_meta_gw" in globals() and name_meta_gw.get("used_paths"):
    print("[info] GW curated sources:", " | ".join(name_meta_gw["used_paths"]))

if not unmatched.empty:
    print(
        "[info] Unmatched examples:",
        ", ".join(unmatched["player_full"].head(8).tolist()),
    )
    print(
        f"[info] See {OUTDIR_GW/'name_link_missing.csv'} for a full list you can curate."
    )
