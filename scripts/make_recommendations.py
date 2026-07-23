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

from pdh.scoring import load_sleeper_scoring, load_stat_map
from pdh import fpl  # used to fetch FPL team ids if missing
from pdh.projections import (
    build_weighted_points90,
    build_team_power,
    project_players,
    fpl_xg_table,
)
from pdh import understat as pdh_understat


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
OUTDIR = ROOT / "data" / "season_2526" / "game_weeks"
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
    ROOT / "data" / "historical" / "players_historical_20251025.csv"
)  # from build_historical.py

team_season_stats = pd.read_csv(
    ROOT / "data" / "historical" / "teams_season_stats.csv"
)  # from build_historical.py

squads = pd.read_csv(
    ROOT / "data" / "season_2526" / "current_squads.csv"
)  # from update_current.py
fixtures = pd.read_csv(
    ROOT / "data" / "season_2526" / "fixtures.csv"
)  # from update_current.py


# ---------------------------
# Helpers
# ---------------------------
# Position/team constants, per-90 scoring, team power model, and gameweek
# projection logic now live in src/pdh/projections.py (Plan B extraction).
FLEX_MAP = {"FM_FLEX": {"F", "M"}, "MD_FLEX": {"M", "D"}}


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
# 1) Name mapping + team ids
# ---------------------------
# Build player_full in squads and name link to FBref. Done before the
# weighted-points build below, since the FPL xG/xA supplement needs
# squads' player_fbref column already attached.
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
# 2) Build season-weighted per90 points per player, per position
# ---------------------------
# See src/pdh/projections.py for the actual per-90 scoring / team-power /
# gameweek-projection engine (Plan B extraction from this script).
try:
    understat_df = pdh_understat.player_season_stats(cfg_seasons["historical_seasons"])
    if understat_df.empty:
        print("[warn] Understat returned no rows (season not started, or soft-blocked); proceeding without xG/xA blending.")
        understat_df = None
except Exception as e:
    print(f"[warn] Understat fetch failed ({e}); proceeding without xG/xA blending.")
    understat_df = None

# FPL's own expected_goals_per_90/expected_assists_per_90 (bootstrap-static)
# supplements Understat for the current season - fills gaps where Understat
# is blocked or a name doesn't link, without needing its own scraper.
fpl_xg_df = fpl_xg_table(squads, season_end_year=cfg_seasons["current_season_end_year"])
if fpl_xg_df.empty:
    fpl_xg_df = None

weighted = build_weighted_points90(
    hist, weights, scoring, stat_map, understat_df=understat_df, fpl_xg_df=fpl_xg_df
)


# ---------------------------
# 3) Team power model (attack + defense)
# ---------------------------
team_power = build_team_power(team_season_stats, hist, squads, weighted, weights)

# ---------------------------
# 4) GW setup (event, output dir, name mapping refresh, chance field)
# ---------------------------
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

# ---------------------------
# 5) Per-GW + multi-week (bench) projections
# ---------------------------
players = project_players(
    squads,
    weighted,
    team_power,
    fixtures,
    gw,
    chance_field,
    weeks_for_bench=args.weeks_for_bench,
)


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
# 9) Top roster planner (stateful, no simulation)
# ---------------------------
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
    if req.get(pos, 0) > 0:
        return True, pos
    if pos in FLEX_MAP["FM_FLEX"] and req.get("FM_FLEX", 0) > 0:
        return True, "FM_FLEX"
    if pos in FLEX_MAP["MD_FLEX"] and req.get("MD_FLEX", 0) > 0:
        return True, "MD_FLEX"
    if req.get("BN", 0) > 0 and starters_remaining(req) == 0:
        return True, "BN"
    return False, ""


def assign_slot(req: dict[str, int], slot: str):
    req[slot] = max(0, req.get(slot, 0) - 1)


# Build the candidate pool from the flex-aware board.
# We do NOT simulate other teams' picks anymore. We only remove players that are
# already unavailable via taken.csv and players already on my snake_state.csv.
board_base = (
    players[players["pos"].isin(["F", "M", "D", "GK"])][
        ["web_name", "pos", "team_name", "proj_points", "proj_nextN", "team"]
    ]
    .dropna(subset=["proj_points"])
    .copy()
)

repl_flex = effective_replacement_ranks_with_flex(
    board_base,
    pos_col="pos",
    proj_col="proj_points",
    roster_positions=roster_positions,
    teams=teams_in_league,
)
draft_board_flex = build_vor(board_base, repl_flex)

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

# Remove globally unavailable players from taken.csv
players_taken = read_taken_csv(OUTDIR_GW)
if players_taken:
    avail = avail[
        ~avail["web_name"].str.lower().isin([n.lower() for n in players_taken])
    ].reset_index(drop=True)

# Load my current state (already picked players and the slot they used)
my_state = read_snake_state(OUTDIR_GW)

# Remove my already-picked players from the candidate pool
if len(my_state):
    avail = avail[
        ~avail["web_name"].str.lower().isin(my_state["web_name"].str.lower())
    ].reset_index(drop=True)

# Remaining roster requirements after applying my already picked players
req = roster_requirements(roster_positions)
if len(my_state) and my_state["slot_used"].notna().any():
    req = subtract_used_slots(req, my_state["slot_used"].dropna())

# If some prior picks are missing slot_used, infer them conservatively
missing_slot_mask = my_state["slot_used"].isna()
if missing_slot_mask.any():
    pos_lookup = players.set_index("web_name")["pos"].to_dict()
    for i, row in my_state[missing_slot_mask].iterrows():
        wn = row["web_name"]
        pos = pos_lookup.get(wn)
        if pos:
            ok, which = can_assign(pos, req)
            if ok:
                assign_slot(req, which)
                my_state.at[i, "slot_used"] = which

# Build the best remaining roster greedily:
# - while starters remain: prioritize VOR, then proj_points
# - after starters are full (bench only): prioritize proj_nextN, then VOR
remaining_picks = []
while any(v > 0 for v in req.values()):
    if avail.empty:
        break

    pick = None
    if starters_remaining(req) > 0:
        for _, row in avail.sort_values(
            ["VOR", "proj_points"], ascending=False
        ).iterrows():
            ok, which = can_assign(row["pos"], req)
            if ok:
                pick = row.copy()
                pick["slot_used"] = which
                assign_slot(req, which)
                break
    else:
        for _, row in avail.sort_values(
            ["proj_nextN", "VOR"], ascending=False
        ).iterrows():
            ok, which = can_assign(row["pos"], req)
            if ok:
                pick = row.copy()
                pick["slot_used"] = which
                assign_slot(req, which)
                break

    if pick is None:
        break

    remaining_picks.append(pick)
    avail = avail[
        avail["web_name"].str.lower() != str(pick["web_name"]).lower()
    ].reset_index(drop=True)

# Enrich already-picked rows so the output is a complete current best roster view
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
    ]
)
if len(my_state):
    enrich = players[
        [
            "web_name",
            "pos",
            "team_name",
            "proj_points",
            "proj_nextN",
        ]
    ].drop_duplicates(subset=["web_name"])
    already = my_state.merge(enrich, on="web_name", how="left")
    # bring VOR/replacement context from the current flex-aware board if available
    enrich_vor = draft_board_flex[
        [
            "web_name",
            "replacement_at_rank",
            "replacement_points",
            "VOR",
        ]
    ].drop_duplicates(subset=["web_name"])
    already = already.merge(enrich_vor, on="web_name", how="left")
    already["already_picked"] = True

new_df = (
    pd.DataFrame(remaining_picks)
    if len(remaining_picks)
    else pd.DataFrame(columns=already.columns)
)
if len(new_df):
    new_df["already_picked"] = False

plan = pd.concat([already, new_df], ignore_index=True, sort=False)

# Order the roster in a human-friendly way: starters before bench, then by projected strength
slot_order = {
    "GK": 0,
    "F": 1,
    "M": 2,
    "D": 3,
    "FM_FLEX": 4,
    "MD_FLEX": 5,
    "BN": 6,
}
plan["slot_order"] = plan["slot_used"].map(slot_order).fillna(99)
plan = plan.sort_values(
    ["slot_order", "already_picked", "proj_points", "proj_nextN", "VOR"],
    ascending=[True, False, False, False, False],
).reset_index(drop=True)

# Save outputs
plan_out = OUTDIR_GW / "top_roster.csv"
plan.to_csv(plan_out, index=False)
print(f"Wrote {plan_out}")

# Also overwrite snake_plan.csv with the same content for backwards compatibility
snake_out = OUTDIR_GW / "snake_plan.csv"
plan.to_csv(snake_out, index=False)
print(f"Wrote {snake_out}")

# Write next-state suggestion: already picked + newly recommended picks
next_state_df = pd.concat(
    [
        (
            my_state[["web_name", "slot_used", "my_round"]]
            if len(my_state)
            else pd.DataFrame(columns=["web_name", "slot_used", "my_round"])
        ),
        (
            new_df.assign(my_round=np.nan)[["web_name", "slot_used", "my_round"]]
            if len(new_df)
            else pd.DataFrame(columns=["web_name", "slot_used", "my_round"])
        ),
    ],
    ignore_index=True,
).drop_duplicates(subset=["web_name"], keep="last")
next_state_path = OUTDIR_GW / "snake_state_next.csv"
next_state_df.to_csv(next_state_path, index=False)
print(
    f"Wrote {next_state_path}  (rename to {OUTDIR_GW/'snake_state.csv'} for the next run)"
)

# Write taken_next.csv = current taken + players already picked by me + newly recommended roster additions
current_taken = set(n.lower() for n in players_taken)
my_existing = (
    set(my_state["web_name"].astype(str).str.lower()) if len(my_state) else set()
)
my_new = set(new_df["web_name"].astype(str).str.lower()) if len(new_df) else set()
all_taken = current_taken | my_existing | my_new

# Preserve original casing by sourcing from the plan rows
name_map = {str(n).lower(): str(n) for n in plan["web_name"].dropna().tolist()}
taken_next_df = pd.DataFrame(
    {"web_name": [name_map.get(k, k) for k in sorted(all_taken)]}
)
taken_next_path = OUTDIR_GW / "taken_next.csv"
taken_next_df.to_csv(taken_next_path, index=False)
print(f"Wrote {taken_next_path}  (rename to {OUTDIR_GW/'taken.csv'} when appropriate)")

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
