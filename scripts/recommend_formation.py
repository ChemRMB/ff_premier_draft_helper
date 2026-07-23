import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from pdh import fpl
from pdh.sleeper import (
    get_sleeper_rosters,
    get_sleeper_players,
    get_my_team_players,
    get_positions,
    normalize_squad_df,
    get_sleeper_name_to_web_name,
)

# ---------- config ----------
ap = argparse.ArgumentParser()
ap.add_argument(
    "--event",
    type=int,
    default=None,
    help="FPL gameweek (event). If omitted, uses the current/next FPL gameweek.",
)
args = ap.parse_args()

project_root = Path(__file__).resolve().parents[1]
path_data = project_root / "data"
SEASON = 2526
GW = args.event if args.event is not None else fpl.current_gameweek()
print(f"Using gameweek: {GW}")

formation_map = {
    "3-5-2": {"D": 3, "M": 5, "F": 2},
    "4-4-2": {"D": 4, "M": 4, "F": 2},
    "4-5-1": {"D": 4, "M": 5, "F": 1},
    "3-4-3": {"D": 3, "M": 4, "F": 3},
    "4-3-3": {"D": 4, "M": 3, "F": 3},
    "5-4-1": {"D": 5, "M": 4, "F": 1},
}

# ---------- load inputs ----------
df_fixtures = pd.read_csv(
    path_data / f"season_{SEASON}/fixtures.csv"
)  # optional in this script
df_squads_norm = pd.read_csv(path_data / f"season_{SEASON}/current_squads.csv")
# df_squads_norm = normalize_squad_df(df_squads)
print("Squads:")
print(df_squads_norm.head())

sleeper_rosters = get_sleeper_rosters()
print("Rosters:")
print(sleeper_rosters)
df_players = pd.read_csv(path_data / f"season_{SEASON}/sleeper_players.csv")
if df_players.empty:
    print("No sleeper players data found, fetching from API...")
    df_players = get_sleeper_players()
    df_players.to_csv(path_data / f"season_{SEASON}/sleeper_players.csv", index=False)
    print("Wrote:", path_data / f"season_{SEASON}/sleeper_players.csv")
df_my_team = get_my_team_players(sleeper_rosters, df_players)
print("My team players (Sleeper):")
print(df_my_team)
# df_my_team["pos"] = get_positions(df_my_team) # just use positions
# map Sleeper names -> FPL web_name
my_web_names = get_sleeper_name_to_web_name(df_my_team, df_squads_norm)
df_my_team["web_name"] = my_web_names
print("My team players (Sleeper -> web_name):")
print(df_my_team[["full_name", "web_name", "position"]])
# inplace override due to no sleeper API access
# df_my_team = pd.read_csv(path_data / f"season_{SEASON}/my_roster.csv")
# my_web_names = df_my_team["web_name"].tolist()

print("load draft board flex aware")
draft_board = pd.read_csv(
    path_data / f"season_{SEASON}/game_weeks/gw{GW}/draft_board_flexaware.csv"
)

# loc players from lst_my_players in draft_board
df_my_team_roster = draft_board[draft_board["web_name"].isin(my_web_names)]
# replace pos with position from df_my_team
# df_my_team_roster = df_my_team_roster.merge(
#     df_my_team[["web_name", "pos"]].drop_duplicates(subset="web_name"),
#     on="web_name",
#     how="left",
# )

# mapping_pos = df_my_team.set_index("web_name")["pos"].to_dict()
mapping_pos = df_my_team.set_index("web_name")["position"].to_dict()
df_my_team_roster = df_my_team_roster.copy().reset_index(drop=True)
df_my_team_roster["pos"] = df_my_team_roster["web_name"].map(mapping_pos)


draft_board = pd.read_csv(
    path_data / f"season_{SEASON}/game_weeks/gw{GW}/draft_board_flexaware.csv"
)

# Focus on my roster present in the board
board_cols = set(draft_board.columns)
needed = {"web_name", "pos", "team_name", "proj_points", "proj_nextN"}
missing_cols = needed - board_cols
if missing_cols:
    raise RuntimeError(f"draft_board missing required columns: {missing_cols}")

my_board = draft_board[draft_board["web_name"].isin(my_web_names)].copy()
# fix positions from df_my_team_roster
my_board = my_board.reset_index(drop=True)
my_board["pos"] = my_board["web_name"].map(mapping_pos)

# Optional: bring proj_nextN if present for bench tie-breaks
has_nextN = "proj_nextN" in my_board.columns
if not has_nextN:
    my_board["proj_nextN"] = my_board["proj_points"]

# Basic sanity: ensure pos is one of F/M/D/GK
my_board = my_board[my_board["pos"].isin(["F", "M", "D", "GK"])].copy()

# ---------- GK pick (not part of formation) ----------
gk_pool = my_board[my_board["pos"] == "GK"].copy()
gk_pick = None
if len(gk_pool):
    gk_pick = gk_pool.sort_values("proj_points", ascending=False).iloc[0]

gk_picks = draft_board[draft_board["pos"] == "GK"].sort_values(
    "proj_points", ascending=False
)
print("Top 10 GK on board (not necessarily in my roster):")
print(gk_picks.head(10)[["web_name", "team_name", "proj_points"]])


# ---------- choose best formation ----------
def pick_best_for_formation(df: pd.DataFrame, d: int, m: int, f: int):
    # pick top D
    d_pool = df[df["pos"] == "D"].sort_values("proj_points", ascending=False)
    m_pool = df[df["pos"] == "M"].sort_values("proj_points", ascending=False)
    f_pool = df[df["pos"] == "F"].sort_values("proj_points", ascending=False)

    if len(d_pool) < d or len(m_pool) < m or len(f_pool) < f:
        # not enough players for this formation
        return None, -np.inf

    d_take = d_pool.head(d)
    m_take = m_pool.head(m)
    f_take = f_pool.head(f)

    starters = pd.concat([d_take, m_take, f_take], ignore_index=True)
    total = starters["proj_points"].sum()
    return starters, float(total)


best_name, best_total, best_starters = None, -np.inf, None
for name, counts in formation_map.items():
    starters, total = pick_best_for_formation(
        my_board, counts["D"], counts["M"], counts["F"]
    )
    if starters is None:
        continue
    if total > best_total:
        best_name, best_total, best_starters = name, total, starters

if best_starters is None:
    raise RuntimeError("No valid formation could be formed from your current roster.")

# Ensure GK is included separately
if gk_pick is not None:
    gk_row = gk_pick.to_frame().T.copy()
    gk_row["role"] = "GK"
else:
    gk_row = pd.DataFrame(columns=best_starters.columns.tolist() + ["role"])


# Tag starters with roles (D1.., M1.., F1..)
def add_roles(starters_df: pd.DataFrame, formation_name: str) -> pd.DataFrame:
    dseq = (
        starters_df[starters_df["pos"] == "D"]
        .sort_values("proj_points", ascending=False)
        .reset_index(drop=True)
    )
    mseq = (
        starters_df[starters_df["pos"] == "M"]
        .sort_values("proj_points", ascending=False)
        .reset_index(drop=True)
    )
    fseq = (
        starters_df[starters_df["pos"] == "F"]
        .sort_values("proj_points", ascending=False)
        .reset_index(drop=True)
    )
    dseq["role"] = [f"D{i+1}" for i in range(len(dseq))]
    mseq["role"] = [f"M{i+1}" for i in range(len(mseq))]
    fseq["role"] = [f"F{i+1}" for i in range(len(fseq))]
    out = pd.concat([dseq, mseq, fseq], ignore_index=True)
    out["formation"] = formation_name
    out["starter"] = True
    return out


starters_tagged = add_roles(best_starters, best_name)

# Build bench from remaining outfielders (top by proj_nextN, then proj_points)
used = set(starters_tagged["web_name"])
bench_pool = my_board[
    (my_board["pos"] != "GK") & (~my_board["web_name"].isin(used))
].copy()
bench = bench_pool.sort_values(
    ["proj_nextN", "proj_points"], ascending=[False, False]
).assign(starter=False, role="BN", formation=best_name)

# Combine final lineup
final_cols = [
    "starter",
    "role",
    "formation",
    "web_name",
    "team_name",
    "pos",
    "proj_points",
    "proj_nextN",
    "VOR",
]
for c in final_cols:
    if c not in starters_tagged.columns:
        starters_tagged[c] = np.nan
if not bench.empty:
    for c in final_cols:
        if c not in bench.columns:
            bench[c] = np.nan

lineup = pd.concat([starters_tagged[final_cols], bench[final_cols]], ignore_index=True)

# Insert GK at the top
if not gk_row.empty:
    # ensure required cols
    for c in final_cols:
        if c not in gk_row.columns:
            gk_row[c] = np.nan
    gk_row = gk_row.assign(starter=True, formation=best_name)[final_cols]
    lineup = pd.concat([gk_row, lineup], ignore_index=True)

# Save + print
out_path = path_data / f"season_{SEASON}/game_weeks/gw{GW}/gw{GW}_lineup.csv"
lineup.to_csv(out_path, index=False)

print(f"Best formation: {best_name}  (sum proj_points={best_total:.2f})")
if gk_pick is not None:
    print(
        f"GK: {gk_pick['web_name']} ({gk_pick['team_name']}) — {gk_pick['proj_points']:.2f} pts"
    )
print("Outfield starters:")
print(
    starters_tagged.sort_values(["pos", "proj_points"], ascending=[True, False])[
        ["role", "web_name", "team_name", "pos", "proj_points", "proj_nextN"]
    ]
)
print(f"Saved lineup to: {out_path}")

# --- LOB (Lineup Over Bench) export ---
# LOB = how many points a player would add if they replaced your worst starter at the same position
# Note: LOB > 0 suggests a swap improves your XI. LOB == 0 means they don't beat the worst starter.

# 1) Find worst starter at each position (include GK)
worst_starter = {}

# GK: if we picked one, that's the lone GK starter
if gk_pick is not None:
    worst_starter["GK"] = float(gk_pick["proj_points"])

for p in ["D", "M", "F"]:
    s = starters_tagged[starters_tagged["pos"] == p]
    if len(s):
        worst_starter[p] = float(s["proj_points"].min())


# 2) Compute LOB for every roster player on your draft board
# Interpreting LOB: Lineup Over Bench
# LOB > 0: this player would increase your expected points vs your current worst starter at that position.
# LOB = 0: no improvement (they don’t beat your worst starter).
def _lob_row(r):
    """Compute LOB for a single row
    LOB = max(0, proj_points − worst_starter_proj_at_same_position)
    """
    thr = worst_starter.get(r["pos"], np.nan)
    if pd.isna(thr):
        return np.nan
    return max(0.0, float(r["proj_points"]) - thr)


lob = my_board.copy()
lob["starter"] = lob["web_name"].isin(starters_tagged["web_name"])
lob["LOB"] = lob.apply(_lob_row, axis=1)
lob = lob.sort_values(
    ["pos", "LOB", "proj_points", "proj_nextN"], ascending=[True, False, False, False]
)

# Optional: show only bench players
lob_view_non_starters = lob[~lob["starter"]].copy()

lob_view = lob

lob_out = path_data / f"season_{SEASON}/game_weeks/gw{GW}/gw{GW}_lob.csv"
lob_view[
    ["web_name", "team_name", "pos", "proj_points", "proj_nextN", "LOB", "starter"]
].to_csv(lob_out, index=False)

print(f"Saved LOB suggestions to: {lob_out}")
print("Top LOB candidates by position:")
print(
    lob_view.sort_values(by="pos")[
        ["web_name", "pos", "proj_points", "proj_nextN", "LOB", "starter"]
    ]
)

print("\nTop LOB candidates (non-starters only):")
print(
    lob_view_non_starters.sort_values(by="pos")[
        ["web_name", "pos", "proj_points", "proj_nextN", "LOB"]
    ]
)

# ---------- Last call: confirmed-to-play filter ----------
# Run this again close to kickoff to catch late fitness news: excludes any
# chosen starter whose FPL status/chance_of_playing signals they're doubtful
# or out, and re-picks the formation from confirmed-available players only.
AVAILABILITY_STATUS_OUT = {"i", "s", "u"}  # injured, suspended, unavailable
CHANCE_DOUBTFUL_BELOW = 75  # FPL chance_of_playing_this_round, percent

avail = df_squads_norm[["web_name", "status", "chance_of_playing_this_round"]].copy()
avail["chance_of_playing_this_round"] = pd.to_numeric(
    avail["chance_of_playing_this_round"], errors="coerce"
)


def _is_doubtful(row) -> bool:
    if row["status"] in AVAILABILITY_STATUS_OUT:
        return True
    chance = row["chance_of_playing_this_round"]
    return bool(pd.notna(chance) and chance < CHANCE_DOUBTFUL_BELOW)


avail["doubtful"] = avail.apply(_is_doubtful, axis=1)
doubtful_names = set(avail.loc[avail["doubtful"], "web_name"])

chosen_names = set(starters_tagged["web_name"])
if gk_pick is not None:
    chosen_names.add(gk_pick["web_name"])
doubtful_chosen = doubtful_names & chosen_names

if doubtful_chosen:
    print("\n[last call] Doubtful/out among your chosen starters:")
    print(
        avail[avail["web_name"].isin(doubtful_chosen)][
            ["web_name", "status", "chance_of_playing_this_round"]
        ]
    )

    my_board_available = my_board[~my_board["web_name"].isin(doubtful_names)].copy()

    best_name_lc, best_total_lc, best_starters_lc = None, -np.inf, None
    for name, counts in formation_map.items():
        starters_lc, total_lc = pick_best_for_formation(
            my_board_available, counts["D"], counts["M"], counts["F"]
        )
        if starters_lc is None:
            continue
        if total_lc > best_total_lc:
            best_name_lc, best_total_lc, best_starters_lc = name, total_lc, starters_lc

    gk_pool_available = gk_pool[~gk_pool["web_name"].isin(doubtful_names)]
    gk_pick_lc = (
        gk_pool_available.sort_values("proj_points", ascending=False).iloc[0]
        if len(gk_pool_available)
        else None
    )

    if best_starters_lc is not None:
        starters_lc_tagged = add_roles(best_starters_lc, best_name_lc)
        print(
            f"\n[last call] Alternate formation excluding doubtful/out: {best_name_lc} "
            f"(sum proj_points={best_total_lc:.2f}, was {best_total:.2f})"
        )
        if gk_pick_lc is not None:
            print(
                f"[last call] GK: {gk_pick_lc['web_name']} ({gk_pick_lc['team_name']}) "
                f"— {gk_pick_lc['proj_points']:.2f} pts"
            )
        lc_cols = ["role", "web_name", "team_name", "pos", "proj_points"]
        print(starters_lc_tagged[lc_cols])

        lc_out = path_data / f"season_{SEASON}/game_weeks/gw{GW}/gw{GW}_lastcall.csv"
        starters_lc_tagged[lc_cols].to_csv(lc_out, index=False)
        print(f"Saved last-call lineup to: {lc_out}")
    else:
        print(
            "[last call] Not enough confirmed-available players to form any "
            "formation - check your roster/bench."
        )
else:
    print("\n[last call] No doubtful/out starters detected - original lineup stands.")
