"""
Streamlit app - live snake-draft assistant (Plan E, first cut).

Run with:
  uv run streamlit run scripts/streamlit_app.py

Focus is the live draft: best-available recommendations (VOR-ranked, taken
players excluded in real time against the actual Sleeper draft), plus a view
of what every team - including yours - has drafted so far. Other reporting
(roster history across weeks, per-gameweek/per-player stats) is deferred;
see config/to_do.md's Flask/Streamlit section for that scope.
"""

from __future__ import annotations
import streamlit as st

from pdh.webapp import data
from pdh.draft_board import positional_scarcity

st.set_page_config(page_title="Prebens Premiership - Draft Assistant", layout="wide")


@st.cache_data(ttl=15)
def _league_config():
    return data.load_league_config()


@st.cache_data(ttl=15)
def _league_teams():
    return data.load_league_teams()


@st.cache_data(ttl=15)
def _live_picks():
    return data.load_live_draft_picks()


@st.cache_data(ttl=60)
def _draft_board(gw_tag: str):
    return data.load_draft_board(gw_tag)


@st.cache_data(ttl=300)
def _available_gws():
    return data.available_gameweeks()


def snake_order(teams: int, round_num: int) -> list[int]:
    base = list(range(1, teams + 1))
    return base if (round_num % 2 == 1) else list(reversed(base))


def on_the_clock(draft: dict, teams_df, n_picks_made: int):
    """Best-effort "who picks next" - only works once the commissioner has
    randomized draft_order (a live event we can't predict ahead of time)."""
    draft_order = draft.get("draft_order")
    rounds = draft.get("settings", {}).get("rounds")
    n_teams = draft.get("settings", {}).get("teams")
    if not draft_order or not rounds or not n_teams:
        return None
    slot_to_user = {v: k for k, v in draft_order.items()}
    round_num = n_picks_made // n_teams + 1
    if round_num > rounds:
        return "Draft complete"
    pick_in_round = n_picks_made % n_teams
    order = snake_order(n_teams, round_num)
    slot = order[pick_in_round]
    user_id = slot_to_user.get(slot)
    row = teams_df[teams_df["user_id"] == user_id]
    team_name = row.iloc[0]["team_name"] if len(row) else user_id
    return f"Round {round_num}, pick {pick_in_round + 1} - **{team_name}**"


setup, draft = _league_config()
teams_df = _league_teams()
my_team_id = setup.get("my_team_id")

st.title(setup.get("name", "Draft Assistant"))

top_cols = st.columns([1, 1, 1, 2])
top_cols[0].metric("Status", draft.get("status", "?"))
top_cols[1].metric("Teams", draft.get("settings", {}).get("teams", "?"))
top_cols[2].metric("Rounds", draft.get("settings", {}).get("rounds", "?"))
if top_cols[3].button("🔄 Refresh live picks"):
    st.cache_data.clear()
    st.rerun()

picks_df = _live_picks()
clock_msg = on_the_clock(draft, teams_df, len(picks_df))
if clock_msg:
    st.info(f"On the clock: {clock_msg}")
elif draft.get("status") == "pre_draft":
    st.info("Draft order not set yet - waiting on the commissioner to start the draft.")

st.divider()

gws = _available_gws()
default_gw = data.current_fpl_gameweek_tag()
default_idx = gws.index(default_gw) if default_gw in gws else max(len(gws) - 1, 0)
gw_tag = st.sidebar.selectbox(
    "Projections gameweek",
    gws,
    index=default_idx if gws else 0,
    help=(
        "Which gwN/draft_board_flexaware.csv to rank players from. Regenerate "
        "with scripts/make_recommendations.py --event N --refresh."
    ),
)
pos_filter = st.sidebar.radio("Position", ["All", "F", "M", "D", "GK"], horizontal=True)
top_n = st.sidebar.slider("Rows to show", 10, 100, 30, step=10)

board = _draft_board(gw_tag) if gw_tag else data.load_draft_board("")
drafted_names = set(picks_df["web_name"].dropna()) if not picks_df.empty else set()
avail_now = data.best_available(board, drafted_names) if not board.empty else board

my_slot = data.my_draft_slot(draft, my_team_id)
slot_map = data.slot_to_roster_id_map(draft, teams_df)
scarcity = None
if my_slot and slot_map and not avail_now.empty:
    scarcity = positional_scarcity(
        avail_now,
        picks_df,
        setup.get("roster_positions", []),
        teams=draft.get("settings", {}).get("teams", len(slot_map)),
        rounds=draft.get("settings", {}).get("rounds", 16),
        my_slot=my_slot,
        n_picks_made=len(picks_df),
        slot_to_roster_id=slot_map,
    )

st.subheader("Positional run risk")
if scarcity is None:
    st.caption(
        "Needs a live draft in progress with draft_order set - shows how many "
        "opponents picking before your next turn still need each position, vs "
        "how many quality (VOR > 0) players are left there. Ratio > 1 means "
        "that position is at real risk of running out before you pick again."
    )
else:
    st.dataframe(
        scarcity.rename(
            columns={
                "pos": "Position",
                "remaining_quality": "Quality left",
                "picks_before_your_turn": "Picks before your turn",
                "teams_needing": "Opponents still needing it",
                "scarcity_ratio": "Risk ratio",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    top_risk = scarcity.iloc[0]
    if top_risk["scarcity_ratio"] > 1:
        st.warning(
            f"**{top_risk['pos']}** is at risk of running out before your next turn - "
            f"{int(top_risk['teams_needing'])} opponents picking before you still need "
            f"one, but only {int(top_risk['remaining_quality'])} quality options remain. "
            f"Worth weighing against pure VOR right now."
        )

st.divider()

col_avail, col_mine = st.columns([2, 1])

with col_avail:
    st.subheader("Best available")
    if board.empty:
        st.warning(
            f"No draft board for {gw_tag} yet - run "
            f"`scripts/make_recommendations.py --event {gw_tag.lstrip('gw') if gw_tag else 1} --refresh`."
        )
    else:
        avail = avail_now[avail_now["pos"] == pos_filter] if pos_filter != "All" else avail_now
        st.dataframe(
            avail[["web_name", "pos", "team_name", "proj_points", "proj_nextN", "VOR"]].head(
                top_n
            ),
            width="stretch",
            hide_index=True,
        )

with col_mine:
    st.subheader("My picks so far")
    if picks_df.empty:
        st.caption("No picks yet.")
    else:
        mine = picks_df[picks_df["picked_by"] == my_team_id]
        if mine.empty:
            st.caption("No picks yet.")
        else:
            st.dataframe(
                mine[["pick_no", "round", "web_name", "full_name"]],
                width="stretch",
                hide_index=True,
            )

st.divider()
st.subheader("Every team's draft so far")
if picks_df.empty:
    st.caption("No picks yet - this fills in once the draft starts.")
else:
    team_names = sorted(picks_df["team_name"].dropna().unique().tolist())
    tabs = st.tabs(team_names) if team_names else []
    for tab, team_name in zip(tabs, team_names):
        with tab:
            sub = picks_df[picks_df["team_name"] == team_name].sort_values("pick_no")
            st.dataframe(
                sub[["pick_no", "round", "web_name", "full_name"]],
                width="stretch",
                hide_index=True,
            )
