"""
Suggest FBref matches for FPL players the auto-linker couldn't match, to make
curating config/name_link_curated.csv a verify-a-suggestion task instead of a
look-it-up-from-scratch one.

For each player in a gameweek's name_link_missing.csv, this fuzzy-matches the
FPL full name against every player in the FBref historical data (all seasons,
NOT scoped to the player's current FPL team - a new signing appears under their
OLD club in history), using a lower cutoff than the pipeline's 0.90 so it
surfaces the near-misses that got rejected. It reports the best candidate plus
one alternate, each player's FBref team/seasons for a sanity check, and a
confidence tier.

Only the "review" tier (and the borderline ones) are worth curating; players
with no plausible FBref match are genuinely new to the PL (promoted-team
players, foreign signings, youth) and correctly fall back to position priors -
there is nothing to map them to.

Usage:
  python scripts/suggest_name_links.py [--missing PATH] [--gw N] [--min-score 0.5]

Writes <gw folder>/name_link_suggestions.csv (columns line up with
config/name_link_curated.csv so accepted rows paste straight in: fill/keep
`player_fbref`, drop the score/context columns).
"""

from __future__ import annotations
import argparse
import difflib
from pathlib import Path

import pandas as pd

from pdh.namelink import canonical_name
from pdh.seasons import current_season_dir

ROOT = Path(__file__).resolve().parents[1]
HIST_PATH = ROOT / "data" / "historical" / "players_historical_20251025.csv"


def _season_label(season_tag: int) -> str:
    """FBref season int (e.g. 2526) -> '2025/26' for readability."""
    s = str(season_tag)
    if len(s) == 4:
        return f"20{s[:2]}/{s[2:]}"
    return s


def build_fbref_index() -> pd.DataFrame:
    """One row per FBref player: canonical name, the teams they've appeared
    for, and the seasons - for matching and for the reviewer's sanity check."""
    hist = pd.read_csv(HIST_PATH)
    grp = hist.groupby("player").agg(
        teams=("team", lambda s: sorted(set(s.dropna()))),
        seasons=("season", lambda s: sorted(set(s.dropna()))),
    )
    grp = grp.reset_index()
    grp["canon"] = grp["player"].map(canonical_name)
    return grp


def _name_match_score(a: str, b: str) -> float:
    """
    Fuzzy score between two canonical names, with a token-containment boost:
    if the shorter name (>= 2 tokens) is fully contained in the longer, treat
    it as a near-certain match (a common pattern here is the FPL name being a
    truncation of the FBref name - "abdul fatawu" vs "abdul fatawu issahaku",
    "jaden philogene" vs "jaden philogene bidace"), which a raw difflib ratio
    under-scores because of the extra tokens.
    """
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if len(short) >= 2 and short <= long_:
        return max(ratio, 0.9)
    return ratio


def suggest(missing: pd.DataFrame, fbref: pd.DataFrame, min_score: float) -> pd.DataFrame:
    canon_to_names: dict[str, list[str]] = {}
    for _, r in fbref.iterrows():
        canon_to_names.setdefault(r["canon"], []).append(r["player"])
    fbref_by_name = fbref.set_index("player")
    pool = list(canon_to_names.keys())

    rows = []
    for _, m in missing.iterrows():
        key = canonical_name(m["player_full"])
        # difflib prefilter (fast), then re-rank by the containment-aware score
        prelim = difflib.get_close_matches(key, pool, n=8, cutoff=min_score)
        cand_keys = sorted(prelim, key=lambda ck: _name_match_score(key, ck), reverse=True)[:3]

        def _detail(ck: str):
            name = canon_to_names[ck][0]
            info = fbref_by_name.loc[name]
            if isinstance(info, pd.DataFrame):  # duplicate name, take first
                info = info.iloc[0]
            score = round(_name_match_score(key, ck), 3)
            teams = info["teams"]
            same_team = str(m["team_name"]).lower() in {str(t).lower() for t in teams}
            return name, score, teams, info["seasons"], same_team

        best = _detail(cand_keys[0]) if cand_keys else (None, 0.0, [], [], False)
        alt = _detail(cand_keys[1]) if len(cand_keys) > 1 else (None, 0.0, [], [], False)

        name, score, teams, seasons, same_team = best
        # Confidence: an exact-ish name that also played for this FPL team is a
        # near-certain match; a strong name match at a *different* club is the
        # classic transfer case (still likely, but verify); weak = review/skip.
        if score >= 0.92 and same_team:
            tier = "high (same team)"
        elif score >= 0.92:
            tier = "high (transfer?)"
        elif score >= 0.75:
            tier = "review"
        elif score >= min_score:
            tier = "weak"
        else:
            tier = "no match - likely new to PL"

        rows.append(
            {
                "player_full": m["player_full"],
                "web_name": m.get("web_name", ""),
                "team_name": m["team_name"],
                # Prefill only high-confidence matches: the 0.75-0.85 band is
                # dominated by same-surname/first-name false positives (verify
                # by hand), so leaving those blank avoids silently curating a
                # wrong mapping if the column is pasted wholesale.
                "player_fbref": name if score >= 0.85 else "",
                "suggestion": name or "",
                "score": score,
                "confidence": tier,
                "fbref_teams": ", ".join(map(str, teams)),
                "fbref_seasons": ", ".join(_season_label(s) for s in seasons),
                "alt_suggestion": alt[0] or "",
                "alt_score": alt[1],
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--missing", type=str, default=None, help="Path to a name_link_missing.csv")
    ap.add_argument("--gw", type=int, default=None, help="Gameweek number (finds its missing file)")
    ap.add_argument("--min-score", type=float, default=0.5, help="Lowest fuzzy score to surface")
    args = ap.parse_args()

    if args.missing:
        missing_path = Path(args.missing)
    else:
        gw_dir = current_season_dir() / "game_weeks"
        if args.gw is not None:
            missing_path = gw_dir / f"gw{args.gw}" / "name_link_missing.csv"
        else:
            gws = sorted(
                (p for p in gw_dir.glob("gw*/name_link_missing.csv")),
                key=lambda p: int(p.parent.name[2:]) if p.parent.name[2:].isdigit() else 0,
            )
            if not gws:
                raise SystemExit(f"No name_link_missing.csv found under {gw_dir}")
            missing_path = gws[-1]

    missing = pd.read_csv(missing_path)
    fbref = build_fbref_index()
    out = suggest(missing, fbref, args.min_score)

    out_path = missing_path.parent / "name_link_suggestions.csv"
    out.to_csv(out_path, index=False)

    tiers = out["confidence"].value_counts()
    print(f"Read {len(missing)} unmatched players from {missing_path}")
    print(f"Wrote suggestions to {out_path}\n")
    print("By confidence:")
    for tier, n in tiers.items():
        print(f"  {n:>3}  {tier}")
    strong = out[out["score"] >= 0.75]
    print(f"\n{len(strong)} worth reviewing (score >= 0.75):")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(
            strong[
                ["player_full", "team_name", "suggestion", "score", "confidence", "fbref_teams"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
