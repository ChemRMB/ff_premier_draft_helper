"""
Shared player-name-linking utilities: canonical normalization, team-scoped fuzzy
matching (difflib), and curated-CSV override support.

This generalizes the FPL<->FBref matcher originally built ad hoc in
scripts/make_recommendations.py (build_player_link/load_or_build_name_link) so
other name-matching problems (Sleeper<->FPL, FPL<->Understat, ...) reuse the same
proven approach instead of separate one-off matchers.
"""

from __future__ import annotations
import re
import unicodedata
import difflib
from pathlib import Path
import pandas as pd


def strip_accents(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)
    )


def canonical_name(s) -> str:
    s = strip_accents(s).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def norm_team_key(s) -> str:
    s = strip_accents(s).lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def link_names(
    source: pd.DataFrame,
    target: pd.DataFrame,
    source_name_col: str,
    target_name_col: str,
    source_team_col: str | None = None,
    target_team_col: str | None = None,
    cutoff: float = 0.90,
) -> pd.DataFrame:
    """
    Link each row of `source` to the best-matching name in `target`.

    Strategy: exact canonical-name match, then team-scoped fuzzy fallback
    (difflib.SequenceMatcher ratio >= cutoff). Falls back to the full target pool
    when no team match is found or team columns aren't provided.

    Returns a copy of `source` with two new columns: `matched_name` (the value
    from target_name_col) and `match_method` ('exact' | 'fuzzy' | None).
    """
    src = source.copy()
    src["_name_key"] = src[source_name_col].map(canonical_name)

    tgt_cols = [target_name_col] + ([target_team_col] if target_team_col else [])
    tgt = target[tgt_cols].drop_duplicates().copy()
    tgt["_name_key"] = tgt[target_name_col].map(canonical_name)
    if target_team_col:
        tgt["_team_key"] = tgt[target_team_col].map(norm_team_key)

    exact_map = dict(zip(tgt["_name_key"], tgt[target_name_col]))
    src["matched_name"] = src["_name_key"].map(exact_map)
    src["match_method"] = src["matched_name"].map(lambda v: "exact" if pd.notna(v) else None)

    if source_team_col:
        src["_team_key"] = src[source_team_col].map(norm_team_key)

    pending_idx = src.index[src["matched_name"].isna()]
    if len(pending_idx):
        pool_by_team = (
            {t: grp[target_name_col].tolist() for t, grp in tgt.groupby("_team_key")}
            if target_team_col
            else {}
        )
        full_pool = tgt[target_name_col].tolist()

        for idx in pending_idx:
            team_key = src.at[idx, "_team_key"] if "_team_key" in src.columns else None
            pool = pool_by_team.get(team_key, full_pool) if team_key else full_pool
            if not pool:
                pool = full_pool
            target_key = src.at[idx, "_name_key"]
            best_name, best_score = None, 0.0
            for cand in pool:
                score = difflib.SequenceMatcher(
                    None, target_key, canonical_name(cand)
                ).ratio()
                if score > best_score:
                    best_name, best_score = cand, score
            if best_score >= cutoff:
                src.at[idx, "matched_name"] = best_name
                src.at[idx, "match_method"] = "fuzzy"

    return src.drop(columns=["_name_key", "_team_key"], errors="ignore")


def apply_curated_overrides(
    link: pd.DataFrame,
    curated_path: Path,
    key_col: str,
    match_col: str = "matched_name",
) -> tuple[pd.DataFrame, int]:
    """Overlay a curated CSV (columns: `key_col`, `match_col`) onto an auto-built link table."""
    if not curated_path.exists():
        return link, 0
    curated = pd.read_csv(curated_path)
    if key_col not in curated.columns or match_col not in curated.columns:
        return link, 0

    merged = link.merge(
        curated[[key_col, match_col]].rename(columns={match_col: "_curated_match"}),
        on=key_col,
        how="left",
    )
    is_override = merged["_curated_match"].notna()
    n_overrides = int(is_override.sum())
    merged[match_col] = merged["_curated_match"].combine_first(merged[match_col])
    merged.loc[is_override, "match_method"] = "curated"
    merged = merged.drop(columns=["_curated_match"])
    return merged, n_overrides
