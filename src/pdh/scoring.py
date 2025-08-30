"""
Compute Sleeper-style fantasy points from normalized stats.
"""

from __future__ import annotations
import json, yaml
import pandas as pd


def load_sleeper_scoring(path: str) -> dict:
    data = json.load(open(path, "r"))
    return data.get("scoring_settings", {})


def load_stat_map(path: str) -> dict:
    return yaml.safe_load(open(path, "r"))


# def points_from_row(row: pd.Series, position: str, scoring: dict, stat_map: dict) -> float:
#     """
#     position: one of 'F','M','D','GK' (flex is resolved before calling)
#     scoring: raw dict from Sleeper with keys like 'pos_f_g', 'pos_m_at', etc.
#     stat_map: maps short keys to column names derived from FBref/FotMob.
#     """
#     pos_key = {"F":"forwards","M":"midfielders","D":"defenders","GK":"goalkeepers"}[position]
#     mapping = stat_map.get(pos_key, {})
#     total = 0.0
#     for short, col in mapping.items():
#         if col not in row.index:
#             continue
#         # Sleeper scoring keys pattern: pos_<pos>_<short>
#         skey = f"pos_{position.lower()}_{short}"
#         if skey in scoring:
#             total += scoring[skey] * float(row[col])
#     return float(total)

# def apply_scoring(df: pd.DataFrame, position_col: str, scoring: dict, stat_map: dict) -> pd.Series:
#     return df.apply(lambda r: points_from_row(r, r[position_col], scoring, stat_map), axis=1)


def points_from_row(
    row: pd.Series, position: str, scoring: dict, stat_map: dict
) -> float:
    """
    position: one of F, M, D, GK
    scoring: dict from sleeper_setup.json["scoring_settings"]
    stat_map: YAML mapping of short keys (e.g. 'g','kp') to canonical column names (e.g. 'goals','key_passes')
    """
    pos_key = {
        "F": "forwards",
        "M": "midfielders",
        "D": "defenders",
        "GK": "goalkeepers",
    }[position]
    mapping = stat_map.get(pos_key, {})
    total = 0.0

    pos_l = position.lower()

    def resolve_weight(short: str) -> float:
        # Try common key patterns, most specific -> least
        candidates = [
            f"pos_{pos_l}_{short}",  # pos_f_g
            f"pos_{pos_l}{short}",  # pos_fg (just in case)
            f"{pos_l}_{short}",  # f_g
            f"{short}",  # g
            f"soccer_{short}",  # soccer_g, just in case
        ]
        for k in candidates:
            if k in scoring and scoring[k] is not None:
                try:
                    return float(scoring[k])
                except Exception:
                    pass
        return 0.0

    for short, col in mapping.items():
        if col in row.index and pd.notna(row[col]):
            w = resolve_weight(short)
            if w != 0.0:
                total += w * float(row[col])
    return float(total)
