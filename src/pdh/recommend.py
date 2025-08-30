\
"""
Season-weighted rates + fixture adjustment + weekly projections
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Iterable

def season_weighted(df: pd.DataFrame, season_col: str, weights: Dict[str,float], stat_cols: Iterable[str]) -> pd.DataFrame:
    tmp = df.copy()
    tmp["_w"] = tmp[season_col].astype(str).map(weights).fillna(1.0)
    for c in stat_cols:
        tmp[c+"_w"] = tmp[c] * tmp["_w"]
    agg = tmp.groupby("player", as_index=False)[[c+"_w" for c in stat_cols]].sum()
    wsum = tmp.groupby("player", as_index=False)["_w"].sum().rename(columns={"_w":"w"})
    out = agg.merge(wsum, on="player")
    for c in stat_cols:
        out[c+"_weighted"] = out[c+"_w"]/out["w"]
    return out.drop(columns=[c+"_w" for c in stat_cols] + ["w"])

def fixture_multiplier(difficulty: int) -> float:
    # Simple mapping: 2=+10%, 3=baseline, 4=-7%, 5=-15% (tweakable)
    return {2:1.10, 3:1.00, 4:0.93, 5:0.85}.get(int(difficulty), 1.00)

def project_gw(points_rate: pd.Series, minutes: float, fixture_diff: int) -> float:
    # points_rate: points per 90 proxy; minutes: expected minutes this GW
    mult = fixture_multiplier(fixture_diff)
    return float(points_rate) * (minutes/90.0) * mult
