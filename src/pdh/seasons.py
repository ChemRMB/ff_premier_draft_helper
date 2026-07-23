"""
Season config + folder naming, centralized so scripts/webapp don't each
hardcode "season_2526". Driven by config/seasons.yaml's
`current_season_end_year`, so a new season only needs that one value bumped
(and new historical data fetched) - the data/season_<tag>/ folder rolls
automatically.
"""

from __future__ import annotations
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SEASONS_YAML = ROOT / "config" / "seasons.yaml"


def load_seasons_config(path: Path = SEASONS_YAML) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def season_tag(end_year: int) -> str:
    """
    Folder tag for a season given its END year, matching the existing
    data/season_<tag>/ convention: season_2526 = the 2025/26 season, so
    end_year=2026 -> "2526" and end_year=2027 -> "2627".
    """
    start_year = end_year - 1
    return f"{start_year % 100:02d}{end_year % 100:02d}"


def current_season_tag(path: Path = SEASONS_YAML) -> str:
    return season_tag(load_seasons_config(path)["current_season_end_year"])


def current_season_dir(path: Path = SEASONS_YAML) -> Path:
    """data/season_<tag>/ for the current season (see config/seasons.yaml)."""
    return ROOT / "data" / f"season_{current_season_tag(path)}"
