"""
Compare soccerdata scraper backends for EPL data: FBref (already used, see
src/pdh/fbref.py), Understat (src/pdh/understat.py, new), and FotMob.

Checks, per source: whether a real fetch succeeds, how long it takes, and
whether it exposes player-level xG/xA - the specific gap FBref has (see
config/to_do.md "Assess scrapers"). Writes a decision doc to
config/scraper_comparison.md (data/ is gitignored, so results wouldn't
otherwise be kept).

Usage:
    python scripts/compare_scrapers.py [--season 2025] [--skip FBref FotMob]
"""

from __future__ import annotations
import argparse
import time
import traceback
from datetime import date
from pathlib import Path

# Import pdh.fbref first: it sets SOCCERDATA_DIR before soccerdata's own
# import-time config is read, so the project-local cache dir (not
# soccerdata's global default) is actually used.
from pdh import fbref
from pdh import understat as pdh_understat

import soccerdata as sd

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "config" / "scraper_comparison.md"


def check_fbref(season: int) -> dict:
    """Lightweight real check: fetch the season schedule (not the full,
    380-request player-match-stats scrape) via the Cloudflare-hardened
    SeleniumBase reader already used in src/pdh/fbref.py."""
    df = fbref.schedule([season], use_safe=True)
    return {
        "rows": len(df),
        "player_level_xg_xa": False,
        "notes": "No xG/xA in the stat tables this project currently pulls from FBref.",
    }


def check_understat(season: int) -> dict:
    df = pdh_understat.player_season_stats([season])
    if df.empty:
        return {
            "rows": 0,
            "player_level_xg_xa": True,
            "notes": (
                "Empty result - either the season hasn't started on Understat yet, "
                "or the request was soft-blocked (observed: understat.com serving a "
                "stub homepage with no embedded season data after repeated requests "
                "in a short window, despite HTTP 200)."
            ),
        }
    xg_cols = [c for c in df.columns if "xg" in c.lower() or c == "xa"]
    return {
        "rows": len(df),
        "player_level_xg_xa": True,
        "notes": f"Player-level columns include: {xg_cols}",
    }


def check_fotmob(season: int) -> dict:
    if not hasattr(sd, "FotMob"):
        return {
            "rows": 0,
            "player_level_xg_xa": None,
            "notes": (
                f"soccerdata {sd.__version__} no longer exposes a FotMob reader at all "
                "(present in 1.8.7, removed by 1.9.0 - likely a takedown/ToS issue). "
                "Not usable via soccerdata regardless of xG coverage."
            ),
        }
    fm = sd.FotMob(leagues="ENG-Premier League", seasons=[season])
    df = fm.read_team_match_stats(stat_type="Expected goals (xG)")
    return {
        "rows": len(df),
        "player_level_xg_xa": False,
        "notes": "xG available at team-match level only (no per-player breakdown in this soccerdata version).",
    }


SOURCES = {
    "FBref": check_fbref,
    "Understat": check_understat,
    "FotMob": check_fotmob,
}


def run(season: int, skip: list[str]) -> list[dict]:
    results = []
    for name, fn in SOURCES.items():
        if name in skip:
            print(f"--- {name} (skipped) ---")
            continue
        print(f"--- {name} ---")
        t0 = time.time()
        try:
            info = fn(season)
            dt = time.time() - t0
            print(f"[ok] {name} in {dt:.1f}s: {info}")
            results.append({"source": name, "ok": True, "seconds": round(dt, 1), **info})
        except Exception as e:
            dt = time.time() - t0
            print(f"[FAIL] {name} after {dt:.1f}s: {e}")
            traceback.print_exc()
            results.append(
                {
                    "source": name,
                    "ok": False,
                    "seconds": round(dt, 1),
                    "rows": 0,
                    "player_level_xg_xa": None,
                    "notes": str(e),
                }
            )
    return results


def write_report(results: list[dict], season: int):
    lines = [
        f"# Scraper comparison ({date.today().isoformat()}, season={season})",
        "",
        "Run via `python scripts/compare_scrapers.py`. See config/to_do.md",
        '"Assess scrapers" / "Assess naming of players for different scrapers".',
        "",
        "| Source | OK | Seconds | Player-level xG/xA | Notes |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['source']} | {r['ok']} | {r['seconds']} | "
            f"{r['player_level_xg_xa']} | {r['notes']} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "- **FBref** stays the primary source for the broad per-90 stat set "
        "(passing, defense, possession, misc) via src/pdh/fbref.py. Reliable "
        "once warm (SeleniumBase undetected-Chromium bypasses Cloudflare), but "
        "requires a real browser and is comparatively slow/heavy.",
        "- **Understat** (src/pdh/understat.py, new) is the source for player-level "
        "xG/xA/np_xG/xG-chain/xG-buildup - FBref has none of this. It's a plain "
        "`requests`-based reader (no browser needed) and returns rich data when "
        "reachable, but has been observed to soft-block (HTTP 200, stub page, no "
        "embedded data) under repeated automated requests in a short window - "
        "treat empty results as 'retry later', not 'no data', and rely on "
        "soccerdata's per-page on-disk cache to avoid re-hitting it unnecessarily.",
        "- **FotMob** is not usable at all: soccerdata removed the FotMob reader "
        "entirely as of 1.9.0 (present in 1.8.7, gone by 1.9.0 - likely a "
        "takedown/ToS issue). Not a source, full stop; drop it from "
        "consideration rather than treating it as a secondary team-power input.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--season", type=int, default=2025, help="FBref-style end-year season (2025 = 2024/25)"
    )
    parser.add_argument("--skip", nargs="*", default=[], help="Source names to skip")
    args = parser.parse_args()

    results = run(args.season, args.skip)
    write_report(results, args.season)
