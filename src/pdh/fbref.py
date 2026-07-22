"""
soccerdata (FBref) accessors for team + player stats (last 3 seasons).
See: https://soccerdata.readthedocs.io/
"""

from __future__ import annotations
from contextlib import suppress
import os
import socket
import pandas as pd
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
SOCCERDATA_BASE_DIR = ROOT / "data" / "cache" / "soccerdata"
os.environ.setdefault("SOCCERDATA_DIR", str(SOCCERDATA_BASE_DIR))

import soccerdata as sd
import soccerdata.fbref as sd_fbref
from seleniumbase import Driver


FBREF_LEAGUE = "ENG-Premier League"
FBREF_CACHE_DIR = SOCCERDATA_BASE_DIR / "data" / "FBref"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _chrome_binary() -> Path | None:
    for env_name in ("PDH_CHROME_PATH", "CHROME_PATH", "GOOGLE_CHROME_BIN"):
        configured = os.environ.get(env_name)
        if configured and Path(configured).exists():
            return Path(configured)

    candidates = [
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path.home() / "Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
        Path("/Applications/Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"),
    ]
    return next((path for path in candidates if path.exists()), None)


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _FBref(sd.FBref):
    """Project FBref reader with safer SeleniumBase defaults."""

    PLAYER_MATCH_STATS = {
        "summary",
        "passing",
        "passing_types",
        "defense",
        "possession",
        "misc",
        "keepers",
    }

    @classmethod
    def _all_leagues(cls) -> dict[str, str]:
        return sd.FBref._all_leagues()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "_driver"):
            raise RuntimeError(
                "ChromeDriver could not start Chrome for soccerdata. Close any "
                "other running build_historical.py jobs and try again. If FBref "
                "still blocks headless Chrome, rerun with PDH_FBREF_HEADLESS=0."
            )

    def _init_webdriver(self):
        if hasattr(self, "_driver"):
            with suppress(Exception):
                self._driver.quit()

        proxy_str = self.proxy()
        resolver_rules = None
        if proxy_str is not None:
            resolver_rules = "MAP * ~NOTFOUND , EXCLUDE 127.0.0.1"

        browser = str(self.path_to_browser) if self.path_to_browser else None
        debug_port = _free_local_port()
        return Driver(
            uc=True,
            headless=self.headless,
            binary_location=browser,
            host_resolver_rules=resolver_rules,
            proxy=proxy_str,
            chromium_arg=f"--remote-debugging-port={debug_port}",
        )

    def read_player_match_stats(
        self,
        stat_type: str = "summary",
        match_id: str | list[str] | None = None,
        force_cache: bool = False,
    ) -> pd.DataFrame:
        if stat_type not in self.PLAYER_MATCH_STATS:
            raise TypeError(
                f"Invalid argument: stat_type should be in {sorted(self.PLAYER_MATCH_STATS)}"
            )

        df_schedule = self.read_schedule(force_cache).reset_index()
        df_schedule = df_schedule[
            ~df_schedule.game_id.isna() & ~df_schedule.match_report.isnull()
        ]
        if match_id is not None:
            requested = [match_id] if isinstance(match_id, str) else match_id
            iterator = df_schedule[df_schedule.game_id.isin(requested)]
            if len(iterator) == 0:
                raise ValueError("No games found with the given IDs in the selected seasons.")
        else:
            iterator = df_schedule

        stats = []
        for i, game in iterator.reset_index().iterrows():
            url = f"{sd_fbref.FBREF_API}/en/matches/{game['game_id']}"
            sd_fbref.logger.info(
                "[%s/%s] Retrieving game with id=%s",
                i + 1,
                len(iterator),
                game["game_id"],
            )
            filepath = self.data_dir / f"match_{game['game_id']}.html"
            reader = self.get(url, filepath)
            tree = sd_fbref.html.parse(reader)
            home_team, away_team = self._parse_teams(tree)
            id_format = (
                "keeper_stats_{}"
                if stat_type == "keepers"
                else f"stats_{{}}_{stat_type}"
            )

            for team in (home_team, away_team):
                html_table = tree.find(f"//table[@id='{id_format.format(team['id'])}']")
                if html_table is None:
                    sd_fbref.logger.warning(
                        "No %s stats found for %s in game with id=%s",
                        stat_type,
                        team["name"],
                        game["game_id"],
                    )
                    continue

                df_table = sd_fbref._parse_table(html_table)
                df_table["team"] = team["name"]
                df_table["game"] = game["game"]
                df_table["league"] = game["league"]
                df_table["season"] = game["season"]
                df_table["game_id"] = game["game_id"]
                stats.append(df_table)

        if not stats:
            raise ValueError(f"No {stat_type} player match stats found.")

        df = sd_fbref._concat(stats, key=["game"])
        df = df[~df.Player.str.contains(r"^\d+\sPlayers$", na=False)]
        return (
            df.rename(columns={"#": "jersey_number"})
            .replace({"team": sd_fbref.TEAMNAME_REPLACEMENTS})
            .pipe(
                sd_fbref.standardize_colnames,
                cols=["Player", "Nation", "Pos", "Age", "Min"],
            )
            .set_index(["league", "season", "game", "team", "player"])
            .sort_index()
        )


def _fbref_reader(seasons: Iterable[int], leagues: str = FBREF_LEAGUE) -> _FBref:
    browser = _chrome_binary()
    kwargs = {
        "leagues": leagues,
        "seasons": list(seasons),
        "data_dir": FBREF_CACHE_DIR,
        "headless": _env_bool("PDH_FBREF_HEADLESS", True),
    }
    if browser is not None:
        kwargs["path_to_browser"] = browser
    return _FBref(**kwargs)


def fetch_fbref_html_with_browser(url: str, out_path: str | Path | None = None) -> str:
    """
    Fetch FBref HTML using SeleniumBase UC mode as a fallback when soccerdata gets blocked.

    Requires `seleniumbase` to be installed. For CAPTCHA-heavy pages, non-headless mode is
    typically more reliable than headless mode.
    """
    # try:
    #     from seleniumbase import SB
    # except ImportError as e:
    #     raise RuntimeError(
    #         "Browser fallback requires `seleniumbase`. Install it with: pip install seleniumbase"
    #     ) from e

    out_path = Path(out_path) if out_path else None

    # with SB(uc=True, test=False, headless=False, incognito=True) as sb:
    #     sb.uc_open_with_reconnect(url, 4)
    #     try:
    #         sb.sleep(5)  # wait a moment for potential CAPTCHA to load
    #         sb.uc_gui_click_captcha()
    #         sb.sleep(15)
    #     except Exception:
    #         pass
    #     sb.wait_for_element("body", timeout=30)
    #     html = sb.get_page_source()
    headless = _env_bool("PDH_FBREF_HEADLESS", True)
    browser = _chrome_binary()
    driver_kwargs = {
        "uc": True,
        "headless": headless,
        "headless2": headless,
        "incognito": True,
        "chromium_arg": f"--remote-debugging-port={_free_local_port()}",
    }
    if browser is not None:
        driver_kwargs["binary_location"] = str(browser)

    with Driver(**driver_kwargs) as driver:
        # driver.get(url)
        try:
            driver.uc_open_with_reconnect(url, 4)
            driver.uc_gui_click_captcha()
            # driver.wait_for_element("body", timeout=8)
        except Exception as e:
            print(f"[warn] Failed to click CAPTCHA: {e}")
        driver.wait_for_element("body", timeout=8)
        html = driver.get_page_source()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")

    return html


# Known FBref league competition ids used in fallback URLs
FBREF_COMP_IDS = {
    "ENG-Premier League": 9,
}


def _schedule_fallback_url(leagues: str) -> str:
    comp_id = FBREF_COMP_IDS.get(leagues)
    if comp_id is None:
        raise ValueError(
            f"No fallback FBref competition id configured for league: {leagues!r}"
        )
    return f"https://fbref.com/en/comps/{comp_id}/schedule/Premier-League-Scores-and-Fixtures"


def safe_read_schedule(
    seasons: Iterable[int], leagues=FBREF_LEAGUE, cache_dir: str | Path | None = None
) -> pd.DataFrame:
    """
    Try soccerdata first; if FBref blocks the request, fall back to a real browser fetch and
    parse the HTML schedule table with pandas.read_html.
    """
    fb = _fbref_reader(leagues=leagues, seasons=seasons)
    try:
        return fb.read_schedule()
    except Exception as e:
        print(
            f"[warn] soccerdata schedule fetch failed, falling back to browser fetch: {e}"
        )

        url = _schedule_fallback_url(leagues)
        cache_dir = Path(cache_dir) if cache_dir else Path("data/cache/fbref")
        html_path = cache_dir / "schedule.html"
        html = fetch_fbref_html_with_browser(url, out_path=html_path)

        tables = pd.read_html(html)
        if not tables:
            raise RuntimeError(
                "Browser fallback fetched HTML but pandas.read_html found no tables."
            )

        # FBref schedule page usually exposes the actual match table as the first large table.
        schedule_df = tables[0]
        if isinstance(schedule_df.columns, pd.MultiIndex):
            schedule_df = flatten_cols(schedule_df)
        return schedule_df


def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    # If MultiIndex columns (e.g., ('Performance','Gls')), flatten to 'Performance_Gls'
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join([str(x) for x in tup if str(x) != ""]).strip("_")
            for tup in df.columns
        ]
    return df


def player_match_stats(seasons: Iterable[int]) -> pd.DataFrame:
    """
    Return player-match stats by merging summary, passing, passing_types, defense, possession, and misc tables.
    Seasons are end years (e.g., 2023 for 2022/23).
    """
    fb = _fbref_reader(seasons)

    def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
        # If MultiIndex columns (e.g., ('Performance','Gls')), flatten to 'Performance_Gls'
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = [
                "_".join([str(x) for x in tup if str(x) != ""]).strip("_")
                for tup in df.columns
            ]
        return df

    def _read(stat_type: str, name: str) -> tuple[pd.DataFrame, set[str]]:
        # Bring index levels back as columns, then flatten column names
        dfi = fb.read_player_match_stats(stat_type=stat_type)
        if dfi.index.names is not None:
            dfi = dfi.reset_index()
        dfi = _flatten_cols(dfi)

        # Candidate ID columns that often exist across tables
        candidates = [
            "league",
            "competition",
            "season",
            "comp_season",
            "game",
            "game_id",
            "date",
            "team",
            "squad",
            "opponent",
            "player",
            "player_id",
        ]
        keys = [c for c in candidates if c in dfi.columns]

        # Prefix only metric (non-key) columns to avoid name clashes across tables
        metric_cols = [c for c in dfi.columns if c not in keys]
        dfi = dfi[keys + metric_cols]
        dfi = dfi.rename(columns={c: f"{name}_{c}" for c in metric_cols})
        return dfi, set(keys)

    summary, ksum = _read("summary", "summary")
    passing, kpas = _read("passing", "passing")
    ptypes, kpt = _read("passing_types", "passing_types")
    defense, kdef = _read("defense", "defense")
    possession, kpos = _read("possession", "possession")
    misc, kmisc = _read("misc", "misc")
    keepers, kkeep = _read("keepers", "keepers")

    # Compute intersection of keys present in ALL tables
    # common_keys = list(ksum & kpas & kpt & kdef & kpos & kmisc)
    common_keys = list(ksum & kpas & kpt & kdef & kpos & kmisc & kkeep)
    if not common_keys:
        raise ValueError(
            "No common join keys found across FBref tables. Inspect raw frames to see available ID columns."
        )

    # Merge on shared keys
    df = (
        summary.merge(passing, on=common_keys, how="outer")
        .merge(ptypes, on=common_keys, how="outer")
        .merge(defense, on=common_keys, how="outer")
        .merge(possession, on=common_keys, how="outer")
        .merge(misc, on=common_keys, how="outer")
        .merge(keepers, on=common_keys, how="outer")
    )

    return df


def team_season_stats(seasons: Iterable[int]) -> pd.DataFrame:
    fb = _fbref_reader(seasons)
    team_std = fb.read_team_season_stats(stat_type="standard")
    if team_std.index.names is not None:
        team_std = team_std.reset_index()
    team_std = flatten_cols(team_std)
    team_std["season"] = team_std["season"].astype(str)
    return team_std


def schedule(
    seasons: Iterable[int],
    leagues=FBREF_LEAGUE,
    use_safe: bool = False,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    if use_safe:
        return safe_read_schedule(seasons=seasons, leagues=leagues, cache_dir=cache_dir)
    fb = _fbref_reader(leagues=leagues, seasons=seasons)
    return fb.read_schedule()
