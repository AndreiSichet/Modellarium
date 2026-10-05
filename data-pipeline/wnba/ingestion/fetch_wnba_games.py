"""Pull WNBA regular-season games, one CSV per season.

A parallel of the NBA's fetch_games.py, deliberately not a generalisation of
it - see ../README.md for why. Imports RateLimiter and nothing else from the
NBA pipeline.
"""

import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

PIPELINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE / "ingestion"))
from fetch_player_boxscores import RateLimiter  # noqa: E402

from nba_api.stats.endpoints import leaguegamefinder  # noqa: E402

RAW_DIR = PIPELINE / "data" / "wnba" / "raw"

WNBA_LEAGUE_ID = "10"

# A scope decision, not a fact that drifts. History reaches 1997; the 1997
# league had 8 teams and a different game. See ../README.md.
FIRST_SEASON = 2015

# May. The WNBA season sits inside ONE calendar year, so a season's label is
# just its year. This is NOT fetch_games.SEASON_START_MONTH (10) and NOT
# build_rolling_features.derive_season's 8 - different league, and a different
# question from either of those. Reusing an NBA boundary here would be wrong
# by construction: it would call May-September 2026 the "2025-26" season.
SEASON_START_MONTH = 5

# Third digit of the zero-padded 10-digit game id, as the NBA scheme does it
# with a '10' league prefix instead of '00'. Verified by probe: '2' is regular
# season (330 of 330 games in 2026), '4' playoffs, '1' preseason.
REGULAR_SEASON_TYPE_DIGIT = "2"
GAME_ID_WIDTH = 10

REQUESTS_PER_SECOND = 1.0
TIMEOUT = 60
limiter = RateLimiter(REQUESTS_PER_SECOND)


def current_season(today: date = None) -> int:
    """The newest WNBA season that has started, by its May boundary."""
    today = today or date.today()
    return today.year if today.month >= SEASON_START_MONTH else today.year - 1


def open_seasons(today: date = None) -> set:
    """The seasons a refresh still has to ask the source about.

    The current season and the one before it, from the SAME May boundary this
    module already uses - no new season constant.

    THE BUG THIS FIXES RAN THE OTHER WAY ROUND HERE. main() used to skip any
    season whose file already existed, with no exception for the live one. So
    a WNBA season was fetched once, on the first run that saw it, and then
    frozen forever: the daily refresh could never pick up a second day of
    games. It never showed, because the regular season was over on every
    occasion this was exercised and the sync writes nothing either way.
    Completed seasons are still reused; the open ones are now re-asked.
    """
    newest = current_season(today)
    return {str(year) for year in (newest - 1, newest)}


def full_refetch_requested() -> bool:
    """Env-var escape hatch, set only by the drift report."""
    return os.environ.get("MODELLARIUM_FULL_REFETCH") == "1"


def seasons_through(today: date = None) -> list:
    """Every season in scope, derived rather than listed."""
    return [str(year) for year in range(FIRST_SEASON, current_season(today) + 1)]


def regular_season_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only regular-season rows, by the game-id type digit.

    season_type_nullable already asks for this; the digit is checked as well
    because the models are regular-season only by design and a silent
    preseason row would change what every feature means.
    """
    padded = frame["GAME_ID"].astype(str).str.zfill(GAME_ID_WIDTH)
    return frame[padded.str[2] == REGULAR_SEASON_TYPE_DIGIT]


def fetch_season(season: str) -> pd.DataFrame:
    limiter.acquire()
    finder = leaguegamefinder.LeagueGameFinder(
        league_id_nullable=WNBA_LEAGUE_ID,
        season_nullable=season,
        season_type_nullable="Regular Season",
        timeout=TIMEOUT,
    )
    return finder.get_data_frames()[0]


def write_atomically(frame: pd.DataFrame, path: Path) -> None:
    """Write to .partial then rename, so an interrupted run cannot leave a
    truncated CSV that a later run mistakes for a finished one."""
    partial = path.with_suffix(".partial")
    frame.to_csv(partial, index=False, encoding="utf-8")
    partial.replace(path)


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    seasons = seasons_through()
    open_now = open_seasons()
    force = full_refetch_requested()
    print(f"WNBA seasons in scope: {len(seasons)}  "
          f"({seasons[0]} .. {seasons[-1]})")
    print(f"Open seasons (fetched): {', '.join(sorted(open_now))}")
    if force:
        print("MODELLARIUM_FULL_REFETCH is set - re-fetching every season "
              "for the drift report.")
    print(f"Writing to {RAW_DIR}\n")

    fetched, skipped, empty, failed = [], [], [], []
    for season in seasons:
        path = RAW_DIR / f"wnba_games_{season}.csv"

        # A COMPLETED SEASON IS REUSED. An OPEN one is re-fetched even though
        # its file exists, which is the half that was missing: the previous
        # test was presence alone, so a live season stopped updating after
        # its first fetch.
        if not force and season not in open_now and path.exists():
            print(f"  {season}: completed, reusing {path.name}")
            skipped.append(season)
            continue

        print(f"  {season}: fetching...", end=" ", flush=True)
        started = time.monotonic()
        try:
            raw = fetch_season(season)
        except Exception as exc:
            print(f"FAILED {type(exc).__name__}: {exc}")
            failed.append(season)
            continue

        if not len(raw):
            # Not written. An empty header-only CSV would satisfy the resume
            # check forever and be globbed by the validator as a passing
            # season on zero rows.
            print("no games yet - season has not started. Nothing written.")
            empty.append(season)
            continue

        regular = regular_season_only(raw)
        dropped = len(raw) - len(regular)
        write_atomically(regular, path)
        print(f"{len(regular):>3} rows "
              f"({dropped} non-regular dropped) "
              f"[{time.monotonic() - started:.1f}s]")
        fetched.append(season)

    print(f"\nfetched {len(fetched)}, reused {len(skipped)}, "
          f"not started {len(empty)}, failed {len(failed)}")
    if empty:
        print(f"  not started (no file written): {', '.join(empty)}")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
