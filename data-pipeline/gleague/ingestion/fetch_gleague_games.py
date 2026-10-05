"""Pull G League games, one CSV per season per competition.

A third parallel of the NBA's fetch_games.py, deliberately not a
generalisation of it or of the WNBA's - see ../README.md, which records that
deferral with its reason and its date. Imports RateLimiter and nothing else
from either existing pipeline.

TWO COMPETITIONS ARE FETCHED, NOT ONE. The G League plays a Showcase Cup
before its regular season, and the probe established that those games carry
type digit '5' - which in the NBA's scheme means the play-in. The digit
meanings do not transfer between leagues even though the scheme does. The Cup
games are kept, in their own file, because a team's last game before its
regular season opener is usually a Cup game and phase 2 needs that date to
compute rest correctly.
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

RAW_DIR = PIPELINE / "data" / "gleague" / "raw"

GLEAGUE_LEAGUE_ID = "20"

# NOT a scope decision like the WNBA's 2015 - this is where the source stops.
# The NBDL began in 2001-02, and 2001-02 and 2002-03 both return zero rows;
# 2003-04 is the first season served. Probed season by season rather than
# inferred, because an empty response for a season that was actually played is
# a property of the API and not an absence of a league.
FIRST_SEASON_START_YEAR = 2003

# NOVEMBER, and it is the FOURTH season boundary in this project. The other
# three answer different questions and must not be unified with this one
# because they look alike:
#
#   build_rolling_features.derive_season   month >= 8   a GAME DATE -> its NBA season
#   fetch_games.SEASON_START_MONTH         10           today -> which NBA seasons exist
#   fetch_wnba_games.SEASON_START_MONTH    5            today -> which WNBA seasons exist
#   this                                   11           today -> which G League seasons exist
#
# The G League spans two calendar years like the NBA, so its label is the
# hyphenated form - but it tips off in November, not October.
SEASON_START_MONTH = 11

GAME_ID_WIDTH = 10

# Third digit of the zero-padded 10-digit game id. '2' is the regular season
# in every league using this scheme. '5' is the play-in for the NBA and the
# SHOWCASE CUP for the G League - established by arithmetic rather than by the
# label: 31 teams x 14 games / 2 = 217, exactly the count returned, and the
# Cup's dates fall before the regular season rather than after it.
REGULAR_SEASON_TYPE_DIGIT = "2"
SHOWCASE_CUP_TYPE_DIGIT = "5"

COMPETITIONS = {
    "regular": REGULAR_SEASON_TYPE_DIGIT,
    "showcase": SHOWCASE_CUP_TYPE_DIGIT,
}

REQUESTS_PER_SECOND = 1.0
TIMEOUT = 60
limiter = RateLimiter(REQUESTS_PER_SECOND)


def current_season_start_year(today: date = None) -> int:
    """Start year of the newest G League season under way, by its November
    boundary."""
    today = today or date.today()
    return today.year if today.month >= SEASON_START_MONTH else today.year - 1


def season_label(start_year: int) -> str:
    """2025 -> '2025-26'. Handles the century roll: 1999 -> '1999-00'."""
    return f"{start_year}-{str(start_year + 1)[-2:].zfill(2)}"


def open_seasons(today: date = None) -> set:
    """The seasons a refresh still has to ask the source about.

    The current season and the one before it, from the SAME November boundary
    this module already uses - no new season constant.

    THIS IS THE LEAGUE THAT FAILED THE FIRST SCHEDULED REFRESH, and the
    reason is below in main(): the presence test required BOTH competition
    files, and the Showcase Cup exists for only 5 of 23 seasons, so 18
    completed seasons were re-fetched every single morning. 481 seconds of a
    21-minute run, and one of those re-fetches came back with its rows in a
    different order, which flipped a game into looking self-contradicting.
    """
    newest = current_season_start_year(today)
    return {season_label(year) for year in (newest - 1, newest)}


def full_refetch_requested() -> bool:
    """Env-var escape hatch, set only by the drift report."""
    return os.environ.get("MODELLARIUM_FULL_REFETCH") == "1"


def seasons_through(today: date = None) -> list:
    """Every season in scope, derived from today rather than listed."""
    newest = current_season_start_year(today)
    return [season_label(y)
            for y in range(FIRST_SEASON_START_YEAR, newest + 1)]


def type_digit(frame: pd.DataFrame) -> pd.Series:
    padded = frame["GAME_ID"].astype(str).str.zfill(GAME_ID_WIDTH)
    return padded.str[2]


def by_digit(frame: pd.DataFrame, digit: str) -> pd.DataFrame:
    return frame[type_digit(frame) == digit]


def fetch_all_types(season: str) -> pd.DataFrame:
    """Every competition in one season, unfiltered."""
    limiter.acquire()
    return leaguegamefinder.LeagueGameFinder(
        league_id_nullable=GLEAGUE_LEAGUE_ID,
        season_nullable=season,
        timeout=TIMEOUT,
    ).get_data_frames()[0]


def fetch_regular_season(season: str) -> pd.DataFrame:
    """What the API itself calls the regular season."""
    limiter.acquire()
    return leaguegamefinder.LeagueGameFinder(
        league_id_nullable=GLEAGUE_LEAGUE_ID,
        season_nullable=season,
        season_type_nullable="Regular Season",
        timeout=TIMEOUT,
    ).get_data_frames()[0]


def assert_filters_agree(season: str, all_types: pd.DataFrame,
                         api_regular: pd.DataFrame) -> None:
    """The API's own season-type filter and the id digit must select the same
    games.

    Two independent routes to the same set. If they ever disagree, one of them
    is admitting something the other excludes and neither can be trusted
    without knowing which - so this raises rather than preferring one.
    """
    by_id = set(by_digit(all_types, REGULAR_SEASON_TYPE_DIGIT)["GAME_ID"])
    by_api = set(api_regular["GAME_ID"].astype(str))
    by_id = {str(g) for g in by_id}

    if by_id != by_api:
        only_digit = sorted(by_id - by_api)[:6]
        only_api = sorted(by_api - by_id)[:6]
        raise SystemExit(
            f"{season}: the two regular-season filters disagree.\n"
            f"  digit '2' only: {len(by_id - by_api)} game(s) {only_digit}\n"
            f"  API filter only: {len(by_api - by_id)} game(s) {only_api}\n"
            f"Neither filter can be trusted until it is known which is wrong."
        )


def write_atomically(frame: pd.DataFrame, path: Path) -> None:
    """Write to .partial then rename, so an interrupted run cannot leave a
    truncated CSV that a later run mistakes for a finished one."""
    partial = path.with_suffix(".partial")
    frame.to_csv(partial, index=False, encoding="utf-8")
    partial.replace(path)


def paths_for(season: str) -> dict:
    return {name: RAW_DIR / f"gleague_{name}_{season}.csv"
            for name in COMPETITIONS}


def main() -> int:
    print(__doc__)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    seasons = seasons_through()
    open_now = open_seasons()
    force = full_refetch_requested()
    print(f"G League seasons in scope: {len(seasons)}  "
          f"({seasons[0]} .. {seasons[-1]})")
    print(f"Open seasons (fetched): {', '.join(sorted(open_now))}")
    if force:
        print("MODELLARIUM_FULL_REFETCH is set - re-fetching every season "
              "for the drift report.")
    print(f"Writing to {RAW_DIR}\n")

    fetched, skipped, empty, failed = [], [], [], []
    totals = {name: 0 for name in COMPETITIONS}

    for season in seasons:
        paths = paths_for(season)

        # PRESENCE IS THE REGULAR-SEASON FILE, NOT BOTH FILES, and that
        # distinction is the whole bug. The Showcase Cup began in 2021-22, so
        # `all(p.exists())` is false for all 18 pre-Cup seasons however many
        # times they are fetched - a condition that can never be satisfied,
        # re-fetching two decades of finished basketball every morning.
        # The Cup's absence is normal for a pre-Cup season, so it cannot
        # stand for "this season is incomplete"; the regular-season file can.
        if not force and season not in open_now and paths["regular"].exists():
            print(f"  {season}: completed, reusing {paths['regular'].name}")
            skipped.append(season)
            continue

        print(f"  {season}: fetching...", end=" ", flush=True)
        started = time.monotonic()
        try:
            all_types = fetch_all_types(season)
            api_regular = fetch_regular_season(season)
        except Exception as exc:
            print(f"FAILED {type(exc).__name__}: {exc}")
            failed.append(season)
            continue

        if not len(all_types):
            # Not written. An empty header-only CSV would satisfy the resume
            # check forever and be globbed by the validator as a passing
            # season on zero rows - the vacuous-pass shape this project has
            # caught before.
            print("0 rows - not served for this season. Nothing written.")
            empty.append(season)
            continue

        assert_filters_agree(season, all_types, api_regular)

        counts = []
        for name, digit in COMPETITIONS.items():
            subset = by_digit(all_types, digit)
            if len(subset):
                write_atomically(subset, paths[name])
            counts.append(f"{name} {len(subset)}")
            totals[name] += len(subset)

        other = len(all_types) - sum(
            len(by_digit(all_types, d)) for d in COMPETITIONS.values())
        print(f"{', '.join(counts)} rows"
              f"{f', {other} other-type dropped' if other else ''} "
              f"[{time.monotonic() - started:.1f}s]")
        fetched.append(season)

    print(f"\nfetched {len(fetched)}, reused {len(skipped)}, "
          f"not served {len(empty)}, failed {len(failed)}")
    for name, n in totals.items():
        print(f"  {name:<9} {n:,} rows this run")
    if empty:
        print(f"  not served by the source (no file written): "
              f"{', '.join(empty)}")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
