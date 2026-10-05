"""Pull raw game data from nba_api's LeagueGameFinder, one season at a time,"""

import os
import time
from datetime import date
from pathlib import Path

from nba_api.stats.endpoints import leaguegamefinder

# A deliberate scope decision, not a fact that drifts: 2015-16 is where this
# project chose to start. The END of the range is derived, because a hardcoded
# one is an expectation with no expiry inside a job that runs unattended for
# months - it would find zero new games every week and report "nothing worth
# retraining", which is indistinguishable from a healthy quiet week.
FIRST_SEASON_START_YEAR = 2015

# NBA seasons open in October. This is NOT build_rolling_features.derive_season,
# which uses an August boundary, and the two must not be harmonised: that one
# maps a GAME DATE to its season and never sees a July date because no games are
# played then, so August is a safe placeholder. This one maps TODAY to "which
# seasons exist", and at August 2027 an August boundary would yield 2027-28 - a
# season that has not started and has no games to fetch.
SEASON_START_MONTH = 10

DELAY_SECONDS = 1
RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def season_label(start_year: int) -> str:
    """2015 -> '2015-16'. Handles the century roll: 1999 -> '1999-00'."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def current_season_start_year(today: date = None) -> int:
    """The newest season that has STARTED, by its October boundary."""
    today = today or date.today()
    return today.year if today.month >= SEASON_START_MONTH else today.year - 1


def open_seasons(today: date = None) -> set:
    """The seasons a refresh still has to ask the source about.

    The current season and the one before it - derived from the SAME October
    boundary as everything else here, so this adds no new season constant to
    a project that already has five answering five different questions.

    WHY ONLY THESE. The daily refresh used to re-fetch all eleven seasons
    every morning, which meant a season from 2015 was rebuilt daily from a
    source free to answer differently - and the G League's equivalent did
    answer differently, failing the first scheduled run. Completed seasons
    are finished: reusing the file already on disk is what makes historical
    rows genuinely fixed between snapshots.

    The PREVIOUS season is included because late corrections happen and
    because the boundary itself has to be crossed safely - on 2026-10-01 the
    current season has no games yet, so the one before it is still the live
    table.
    """
    newest = current_season_start_year(today)
    return {season_label(year) for year in (newest - 1, newest)}


def full_refetch_requested() -> bool:
    """Env-var escape hatch, set only by the drift report.

    An environment variable rather than a flag because the daily refresh runs
    these scripts as subprocesses by path, so there is no argument list to
    thread an option through.
    """
    return os.environ.get("MODELLARIUM_FULL_REFETCH") == "1"


def seasons_through(today: date = None) -> list:
    """Every season from the fixed start through the one currently underway."""
    newest = current_season_start_year(today)
    return [season_label(y) for y in range(FIRST_SEASON_START_YEAR, newest + 1)]

def fetch_season(season: str):
    finder = leaguegamefinder.LeagueGameFinder(
        # Without season_type_nullable the pull mixes in preseason,
        # playoff, All-Star and NBA Cup games.
        season_nullable=season, season_type_nullable="Regular Season"
    )
    return finder.get_data_frames()[0]

def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    seasons = seasons_through()
    open_now = open_seasons()
    force = full_refetch_requested()
    print(f"Open seasons (fetched): {', '.join(sorted(open_now))}")
    if force:
        print("MODELLARIUM_FULL_REFETCH is set - re-fetching every season "
              "for the drift report.")
    print(f"Seasons derived from today's date: {len(seasons)}, "
          f"{seasons[0]} through {seasons[-1]}.\n")

    successes = 0
    failures = []
    not_started = []
    reused = []

    for season in seasons:
        out_path = RAW_DATA_DIR / f"games_{season}.csv"

        # A COMPLETED SEASON IS REUSED, NOT RE-FETCHED. Fetched anyway when
        # its file is absent, so a cold checkout or an orphaned corpus still
        # builds from scratch - the same resume-by-existence discipline the
        # box-score and quarter-score fetchers already use.
        if not force and season not in open_now and out_path.exists():
            print(f"Season {season}: completed, reusing {out_path.name}")
            reused.append(season)
            continue

        print(f"Fetching season {season}...")

        try:
            games_df = fetch_season(season)
        except Exception as exc:
            print(f"  FAILED: {season} -> {exc}")
            failures.append(season)
            time.sleep(DELAY_SECONDS)
            continue

        # A season derived from the calendar can legitimately have no games
        # yet: October 1st is inside the season by the boundary above, but
        # opening night is around the 20th. Writing a header-only file would
        # make validate_games.py report an empty season as PASSED, which is a
        # vacuous pass of exactly the kind this project keeps removing.
        if games_df.empty:
            print(f"  no games yet - season has not started. Nothing written.")
            not_started.append(season)
            time.sleep(DELAY_SECONDS)
            continue

        games_df.to_csv(out_path, index=False)

        print(f"  OK: {len(games_df)} rows -> {out_path.name}")
        successes += 1

        time.sleep(DELAY_SECONDS)

    print(f"\nDone. {successes} season(s) fetched, {len(reused)} reused unchanged, of {len(seasons)} in scope.")
    if not_started:
        print(f"Not started yet (no file written): {', '.join(not_started)}")
    if failures:
        print(f"Failed seasons: {', '.join(failures)}")

if __name__ == "__main__":
    main()
