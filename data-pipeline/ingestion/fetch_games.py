"""Pull raw game data from nba_api's LeagueGameFinder, one season at a time,"""

import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

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

REGULAR_SEASON_GAME_ID_DIGIT = "2"
SCHEDULE_TIMEOUT_SECONDS = 30

# AFTER THIS DATE, "the schedule is not published" STOPS BEING AN EXPLANATION.
#
# A SCOPE DECISION KEPT AS A LITERAL, because it cannot be derived. Measured
# 2026-10-22: ScheduleLeagueV2 exposes 195 columns and NOT ONE of them carries
# a publication, release, updated or version date, so there is nothing in the
# API to compute "when was this schedule published" from. What the API does
# show is that a published season's rows begin in early October (2025-26 opens
# 2025-10-02, 2026-27 opens 2026-10-03, both preseason), and the NBA has
# historically released the full regular-season schedule in mid-to-late
# August. Mid-September is therefore about three weeks clear of the first
# game and a month clear of the usual release.
#
# Why it matters: an unpublished season and a source outage are the SAME
# exception. nba_api's parser raises IndexError from the constructor for both
# - `["gameDates"][0]["games"][0]` on an empty list - so absorbing IndexError
# unconditionally would read a real outage as "the season has not started" and
# accept an empty fetch. The date bound is what closes that: before the
# cutoff an absent schedule is ordinary, after it the only honest reading is
# that we cannot tell, which is a failure.
SCHEDULE_PUBLISHED_BY_MONTH = 9
SCHEDULE_PUBLISHED_BY_DAY = 15


def season_start_year(season: str) -> int:
    """"2026-27" -> 2026."""
    return int(season.split("-")[0])


def schedule_cutoff(season: str) -> "pd.Timestamp":
    """The date after which this season's schedule must exist."""
    return pd.Timestamp(year=season_start_year(season),
                        month=SCHEDULE_PUBLISHED_BY_MONTH,
                        day=SCHEDULE_PUBLISHED_BY_DAY)


def first_regular_season_game(season: str):
    """The earliest regular-season date the schedule shows, or None.

    Raises on anything that is not a clean answer. The caller turns an
    exception into a FAILED season on purpose: an unreadable schedule cannot
    prove a season has not started.
    """
    from nba_api.stats.endpoints import scheduleleaguev2

    # AN UNPUBLISHED SEASON RAISES IndexError FROM THE CONSTRUCTOR, not from
    # get_data_frames() and not from indexing its result. Measured on 2027-28:
    # nba_api's own parser does `["gameDates"][0]["games"][0]` in
    # get_games_headers() while building the object, so the failure happens
    # before any frame exists. The inference service records the same class of
    # behaviour for WNBA 2027 but catches it one level out, which is why this
    # needed measuring rather than copying.
    #
    # That exception is the API answering "nothing is scheduled" - an ordinary
    # state of the world, NOT an unreadable schedule, so it must not become a
    # failure. ONLY IndexError is absorbed; anything else propagates and the
    # caller turns it into a failed season.
    try:
        frames = scheduleleaguev2.ScheduleLeagueV2(
            season=season, league_id="00",
            timeout=SCHEDULE_TIMEOUT_SECONDS).get_data_frames()
    except IndexError:
        return None
    if not frames or frames[0].empty:
        return None

    frame = frames[0]
    padded = frame["gameId"].astype(str).str.zfill(10)
    regular = frame[padded.str[2] == REGULAR_SEASON_GAME_ID_DIGIT]
    if regular.empty:
        return None
    dates = pd.to_datetime(regular["gameDate"], format="%m/%d/%Y %H:%M:%S",
                           errors="coerce")
    if dates.notna().sum() == 0:
        raise RuntimeError(
            f"the schedule for {season} has regular-season rows but no "
            f"parseable gameDate, so the first game date is unknown")
    return dates.min()


def season_not_started(season: str, today=None):
    """(True, why) if an empty fetch is legitimate; (False, why) otherwise."""
    if today is None:
        today = pd.Timestamp.now().normalize()

    try:
        first = first_regular_season_game(season)
    except Exception as error:                                  # noqa: BLE001
        return (False, f"the schedule could not be read "
                       f"({type(error).__name__}: {error}), so it cannot be "
                       f"shown that {season} has not started")

    if first is None:
        cutoff = schedule_cutoff(season)
        if today >= cutoff:
            return (False, f"the schedule publishes no regular-season game "
                           f"for {season}, and it is {today.date()}, past the "
                           f"{cutoff.date()} cutoff by which that schedule "
                           f"should exist - so this is UNREADABLE, not "
                           f"not-started, and an empty fetch cannot be "
                           f"explained by the season being in the future")
        return (True, f"the schedule publishes no regular-season game for "
                      f"{season} yet, and it is {today.date()}, before the "
                      f"{cutoff.date()} cutoff")
    if first > today:
        return (True, f"the schedule's first {season} game is "
                      f"{first.date()}, still ahead of {today.date()}")
    return (False, f"THE SCHEDULE SHOWS {season} STARTED {first.date()} - "
                   f"games have been played and the fetch returned none")


def main() -> int:
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
        #
        # BUT "no games yet" IS ONLY ACCEPTED WHEN THE SCHEDULE AGREES. An
        # empty result is indistinguishable, on its own, from a source that
        # answered with nothing because it was broken - and that silence used
        # to be recorded as a clean skip. The schedule is the second opinion:
        # if its first regular-season game is still ahead, an empty fetch is
        # the truth; if games have already been played, an empty fetch is a
        # failure; and if the schedule cannot be read at all we cannot prove
        # the season has not started, so that is a failure too.
        if games_df.empty:
            verdict, detail = season_not_started(season)
            if verdict is True:
                print(f"  no games yet - {detail}. Nothing written.")
                not_started.append(season)
                time.sleep(DELAY_SECONDS)
                continue
            print(f"  EMPTY RESULT BUT {detail}")
            failures.append(season)
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

    # ZERO TOLERANCE, AND IT IS THE ISOLATION THAT MAKES THAT AFFORDABLE. One
    # failed season is one season of wrong data, and before 2026-10-22 this
    # script printed the list and exited 0, so the daily refresh staged and
    # swapped it. With per-league isolation a failure here means the NBA keeps
    # yesterday's tables and is retried tomorrow, which is the safe outcome -
    # so there is no longer any reason to tolerate a partial fetch.
    if failures:
        print(f"\nFAIL: {len(failures)} season(s) did not fetch. The NBA's "
              f"tables will be carried forward rather than rebuilt from a "
              f"partial corpus.")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
