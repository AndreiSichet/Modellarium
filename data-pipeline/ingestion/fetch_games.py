"""Pull raw game data from nba_api's LeagueGameFinder, one season at a time,"""

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
    print(f"Seasons derived from today's date: {len(seasons)}, "
          f"{seasons[0]} through {seasons[-1]}.\n")

    successes = 0
    failures = []
    not_started = []

    for season in seasons:
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

        out_path = RAW_DATA_DIR / f"games_{season}.csv"
        games_df.to_csv(out_path, index=False)

        print(f"  OK: {len(games_df)} rows -> {out_path.name}")
        successes += 1

        time.sleep(DELAY_SECONDS)

    print(f"\nDone. {successes}/{len(seasons)} seasons saved.")
    if not_started:
        print(f"Not started yet (no file written): {', '.join(not_started)}")
    if failures:
        print(f"Failed seasons: {', '.join(failures)}")

if __name__ == "__main__":
    main()
