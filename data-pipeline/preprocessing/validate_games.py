"""Read-only quality gate for the raw game data in data-pipeline/data/raw/.

IT VALIDATES THE RAW PER-SEASON FILES, NOT THE PROCESSED TABLE, AND THAT IS
DELIBERATE. Applying these same checks to `games_final.csv` sliced by its
`SEASON` column fails 2020-21 with "160 GAME_DATE values outside expected
season range", every time. The cause is not bad data: the 2019-20 COVID
bubble's 80 regular-season games of 2020-08-01 to 2020-08-14 are labelled
2020-21 by `build_rolling_features.derive_season`, which starts a season at
`month >= 8`, while `season_date_bounds` below starts it on September 1. In
the raw files those games sit in `games_2019-20.csv`, whose bounds run to
2020-08-31, so they pass. The GAME_ID is the authority - `0021900001` is
2019-20 - and moving SEASON onto it belongs to the drop-and-retrain spec,
because it shifts Elo on 14,612 rows by up to 66 points. Until then: raw
files only. Measured 2026-10-22.

EXIT CODE IS A CONTRACT. Until 2026-10-22 this script printed `RESULT: FAIL`
and exited 0, so the daily refresh's NBA validation step could not fail - and
`run_step` discards a non-fetch step's stdout, so the message was invisible
too. It now exits 1 on any failed season, and on a season that is in scope
with no file at all.
"""

import re
import sys
from pathlib import Path

import pandas as pd

from build_rolling_features import season_from_game_id

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
INGESTION_DIR = Path(__file__).resolve().parents[1] / "ingestion"
if str(INGESTION_DIR) not in sys.path:
    sys.path.insert(0, str(INGESTION_DIR))

EXIT_OK = 0
EXIT_FAILED = 1

CRITICAL_COLUMNS = ["TEAM_ID", "GAME_ID", "GAME_DATE", "WL", "PTS"]
VALID_WL_VALUES = {"W", "L"}

SEASON_FILE_RE = re.compile(r"games_(\d{4}-\d{2})\.csv$")

def find_season_files():
    files = sorted(RAW_DATA_DIR.glob("games_*.csv"))
    seasons = []
    for f in files:
        match = SEASON_FILE_RE.search(f.name)
        if match:
            seasons.append((match.group(1), f))
    return seasons

def season_date_bounds(season: str):
    """The calendar window a season's games may fall in.

    SEPTEMBER TO AUGUST, which is deliberately wider than a normal season and
    is why the 2019-20 bubble's August 2020 games pass here: they are in
    `games_2019-20.csv`, and that file's window runs to 2020-08-31.

    This is a SANITY check on the dates, not the season rule. The rule is the
    GAME_ID - see `season_matches_game_id` below, and
    `build_rolling_features.season_from_game_id`, which is the one definition
    both this validator and the feature builders now read.
    """
    start_year = int(season[:4])
    end_year = start_year + 1
    lower = pd.Timestamp(f"{start_year}-09-01")
    upper = pd.Timestamp(f"{end_year}-08-31")
    return lower, upper


def season_matches_game_id(season: str, df: pd.DataFrame) -> int:
    """How many rows the GAME_ID says belong to a DIFFERENT season.

    THE RAW FILES BECOME SELF-CHECKING. `games_2019-20.csv` should contain
    only games whose id starts the 2019 season, and the id is the authority
    for that - the dates cannot settle it, which is the whole lesson of the
    bubble. A non-zero count here means a game is filed under the wrong
    season, which no date window can detect.

    Imported rather than restated: one definition, two callers.
    """
    if "GAME_ID" not in df.columns or "GAME_DATE" not in df.columns:
        return 0
    dates = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    by_id = season_from_game_id(df["GAME_ID"], dates)
    expected = int(season[:4])
    return int((by_id.notna() & (by_id != expected)).sum())

def validate_file(season: str, path: Path, reference_columns):
    issues = []
    df = pd.read_csv(path)

    row_count = len(df)
    col_count = len(df.columns)

    schema_ok = True
    if reference_columns is not None and list(df.columns) != list(reference_columns):
        schema_ok = False
        missing = set(reference_columns) - set(df.columns)
        extra = set(df.columns) - set(reference_columns)
        issues.append(
            f"schema mismatch (missing={sorted(missing)}, extra={sorted(extra)})"
        )

    missing_counts = {}
    for col in CRITICAL_COLUMNS:
        if col in df.columns:
            missing_counts[col] = int(df[col].isna().sum())
            if missing_counts[col] > 0:
                issues.append(f"{missing_counts[col]} missing values in {col}")
        else:
            missing_counts[col] = None
            issues.append(f"critical column {col} not present")

    date_parse_failures = 0
    date_out_of_range = 0
    if "GAME_DATE" in df.columns:
        parsed_dates = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        date_parse_failures = int(parsed_dates.isna().sum() - df["GAME_DATE"].isna().sum())
        lower, upper = season_date_bounds(season)
        in_range = parsed_dates.between(lower, upper)
        date_out_of_range = int((~in_range & parsed_dates.notna()).sum())
        if date_parse_failures > 0:
            issues.append(f"{date_parse_failures} GAME_DATE values failed to parse")
        if date_out_of_range > 0:
            issues.append(f"{date_out_of_range} GAME_DATE values outside expected season range")

        # THE GAME_ID IS THE AUTHORITY on which season a game belongs to, and
        # this is the check the date window cannot make.
        filed_wrong = season_matches_game_id(season, df)
        if filed_wrong > 0:
            issues.append(
                f"{filed_wrong} row(s) whose GAME_ID belongs to a different "
                f"season than {season}")

    invalid_wl_count = 0
    if "WL" in df.columns:
        invalid_wl = df.loc[df["WL"].notna() & ~df["WL"].isin(VALID_WL_VALUES)]
        invalid_wl_count = len(invalid_wl)
        if invalid_wl_count > 0:
            issues.append(f"{invalid_wl_count} WL values outside {{'W', 'L'}}")

    bad_game_id_count = 0
    if "GAME_ID" in df.columns:
        game_id_counts = df["GAME_ID"].value_counts()
        bad_game_ids = game_id_counts[game_id_counts != 2]
        bad_game_id_count = len(bad_game_ids)
        if bad_game_id_count > 0:
            issues.append(f"{bad_game_id_count} GAME_IDs do not appear exactly twice")

    duplicate_row_count = int(df.duplicated().sum())
    if duplicate_row_count > 0:
        issues.append(f"{duplicate_row_count} exact duplicate rows")

    passed = len(issues) == 0

    print(f"\nSeason {season} ({path.name})")
    print(f"  rows: {row_count}, columns: {col_count}, schema_ok: {schema_ok}")
    print(f"  missing values (critical columns): {missing_counts}")
    print(f"  GAME_DATE parse failures: {date_parse_failures}, out-of-range: {date_out_of_range}")
    print(f"  invalid WL values: {invalid_wl_count}")
    print(f"  GAME_IDs not paired exactly twice: {bad_game_id_count}")
    print(f"  exact duplicate rows: {duplicate_row_count}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}" + (f" ({'; '.join(issues)})" if issues else ""))

    return passed, list(df.columns)

def seasons_in_scope() -> list:
    """The seasons the fetcher is responsible for, DERIVED not listed.

    Read from `fetch_games.seasons_through()` so the two cannot disagree: the
    question "is a season missing" is only meaningful against the same scope
    the fetcher works from, and a second copy of that list is how they drift.
    """
    from fetch_games import seasons_through

    return seasons_through()


def main() -> int:
    season_files = find_season_files()
    on_disk = {season for season, _ in season_files}

    try:
        scope = seasons_in_scope()
    except Exception as error:                                  # noqa: BLE001
        # Refusing rather than guessing: without the scope there is no way to
        # tell a missing season from a season that does not exist yet.
        print(f"FAIL: cannot derive the seasons in scope ({type(error).__name__}:"
              f" {error}). Refusing to validate against an unknown scope.")
        return EXIT_FAILED

    absent = [season for season in scope if season not in on_disk]

    if not season_files:
        # STILL A FAILURE, and it used to be a silent success. An empty raw
        # directory means the fetch produced nothing at all.
        print(f"FAIL: no games_<season>.csv files found in {RAW_DATA_DIR}, "
              f"but {len(scope)} season(s) are in scope.")
        return EXIT_FAILED

    reference_columns = None
    results = []

    for season, path in season_files:
        passed, columns = validate_file(season, path, reference_columns)
        if reference_columns is None:
            reference_columns = columns
        results.append((season, passed))

    print("\n" + "=" * 40)
    failed = [season for season, passed in results if not passed]
    if failed:
        print(f"{len(results) - len(failed)}/{len(results)} seasons passed validation.")
        print(f"Failed seasons: {', '.join(failed)}")
    else:
        print(f"All {len(results)} seasons passed validation.")

    print(f"Seasons in scope: {len(scope)} ({scope[0]} .. {scope[-1]}), "
          f"on disk: {len(on_disk)}")
    if absent:
        # A SEASON IN SCOPE WITH NO FILE IS A FAILURE. find_season_files()
        # only sees what exists, so before this check a missing current season
        # was invisible: the script reported "All 11 seasons passed" with the
        # season in progress simply not there.
        print(f"FAIL: in scope but no file: {', '.join(absent)}")

    if failed or absent:
        return EXIT_FAILED
    return EXIT_OK

if __name__ == "__main__":
    sys.exit(main())
