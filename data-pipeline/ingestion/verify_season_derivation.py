"""Check the derived season list against the hardcoded one it replaced.

The hardcoded list was an expectation with no expiry inside a job that runs
unattended for months: it would have found zero new games every Monday and
reported "nothing worth retraining", which is indistinguishable from a healthy
quiet week. This checks the replacement is exactly equivalent up to the moment
the new season starts, and correct after it.
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_games import (  # noqa: E402
    FIRST_SEASON_START_YEAR,
    current_season_start_year,
    season_label,
    seasons_through,
)

# Verbatim, as it stood before the derivation replaced it.
HARDCODED = [
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]

SEASON_OPENS = date(2026, 10, 20)

failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail:
        print(f"        {detail}")
    if not ok:
        failures.append(name)


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def equivalence_before_the_new_season():
    section("1. IDENTICAL TO THE HARDCODED LIST, UP TO THE NEW SEASON")

    for when in (date(2026, 4, 13), date(2026, 8, 1), date(2026, 9, 27),
                 date(2026, 10, 1) - __import__("datetime").timedelta(days=1)):
        got = seasons_through(when)
        check(f"{when} derives the hardcoded list exactly",
              got == HARDCODED,
              f"{len(got)} seasons, newest {got[-1]}")

    # The last day before the October boundary, which is the tightest case.
    eve = date(2026, 9, 30)
    check(f"{eve} (last day before the October boundary) still ends 2025-26",
          seasons_through(eve)[-1] == "2025-26")


def includes_the_new_season_after_it_starts():
    section("2. PICKS UP 2026-27 ONCE IT EXISTS")

    for when in (date(2026, 10, 1), SEASON_OPENS, date(2026, 11, 3),
                 date(2027, 4, 11)):
        got = seasons_through(when)
        check(f"{when} includes 2026-27 as the newest",
              got[-1] == "2026-27" and "2026-27" in got,
              f"{len(got)} seasons, newest {got[-1]}")

    before, after = seasons_through(date(2026, 9, 30)), seasons_through(SEASON_OPENS)
    check("the new season is APPENDED, nothing else changes",
          after == before + ["2026-27"],
          f"{len(before)} -> {len(after)} seasons, only addition is "
          f"{set(after) - set(before)}")


def the_offseason_negative_test():
    """August 2027 must give 2026-27, not 2027-28.

    This is the case an August boundary gets wrong. build_rolling_features
    .derive_season uses month >= 8 because it maps a GAME DATE to a season and
    never sees a July date; reusing it here would fetch a season that has not
    started.
    """
    section("3. NEGATIVE TEST - THE OFFSEASON MUST NOT INVENT A SEASON")

    for when in (date(2027, 7, 1), date(2027, 8, 15), date(2027, 9, 30)):
        got = seasons_through(when)
        ok = got[-1] == "2026-27"
        check(f"{when} (offseason) newest is 2026-27, not 2027-28",
              ok, f"got newest {got[-1]}")

    august = seasons_through(date(2027, 8, 15))
    check("2027-28 is absent entirely during its own offseason",
          "2027-28" not in august)

    # What the wrong boundary would have produced, stated so the distinction is
    # on the record rather than implied.
    august_boundary = 2027 if 8 >= 8 else 2026
    check("an August boundary WOULD have been wrong here",
          season_label(august_boundary) == "2027-28",
          "month >= 8 yields 2027-28 at 2027-08-15 - a season with no games")


def structure_and_edges():
    section("4. STRUCTURE")

    got = seasons_through(date(2026, 11, 3))
    check("starts at the fixed scope decision, 2015-16",
          got[0] == season_label(FIRST_SEASON_START_YEAR) == "2015-16")
    check("contiguous, no gaps and no duplicates",
          got == [season_label(y) for y in
                  range(FIRST_SEASON_START_YEAR,
                        FIRST_SEASON_START_YEAR + len(got))]
          and len(set(got)) == len(got),
          f"{got[0]} .. {got[-1]}, {len(got)} entries")

    check("label format handles the century roll",
          season_label(1999) == "1999-00" and season_label(2009) == "2009-10",
          f"1999 -> {season_label(1999)}, 2009 -> {season_label(2009)}")

    check("boundary is October, checked on both sides of it",
          current_season_start_year(date(2026, 9, 30)) == 2025
          and current_season_start_year(date(2026, 10, 1)) == 2026)


def main() -> int:
    print(f"Today is {date.today()}; live derivation gives "
          f"{len(seasons_through())} seasons, newest {seasons_through()[-1]}.")

    equivalence_before_the_new_season()
    includes_the_new_season_after_it_starts()
    the_offseason_negative_test()
    structure_and_edges()

    section("RESULT")
    if failures:
        print(f"  {len(failures)} FAILED: {failures}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
