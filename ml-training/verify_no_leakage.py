r"""No feature may be computable only after the game has been played.

THE TRUNCATION TEST. For a sampled game, rebuild every feature from history
cut off strictly before that game's date, through the SERVING path, and
compare with the row the training pipeline stored. A feature that needs the
game's own result cannot be reproduced that way, and that is the whole test.

WHY THIS EXISTS. The NBA team models carry four availability features -
`{HOME,AWAY}_ABSENT_COUNT` and `{HOME,AWAY}_WEIGHTED_ABSENT_MIN` - built from
the game's own box score: a player counts as absent when he recorded no
minutes in THAT game, which is knowable only after tip-off. Serving cannot
know it, so it substitutes the injury report, which counts a different
population. Measured on 2026-10-22, that skew costs 12% on spread - worse than
having no availability features at all.

**AND A TEST THAT WOULD HAVE CAUGHT IT ALREADY EXISTED.**
`live_features.verify_against_training_data` replays 200 historical games
against the pipeline's own columns, strictly truncated, to 1e-9 - and it
excludes these four by name, printing "Not replayable (live-only, no
historical equivalent)". That description is backwards. They have a historical
equivalent; it is built from the future. The exclusion is why 34 of 38
features were audited for a year and the 4 that leak were not.

So the four are this suite's POSITIVE CONTROL. They must FAIL. If they pass,
the test is broken and says so rather than reporting a clean bill of health.

    python ml-training/verify_no_leakage.py
    python ml-training/verify_no_leakage.py --plant        # must go red
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for extra in (HERE, HERE / "nfl", HERE / "wnba", HERE / "gleague"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

TOLERANCE = 1e-9
RESULTS = []


class HarnessFailure(RuntimeError):
    """The suite could not set up the comparison it meant to make."""


def record(name, passed, detail):
    RESULTS.append((name, passed, detail))
    print("  %-4s %-54s %s" % ("PASS" if passed else "FAIL", name, detail))


# ==========================================================================
# the sample, derived
# ==========================================================================

def completed_seasons(frame: pd.DataFrame, season_col="SEASON") -> list:
    """Seasons with a full-looking complement of games, newest last.

    DERIVED, not listed. A season still being played has far fewer games than
    a finished one, and comparing each season's count against the MEDIAN
    season's count separates the two without naming a league's schedule
    length - which would be a figure that expires the next time a league
    changes its calendar or adds a team.
    """
    counts = frame.groupby(season_col).size()
    if counts.empty:
        raise HarnessFailure("no seasons in the table")
    threshold = 0.8 * counts.median()
    return sorted(int(s) for s, n in counts.items() if n >= threshold)


def sample_dates(frame: pd.DataFrame, seasons, date_col="GAME_DATE") -> list:
    """Early, middle and late game-days from each season given.

    THREE POINTS IN THE SEASON BECAUSE THE WARM-UP MATTERS. Early games have
    incomplete rolling windows and are where a NaN boundary can hide a
    difference; late games have the longest history behind them and are where
    an off-by-one in a window shows up. Taking only mid-season games would
    miss both.
    """
    out = []
    for season in seasons:
        days = sorted(frame.loc[frame["SEASON"] == season, date_col].unique())
        if not days:
            continue
        for label, position in (("early", 0.10), ("mid", 0.50),
                                ("late", 0.90)):
            out.append((season, label, days[int((len(days) - 1) * position)]))
    return out


def games_on(frame: pd.DataFrame, day, limit: int) -> pd.DataFrame:
    return frame[frame["GAME_DATE"] == day].head(limit)


# ==========================================================================
# the comparison
# ==========================================================================

def compare_row(got: dict, want, columns, game_id, mismatches: list) -> int:
    """Compare one rebuilt row against the stored one. Returns values checked."""
    checked = 0
    for column in columns:
        a, b = got.get(column), want[column]
        checked += 1
        a_nan = a is None or (isinstance(a, float) and np.isnan(a)) or pd.isna(a)
        b_nan = pd.isna(b)
        if a_nan and b_nan:
            continue
        if a_nan != b_nan:
            mismatches.append((game_id, column, a, b, "NaN on one side only"))
            continue
        if not np.isclose(float(a), float(b), rtol=TOLERANCE, atol=TOLERANCE):
            mismatches.append((game_id, column, float(a), float(b),
                               f"delta {abs(float(a) - float(b)):.3e}"))
    return checked


def report_family(name, columns, checked, games, mismatches,
                  expected_failures=()) -> None:
    """One family's verdict, separating expected failures from findings.

    THE SEPARATION IS THE POINT. The availability features are KNOWN to leak;
    counting them as failures would bury any new finding in noise, and
    excluding them would repeat the mistake this suite exists to correct. They
    are named, required to fail, and reported apart from everything else.
    """
    by_column = {}
    for _gid, column, _a, _b, _why in mismatches:
        by_column[column] = by_column.get(column, 0) + 1

    expected = {c: n for c, n in by_column.items() if c in expected_failures}
    findings = {c: n for c, n in by_column.items() if c not in expected_failures}

    print(f"\n  {name}: {len(columns)} features x {games} games "
          f"= {checked} values")

    if expected_failures:
        missing = [c for c in expected_failures if c not in expected]
        record(f"{name}: the known-leaking features DO fail (positive control)",
               not missing,
               f"{sorted(expected)} failed"
               if not missing else
               f"THESE PASSED AND MUST NOT: {missing} - the test is broken")

    record(f"{name}: no OTHER feature fails the truncation test",
           not findings,
           "clean" if not findings else
           f"{len(findings)} column(s): " + ", ".join(
               f"{c} x{n}" for c, n in sorted(findings.items())[:6]))

    if findings:
        print("    FINDINGS - first 6 values:")
        for gid, column, a, b, why in mismatches[:40]:
            if column in findings:
                print(f"      game {gid} {column}: served {a}, "
                      f"training {b} ({why})")


# ==========================================================================
# NBA team models - 38 features, and the positive control
# ==========================================================================

AVAILABILITY_COLUMNS = (
    "HOME_ABSENT_COUNT", "AWAY_ABSENT_COUNT",
    "HOME_WEIGHTED_ABSENT_MIN", "AWAY_WEIGHTED_ABSENT_MIN",
)


def check_nba_team(per_day: int, plant: bool) -> None:
    from common import FEATURE_COLUMNS
    from live_features import get_live_features, load_games_final
    from train_baseline import load_dataset

    dataset = load_dataset()
    games_final = load_games_final()

    seasons = completed_seasons(dataset)[-2:]
    sample = sample_dates(dataset, seasons)

    columns = list(FEATURE_COLUMNS)
    # THE EXPECTED-FAILURE SET IS NOW EMPTY, AND A NON-EMPTY ONE IS FATAL.
    # Before A3 the four availability columns were this suite's positive
    # control: they HAD to fail, because they were built from the game's own
    # box score. They have been dropped from the feature contract, so nothing
    # in it may fail any more - and if one of them is still present, the drop
    # did not take, which is a worse problem than a failing check.
    expected = tuple(c for c in AVAILABILITY_COLUMNS if c in columns)
    if expected:
        raise HarnessFailure(
            f"these were dropped by A3 and are still in FEATURE_COLUMNS: "
            f"{sorted(expected)}. The feature contract and this suite "
            f"disagree about what shipped.")
    if plant:
        # THE PLANT: a column that is pure post-game information - the home
        # team's actual points - added to the feature list. Serving cannot
        # produce it, so the suite must report it as a finding rather than as
        # an expected failure.
        columns = columns + ["HOME_PTS"]
        if "HOME_PTS" not in dataset.columns:
            raise HarnessFailure("HOME_PTS is not in the dataset; the plant "
                                 "would have tested nothing")

    mismatches, checked, games = [], 0, 0
    for _season, _label, day in sample:
        for _i, row in games_on(dataset, day, per_day).iterrows():
            served = get_live_features(
                row["HOME_TEAM_ID"], row["AWAY_TEAM_ID"], row["GAME_DATE"],
                games_final).iloc[0].to_dict()
            if plant:
                # Serving has no such feature at all, which is exactly what a
                # post-game column looks like from the serving side.
                served.setdefault("HOME_PTS", float("nan"))
            checked += compare_row(served, row, columns, row["GAME_ID"],
                                   mismatches)
            games += 1

    print(f"\n  sample: seasons {seasons}, "
          f"{len(sample)} game-days (early/mid/late x {len(seasons)}), "
          f"{games} games")
    report_family("NBA team", columns, checked, games, mismatches, expected)


# ==========================================================================
# NBA quarter / half models - 30 features
# ==========================================================================

def check_nba_quarter_half(per_day: int) -> None:
    from live_features import load_games_final
    from live_quarter_half_features import (
        InsufficientQuarterHalfHistory, get_live_quarter_half_features,
        load_quarter_half_history,
    )
    from train_quarter_half_baseline import FEATURE_COLUMNS, load_dataset

    dataset = load_dataset()
    history = load_quarter_half_history()
    games_final = load_games_final()

    if "SEASON" not in dataset.columns:
        from train_baseline import derive_season
        dataset = dataset.assign(
            SEASON=derive_season(pd.to_datetime(dataset["GAME_DATE"])))
    dataset["GAME_DATE"] = pd.to_datetime(dataset["GAME_DATE"])

    seasons = completed_seasons(dataset)[-2:]
    sample = sample_dates(dataset, seasons)

    mismatches, checked, games, refused = [], 0, 0, 0
    for _season, _label, day in sample:
        for _i, row in games_on(dataset, day, per_day).iterrows():
            try:
                served = get_live_quarter_half_features(
                    row["HOME_TEAM_ID"], row["AWAY_TEAM_ID"],
                    row["GAME_DATE"], history, games_final
                ).iloc[0].to_dict()
            except InsufficientQuarterHalfHistory:
                # A REFUSAL IS NOT A LEAK. These models cannot score an
                # incomplete rolling window and say so; counting that as a
                # mismatch would report the correct behaviour as a fault.
                refused += 1
                continue
            checked += compare_row(served, row, FEATURE_COLUMNS,
                                   row["GAME_ID"], mismatches)
            games += 1

    print(f"\n  sample: seasons {seasons}, {len(sample)} game-days, "
          f"{games} scored, {refused} refused for short history")
    report_family("NBA quarter/half", FEATURE_COLUMNS, checked, games,
                  mismatches)


# ==========================================================================
# NBA player props - 17 features, per player rather than per game
# ==========================================================================

def check_nba_player_props(per_day: int) -> None:
    from live_features import load_games_final
    from live_player_features import get_live_player_features
    from train_player_baseline import FEATURE_COLUMNS, load_dataset

    dataset = load_dataset()
    dataset["GAME_DATE"] = pd.to_datetime(dataset["GAME_DATE"])
    games_final = load_games_final()

    history = dataset
    seasons = completed_seasons(dataset)[-2:]
    sample = sample_dates(dataset, seasons)

    mismatches, checked, rows = [], 0, 0
    for _season, _label, day in sample:
        today = dataset[dataset["GAME_DATE"] == day]
        for team_id in list(today["TEAM_ID"].unique())[:per_day]:
            side = today[today["TEAM_ID"] == team_id]
            if side.empty:
                continue
            first = side.iloc[0]
            # THE OPPONENT IS NOT A COLUMN HERE, so it is looked up rather
            # than assumed: the player table carries TEAM_ID and GAME_ID, and
            # games_final has both sides of every game. Guessing it from
            # IS_HOME would have silently paired the wrong teams.
            both = games_final[games_final["GAME_ID"] == first["GAME_ID"]]
            others = [int(t) for t in both["TEAM_ID"].unique()
                      if int(t) != int(team_id)]
            if len(others) != 1:
                continue
            served = get_live_player_features(
                int(team_id), others[0],
                first["GAME_DATE"], bool(first.get("IS_HOME", True)),
                history, games_final)
            if served is None or served.empty:
                continue
            by_player = {int(r["PLAYER_ID"]): r
                         for _j, r in served.iterrows()
                         if "PLAYER_ID" in served.columns}
            for _j, want in side.iterrows():
                got = by_player.get(int(want["PLAYER_ID"]))
                if got is None:
                    continue
                checked += compare_row(got.to_dict(), want, FEATURE_COLUMNS,
                                       want.get("GAME_ID", "?"), mismatches)
                rows += 1

    print(f"\n  sample: seasons {seasons}, {len(sample)} game-days, "
          f"{rows} player-games compared")
    report_family("NBA player props", FEATURE_COLUMNS, checked, rows,
                  mismatches)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-day", type=int, default=3,
                        help="games (or teams) sampled per selected game-day")
    parser.add_argument("--plant", action="store_true",
                        help="add a post-game column to the NBA feature list; "
                             "the suite must report it as a finding")
    args = parser.parse_args()

    print("TRUNCATION TEST - no feature may need the game's own result")
    print(f"  tolerance {TOLERANCE:g}, NaN == NaN")
    if args.plant:
        print("  PLANT ACTIVE: HOME_PTS added to the NBA team feature list")
    else:
        print("  the positive control is now the PLANT alone - the four "
              "availability")
        print("  features are gone, so --plant is what proves this suite can "
              "go red")

    check_nba_team(args.per_day, args.plant)
    check_nba_quarter_half(args.per_day)
    check_nba_player_props(args.per_day)

    passed = sum(1 for _n, ok, _d in RESULTS if ok)
    print()
    print(f"  {passed} of {len(RESULTS)} checks passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except HarnessFailure as error:
        print(f"\nHARNESS FAILURE, no verdict: {error}")
        raise SystemExit(2)
    except Exception as error:                                  # noqa: BLE001
        print(f"\nCRASHED before a verdict: {type(error).__name__}: {error}")
        raise SystemExit(2)
