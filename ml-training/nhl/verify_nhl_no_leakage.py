r"""The NHL's truncation test: every feature recomputable from the past alone.

THE METHOD IS `verify_no_leakage.py`'S, APPLIED TO A NEW LEAGUE. For a
sampled fixture, throw away everything from its own date onwards, recompute
its features from the surviving history THROUGH THE SAME CODE SERVING WILL
USE, and require the result to equal the training row. A feature that cannot
be reproduced that way is reading something a pre-game caller does not have.

WHY IT RUNS BEFORE ANY MODEL NUMBER. A3 is the reason: four features that
turned out to be post-game were measured, reported and shipped first, and
the error only surfaced at a retrain, by which time the numbers everyone had
memorised were describing a configuration production was not in. A model
score computed on a leaking feature set is not a weak result, it is a
meaningless one, so the order matters.

WHAT IT COMPARES. Every column of the built frame that is not a key, a
target or `GAME_END` - derived from the FRAME, not from `feature_columns()`.
That is deliberate: a list-driven comparison would silently skip a column
somebody added to the frame outside `row_features`, which is exactly what
plant (a) does.

    python ml-training/nhl/verify_nhl_no_leakage.py
    python ml-training/nhl/verify_nhl_no_leakage.py --plant a
    python ml-training/nhl/verify_nhl_no_leakage.py --table <path>

EXIT CODES. 0 every check passed, 1 a check failed, 2 crashed before
reaching a verdict - the project's contract.
"""
import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import nhl_features as F                                        # noqa: E402

TOLERANCE = 1e-12


class PlantDidNothing(Exception):
    """A plant that changes no value is a harness bug, not a passing check."""


# ==========================================================================
# the sample
# ==========================================================================

def choose_sample(fixtures, every=80):
    """Fixtures to recompute: spread out, plus the cases that go wrong.

    NOT A RANDOM SAMPLE. A uniform draw would almost surely miss the three
    places a truncation bug actually lives - the first game of a season
    (where the Elo carry-over has to be applied by hand), a team's first game
    ever (where there is no history at all) and the relocation (where one
    franchise's past becomes another's). Each is included by name, and the
    evenly spaced draw covers the ordinary middle.
    """
    chosen = set(range(0, len(fixtures), every))

    # The first game-date of every season, both sides of it.
    for season in F.season_order(fixtures):
        block = fixtures.index[fixtures["SEASON"] == season]
        if len(block):
            chosen.update(list(block[:4]))

    # The relocation: the receiving franchise's first fixtures.
    new_ids = set(F.RELOCATIONS.values())
    moved = fixtures.index[fixtures["HOME_ID"].isin(new_ids)
                           | fixtures["AWAY_ID"].isin(new_ids)]
    chosen.update(list(moved[:6]))

    return sorted(chosen)


# ==========================================================================
# the recomputation
# ==========================================================================

def recompute(fixtures, games, fixture, params, carry_rolling, carry_rest,
              windows):
    """One fixture's features, from history strictly before its date.

    THE TRUNCATION IS DONE TO THE INPUTS, not inside the feature code. Both
    the games table and the fixture table are cut to rows dated before the
    game, a fresh History and a fresh Elo replay are built from what is left,
    and `row_features` - the production function - is called on that. So the
    thing under test is the feature code, and the thing supplying it is a
    world in which the game has not happened.
    """
    past_games = games[games["GAME_DATE"] < fixture.GAME_DATE]
    if (past_games["GAME_DATE"] >= fixture.GAME_DATE).any():
        raise AssertionError("the truncated games table still holds the "
                             "fixture's own date")

    ratings = F.elo_as_of(fixtures, fixture.GAME_DATE, fixture.SEASON,
                          params["k"], params["home_advantage"],
                          params["carryover"])
    history = F.History(past_games) if len(past_games) else F.History(
        past_games.assign(**{}))

    return F.row_features(
        history, fixture.HOME_ID, fixture.AWAY_ID, fixture.GAME_DATE,
        fixture.SEASON,
        ratings.get(fixture.HOME_ID, F.ELO_BASELINE),
        ratings.get(fixture.AWAY_ID, F.ELO_BASELINE),
        carry_rolling=carry_rolling, carry_rest=carry_rest, windows=windows)


def compared_columns(frame, windows):
    """Every column of the frame that claims to be a feature.

    DERIVED FROM THE FRAME, which is what makes plant (a) go red: a post-game
    column added to the frame is compared whether or not anything declares
    it.
    """
    skip = set(F.KEY_COLUMNS) | set(F.TARGET_COLUMNS) | {"GAME_END"}
    return [c for c in frame.columns if c not in skip]


def disagreements(frame, fixtures, games, sample, params, carry_rolling,
                  carry_rest, windows):
    """Every (fixture, column) where the recomputed value differs."""
    columns = compared_columns(frame, windows)
    out = []
    for index in sample:
        fixture = fixtures.iloc[index]
        got = recompute(fixtures, games, fixture, params, carry_rolling,
                        carry_rest, windows)
        stored = frame.iloc[index]
        for column in columns:
            want = stored[column]
            have = got.get(column, "<not produced>")
            if have == "<not produced>":
                out.append((index, column, want, have))
                continue
            if pd.isna(want) and pd.isna(have):
                continue
            if pd.isna(want) or pd.isna(have):
                out.append((index, column, want, have))
                continue
            if abs(float(want) - float(have)) > TOLERANCE:
                out.append((index, column, want, have))
    return out, columns


# ==========================================================================
# the checks
# ==========================================================================

def run_checks(table, plant=None, every=80):
    results = []

    def check(name, ok, detail):
        results.append((name, bool(ok), detail))

    params = dict(F.ELO_DEFAULTS)
    windows = F.ROLLING_WINDOWS
    carry_rolling = carry_rest = False

    games = F.load_games(table)
    fixtures = F.as_fixtures(games)
    frame = F.build_features(games, elo=params, carry_rolling=carry_rolling,
                             carry_rest=carry_rest, windows=windows)

    # ---------------------------------------------------------- plants
    planted_column = None
    original_rolling = F.rolling_form
    if plant == "a":
        # (a) A POST-GAME COLUMN. The game's own goals, written into the
        # frame as though they were a feature. A pre-game caller cannot have
        # them, so the recomputation produces nothing for the column and the
        # value comparison has to say so.
        planted_column = "HOME_GOALS_THIS_GAME"
        frame[planted_column] = fixtures["HOME_GOALS"].to_numpy(dtype=float)
        if frame[planted_column].isna().all():
            raise PlantDidNothing("the planted column is entirely NaN")
    elif plant == "b":
        # (b) THE OFF-BY-ONE. The rolling window includes the fixture's own
        # row - the classic leak, and the one a reviewer reading
        # `shift(1)`-free code is looking for. Planted by replacing the
        # production function, so the bulk build leaks and the truncated
        # recomputation cannot.
        def leaking_rolling_form(history, team, date, season, window,
                                 carry_across_seasons):
            dates = history.dates.get(team) or []
            cut = history.cut(team, date)
            # include the game's own row, if this history has it
            if cut < len(dates) and dates[cut] == date:
                cut += 1
            floor = 0
            if not carry_across_seasons:
                start = history.season_start(team, season)
                if start is None:
                    return {f"ROLL{window}_{m}": np.nan
                            for m in F.ROLLING_METRICS}
                floor = start
            if cut - floor < window:
                return {f"ROLL{window}_{m}": np.nan
                        for m in F.ROLLING_METRICS}
            lo = cut - window
            gf = history.goals_for[team][lo:cut]
            ga = history.goals_against[team][lo:cut]
            wins = history.wins[team][lo:cut]
            return {
                f"ROLL{window}_GOALS_FOR": float(gf.mean()),
                f"ROLL{window}_GOALS_AGAINST": float(ga.mean()),
                f"ROLL{window}_GOAL_DIFF": float((gf - ga).mean()),
                f"ROLL{window}_WIN_SHARE": float(wins.mean()),
            }

        F.rolling_form = leaking_rolling_form
        leaked = F.build_features(games, elo=params,
                                  carry_rolling=carry_rolling,
                                  carry_rest=carry_rest, windows=windows)
        moved = (leaked["HOME_ROLL10_GOAL_DIFF"]
                 .ne(frame["HOME_ROLL10_GOAL_DIFF"])
                 & leaked["HOME_ROLL10_GOAL_DIFF"].notna()).sum()
        if moved == 0:
            raise PlantDidNothing(
                "the leaking window changed no ROLL10 value")
        frame = leaked
        F.rolling_form = original_rolling

    sample = choose_sample(fixtures, every=every)

    # ---------------------------------------------------------- check 1
    declared = F.feature_columns(windows)
    present = compared_columns(frame, windows)
    check("1 the frame's feature columns are exactly the declared set",
          present == declared,
          f"{len(present)} column(s) present, {len(declared)} declared"
          + (f"; extra {sorted(set(present) - set(declared))}"
             if set(present) - set(declared) else "")
          + (f"; missing {sorted(set(declared) - set(present))}"
             if set(declared) - set(present) else ""))

    # ---------------------------------------------------------- check 2
    bad, columns = disagreements(frame, fixtures, games, sample, params,
                                 carry_rolling, carry_rest, windows)
    detail = (f"{len(sample)} fixture(s) x {len(columns)} column(s) = "
              f"{len(sample) * len(columns)} comparison(s), "
              f"{len(bad)} disagreement(s)")
    if bad:
        shown = ", ".join(f"row {i} {c}: stored {w!r} recomputed {h!r}"
                          for i, c, w, h in bad[:4])
        detail += "; " + shown + (f"; +{len(bad) - 4} more" if len(bad) > 4
                                  else "")
    check("2 every sampled feature recomputes from the past alone",
          not bad, detail)

    # ---------------------------------------------------------- check 3
    # THE SAMPLE MUST CONTAIN THE CASES IT CLAIMS TO. A truncation test over
    # 200 mid-season fixtures would pass while the season-boundary carry-over
    # was wrong, so the sample's own coverage is a check.
    seasons_hit = set(fixtures.iloc[sample]["SEASON"])
    openers = sum(
        1 for i in sample
        if fixtures.iloc[i]["GAME_DATE"]
        == fixtures[fixtures["SEASON"] == fixtures.iloc[i]["SEASON"]]
        ["GAME_DATE"].min())
    new_ids = set(F.RELOCATIONS.values())
    relocated = sum(1 for i in sample
                    if fixtures.iloc[i]["HOME_ID"] in new_ids
                    or fixtures.iloc[i]["AWAY_ID"] in new_ids)
    check("3 the sample covers every season, season openers and the "
          "relocation",
          (len(seasons_hit) == fixtures["SEASON"].nunique()
           and openers >= fixtures["SEASON"].nunique() and relocated >= 1),
          f"{len(seasons_hit)} of {fixtures['SEASON'].nunique()} season(s), "
          f"{openers} opening-date fixture(s), {relocated} fixture(s) "
          f"involving the relocated franchise")

    # ---------------------------------------------------------- check 4
    # A team's FIRST EVER game has no history, so every rolling and rest
    # column must be NaN rather than a number invented from nothing.
    #
    # CORRECTED AFTER GOING RED AGAINST WORKING CODE. The first version
    # tested every column beginning `HOME_`/`AWAY_` and so demanded that
    # HOME_ELO be NaN too - but a team with no history correctly carries the
    # 1500 BASELINE, which is a defined starting value and not an invented
    # one. The Elo columns are therefore checked for the baseline instead of
    # for absence, which is the assertion that was meant.
    elo_columns = {"HOME_ELO", "AWAY_ELO", "ELO_DIFF"}
    first_ever = {}
    for row in fixtures.itertuples():
        for side, team in (("HOME", row.HOME_ID), ("AWAY", row.AWAY_ID)):
            first_ever.setdefault(team, (row.Index, side))
    offenders = []
    baseline_misses = []
    for team, (index, side) in first_ever.items():
        row = frame.iloc[index]
        for column in columns:
            if not column.startswith(f"{side}_"):
                continue
            if column in elo_columns:
                if abs(float(row[column]) - F.ELO_BASELINE) > TOLERANCE:
                    baseline_misses.append((team, column, row[column]))
            elif not pd.isna(row[column]):
                offenders.append((team, column))
    check("4 a team's first ever game carries no rolling or rest value, and "
          "baseline Elo",
          not offenders and not baseline_misses,
          f"{len(first_ever)} franchise(s) checked, {len(offenders)} "
          f"non-null rolling/rest column(s), {len(baseline_misses)} "
          f"off-baseline Elo"
          + (f": {offenders[:3]}" if offenders else "")
          + (f": {baseline_misses[:3]}" if baseline_misses else ""))

    # ---------------------------------------------------------- check 5
    # THE ELO RECORDED IS THE PRE-GAME ONE. Read directly rather than
    # inferred: the first fixture of the corpus must be the baseline for both
    # sides, because nothing has been played.
    first = frame.iloc[0]
    check("5 the first fixture of the corpus carries baseline Elo on both "
          "sides",
          (abs(first["HOME_ELO"] - F.ELO_BASELINE) < TOLERANCE
           and abs(first["AWAY_ELO"] - F.ELO_BASELINE) < TOLERANCE),
          f"home {first['HOME_ELO']}, away {first['AWAY_ELO']}, baseline "
          f"{F.ELO_BASELINE}")

    # ---------------------------------------------------------- check 6
    # NO FEATURE COLUMN *IS* A TARGET OR A SOURCE OUTCOME COLUMN.
    #
    # CORRECTED AFTER GOING RED AGAINST WORKING CODE, and the wrong version
    # is worth recording because it is the tempting one. It tested whether a
    # feature name ENDED IN an outcome name, and flagged eight real
    # features: `ROLL10_GOALS_FOR` ends in `GOALS_FOR` and is a trailing
    # mean of PAST goals, which is the whole point of it. A name-suffix test
    # cannot tell "this game's goals" from "the last ten games' goals", so
    # it was answering a question it could not answer.
    #
    # What a name CAN settle is whether a feature column is literally one of
    # the source table's outcome columns or one of the targets. Whether a
    # correctly-named column reads the game's own row is check 2's question,
    # and check 2 can actually answer it.
    outcomes = {"GOALS_FOR", "GOALS_AGAINST", "MARGIN", "RESULT", "GAME_END",
                "DERIVED", "END_DISPUTED"}
    forbidden = outcomes | set(F.TARGET_COLUMNS)
    collisions = sorted(set(present) & forbidden)
    check("6 no feature column is a target or a source outcome column",
          not collisions, f"{len(present)} feature column(s) checked against "
          f"{len(forbidden)} forbidden name(s), {len(collisions)} collision(s)"
          + (f": {collisions}" if collisions else ""))

    return results, frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=None)
    parser.add_argument("--plant", choices=("a", "b"), default=None,
                        help="a: a post-game column; b: the window includes "
                             "the game's own row")
    parser.add_argument("--every", type=int, default=80,
                        help="sample stride through the fixture list")
    args = parser.parse_args()

    try:
        results, _frame = run_checks(args.table, plant=args.plant,
                                     every=args.every)
    except PlantDidNothing as error:
        print(f"HARNESS BUG: {error}")
        return 2
    except Exception:                                           # noqa: BLE001
        traceback.print_exc()
        return 2

    if args.plant:
        print(f"PLANT ({args.plant}) ACTIVE - checks are expected to FAIL\n")
    passed = 0
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"         {detail}")
        passed += int(ok)
    print()
    print(f"{passed} of {len(results)} check(s) passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
