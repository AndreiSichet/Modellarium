r"""Serving's absence WEIGHT must equal training's, given the same absence set.

This is the guard that would have caught the 2026-10-21 skew: serving took each
absent player's last known ROLL10_MIN through `groupby("PLAYER_ID").last()`
with no season filter, so in October it served the previous April's minutes.

SCOPE, STATED NARROWLY ON PURPOSE. The check fixes the absence set to
TRAINING's - the players with no minutes in that game's box score - and asks
only whether the two sides agree on what those absences WEIGH. It deliberately
does NOT check `ABSENT_COUNT`, because the two sides do not agree about who is
absent at all (training reads the box score, serving reads the injury report)
and that mismatch needs a retrain, not a verifier. See A3 in CLAUDE.md and the
stopgap note in `injury_availability.py`.

    training weight = build_team_availability.absence_weights:
                      players[ROLL10_MIN].where(IS_ABSENT).fillna(0.0), summed
    serving weight  = injury_availability.in_season_trailing_minutes, summed
                      over the same players, NaN -> 0 at the same place

Sampling is DERIVED, never hand-picked: the first game-days of each of the last
three seasons, where the bug lived and every weight is 0, plus game-days around
each season's midpoint, where the weights are non-zero and the rule has to
match value for value. A parity test over all-zero data proves nothing, so the
mid-season set's non-zero count is asserted against a derived floor and printed.

Exit codes follow the project's contract: 0 verdict reached, 1 a check failed,
2 crashed before reaching a verdict.

    python ml-training/verify_availability_weight_parity.py
    python ml-training/verify_availability_weight_parity.py --plants
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent / "data-pipeline" / "preprocessing"
for directory in (HERE, PIPELINE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import numpy as np                                             # noqa: E402
import pandas as pd                                            # noqa: E402

import injury_availability as IA                                # noqa: E402
import build_player_rolling_minutes as PRM                      # noqa: E402
from build_team_availability import absence_weights             # noqa: E402

TOLERANCE = 1e-9
EARLY_DAYS = 5
MID_DAYS = 5
SEASONS_SAMPLED = 3


class PlantDidNothing(RuntimeError):
    """A plant that changes no behaviour is a harness bug, not a pass."""


def load_history() -> pd.DataFrame:
    frame = pd.read_csv(
        IA.player_history_path(),
        usecols=["GAME_ID", "GAME_DATE", "TEAM_ID", "PLAYER_ID", "PLAYER_NAME",
                 "SEASON", "MIN_NUMERIC", IA.ROLLING_COLUMN],
        dtype={"GAME_ID": str, "TEAM_ID": "Int64", "PLAYER_ID": "Int64"},
        low_memory=False,
    )
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])
    for column in (IA.ROLLING_COLUMN, "MIN_NUMERIC"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["SEASON"] = pd.to_numeric(frame["SEASON"], errors="coerce")
    return frame


def sample_game_days(history: pd.DataFrame) -> list:
    """(label, season, date) triples, derived from the data."""
    seasons = sorted(int(s) for s in history["SEASON"].dropna().unique())
    chosen = seasons[-SEASONS_SAMPLED:]
    picks = []
    for season in chosen:
        days = sorted(history.loc[history["SEASON"] == season,
                                  "GAME_DATE"].unique())
        if len(days) < EARLY_DAYS + MID_DAYS:
            continue
        for date in days[:EARLY_DAYS]:
            picks.append(("early", season, pd.Timestamp(date)))
        middle = len(days) // 2
        for date in days[middle:middle + MID_DAYS]:
            picks.append(("mid", season, pd.Timestamp(date)))
    return picks


def pairs_for(history: pd.DataFrame, picks: list) -> pd.DataFrame:
    """One row per (game, team) on the sampled days."""
    wanted = pd.DataFrame(picks, columns=["BAND", "SEASON", "GAME_DATE"])
    rows = history.merge(wanted, on=["SEASON", "GAME_DATE"], how="inner")
    return rows


def compare(history: pd.DataFrame, rows: pd.DataFrame) -> tuple:
    """Returns (results dataframe, mismatches list)."""
    results, mismatches = [], []
    for (game_id, team_id, band, game_date), block in rows.groupby(
            ["GAME_ID", "TEAM_ID", "BAND", "GAME_DATE"], sort=True):
        block = block.assign(IS_ABSENT=block["MIN_NUMERIC"].isna())
        absent_ids = [int(p) for p in block.loc[block["IS_ABSENT"], "PLAYER_ID"]
                      if pd.notna(p)]
        training = float(absence_weights(block).sum())

        season = int(block["SEASON"].iloc[0])
        weights = IA.in_season_trailing_minutes(history, season, game_date)
        served = float(sum(0.0 if pd.isna(weights.get(pid, np.nan))
                           else float(weights.get(pid)) for pid in absent_ids))

        results.append({"game_id": game_id, "team_id": int(team_id),
                        "band": band, "date": game_date.date().isoformat(),
                        "absences": len(absent_ids),
                        "training": training, "served": served,
                        "delta": served - training})
        if abs(served - training) > TOLERANCE:
            mismatches.append(results[-1])
    return pd.DataFrame(results), mismatches


def run(verbose=True) -> tuple:
    history = load_history()
    picks = sample_game_days(history)
    rows = pairs_for(history, picks)
    frame, mismatches = compare(history, rows)

    early = frame[frame["band"] == "early"]
    mid = frame[frame["band"] == "mid"]
    non_zero_mid = int((mid["training"].abs() > TOLERANCE).sum())
    # DERIVED FLOOR, not a literal: most of a mid-season team-game's absences
    # are players well past the 10-appearance warm-up, so a clear majority of
    # mid-season pairs must carry weight. Half is a floor, not a target.
    floor = len(mid) // 2

    if verbose:
        print(f"  history rows            : {len(history):,}")
        print(f"  seasons sampled         : "
              f"{sorted(set(s for _, s, _ in picks))}")
        print(f"  game-days sampled       : {len(picks)} "
              f"({EARLY_DAYS} early + {MID_DAYS} mid per season)")
        print(f"  (game, team) pairs      : {len(frame)}  "
              f"[early {len(early)}, mid {len(mid)}]")
        print(f"  absences covered        : {int(frame['absences'].sum()):,}")
        print(f"  mid-season pairs with a NON-ZERO training weight: "
              f"{non_zero_mid} of {len(mid)}  (derived floor {floor})")
        print(f"  early-season pairs with a non-zero weight       : "
              f"{int((early['training'].abs() > TOLERANCE).sum())} of {len(early)}")
        if len(mid):
            print(f"  mid-season training weight: mean "
                  f"{mid['training'].mean():.3f}  max {mid['training'].max():.3f}")
        print(f"  largest |served - training|: "
              f"{frame['delta'].abs().max():.3e}  (tolerance {TOLERANCE:g})")

    problems = []
    if non_zero_mid < floor:
        problems.append(
            f"VACUOUS: only {non_zero_mid} of {len(mid)} mid-season pairs "
            f"carry weight, under the derived floor of {floor}")
    if mismatches:
        problems.append(f"{len(mismatches)} pair(s) disagree beyond {TOLERANCE:g}")
        if verbose:
            for row in mismatches[:8]:
                print(f"    {row['date']} game {row['game_id']} team "
                      f"{row['team_id']} [{row['band']}]: served "
                      f"{row['served']:.10f} training {row['training']:.10f}")

    if verbose:
        print()
        if problems:
            for problem in problems:
                print(f"  FAIL - {problem}")
        else:
            print(f"  PASS - {len(frame)} pair(s) agree to within "
                  f"{TOLERANCE:g}, and the mid-season set is not vacuous.")
    return frame, problems


# ------------------------------------------------------------------- plants
def plant_groupby_last(history_unused):
    """Revert to the defect: last known value, no season filter."""
    def broken(history, season, before):
        appearances = history[history["GAME_DATE"] < pd.Timestamp(before)]
        appearances = appearances.sort_values(
            ["PLAYER_ID", "GAME_DATE", "GAME_ID"])
        return appearances.groupby("PLAYER_ID")[IA.ROLLING_COLUMN].last()
    return broken


def plant_no_shift(history_unused):
    """Let the window include the game's own row: drop shift(1)."""
    def leaky(values, window):
        return values.rolling(window).mean()
    return leaky


def plant_noop(history_unused):
    """Changes nothing. The harness must notice and raise."""
    return IA.in_season_trailing_minutes


PLANTS = [
    ("revert to groupby().last(), no season filter",
     "in_season_trailing_minutes", plant_groupby_last, IA, "early"),
    ("window includes the game's own row (no shift(1))",
     "trailing_appearance_mean", plant_no_shift, PRM, "mid"),
    ("no-op: the same function back again",
     "in_season_trailing_minutes", plant_noop, IA, None),
]


def run_plants() -> int:
    print("  planting each defect, then reverting")
    baseline, problems = run(verbose=False)
    if problems:
        print("  HARNESS REFUSES: the clean run is already failing")
        for problem in problems:
            print(f"    {problem}")
        return 2
    print(f"  clean run: {len(baseline)} pairs, 0 problems")
    print()

    failures = 0
    for label, attribute, factory, module, must_hit in PLANTS:
        original = getattr(module, attribute)
        replacement = factory(None)
        try:
            if replacement is original:
                raise PlantDidNothing(
                    f"{module.__name__}.{attribute} was replaced with itself, "
                    f"so this plant cannot prove anything")
        except PlantDidNothing as caught:
            print(f"  PASS  no-op guard: {label}")
            print(f"        raised {type(caught).__name__}: {caught}")
            continue
        setattr(module, attribute, replacement)
        try:
            frame, problems = run(verbose=False)
        finally:
            setattr(module, attribute, original)

        if not problems:
            print(f"  FAIL  {label}")
            print("        the suite stayed GREEN - this check is not doing "
                  "its job")
            failures += 1
            continue

        band_hit = None
        merged = frame.merge(baseline, on=["game_id", "team_id", "band"],
                             suffixes=("_planted", "_clean"))
        moved = merged[(merged["served_planted"]
                        - merged["served_clean"]).abs() > TOLERANCE]
        if len(moved):
            band_hit = sorted(moved["band"].unique())
        print(f"  PASS  {label}")
        print(f"        red as intended; {len(moved)} pair(s) moved, "
              f"band(s) {band_hit}")
        if must_hit and (not band_hit or must_hit not in band_hit):
            print(f"        NOTE it was expected to move the '{must_hit}' "
                  f"band and moved {band_hit}")

    print()
    clean_frame, clean_problems = run(verbose=False)
    print(f"  after reverting every plant: {len(clean_frame)} pairs, "
          f"{len(clean_problems)} problem(s)")
    return 1 if (failures or clean_problems) else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plants", action="store_true",
                        help="make each check fail once, then revert")
    args = parser.parse_args(argv)

    print("AVAILABILITY WEIGHT PARITY (serving vs training, fixed absence set)")
    print(f"  history table: {IA.player_history_path()}")
    print()

    if args.plants:
        return run_plants()

    _frame, problems = run(verbose=True)
    return 1 if problems else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:                                  # noqa: BLE001
        print(f"\nCRASHED before reaching a verdict: "
              f"{type(error).__name__}: {error}")
        raise SystemExit(2)
