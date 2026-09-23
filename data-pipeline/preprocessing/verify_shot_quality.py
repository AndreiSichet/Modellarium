"""Exercise the shot-quality guards on synthetic data.

The 20-game verification corpus is all one training season, so the two guards
that matter - "the baseline never reads test seasons" and "a rolling value
never reads its own game" - could not fire at all. A guard that has not been
shown to fail has not been shown to work, so they are exercised here against
data built to trigger them, with no API calls.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_shot_quality import (  # noqa: E402
    ROLLING_WINDOW,
    TEST_SEASONS,
    TRAIN_SEASONS,
    ZONE,
    add_rolling,
    assert_baseline_is_train_only,
    assert_no_lookahead,
    team_game_quality,
    zone_baseline,
)

ZONES = ["Restricted Area", "In The Paint (Non-RA)", "Mid-Range",
         "Above the Break 3", "Left Corner 3", "Right Corner 3"]
THREE_ZONES = {"Above the Break 3", "Left Corner 3", "Right Corner 3"}

TEAMS = [1610612737, 1610612738, 1610612739, 1610612740]
GAMES_PER_TEAM_PER_SEASON = 20
SHOTS_PER_TEAM_GAME = 80

failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail:
        print(f"        {detail}")
    if not ok:
        failures.append(name)


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def synthetic_shots(seed: int = 0) -> pd.DataFrame:
    """Shots spanning train and test seasons, enough games to fill a window."""
    rng = np.random.default_rng(seed)
    rows = []
    game_id = 22100000

    for season in TRAIN_SEASONS + TEST_SEASONS:
        for index in range(GAMES_PER_TEAM_PER_SEASON):
            for team in TEAMS:
                game_id += 1
                date = pd.Timestamp(f"{season}-11-01") + pd.Timedelta(days=index * 2)
                zones = rng.choice(ZONES, SHOTS_PER_TEAM_GAME)
                for zone in zones:
                    rows.append({
                        "GAME_ID_INT": game_id,
                        "TEAM_ID": team,
                        "SEASON": season,
                        "GAME_DATE": date,
                        ZONE: zone,
                        "SHOT_TYPE": ("3PT Field Goal" if zone in THREE_ZONES
                                      else "2PT Field Goal"),
                        "SHOT_MADE_FLAG": int(rng.random() < 0.45),
                    })

    return pd.DataFrame(rows)


def baseline_ignores_test_seasons(shots: pd.DataFrame) -> None:
    section("1. THE ZONE BASELINE MUST NOT READ TEST SEASONS")

    baseline = zone_baseline(shots)
    test_rows = int(shots["SEASON"].isin(TEST_SEASONS).sum())
    check("the synthetic corpus actually contains test-season shots",
          test_rows > 0,
          f"{test_rows:,} test-season rows - without these the guard is vacuous")

    poisoned = shots.copy()
    is_test = poisoned["SEASON"].isin(TEST_SEASONS)
    poisoned.loc[is_test, "SHOT_MADE_FLAG"] = 1
    after = zone_baseline(poisoned)
    check("forcing every test-season shot to a make leaves the baseline identical",
          baseline[["LEAGUE_FG_PCT"]].equals(after[["LEAGUE_FG_PCT"]]))

    control = shots.copy()
    control.loc[control["SEASON"].isin(TRAIN_SEASONS), "SHOT_MADE_FLAG"] = 1
    moved = not baseline[["LEAGUE_FG_PCT"]].equals(
        zone_baseline(control)[["LEAGUE_FG_PCT"]])
    check("POSITIVE CONTROL: corrupting train seasons does move it",
          moved,
          "without this, the check above would pass on a baseline of nothing")

    print("\n  Running the module's own assertion against the same data:")
    assert_baseline_is_train_only(shots, baseline)


def rolling_does_not_look_ahead(shots: pd.DataFrame) -> None:
    section("2. A ROLLING VALUE MUST NOT READ ITS OWN GAME OR LATER")

    quality = add_rolling(team_game_quality(shots, zone_baseline(shots)))
    complete = quality[f"ROLL{ROLLING_WINDOW}_SHOOTING_LUCK"].notna().sum()
    check(f"some rows have a complete {ROLLING_WINDOW}-game window",
          complete > 0,
          f"{complete:,} rows - without these the lag guard is vacuous")

    print("\n  Running the module's own assertion:")
    assert_no_lookahead(quality)
    check("assert_no_lookahead accepts correctly-lagged features", True)


def guard_catches_a_real_leak(shots: pd.DataFrame) -> None:
    """The sharpest test: break the lag deliberately and confirm it is caught."""
    section("3. THE LAG GUARD MUST CATCH A DELIBERATE LEAK")

    quality = team_game_quality(shots, zone_baseline(shots))

    def leaky_rolling(frame):
        """add_rolling WITHOUT shift(1) - a window including the row's own game.

        This is the exact bug the lag convention exists to prevent, and it is
        passed in as the implementation under test rather than written into a
        column, because the guard recomputes and would otherwise just re-apply
        the correct shift and test nothing.
        """
        frame = frame.sort_values(["TEAM_ID", "SEASON", "GAME_DATE", "GAME_ID"])
        frame = frame.reset_index(drop=True)
        for column in ("SHOT_QUALITY", "SHOOTING_LUCK"):
            frame[f"ROLL{ROLLING_WINDOW}_{column}"] = (
                frame.groupby(["TEAM_ID", "SEASON"])[column]
                .transform(lambda s: s.rolling(ROLLING_WINDOW).mean())
            )
        return frame

    try:
        assert_no_lookahead(quality, recompute_with=leaky_rolling)
    except SystemExit as error:
        message = str(error)
        caught_the_leak = "reads its own game" in message
        check("an unshifted rolling mean is rejected", True, message[:70])
        check("rejected BY THE LEAK CHECK, not by the positive control",
              caught_the_leak,
              "if the positive control fires instead, the guard raised for an "
              "unrelated reason and the leak went undetected")
    else:
        check("an unshifted rolling mean is rejected", False,
              "the guard accepted a feature that reads its own game")


def luck_is_zero_mean_on_training_data(shots: pd.DataFrame) -> None:
    """Self-consistency: expected and actual must balance where the baseline came from."""
    section("4. SHOOTING LUCK MUST AVERAGE TO ZERO ON THE BASELINE'S OWN DATA")

    baseline = zone_baseline(shots)
    quality = team_game_quality(shots, baseline)
    train = quality[quality["SEASON"].isin(TRAIN_SEASONS)]
    mean_luck = float(train["SHOOTING_LUCK"].mean())

    check("training-season luck averages ~0",
          abs(mean_luck) < 1e-3,
          f"mean {mean_luck:+.2e} - expected and actual are two views of the "
          f"same shots here, so any real offset would be an arithmetic error")


def main() -> int:
    shots = synthetic_shots()
    print(f"Synthetic corpus: {len(shots):,} shots, "
          f"{shots['GAME_ID_INT'].nunique():,} team-games, "
          f"seasons {sorted(shots['SEASON'].unique())}")

    baseline_ignores_test_seasons(shots)
    rolling_does_not_look_ahead(shots)
    guard_catches_a_real_leak(shots)
    luck_is_zero_mean_on_training_data(shots)

    section("RESULT")
    if failures:
        print(f"  {len(failures)} FAILED: {failures}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
