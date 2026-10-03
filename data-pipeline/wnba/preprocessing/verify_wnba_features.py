"""Verify the WNBA feature builders: lag guard, and Elo's leakage boundary.

Both are negative-tested AND positive-controlled. The positive control is the
half that matters: without it, a negative test passes identically on a feature
that reads no data at all, which this project has caught three times.

Operates on in-memory copies. Writes nothing.
"""

import sys
from pathlib import Path

import pandas as pd

WNBA = Path(__file__).resolve().parent
PIPELINE = WNBA.parents[1]
sys.path.insert(0, str(WNBA))
sys.path.insert(0, str(PIPELINE / "preprocessing"))

from build_rolling_features import trailing_mean  # noqa: E402
from build_wnba_elo import (  # noqa: E402
    CARRYOVER_GRID, K_GRID, TEST_SEASONS, TRAIN_SEASONS, fit_parameters,
    log_loss_on, prepare, run_elo,
)

PROCESSED = PIPELINE / "data" / "wnba" / "processed"
FEATURES_PATH = PROCESSED / "wnba_games_final_features.csv"
TEAM_KEY = "TEAM_ID"

results = []


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def record(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"         {detail}")
    results.append(ok)


def load():
    frame = pd.read_csv(FEATURES_PATH, dtype={"GAME_ID": str})
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])
    return frame.sort_values([TEAM_KEY, "GAME_DATE"]).reset_index(drop=True)


def recompute(frame, prefix, window, metric):
    keys = [TEAM_KEY, "SEASON"] if prefix == "ROLL" else [TEAM_KEY]
    return trailing_mean(frame.groupby(keys), metric, window)


def lag_guard(frame):
    """A rolling value must depend only on that team's EARLIER games."""
    section("LAG GUARD - NEGATIVE-TESTED AND POSITIVE-CONTROLLED")

    for prefix in ("ROLL", "CARRY"):
        window, metric = 5, "PTS"
        col = f"{prefix}{window}_{metric}"
        honest = recompute(frame, prefix, window, metric)

        # A probe with a complete window and plenty of games on both sides.
        candidates = frame.index[honest.notna()]
        # Pick one in the middle of its team's season so both directions exist.
        probe = None
        for idx in candidates:
            team, season = frame.at[idx, TEAM_KEY], frame.at[idx, "SEASON"]
            same = frame[(frame[TEAM_KEY] == team) & (frame["SEASON"] == season)]
            pos = list(same.index).index(idx)
            if 10 <= pos <= len(same) - 10:
                probe = idx
                break
        team = frame.at[probe, TEAM_KEY]
        when = frame.at[probe, "GAME_DATE"]
        print(f"\n  {col}: probe row {probe}, team {team}, {when.date()}, "
              f"honest value {honest[probe]:.3f}")

        # NEGATIVE: wreck this team's LATER games.
        future = frame.copy()
        mask = (future[TEAM_KEY] == team) & (future["GAME_DATE"] >= when)
        future.loc[mask, metric] = 999
        moved = recompute(future, prefix, window, metric)[probe]
        record(f"{col}: corrupting {int(mask.sum())} LATER games leaves it unchanged",
               abs(moved - honest[probe]) < 1e-9,
               f"{honest[probe]:.3f} -> {moved:.3f}")

        # POSITIVE CONTROL: wreck this team's EARLIER games. Without this, the
        # negative test above would pass on a column that reads nothing.
        past = frame.copy()
        mask = (past[TEAM_KEY] == team) & (past["GAME_DATE"] < when)
        past.loc[mask, metric] = 999
        shifted = recompute(past, prefix, window, metric)[probe]
        record(f"{col}: corrupting {int(mask.sum())} EARLIER games DOES move it",
               abs(shifted - honest[probe]) > 1e-9,
               f"{honest[probe]:.3f} -> {shifted:.3f}")


def season_boundary(frame):
    """Within-season windows must be NaN at a season opener; carried ones not."""
    section("SEASON-BOUNDARY HANDLING")
    openers = frame.sort_values("GAME_DATE").groupby([TEAM_KEY, "SEASON"]).head(1)
    nan_roll = int(openers["ROLL5_PTS"].isna().sum())
    zero_roll = int((openers["ROLL5_PTS"] == 0).sum())
    record("every season opener has ROLL5_PTS NaN",
           nan_roll == len(openers), f"{nan_roll} of {len(openers)}")
    record("no season opener has ROLL5_PTS == 0 (an unknown is not a zero)",
           zero_roll == 0, f"{zero_roll} zeros")
    carried = int(openers["CARRY5_PTS"].notna().sum())
    record("the CARRY variant DOES have history at openers (that is its purpose)",
           carried > 0, f"{carried} of {len(openers)} openers carry history")


def corrected_margin(frame):
    section("THE CORRECTED MARGIN IS WHAT WAS ROLLED")
    opponent = frame.groupby("GAME_ID")["PTS"].transform(
        lambda s: s.values[::-1] if len(s) == 2 else None)
    derived = frame["PTS"] - opponent
    record("PLUS_MINUS == PTS - opponent PTS on every row",
           int((derived != frame["PLUS_MINUS"]).sum()) == 0)
    record("PTS_ALLOWED == PTS - PLUS_MINUS on every row",
           bool((frame["PTS_ALLOWED"] == frame["PTS"] - frame["PLUS_MINUS"]).all()))


def elo_leakage(frame):
    """The fitted parameters must not move when a TEST-season result changes.

    Structurally they cannot - the objective sums only over training seasons,
    and the test seasons come after them chronologically - but 'structurally
    cannot' is exactly the claim a positive control exists to check.
    """
    section("ELO PARAMETERS: FITTED ON TRAINING SEASONS ONLY")
    ordered = frame.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    print("  fitting on the real data...")
    import io
    from contextlib import redirect_stdout
    with redirect_stdout(io.StringIO()):
        honest_k, honest_c = fit_parameters(prepare(ordered))
    print(f"  honest fit: K={honest_k}, carryover={honest_c:.3f}")

    # NEGATIVE: flip every TEST-season result. The fit must not budge.
    corrupted = ordered.copy()
    mask = corrupted["SEASON"].isin(TEST_SEASONS)
    corrupted.loc[mask, "WL"] = corrupted.loc[mask, "WL"].map({"W": "L", "L": "W"})
    with redirect_stdout(io.StringIO()):
        k2, c2 = fit_parameters(prepare(corrupted))
    record(f"flipping all {int(mask.sum())} TEST-season rows leaves the fit unchanged",
           (k2, c2) == (honest_k, honest_c),
           f"K={k2}, carryover={c2:.3f}")

    # POSITIVE CONTROL. The first attempt flipped EVERY training result and the
    # fit did not budge - and that was the test being wrong, not the code.
    # Flipping all of them is SELF-INVERTING: Elo learns exactly mirrored
    # ratings, predicts the mirrored outcomes exactly as well, and the loss
    # surface is identical. A valid control has to break that symmetry, so this
    # randomises HALF of the training results instead, destroying signal rather
    # than reflecting it.
    import numpy as np
    corrupted = ordered.copy()
    mask = corrupted["SEASON"].isin(TRAIN_SEASONS)
    rng = np.random.default_rng(0)
    # Flip whole GAMES, so the two rows stay consistent with each other.
    train_games = corrupted.loc[mask, "GAME_ID"].unique()
    flip = set(rng.choice(train_games, size=len(train_games) // 2, replace=False))
    hit = corrupted["GAME_ID"].isin(flip)
    corrupted.loc[hit, "WL"] = corrupted.loc[hit, "WL"].map({"W": "L", "L": "W"})
    with redirect_stdout(io.StringIO()):
        k3, c3 = fit_parameters(prepare(corrupted))
    record(f"randomising half the training games ({len(flip)}) DOES move the fit",
           (k3, c3) != (honest_k, honest_c),
           f"K={k3}, carryover={c3:.3f}  (vs K={honest_k}, "
           f"carryover={honest_c:.3f}) - less signal should want a smaller K")

    # And the objective itself must ignore test seasons.
    games = prepare(ordered)
    _, _, exp, _ = run_elo(games, honest_k, honest_c)
    train_loss = log_loss_on(exp, set(TRAIN_SEASONS))
    test_loss = log_loss_on(exp, set(TEST_SEASONS))
    print(f"\n  train log loss {train_loss:.4f}, test {test_loss:.4f} "
          "(reported, never optimised)")


def main() -> int:
    print(__doc__)
    frame = load()
    print(f"{len(frame):,} team-game rows, {frame['GAME_ID'].nunique():,} games.")

    lag_guard(frame)
    season_boundary(frame)
    corrected_margin(frame)
    elo_leakage(frame)

    section("RESULT")
    print(f"  {sum(results)} of {len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
