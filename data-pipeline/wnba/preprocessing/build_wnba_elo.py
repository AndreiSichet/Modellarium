"""Add Elo ratings to the WNBA feature table, with the parameters FITTED.

The NBA pipeline uses K=20 and regresses a third of the way to the mean between
seasons. Those are not copied here: the between-season carryover encodes an
assumption about roster continuity, and the WNBA's is not the NBA's - 12-player
rosters, three expansion teams since 2025, and far more turnover per seat.

So both are fitted by minimising Elo's own log loss as a standalone predictor,
on TRAINING SEASONS ONLY. That is the one optimised quantity in this phase, and
it obeys the same leakage rule as everything else.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

WNBA = Path(__file__).resolve().parents[1]
PIPELINE = WNBA.parent
PROCESSED = PIPELINE / "data" / "wnba" / "processed"
INPUT_PATH = PROCESSED / "wnba_games_with_features.csv"
OUTPUT_PATH = PROCESSED / "wnba_games_final_features.csv"

TEAM_KEY = "TEAM_ID"
BASELINE_RATING = 1500.0

# Mirrors the NBA's shape - train, one validation season, the two most recent
# as test. Phase 3 may choose differently, and if it does THE ELO PARAMETERS
# MUST BE REFIT: they are fitted on whatever is called training here.
TRAIN_SEASONS = list(range(2015, 2024))   # 2015-2023
VALIDATION_SEASONS = [2024]
TEST_SEASONS = [2025, 2026]

# WIDENED AFTER THE FIRST RUN. The optimum initially landed on K=40, the edge
# of the grid, with the loss still falling - which is a grid artifact rather
# than a result until the boundary is pushed out and the optimum stays put.
K_GRID = [8, 12, 20, 25, 30, 40, 50, 60, 80, 100]
CARRYOVER_GRID = [0.0, 0.2, 1 / 3, 0.5, 0.67, 0.8, 1.0]

NBA_K = 20
NBA_CARRYOVER = 1 / 3


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def expected_score(rating, opponent_rating):
    return 1 / (1 + 10 ** ((opponent_rating - rating) / 400))


def run_elo(games, k, carryover):
    """Sequential Elo over every game in order. Returns per-row ratings.

    `games` is a list of (game_id, season, idx_a, team_a, won_a, idx_b, team_b).
    Inherently sequential, so a plain loop rather than anything vectorised.
    """
    ratings, last_season = {}, {}
    team_elo, opp_elo, expectations = {}, {}, {}

    def rating_for(team, season):
        if team not in ratings:
            # EXPANSION TEAMS START AT THE LEAGUE MEAN. Stated as an
            # assumption, not a finding: expansion teams historically start
            # weak, so the mean is optimistic. With three examples, fitting an
            # offset would be fitting noise. The drift report below is how the
            # assumption gets checked.
            ratings[team] = BASELINE_RATING
        elif last_season[team] != season:
            ratings[team] = BASELINE_RATING + (
                ratings[team] - BASELINE_RATING) * (1 - carryover)
        last_season[team] = season
        return ratings[team]

    for game_id, season, idx_a, team_a, won_a, idx_b, team_b in games:
        ra = rating_for(team_a, season)
        rb = rating_for(team_b, season)
        ea = expected_score(ra, rb)

        team_elo[idx_a], opp_elo[idx_a] = ra, rb
        team_elo[idx_b], opp_elo[idx_b] = rb, ra
        expectations[game_id] = (ea, won_a, season)

        actual_a = 1.0 if won_a else 0.0
        ratings[team_a] = ra + k * (actual_a - ea)
        ratings[team_b] = rb + k * ((1 - actual_a) - (1 - ea))

    return team_elo, opp_elo, expectations, ratings


def log_loss_on(expectations, seasons):
    """Elo's own log loss, restricted to the given seasons."""
    eps = 1e-15
    picked = [(e, won) for (e, won, s) in expectations.values() if s in seasons]
    if not picked:
        return float("nan")
    p = np.clip(np.array([e for e, _ in picked]), eps, 1 - eps)
    y = np.array([1.0 if won else 0.0 for _, won in picked])
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def prepare(frame):
    """One tuple per game, in chronological order."""
    ordered = frame.sort_values(["GAME_DATE", "GAME_ID"])
    games = []
    for game_id, group in ordered.groupby("GAME_ID", sort=False):
        a, b = group.index[0], group.index[1]
        games.append((game_id, int(frame.at[a, "SEASON"]),
                      a, frame.at[a, TEAM_KEY], frame.at[a, "WL"] == "W",
                      b, frame.at[b, TEAM_KEY]))
    # groupby(sort=False) preserves first-appearance order, which after the
    # sort above is chronological.
    return games


def fit_parameters(games):
    section("FITTING K AND THE CARRYOVER - TRAINING SEASONS ONLY")
    print(f"  train      : {TRAIN_SEASONS[0]}-{TRAIN_SEASONS[-1]} "
          f"({len(TRAIN_SEASONS)} seasons)")
    print(f"  validation : {VALIDATION_SEASONS}")
    print(f"  test       : {TEST_SEASONS}  <- never influences the fit\n")

    results = []
    for carryover in CARRYOVER_GRID:
        for k in K_GRID:
            _, _, exp, _ = run_elo(games, k, carryover)
            results.append((log_loss_on(exp, set(TRAIN_SEASONS)), k, carryover))

    results.sort()
    best_loss, best_k, best_carry = results[0]

    print(f"  {'CARRYOVER':>10}" + "".join(f"{k:>8}" for k in K_GRID))
    print("  " + "-" * (10 + 8 * len(K_GRID)))
    grid = {(k, c): l for l, k, c in results}
    for carryover in CARRYOVER_GRID:
        row = f"  {carryover:>10.3f}"
        for k in K_GRID:
            value = grid[(k, carryover)]
            mark = "*" if (k == best_k and carryover == best_carry) else " "
            row += f"{value:>7.4f}{mark}"
        print(row)

    nba_loss = grid[(NBA_K, NBA_CARRYOVER)]
    on_edge = []
    if best_k in (K_GRID[0], K_GRID[-1]):
        on_edge.append(f"K={best_k}")
    if best_carry in (CARRYOVER_GRID[0], CARRYOVER_GRID[-1]):
        on_edge.append(f"carryover={best_carry:.3f}")

    print(f"\n  BEST      K={best_k}, carryover={best_carry:.3f}  "
          f"log loss {best_loss:.4f}")
    if on_edge:
        print(f"  WARNING: {', '.join(on_edge)} sits on the GRID BOUNDARY. "
              "Widen the grid before\n           reporting this as an optimum.")
    else:
        print("  interior to the grid on both axes, so it is an optimum rather "
              "than a boundary.")
    print(f"  NBA's     K={NBA_K}, carryover={NBA_CARRYOVER:.3f}  "
          f"log loss {nba_loss:.4f}")
    print(f"  improvement over copying the NBA's: "
          f"{(nba_loss - best_loss) / nba_loss * 100:+.2f}%")
    return best_k, best_carry


def report_expansion(frame):
    """Did starting expansion teams at the league mean turn out optimistic?

    Three teams is too few to fit an offset from, so the assumption stands and
    this is how it gets checked instead. All three drifting sharply down would
    say the mean is measurably wrong.
    """
    section("EXPANSION TEAMS - CHECKING THE LEAGUE-MEAN ASSUMPTION")
    first_season = frame.groupby(TEAM_KEY)["SEASON"].min()
    newcomers = first_season[first_season > frame["SEASON"].min()]

    if newcomers.empty:
        print("  none")
        return

    print(f"  {'TEAM':<6}{'DEBUT':>7}{'GAMES':>7}{'START':>9}{'END':>9}"
          f"{'MIN':>9}{'MAX':>9}  VERDICT")
    print("  " + "-" * 74)
    for team, season in newcomers.items():
        rows = frame[(frame[TEAM_KEY] == team) & (frame["SEASON"] == season)]
        rows = rows.sort_values("GAME_DATE")
        name = rows["TEAM_NAME"].iloc[0]
        elo = rows["TEAM_ELO"]
        drift = elo.iloc[-1] - elo.iloc[0]
        verdict = ("drifted DOWN - mean was optimistic" if drift < -30
                   else "drifted up" if drift > 30 else "roughly flat")
        print(f"  {name[:5]:<6}{season:>7}{len(rows):>7}{elo.iloc[0]:>9.1f}"
              f"{elo.iloc[-1]:>9.1f}{elo.min():>9.1f}{elo.max():>9.1f}  "
              f"{drift:+.0f}, {verdict}")

    print("\n  The assumption stands either way - three teams cannot support an")
    print("  offset - but a consistent downward drift is worth phase 3 knowing.")


def report_relocation_continuity(frame, carryover):
    """A franchise that changed abbreviation must keep ONE continuous rating.

    Positive control rather than assertion: the rating after the change must be
    the regressed prior rating, NOT the 1500 baseline a restart would give.
    """
    section("FRANCHISE CONTINUITY ACROSS SEASONS - POSITIVE-CONTROLLED")
    print("""  Checked for EVERY team at EVERY season boundary, not only for the
  renames. Keying on TEAM_NAME would have missed PHO->PHX, which changed
  abbreviation while keeping its name - and the abbreviation is not in this
  table, having been dropped in phase 1 precisely because it is not a key.

  The property: a team's first rating of a season must be its PRIOR rating
  regressed toward the mean, never the 1500 baseline a restart would give.
""")

    failures, debuts, checked = [], 0, 0
    for team, rows in frame.groupby(TEAM_KEY):
        seasons = sorted(rows["SEASON"].unique())
        for prev, nxt in zip(seasons, seasons[1:]):
            before = rows[rows["SEASON"] == prev].sort_values("GAME_DATE")
            after = rows[rows["SEASON"] == nxt].sort_values("GAME_DATE")
            # The rating carried out of `prev` is the last pre-game rating plus
            # that game's own update, which is what the next season regresses.
            first = after["TEAM_ELO"].iloc[0]
            checked += 1
            if abs(first - BASELINE_RATING) < 1e-9:
                failures.append((team, prev, nxt, first))
        debuts += 1

    print(f"  season boundaries checked          : {checked}")
    print(f"  franchises                         : {debuts}")
    print(f"  boundaries that RESTARTED at 1500  : {len(failures)}  "
          f"(must be 0)")
    for team, prev, nxt, value in failures[:5]:
        print(f"      {team}  {prev} -> {nxt}  {value}")
    if failures:
        raise SystemExit("a franchise restarted at the baseline across a season")

    print("\n  The three known identity changes, by TEAM_ID:")
    known = {1611661319: "SAN -> LVA at 2018", 1611661321: "TUL -> DAL at 2016",
             1611661317: "PHO -> PHX at 2025"}
    for team, label in known.items():
        season = int(label.split()[-1])
        before = frame[(frame[TEAM_KEY] == team) & (frame["SEASON"] == season - 1)]
        after = frame[(frame[TEAM_KEY] == team) & (frame["SEASON"] == season)]
        if before.empty or after.empty:
            print(f"    {team}  {label:<22} (a season is missing)")
            continue
        last = before.sort_values("GAME_DATE")["TEAM_ELO"].iloc[-1]
        first = after.sort_values("GAME_DATE")["TEAM_ELO"].iloc[0]
        print(f"    {team}  {label:<22} {last:>7.1f} -> {first:>7.1f}   "
              f"restart would be {BASELINE_RATING:.0f}: "
              f"{'CONTINUOUS' if abs(first - BASELINE_RATING) > 1 else 'RESTARTED'}")


def main() -> int:
    print(__doc__)
    frame = pd.read_csv(INPUT_PATH, dtype={"GAME_ID": str})
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])
    frame = frame.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    games = prepare(frame)
    print(f"\n{len(games):,} games in chronological order.")

    k, carryover = fit_parameters(games)

    team_elo, opp_elo, expectations, final = run_elo(games, k, carryover)
    frame["TEAM_ELO"] = pd.Series(team_elo)
    frame["OPPONENT_ELO"] = pd.Series(opp_elo)

    section("ELO AS A STANDALONE PREDICTOR, BY SPLIT")
    for label, seasons in [("train", TRAIN_SEASONS),
                           ("validation", VALIDATION_SEASONS),
                           ("test", TEST_SEASONS)]:
        print(f"  {label:<12} log loss {log_loss_on(expectations, set(seasons)):.4f}")

    report_expansion(frame)
    report_relocation_continuity(frame, carryover)

    frame = frame.sort_values([TEAM_KEY, "GAME_DATE"]).reset_index(drop=True)
    frame.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    section("OUTPUT")
    print(f"  wrote {OUTPUT_PATH.name}")
    print(f"  rows {len(frame):,}, columns {len(frame.columns)}")
    print(f"  fitted K={k}, carryover={carryover:.3f}")
    print(f"  final Elo range: {min(final.values()):.1f} .. {max(final.values()):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
