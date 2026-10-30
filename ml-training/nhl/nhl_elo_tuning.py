r"""Fit the NHL's three Elo parameters on the TRAINING seasons.

WHAT IS SEARCHED: k (how hard one result moves a rating), home_advantage
(rating points added to the home side inside the expectation) and carryover
(how much of a rating survives the summer). Each is reported with its
validation effect, one at a time, so a reader can see which of the three the
answer actually turns on.

THE SPLIT IS NOT TOUCHED. The replay runs through the validation season and
the log loss is read on the validation season ALONE. The two test seasons are
never loaded here - `--allow-test` does not exist, deliberately.

THE SEARCH IS FAST AND THE PRODUCTION FUNCTION IS SLOW, so the fast path is
CHECKED AGAINST IT rather than trusted. `replay_elo` in nhl_features is the
definition; `_fast_replay` below is the same arithmetic over integer indices
and plain arrays, and check zero asserts the two agree to 1e-12 on the
default triple before any searching happens. Two implementations of one rule
is how a project ends up with a gate that grades the wrong thing, so the
second one has to be held to the first every run.

    python ml-training/nhl/nhl_elo_tuning.py
    python ml-training/nhl/nhl_elo_tuning.py --table <path> --json out.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import nhl_features as F                                        # noqa: E402
from nhl_features import refuse_undeclared_seasons        # noqa: E402

# The grid. Coarse on purpose: the point is to find the region, and a search
# fine enough to distinguish 0.6610 from 0.6609 on 1,312 games would be
# fitting the validation season rather than choosing a parameter.
K_GRID = (4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0)
HOME_GRID = (0.0, 10.0, 20.0, 30.0, 35.0, 40.0, 50.0, 60.0, 80.0)
CARRY_GRID = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

EPSILON = 1e-15


def log_loss(actual, predicted):
    predicted = np.clip(np.asarray(predicted, dtype=float), EPSILON,
                        1.0 - EPSILON)
    actual = np.asarray(actual, dtype=float)
    return float(-np.mean(actual * np.log(predicted)
                          + (1.0 - actual) * np.log(1.0 - predicted)))


def brier(actual, predicted):
    return float(np.mean((np.asarray(predicted, dtype=float)
                          - np.asarray(actual, dtype=float)) ** 2))


class Replay:
    """The fixture list as plain arrays, replayable at any parameter triple."""

    def __init__(self, fixtures):
        teams = sorted(set(fixtures["HOME_ID"]) | set(fixtures["AWAY_ID"]))
        index = {team: i for i, team in enumerate(teams)}
        self.n_teams = len(teams)
        self.home = fixtures["HOME_ID"].map(index).to_numpy(dtype=np.int32)
        self.away = fixtures["AWAY_ID"].map(index).to_numpy(dtype=np.int32)
        self.home_win = (fixtures["HOME_GOALS"].to_numpy()
                         > fixtures["AWAY_GOALS"].to_numpy()).astype(float)
        seasons = fixtures["SEASON"].to_numpy()
        self.new_season = np.zeros(len(fixtures), dtype=bool)
        self.new_season[1:] = seasons[1:] != seasons[:-1]
        self.seasons = seasons

    def run(self, k, home_advantage, carryover):
        """Pre-game expectations for every fixture, and the pre-game ratings.

        THE SAME ARITHMETIC AS `replay_elo`, over arrays. Checked against it.
        """
        ratings = np.full(self.n_teams, F.ELO_BASELINE, dtype=float)
        expected = np.empty(len(self.home), dtype=float)
        pre_home = np.empty(len(self.home), dtype=float)
        pre_away = np.empty(len(self.home), dtype=float)
        for i in range(len(self.home)):
            if self.new_season[i]:
                ratings = F.ELO_BASELINE + carryover * (ratings
                                                        - F.ELO_BASELINE)
            h = self.home[i]
            a = self.away[i]
            home = ratings[h]
            away = ratings[a]
            pre_home[i] = home
            pre_away[i] = away
            e = 1.0 / (1.0 + 10.0 ** (
                -(home + home_advantage - away) / 400.0))
            expected[i] = e
            change = k * (self.home_win[i] - e)
            ratings[h] = home + change
            ratings[a] = away - change
        return expected, pre_home, pre_away


def score(replay, mask, k, home_advantage, carryover):
    expected, _ph, _pa = replay.run(k, home_advantage, carryover)
    return log_loss(replay.home_win[mask], expected[mask])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    # THE TEST SEASONS ARE NOT LOADED. Not filtered later - not loaded.
    #
    # AND A SEASON LATER THAN THEM IS REFUSED, NOT KEPT. `kept` is defined
    # by excluding the named test seasons, so an UNDECLARED later season -
    # the open 2026-27 in the default table - passes the filter and lands in
    # the training set. Chronology happens to spare the validation figure
    # here, because the replay is ordered and 2026-27 comes after 2023-24,
    # but the printed training-season list would have been wrong and the
    # next split would not be so lucky. The guard lives in
    # nhl_features beside TEST_SEASONS, and is imported rather than copied.
    games = F.load_games(args.table)
    refuse_undeclared_seasons(F.as_fixtures(games))
    kept = [s for s in F.season_order(games) if s not in F.TEST_SEASONS]
    games = games[games["SEASON"].isin(set(kept))]
    fixtures = F.as_fixtures(games)

    train_seasons = [s for s in kept if s not in F.VALIDATION_SEASONS]
    print(f"training seasons : {len(train_seasons)} "
          f"({train_seasons[0]} .. {train_seasons[-1]})")
    print(f"validation season: {', '.join(F.VALIDATION_SEASONS)}")
    print(f"test seasons     : {', '.join(F.TEST_SEASONS)}  "
          f"- NOT LOADED")
    print(f"fixtures replayed: {len(fixtures)}")
    print()

    replay = Replay(fixtures)
    validation = np.isin(replay.seasons, list(F.VALIDATION_SEASONS))
    print(f"validation fixtures: {int(validation.sum())}")

    # ---- check 0: the fast path equals the production function --------
    d = F.ELO_DEFAULTS
    _e, fast_home, fast_away = replay.run(d["k"], d["home_advantage"],
                                          d["carryover"])
    slow_home, slow_away, _final = F.replay_elo(
        fixtures, d["k"], d["home_advantage"], d["carryover"])
    worst = max(float(np.abs(fast_home - slow_home).max()),
                float(np.abs(fast_away - slow_away).max()))
    if worst > 1e-12:
        print(f"ABORT: the search's replay disagrees with nhl_features."
              f"replay_elo by {worst:.3e} Elo points")
        return 2
    print(f"check 0: the search's replay matches nhl_features.replay_elo "
          f"exactly (worst {worst:.1e} Elo)")
    print()

    # ---- the search ---------------------------------------------------
    best = None
    for k in K_GRID:
        for home_advantage in HOME_GRID:
            for carryover in CARRY_GRID:
                value = score(replay, validation, k, home_advantage,
                              carryover)
                if best is None or value < best[0]:
                    best = (value, k, home_advantage, carryover)
    loss, k, home_advantage, carryover = best
    print(f"BEST ON VALIDATION: k={k:g}  home_advantage={home_advantage:g}  "
          f"carryover={carryover:g}")
    print(f"  validation log loss {loss:.6f}")

    expected, _ph, _pa = replay.run(k, home_advantage, carryover)
    actual = replay.home_win[validation]
    predicted = expected[validation]
    print(f"  validation accuracy {np.mean((predicted > 0.5) == (actual > 0.5)):.6f}"
          f"   Brier {brier(actual, predicted):.6f}")
    print(f"  the always-home rate on validation is "
          f"{actual.mean():.6f}, so no-skill log loss is "
          f"{log_loss(actual, np.full(len(actual), actual.mean())):.6f}")
    print()

    # ---- one parameter at a time --------------------------------------
    # EACH PARAMETER'S VALIDATION EFFECT, with the other two held at the
    # chosen values. This is the part that says whether a number was chosen
    # or merely landed on: a flat sweep means the parameter does not matter
    # and the grid's winner is noise.
    print("EACH PARAMETER'S VALIDATION EFFECT, others held at the choice")
    sweeps = (("k", K_GRID, 0), ("home_advantage", HOME_GRID, 1),
              ("carryover", CARRY_GRID, 2))
    report = {}
    for name, grid, position in sweeps:
        print(f"  {name}")
        selected = [k, home_advantage, carryover][position]
        values = []
        for candidate in grid:
            triple = [k, home_advantage, carryover]
            triple[position] = candidate
            values.append((candidate, score(replay, validation, *triple)))
        floor = min(v for _c, v in values)
        span = max(v for _c, v in values) - floor
        for candidate, value in values:
            marker = "  <- chosen" if candidate == selected else ""
            bar = "#" * int(round(60 * (value - floor) / (span or 1)))
            print(f"      {candidate:>6g}  {value:.6f}  {bar}{marker}")
        print(f"      span across the sweep: {span:.6f} log loss")
        report[name] = {"grid": list(grid),
                        "losses": [v for _c, v in values],
                        "span": span}

    print()
    print("WHAT THE SWEEPS SAY")
    for name in ("k", "home_advantage", "carryover"):
        span = report[name]["span"]
        print(f"  {name:<16} span {span:.6f}"
              + ("   - matters" if span > 0.002 else
                 "   - nearly flat; the grid's winner here is close to "
                 "arbitrary"))

    chosen = {"k": k, "home_advantage": home_advantage,
              "carryover": carryover, "validation_log_loss": loss,
              "validation_brier": brier(actual, predicted),
              "validation_accuracy": float(
                  np.mean((predicted > 0.5) == (actual > 0.5))),
              "validation_fixtures": int(validation.sum()),
              "training_seasons": train_seasons,
              "validation_seasons": list(F.VALIDATION_SEASONS),
              "sweeps": report}
    if args.json:
        args.json.write_text(json.dumps(chosen, indent=2), encoding="utf-8")
        print()
        print(f"written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
