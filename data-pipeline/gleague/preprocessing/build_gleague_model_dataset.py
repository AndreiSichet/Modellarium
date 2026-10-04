"""Assemble the G League model dataset - one row per regular-season game.

Wide format, HOME_/AWAY_ prefixed, matching the NBA's model_dataset.csv and
the WNBA's wnba_model_dataset.csv so phase 3 reads the same shape against any
league. All five candidate feature sets ship side by side; phase 3 chooses.

TARGETS ARE REGULAR-SEASON ONLY. The Showcase Cup feeds rest days and the
Cup-inclusive rolling windows, and never appears as a row here - asserted
below by game-id type digit rather than trusted.
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parents[1]
sys.path.insert(0, str(HERE))
from build_gleague_rolling_features import CANDIDATES, METRICS  # noqa: E402

PROCESSED = PIPELINE / "data" / "gleague" / "processed"
REGULAR = PROCESSED / "gleague_games_final.csv"
ROLLING = PROCESSED / "gleague_rolling_features.csv"
REST = PROCESSED / "gleague_rest_days.csv"
ELO = PROCESSED / "gleague_elo.csv"
SHOWCASE = PROCESSED / "gleague_showcase_games.csv"
OUT = PROCESSED / "gleague_model_dataset.csv"

GAME_ID_WIDTH = 10
REGULAR_SEASON_TYPE_DIGIT = "2"

PER_SIDE = ([f"{n}_{m}" for n in CANDIDATES for m in METRICS]
            + ["REST_DAYS_RAW", "REST_DAYS", "IS_LONG_BREAK"])


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    print(__doc__)

    games = pd.read_csv(REGULAR, dtype={"GAME_ID": str},
                        parse_dates=["GAME_DATE"])
    rolling = pd.read_csv(ROLLING, dtype={"GAME_ID": str},
                          parse_dates=["GAME_DATE"])
    rest = pd.read_csv(REST, dtype={"GAME_ID": str},
                       parse_dates=["GAME_DATE"])
    elo = pd.read_csv(ELO, dtype={"GAME_ID": str},
                      parse_dates=["GAME_DATE"])

    section("MERGING, STRICTLY")
    keys = ["GAME_ID", "TEAM_ID"]
    side = games[keys + ["SEASON", "GAME_DATE", "IS_HOME", "PTS", "REB",
                         "AST", "PLUS_MINUS", "WIN"]].copy()

    # THE MERGE IS CHECKED ON KEY COVERAGE, NOT ON VALUES, and the first
    # version of this check got that wrong. A team's first-ever game
    # legitimately has every rolling window and every rest value NaN - no
    # history exists - so "all feature columns null" is a correct state for
    # 36 rows and says nothing about whether the merge matched.
    for name, frame in (("rolling", rolling), ("rest", rest)):
        before = len(side)
        theirs = set(map(tuple, frame[keys].to_numpy()))
        ours = set(map(tuple, side[keys].to_numpy()))
        if ours - theirs:
            raise SystemExit(f"{len(ours - theirs)} row(s) have no {name} "
                             f"counterpart, e.g. {sorted(ours - theirs)[:3]}")
        side = side.merge(
            frame.drop(columns=["SEASON", "GAME_DATE"]), on=keys, how="left",
            validate="one_to_one")
        if len(side) != before:
            raise SystemExit(f"{name} merge changed the row count")
        print(f"  {name:<9} merged on {keys}, every key matched, "
              f"{len(side):,} rows")

    no_history = side[[c for c in PER_SIDE
                       if c in side.columns]].isna().all(axis=1)
    debuts = (side.sort_values(["TEAM_ID", "GAME_DATE"])
              .groupby("TEAM_ID").head(1).index)
    print(f"  {int(no_history.sum())} row(s) carry no feature value at all, "
          f"against {len(debuts)} franchise debut(s)")
    if not set(side.index[no_history]) <= set(debuts):
        raise SystemExit(
            "a row with no features at all that is not a franchise debut - "
            "that is a merge or windowing fault rather than absent history")
    print("  all of them are franchise debuts, so absent history rather "
          "than a failed merge")

    section("RESHAPING TO ONE ROW PER GAME")
    home = side[side["IS_HOME"]].set_index("GAME_ID")
    away = side[~side["IS_HOME"]].set_index("GAME_ID")
    if set(home.index) != set(away.index):
        raise SystemExit("a game without exactly one home and one away side")

    out = pd.DataFrame(index=home.index)
    out["SEASON"] = home["SEASON"]
    out["GAME_DATE"] = home["GAME_DATE"]
    out["HOME_TEAM_ID"] = home["TEAM_ID"]
    out["AWAY_TEAM_ID"] = away["TEAM_ID"]

    for column in PER_SIDE:
        if column in home.columns:
            out[f"HOME_{column}"] = home[column]
            out[f"AWAY_{column}"] = away[column]

    out = out.join(elo.set_index("GAME_ID")[
        ["HOME_TEAM_ELO", "AWAY_TEAM_ELO"]], how="left")
    if out[["HOME_TEAM_ELO", "AWAY_TEAM_ELO"]].isna().any().any():
        raise SystemExit("a game with no Elo rating")

    # Labels, never features. Derived from PTS and WL, which the validator
    # cross-checks against each other - not from PLUS_MINUS, which is wrong
    # on 6.79% of games here.
    out["HOME_PTS"] = home["PTS"]
    out["AWAY_PTS"] = away["PTS"]
    out["HOME_MARGIN"] = home["PTS"] - away["PTS"]
    out["TOTAL_PTS"] = home["PTS"] + away["PTS"]
    out["HOME_WIN"] = home["WIN"]

    section("LABEL GATES")
    agree = (out["HOME_WIN"] == (out["HOME_MARGIN"] > 0).astype("Int64"))
    known = out["HOME_WIN"].notna()
    if not bool(agree[known].all()):
        raise SystemExit(f"HOME_WIN disagrees with the margin on "
                         f"{int((~agree[known]).sum())} game(s)")
    print(f"  HOME_WIN agrees with the margin on all "
          f"{int(known.sum()):,} game(s) where it is known")
    if (out["HOME_MARGIN"] == 0).any():
        raise SystemExit("a tied game reached the dataset")
    print("  no tied games - basketball has no draws, and the cleaner now "
          "drops the one\n  game whose record claimed otherwise")

    # The margin must equal the recomputed PLUS_MINUS, which is the column the
    # rolling features were built from. If these disagree, the features and
    # the labels describe different games.
    if not bool((out["HOME_MARGIN"] == home["PLUS_MINUS"]).all()):
        raise SystemExit("HOME_MARGIN differs from the recomputed PLUS_MINUS")
    print("  HOME_MARGIN == the recomputed PLUS_MINUS on every game, so the "
          "features and\n  the labels describe the same games")

    section("NO SHOWCASE CUP ROW IS A TARGET")
    digits = (out.index.astype(str).str.zfill(GAME_ID_WIDTH).str[2])
    if set(digits) != {REGULAR_SEASON_TYPE_DIGIT}:
        raise SystemExit(f"non-regular-season type digit(s) present: "
                         f"{sorted(set(digits))}")
    print(f"  every one of {len(out):,} rows carries type digit "
          f"'{REGULAR_SEASON_TYPE_DIGIT}'")

    cup = set(pd.read_csv(SHOWCASE, dtype={"GAME_ID": str})["GAME_ID"])
    overlap = cup & set(out.index)
    if overlap:
        raise SystemExit(f"{len(overlap)} Cup game(s) appear as targets")
    print(f"  none of the {len(cup):,} Cup games appears here, though their "
          f"results do\n  reach the CUP5/CUP10 features and the rest-day "
          f"column")

    section("OUTPUT")
    out = out.reset_index().sort_values(["GAME_DATE", "GAME_ID"])
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"  wrote {OUT.name}: {len(out):,} games x {out.shape[1]} columns")
    print(f"  home win rate {out['HOME_WIN'].mean() * 100:.1f}%  "
          f"(NBA 56.4%, WNBA 55.4%)")
    print(f"  {out['SEASON'].nunique()} seasons, "
          f"{out['GAME_DATE'].min().date()} to "
          f"{out['GAME_DATE'].max().date()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
