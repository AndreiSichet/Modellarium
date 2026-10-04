"""Rest days for G League regular-season games, counting Showcase Cup games.

FATIGUE IS PHYSICAL, SO EVERY GAME COUNTS. A Cup game three days before a
team's regular-season opener tires it exactly as much as a league game would,
so REST_DAYS looks back at the previous game of ANY kind. This is the reason
phase 1 kept the Cup in its own table rather than discarding it: without those
rows, every team's first regular-season game since the Cup began would read as
coming off a seven-month off-season when it actually follows a Cup game by
days.

Three columns ship rather than one treatment, the arrangement the WNBA phase
settled on:

  REST_DAYS_RAW   uncapped, so a tree can see a genuine three-week gap
  REST_DAYS       clipped at 7, so the column means what the NBA's means
  IS_LONG_BREAK   the gap was over the cap

A tree splits on order, so 30 is just "a lot". A linear model cannot take that
- one 30 among values of 1-4 is a high-leverage point - and phase 3 may well
pick linear again, so both are available.

Season-scoped, like the WNBA's and unlike the NBA's: a season opener is NaN
rather than "7 days rest", because a seven-month off-season and a week off are
different facts.
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parents[1]

PROCESSED = PIPELINE / "data" / "gleague" / "processed"
REGULAR = PROCESSED / "gleague_games_final.csv"
SHOWCASE = PROCESSED / "gleague_showcase_games.csv"
OUT = PROCESSED / "gleague_rest_days.csv"

REST_DAYS_CAP = 7


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def load(path: Path, competition: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"GAME_ID": str})
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])
    frame["COMPETITION"] = competition
    return frame


def rest_from(stream: pd.DataFrame) -> pd.Series:
    """Days since each team's previous game in the same season.

    The stream is whatever games the caller chose to count; passing only
    regular-season rows is what produces the comparison figure below.
    """
    ordered = stream.sort_values(["TEAM_ID", "SEASON", "GAME_DATE"])
    delta = (ordered.groupby(["TEAM_ID", "SEASON"])["GAME_DATE"]
             .diff().dt.days)
    return delta.reindex(stream.index)


def report_cup_effect(both: pd.DataFrame, regular_only: pd.Series) -> None:
    """How many regular-season team-games have their rest changed by counting
    Cup games - the size of the problem phase 1's design exists to solve."""
    section("WHAT COUNTING CUP GAMES ACTUALLY CHANGES")

    regular = both[both["COMPETITION"] == "regular"].copy()
    regular["REST_REGULAR_ONLY"] = regular_only

    differs = (regular["REST_DAYS_RAW"].notna()
               & regular["REST_REGULAR_ONLY"].notna()
               & (regular["REST_DAYS_RAW"] != regular["REST_REGULAR_ONLY"]))
    # A team's first regular-season game of a Cup season has no
    # regular-season predecessor at all, so counting the Cup turns a NaN into
    # a number. That is the largest part of the effect and a plain inequality
    # cannot see it.
    gained = (regular["REST_DAYS_RAW"].notna()
              & regular["REST_REGULAR_ONLY"].isna())

    print(f"  {'season':<9}{'team-games':>12}{'value changed':>15}"
          f"{'NaN -> value':>14}{'median shift':>14}")
    print("  " + "-" * 64)
    for season in sorted(regular["SEASON"].unique()):
        rows = regular["SEASON"] == season
        changed = int((differs & rows).sum())
        new = int((gained & rows).sum())
        if changed:
            shift = (regular.loc[differs & rows, "REST_REGULAR_ONLY"]
                     - regular.loc[differs & rows, "REST_DAYS_RAW"]).median()
            shift_text = f"{shift:>13.1f}"
        else:
            shift_text = f"{'-':>13}"
        print(f"  {season:<9}{int(rows.sum()):>12}{changed:>15}{new:>14}"
              f"{shift_text}")

    print(f"\n  total: {int(differs.sum())} value(s) changed, "
          f"{int(gained.sum())} NaN replaced by a real value, across "
          f"{len(regular):,} regular-season team-games")
    print("""
  Both columns are the point. A CHANGED value means the Cup game sat between
  two league games and the team had less rest than the league schedule alone
  implies. A NaN REPLACED means the team's first league game of the season now
  has a rest value at all - which is the case phase 1 kept the Cup for, and it
  cannot be seen by comparing two numbers.""")

    # THE WHOLE EFFECT IS ON OPENERS, AND THAT IS STRUCTURAL RATHER THAN LUCK.
    # The Cup is a strict PREFIX: in all five Cup seasons its last game falls
    # before the first regular-season game, so a Cup game can never sit
    # between two league games. Asserted rather than inferred from the zero
    # above, because "no values changed" and "the two never interleave" are
    # different claims and only the second explains it.
    cup = both[both["COMPETITION"] == "showcase"]
    for season, group in cup.groupby("SEASON"):
        league = regular[regular["SEASON"] == season]
        if len(league) and group["GAME_DATE"].max() >= league["GAME_DATE"].min():
            raise SystemExit(
                f"{season}: a Cup game falls on or after the first "
                f"regular-season game, so the prefix assumption behind the "
                f"zero above does not hold and the report is misleading")

    openers = regular.sort_values(["TEAM_ID", "SEASON", "GAME_DATE"]) \
        .groupby(["TEAM_ID", "SEASON"]).head(1)
    with_value = openers[openers["REST_DAYS_RAW"].notna()]
    if len(with_value):
        within = int((with_value["REST_DAYS_RAW"] <= REST_DAYS_CAP).sum())
        print(f"""
  The Cup is a strict PREFIX in all {cup['SEASON'].nunique()} Cup seasons - """
              f"""asserted, not inferred - so it
  cannot interleave and every affected row is an opener. Their new rest
  values: min {with_value['REST_DAYS_RAW'].min():.0f}, median """
              f"""{with_value['REST_DAYS_RAW'].median():.0f}, max """
              f"""{with_value['REST_DAYS_RAW'].max():.0f} days, with {within} of """
              f"""{len(with_value)} at or under the
  {REST_DAYS_CAP}-day cap. So for most openers this is a real rest figure """
              f"""rather than one
  clipped to the cap, and for all of them it is the difference between a row
  a linear model can score and one it cannot.""")


def report_long_breaks(frame: pd.DataFrame) -> None:
    section("LONG BREAKS")
    regular = frame[frame["COMPETITION"] == "regular"]
    over = regular[regular["REST_DAYS_RAW"] > REST_DAYS_CAP]
    print(f"  {len(over):,} of {len(regular):,} regular-season team-games "
          f"exceed the {REST_DAYS_CAP}-day cap "
          f"({len(over) / len(regular) * 100:.1f}%)")
    if len(over):
        print(f"  longest {over['REST_DAYS_RAW'].max():.0f} days, "
              f"median over the cap {over['REST_DAYS_RAW'].median():.0f}")

    # A gap shared by most of the league is a scheduled break; one team's long
    # gap is its own schedule. Reported by how many teams share the window
    # rather than from a remembered calendar, which is how the WNBA phase
    # identified its four league-wide breaks.
    per_season = []
    for season, group in regular.groupby("SEASON"):
        teams = group["TEAM_ID"].nunique()
        idle = group[group["REST_DAYS_RAW"] > 10]
        if len(idle):
            per_season.append((season, teams, idle["TEAM_ID"].nunique(),
                               int(idle["REST_DAYS_RAW"].max())))
    if per_season:
        print(f"\n  {'season':<9}{'teams':>7}{'idle >10d':>11}{'longest':>9}")
        print("  " + "-" * 36)
        for season, teams, idle_teams, longest in per_season:
            flag = "  <- most of the league" if idle_teams > teams / 2 else ""
            print(f"  {season:<9}{teams:>7}{idle_teams:>11}"
                  f"{longest:>9}{flag}")


def main() -> int:
    print(__doc__)

    regular = load(REGULAR, "regular")
    showcase = load(SHOWCASE, "showcase") if SHOWCASE.exists() \
        else pd.DataFrame()
    print(f"  regular  {len(regular):,} team-games, "
          f"{regular['SEASON'].nunique()} seasons")
    print(f"  showcase {len(showcase):,} team-games, "
          f"{showcase['SEASON'].nunique() if len(showcase) else 0} seasons")

    both = pd.concat([regular, showcase], ignore_index=True)
    both = both.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    both["REST_DAYS_RAW"] = rest_from(both)
    both["REST_DAYS"] = both["REST_DAYS_RAW"].clip(upper=REST_DAYS_CAP)
    both["IS_LONG_BREAK"] = (both["REST_DAYS_RAW"] > REST_DAYS_CAP) \
        .astype("Int64")
    both.loc[both["REST_DAYS_RAW"].isna(), "IS_LONG_BREAK"] = pd.NA

    # The same computation over regular-season rows only, which is what the
    # comparison needs. Computed on a separate frame so the shipped column
    # cannot accidentally be the regular-only one.
    only = both[both["COMPETITION"] == "regular"].copy()
    only["REST_REGULAR_ONLY"] = rest_from(only)
    regular_only = only.set_index(only.index)["REST_REGULAR_ONLY"]

    report_cup_effect(both, regular_only)
    report_long_breaks(both)

    section("OUTPUT")
    out = both[both["COMPETITION"] == "regular"][
        ["GAME_ID", "TEAM_ID", "SEASON", "GAME_DATE",
         "REST_DAYS_RAW", "REST_DAYS", "IS_LONG_BREAK"]].copy()

    openers = out.groupby(["TEAM_ID", "SEASON"]).size()
    nan_count = int(out["REST_DAYS_RAW"].isna().sum())
    print(f"  {len(out):,} regular-season team-games")
    print(f"  {nan_count} with no rest value")

    # A season opener is NaN unless a Cup game preceded it, so the count must
    # sit between zero and the number of team-seasons. Asserting equality
    # would be wrong: that is exactly what the Cup rows change.
    if not 0 <= nan_count <= len(openers):
        raise SystemExit(f"{nan_count} NaN rest values against "
                         f"{len(openers)} team-seasons")
    print(f"  against {len(openers)} team-seasons - so {len(openers) - nan_count} "
          f"season opener(s) have a rest value because a Cup game preceded "
          f"them")

    if (out["REST_DAYS_RAW"] <= 0).any():
        raise SystemExit("a non-positive rest value: two games on one date")

    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"  wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
