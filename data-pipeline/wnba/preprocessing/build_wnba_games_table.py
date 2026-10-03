"""Build the canonical WNBA games table from the raw per-season CSVs.

Separate from the validator so a failed gate cannot leave a half-written
output behind - the same split the NBA pipeline keeps between
validate_games.py and build_games_table.py.

Long format, one row per team per game, matching games_final.csv's concepts so
phase 2's feature code reads naturally against either league. Phase 2 adds the
rolling, rest-day and Elo columns.
"""

import sys
from pathlib import Path

import pandas as pd

PIPELINE = Path(__file__).resolve().parents[2]
RAW_DIR = PIPELINE / "data" / "wnba" / "raw"
PROCESSED_DIR = PIPELINE / "data" / "wnba" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "wnba_games_final.csv"

# Dropped to match games_final.csv. TEAM_ABBREVIATION is used to derive
# OPPONENT first, then dropped: it is not a stable key (PHO->PHX, SAN->LVA,
# TUL->DAL all kept their TEAM_ID through the change).
COLUMNS_TO_DROP = ["SEASON_ID", "MIN", "TEAM_ABBREVIATION", "MATCHUP"]

GAME_ID_WIDTH = 10
REGULAR_SEASON_TYPE_DIGIT = "2"


def load_all_seasons() -> pd.DataFrame:
    frames = []
    for path in sorted(RAW_DIR.glob("wnba_games_*.csv")):
        season = int(path.stem.replace("wnba_games_", ""))
        frame = pd.read_csv(path, dtype={"GAME_ID": str})
        # A WNBA season sits inside one calendar year, so the label IS the
        # year. No month boundary is involved, unlike derive_season for the NBA.
        frame["SEASON"] = season
        frames.append(frame)
    if not frames:
        raise SystemExit(f"No season files in {RAW_DIR}. Run the fetcher first.")
    return pd.concat(frames, ignore_index=True)


def pair_on_game_id(frame: pd.DataFrame, column: str) -> pd.Series:
    """The other row's value for `column`, taken by pairing on GAME_ID.

    Deliberately NOT parsed out of MATCHUP: the NBA pipeline found ~10 games
    whose MATCHUP string was identical on both rows, so the text cannot be
    trusted to say who the opponent was.
    """
    def other(group):
        if len(group) != 2:
            return pd.Series(pd.NA, index=group.index)
        return pd.Series(group[column].values[::-1], index=group.index)

    return frame.groupby("GAME_ID", group_keys=False)[[column]].apply(
        lambda g: other(g)[g.index])


def drop_unreliable_home_away(frame: pd.DataFrame) -> pd.DataFrame:
    home_counts = frame.groupby("GAME_ID")["IS_HOME"].sum()
    bad = home_counts[home_counts != 1].index
    if not len(bad):
        print("  IS_HOME: every game has exactly one home team. Nothing dropped.")
        return frame
    dropped = frame[frame["GAME_ID"].isin(bad)]
    print(f"  IS_HOME: dropping {len(bad)} game(s) ({len(dropped)} rows) "
          "without exactly one home team")
    return frame[~frame["GAME_ID"].isin(bad)]


def recompute_plus_minus(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace PLUS_MINUS with the margin derived from PTS.

    The raw column does not reconcile on ~1.2% of WNBA games - and the NBA's
    own raw data carries the same defect at 1.02%, so this is a property of
    LeagueGameFinder rather than of this league. PTS and WL are validated
    upstream and agree with each other on every game, so the margin is
    arithmetic over trusted columns rather than invented data.

    The column keeps its name so phase 2 reads the same against both leagues.
    """
    opponent_pts = pair_on_game_id(frame, "PTS")
    derived = (frame["PTS"] - opponent_pts).astype(float)

    changed = int((derived != frame["PLUS_MINUS"].astype(float)).sum())
    worst = float((derived - frame["PLUS_MINUS"].astype(float)).abs().max())
    print(f"  PLUS_MINUS: recomputed from PTS. {changed:,} of {len(frame):,} "
          f"rows changed, worst correction {worst:.0f} pts.")

    # The sign must agree with the recorded result on every row, or the
    # recomputation is wrong rather than the source.
    disagree = int(((derived > 0) != (frame["WL"] == "W")).sum())
    if disagree:
        raise SystemExit(
            f"{disagree} rows where the derived margin's sign disagrees with WL")
    print("  PLUS_MINUS: derived sign agrees with WL on every row.")

    frame["PLUS_MINUS"] = derived
    return frame


def assert_regular_season_only(frame: pd.DataFrame) -> None:
    padded = frame["GAME_ID"].astype(str).str.zfill(GAME_ID_WIDTH)
    offenders = padded[padded.str[2] != REGULAR_SEASON_TYPE_DIGIT]
    if len(offenders):
        raise SystemExit(f"{len(offenders)} non-regular-season rows in the output")
    print(f"  game-id type digit: all {len(frame):,} rows are "
          f"'{REGULAR_SEASON_TYPE_DIGIT}' (regular season).")


def main() -> int:
    frame = load_all_seasons()
    print(f"Loaded {len(frame):,} team-game rows from "
          f"{len(sorted(RAW_DIR.glob('wnba_games_*.csv')))} season files.\n")

    frame["IS_HOME"] = frame["MATCHUP"].str.contains("vs.", regex=False)
    frame["OPPONENT"] = pair_on_game_id(frame, "TEAM_ABBREVIATION")
    # The authoritative pairing key. OPPONENT (abbreviation) is for reading;
    # abbreviations are not stable across seasons.
    frame["OPPONENT_TEAM_ID"] = pair_on_game_id(frame, "TEAM_ID")

    frame = drop_unreliable_home_away(frame)
    frame = recompute_plus_minus(frame)
    assert_regular_season_only(frame)

    unpaired = int(frame["OPPONENT_TEAM_ID"].isna().sum())
    if unpaired:
        raise SystemExit(f"{unpaired} rows could not be paired to an opponent")
    print(f"  opponent pairing: every row resolved.")

    frame = frame.drop(columns=COLUMNS_TO_DROP)
    frame = frame.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\nWrote {OUTPUT_PATH}")
    print(f"  rows    : {len(frame):,}")
    print(f"  games   : {frame['GAME_ID'].nunique():,}")
    print(f"  teams   : {frame['TEAM_ID'].nunique()}")
    print(f"  seasons : {frame['SEASON'].min()} .. {frame['SEASON'].max()}")
    print(f"  columns ({len(frame.columns)}): {list(frame.columns)}")

    print("\n  per season:")
    per = frame.groupby("SEASON").agg(
        games=("GAME_ID", "nunique"), teams=("TEAM_ID", "nunique"),
        rows=("GAME_ID", "size"))
    for season, row in per.iterrows():
        print(f"    {season}  {row['games']:>4} games  {row['teams']:>3} teams  "
              f"{row['rows']:>4} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
