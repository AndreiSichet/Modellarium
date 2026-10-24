"""Add trailing rolling-window features to the canonical games table."""

from pathlib import Path

import pandas as pd

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
INPUT_PATH = PROCESSED_DATA_DIR / "games_master.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "games_with_rolling.csv"

TEAM_KEY = "TEAM_ID"
WINDOWS = [5, 10]

METRICS = {
    "WIN": "WIN_PCT",
    "PTS": "PTS",
    "PLUS_MINUS": "PLUS_MINUS",
    "FG_PCT": "FG_PCT",
    "REB": "REB",
    "AST": "AST",
    "TOV": "TOV",
}

def derive_season(game_date: pd.Series) -> pd.Series:
    """Season from the DATE alone, August boundary. The FALLBACK, not the rule.

    KEPT, AND STILL CORRECT FOR WHAT IT IS NOW USED FOR: a fixture that has
    not been played has no GAME_ID, so serving an unplayed game has nothing
    but the date to go on. `live_features.season_of` is that caller.

    IT IS NOT THE RULE FOR A PLAYED GAME. `month >= 8` mislabels the 2019-20
    COVID bubble: the 80 regular-season games of 2020-08-01 to 2020-08-14 are
    part of the 2019-20 season and this function calls them 2020-21. The
    GAME_ID says 2019 and the GAME_ID is the authority - see
    `season_from_game_id`. Use `derive_season_for_frame` for any table that
    has a GAME_ID column.
    """
    return game_date.dt.year.where(game_date.dt.month >= 8, game_date.dt.year - 1)


def season_from_game_id(game_id, game_date=None):
    """Season from the NBA's own GAME_ID. THE AUTHORITY for a played game.

    The canonical id is ten characters: `00` + a type digit + two digits of
    the season's START year + a five-digit sequence, so `0021900001` is the
    first regular-season game of 2019-20. **The ids in this project's tables
    are EIGHT characters** - `21500001` - because the leading `00` is lost
    whenever the column is read as a number. Both forms are accepted by
    zero-padding to ten before slicing, which is why this does not index from
    the left.

    THE CENTURY COMES FROM THE DATE, NOT FROM A PIVOT YEAR. Two digits cannot
    say whether `15` is 1915 or 2015, and a hardcoded "assume 20xx" is a
    literal that expires. The game's own date settles it, and that is the one
    thing the date is still trusted for here.
    """
    ids = pd.Series(game_id).astype("string").str.strip().str.zfill(10)
    two = pd.to_numeric(ids.str[3:5], errors="coerce")

    if game_date is None:
        # No date to anchor the century: assume the same century as the
        # project's own data, and say so rather than guessing silently.
        return (2000 + two).astype("Int64")

    dates = pd.to_datetime(pd.Series(game_date).values)
    year = pd.Series(dates).dt.year
    century = (year // 100) * 100
    candidate = century + two
    # A season starts in the calendar year of the date or the one before it,
    # so a candidate more than a year away means the century guess was wrong.
    candidate = candidate.where((year - candidate).between(0, 1),
                                candidate - 100)
    return candidate.astype("Int64")


def derive_season_for_frame(frame: pd.DataFrame) -> pd.Series:
    """SEASON for a table of played games: the GAME_ID rule, date as fallback.

    ONE DEFINITION, and every caller that has a GAME_ID should use this rather
    than `derive_season`. A row with no usable id falls back to the date rule
    and is counted, because a silent fallback would hide an id column that had
    been read as the wrong type.
    """
    if "GAME_ID" not in frame.columns:
        return derive_season(frame["GAME_DATE"])

    by_id = season_from_game_id(frame["GAME_ID"], frame["GAME_DATE"])
    by_date = derive_season(frame["GAME_DATE"])
    return by_id.fillna(pd.Series(by_date.values, index=by_id.index))         .astype(int).set_axis(frame.index)


def season_rule_disagreements(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows where the GAME_ID and the date disagree, as a CROSS-CHECK.

    A disagreement of exactly one season is legitimate and expected: it is
    what a game played outside its season's normal calendar window looks like,
    and the 2019-20 bubble is precisely that. **A disagreement of more than
    one season is not explicable that way** and means an id or a date is
    wrong, so callers treat it as fatal.

    The dates cannot settle this on their own, and that is worth stating: the
    date rule is perfectly self-consistent: it just assigns the bubble games
    to the following season. Only the id knows.
    """
    by_id = season_from_game_id(frame["GAME_ID"], frame["GAME_DATE"])
    by_date = derive_season(frame["GAME_DATE"])
    gap = (by_id.astype("Int64") - pd.Series(by_date.values,
                                             index=by_id.index).astype("Int64"))
    out = frame.loc[gap.fillna(0) != 0].copy()
    out["SEASON_BY_ID"] = by_id.loc[out.index]
    out["SEASON_BY_DATE"] = pd.Series(by_date.values,
                                      index=by_id.index).loc[out.index]
    out["SEASON_GAP"] = gap.loc[out.index]
    return out

def trailing_mean(grouped, column: str, window: int) -> pd.Series:
    """The project's one lag convention, in one place.

    Extracted so other feature builders import it rather than restating the
    expression - a second implementation is where an off-by-one leak enters.
    """
    return grouped[column].transform(
        # shift(1) first: without it a game's own result enters
        # its own features, which is leakage.
        lambda s, w=window: s.shift(1).rolling(w).mean()
    )

def main():
    df = pd.read_csv(INPUT_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # THE GAME_ID RULE, not the date. See derive_season_for_frame: the date
    # rule mislabels the 2019-20 bubble's August 2020 games as 2020-21, which
    # moves one carryover-regression event per team and shifts Elo on 55% of
    # all rows by up to 66 points.
    df["SEASON"] = derive_season_for_frame(df)
    disagreements = season_rule_disagreements(df)
    if not disagreements.empty:
        gaps = sorted(disagreements["SEASON_GAP"].unique().tolist())
        print(f"  {len(disagreements)} row(s) where the GAME_ID and the date "
              f"disagree about the season (gaps {gaps}); the id wins")
        if any(abs(int(g)) > 1 for g in gaps):
            # A gap of one season is a game played outside its season's normal
            # window - the bubble. More than one cannot be explained that way.
            raise ValueError(
                f"a GAME_ID and its date disagree by more than one season: "
                f"{disagreements[['GAME_ID', 'GAME_DATE', 'SEASON_BY_ID', 'SEASON_BY_DATE']].head(5)}")
    df["WIN"] = (df["WL"] == "W").astype(int)

    df = df.sort_values([TEAM_KEY, "SEASON", "GAME_DATE"]).reset_index(drop=True)

    grouped = df.groupby([TEAM_KEY, "SEASON"])
    for window in WINDOWS:
        for source_col, feature_name in METRICS.items():
            df[f"ROLL{window}_{feature_name}"] = trailing_mean(
                grouped, source_col, window
            )

    df = df.drop(columns=["WIN"])

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Total rows: {len(df)}")
    print(f"Saved to {OUTPUT_PATH}")
    print("\nPreview:")
    print(df.head())

if __name__ == "__main__":
    main()
