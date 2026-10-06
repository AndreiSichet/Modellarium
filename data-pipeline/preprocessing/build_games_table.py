"""Build the canonical games table from the raw per-season CSVs."""

from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUT_PATH = PROCESSED_DATA_DIR / "games_master.csv"

COLUMNS_TO_DROP = ["SEASON_ID", "MIN", "TEAM_ABBREVIATION", "MATCHUP"]

TEAM_KEY = "TEAM_ID"

def load_all_seasons():
    season_files = sorted(RAW_DATA_DIR.glob("games_*.csv"))
    frames = [pd.read_csv(f) for f in season_files]
    return pd.concat(frames, ignore_index=True)

def derive_opponent(df: pd.DataFrame) -> pd.Series:
    """Opponent abbreviation, taken from the other row sharing the same"""

    def other_team(group: pd.DataFrame) -> pd.Series:
        if len(group) != 2:
            return pd.Series(pd.NA, index=group.index)
        return pd.Series(group["TEAM_ABBREVIATION"].values[::-1], index=group.index)

    return df.groupby("GAME_ID", group_keys=False).apply(other_team)

def drop_unreliable_home_away(df: pd.DataFrame) -> pd.DataFrame:
    """Every GAME_ID should have exactly one home team. Where that's not"""

    home_counts = df.groupby("GAME_ID")["IS_HOME"].sum()
    bad_game_ids = home_counts[home_counts != 1].index

    if len(bad_game_ids) == 0:
        print("IS_HOME check: every GAME_ID has exactly one home team. Nothing dropped.")
        return df

    dropped = df[df["GAME_ID"].isin(bad_game_ids)]
    print(
        f"IS_HOME check: dropping {len(bad_game_ids)} games ({len(dropped)} rows) "
        f"with an unreliable home/away signal (not exactly one home team per GAME_ID):"
    )
    print(
        dropped[["GAME_ID", "GAME_DATE", "TEAM_NAME", "MATCHUP", "IS_HOME"]]
        .sort_values(["GAME_ID", "GAME_DATE"])
        .to_string(index=False)
    )

    return df[~df["GAME_ID"].isin(bad_game_ids)]

def recompute_margin(df: pd.DataFrame) -> pd.DataFrame:
    """Replace PLUS_MINUS with the margin derived from PTS, keeping the source.

    LeagueGameFinder's PLUS_MINUS does not agree with the points it reports on
    about 1% of NBA games, the two rows of a game do not mirror each other, and
    neither matches the real margin. PTS and WL are both validated upstream, so
    the quantity is recoverable exactly. The WNBA and G League builders already
    do this; the NBA was the last to carry the raw column. See A7.

    Placed here because this is the earliest table everything downstream reads:
    the rolling features are built from it, and serving recomputes those same
    windows from the served copy. One implementation, no second arithmetic.
    """
    counts = df.groupby("GAME_ID").size()
    wrong_shape = counts[counts != 2]
    if len(wrong_shape):
        raise SystemExit(
            f"{len(wrong_shape)} game(s) without exactly two team rows, so there "
            f"is no opponent to take points from: "
            f"{sorted(int(g) for g in wrong_shape.index)[:10]}"
        )

    # Two rows per game is guaranteed above, so the other side's points are the
    # game's total minus this row's.
    opponent_points = df.groupby("GAME_ID")["PTS"].transform("sum") - df["PTS"]
    # float64, matching the dtype of the column it replaces. A margin is a whole
    # number so int64 reads as the honest type, but games_final.csv is a SERVED
    # artifact and changing a served column's dtype for tidiness is a change
    # nobody asked for.
    derived = (df["PTS"] - opponent_points).astype("float64")

    # Tested BEFORE the WL comparison, not after. A tie is unlabelable whatever
    # WL says, and the G League's 95-95 game walked straight past a WL-validity
    # guard precisely because WL was absent on both rows.
    tied = sorted(int(g) for g in df.loc[derived == 0, "GAME_ID"].unique())
    if tied:
        raise SystemExit(
            f"{len(tied)} game(s) with a recomputed margin of zero. Basketball "
            f"has no draws - overtime decides: {tied[:10]}"
        )

    disagrees = sorted(
        int(g) for g in df.loc[(derived > 0) != df["WL"].eq("W"), "GAME_ID"].unique()
    )
    if disagrees:
        raise SystemExit(
            f"{len(disagrees)} game(s) where the recomputed margin's sign "
            f"disagrees with WL: {disagrees[:10]}"
        )

    sums = derived.groupby(df["GAME_ID"]).sum()
    unbalanced = sums[sums != 0]
    if len(unbalanced):
        raise SystemExit(
            f"{len(unbalanced)} game(s) whose two margins do not sum to zero, so "
            f"the rows were paired wrongly: {sorted(int(g) for g in unbalanced.index)[:10]}"
        )

    source = df["PLUS_MINUS"].astype(float)
    missing = int(source.isna().sum())
    differs = source.notna() & (source != derived.astype(float))
    worst = float((source[differs] - derived[differs]).abs().max()) if differs.any() else 0.0
    print(
        f"PLUS_MINUS: recomputed as PTS - opponent PTS. The source value differed "
        f"on {int(differs.sum()):,} of {len(df):,} rows across "
        f"{df.loc[differs, 'GAME_ID'].nunique():,} of {df['GAME_ID'].nunique():,} "
        f"games (worst error {worst:.0f} points"
        f"{f', {missing:,} rows had no source value' if missing else ''}). "
        f"The source value is kept as PLUS_MINUS_SOURCE."
    )

    df = df.copy()
    df["PLUS_MINUS_SOURCE"] = df["PLUS_MINUS"]
    df["PLUS_MINUS"] = derived
    return df

def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all_seasons()

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    df["IS_HOME"] = df["MATCHUP"].str.contains("vs.", regex=False)
    df["OPPONENT"] = derive_opponent(df)

    df = drop_unreliable_home_away(df)

    df = recompute_margin(df)

    df = df.drop(columns=COLUMNS_TO_DROP)

    df = df.sort_values([TEAM_KEY, "GAME_DATE"], ascending=True).reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['GAME_DATE'].min().date()} to {df['GAME_DATE'].max().date()}")
    print(f"\nSaved to {OUTPUT_PATH}")
    print("\nPreview:")
    print(df.head())

if __name__ == "__main__":
    main()
