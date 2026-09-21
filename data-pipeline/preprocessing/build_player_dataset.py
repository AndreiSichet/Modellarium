"""Assemble the player-prop training table: one row per player per game."""

from pathlib import Path

import pandas as pd

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PLAYER_ROLLING_PATH = PROCESSED_DATA_DIR / "player_boxscores_with_rolling.csv"
GAMES_FINAL_PATH = PROCESSED_DATA_DIR / "games_final.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "player_dataset.csv"

MERGE_KEYS = ["GAME_ID", "TEAM_ID"]

TEAM_CONTEXT_COLUMNS = [
    "IS_HOME",
    "REST_DAYS",
    "IS_BACK_TO_BACK",
    "TEAM_ELO",
    "OPPONENT_ELO",
]

CONTEXT_ALWAYS_PRESENT = [
    "IS_HOME",
    "IS_BACK_TO_BACK",
    "TEAM_ELO",
    "OPPONENT_ELO",
]
CONTEXT_MAY_BE_NAN = ["REST_DAYS"]

ROLLING_STATS = ["MIN", "PTS", "REB", "AST", "FG3M", "PRA"]
WINDOWS = [5, 10]

PLAYER_FEATURE_COLUMNS = [
    f"ROLL{window}_{stat}" for window in WINDOWS for stat in ROLLING_STATS
]
FEATURE_COLUMNS = PLAYER_FEATURE_COLUMNS + TEAM_CONTEXT_COLUMNS

LABEL_COLUMNS = ["MIN_NUMERIC", "PTS", "REB", "AST", "FG3M", "PRA"]

ID_COLUMNS = [
    "GAME_ID",
    "GAME_DATE",
    "SEASON",
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
]

EXPECTED_ROWS = 280_943

def load_players() -> pd.DataFrame:
    """Player-game rows, with merge keys converted deliberately."""
    needed = ID_COLUMNS + LABEL_COLUMNS + PLAYER_FEATURE_COLUMNS
    players = pd.read_csv(
        PLAYER_ROLLING_PATH,
        usecols=needed,
        dtype={"GAME_ID": str, "TEAM_ID": str, "PLAYER_ID": str},
        low_memory=False,
    )

    players["GAME_ID"] = players["GAME_ID"].astype(int)
    players["TEAM_ID"] = players["TEAM_ID"].astype(int)
    players["PLAYER_ID"] = players["PLAYER_ID"].astype(int)
    players["GAME_DATE"] = pd.to_datetime(players["GAME_DATE"])

    print(f"Loaded {len(players):,} player-game rows from "
          f"{PLAYER_ROLLING_PATH.name}")
    return players

def load_team_context() -> pd.DataFrame:
    """One row per team-game: the pre-game context a player inherits."""
    games = pd.read_csv(
        GAMES_FINAL_PATH, usecols=MERGE_KEYS + TEAM_CONTEXT_COLUMNS
    )

    duplicates = int(games.duplicated(subset=MERGE_KEYS).sum())
    if duplicates:
        raise RuntimeError(
            f"games_final.csv has {duplicates:,} duplicate (GAME_ID, TEAM_ID) "
            f"pairs; the many-to-one merge below assumes it is unique."
        )

    print(f"Loaded {len(games):,} team-game context rows from "
          f"{GAMES_FINAL_PATH.name}")
    return games

def attach_team_context(players: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Give every player the pre-game context of the team he played for."""
    before = len(players)
    merged = players.merge(
        games,
        on=MERGE_KEYS,
        how="left",
        validate="many_to_one",
        indicator="_context_merge",
    )

    if len(merged) != before:
        raise RuntimeError(f"merge changed rows: {before:,} -> {len(merged):,}")

    unmatched = merged[merged["_context_merge"] != "both"]
    print(f"\nTeam-context merge: {len(merged):,} rows, "
          f"{len(unmatched):,} without a match")
    if not unmatched.empty:
        sample = unmatched[MERGE_KEYS].drop_duplicates().head(5)
        raise RuntimeError(
            f"{len(unmatched):,} player rows found no team-game context. Every "
            f"player row comes from a game in games_final.csv, so this is a "
            f"failed merge - check the GAME_ID/TEAM_ID dtypes. First few:\n"
            f"{sample.to_string(index=False)}"
        )

    return merged.drop(columns=["_context_merge"])

def drop_players_who_sat(players: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with a real line to predict."""
    played = players["MIN_NUMERIC"].notna()
    dropped = int((~played).sum())
    kept = players[played].reset_index(drop=True)

    print(f"\nDropped {dropped:,} rows where the player did not appear "
          f"({dropped / len(players) * 100:.1f}%), leaving {len(kept):,}.")
    return kept

def check_column_split(dataset: pd.DataFrame) -> None:
    """Every column is exactly one of: id, feature, label."""
    overlap = set(FEATURE_COLUMNS) & set(LABEL_COLUMNS)
    if overlap:
        raise RuntimeError(f"columns are both feature and label: {sorted(overlap)}")

    classified = set(ID_COLUMNS) | set(FEATURE_COLUMNS) | set(LABEL_COLUMNS)
    unclassified = set(dataset.columns) - classified
    missing = classified - set(dataset.columns)
    if unclassified or missing:
        raise RuntimeError(
            "column groups are out of sync with the dataset.\n"
            f"  in the frame but unclassified: {sorted(unclassified)}\n"
            f"  expected but absent: {sorted(missing)}"
        )

def main():
    players = load_players()
    games = load_team_context()

    dataset = attach_team_context(players, games)
    dataset = drop_players_who_sat(dataset)

    dataset = dataset[ID_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMNS]
    check_column_split(dataset)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\nWrote {OUTPUT_PATH}")
    print(f"  rows     : {len(dataset):,}  (expected {EXPECTED_ROWS:,} -> "
          f"{'MATCH' if len(dataset) == EXPECTED_ROWS else 'MISMATCH'})")
    print(f"  columns  : {len(dataset.columns)} "
          f"({len(ID_COLUMNS)} id + {len(FEATURE_COLUMNS)} feature "
          f"+ {len(LABEL_COLUMNS)} label)")
    print(f"  players  : {dataset['PLAYER_ID'].nunique():,}")
    print(f"  games    : {dataset['GAME_ID'].nunique():,}")
    print(f"  seasons  : {dataset['SEASON'].min()} - {dataset['SEASON'].max()}")

    print("\n  NaN in team-context columns that must never be NaN:")
    strict_nan = dataset[CONTEXT_ALWAYS_PRESENT].isna().sum()
    print(strict_nan.to_string())
    print(f"  context check: {'PASS' if strict_nan.sum() == 0 else 'FAIL'}")
    if strict_nan.sum():
        raise RuntimeError(
            "These columns are never NaN in games_final.csv, so a NaN here "
            "means the join produced rows it should not have."
        )

    print("\n  NaN in REST_DAYS (expected: each team's first game in the data):")
    rest_nan = int(dataset["REST_DAYS"].isna().sum())
    affected = dataset.loc[dataset["REST_DAYS"].isna(), "TEAM_ID"].nunique()
    print(f"    {rest_nan:,} player-rows across {affected} teams "
          f"- legitimate, not a merge failure")

    print("\n  NaN in the player rolling features (early-season warm-up, expected):")
    for column in PLAYER_FEATURE_COLUMNS:
        print(f"    {column:<16} {int(dataset[column].isna().sum()):>7,}")

    print("\n  Labels:")
    print(dataset[LABEL_COLUMNS].describe().to_string())

    print("\n  Preview:")
    preview = dataset[dataset["ROLL10_PRA"].notna()].head(5)
    print(preview[["GAME_DATE", "PLAYER_NAME", "TEAM_ABBREVIATION", "IS_HOME",
                   "TEAM_ELO", "ROLL10_PTS", "ROLL10_PRA", "PTS", "PRA"]]
          .to_string(index=False))

if __name__ == "__main__":
    main()
