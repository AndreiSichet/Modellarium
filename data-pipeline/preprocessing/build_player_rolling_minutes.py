"""Combine every player box score into one table and add each player's"""

import sys
from pathlib import Path

import pandas as pd

INGESTION_DIR = Path(__file__).resolve().parents[1] / "ingestion"
sys.path.insert(0, str(INGESTION_DIR))
from fetch_player_boxscores import (  # noqa: E402
    OUTPUT_DIR as BOXSCORE_DIR,
    TARGET_COLUMNS,
    has_played,
)

from build_rolling_features import WINDOWS, derive_season  # noqa: E402

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
GAMES_FINAL_PATH = PROCESSED_DATA_DIR / "games_final.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "player_boxscores_with_rolling.csv"

COUNTING_STATS = ["PTS", "REB", "AST", "FG3M"]

PRA_COLUMN = "PRA"

ROLLING_SOURCES = {
    "MIN_NUMERIC": "MIN",
    "PTS": "PTS",
    "REB": "REB",
    "AST": "AST",
    "FG3M": "FG3M",
    PRA_COLUMN: "PRA",
}

PLAYER_SEASON_KEYS = ["PLAYER_ID", "SEASON"]
SORT_KEYS = ["PLAYER_ID", "SEASON", "GAME_DATE", "GAME_ID"]

REGRESSION_GUARD_COLUMN = "ROLL10_MIN"

PROGRESS_EVERY = 2000

def load_all_boxscores() -> pd.DataFrame:
    """Read every per-game file into one frame."""
    paths = sorted(BOXSCORE_DIR.glob("*.csv"))
    print(f"Reading {len(paths):,} box score files...")

    frames = []
    for i, path in enumerate(paths, start=1):
        frames.append(pd.read_csv(path, dtype=str, keep_default_na=False))
        if i % PROGRESS_EVERY == 0:
            print(f"  {i:,} / {len(paths):,}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  combined: {len(combined):,} player-rows\n")
    return combined

def attach_game_date(players: pd.DataFrame) -> pd.DataFrame:
    """Join GAME_DATE in from the already-trusted game table."""
    games = pd.read_csv(GAMES_FINAL_PATH, usecols=["GAME_ID", "GAME_DATE"])
    games["GAME_ID"] = games["GAME_ID"].astype(str).str.zfill(10)

    before = len(games)
    games = games.drop_duplicates(subset="GAME_ID")
    print(f"games_final.csv: {before:,} rows -> {len(games):,} unique GAME_IDs")

    merged = players.merge(games, on="GAME_ID", how="left", validate="m:1")

    if len(merged) != len(players):
        raise RuntimeError(
            f"join changed the row count: {len(players):,} -> {len(merged):,}"
        )

    unmatched = merged["GAME_DATE"].isna().sum()
    if unmatched:
        raise RuntimeError(f"{unmatched:,} player-rows have no matching GAME_DATE")

    merged["GAME_DATE"] = pd.to_datetime(merged["GAME_DATE"])
    return merged

def parse_minutes(players: pd.DataFrame) -> pd.DataFrame:
    """MIN ("MM:SS") -> MIN_NUMERIC (float), NaN where the player sat."""
    played = has_played(players["MIN"])

    parts = players.loc[played, "MIN"].str.split(":", expand=True)
    minutes = pd.to_numeric(parts[0], errors="coerce")
    seconds = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else 0

    players["MIN_NUMERIC"] = float("nan")
    players.loc[played, "MIN_NUMERIC"] = minutes + seconds / 60.0

    unparsed = played & players["MIN_NUMERIC"].isna()
    if unparsed.any():
        examples = players.loc[unparsed, "MIN"].head(5).tolist()
        raise RuntimeError(
            f"{int(unparsed.sum()):,} rows have minutes that would not parse, "
            f"e.g. {examples}"
        )

    return players

def parse_counting_stats(players: pd.DataFrame) -> pd.DataFrame:
    """Numeric PTS/REB/AST/FG3M plus PRA, all NaN where the player sat."""
    played = has_played(players["MIN"])

    for column in COUNTING_STATS:
        players[column] = pd.to_numeric(players[column], errors="coerce")
        players.loc[~played, column] = float("nan")

        unparsed = played & players[column].isna()
        if unparsed.any():
            raise RuntimeError(
                f"{int(unparsed.sum()):,} rows played but have no parseable "
                f"{column}. MIN and the counting stats must agree about whether "
                f"a player appeared."
            )

    players[PRA_COLUMN] = players["PTS"] + players["REB"] + players["AST"]

    return players

def add_rolling(players: pd.DataFrame) -> pd.DataFrame:
    """Trailing averages over played games, carried across absences."""
    players = players.sort_values(SORT_KEYS).reset_index(drop=True)
    played = players["MIN_NUMERIC"].notna()
    played_only = players.loc[played]

    for window in WINDOWS:
        for source, suffix in ROLLING_SOURCES.items():
            column = f"ROLL{window}_{suffix}"

            rolled = played_only.groupby(PLAYER_SEASON_KEYS)[source].transform(
                lambda s, w=window: s.shift(1).rolling(w).mean()
            )

            players[column] = float("nan")
            players.loc[played, column] = rolled

            players[column] = players.groupby(PLAYER_SEASON_KEYS)[column].ffill()

    return players

def expected_nan_count(players: pd.DataFrame, window: int) -> int:
    """How many rows must be NaN, derived rather than eyeballed."""
    played = players["MIN_NUMERIC"].notna().astype(int)
    appearances_so_far = played.groupby(
        [players["PLAYER_ID"], players["SEASON"]]
    ).cumsum()
    return int((appearances_so_far <= window).sum())

def check_pra_linearity(players: pd.DataFrame) -> bool:
    """ROLL{w}_PRA must equal ROLL{w}_PTS + ROLL{w}_REB + ROLL{w}_AST."""
    print("\n  PRA linearity (mean is linear, so this must hold exactly):")
    all_ok = True

    for window in WINDOWS:
        direct = players[f"ROLL{window}_PRA"]
        summed = (players[f"ROLL{window}_PTS"]
                  + players[f"ROLL{window}_REB"]
                  + players[f"ROLL{window}_AST"])

        both_nan = direct.isna() & summed.isna()
        difference = (direct - summed).abs()
        worst = float(difference.max(skipna=True))
        disagreeing = int((~both_nan & ~(difference < 1e-9)).sum())

        all_ok &= disagreeing == 0
        print(f"    ROLL{window}_PRA vs sum of parts: {disagreeing:,} disagreeing rows, "
              f"largest difference {worst:.2e}  {'OK' if disagreeing == 0 else 'FAIL'}")

    return all_ok

def check_minutes_regression(players: pd.DataFrame) -> None:
    """ROLL10_MIN must be byte-identical to the previous run's."""
    if not OUTPUT_PATH.exists():
        print(f"\n  {REGRESSION_GUARD_COLUMN} regression check: no previous output "
              f"to compare against (first run).")
        return

    previous = pd.read_csv(
        OUTPUT_PATH,
        usecols=["GAME_ID", "PLAYER_ID", REGRESSION_GUARD_COLUMN],
        dtype={"GAME_ID": str, "PLAYER_ID": str},
    )
    merged = players[["GAME_ID", "PLAYER_ID", REGRESSION_GUARD_COLUMN]].merge(
        previous, on=["GAME_ID", "PLAYER_ID"], how="inner",
        suffixes=("_new", "_old"), validate="one_to_one",
    )

    new = merged[f"{REGRESSION_GUARD_COLUMN}_new"]
    old = merged[f"{REGRESSION_GUARD_COLUMN}_old"]
    both_nan = new.isna() & old.isna()
    differing = int((~both_nan & ~((new - old).abs() < 1e-9)).sum())

    print(f"\n  {REGRESSION_GUARD_COLUMN} regression check "
          f"(availability features depend on this):")
    print(f"    rows compared: {len(merged):,}   differing: {differing:,}  "
          f"{'UNCHANGED' if differing == 0 else 'CHANGED - investigate'}")

def main():
    players = load_all_boxscores()
    players = attach_game_date(players)
    players["SEASON"] = derive_season(players["GAME_DATE"])
    players = parse_minutes(players)
    players = parse_counting_stats(players)
    players = add_rolling(players)

    rolling_columns = [
        f"ROLL{window}_{suffix}"
        for window in WINDOWS
        for suffix in ROLLING_SOURCES.values()
    ]
    output_columns = (
        TARGET_COLUMNS
        + ["GAME_DATE", "SEASON", "MIN_NUMERIC", PRA_COLUMN]
        + rolling_columns
    )
    players = players[output_columns]

    check_minutes_regression(players)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    players.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\nWrote {OUTPUT_PATH}")
    print(f"  rows              : {len(players):,}")
    print(f"  columns           : {len(players.columns)} "
          f"({len(rolling_columns)} rolling)")
    print(f"  date range        : {players['GAME_DATE'].min().date()} "
          f"-> {players['GAME_DATE'].max().date()}")
    print(f"  unique players    : {players['PLAYER_ID'].nunique():,}")
    print(f"  player-seasons    : {players.groupby(PLAYER_SEASON_KEYS).ngroups:,}")
    print(f"  rows with minutes : {int(players['MIN_NUMERIC'].notna().sum()):,}")

    print("\n  NaN counts (early-season warm-up, by design):")
    all_match = True
    for window in WINDOWS:
        expected = expected_nan_count(players, window)
        for suffix in ROLLING_SOURCES.values():
            column = f"ROLL{window}_{suffix}"
            actual = int(players[column].isna().sum())
            ok = actual == expected
            all_match &= ok
            print(f"    {column:<16} {actual:>7,}  expected {expected:>7,}  "
                  f"{'OK' if ok else 'MISMATCH'}")
    print(f"\n  all NaN counts as predicted: {all_match}")

    linear_ok = check_pra_linearity(players)

    print("\n  Preview (a player past the ROLL10 warm-up):")
    preview = players[players["ROLL10_PRA"].notna()].head(6)
    print(preview[["GAME_DATE", "PLAYER_NAME", "MIN", "PTS", "REB", "AST", "PRA",
                   "ROLL5_PTS", "ROLL10_PTS", "ROLL10_PRA"]].to_string(index=False))

    if not (all_match and linear_ok):
        raise SystemExit("Checks failed - see above.")

if __name__ == "__main__":
    main()
