"""Turn per-player availability into a team-level feature, one row per"""

from pathlib import Path

import pandas as pd

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
INPUT_PATH = PROCESSED_DATA_DIR / "player_boxscores_with_rolling.csv"
GAMES_FINAL_PATH = PROCESSED_DATA_DIR / "games_final.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "team_availability.csv"

ROLLING_COLUMN = "ROLL10_MIN"
GROUP_KEYS = ["GAME_ID", "TEAM_ID"]

EXPECTED_TEAM_GAMES = 26_398

def load_players() -> pd.DataFrame:
    """Load the player-game table, keeping GAME_ID padded."""
    players = pd.read_csv(
        INPUT_PATH,
        dtype={"GAME_ID": str, "TEAM_ID": str, "PLAYER_ID": str},
        low_memory=False,
    )
    for column in ("MIN_NUMERIC", ROLLING_COLUMN):
        players[column] = pd.to_numeric(players[column], errors="coerce")

    print(f"Loaded {len(players):,} player-rows from {INPUT_PATH.name}")
    return players

def build_team_availability(players: pd.DataFrame) -> pd.DataFrame:
    """Collapse player rows into one row per team-game."""
    players["IS_ABSENT"] = players["MIN_NUMERIC"].isna()

    players["ABSENT_WEIGHT"] = (
        players[ROLLING_COLUMN].where(players["IS_ABSENT"]).fillna(0.0)
    )

    availability = (
        players.groupby(GROUP_KEYS, as_index=False)
        .agg(
            ABSENT_COUNT=("IS_ABSENT", "sum"),
            WEIGHTED_ABSENT_MIN=("ABSENT_WEIGHT", "sum"),
        )
    )
    availability["ABSENT_COUNT"] = availability["ABSENT_COUNT"].astype(int)

    return availability

def report_unknown_roles(players: pd.DataFrame):
    """How often the fill_value=0 choice is actually exercised."""
    absent = players["IS_ABSENT"]
    unknown = absent & players[ROLLING_COLUMN].isna()

    print(f"\n  absent player-rows            : {int(absent.sum()):,}")
    print(f"  of those, no ROLL10_MIN yet   : {int(unknown.sum()):,} "
          f"({unknown.sum() / absent.sum() * 100:.1f}% of absences) -> weighted as 0")

    affected_team_games = players.loc[unknown, GROUP_KEYS].drop_duplicates()
    print(f"  team-games touched by that     : {len(affected_team_games):,} "
          f"({len(affected_team_games) / EXPECTED_TEAM_GAMES * 100:.1f}%)")

def sanity_check_against_plus_minus(availability: pd.DataFrame):
    """Does missing more, heavier players actually go with playing worse?"""
    games = pd.read_csv(
        GAMES_FINAL_PATH,
        usecols=["GAME_ID", "TEAM_ID", "PLUS_MINUS"],
        dtype={"TEAM_ID": str},
    )
    games["GAME_ID"] = games["GAME_ID"].astype(str).str.zfill(10)

    merged = availability.merge(games, on=GROUP_KEYS, how="left", validate="1:1")
    unmatched = merged["PLUS_MINUS"].isna().sum()
    if unmatched:
        raise RuntimeError(f"{unmatched:,} team-games have no PLUS_MINUS to check against")

    print("\n" + "=" * 66)
    print("SANITY CHECK vs actual PLUS_MINUS (same game)")
    print("=" * 66)
    for column in ("WEIGHTED_ABSENT_MIN", "ABSENT_COUNT"):
        r = merged[column].corr(merged["PLUS_MINUS"])
        direction = "negative (expected)" if r < 0 else "POSITIVE - investigate"
        print(f"  corr({column:<20}, PLUS_MINUS) = {r:+.4f}   {direction}")

    quartiles = pd.qcut(merged["WEIGHTED_ABSENT_MIN"], 4, labels=False, duplicates="drop")
    print("\n  mean PLUS_MINUS by WEIGHTED_ABSENT_MIN quartile:")
    for q, group in merged.groupby(quartiles):
        print(f"    Q{int(q) + 1}  n={len(group):>6,}  "
              f"absent-min {group['WEIGHTED_ABSENT_MIN'].min():>6.1f}-"
              f"{group['WEIGHTED_ABSENT_MIN'].max():>6.1f}  "
              f"mean PLUS_MINUS {group['PLUS_MINUS'].mean():+.3f}")

def main():
    players = load_players()
    availability = build_team_availability(players)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    availability.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\nWrote {OUTPUT_PATH}")
    print(f"  rows     : {len(availability):,}  "
          f"(expected {EXPECTED_TEAM_GAMES:,} -> "
          f"{'MATCH' if len(availability) == EXPECTED_TEAM_GAMES else 'MISMATCH'})")
    print(f"  games    : {availability['GAME_ID'].nunique():,}")
    print(f"  teams    : {availability['TEAM_ID'].nunique()}")

    report_unknown_roles(players)

    print("\n  WEIGHTED_ABSENT_MIN distribution:")
    print(availability["WEIGHTED_ABSENT_MIN"].describe().to_string())
    print("\n  ABSENT_COUNT distribution:")
    print(availability["ABSENT_COUNT"].describe().to_string())

    sanity_check_against_plus_minus(availability)

if __name__ == "__main__":
    main()
