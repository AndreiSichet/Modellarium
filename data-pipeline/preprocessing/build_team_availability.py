"""Turn per-player availability into a team-level feature, one row per"""

from pathlib import Path

import pandas as pd

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
INPUT_PATH = PROCESSED_DATA_DIR / "player_boxscores_with_rolling.csv"
GAMES_FINAL_PATH = PROCESSED_DATA_DIR / "games_final.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "team_availability.csv"

ROLLING_COLUMN = "ROLL10_MIN"
GROUP_KEYS = ["GAME_ID", "TEAM_ID"]

def universe_team_games() -> int:
    """How many team-games exist, read from games_final.csv.

    REPLACES A HARDCODED 26,398. That literal was print-only - a percentage
    denominator and a MATCH/MISMATCH label - so unlike the one in
    build_quarter_half_rolling it would not have FAILED anything on the first
    new NBA day. It would have printed "MISMATCH" and a slightly wrong
    percentage every run from the 21st onward, which is its own problem: a
    report that cries mismatch every day is a report people stop reading.

    Derived from the same table the availability rows are built against, so
    it is right by construction rather than right until the schedule moves.
    """
    import pandas as pd

    return len(pd.read_csv(GAMES_FINAL_PATH, usecols=["GAME_ID"]))

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

def absence_weights(players: pd.DataFrame, weight_column: str = ROLLING_COLUMN) -> pd.Series:
    """Per-player absence weight: the rolling column where absent, else 0.

    An unknown role weighs 0 rather than NaN, which is a real and measured
    deflation early in a season - see the ROLL10_MIN warm-up in CLAUDE.md.
    """
    return players[weight_column].where(players["IS_ABSENT"]).fillna(0.0)


def reduce_absence_weights(players: pd.DataFrame, weights: pd.Series,
                           how="sum") -> pd.DataFrame:
    """Collapse per-absence weights into one number per team-game.

    The reduction is a parameter because the aggregation study varies it while
    the pipeline always sums. Extracted rather than forked so the two cannot
    drift apart - the same reason trailing_mean() is shared.
    """
    frame = players[GROUP_KEYS].assign(ABSENT_WEIGHT=weights)
    return frame.groupby(GROUP_KEYS, as_index=False).agg(
        WEIGHTED=("ABSENT_WEIGHT", how))


def build_team_availability(players: pd.DataFrame) -> pd.DataFrame:
    """Collapse player rows into one row per team-game."""
    players["IS_ABSENT"] = players["MIN_NUMERIC"].isna()
    players["ABSENT_WEIGHT"] = absence_weights(players)

    counts = (
        players.groupby(GROUP_KEYS, as_index=False)
        .agg(ABSENT_COUNT=("IS_ABSENT", "sum"))
    )
    weighted = reduce_absence_weights(players, players["ABSENT_WEIGHT"], how="sum")

    availability = counts.merge(weighted, on=GROUP_KEYS, validate="one_to_one")
    availability = availability.rename(columns={"WEIGHTED": "WEIGHTED_ABSENT_MIN"})
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
    # Denominated by the team-games actually present in this player table,
    # which is self-contained and needs no second file.
    all_team_games = len(players[GROUP_KEYS].drop_duplicates())
    print(f"  team-games touched by that     : {len(affected_team_games):,} "
          f"({len(affected_team_games) / all_team_games * 100:.1f}% of "
          f"{all_team_games:,})")

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
    expected = universe_team_games()
    print(f"  rows     : {len(availability):,}  "
          f"(games_final.csv has {expected:,} -> "
          f"{'MATCH' if len(availability) == expected else 'MISMATCH'})")
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
