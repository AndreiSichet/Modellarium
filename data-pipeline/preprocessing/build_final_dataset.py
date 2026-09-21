"""Reshape games_final.csv from one-row-per-team-per-game into one-row-per-game,"""

from pathlib import Path

import pandas as pd

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
INPUT_PATH = PROCESSED_DATA_DIR / "games_final.csv"
AVAILABILITY_PATH = PROCESSED_DATA_DIR / "team_availability.csv"
ADVANCED_ROLLING_PATH = PROCESSED_DATA_DIR / "team_advanced_rolling.csv"
QUARTER_HALF_PATH = PROCESSED_DATA_DIR / "quarter_half_raw.csv"
QUARTER_HALF_ROLLING_PATH = PROCESSED_DATA_DIR / "quarter_half_rolling.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "model_dataset.csv"
QUARTER_SCORES_FAILURES_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "raw"
    / "quarter_scores_failures.log"
)

PREGAME_FEATURE_COLUMNS = [
    "REST_DAYS",
    "IS_BACK_TO_BACK",
    "TEAM_ELO",
    "ABSENT_COUNT",
    "WEIGHTED_ABSENT_MIN",
]

AVAILABILITY_COLUMNS = ["ABSENT_COUNT", "WEIGHTED_ABSENT_MIN"]

POSTGAME_OUTCOME_COLUMNS = ["WL", "PTS", "REB", "AST", "Q1_PTS", "HALF1_PTS"]

ID_COLUMNS = ["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_NAME"]

GAME_LEVEL_COLUMNS = ["GAME_ID", "GAME_DATE"]

LABEL_COLUMNS = [
    "HOME_WIN",
    "HOME_PTS",
    "AWAY_PTS",
    "HOME_MARGIN",
    "TOTAL_PTS",
    "HOME_REB",
    "AWAY_REB",
    "REB_MARGIN",
    "TOTAL_REB",
    "HOME_AST",
    "AWAY_AST",
    "AST_MARGIN",
    "TOTAL_AST",
    "HOME_Q1_PTS",
    "AWAY_Q1_PTS",
    "HOME_Q1_MARGIN",
    "TOTAL_Q1_PTS",
    "HOME_Q1_WIN",
    "HOME_HALF1_PTS",
    "AWAY_HALF1_PTS",
    "HOME_HALF1_MARGIN",
    "TOTAL_HALF1_PTS",
    "HOME_HALF1_WIN",
]

QUARTER_HALF_LABELS = [c for c in LABEL_COLUMNS if "Q1" in c or "HALF1" in c]

def attach_availability(df: pd.DataFrame) -> pd.DataFrame:
    """Join the team-level availability features onto each team-game row."""
    availability = pd.read_csv(AVAILABILITY_PATH, dtype={"GAME_ID": str})
    availability["GAME_ID"] = availability["GAME_ID"].astype(int)

    before = len(df)
    merged = df.merge(
        availability, on=["GAME_ID", "TEAM_ID"], how="left", validate="one_to_one"
    )

    if len(merged) != before:
        raise RuntimeError(f"availability join changed rows: {before} -> {len(merged)}")

    missing = int(merged[AVAILABILITY_COLUMNS].isna().any(axis=1).sum())
    if missing:
        raise RuntimeError(
            f"{missing:,} team-game rows got no availability data. These columns "
            f"have no legitimate NaN, so this is a failed merge - check that the "
            f"GAME_ID dtypes match on both sides."
        )

    print(f"Availability merged: {len(availability):,} team-games, 0 unmatched.")
    return merged

def attach_advanced_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Join the pace/rating rolling features onto each team-game row."""
    advanced = pd.read_csv(ADVANCED_ROLLING_PATH, dtype={"GAME_ID": str})
    advanced["GAME_ID"] = advanced["GAME_ID"].astype(int)

    before = len(df)
    merged = df.merge(
        advanced,
        on=["GAME_ID", "TEAM_ID"],
        how="left",
        validate="one_to_one",
        indicator="_advanced_merge",
    )

    if len(merged) != before:
        raise RuntimeError(f"advanced join changed rows: {before} -> {len(merged)}")

    unmatched = int((merged["_advanced_merge"] != "both").sum())

    merged = merged.drop(columns=["_advanced_merge"])
    added = [c for c in advanced.columns if c not in ("GAME_ID", "TEAM_ID")]
    print(f"Advanced rolling merged: {len(advanced):,} team-games, "
          f"{len(added)} columns, {unmatched:,} unmatched.")
    if unmatched:
        print(f"  {unmatched:,} row(s) have no advanced-stats match and will carry "
              f"NaN in those columns. Not fatal: they are held out of "
              f"FEATURE_COLUMNS and feed no prediction. A count near "
              f"{len(merged):,} would mean a broken merge instead.")
    return merged

def read_known_fetch_failures() -> set:
    """Game ids fetch_quarter_scores.py has ever failed on."""
    if not QUARTER_SCORES_FAILURES_PATH.exists():
        return set()

    failures = set()
    with QUARTER_SCORES_FAILURES_PATH.open(encoding="utf-8") as log:
        for number, line in enumerate(log, start=1):
            text = line.strip()
            if not text:
                continue
            game_id, _, reason = text.partition("\t")
            if not reason or not game_id.isdigit():
                raise RuntimeError(
                    f"{QUARTER_SCORES_FAILURES_PATH.name} line {number} is not "
                    f"'<game_id>\\t<reason>': {line!r}. Refusing to guess which "
                    f"games are allowed to be missing."
                )
            failures.add(int(game_id))
    return failures

def attach_quarter_half(df: pd.DataFrame) -> pd.DataFrame:
    """Join Q1 and first-half scoring onto each team-game row."""
    quarters = pd.read_csv(QUARTER_HALF_PATH, dtype={"GAME_ID": str})
    quarters["GAME_ID"] = quarters["GAME_ID"].astype(int)

    before = len(df)
    merged = df.merge(
        quarters,
        on=["GAME_ID", "TEAM_ID"],
        how="left",
        validate="one_to_one",
        indicator="_quarter_merge",
    )

    if len(merged) != before:
        raise RuntimeError(f"quarter join changed rows: {before} -> {len(merged)}")

    unmatched = merged.loc[merged["_quarter_merge"] != "both", "GAME_ID"]
    unmatched_games = set(unmatched)
    known_failures = read_known_fetch_failures()

    unexpected = sorted(unmatched_games - known_failures)
    if unexpected:
        raise RuntimeError(
            f"quarter/half merge left {len(unmatched):,} rows unmatched across "
            f"{len(unmatched_games)} game(s), and {len(unexpected)} of those "
            f"game(s) are NOT in {QUARTER_SCORES_FAILURES_PATH.name}: "
            f"{unexpected[:10]}. Either the fetch genuinely failed without "
            f"being logged, or the GAME_ID dtypes disagree on the two sides."
        )

    if len(unmatched) != 2 * len(unmatched_games):
        raise RuntimeError(
            f"quarter/half merge left {len(unmatched):,} unmatched rows across "
            f"{len(unmatched_games)} game(s); expected exactly 2 per game. "
            f"A game should strand its home and away rows together or not at "
            f"all, so this means the merge key is wrong."
        )

    merged = merged.drop(columns=["_quarter_merge"])
    print(f"Quarter/half merged: {len(quarters):,} team-games, "
          f"{len(unmatched)} unmatched row(s) across {len(unmatched_games)} game(s), "
          f"all present in {QUARTER_SCORES_FAILURES_PATH.name} "
          f"(which records {len(known_failures)} distinct game(s)).")
    return merged

def attach_quarter_half_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Join trailing Q1/1H form onto each team-game row."""
    rolling = pd.read_csv(QUARTER_HALF_ROLLING_PATH, dtype={"GAME_ID": str})
    rolling["GAME_ID"] = rolling["GAME_ID"].astype(int)

    before = len(df)
    merged = df.merge(
        rolling,
        on=["GAME_ID", "TEAM_ID"],
        how="left",
        validate="one_to_one",
        indicator="_qh_rolling_merge",
    )

    if len(merged) != before:
        raise RuntimeError(
            f"quarter/half rolling join changed rows: {before} -> {len(merged)}"
        )

    unmatched = int((merged["_qh_rolling_merge"] != "both").sum())
    if unmatched:
        raise RuntimeError(
            f"{unmatched:,} team-game rows found no quarter/half rolling match. "
            f"That file is built against the full universe, so this is a failed "
            f"merge, not a warm-up gap - check the GAME_ID dtypes."
        )

    merged = merged.drop(columns=["_qh_rolling_merge"])
    added = [c for c in rolling.columns if c not in ("GAME_ID", "TEAM_ID")]
    print(f"Quarter/half rolling merged: {len(rolling):,} team-games, "
          f"{len(added)} columns, 0 unmatched.")
    return merged

def home_win_label(home_points: pd.Series, away_points: pd.Series) -> pd.Series:
    """1 if the home team led this period, 0 if it trailed, NA if tied."""
    known = home_points.notna() & away_points.notna()
    decided = known & (home_points != away_points)
    return (home_points > away_points).astype("Int64").where(decided)

def add_period_labels(merged: pd.DataFrame, period: str) -> pd.DataFrame:
    """Margin / total / win for one period, mirroring the full-game trio."""
    home, away = f"HOME_{period}_PTS", f"AWAY_{period}_PTS"
    merged[f"HOME_{period}_MARGIN"] = merged[home] - merged[away]
    merged[f"TOTAL_{period}_PTS"] = merged[home] + merged[away]
    merged[f"HOME_{period}_WIN"] = home_win_label(merged[home], merged[away])
    return merged

def build_keep_columns(df: pd.DataFrame) -> list:
    rolling_cols = [c for c in df.columns if c.startswith("ROLL5_") or c.startswith("ROLL10_")]
    return ID_COLUMNS + POSTGAME_OUTCOME_COLUMNS + rolling_cols + PREGAME_FEATURE_COLUMNS

def split_and_prefix(df: pd.DataFrame, keep_cols: list, prefix: str) -> pd.DataFrame:
    subset = df.loc[df["IS_HOME"] == (prefix == "HOME"), keep_cols]
    rename_map = {c: f"{prefix}_{c}" for c in keep_cols if c not in GAME_LEVEL_COLUMNS}
    return subset.rename(columns=rename_map)

def main():
    df = pd.read_csv(INPUT_PATH)
    df = attach_availability(df)
    df = attach_advanced_rolling(df)
    df = attach_quarter_half(df)
    df = attach_quarter_half_rolling(df)

    keep_cols = build_keep_columns(df)

    home = split_and_prefix(df, keep_cols, "HOME")
    away = split_and_prefix(df, keep_cols, "AWAY")

    merged = home.merge(away, on=GAME_LEVEL_COLUMNS, how="inner", validate="one_to_one")

    merged["HOME_WIN"] = (merged["HOME_WL"] == "W").astype(int)
    merged["HOME_MARGIN"] = merged["HOME_PTS"] - merged["AWAY_PTS"]
    merged["TOTAL_PTS"] = merged["HOME_PTS"] + merged["AWAY_PTS"]
    merged["REB_MARGIN"] = merged["HOME_REB"] - merged["AWAY_REB"]
    merged["TOTAL_REB"] = merged["HOME_REB"] + merged["AWAY_REB"]
    merged["AST_MARGIN"] = merged["HOME_AST"] - merged["AWAY_AST"]
    merged["TOTAL_AST"] = merged["HOME_AST"] + merged["AWAY_AST"]
    for period in ("Q1", "HALF1"):
        merged = add_period_labels(merged, period)
    merged = merged.drop(columns=["HOME_WL", "AWAY_WL"])

    merged.to_csv(OUTPUT_PATH, index=False)

    expected_rows = len(df) // 2
    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(merged)} (expected {expected_rows})")
    print(
        "Merge check: "
        + ("PASS, no games dropped or duplicated." if len(merged) == expected_rows else "FAIL, mismatch.")
    )

    print(f"\nSaved to {OUTPUT_PATH}")

    print("\nNaN counts (early-season rolling windows still to be decided later):")
    print(merged[["HOME_ROLL5_PTS", "AWAY_ROLL5_PTS"]].isna().sum())

    availability_output = [
        f"{side}_{col}" for side in ("HOME", "AWAY") for col in AVAILABILITY_COLUMNS
    ]
    availability_nan = merged[availability_output].isna().sum()
    print("NaN counts in availability features (must all be 0):")
    print(availability_nan.to_string())
    print("Availability check: " + ("PASS" if availability_nan.sum() == 0 else "FAIL"))

    print("\nQuarter/half label NaN, by cause:")
    no_data = merged["HOME_Q1_PTS"].isna()
    print(f"  games with no quarter data : {int(no_data.sum())}  "
          f"(affects all {len(QUARTER_HALF_LABELS)} columns)")
    for period in ("Q1", "HALF1"):
        tied = (~no_data) & (merged[f"HOME_{period}_PTS"]
                             == merged[f"AWAY_{period}_PTS"])
        print(f"  {period + ' tied':<26} : {int(tied.sum()):,}  "
              f"({tied.mean():.1%} - HOME_{period}_WIN only)")

    print("\n  resulting NaN per column:")
    for col in QUARTER_HALF_LABELS:
        print(f"    {col:<20}{int(merged[col].isna().sum()):>6,}")

    print(f"\nLabel columns (exclude from training features): {LABEL_COLUMNS}")
    print(merged[LABEL_COLUMNS].describe())

    print("\nPreview:")
    print(merged.head())

if __name__ == "__main__":
    main()
