"""Build a model-ready feature row for a game that hasn't been played yet."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import FEATURE_COLUMNS

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "data-pipeline" / "preprocessing"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

# Constants and formulas are IMPORTED from the pipeline, never restated:
# a second copy is how training and serving drift apart silently.
from build_elo_ratings import (  # noqa: E402
    BASELINE_RATING,
    K_FACTOR,
    SEASON_REGRESSION_FRACTION,
    expected_score,
)
from build_rest_days import REST_DAYS_CAP  # noqa: E402
from build_rolling_features import METRICS, WINDOWS, derive_season  # noqa: E402

def season_of(game_date: pd.Timestamp) -> int:
    """Season label for one date."""
    return int(derive_season(pd.Series([pd.Timestamp(game_date)])).iloc[0])

def team_history(games_final_df: pd.DataFrame, team_id: int, before: pd.Timestamp) -> pd.DataFrame:
    """That team's games strictly before the given date, oldest first."""
    rows = games_final_df[
        (games_final_df["TEAM_ID"] == team_id) & (games_final_df["GAME_DATE"] < before)
    ]
    return rows.sort_values(["GAME_DATE", "GAME_ID"])

def rolling_features(history: pd.DataFrame, season: int) -> dict:
    """Trailing means over the team's most recent games this season."""
    in_season = history[history["SEASON"] == season]

    source = in_season.assign(WIN=(in_season["WL"] == "W").astype(int))

    features = {}
    for window in WINDOWS:
        recent = source.tail(window)
        complete = len(recent) == window
        for source_col, feature_name in METRICS.items():
            features[f"ROLL{window}_{feature_name}"] = (
                recent[source_col].mean() if complete else np.nan
            )
    return features

def rest_features(history: pd.DataFrame, game_date: pd.Timestamp) -> dict:
    """Capped days since the team's last game, plus the back-to-back flag."""
    if history.empty:
        return {"REST_DAYS": np.nan, "IS_BACK_TO_BACK": 0}

    last_date = history["GAME_DATE"].iloc[-1]
    rest_days = min((pd.Timestamp(game_date) - last_date).days, REST_DAYS_CAP)
    return {"REST_DAYS": float(rest_days), "IS_BACK_TO_BACK": int(rest_days == 1)}

def current_elo(history: pd.DataFrame, season: int) -> float:
    """The team's rating going into this game."""
    if history.empty:
        return BASELINE_RATING

    last = history.iloc[-1]
    rating, opponent_rating = last["TEAM_ELO"], last["OPPONENT_ELO"]

    actual = 1.0 if last["WL"] == "W" else 0.0
    rating = rating + K_FACTOR * (actual - expected_score(rating, opponent_rating))

    if last["SEASON"] != season:
        rating = BASELINE_RATING + (rating - BASELINE_RATING) * (1 - SEASON_REGRESSION_FRACTION)

    return rating

def team_features(games_final_df: pd.DataFrame, team_id: int, game_date: pd.Timestamp) -> dict:
    """All 17 pre-game features for one team, keyed by unprefixed name."""
    season = season_of(game_date)
    history = team_history(games_final_df, team_id, game_date)

    features = rolling_features(history, season)
    features.update(rest_features(history, game_date))
    features["TEAM_ELO"] = current_elo(history, season)
    return features

AVAILABILITY_FEATURES = ("ABSENT_COUNT", "WEIGHTED_ABSENT_MIN")

def availability_is_required() -> bool:
    return any(
        f"{prefix}_{name}" in FEATURE_COLUMNS
        for prefix in ("HOME", "AWAY")
        for name in AVAILABILITY_FEATURES
    )

def _availability_for_both_teams(home_team_id: int, away_team_id: int) -> dict:
    """Live ABSENT_COUNT / WEIGHTED_ABSENT_MIN, or NaN if unknowable."""
    blank = {
        f"{prefix}_{name}": float("nan")
        for prefix in ("HOME", "AWAY")
        for name in AVAILABILITY_FEATURES
    }

    try:
        import injury_availability as availability

        reconciled, pending_team_ids = availability.get_live_availability()
    except Exception as error:
        print(f"  availability unknown for this prediction: "
              f"{type(error).__name__}: {error}")
        return blank

    values = {}
    for prefix, team_id in (("HOME", home_team_id), ("AWAY", away_team_id)):
        team = availability.get_team_live_availability(
            team_id, reconciled, pending_team_ids
        )
        for name in AVAILABILITY_FEATURES:
            values[f"{prefix}_{name}"] = team[name]

        if pd.isna(team[AVAILABILITY_FEATURES[0]]):
            print(f"  {prefix} team {team_id}: injury report NOT YET SUBMITTED "
                  f"- availability recorded as unknown, not zero")

    return values

def get_live_features(
    home_team_id: int,
    away_team_id: int,
    game_date,
    games_final_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the feature row for one matchup, in FEATURE_COLUMNS order."""
    game_date = pd.Timestamp(game_date)

    row = {}
    for prefix, team_id in (("HOME", home_team_id), ("AWAY", away_team_id)):
        for name, value in team_features(games_final_df, team_id, game_date).items():
            row[f"{prefix}_{name}"] = value

    if availability_is_required():
        row.update(_availability_for_both_teams(home_team_id, away_team_id))

    missing = set(FEATURE_COLUMNS) - set(row)
    if missing:
        raise ValueError(f"Feature assembly incomplete, missing: {sorted(missing)}")

    return pd.DataFrame([row])[FEATURE_COLUMNS]

def load_games_final() -> pd.DataFrame:
    """Load the history table. A service should call this once at startup."""
    path = (
        Path(__file__).resolve().parents[1]
        / "data-pipeline"
        / "data"
        / "processed"
        / "games_final.csv"
    )
    df = pd.read_csv(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df

def verify_against_training_data(sample_size: int = 200, seed: int = 42) -> bool:
    """Rebuild features for past games and diff against the pipeline's rows."""
    from train_baseline import load_dataset

    replayable_columns = [c for c in FEATURE_COLUMNS
                          if not any(c.endswith(f"_{name}")
                                     for name in AVAILABILITY_FEATURES)]
    skipped = [c for c in FEATURE_COLUMNS if c not in replayable_columns]

    games_final_df = load_games_final()
    dataset = load_dataset()

    if skipped:
        print(f"Replaying {len(replayable_columns)} of {len(FEATURE_COLUMNS)} features.")
        print(f"Not replayable (live-only, no historical equivalent): {skipped}")

    sample = dataset.sample(n=min(sample_size, len(dataset)), random_state=seed)

    mismatches = []
    max_difference = 0.0

    for _, expected_row in sample.iterrows():
        actual = get_live_features(
            expected_row["HOME_TEAM_ID"],
            expected_row["AWAY_TEAM_ID"],
            expected_row["GAME_DATE"],
            games_final_df,
        )

        for column in replayable_columns:
            got, want = actual[column].iloc[0], expected_row[column]

            if pd.isna(got) and pd.isna(want):
                continue
            if pd.isna(got) != pd.isna(want):
                mismatches.append((expected_row["GAME_ID"], column, got, want))
                continue

            difference = abs(float(got) - float(want))
            max_difference = max(max_difference, difference)
            if not np.isclose(got, want, rtol=1e-9, atol=1e-9):
                mismatches.append((expected_row["GAME_ID"], column, got, want))

    print(f"Checked {len(sample)} games x {len(replayable_columns)} features "
          f"= {len(sample) * len(replayable_columns)} values")
    print(f"Largest absolute difference: {max_difference:.2e}")

    if mismatches:
        print(f"\nFAIL - {len(mismatches)} mismatches. First 10:")
        for game_id, column, got, want in mismatches[:10]:
            print(f"  game {game_id} {column}: got {got}, pipeline had {want}")
        return False

    print("\nPASS - every value reproduces the pipeline exactly.")
    return True

def main():
    print("=" * 78)
    print("LIVE FEATURE VERIFICATION")
    print("=" * 78)
    print("Rebuilding features for historical games from team ids + date alone,")
    print("then diffing against the rows the pipeline computed for those games.\n")

    ok = verify_against_training_data()

    print("\n" + "=" * 78)
    print("EXAMPLE OUTPUT")
    print("=" * 78)
    games_final_df = load_games_final()
    latest = pd.Timestamp(games_final_df["GAME_DATE"].max())
    print(f"games_final.csv newest game: {latest.date()} "
          f"- features are only as current as this.\n")

    last_game = games_final_df[games_final_df["GAME_DATE"] == latest]
    home_id = last_game.loc[last_game["IS_HOME"], "TEAM_ID"].iloc[0]
    away_id = last_game.loc[~last_game["IS_HOME"], "TEAM_ID"].iloc[0]
    tip_off = latest + pd.DateOffset(days=1)

    print(f"Hypothetical: team {home_id} hosting team {away_id} on {tip_off.date()}")
    example = get_live_features(home_id, away_id, tip_off, games_final_df)
    print(example.T.to_string(header=["value"]))

    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
