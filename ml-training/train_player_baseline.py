"""Baselines for the five player-prop targets, before any XGBoost is touched."""

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "data-pipeline" / "preprocessing"
sys.path.insert(0, str(PIPELINE_DIR))
from build_player_dataset import (  # noqa: E402
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    OUTPUT_PATH as DATASET_PATH,
    PLAYER_FEATURE_COLUMNS,
    TEAM_CONTEXT_COLUMNS,
)

from train_baseline import TEST_SEASON_COUNT  # noqa: E402

TARGETS = ["PTS", "REB", "AST", "FG3M", "PRA"]

NOT_A_TARGET = "MIN_NUMERIC"

TARGET_LABELS = {
    "PTS": "Points",
    "REB": "Rebounds",
    "AST": "Assists",
    "FG3M": "Threes made",
    "PRA": "Points+Reb+Ast",
}

def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")

def check_no_leakage() -> None:
    """No post-game outcome may appear among the inputs."""
    leaked = [c for c in FEATURE_COLUMNS if c in LABEL_COLUMNS]
    if leaked:
        raise ValueError(f"post-game columns present in FEATURE_COLUMNS: {leaked}")

    if NOT_A_TARGET in FEATURE_COLUMNS:
        raise ValueError(
            f"{NOT_A_TARGET} is a post-game outcome and must never be a feature."
        )

    for target in TARGETS:
        if target in FEATURE_COLUMNS:
            raise ValueError(f"target {target} is also listed as a feature.")

def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    print(f"Loaded {len(df):,} player-games from {DATASET_PATH.name}")
    return df

def drop_incomplete(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows LinearRegression cannot consume, and say why separately."""
    before = len(df)

    warmup = df[PLAYER_FEATURE_COLUMNS].isna().any(axis=1)
    context = df[TEAM_CONTEXT_COLUMNS].isna().any(axis=1)

    print(f"\n  incomplete player rolling windows : {int(warmup.sum()):>7,}")
    print(f"  missing team context (REST_DAYS)  : {int(context.sum()):>7,}")
    print(f"  overlap                           : {int((warmup & context).sum()):>7,}")

    kept = df[~(warmup | context)].reset_index(drop=True)
    dropped = before - len(kept)
    print(f"\nDropped {dropped:,} of {before:,} rows ({dropped / before:.1%}), "
          f"leaving {len(kept):,}.")

    residual = kept[FEATURE_COLUMNS].isna().sum()
    residual = residual[residual > 0]
    if not residual.empty:
        raise ValueError(f"NaN remains in the feature matrix:\n{residual}")

    return kept

def split_by_season(df: pd.DataFrame) -> tuple:
    """Chronological hold-out, identical boundary to the team models."""
    seasons = sorted(df["SEASON"].unique())
    test_seasons = seasons[-TEST_SEASON_COUNT:]

    train = df[~df["SEASON"].isin(test_seasons)].reset_index(drop=True)
    test = df[df["SEASON"].isin(test_seasons)].reset_index(drop=True)

    train_seasons = seasons[:-TEST_SEASON_COUNT]
    print(f"\nTrain: seasons {train_seasons[0]}-{train_seasons[-1]} "
          f"({len(train_seasons)} seasons, {len(train):,} player-games)")
    print(f"Test:  seasons {test_seasons[0]}-{test_seasons[-1]} "
          f"({len(test_seasons)} seasons, {len(test):,} player-games)")

    return train, test, test_seasons

def scale_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Standardise, fitting on train only."""
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train[FEATURE_COLUMNS])
    test_x = scaler.transform(test[FEATURE_COLUMNS])
    print(f"Standardized {len(FEATURE_COLUMNS)} features "
          f"(scaler fit on train only).")
    return train_x, test_x

def evaluate(target: str, train: pd.DataFrame, test: pd.DataFrame,
             train_x, test_x) -> dict:
    section(f"{TARGET_LABELS[target].upper()} (target: {target})")

    naive_column = f"ROLL10_{target}"
    naive_mae = mean_absolute_error(test[target], test[naive_column])

    model = LinearRegression()
    model.fit(train_x, train[target])
    linear_mae = mean_absolute_error(test[target], model.predict(test_x))

    change = (linear_mae - naive_mae) / naive_mae * 100

    print(f"Naive: {naive_column:<14} MAE {naive_mae:7.3f}")
    print(f"LinearRegression              MAE {linear_mae:7.3f}   "
          f"{change:+.1f}% vs naive")
    print(f"Test-set mean {target}: {test[target].mean():.2f}  "
          f"(MAE in context)")

    return {"target": target, "naive": naive_mae,
            "linear": linear_mae, "change": change}

def main():
    check_no_leakage()

    section("DATA PREP")
    df = load_dataset()
    df = drop_incomplete(df)
    train, test, test_seasons = split_by_season(df)
    train_x, test_x = scale_features(train, test)

    results = [evaluate(t, train, test, train_x, test_x) for t in TARGETS]

    section(f"SUMMARY (test: seasons {test_seasons[0]}-{test_seasons[-1]}, "
            f"{len(test):,} player-games)")
    print(f"{'TARGET':<16}{'NAIVE (ROLL10)':>16}{'LINEAR':>10}{'CHANGE':>10}")
    print("-" * 52)
    for row in results:
        print(f"{TARGET_LABELS[row['target']]:<16}{row['naive']:>16.3f}"
              f"{row['linear']:>10.3f}{row['change']:>9.1f}%")

    print(f"\n{NOT_A_TARGET} deliberately not modelled here - predicting playing "
          f"time is a\ndifferent problem and deserves its own pass.")

if __name__ == "__main__":
    main()
