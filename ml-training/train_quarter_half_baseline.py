"""Baselines for the Q1 and first-half markets."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.preprocessing import StandardScaler

from train_baseline import (
    format_metric,
    load_dataset,
    naive_prediction,
    print_summary,
    section,
    split_by_season,
)

PERIODS = ["Q1", "HALF1"]
ROLLING_METRICS = ["MARGIN", "PTS", "PTS_ALLOWED"]

QUARTER_HALF_FEATURES = [
    f"{side}_{window}_{period}_{metric}"
    for side in ("HOME", "AWAY")
    for window in ("ROLL5", "ROLL10")
    for period in PERIODS
    for metric in ROLLING_METRICS
]

CONTEXT_FEATURES = [
    "HOME_TEAM_ELO",
    "AWAY_TEAM_ELO",
    "HOME_REST_DAYS",
    "AWAY_REST_DAYS",
    "HOME_IS_BACK_TO_BACK",
    "AWAY_IS_BACK_TO_BACK",
]

FEATURE_COLUMNS = QUARTER_HALF_FEATURES + CONTEXT_FEATURES

REGRESSION_TARGETS = [
    ("HOME_Q1_MARGIN", "Q1 spread", "Q1_PTS", "diff"),
    ("TOTAL_Q1_PTS", "Q1 total", "Q1_PTS", "sum"),
    ("HOME_HALF1_MARGIN", "1H spread", "HALF1_PTS", "diff"),
    ("TOTAL_HALF1_PTS", "1H total", "HALF1_PTS", "sum"),
]

CLASSIFICATION_TARGETS = [
    ("HOME_Q1_WIN", "Q1 winner", "Q1"),
    ("HOME_HALF1_WIN", "1H winner", "HALF1"),
]

def drop_incomplete_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter 1. Rows whose trailing Q1/1H form is not yet complete."""
    before = len(df)
    kept = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    dropped = before - len(kept)

    print(f"\nFilter 1 - incomplete rolling window (linear models cannot take NaN)")
    print(f"  dropped {dropped:,} of {before:,} rows ({dropped / before:.1%}), "
          f"leaving {len(kept):,}.")
    print(f"  includes the 3 games with no quarter data and the games that "
          f"follow them within one window.")

    residual = kept[FEATURE_COLUMNS].isna().sum()
    residual = residual[residual > 0]
    if not residual.empty:
        raise ValueError(f"NaN remains in the feature matrix:\n{residual}")

    return kept

def report_ties(df: pd.DataFrame) -> None:
    """Filter 2, quantified before it is applied. Classification only."""
    print(f"\nFilter 2 - tied period (the two winner targets only)")
    for period in PERIODS:
        tied = int((df[f"HOME_{period}_PTS"] == df[f"AWAY_{period}_PTS"]).sum())
        print(f"  {period:<6} {tied:,} of {len(df):,} rows tied ({tied / len(df):.1%}) "
              f"- no honest binary label, dropped from HOME_{period}_WIN only.")
    print("  All four regression targets keep these rows: a 0 margin and a "
          "real total are valid data.")

def scale(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Standardize, scaler fit on train only - never on the full frame."""
    scaler = StandardScaler()
    return (scaler.fit_transform(train[FEATURE_COLUMNS]),
            scaler.transform(test[FEATURE_COLUMNS]))

def evaluate_regression(target, label, stat, combine, train, test) -> list:
    section(f"{label.upper()} (target: {target})")

    train = train.dropna(subset=[target])
    test = test.dropna(subset=[target])
    x_train, x_test = scale(train, test)
    y_train, y_test = train[target], test[target]

    results = []
    naive_name = f"Naive: ROLL10 {stat} {combine}"
    naive_pred = naive_prediction(test, stat, combine)
    naive_mae = mean_absolute_error(y_test, naive_pred)
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
    print(f"{naive_name:<30} MAE {naive_mae:6.2f}  RMSE {naive_rmse:6.2f}")
    results.append((label, naive_name, [("MAE", naive_mae), ("RMSE", naive_rmse)]))

    model = LinearRegression()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"{'LinearRegression':<30} MAE {mae:6.2f}  RMSE {rmse:6.2f}")
    results.append((label, "LinearRegression", [("MAE", mae), ("RMSE", rmse)]))

    print(f"  ({len(train):,} train / {len(test):,} test rows)")
    return results

def evaluate_classification(target, label, period, train, test) -> list:
    section(f"{label.upper()} (target: {target})")

    before_train, before_test = len(train), len(test)
    train = train.dropna(subset=[target])
    test = test.dropna(subset=[target])
    print(f"Dropped {before_train - len(train):,} train and "
          f"{before_test - len(test):,} test rows with no decided winner "
          f"(tied {period}).")

    x_train, x_test = scale(train, test)
    y_train = train[target].astype(int)
    y_test = test[target].astype(int)

    results = []
    naive_acc = accuracy_score(y_test, np.ones(len(y_test), dtype=int))
    print(f"{'Naive: always home':<30} accuracy {naive_acc:.4f}  "
          f"<- the tie-excluded home rate")
    results.append((label, "Naive: always home", [("Accuracy", naive_acc)]))

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    acc = accuracy_score(y_test, model.predict(x_test))
    loss = log_loss(y_test, proba)
    print(f"{'LogisticRegression':<30} accuracy {acc:.4f}  log loss {loss:.4f}")
    results.append((label, "LogisticRegression",
                    [("Accuracy", acc), ("Log loss", loss)]))

    print(f"  ({len(train):,} train / {len(test):,} test rows)")
    return results

def main():
    section("DATA PREP")
    df = load_dataset()
    print(f"Loaded {len(df):,} games from model_dataset.csv")
    print(f"Feature set: {len(FEATURE_COLUMNS)} columns "
          f"({len(QUARTER_HALF_FEATURES)} quarter/half rolling + "
          f"{len(CONTEXT_FEATURES)} reused context)")
    print("  held out of common.FEATURE_COLUMNS - the 7 shipped models are "
          "untouched by this script.")

    df = drop_incomplete_windows(df)
    report_ties(df)

    section("SPLIT")
    train, test, test_seasons = split_by_season(df)

    results = []
    for target, label, stat, combine in REGRESSION_TARGETS:
        results += evaluate_regression(target, label, stat, combine, train, test)
    for target, label, period in CLASSIFICATION_TARGETS:
        results += evaluate_classification(target, label, period, train, test)

    print_summary(results, test_seasons, len(test))

    section("READ THIS BEFORE COMPARING TO THE TEAM-LEVEL TABLE")
    print("These MAEs are NOT comparable to the full-game spread/total numbers.")
    print("A quarter is a smaller quantity with less to be wrong about, so a")
    print("lower MAE here is arithmetic, not skill. The only meaningful")
    print("comparison is naive vs LinearRegression within each row above.")
    print()
    print("The two winner accuracies are conditional on the period being")
    print("decided - tied periods are excluded, which a sportsbook would push.")

if __name__ == "__main__":
    main()
