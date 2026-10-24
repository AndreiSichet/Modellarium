"""Baseline models for moneyline, spread, totals, rebounds and assists."""

import sys
from pathlib import Path

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

from common import FEATURE_COLUMNS, ROLLING_FEATURE_COLUMNS, TEST_SEASONS

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "data-pipeline" / "preprocessing"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from build_rolling_features import (  # noqa: E402
    derive_season, derive_season_for_frame,
)

DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "data-pipeline"
    / "data"
    / "processed"
    / "model_dataset.csv"
)

# TEST_SEASON_COUNT is gone: the test seasons come from common.TEST_SEASONS,
# which is what production's split reads. See split_by_season.

BOOLEAN_FEATURE_COLUMNS = ["HOME_IS_BACK_TO_BACK", "AWAY_IS_BACK_TO_BACK"]

UNUSED_FEATURE_COLUMNS = [
    f"{side}_{window}_{metric}"
    for side in ("HOME", "AWAY")
    for window in ("ROLL5", "ROLL10")
    for metric in ("OFF_RATING", "DEF_RATING", "NET_RATING", "PACE", "TS_PCT")
] + [
    # THE FOUR AVAILABILITY COLUMNS, DROPPED FROM THE FEATURE SET BY A3 AND
    # STILL PRESENT IN THE TABLE. The pipeline keeps building them because
    # they are a TRUE record of who did not play, which is a fact worth
    # keeping; what they are not is a feature, because that fact is only
    # knowable after tip-off.
    #
    # LISTED HERE RATHER THAN DELETED FROM THE TABLE SO THE GUARD STAYS. The
    # `unaccounted` check below is what caught the drop being incomplete -
    # removing the columns from FEATURE_COLUMNS made them unclassified and
    # load_dataset refused the table by name. Anyone who later wants to use
    # them has to move them out of this list deliberately, in a diff, with
    # this comment in front of them.
    f"{side}_{name}"
    for side in ("HOME", "AWAY")
    for name in ("ABSENT_COUNT", "WEIGHTED_ABSENT_MIN")
]

UNUSED_FEATURE_COLUMNS += [
    f"{side}_{window}_{metric}"
    for side in ("HOME", "AWAY")
    for window in ("ROLL5", "ROLL10")
    for metric in ("Q1_MARGIN", "Q1_PTS", "Q1_PTS_ALLOWED",
                   "HALF1_MARGIN", "HALF1_PTS", "HALF1_PTS_ALLOWED")
]

ID_COLUMNS = [
    "GAME_ID",
    "GAME_DATE",
    "HOME_TEAM_ID",
    "HOME_TEAM_NAME",
    "AWAY_TEAM_ID",
    "AWAY_TEAM_NAME",
]
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

METRIC_PRECISION = {"Accuracy": 4, "Log loss": 4, "MAE": 2, "RMSE": 2}

REGRESSION_TARGETS = [
    ("HOME_MARGIN", "Spread", "PTS", "diff"),
    ("TOTAL_PTS", "Totals", "PTS", "sum"),
    ("REB_MARGIN", "REB margin", "REB", "diff"),
    ("TOTAL_REB", "REB total", "REB", "sum"),
    ("AST_MARGIN", "AST margin", "AST", "diff"),
    ("TOTAL_AST", "AST total", "AST", "sum"),
]

# derive_season WAS DEFINED HERE, a second copy of the pipeline's. Both are
# now imported from build_rolling_features, which is the one definition - a
# duplicated season rule is how the training and pipeline sides would drift
# apart on exactly the change this spec makes. Re-exported because other
# modules import `derive_season` from here.

def elo_expected_score(rating: pd.Series, opponent_rating: pd.Series) -> pd.Series:
    """Elo win probability. Same formula as build_elo_ratings.py, no training."""
    return 1 / (1 + 10 ** ((opponent_rating - rating) / 400))

def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")

def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    # model_dataset.csv carries GAME_ID, so the season comes from the id.
    df["SEASON"] = derive_season_for_frame(df)

    for col in BOOLEAN_FEATURE_COLUMNS:
        df[col] = df[col].astype(int)

    unaccounted = (
        set(df.columns)
        - set(FEATURE_COLUMNS)
        - set(ID_COLUMNS)
        - set(LABEL_COLUMNS)
        - set(UNUSED_FEATURE_COLUMNS)
        - {"SEASON"}
    )
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if unaccounted or missing:
        raise ValueError(
            "FEATURE_COLUMNS is out of sync with the dataset.\n"
            f"  Columns in the file but unclassified: {sorted(unaccounted)}\n"
            f"  Features expected but absent: {sorted(missing)}"
        )

    return df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

def drop_incomplete_windows(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=ROLLING_FEATURE_COLUMNS).reset_index(drop=True)
    dropped = before - len(df)

    print(
        f"Dropped {dropped} of {before} rows ({dropped / before:.1%}) with incomplete "
        f"rolling windows, leaving {len(df)}."
    )

    residual = df[FEATURE_COLUMNS].isna().sum()
    residual = residual[residual > 0]
    if not residual.empty:
        raise ValueError(f"NaN remains in the feature matrix after the drop:\n{residual}")

    return df

def split_by_season(df: pd.DataFrame) -> tuple:
    """Train / test by season, from common.TEST_SEASONS. ONE definition.

    THIS USED TO DERIVE ITS OWN TEST SET as "the last two seasons present",
    while production's `common.split_three_way` reads the literal
    `common.TEST_SEASONS`. The two agreed only for as long as the newest
    season in the table happened to be the newest test season - so the first
    rebuild of model_dataset.csv carrying the 2026 season would have moved
    THIS script's test set to (2025, 2026) and left production's at
    (2024, 2025). Every frozen baseline figure would have changed, and the
    cause would have looked like the data rather than the split.
    That is not hypothetical: the NBA season opened on 2026-10-20.

    TRAIN IS EVERYTHING STRICTLY BEFORE THE TEST SEASONS, not "everything
    that is not a test season". The second form would pull a season LATER
    than the test set into training the moment one exists - training on
    2026 to predict 2024 - which is leakage of the plainest kind. The
    comparison with production stays honest because production's own split
    is chronological too: train < validation < test.

    `VALIDATION_SEASON` is deliberately inside train here. These are baselines
    with nothing to early-stop, so there is no held-out set to reserve, and
    folding it in is what makes the baseline and production numbers comparable
    on the same test games.
    """
    seasons = sorted(df["SEASON"].unique())
    test_seasons = [s for s in seasons if s in TEST_SEASONS]
    train_seasons = [s for s in seasons if s < min(TEST_SEASONS)]

    if not test_seasons or not train_seasons:
        raise ValueError(
            f"Empty split - seasons present: {seasons}, test seasons "
            f"{list(TEST_SEASONS)}. Expected data in the test seasons and "
            f"at least one season before them."
        )

    train = df[df["SEASON"] < min(TEST_SEASONS)].reset_index(drop=True)
    test = df[df["SEASON"].isin(TEST_SEASONS)].reset_index(drop=True)

    ignored = [s for s in seasons if s not in test_seasons
               and s not in train_seasons]
    print(
        f"Train: seasons {train_seasons[0]}-{train_seasons[-1]} "
        f"({len(train_seasons)} seasons, {len(train)} games)"
    )
    print(
        f"Test:  seasons {test_seasons[0]}-{test_seasons[-1]} "
        f"({len(test_seasons)} seasons, {len(test)} games)"
    )
    if ignored:
        # NAMED, not silently dropped. A season after the test set belongs to
        # neither side, and saying so is what stops someone reading an
        # unchanged figure as "the new season had no effect on the model".
        print(
            f"Ignored: seasons {ignored} - later than the test set, so in "
            f"neither train nor test. These baselines are frozen against "
            f"seasons {TEST_SEASONS[0]}-{TEST_SEASONS[-1]}."
        )

    return train, test, test_seasons

def scale_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[FEATURE_COLUMNS])
    x_test = scaler.transform(test[FEATURE_COLUMNS])
    print(f"Standardized {len(FEATURE_COLUMNS)} features (scaler fit on train only).")
    return x_train, x_test

def evaluate_moneyline(train, test, x_train, x_test) -> list:
    section("MONEYLINE (target: HOME_WIN)")
    y_train, y_test = train["HOME_WIN"], test["HOME_WIN"]
    results = []

    naive_pred = np.ones(len(y_test), dtype=int)
    naive_acc = accuracy_score(y_test, naive_pred)
    print(f"Naive (always home)  accuracy {naive_acc:.4f}  <- the test set's home-win rate")
    results.append(("Moneyline", "Naive: always home", [("Accuracy", naive_acc)]))

    elo_prob = elo_expected_score(test["HOME_TEAM_ELO"], test["AWAY_TEAM_ELO"])
    elo_acc = accuracy_score(y_test, (elo_prob > 0.5).astype(int))
    elo_loss = log_loss(y_test, elo_prob)
    print(f"Elo win probability  accuracy {elo_acc:.4f}  log loss {elo_loss:.4f}")
    results.append(("Moneyline", "Elo win probability", [("Accuracy", elo_acc), ("Log loss", elo_loss)]))

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    lr_acc = accuracy_score(y_test, model.predict(x_test))
    lr_loss = log_loss(y_test, proba)
    print(f"LogisticRegression   accuracy {lr_acc:.4f}  log loss {lr_loss:.4f}")
    results.append(("Moneyline", "LogisticRegression", [("Accuracy", lr_acc), ("Log loss", lr_loss)]))

    return results

def naive_prediction(test: pd.DataFrame, stat: str, combine: str) -> pd.Series:
    """Each team's own ROLL10 average, differenced or summed."""
    home, away = test[f"HOME_ROLL10_{stat}"], test[f"AWAY_ROLL10_{stat}"]
    return home - away if combine == "diff" else home + away

def evaluate_regression_target(
    target: str, label: str, stat: str, combine: str, train, test, x_train, x_test
) -> list:
    section(f"{label.upper()} (target: {target})")
    y_train, y_test = train[target], test[target]
    results = []

    naive_name = f"Naive: ROLL10 {stat} {combine}"
    naive_pred = naive_prediction(test, stat, combine)
    naive_mae = mean_absolute_error(y_test, naive_pred)
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
    print(f"{naive_name:<28} MAE {naive_mae:6.2f}  RMSE {naive_rmse:6.2f}")
    results.append((label, naive_name, [("MAE", naive_mae), ("RMSE", naive_rmse)]))

    model = LinearRegression()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"{'LinearRegression':<28} MAE {mae:6.2f}  RMSE {rmse:6.2f}")
    results.append((label, "LinearRegression", [("MAE", mae), ("RMSE", rmse)]))

    return results

def format_metric(name: str, value: float) -> str:
    return f"{name} {value:.{METRIC_PRECISION[name]}f}"

def print_summary(results: list, test_seasons: list, test_rows: int) -> None:
    section(f"SUMMARY (test set: seasons {test_seasons[0]}-{test_seasons[-1]}, {test_rows} games)")

    rows = []
    for target, method, metrics in results:
        cells = [format_metric(name, value) for name, value in metrics]
        cells += ["-"] * (2 - len(cells))
        rows.append((target, method, cells[0], cells[1]))

    headers = ("TARGET", "METHOD", "METRIC", "")
    widths = [max(len(str(r[i])) for r in (*rows, headers)) for i in range(4)]

    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line.rstrip())
    print("-" * len(line))

    previous_target = None
    for row in rows:
        if previous_target is not None and row[0] != previous_target:
            print()
        previous_target = row[0]
        print("  ".join(str(cell).ljust(w) for cell, w in zip(row, widths)).rstrip())

def main():
    section("DATA PREP")
    df = load_dataset()
    print(f"Loaded {len(df)} games from {DATASET_PATH.name}")

    df = drop_incomplete_windows(df)
    train, test, test_seasons = split_by_season(df)
    x_train, x_test = scale_features(train, test)

    results = evaluate_moneyline(train, test, x_train, x_test)

    for target, label, stat, combine in REGRESSION_TARGETS:
        results += evaluate_regression_target(
            target, label, stat, combine, train, test, x_train, x_test
        )

    print_summary(results, test_seasons, len(test))

if __name__ == "__main__":
    main()
