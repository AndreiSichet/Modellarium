"""XGBoost for the five player-prop targets, against the baselines already"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from common import setup_mlflow, split_three_way

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "data-pipeline" / "preprocessing"
sys.path.insert(0, str(PIPELINE_DIR))
from build_player_dataset import (  # noqa: E402
    FEATURE_COLUMNS,
    OUTPUT_PATH as DATASET_PATH,
    PLAYER_FEATURE_COLUMNS,
    TEAM_CONTEXT_COLUMNS,
)

from train_player_baseline import (  # noqa: E402
    TARGET_LABELS,
    TARGETS,
    check_no_leakage,
)

PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "early_stopping_rounds": 50,
    "random_state": 42,
}

IMPROVEMENT_BAR_PCT = 3.0

FLAG_TARGET = "FG3M"

def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")

def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df

def complete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows a linear model could also use - no NaN anywhere in the inputs."""
    return df.dropna(subset=FEATURE_COLUMNS)

def fit_baselines(train: pd.DataFrame, validation: pd.DataFrame,
                  test_comparable: pd.DataFrame, target: str) -> tuple:
    """Naive and LinearRegression on exactly the rows the baseline used."""
    fit_set = complete_rows(pd.concat([train, validation]))

    scaler = StandardScaler()
    fit_x = scaler.fit_transform(fit_set[FEATURE_COLUMNS])
    test_x = scaler.transform(test_comparable[FEATURE_COLUMNS])

    model = LinearRegression()
    model.fit(fit_x, fit_set[target])

    naive = mean_absolute_error(test_comparable[target],
                                test_comparable[f"ROLL10_{target}"])
    linear = mean_absolute_error(test_comparable[target], model.predict(test_x))
    return naive, linear

def train_target(target: str, train: pd.DataFrame, validation: pd.DataFrame,
                 test: pd.DataFrame, test_comparable: pd.DataFrame) -> dict:
    section(f"{TARGET_LABELS[target].upper()} (target: {target})")

    model = XGBRegressor(**PARAMS)
    model.fit(
        train[FEATURE_COLUMNS], train[target],
        eval_set=[(validation[FEATURE_COLUMNS], validation[target])],
        verbose=False,
    )
    trees = int(model.best_iteration)
    print(f"Early stopping at iteration {trees} of {PARAMS['n_estimators']}.")

    xgb_comparable = mean_absolute_error(
        test_comparable[target], model.predict(test_comparable[FEATURE_COLUMNS]))
    xgb_full = mean_absolute_error(test[target], model.predict(test[FEATURE_COLUMNS]))

    naive, linear = fit_baselines(train, validation, test_comparable, target)
    change = (xgb_comparable - linear) / linear * 100
    verdict = ("REAL GAIN" if change <= -IMPROVEMENT_BAR_PCT
               else "no meaningful gain")

    print(f"\nMETHOD                          MAE   vs LINEAR")
    print(f"{'-' * 46}")
    print(f"Naive: ROLL10_{target:<16}{naive:7.3f}")
    print(f"LinearRegression            {linear:7.3f}          -")
    print(f"XGBoost                     {xgb_comparable:7.3f}   {change:+8.1f}%")
    print(f"\nXGBoost on the full test set (incl. incomplete-window rows): "
          f"{xgb_full:.3f}")
    print("  No baseline counterpart - neither naive nor linear can score those rows.")

    importance = pd.Series(
        model.get_booster().get_score(importance_type="gain")
    ).sort_values(ascending=False)
    importance = importance / importance.sum() * 100

    if target == FLAG_TARGET:
        print(f"\n  {FLAG_TARGET} feature importance, in full "
              f"(% of total gain) - see the module docstring:")
        for name, value in importance.items():
            marker = "  <-- team context" if name in TEAM_CONTEXT_COLUMNS else ""
            print(f"    {name:<22}{value:6.2f}%{marker}")
        context_share = importance[
            [c for c in importance.index if c in TEAM_CONTEXT_COLUMNS]
        ].sum()
        print(f"    team-context share of total gain: {context_share:.2f}%")
    else:
        print("\n  top 5 by gain: " + ", ".join(
            f"{n} {v:.1f}%" for n, v in importance.head(5).items()))

    log_run(target, trees, naive, linear, xgb_comparable, xgb_full, change)

    return {"target": target, "naive": naive, "linear": linear,
            "xgb": xgb_comparable, "xgb_full": xgb_full,
            "change": change, "trees": trees, "verdict": verdict,
            "context_share": float(importance[
                [c for c in importance.index if c in TEAM_CONTEXT_COLUMNS]
            ].sum())}

def log_run(target, trees, naive, linear, xgb_comparable, xgb_full, change):
    import mlflow

    experiment = f"player_{target.lower()}"
    setup_mlflow(experiment)
    with mlflow.start_run(run_name=f"xgb-{experiment}"):
        mlflow.log_params(PARAMS)
        mlflow.log_metrics({
            "naive_mae": naive,
            "linear_mae": linear,
            "test_mae": xgb_comparable,
            "test_mae_full": xgb_full,
            "mae_change_pct_vs_linear": change,
            "best_iteration": trees,
        })

def main():
    check_no_leakage()

    section("DATA PREP")
    df = load_dataset()
    print(f"Loaded {len(df):,} player-games - no rows dropped "
          f"(XGBoost handles NaN natively).")

    usable_by_baseline = len(complete_rows(df))
    print(f"  the baselines could only use {usable_by_baseline:,} of these; "
          f"XGBoost trains on {len(df) - usable_by_baseline:,} more rows.")

    train, validation, test = split_three_way(df)
    test_comparable = complete_rows(test)
    print(f"\nThree-way table scored on {len(test_comparable):,} complete-feature "
          f"test rows (what the baselines could score); "
          f"{len(test) - len(test_comparable):,} held aside for the secondary number.")

    results = [train_target(t, train, validation, test, test_comparable)
               for t in TARGETS]

    section(f"VERDICTS (bar: MAE improvement over LinearRegression > "
            f"{IMPROVEMENT_BAR_PCT:.1f}%)")
    print(f"{'TARGET':<16}{'NAIVE':>9}{'LINEAR':>9}{'XGBOOST':>9}"
          f"{'CHANGE':>9}{'TREES':>7}   VERDICT")
    print("-" * 76)
    for row in results:
        print(f"{TARGET_LABELS[row['target']]:<16}{row['naive']:>9.3f}"
              f"{row['linear']:>9.3f}{row['xgb']:>9.3f}{row['change']:>8.1f}%"
              f"{row['trees']:>7}   {row['verdict']}")

    cleared = sum(1 for r in results if r["verdict"] == "REAL GAIN")
    print(f"\n{cleared} of {len(results)} targets clear the bar.")

    section("TREE COUNTS - is FG3M converging early?")
    print("A much lower count than its peers is the same signature the rejected")
    print("advanced-stats experiment produced: added variance, not added signal.\n")
    for row in results:
        flag = "  <-- the flagged target" if row["target"] == FLAG_TARGET else ""
        print(f"  {TARGET_LABELS[row['target']]:<16}{row['trees']:>5} trees   "
              f"team-context gain {row['context_share']:5.1f}%{flag}")

if __name__ == "__main__":
    main()
