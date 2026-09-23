"""Does shot-location data add what the box score cannot carry?

Study only. Trains nothing that ships, writes no artifacts, mutates no pipeline
file - the variant columns are merged onto an in-memory copy of
model_dataset.csv.

Scoped to five seasons, because the full ingestion is justified by this result
or not at all.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

from common import FEATURE_COLUMNS
from train_baseline import REGRESSION_TARGETS, ROLLING_FEATURE_COLUMNS, load_dataset, section
from train_moneyline_xgb import PARAMS as CLASSIFIER_PARAMS, TARGET as WIN_TARGET
from train_regression_xgb import PARAMS as REGRESSION_PARAMS

PROCESSED = Path(__file__).resolve().parents[1] / "data-pipeline" / "data" / "processed"
SHOT_QUALITY_PATH = PROCESSED / "shot_quality.csv"
GAMES_FINAL_PATH = PROCESSED / "games_final.csv"

# The frozen split's shape, moved onto the subset: validation is the season
# immediately before test, because the frozen architecture early-stops and
# needs a validation window. "Three training seasons" is therefore two for
# fitting plus one for early stopping, exactly as 2015-2022 / 2023 / 2024-25 is.
TRAIN_SEASONS = [2021, 2022]
VALIDATION_SEASON = 2023
TEST_SEASONS = [2024, 2025]

FULL_DATA_SPREAD_MAE = 10.7368

VARIANT_1 = ["ROLL10_SHOOTING_LUCK"]
VARIANT_2 = ["ROLL10_SHOOTING_LUCK", "ROLL10_SHOT_QUALITY"]

ADOPTION_BAR_PCT = 3.0
BOOTSTRAP_ITERATIONS = 2000


def sided(columns) -> list:
    return [f"{side}_{c}" for side in ("HOME", "AWAY") for c in columns]


def load_shot_features() -> pd.DataFrame:
    """Team-game shot quality, reshaped to one row per game with HOME_/AWAY_."""
    if not SHOT_QUALITY_PATH.exists():
        return None

    quality = pd.read_csv(SHOT_QUALITY_PATH)
    sides = pd.read_csv(GAMES_FINAL_PATH, usecols=["GAME_ID", "TEAM_ID", "IS_HOME"])

    merged = quality.merge(sides, on=["GAME_ID", "TEAM_ID"], how="left",
                           validate="one_to_one")
    if merged["IS_HOME"].isna().any():
        raise SystemExit("a shot-quality team-game could not be assigned a side")

    frames = []
    for side, mask in (("HOME", merged["IS_HOME"]), ("AWAY", ~merged["IS_HOME"])):
        part = merged[mask].set_index("GAME_ID")[VARIANT_2]
        part.columns = [f"{side}_{c}" for c in part.columns]
        frames.append(part)

    wide = pd.concat(frames, axis=1).sort_index()
    return wide


def split(dataset: pd.DataFrame) -> tuple:
    train = dataset[dataset["SEASON"].isin(TRAIN_SEASONS)]
    validation = dataset[dataset["SEASON"] == VALIDATION_SEASON]
    test = dataset[dataset["SEASON"].isin(TEST_SEASONS)]
    return train, validation, test


def train_and_score(dataset: pd.DataFrame, features: list, scored_index) -> dict:
    """Every target on one feature set, held out. Never loads a shipped model."""
    train, validation, test = split(dataset)
    comparable = test.loc[scored_index]

    results = {}

    classifier = XGBClassifier(**CLASSIFIER_PARAMS)
    classifier.fit(train[features], train[WIN_TARGET],
                   eval_set=[(validation[features], validation[WIN_TARGET])],
                   verbose=False)
    proba = classifier.predict_proba(comparable[features])[:, 1].astype(np.float64)
    results["Moneyline"] = {
        "accuracy": float(accuracy_score(comparable[WIN_TARGET], (proba > 0.5).astype(int))),
        "log_loss": float(log_loss(comparable[WIN_TARGET], proba)),
        "trees": int(classifier.best_iteration),
    }

    for target, label, _stat, _combine in REGRESSION_TARGETS:
        model = XGBRegressor(**REGRESSION_PARAMS)
        model.fit(train[features], train[target],
                  eval_set=[(validation[features], validation[target])],
                  verbose=False)
        predicted = model.predict(comparable[features]).astype(np.float64)
        results[label] = {
            "mae": float(mean_absolute_error(comparable[target], predicted)),
            "trees": int(model.best_iteration),
        }
        if label == "Spread":
            results[label]["errors"] = np.abs(
                comparable[target].to_numpy() - predicted)

    return results


def significance(base_errors, variant_errors, label: str) -> dict:
    """Paired bootstrap. The last three studies all needed this to tell a small
    gain from nothing, and two would have reported noise as a finding without it."""
    rng = np.random.default_rng(0)
    n = len(base_errors)
    deltas = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        idx = rng.integers(0, n, n)
        deltas.append(variant_errors[idx].mean() - base_errors[idx].mean())
    deltas = np.array(deltas)
    ci = np.percentile(deltas, [2.5, 97.5])
    base_mae = base_errors.mean()
    return {
        "label": label,
        "mean": float(deltas.mean()),
        "ci": ci,
        "pct": float(deltas.mean() / base_mae * 100),
        "pct_ci": (ci[0] / base_mae * 100, ci[1] / base_mae * 100),
        "spans_zero": bool(ci[0] * ci[1] <= 0),
    }


def report_subset_baseline(results: dict, train, validation, test, comparable) -> None:
    section("THE SUBSET BASELINE - COMPUTED BEFORE ANY VARIANT IS SCORED")
    print("A model trained on two seasons is worse than one trained on nine.")
    print("Comparing a variant against the frozen 10.7368 would make shot")
    print("features look catastrophic regardless of merit, so BOTH arms train")
    print("on the identical subset and this number is the only baseline used.\n")

    print(f"  train      seasons {TRAIN_SEASONS}  {len(train):,} games")
    print(f"  validation season  {VALIDATION_SEASON}        {len(validation):,} games")
    print(f"  test       seasons {TEST_SEASONS}  {len(test):,} games "
          f"({len(comparable):,} with complete rolling windows)\n")

    spread = results["Spread"]["mae"]
    cost = (spread - FULL_DATA_SPREAD_MAE) / FULL_DATA_SPREAD_MAE * 100
    print(f"  Subset baseline spread MAE   {spread:.4f}")
    print(f"  Full-data spread MAE         {FULL_DATA_SPREAD_MAE:.4f}")
    print(f"  Cost of the reduced training set   {cost:+.2f}%")
    print("\n  That gap is the price of five seasons instead of eleven. It is")
    print("  context, not a result - and it is why the frozen number cannot be")
    print("  the comparison.")


def print_arms(all_results: dict, labels: list) -> None:
    section("SPREAD - THE HEADLINE")
    base = all_results[labels[0]]["Spread"]["mae"]
    print(f"{'ARM':<34}{'FEATURES':>10}{'MAE':>10}{'CHANGE':>10}{'TREES':>8}")
    print("-" * 74)
    for label, count in labels:
        r = all_results[(label, count)]["Spread"]
        change = (r["mae"] - base) / base * 100
        print(f"{label:<34}{count:>10}{r['mae']:>10.4f}"
              f"{change:>+9.2f}%{r['trees']:>8}")

    section("ALL SEVEN TARGETS")
    targets = ["Moneyline"] + [l for _t, l, _s, _c in REGRESSION_TARGETS]
    header = f"{'TARGET':<12}" + "".join(f"{l.split(' ')[0]:>16}" for l, _c in labels)
    print(header)
    print("-" * len(header))
    for target in targets:
        row = f"{target:<12}"
        for key in labels:
            r = all_results[key][target]
            row += f"{(r['log_loss'] if target == 'Moneyline' else r['mae']):>16.4f}"
        print(row)
    print("(moneyline is log loss, the rest are MAE; lower is better)")

    print(f"\n{'TARGET':<12}" + "".join(f"{l.split(' ')[0]:>16}" for l, _c in labels)
          + "   <- TREE COUNTS")
    for target in targets:
        row = f"{target:<12}"
        for key in labels:
            row += f"{all_results[key][target]['trees']:>16}"
        print(row)
    print("Collapsing counts alongside a better MAE indicates dilution rather")
    print("than signal - how the advanced-stats experiment was diagnosed (S16).")


def main():
    section("DATA")
    dataset = load_dataset()
    subset = dataset[dataset["SEASON"].isin(
        TRAIN_SEASONS + [VALIDATION_SEASON] + TEST_SEASONS)].copy()
    print(f"\nSubset: {len(subset):,} of {len(dataset):,} games, seasons "
          f"{TRAIN_SEASONS + [VALIDATION_SEASON] + TEST_SEASONS}")

    shots = load_shot_features()

    # Every arm scores identical rows, or the comparison is between populations
    # rather than between feature sets.
    _train, _validation, test = split(subset)
    scored = test.dropna(subset=ROLLING_FEATURE_COLUMNS).index

    all_results = {}
    baseline_key = ("Subset baseline (38 features)", 38)
    all_results[baseline_key] = train_and_score(subset, FEATURE_COLUMNS, scored)
    report_subset_baseline(all_results[baseline_key],
                           _train, _validation, test, scored)

    if shots is None:
        section("VARIANTS - NOT RUN")
        print(f"{SHOT_QUALITY_PATH.name} does not exist yet, so only the subset")
        print("baseline above could be computed. Run, in order:")
        print("  data-pipeline/ingestion/fetch_shot_charts.py")
        print("  data-pipeline/preprocessing/build_shot_quality.py")
        print("then rerun this script for the variant comparison.")
        return

    merged = subset.merge(shots, left_on="GAME_ID", right_index=True, how="left")

    # Coverage means "was this game ingested", NOT "does it have a complete
    # rolling window". The first ROLLING_WINDOW games of every team-season are
    # NaN by construction, exactly as every other rolling feature is, so
    # measuring the rolled columns would read the normal warm-up as a partial
    # corpus - which it did, reporting 88.3% on a 99.93%-complete ingestion.
    ingested = shots.index
    coverage = subset["GAME_ID"].isin(ingested).mean()
    print(f"\nShot data ingested for {coverage:.2%} of subset games "
          f"({int(subset['GAME_ID'].isin(ingested).sum()):,} of {len(subset):,}).")
    if coverage < 0.99:
        section("VARIANTS - NOT RUN")
        print(f"Only {coverage:.2%} of subset games were ingested. Scoring a")
        print("variant on a partial corpus would compare feature sets AND")
        print("populations at once. Finish the ingestion first.")
        return

    scored_rows = merged.set_index(subset.index).loc[scored]
    nan_shot = int(scored_rows[sided(VARIANT_1)].isna().any(axis=1).sum())
    print(f"Of the {len(scored):,} scored test games, {nan_shot:,} "
          f"({nan_shot / len(scored):.2%}) have a NaN shot feature on at least\n"
          f"one side - the rolling warm-up plus the reach of the missing games. "
          f"XGBoost\nscores those rows on a learned default split rather than "
          f"dropping them.")

    labels = [baseline_key]
    for name, columns in (("Variant 1: luck only", VARIANT_1),
                          ("Variant 2: luck + quality", VARIANT_2)):
        features = FEATURE_COLUMNS + sided(columns)
        key = (name, len(features))
        all_results[key] = train_and_score(merged, features, scored)
        labels.append(key)

        # Variant 1 first: if it does nothing, variant 2 adding two correlated
        # columns is unlikely to rescue it, and saying so beats running both
        # reflexively.
        if columns is VARIANT_1:
            base_mae = all_results[baseline_key]["Spread"]["mae"]
            change = (all_results[key]["Spread"]["mae"] - base_mae) / base_mae * 100
            print(f"\n  Variant 1 spread change: {change:+.2f}%")

    print_arms(all_results, labels)

    section("IS ANY CHANGE REAL?")
    base_errors = all_results[baseline_key]["Spread"]["errors"]
    stats = []
    for key in labels[1:]:
        result = significance(base_errors, all_results[key]["Spread"]["errors"], key[0])
        stats.append(result)
        print(f"  {result['label']:<30}{result['pct']:>+8.2f}%   "
              f"95% CI [{result['pct_ci'][0]:+.2f}%, {result['pct_ci'][1]:+.2f}%]   "
              f"{'SPANS ZERO' if result['spans_zero'] else 'excludes zero'}")

    section("VERDICT")
    best = min(stats, key=lambda s: s["pct"])
    clears = best["pct"] <= -ADOPTION_BAR_PCT and not best["spans_zero"]
    if clears:
        print(f"{best['label']} clears the bar at {best['pct']:+.2f}% with a CI")
        print("excluding zero. The remaining ingestion is justified.")
    else:
        print(f"NO VARIANT CLEARS THE BAR. Best is {best['label']} at "
              f"{best['pct']:+.2f}%,")
        print(f"against a {ADOPTION_BAR_PCT:.0f}% bar"
              + (", and its interval spans zero." if best["spans_zero"] else "."))
        print("\nA TIE MEANS STOP. If shot quality carries nothing on five")
        print("seasons it will not appear on eleven, so the full pull does not")
        print("happen.")


if __name__ == "__main__":
    main()
