"""Do travel, time-zone, road-trip and density features beat REST_DAYS alone?

Study only. Trains nothing that ships, writes no artifacts, mutates no pipeline
file - the variant columns are merged onto an in-memory copy of
model_dataset.csv.

Full data, frozen split, frozen architecture. There is no ingestion cost here,
so there is no reason to work on a subset.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

from common import FEATURE_COLUMNS, split_three_way
from train_baseline import REGRESSION_TARGETS, ROLLING_FEATURE_COLUMNS, load_dataset, section
from train_moneyline_xgb import PARAMS as CLASSIFIER_PARAMS, TARGET as WIN_TARGET
from train_regression_xgb import PARAMS as REGRESSION_PARAMS

PROCESSED = Path(__file__).resolve().parents[1] / "data-pipeline" / "data" / "processed"
TRAVEL_PATH = PROCESSED / "travel_features.csv"
GAMES_FINAL_PATH = PROCESSED / "games_final.csv"

RECORDED = {"Spread": 10.7368, "Totals": 15.2322, "REB margin": 7.5095,
            "REB total": 7.3382, "AST margin": 5.3928, "AST total": 5.8356}
RECORDED_ACCURACY = 0.6698
RECORDED_LOG_LOSS = 0.5979
TOLERANCE = 5e-4

DENSITY = ["GAMES_LAST_7"]
TRAVEL = ["TRAVEL_KM", "TZ_SHIFT"]
ALL_FOUR = ["GAMES_LAST_7", "TRAVEL_KM", "TZ_SHIFT", "ROAD_TRIP_GAME"]

ADOPTION_BAR_PCT = 3.0
BOOTSTRAP_ITERATIONS = 2000


def sided(columns) -> list:
    return [f"{side}_{c}" for side in ("HOME", "AWAY") for c in columns]


def load_travel() -> pd.DataFrame:
    """Team-game travel features, reshaped to one row per game."""
    if not TRAVEL_PATH.exists():
        raise SystemExit(f"{TRAVEL_PATH.name} missing. Run "
                         "data-pipeline/preprocessing/build_travel_features.py")

    travel = pd.read_csv(TRAVEL_PATH)
    sides = pd.read_csv(GAMES_FINAL_PATH, usecols=["GAME_ID", "TEAM_ID", "IS_HOME"])
    merged = travel.merge(sides, on=["GAME_ID", "TEAM_ID"], how="left",
                          validate="one_to_one")
    if merged["IS_HOME"].isna().any():
        raise SystemExit("a travel team-game could not be assigned a side")

    frames = []
    for side, mask in (("HOME", merged["IS_HOME"]), ("AWAY", ~merged["IS_HOME"])):
        part = merged[mask].set_index("GAME_ID")[ALL_FOUR]
        part.columns = [f"{side}_{c}" for c in part.columns]
        frames.append(part)
    return pd.concat(frames, axis=1).sort_index()


def train_and_score(dataset: pd.DataFrame, features: list, scored_index) -> dict:
    """Every target on one feature set, HELD OUT on the frozen split.

    Never loads ml-training/models/: those train with no holdout, so the test
    window sits inside their training set and every metric is flattered.
    """
    train, validation, test = split_three_way(dataset)
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


def verify_baseline(results: dict) -> None:
    section("BASELINE GATE - THE FROZEN 38-FEATURE NUMBERS MUST REPRODUCE")
    failures = []
    for name, got, want in [
        ("Moneyline accuracy", results["Moneyline"]["accuracy"], RECORDED_ACCURACY),
        ("Moneyline log loss", results["Moneyline"]["log_loss"], RECORDED_LOG_LOSS),
    ]:
        ok = abs(got - want) < TOLERANCE
        failures += [] if ok else [name]
        print(f"  {name:<22} got {got:.4f}   recorded {want:.4f}   "
              f"{'MATCH' if ok else 'NO MATCH'}")
    for label, want in RECORDED.items():
        got = results[label]["mae"]
        ok = abs(got - want) < TOLERANCE
        failures += [] if ok else [label]
        print(f"  {label + ' MAE':<22} got {got:.4f}   recorded {want:.4f}   "
              f"{'MATCH' if ok else 'NO MATCH'}")
    if failures:
        raise SystemExit(f"{failures} did not reproduce; every comparison below "
                         "would be against a different baseline.")
    print("\n  PASS - all seven reproduce.")


def significance(base_errors, variant_errors) -> dict:
    rng = np.random.default_rng(0)
    n = len(base_errors)
    deltas = np.array([
        variant_errors[i].mean() - base_errors[i].mean()
        for i in (rng.integers(0, n, n) for _ in range(BOOTSTRAP_ITERATIONS))
    ])
    ci = np.percentile(deltas, [2.5, 97.5])
    base = base_errors.mean()
    return {"pct": float(deltas.mean() / base * 100),
            "pct_ci": (ci[0] / base * 100, ci[1] / base * 100),
            "spans_zero": bool(ci[0] * ci[1] <= 0)}


def main():
    section("DATA")
    dataset = load_dataset()
    travel = load_travel()
    merged = dataset.merge(travel, left_on="GAME_ID", right_index=True, how="left")
    if len(merged) != len(dataset):
        raise SystemExit("the travel merge changed the row count")

    coverage = merged["GAME_ID"].isin(travel.index).mean()
    print(f"\nTravel features present for {coverage:.2%} of all "
          f"{len(dataset):,} games.")

    _train, _validation, test = split_three_way(merged)
    scored = test.dropna(subset=ROLLING_FEATURE_COLUMNS).index
    print(f"Scoring on {len(scored):,} complete-window test games - every arm "
          "scores identical rows.")

    arms = [("Baseline (38 features)", FEATURE_COLUMNS)]
    all_results = {arms[0][0]: train_and_score(merged, FEATURE_COLUMNS, scored)}
    verify_baseline(all_results[arms[0][0]])

    for name, columns in (("Variant 1: density only", DENSITY),
                          ("Variant 2: travel only", TRAVEL),
                          ("Variant 3: all eight", ALL_FOUR)):
        features = FEATURE_COLUMNS + sided(columns)
        label = f"{name} ({len(features)})"
        arms.append((label, features))
        all_results[label] = train_and_score(merged, features, scored)

    section("SPREAD - THE HEADLINE")
    base = all_results[arms[0][0]]["Spread"]["mae"]
    print(f"{'ARM':<34}{'FEATURES':>10}{'MAE':>10}{'CHANGE':>10}{'TREES':>8}")
    print("-" * 74)
    for label, features in arms:
        r = all_results[label]["Spread"]
        print(f"{label:<34}{len(features):>10}{r['mae']:>10.4f}"
              f"{(r['mae'] - base) / base * 100:>+9.2f}%{r['trees']:>8}")

    section("ALL SEVEN TARGETS")
    targets = ["Moneyline"] + [l for _t, l, _s, _c in REGRESSION_TARGETS]
    short = [l.split(":")[0].split(" (")[0] for l, _f in arms]
    print(f"{'TARGET':<12}" + "".join(f"{s:>13}" for s in short))
    print("-" * (12 + 13 * len(arms)))
    for target in targets:
        row = f"{target:<12}"
        for label, _f in arms:
            r = all_results[label][target]
            row += f"{(r['log_loss'] if target == 'Moneyline' else r['mae']):>13.4f}"
        print(row)
    print("(moneyline is log loss, the rest are MAE; lower is better)")

    print(f"\n{'TARGET':<12}" + "".join(f"{s:>13}" for s in short) + "   <- TREES")
    for target in targets:
        row = f"{target:<12}"
        for label, _f in arms:
            row += f"{all_results[label][target]['trees']:>13}"
        print(row)

    section("IS ANY CHANGE REAL?")
    base_errors = all_results[arms[0][0]]["Spread"]["errors"]
    stats = {}
    for label, _f in arms[1:]:
        result = significance(base_errors, all_results[label]["Spread"]["errors"])
        stats[label] = result
        print(f"  {label:<34}{result['pct']:>+8.2f}%   "
              f"95% CI [{result['pct_ci'][0]:+.2f}%, {result['pct_ci'][1]:+.2f}%]"
              f"   {'SPANS ZERO' if result['spans_zero'] else 'excludes zero'}")

    section("VERDICT")
    best = min(stats, key=lambda k: stats[k]["pct"])
    result = stats[best]
    if result["pct"] <= -ADOPTION_BAR_PCT and not result["spans_zero"]:
        print(f"{best} clears the bar at {result['pct']:+.2f}%.")
        print("NOT SHIPPED IN THIS PHASE - adoption is a separate change.")
        return

    print(f"NO VARIANT CLEARS THE BAR. Best is {best} at {result['pct']:+.2f}%,")
    print(f"against a {ADOPTION_BAR_PCT:.0f}% bar"
          + (", and its interval spans zero." if result["spans_zero"] else "."))
    print("\nHOW TO READ THIS NULL - the point of the correlation diagnostic:")
    print("  TRAVEL_KM, TZ_SHIFT and ROAD_TRIP_GAME correlate with REST_DAYS at")
    print("  |r| < 0.12, so they are genuinely new quantities. Their null is")
    print("  INFORMATIVE: the model was handed information it did not have and")
    print("  could not use it.")
    print("  GAMES_LAST_7 correlates at -0.65 - substantially overlapping. Its")
    print("  null is WEAKER evidence, because the density term is not as")
    print("  independent of recency as the hypothesis assumed.")


if __name__ == "__main__":
    main()
