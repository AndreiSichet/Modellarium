"""Retrain the seven team-level models on current data, and refuse to promote"""

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from common import FEATURE_COLUMNS
from finalize_models import final_params
from model_evaluation import TASKS, evaluate, load_model, primary
from train_baseline import load_dataset, section
from train_moneyline_xgb import PARAMS as MONEYLINE_PARAMS
from train_regression_xgb import PARAMS as REGRESSION_PARAMS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = PROJECT_ROOT / "data-pipeline"
GAMES_FINAL = PIPELINE / "data" / "processed" / "games_final.csv"

PRODUCTION_DIR = Path(__file__).resolve().parent / "models"
CANDIDATE_DIR = Path(__file__).resolve().parent / "models_candidate"

TRAINED_THROUGH = "trained_through.json"

REFRESH_STEPS = [
    PIPELINE / "ingestion" / "fetch_games.py",
    PIPELINE / "ingestion" / "fetch_player_boxscores.py",
    PIPELINE / "preprocessing" / "validate_games.py",
    PIPELINE / "preprocessing" / "build_games_table.py",
    PIPELINE / "preprocessing" / "build_rolling_features.py",
    PIPELINE / "preprocessing" / "build_rest_days.py",
    PIPELINE / "preprocessing" / "build_elo_ratings.py",
    PIPELINE / "preprocessing" / "build_player_rolling_minutes.py",
    PIPELINE / "preprocessing" / "build_team_availability.py",
    PIPELINE / "ingestion" / "fetch_quarter_scores.py",
    PIPELINE / "preprocessing" / "build_quarter_half_raw.py",
    PIPELINE / "preprocessing" / "build_quarter_half_rolling.py",
    PIPELINE / "preprocessing" / "build_final_dataset.py",
]

DEFAULT_TEST_GAMES = 500

DEFAULT_TOLERANCE_PCT = 2.0

DEFAULT_MIN_NEW_GAMES = 100

# A crash and a refusal must not share a code: one is the absence of a
# measurement, the other is a measurement.
EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_ERROR = 2

def snapshot(path: Path) -> dict:
    """Enough of games_final.csv to tell whether anything new arrived."""
    if not path.exists():
        return {"rows": 0, "latest": None}
    frame = pd.read_csv(path, usecols=["GAME_ID", "GAME_DATE"])
    return {
        "rows": len(frame),
        "latest": str(pd.to_datetime(frame["GAME_DATE"]).max().date()),
    }

def read_trained_through(directory: Path):
    """What data the models in this directory were trained on, or None."""
    path = directory / TRAINED_THROUGH
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def write_trained_through(directory: Path, dataset: pd.DataFrame) -> None:
    (directory / TRAINED_THROUGH).write_text(
        json.dumps({
            "trained_through": str(dataset["GAME_DATE"].max().date()),
            "games": int(len(dataset)),
        }, indent=2),
        encoding="utf-8",
    )

def refresh_data() -> None:
    """Rerun ingestion and preprocessing, in order, stopping on failure."""
    section("REFRESHING DATA")
    for step in REFRESH_STEPS:
        print(f"\n--- {step.name}")
        result = subprocess.run(
            [sys.executable, str(step)], cwd=str(PROJECT_ROOT), check=False
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{step.name} failed with exit code {result.returncode}. "
                f"Stopping: every later step reads what this one writes, so "
                f"continuing would train on a half-built dataset."
            )

def rolling_split(dataset: pd.DataFrame, test_games: int) -> tuple:
    """Most recent games as test, the ones before as validation, rest train."""
    ordered = dataset.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    if len(ordered) < test_games * 3:
        raise RuntimeError(
            f"{len(ordered):,} games is too few to carve a {test_games}-game "
            f"test and validation window out of and still leave a training set."
        )

    test = ordered.iloc[-test_games:]
    validation = ordered.iloc[-2 * test_games:-test_games]
    train = ordered.iloc[: -2 * test_games]

    def span(frame):
        return f"{frame['GAME_DATE'].min().date()} to {frame['GAME_DATE'].max().date()}"

    print(f"  train      {len(train):>6,} games   {span(train)}")
    print(f"  validation {len(validation):>6,} games   {span(validation)}")
    print(f"  test       {len(test):>6,} games   {span(test)}   <- the gate's window")
    return train, validation, test

def train_candidate(target: str, classification: bool, train, validation):
    """One candidate model, on the frozen architecture."""
    params = MONEYLINE_PARAMS if classification else REGRESSION_PARAMS
    model = (XGBClassifier if classification else XGBRegressor)(**params)
    model.fit(
        train[FEATURE_COLUMNS], train[target],
        eval_set=[(validation[FEATURE_COLUMNS], validation[target])],
        verbose=False,
    )
    return model

def refit_production(shipped, target: str, classification: bool, train):
    """Production's architecture, re-fit on the candidate's training rows."""
    trees = shipped.get_booster().num_boosted_rounds()
    params = final_params(classification, trees)
    model = (XGBClassifier if classification else XGBRegressor)(**params)
    model.fit(train[FEATURE_COLUMNS], train[target], verbose=False)
    return model, trees

def run_gate(train, validation, test, production_dir: Path,
             tolerance_pct: float) -> list:
    """Train each candidate, re-fit production on the same rows, compare."""
    results = []

    for key, label, target, classification in TASKS:
        candidate = train_candidate(target, classification, train, validation)
        candidate_metrics = evaluate(candidate, test, target, classification)

        shipped = load_model(key, classification, production_dir)
        incumbent, trees = refit_production(shipped, target, classification, train)
        incumbent_metrics = evaluate(incumbent, test, target, classification)

        shipped_metrics = evaluate(shipped, test, target, classification)

        cand = primary(candidate_metrics, classification)
        prod = primary(incumbent_metrics, classification)
        change_pct = (cand - prod) / prod * 100
        passed = change_pct <= tolerance_pct

        results.append({
            "key": key, "label": label, "target": target,
            "classification": classification,
            "candidate": cand, "production": prod,
            "shipped_as_is": primary(shipped_metrics, classification),
            "change_pct": change_pct, "passed": passed,
            "candidate_trees": int(candidate.best_iteration),
            "production_trees": trees,
            "candidate_metrics": candidate_metrics,
            "production_metrics": incumbent_metrics,
            "model": candidate,
        })

    return results

ECE_WARN_MULTIPLE = 2.0

def calibration_note(results) -> None:
    """Report calibration. It informs the verdict; it does not decide it.

    Brier and ECE are shown, not gated on. The gate's tolerance is 2%, a number
    chosen against observed month-to-month movement in log loss and MAE; there
    is no equivalent observation for ECE, because no real retrain has run yet.
    A threshold invented now would be a guess wearing a number's clothes, and
    the failure mode is the expensive direction - refusing a good candidate.

    So this warns and lets the log-loss verdict stand. Revisit once several
    real retrains have recorded ECE, and set the bar from that spread.
    """
    classified = [r for r in results if r["classification"]]
    if not classified:
        return

    print()
    for r in classified:
        c, p = r["candidate_metrics"], r["production_metrics"]
        if p["ece"] > 0 and c["ece"] > p["ece"] * ECE_WARN_MULTIPLE:
            state = ("PASSED on log loss" if r["passed"]
                     else "already FAILED on log loss")
            print(f"  WARNING: {r['label']} {state}, and its ECE is "
                  f"{c['ece'] / p['ece']:.1f}x production's "
                  f"({c['ece']:.4f} against {p['ece']:.4f}).")
            if r["passed"]:
                print("  The ranking held up but the probabilities did not.")
                print("  Look at the reliability curve before merging -")
                print("  measure_calibration.py prints it.")
            else:
                print("  Both are worse, so this adds to the refusal rather")
                print("  than complicating it.")
        else:
            print(f"  Calibration: {r['label']} ECE {c['ece']:.4f} against "
                  f"production's {p['ece']:.4f} - no warning.")
    print(f"  ECE is not a gate criterion yet; see calibration_note.")

def print_table(results, tolerance_pct: float) -> None:
    section(f"PROMOTION GATE (a candidate may be up to {tolerance_pct:.1f}% worse)")
    print("PRODUCTION is the shipped architecture RE-FIT on the candidate's own")
    print("training rows, so neither model has seen the test window. The shipped")
    print("weights are not the comparison - see the rightmost column.\n")

    width = max(len(r["label"]) for r in results)
    print(f"{'TARGET':<{width}}  {'METRIC':<9}{'PRODUCTION':>11}{'CANDIDATE':>11}"
          f"{'CHANGE':>9}{'TREES c/p':>11}{'BRIER c/p':>19}{'ECE c/p':>17}"
          f"   {'VERDICT':<18}{'shipped as-is':>14}")
    print("-" * (width + 114))
    for r in results:
        metric = "log loss" if r["classification"] else "MAE"
        verdict = "pass" if r["passed"] else "FAIL - regressed"
        trees = f"{r['candidate_trees']}/{r['production_trees']}"
        if r["classification"]:
            c, p = r["candidate_metrics"], r["production_metrics"]
            brier = f"{c['brier']:.4f}/{p['brier']:.4f}"
            ece = f"{c['ece']:.4f}/{p['ece']:.4f}"
        else:
            # Calibration is a property of a probability. A regression target
            # has no such column, and a dash says so rather than a zero.
            brier = ece = "-"
        print(f"{r['label']:<{width}}  {metric:<9}{r['production']:>11.4f}"
              f"{r['candidate']:>11.4f}{r['change_pct']:>+8.2f}%{trees:>11}"
              f"{brier:>19}{ece:>17}"
              f"   {verdict:<18}{r['shipped_as_is']:>14.4f}")

    calibration_note(results)

    print("\n(negative change = the candidate is better)")
    print("'shipped as-is' scores the deployed weights directly on this window,")
    print("and is never compared - it is shown so the bias stays visible. For")
    print("models built by finalize_models.py it is optimistic, because those")
    print("games ARE in their training set. For a model that genuinely predates")
    print("the window it is honest, and the gap to PRODUCTION is simply the")
    print("value of the extra training data.")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-refresh", action="store_true",
                        help="do not rerun ingestion/preprocessing")
    parser.add_argument("--production-dir", type=Path, default=PRODUCTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=CANDIDATE_DIR)
    parser.add_argument("--test-games", type=int, default=DEFAULT_TEST_GAMES)
    parser.add_argument("--tolerance-pct", type=float, default=DEFAULT_TOLERANCE_PCT)
    parser.add_argument("--min-new-games", type=int, default=DEFAULT_MIN_NEW_GAMES)
    parser.add_argument("--force", action="store_true",
                        help="retrain regardless of how little new data exists")
    args = parser.parse_args()

    section("NEW DATA CHECK")
    before = snapshot(GAMES_FINAL)
    print(f"  before refresh: {before['rows']:,} team-game rows, "
          f"latest {before['latest']}")

    if args.skip_refresh:
        print("  --skip-refresh: pipeline not rerun.")
    else:
        refresh_data()

    after = snapshot(GAMES_FINAL)
    print(f"\n  after refresh:  {after['rows']:,} team-game rows, "
          f"latest {after['latest']}")

    section("DATA PREP")
    dataset = load_dataset()
    print(f"Loaded {len(dataset):,} games.")

    marker = read_trained_through(args.production_dir)
    if marker is None:
        print(f"\n  No {TRAINED_THROUGH} beside the production models, so how "
              f"much data\n  they have already seen is unknown. Not guessing.")
        if not args.force:
            print("  Skipping. Pass --force to retrain anyway; the marker is "
                  "written\n  alongside any candidates produced, so this "
                  "resolves itself after one run.")
            return EXIT_OK
        print("  --force given: continuing.")
    else:
        cutoff = pd.Timestamp(marker["trained_through"])
        new_games = int((dataset["GAME_DATE"] > cutoff).sum())
        print(f"\n  production trained through {marker['trained_through']} "
              f"({marker['games']:,} games)")
        print(f"  genuinely new games since then: {new_games:,}")

        if new_games < args.min_new_games and not args.force:
            print(f"\n  Fewer than {args.min_new_games} new games. Nothing worth "
                  f"retraining on.")
            print("  Pass --force to retrain anyway.")
            return EXIT_OK
        if new_games < args.min_new_games:
            print(f"\n  Below the {args.min_new_games}-game threshold, but "
                  f"--force was given.")

    print()
    train, validation, test = rolling_split(dataset, args.test_games)

    section("TRAINING CANDIDATES AND RE-FITTING PRODUCTION")
    print(f"Production architecture read from {args.production_dir}")
    print("Both models fit on the same training rows; neither sees the test window.")
    results = run_gate(train, validation, test, args.production_dir,
                       args.tolerance_pct)

    print_table(results, args.tolerance_pct)

    passed = [r for r in results if r["passed"]]
    section("RESULT")
    print(f"{len(passed)} of {len(results)} targets passed the gate.")

    if len(passed) != len(results):
        failed = ", ".join(r["label"] for r in results if not r["passed"])
        print(f"\nNOT PROMOTING ANY MODEL. Regressed: {failed}")
        print("All seven ship together - they share a feature set and a")
        print("dataset, so promoting a subset would leave the served models")
        print("trained on different data as each other.")
        return EXIT_REFUSED

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.output_dir.glob("*.json"):
        stale.unlink()
    for r in results:
        r["model"].save_model(str(args.output_dir / f"{r['key']}.json"))
    write_trained_through(args.output_dir, dataset)

    print(f"\nAll passed. {len(results)} candidate models written to "
          f"{args.output_dir}")
    print(f"A {TRAINED_THROUGH} marker was written alongside them, so the next")
    print("run can tell how much data these have already seen.")
    print("ml-training/models/ is UNTOUCHED. Promoting these is a separate,")
    print("human decision - a pull request once the scheduled job exists.")
    return EXIT_OK

if __name__ == "__main__":
    try:
        code = main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        print()
        print("=" * 78)
        print("CRASHED BEFORE REACHING A VERDICT")
        print("=" * 78)
        print("This is NOT a gate refusal. No comparison was made, so nothing")
        print("above says anything about whether a retrained model would be")
        print("better or worse than production.")
        print()
        print(f"Exiting {EXIT_ERROR} so the caller can tell the two "
              f"apart. A real refusal exits {EXIT_REFUSED}.")
        code = EXIT_ERROR
    raise SystemExit(code)
