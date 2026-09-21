"""Prove the retrain mechanism now, using history in place of "the future"."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from common import FEATURE_COLUMNS
from continuous_retrain import write_trained_through
from finalize_models import final_params
from model_evaluation import TASKS
from train_baseline import load_dataset
from train_moneyline_xgb import PARAMS as MONEYLINE_PARAMS
from train_regression_xgb import PARAMS as REGRESSION_PARAMS
from xgboost import XGBClassifier, XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data-pipeline" / "data" / "processed" / "model_dataset.csv"
RETRAIN = Path(__file__).resolve().parent / "continuous_retrain.py"

CUTOFF = "2026-02-01"

TEST_GAMES = 300

def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")

def train_simulated_production(truncated, directory: Path):
    """Train the seven models on pre-cutoff data only."""
    directory.mkdir(parents=True, exist_ok=True)
    ordered = truncated.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    validation = ordered.iloc[-TEST_GAMES:]
    train = ordered.iloc[:-TEST_GAMES]

    print(f"  training on {len(train):,} games, early-stopping on "
          f"{len(validation):,}")
    for key, label, target, classification in TASKS:
        params = MONEYLINE_PARAMS if classification else REGRESSION_PARAMS
        searching = (XGBClassifier if classification else XGBRegressor)(**params)
        searching.fit(
            train[FEATURE_COLUMNS], train[target],
            eval_set=[(validation[FEATURE_COLUMNS], validation[target])],
            verbose=False,
        )
        trees = int(searching.best_iteration)

        model = (XGBClassifier if classification else XGBRegressor)(
            **final_params(classification, trees))
        model.fit(train[FEATURE_COLUMNS], train[target], verbose=False)
        model.save_model(str(directory / f"{key}.json"))
        print(f"    {label:<12} {trees:>4} trees "
              f"(read back: {model.get_booster().num_boosted_rounds()})")

    write_trained_through(directory, truncated)
    print(f"  marker written: trained through "
          f"{truncated['GAME_DATE'].max().date()}")

def main():
    section(f"SIMULATED CUTOFF: pretending today is {CUTOFF}")

    full = load_dataset()
    truncated = full[full["GAME_DATE"] < CUTOFF]
    new_games = full[full["GAME_DATE"] >= CUTOFF]

    print(f"  full dataset          : {len(full):,} games, "
          f"{full['GAME_DATE'].min().date()} to {full['GAME_DATE'].max().date()}")
    print(f"  'old production' sees : {len(truncated):,} games (before {CUTOFF})")
    print(f"  standing in for 'new' : {len(new_games):,} games "
          f"({new_games['GAME_DATE'].min().date()} to "
          f"{new_games['GAME_DATE'].max().date()})")

    if len(new_games) < TEST_GAMES:
        raise SystemExit(
            f"only {len(new_games)} games after the cutoff, need at least "
            f"{TEST_GAMES} for the test window to sit in the new region."
        )

    workdir = Path(tempfile.mkdtemp(prefix="retrain_verify_"))
    old_models = workdir / "models_february"
    candidates = workdir / "models_candidate"

    try:
        section("STEP 1: train the stand-in for February's production models")
        train_simulated_production(truncated, old_models)

        section("STEP 2: run the real gate, unmodified")
        print("continuous_retrain.py, with:")
        print(f"  --production-dir  {old_models.name}   (the February stand-in)")
        print(f"  --test-games      {TEST_GAMES}")
        print("  --skip-refresh    (the cutoff simulation replaces re-fetching)")
        print("  NO --force        (the new-data trigger must fire on its own,")
        print("                     off the marker written in step 1)\n")

        result = subprocess.run(
            [sys.executable, str(RETRAIN),
             "--skip-refresh",
             "--production-dir", str(old_models),
             "--output-dir", str(candidates),
             "--test-games", str(TEST_GAMES)],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True,
        )
        tail = result.stdout.split("PROMOTION GATE")
        print("PROMOTION GATE" + tail[-1] if len(tail) > 1 else result.stdout[-3000:])
        if result.returncode not in (0, 1):
            print("STDERR:", result.stderr[-2000:])

        section("STEP 3: what the run proved")
        promoted = [p for p in candidates.glob("*.json")
                    if p.name != "trained_through.json"] if candidates.exists() else []
        checks = [
            ("the gate ran to a verdict", result.returncode in (0, 1)),
            ("the new-data trigger fired on its own, without --force",
             "genuinely new games since then" in result.stdout
             and "Nothing worth retraining" not in result.stdout),
            ("production was RE-FIT, not scored as shipped",
             "RE-FITTING PRODUCTION" in result.stdout
             and "shipped as-is" in result.stdout),
            ("the test window sits in the post-cutoff region",
             "<- the gate's window" in result.stdout),
            ("a per-target verdict was produced for all 7",
             result.stdout.count("pass") + result.stdout.count("FAIL - regressed") >= 7),
            ("ml-training/models/ was not touched",
             not any(p.stat().st_mtime > (workdir.stat().st_mtime)
                     for p in (PROJECT_ROOT / "ml-training" / "models").glob("*.json"))),
        ]
        for name, ok in checks:
            print(f"  [{'OK  ' if ok else 'FAIL'}] {name}")

        print(f"\n  candidate models written: {len(promoted)}"
              f"{' (gate passed)' if promoted else ' (gate blocked promotion)'}")
        meaning = {0: "promoted", 1: "refused to promote"}.get(
            result.returncode, "CRASHED - no verdict reached")
        print(f"  retrain exit code: {result.returncode} ({meaning})")

        return 0 if all(ok for _, ok in checks) else 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

if __name__ == "__main__":
    raise SystemExit(main())
