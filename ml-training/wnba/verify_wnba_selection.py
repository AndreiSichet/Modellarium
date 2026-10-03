"""Verify the selection machinery BEFORE the test seasons are touched.

Four things, each negative-tested where a negative test means anything:
  - Elo is refit per fold on that fold's training seasons only
  - the scaler sees training rows only
  - phase 2's lag guard still passes against the dataset being used
  - the test-once guard actually refuses
"""

import io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from wnba_common import (  # noqa: E402
    MODELS_DIR, TEST_SEASONS, feature_columns, fit_elo_on, folds,
    load_dataset, load_long, section,
)

results = []


def record(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"         {detail}")
    results.append(ok)


def elo_refit_isolation(long_frame):
    section("1. ELO IS REFIT PER FOLD, ON THAT FOLD'S TRAINING SEASONS ONLY")

    fitted = {}
    for validation_season, training in folds():
        k, carryover, _ = fit_elo_on(long_frame, training)
        fitted[validation_season] = (k, carryover)
        print(f"  fold {validation_season}: trained on "
              f"{training[0]}-{training[-1]} -> K={k}, carryover={carryover:.3f}")

    record("the fitted parameters are not identical across all folds",
           len(set(fitted.values())) > 1,
           f"{len(set(fitted.values()))} distinct (K, carryover) pairs - if this "
           "were 1, a single fit might be being reused")

    # NEGATIVE: corrupt the season a fold is scored on. Its parameters must not
    # move, because the objective sums only over that fold's training seasons.
    validation_season, training = folds()[0]
    corrupted = long_frame.copy()
    mask = corrupted["SEASON"] == validation_season
    corrupted.loc[mask, "WL"] = corrupted.loc[mask, "WL"].map({"W": "L", "L": "W"})
    k2, c2, _ = fit_elo_on(corrupted, training)
    record(f"flipping all {int(mask.sum())} rows of validation season "
           f"{validation_season} leaves fold {validation_season}'s fit unchanged",
           (k2, c2) == fitted[validation_season],
           f"K={k2}, carryover={c2:.3f}")

    # POSITIVE CONTROL. Phase 2's lesson applies: flipping EVERY result of a
    # season is self-inverting, so half the training games are randomised
    # instead, which destroys signal rather than mirroring it.
    rng = np.random.default_rng(0)
    corrupted = long_frame.copy()
    train_games = corrupted.loc[corrupted["SEASON"].isin(training),
                                "GAME_ID"].unique()
    flip = set(rng.choice(train_games, size=len(train_games) // 2, replace=False))
    hit = corrupted["GAME_ID"].isin(flip)
    corrupted.loc[hit, "WL"] = corrupted.loc[hit, "WL"].map({"W": "L", "L": "W"})
    k3, c3, _ = fit_elo_on(corrupted, training)
    record(f"randomising half of fold {validation_season}'s TRAINING games "
           "DOES move its fit",
           (k3, c3) != fitted[validation_season],
           f"K={k3}, carryover={c3:.3f} vs K={fitted[validation_season][0]}, "
           f"carryover={fitted[validation_season][1]:.3f}")


def scaler_isolation(dataset):
    section("2. THE SCALER SEES TRAINING ROWS ONLY")
    validation_season, training = folds()[0]
    features = feature_columns("CARRY5")
    usable = dataset.dropna(subset=features)
    train = usable[usable["SEASON"].isin(training)]
    validate = usable[usable["SEASON"] == validation_season]

    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(train[features], train["HOME_MARGIN"])
    scaler = model.named_steps["standardscaler"]

    train_mean = train[features].to_numpy().mean(axis=0)
    both_mean = pd.concat([train, validate])[features].to_numpy().mean(axis=0)

    record("the fitted scaler's means equal the TRAINING rows' means",
           np.allclose(scaler.mean_, train_mean),
           f"largest difference {np.abs(scaler.mean_ - train_mean).max():.2e}")
    record("and differ from train+validation means (so validation was excluded)",
           not np.allclose(scaler.mean_, both_mean),
           f"largest difference {np.abs(scaler.mean_ - both_mean).max():.4f}")


def rerun_phase2_guard():
    section("3. PHASE 2'S LAG GUARD, RERUN AGAINST THE SAME DATASET")
    script = (PROJECT / "data-pipeline" / "wnba" / "preprocessing"
              / "verify_wnba_features.py")
    proc = subprocess.run([sys.executable, str(script)],
                          capture_output=True, text=True, cwd=PROJECT)
    tail = [l for l in proc.stdout.splitlines() if "checks passed" in l]
    record("phase 2's feature verifier still passes", proc.returncode == 0,
           tail[0].strip() if tail else f"exit {proc.returncode}")


def test_guard_refuses():
    section("4. THE TEST-ONCE GUARD REFUSES")
    script = HERE / "test_wnba_models.py"

    proc = subprocess.run([sys.executable, str(script)],
                          capture_output=True, text=True, cwd=PROJECT)
    record("refuses with no flag", proc.returncode != 0,
           [l for l in proc.stdout.splitlines() if "Refusing" in l][:1])

    receipt = MODELS_DIR / "TEST_WAS_EVALUATED.json"
    existed = receipt.exists()
    if not existed:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        receipt.write_text('{"when": "probe", "selection": "probe"}')
    proc = subprocess.run([sys.executable, str(script),
                           "--i-am-ready-to-touch-test"],
                          capture_output=True, text=True, cwd=PROJECT)
    record("refuses WITH the flag when a receipt already exists",
           proc.returncode != 0,
           [l for l in proc.stdout.splitlines() if "REFUSING" in l][:1])
    if not existed:
        receipt.unlink()
        print("         (probe receipt removed - test has NOT been scored yet)")


def main() -> int:
    print(__doc__)
    dataset = load_dataset()
    long_frame = load_long()
    print(f"{len(dataset):,} games. Test seasons {TEST_SEASONS} are not read "
          "by any check here.")

    elo_refit_isolation(long_frame)
    scaler_isolation(dataset)
    rerun_phase2_guard()
    test_guard_refuses()

    section("RESULT")
    print(f"  {sum(results)} of {len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
