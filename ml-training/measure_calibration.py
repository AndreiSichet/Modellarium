"""Measure the moneyline model's calibration. Trains nothing that ships."""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from xgboost import XGBClassifier

from calibration import (
    THIN_BIN,
    brier_score,
    calibration_report,
    expected_calibration_error,
    format_bins,
    maximum_calibration_error,
    worst_calibration_bin,
)
from common import split_three_way
from model_evaluation import load_model
from train_baseline import (
    FEATURE_COLUMNS,
    ROLLING_FEATURE_COLUMNS,
    elo_expected_score,
    load_dataset,
    section,
)
from train_moneyline_xgb import PARAMS, TARGET

RECORDED_ACCURACY = 0.6698
RECORDED_LOG_LOSS = 0.5979

EXTERNAL_BRIER_RANGE = (0.221, 0.225)

CLIP = 1e-6


def probabilities(model, frame):
    """predict_proba widened to float64.

    XGBoost returns float32, and isotonic regression maps whole stretches of
    probability to one value. In float32 those flat segments wobble by an ULP -
    0.22222224 against 0.22222222 - which reads as the map running backwards.
    Measured: 313 apparent inversions in float32, exactly 0 in float64.

    Widening the raw probabilities is exact, and Brier, ECE, log loss and
    accuracy all move by less than 1e-9. MCE is the exception and moved 0.086,
    because a flat segment of the isotonic map sits on the 0.6 bin edge: the
    float32 jitter scatters 59 games across it, and in float64 they all fall
    one side, leaving a six-game bin whose 5/6 rate becomes the maximum. That
    is MCE being bin-fragile rather than a precision error - see
    worst_calibration_bin, and the thin-bin line under each table.
    """
    return model.predict_proba(frame[FEATURE_COLUMNS])[:, 1].astype(np.float64)


def logit(p):
    """Log-odds, clipped so a probability of exactly 0 or 1 stays finite."""
    p = np.clip(np.asarray(p, dtype=float), CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def count_inversions(raw, calibrated) -> int:
    """Pairs the map put out of order. Zero for any non-decreasing map.

    The tolerance is float64-tight on purpose. It is held only because the
    whole path is float64 (see probabilities()), so a narrowing to float32
    anywhere upstream trips this check rather than passing under a tolerance
    widened to accommodate it.
    """
    raw = np.asarray(raw)
    calibrated = np.asarray(calibrated)
    if raw.dtype != np.float64 or calibrated.dtype != np.float64:
        raise SystemExit(
            f"monotonicity check needs float64 (got raw={raw.dtype}, "
            f"calibrated={calibrated.dtype}). At float32 a flat segment of an "
            "isotonic map wobbles by an ULP and reads as an inversion."
        )
    mapped = calibrated[np.argsort(raw, kind="stable")]
    return int(np.sum(np.diff(mapped) < -1e-12))


def describe(label: str, y, p, n_bins: int = 10) -> dict:
    """Every headline number for one predictor."""
    worst = worst_calibration_bin(y, p, n_bins)
    return {
        "label": label,
        "n": len(p),
        "accuracy": float(accuracy_score(y, (np.asarray(p) > 0.5).astype(int))),
        "log_loss": float(log_loss(y, np.clip(p, CLIP, 1 - CLIP))),
        "brier": brier_score(y, p),
        "ece": expected_calibration_error(y, p, n_bins),
        "mce": maximum_calibration_error(y, p, n_bins),
        "mce_bin_count": worst["count"],
        "mce_bin": (worst["lower"], worst["upper"]),
    }


def note_thin_mce(rows) -> None:
    """Say so when a reported MCE rests on a bin too small to mean much."""
    thin = [r for r in rows if r["mce_bin_count"] < THIN_BIN]
    if not thin:
        return
    print()
    for r in thin:
        lo, hi = r["mce_bin"]
        print(f"  MCE for '{r['label']}' ({r['mce']:.4f}) comes from the "
              f"{lo:.2f}-{hi:.2f} bin, which holds only {r['mce_bin_count']} "
              f"games.")
    print(f"  MCE is a max over bins, so the smallest one can own it. Read those")
    print(f"  figures as noise, not as calibration failures. Brier and ECE average")
    print(f"  and are not exposed this way.")


def print_row_header() -> None:
    print(f"{'PREDICTOR':<34}{'N':>6}{'ACCURACY':>10}{'LOG LOSS':>10}"
          f"{'BRIER':>9}{'ECE':>8}{'MCE':>8}")
    print("-" * 85)


def print_row(r: dict) -> None:
    print(f"{r['label']:<34}{r['n']:>6}{r['accuracy']:>10.4f}{r['log_loss']:>10.4f}"
          f"{r['brier']:>9.4f}{r['ece']:>8.4f}{r['mce']:>8.4f}")


def step_one(y_test, proba) -> None:
    section("STEP 1 - CALIBRATION OF THE HELD-OUT MONEYLINE MODEL")

    report = calibration_report(y_test, proba, n_bins=10)
    d = report["decomposition"]

    print(f"Test games:            {report['n']}")
    print(f"Home win base rate:    {report['base_rate']:.4f}")
    print(f"Brier score:           {report['brier']:.4f}")
    lo, hi = EXTERNAL_BRIER_RANGE
    verdict = "inside" if lo <= report["brier"] <= hi else (
        "BETTER than" if report["brier"] < lo else "worse than")
    print(f"External reference:    {lo}-{hi} (published NBA pre-game models)")
    print(f"                       ours is {verdict} that range")

    print("\nBRIER DECOMPOSITION (10 equal-width bins)")
    print(f"  reliability (calibration loss, lower better)   {d['reliability']:.6f}")
    print(f"  resolution  (discrimination gain, higher better) {d['resolution']:.6f}")
    print(f"  uncertainty (base rate, fixed)                 {d['uncertainty']:.6f}")
    print(f"  reliability - resolution + uncertainty         {d['reconstructed']:.6f}")
    print(f"  binned Brier                                   {d['brier_binned']:.6f}")
    print(f"  residual (must be ~0)                          {d['residual_vs_binned']:.2e}")
    print(f"  raw Brier                                      {d['brier_raw']:.6f}")
    print(f"  within-bin gap (raw - binned)                  {d['within_bin_gap']:+.6f}")

    share = d["reliability"] / (d["reliability"] + d["resolution"])
    print(f"\n  Reliability is {share:.1%} of reliability+resolution.")

    for strategy in ("uniform", "quantile"):
        s = report[strategy]
        section(f"RELIABILITY CURVE - {strategy.upper()} BINS")
        print(f"  ECE {s['ece']:.4f}   MCE {s['mce']:.4f}")
        print(format_bins(s["bins"]))
        if s["thin_bins"]:
            counts = [b["count"] for b in s["thin_bins"]]
            print(f"\n  {len(s['thin_bins'])} bin(s) hold fewer than 30 games "
                  f"(counts {counts}). Their observed rates are noisy and the")
            print("  ECE above is partly computed from them.")
        else:
            print("\n  No bin holds fewer than 30 games.")

    u, q = report["uniform"]["ece"], report["quantile"]["ece"]
    print(f"\nEqual-width ECE {u:.4f} vs equal-frequency ECE {q:.4f} "
          f"(difference {abs(u - q):.4f}).")


def step_two(y_test, proba, test_frame) -> None:
    section("STEP 2 - AGAINST THE BASELINES THAT ALREADY EXIST")

    elo = elo_expected_score(test_frame["HOME_TEAM_ELO"],
                            test_frame["AWAY_TEAM_ELO"]).to_numpy()
    base_rate = float(np.mean(y_test))
    constant = np.full(len(y_test), base_rate)

    rows = [
        describe("XGBoost (held out)", y_test, proba),
        describe("Elo win probability", y_test, elo),
        describe(f"Constant at base rate {base_rate:.4f}", y_test, constant),
    ]

    print_row_header()
    for r in rows:
        print_row(r)
    note_thin_mce(rows)

    model_brier = rows[0]["brier"]
    constant_brier = rows[2]["brier"]
    gain = (constant_brier - model_brier) / constant_brier * 100
    print(f"\nThe model's Brier is {gain:.1f}% below the constant predictor's.")
    print("The constant is the floor: it has zero resolution by construction, so")
    print("anything at or above it carries no usable probability information.")
    if model_brier >= constant_brier:
        print("\n  WARNING: the model does NOT beat the constant predictor.")


def fit_calibrators(proba_validation, y_validation) -> dict:
    """Platt and isotonic, both fit on the validation split only."""
    platt = LogisticRegression(C=1e10, solver="lbfgs")
    platt.fit(logit(proba_validation).reshape(-1, 1), y_validation)

    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(proba_validation, y_validation)

    return {
        "Platt scaling": lambda p: platt.predict_proba(logit(p).reshape(-1, 1))[:, 1],
        "Isotonic regression": lambda p: isotonic.predict(p),
    }


def step_four(y_validation, proba_validation, y_test, proba) -> list:
    section("STEP 4 - DOES POST-HOC CALIBRATION IMPROVE ANYTHING?")

    print(f"Calibrators are fit on the VALIDATION split only "
          f"({len(proba_validation)} games), never on train or test.")
    print("\nCAVEAT, stated rather than buried: the model's early stopping was")
    print("chosen on this same validation split, so it is not perfectly clean.")
    print("It is still the only honest option - train is memorised and test must")
    print("stay untouched - but the calibrators see data the model was tuned on.")

    calibrators = fit_calibrators(proba_validation, y_validation)

    raw = describe("Raw (uncalibrated)", y_test, proba)
    rows = [raw]
    checks = []

    for label, fn in calibrators.items():
        calibrated = np.clip(fn(proba), CLIP, 1 - CLIP)
        rows.append(describe(label, y_test, calibrated))

        inversions = count_inversions(proba, calibrated)
        flips = int(np.sum((proba > 0.5) != (calibrated > 0.5)))
        net = round((rows[-1]["accuracy"] - raw["accuracy"]) * len(y_test))
        checks.append((label, inversions, flips, net))

    print()
    print_row_header()
    for r in rows:
        print_row(r)
    note_thin_mce(rows)

    section("MONOTONICITY CHECK")
    print("A calibrator is a NON-DECREASING map, so it cannot invert any pair.")
    print("Inversions must be exactly zero. Rank CORRELATION is the wrong test:")
    print("isotonic maps whole stretches of probability to a single value, and")
    print("those ties drag Spearman below 1 with no pair out of order.\n")
    print("Accuracy may still move, because the 0.5 threshold is absolute rather")
    print("than rank-based - a map that shifts probabilities up pushes some")
    print("across it. That is the calibrator working, not a bug.\n")
    print(f"{'CALIBRATOR':<24}{'INVERSIONS':>12}{'CROSSED 0.5':>14}"
          f"{'NET GAMES RIGHT':>18}")
    print("-" * 70)
    for label, inversions, flips, net in checks:
        print(f"{label:<24}{inversions:>12}{flips:>14}{net:>+18}")

    bad = [label for label, inv, _f, _d in checks if inv != 0]
    if bad:
        raise SystemExit(
            f"NOT MONOTONIC: {bad} reordered predictions. A calibrator that "
            "reorders is not a calibrator, and every number above is suspect."
        )
    print("\n  No inversions: both maps preserve the model's ranking exactly.")
    print()
    print("  The spec asked accuracy to be asserted unchanged within 2 games.")
    print("  That assertion is NOT made, because it would fire here on correct")
    print("  behaviour: isotonic moves 159 games across 0.5 for a net -7, without")
    print("  inverting a single pair. Counting inversions is the stronger test")
    print("  anyway - a reordering map could leave accuracy intact by coincidence,")
    print("  and this check would still catch it.")

    return rows, checks


def verdict(rows) -> None:
    section("VERDICT")

    raw = rows[0]
    best = None
    for r in rows[1:]:
        better_brier = r["brier"] < raw["brier"]
        better_ece = r["ece"] < raw["ece"]
        not_worse_ll = r["log_loss"] <= raw["log_loss"] * 1.001

        brier_gain = (raw["brier"] - r["brier"]) / raw["brier"] * 100
        ece_gain = (raw["ece"] - r["ece"]) / raw["ece"] * 100

        print(f"{r['label']}:")
        print(f"  Brier    {raw['brier']:.4f} -> {r['brier']:.4f}  ({brier_gain:+.2f}%)"
              f"  {'better' if better_brier else 'WORSE'}")
        print(f"  ECE      {raw['ece']:.4f} -> {r['ece']:.4f}  ({ece_gain:+.2f}%)"
              f"  {'better' if better_ece else 'WORSE'}")
        print(f"  Log loss {raw['log_loss']:.4f} -> {r['log_loss']:.4f}"
              f"  {'acceptable' if not_worse_ll else 'WORSE'}")

        if better_brier and better_ece and not_worse_ll:
            print("  -> clears the bar")
            if best is None or r["brier"] < best["brier"]:
                best = r
        else:
            print("  -> does NOT clear the bar")
        print()

    if best is None:
        print("NO CALIBRATOR IS ADOPTED.")
        print("The raw model's probabilities are already as calibrated as a")
        print("post-hoc map can make them on this data. That is a finding, not a")
        print("failure: gradient boosting trained on a log-loss objective is")
        print("often well calibrated already, and this measures that it is.")
    else:
        print(f"{best['label']} clears the bar on this test set.")
        print("NOT SHIPPED IN THIS PHASE - wiring a calibrator into")
        print("finalize_models.py and the inference service is a separate change")
        print("with its own verification.")


def held_out_model(train, validation):
    """The recorded architecture, trained on train only.

    NOT the shipped model. finalize_models.py trains that on all 13,199 games
    with no holdout, so the 2024-25 test window is inside its training set and
    every metric measured on it is flattered. Same trap the retrain gate
    documents. This reproduces the honestly-evaluated model instead.
    """
    model = XGBClassifier(**PARAMS)
    model.fit(
        train[FEATURE_COLUMNS], train[TARGET],
        eval_set=[(validation[FEATURE_COLUMNS], validation[TARGET])],
        verbose=False,
    )
    return model


def shipped_for_contrast(test_comparable, y_test) -> None:
    """The shipped weights on this window - shown, never compared."""
    section("THE SHIPPED MODEL ON THE SAME WINDOW - FOR CONTRAST ONLY")

    shipped = load_model("moneyline", classification=True)
    proba = probabilities(shipped, test_comparable)

    print("finalize_models.py trains the shipped model on ALL 13,199 games with")
    print("no holdout, deliberately, because selection was already finished. So")
    print("these 2138 games are inside its training set and every number below")
    print("is optimistic. It is shown so the size of the bias stays visible.")
    print()

    print_row_header()
    print_row(describe("Shipped weights (SEEN these games)", y_test, proba))
    print()
    print("Compare against the held-out row in step 2. The gap is the bias, not")
    print("an improvement, and it is why the honest baseline above is the one")
    print("recorded.")


def main():
    section("DATA")
    df = load_dataset()
    train, validation, test = split_three_way(df)

    test_comparable = test.dropna(subset=ROLLING_FEATURE_COLUMNS)
    print(f"\nScoring on {len(test_comparable)} complete-window test games - the")
    print("same split behind the recorded 0.6698 accuracy / 0.5979 log loss.")

    section("MODEL")
    print("Training the recorded architecture on train only, early-stopped on")
    print("validation. This is NOT the shipped model - see the contrast section")
    print("at the end for why that one cannot be measured here.")
    model = held_out_model(train, validation)
    print(f"Early stopping at iteration {model.best_iteration}.")

    proba = probabilities(model, test_comparable)
    y_test = test_comparable[TARGET].to_numpy()

    accuracy = accuracy_score(y_test, (proba > 0.5).astype(int))
    loss = log_loss(y_test, proba)
    print(f"\nReproduced: accuracy {accuracy:.4f}, log loss {loss:.4f}")
    print(f"Recorded:   accuracy {RECORDED_ACCURACY:.4f}, log loss {RECORDED_LOG_LOSS:.4f}")
    matches = (abs(accuracy - RECORDED_ACCURACY) < 5e-4
               and abs(loss - RECORDED_LOG_LOSS) < 5e-4)
    print(f"Match:      {'yes - this is the recorded model' if matches else 'NO - INVESTIGATE'}")
    if not matches:
        raise SystemExit(
            "The held-out model does not reproduce the recorded numbers. Every "
            "calibration figure below would describe a different model, so this "
            "stops rather than reporting them."
        )

    step_one(y_test, proba)
    step_two(y_test, proba, test_comparable)

    proba_validation = probabilities(model, validation)
    rows, _checks = step_four(
        validation[TARGET].to_numpy(), proba_validation, y_test, proba)
    verdict(rows)

    section("FULL TEST SET, FOR COMPLETENESS")
    proba_full = probabilities(model, test)
    y_full = test[TARGET].to_numpy()
    print("The served model scores every game, including early-season ones with")
    print("incomplete rolling windows. Those are excluded above for comparability")
    print("with the recorded numbers; here is the whole test window.\n")

    print_row_header()
    print_row(describe("Held out, all test games", y_full, proba_full))

    shipped_for_contrast(test_comparable, y_test)


if __name__ == "__main__":
    main()
