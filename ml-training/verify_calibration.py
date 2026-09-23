"""Check the calibration metrics against arithmetic, not against themselves."""

import sys

import numpy as np

from calibration import (
    brier_decomposition,
    brier_score,
    calibration_bins,
    expected_calibration_error,
    maximum_calibration_error,
)

TOLERANCE = 1e-12

failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail:
        print(f"        {detail}")
    if not ok:
        failures.append(name)


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def hand_worked_ece():
    """Ten predictions in known bins, ECE computed by hand."""
    section("1. ECE AGAINST HAND ARITHMETIC")

    y = [0, 0, 1, 0, 1, 1, 0, 1, 1, 1]
    p = [0.05, 0.05, 0.15, 0.15, 0.45, 0.45, 0.85, 0.85, 0.95, 0.95]

    # Bin 0.0-0.1 : preds .05 .05        mean .05,  observed 0/2 = 0.00, gap .05
    # Bin 0.1-0.2 : preds .15 .15        mean .15,  observed 1/2 = 0.50, gap .35
    # Bin 0.4-0.5 : preds .45 .45        mean .45,  observed 2/2 = 1.00, gap .55
    # Bin 0.8-0.9 : preds .85 .85        mean .85,  observed 1/2 = 0.50, gap .35
    # Bin 0.9-1.0 : preds .95 .95        mean .95,  observed 2/2 = 1.00, gap .05
    # Every bin holds 2 of 10, so ECE is the plain mean of the gaps:
    expected_ece = (0.05 + 0.35 + 0.55 + 0.35 + 0.05) / 5
    expected_mce = 0.55

    got_ece = expected_calibration_error(y, p, n_bins=10)
    got_mce = maximum_calibration_error(y, p, n_bins=10)

    check("ECE matches hand arithmetic",
          abs(got_ece - expected_ece) < TOLERANCE,
          f"by hand {expected_ece:.6f}, computed {got_ece:.6f}")
    check("MCE is the worst single bin",
          abs(got_mce - expected_mce) < TOLERANCE,
          f"by hand {expected_mce:.6f}, computed {got_mce:.6f}")

    bins = calibration_bins(y, p, n_bins=10)
    check("five non-empty bins, two games each",
          len(bins) == 5 and all(b["count"] == 2 for b in bins),
          f"{len(bins)} bins, counts {[b['count'] for b in bins]}")

    # Brier by hand: each pair contributes 2*(p - y)^2, averaged over 10.
    expected_brier = (
        2 * 0.05 ** 2 + (0.15 ** 2 + 0.85 ** 2) + (0.55 ** 2 + 0.55 ** 2)
        + (0.85 ** 2 + 0.15 ** 2) + 2 * 0.05 ** 2
    ) / 10
    check("Brier matches hand arithmetic",
          abs(brier_score(y, p) - expected_brier) < TOLERANCE,
          f"by hand {expected_brier:.6f}, computed {brier_score(y, p):.6f}")


def decomposition_identity():
    """reliability - resolution + uncertainty must equal the BINNED Brier."""
    section("2. BRIER DECOMPOSITION RECONCILES")

    rng = np.random.default_rng(42)
    for label, p in [
        ("well calibrated", rng.uniform(0.2, 0.8, 4000)),
        ("clustered mid",   rng.normal(0.55, 0.08, 4000).clip(0.01, 0.99)),
        ("overconfident",   rng.beta(0.5, 0.5, 4000).clip(0.01, 0.99)),
    ]:
        y = (rng.uniform(size=len(p)) < p).astype(int)
        d = brier_decomposition(y, p)
        check(f"{label}: identity holds against the binned Brier",
              abs(d["residual_vs_binned"]) < 1e-10,
              f"binned {d['brier_binned']:.8f}, "
              f"rel-res+unc {d['reconstructed']:.8f}, "
              f"residual {d['residual_vs_binned']:.2e}")
        print(f"        raw Brier {d['brier_raw']:.6f}, "
              f"within-bin gap {d['within_bin_gap']:+.6f}")


def responds_to_miscalibration():
    """A deliberately miscalibrated predictor must score worse on calibration
    while its accuracy is untouched."""
    section("3. NEGATIVE TEST - DELIBERATE MISCALIBRATION")

    rng = np.random.default_rng(7)
    p = rng.uniform(0.05, 0.95, 5000)
    y = (rng.uniform(size=len(p)) < p).astype(int)

    # Push predictions toward the extremes without reordering them: sqrt above
    # 0.5, square below. Monotonic, so the ranking - and therefore accuracy -
    # is unchanged.
    skewed = np.where(p > 0.5, np.sqrt(p), p ** 2)

    base_acc = float(((p > 0.5).astype(int) == y).mean())
    skew_acc = float(((skewed > 0.5).astype(int) == y).mean())

    base_brier, skew_brier = brier_score(y, p), brier_score(y, skewed)
    base_ece = expected_calibration_error(y, p)
    skew_ece = expected_calibration_error(y, skewed)

    check("skewing made Brier worse",
          skew_brier > base_brier,
          f"{base_brier:.4f} -> {skew_brier:.4f}")
    check("skewing made ECE worse",
          skew_ece > base_ece,
          f"{base_ece:.4f} -> {skew_ece:.4f}")
    check("accuracy is unchanged (the map is monotonic)",
          abs(base_acc - skew_acc) < TOLERANCE,
          f"{base_acc:.6f} vs {skew_acc:.6f}")


def perfect_and_worst_cases():
    """Sanity anchors at both ends of the range."""
    section("4. ANCHORS")

    y = [0, 1] * 500
    perfect = [0.0, 1.0] * 500
    check("a perfect predictor scores Brier 0 and ECE 0",
          brier_score(y, perfect) < TOLERANCE
          and expected_calibration_error(y, perfect) < TOLERANCE)

    inverted = [1.0, 0.0] * 500
    check("an inverted predictor scores Brier 1 and ECE 1",
          abs(brier_score(y, inverted) - 1.0) < TOLERANCE
          and abs(expected_calibration_error(y, inverted) - 1.0) < TOLERANCE)

    constant = [0.5] * 1000
    d = brier_decomposition(y, constant)
    check("a constant predictor has zero resolution",
          abs(d["resolution"]) < TOLERANCE,
          f"resolution {d['resolution']:.2e}, uncertainty {d['uncertainty']:.4f}")


def main() -> int:
    hand_worked_ece()
    decomposition_identity()
    responds_to_miscalibration()
    perfect_and_worst_cases()

    section("RESULT")
    if failures:
        print(f"  {len(failures)} FAILED: {failures}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
