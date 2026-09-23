"""Calibration metrics for probability predictions. Measurement only."""

import numpy as np

DEFAULT_BINS = 10

THIN_BIN = 30


def brier_score(y_true, y_prob) -> float:
    """Mean squared error between predicted probability and outcome."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def calibration_bins(y_true, y_prob, n_bins: int = DEFAULT_BINS,
                     strategy: str = "uniform") -> list:
    """Per-bin count, mean prediction and observed rate. Empty bins dropped."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.quantile(y_prob, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    # right=True with a 0.0 left edge, so a prediction of exactly 0 lands in
    # bin 1 rather than bin 0, and every bin is (lo, hi].
    index = np.digitize(y_prob, edges[1:-1], right=True)

    bins = []
    for k in range(len(edges) - 1):
        mask = index == k
        count = int(mask.sum())
        if count == 0:
            continue
        bins.append({
            "lower": float(edges[k]),
            "upper": float(edges[k + 1]),
            "count": count,
            "mean_predicted": float(y_prob[mask].mean()),
            "observed_rate": float(y_true[mask].mean()),
        })
    return bins


def expected_calibration_error(y_true, y_prob, n_bins: int = DEFAULT_BINS,
                               strategy: str = "uniform") -> float:
    """Count-weighted mean gap between predicted and observed rate."""
    bins = calibration_bins(y_true, y_prob, n_bins, strategy)
    total = sum(b["count"] for b in bins)
    return float(sum(
        b["count"] * abs(b["mean_predicted"] - b["observed_rate"]) for b in bins
    ) / total)


def worst_calibration_bin(y_true, y_prob, n_bins: int = DEFAULT_BINS,
                          strategy: str = "uniform") -> dict:
    """The bin MCE reports. Returned whole so its count stays visible.

    MCE is a max over bins, so it is set by whichever bin is smallest and
    unluckiest - unlike ECE and Brier, which average. A six-game bin can own
    it outright, which is why callers get the count and not just the gap.
    """
    bins = calibration_bins(y_true, y_prob, n_bins, strategy)
    return max(bins, key=lambda b: abs(b["mean_predicted"] - b["observed_rate"]))


def maximum_calibration_error(y_true, y_prob, n_bins: int = DEFAULT_BINS,
                              strategy: str = "uniform") -> float:
    """The worst single bin's gap."""
    b = worst_calibration_bin(y_true, y_prob, n_bins, strategy)
    return float(abs(b["mean_predicted"] - b["observed_rate"]))


def brier_decomposition(y_true, y_prob, n_bins: int = DEFAULT_BINS,
                        strategy: str = "uniform") -> dict:
    """Murphy decomposition: reliability, resolution, uncertainty.

    The identity reliability - resolution + uncertainty holds exactly against
    the BINNED Brier score, not the raw one. Binning replaces each forecast
    with its bin's mean, and the difference between the two Briers is the
    within-bin spread that the decomposition cannot see. Both are returned so
    that gap is visible rather than absorbed into a tolerance.
    """
    y_true = np.asarray(y_true, dtype=float)
    bins = calibration_bins(y_true, y_prob, n_bins, strategy)
    total = sum(b["count"] for b in bins)
    base_rate = float(y_true.mean())

    reliability = sum(
        b["count"] * (b["mean_predicted"] - b["observed_rate"]) ** 2 for b in bins
    ) / total
    resolution = sum(
        b["count"] * (b["observed_rate"] - base_rate) ** 2 for b in bins
    ) / total
    uncertainty = base_rate * (1.0 - base_rate)

    brier_binned = sum(
        b["count"] * ((b["mean_predicted"] - b["observed_rate"]) ** 2
                      + b["observed_rate"] * (1.0 - b["observed_rate"]))
        for b in bins
    ) / total

    raw = brier_score(y_true, y_prob)
    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "brier_binned": float(brier_binned),
        "brier_raw": float(raw),
        "reconstructed": float(reliability - resolution + uncertainty),
        "residual_vs_binned": float(brier_binned - (reliability - resolution + uncertainty)),
        "within_bin_gap": float(raw - brier_binned),
    }


def thin_bins(bins: list, threshold: int = THIN_BIN) -> list:
    """Bins holding too few games for their rate to mean much."""
    return [b for b in bins if b["count"] < threshold]


def calibration_report(y_true, y_prob, n_bins: int = DEFAULT_BINS) -> dict:
    """Every calibration number for one set of predictions, both binnings."""
    report = {
        "n": int(len(y_prob)),
        "base_rate": float(np.mean(np.asarray(y_true, dtype=float))),
        "brier": brier_score(y_true, y_prob),
        "decomposition": brier_decomposition(y_true, y_prob, n_bins, "uniform"),
    }
    for strategy in ("uniform", "quantile"):
        bins = calibration_bins(y_true, y_prob, n_bins, strategy)
        report[strategy] = {
            "ece": expected_calibration_error(y_true, y_prob, n_bins, strategy),
            "mce": maximum_calibration_error(y_true, y_prob, n_bins, strategy),
            "bins": bins,
            "thin_bins": thin_bins(bins),
        }
    return report


def format_bins(bins: list, threshold: int = THIN_BIN) -> str:
    """The reliability curve as a table, thin bins marked."""
    lines = [f"  {'BIN':<14}{'COUNT':>7}{'PREDICTED':>11}{'OBSERVED':>10}{'GAP':>9}   "]
    for b in bins:
        gap = b["mean_predicted"] - b["observed_rate"]
        flag = "  THIN" if b["count"] < threshold else ""
        lines.append(
            f"  {b['lower']:.2f}-{b['upper']:.2f}     {b['count']:>7}"
            f"{b['mean_predicted']:>11.4f}{b['observed_rate']:>10.4f}"
            f"{gap:>+9.4f}{flag}"
        )
    return "\n".join(lines)
