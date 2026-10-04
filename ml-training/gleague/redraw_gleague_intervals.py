"""Re-draw the test bootstrap intervals across seeds. Robustness only.

WHY THIS EXISTS. The single test run reported `model - Elo alone` on the
moneyline at -0.0104 with a 95% CI of [-0.0202, -0.0004]. That excludes zero,
and its nearer bound sits four ten-thousandths from it. §40 established the
rule after an interval landed at [-0.0908, -0.0011]: any interval whose nearer
bound is that close should be re-drawn across seeds before "excludes zero" is
claimed, because a single draw can place it either side.

THIS MAKES NO SELECTION AND WRITES NOTHING. The configurations are those
already in selection.json, the test receipt is left intact, and no number here
can change which model ships - it only reports how stable an
already-published interval is. That is the same shape as the WNBA's
moneyline_vs_elo.py, which bounded a difference phase 3 had reported as a bare
point estimate.

It reproduces the recorded test scores before reporting anything, and refuses
if any of them misses.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gleague_common import (  # noqa: E402
    IDENTITY_PATH, MODELS_DIR, TARGETS, TEST_SEASONS, bootstrap_difference,
    errors, feature_columns, fit_elo_on, load_dataset, load_long, score,
    section, training_seasons, usable, with_fold_elo)
from select_gleague_models import (  # noqa: E402
    fit_predict, naive_prediction, split_inner)

SEEDS = list(range(10))
TOLERANCE = 1e-9


def main() -> int:
    print(__doc__)
    selection = json.loads(
        (MODELS_DIR / "selection.json").read_text(encoding="utf-8"))
    receipt = json.loads(
        (MODELS_DIR / "TEST_WAS_EVALUATED.json").read_text(encoding="utf-8"))
    span = selection["span"]

    dataset = load_dataset()
    long_frame = load_long()
    identity = pd.read_csv(IDENTITY_PATH, dtype={"TEAM_ID": "int64"})

    training = training_seasons(dataset, span, TEST_SEASONS[0])
    k, carryover = fit_elo_on(long_frame, identity, training)
    frame = with_fold_elo(dataset, long_frame, identity, k, carryover)

    section("REPRODUCING THE RECORDED TEST SCORES FIRST")
    computed = {}
    for target, recorded in receipt["results"].items():
        spec = TARGETS[target]
        window, family = recorded["window"], recorded["family"]
        columns = feature_columns(window)

        train = frame[frame["SEASON"].isin(training)]
        train = train[usable(train, window, target)]
        test = frame[frame["SEASON"].isin(TEST_SEASONS)]
        test = test[usable(test, window, target)]

        outer, inner = split_inner(train)
        predicted = fit_predict(family, spec["kind"], outer, test, columns,
                                spec["label"], inner)
        y = test[spec["label"]].to_numpy(dtype=float)

        model_score = score(spec["kind"], y, predicted)
        gap = abs(model_score - recorded["model"])
        print(f"  {target:<11}{model_score:.10f} against recorded "
              f"{recorded['model']:.10f}   diff {gap:.2e}")
        if gap > TOLERANCE:
            raise SystemExit(
                f"{target} does not reproduce its recorded test score, so no "
                f"interval reported here would describe the published result")

        computed[target] = {
            "model": errors(spec["kind"], y, predicted),
            "naive": errors(spec["kind"], y,
                            naive_prediction(target, window, train, test)),
        }
        if spec["kind"] == "classification":
            computed[target]["elo"] = errors(
                spec["kind"], y, test["ELO_EXPECTED"].to_numpy(dtype=float))

    section(f"THE SAME INTERVALS RE-DRAWN ACROSS {len(SEEDS)} SEEDS")
    print(f"  {'comparison':<28}{'mean':>10}{'low':>11}{'high':>11}"
          f"{'excludes 0':>12}")
    print("  " + "-" * 72)

    verdicts = {}
    for target, parts in computed.items():
        for name in ("naive", "elo"):
            if name not in parts:
                continue
            excluding = 0
            lows, highs, means = [], [], []
            for seed in SEEDS:
                mean, low, high = bootstrap_difference(
                    parts["model"], parts[name], seed=seed)
                means.append(mean)
                lows.append(low)
                highs.append(high)
                if not (low <= 0 <= high):
                    excluding += 1
            label = f"{target} model - {name}"
            print(f"  {label:<28}{np.mean(means):>10.4f}"
                  f"{np.mean(lows):>11.4f}{np.mean(highs):>11.4f}"
                  f"{excluding:>8} / {len(SEEDS)}")
            verdicts[label] = (excluding, len(SEEDS), float(np.mean(highs)),
                               float(np.max(highs)))

    section("READ")
    for label, (excluding, total, mean_high, worst_high) in verdicts.items():
        if excluding == total:
            margin = abs(mean_high)
            print(f"  {label}: {excluding}/{total} seeds exclude zero.")
            if margin < 0.01:
                print(f"    RELIABLE BUT NARROW - the nearer bound averages "
                      f"{mean_high:+.4f} and at worst\n"
                      f"    reaches {worst_high:+.4f}. So the difference is "
                      f"real and small, which is a\n"
                      f"    different claim from one that clears zero "
                      f"comfortably.")
        else:
            print(f"  {label}: only {excluding}/{total} seeds exclude zero, "
                  f"so the single-draw\n    interval was not stable and this "
                  f"is a TIE.")

    print("""
  Nothing was selected or written here. The test receipt stands, and the
  configurations are the ones selection.json already named.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
