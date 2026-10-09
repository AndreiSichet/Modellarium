"""NFL phase 3 §5.3: re-draw a borderline interval across ten seeds.

Runs only when the test step reported a nearer bound inside the trigger -
0.0005 on log loss, 0.02 on MAE. §40 established the rule for the NBA and the
G League then had to act on it at a bound of -0.0004: an interval that clears
zero by four ten-thousandths is not making the same claim as one that clears it
comfortably, and one draw cannot tell them apart.

THE ORDER MATTERS, and it is the G League's: reproduce the recorded test scores
to 0.0 FIRST, then re-draw. A re-draw on numbers that do not reproduce would be
describing a different model than the receipt's.

This script SELECTS NOTHING and WRITES NOTHING. The receipt is left exactly as
the test step wrote it.

    python ml-training/nfl/redraw_nfl_intervals.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import nfl_common as C      # noqa: E402


def parse(name):
    family, elo_variant, feature_set = name.split("/")
    return feature_set, elo_variant, family


def reproduce(market, span_start, selected):
    """Refit the held-out model and return its per-row losses and baselines."""
    training = list(range(span_start, min(C.TEST_SEASONS)))
    frame = C.fold_frame(training)
    feature_set, elo_variant, family = parse(selected)
    columns = C.feature_list(feature_set, elo_variant)
    train = C.rows_for(frame, training, market)
    test = C.rows_for(frame, C.TEST_SEASONS, market)

    model, _setting, _trees = C.fit_family(market, family, train, columns)
    labels, proba = C.predict(market, model, test, columns)
    metrics = C.score(market, test[market.target], labels, proba)
    model_losses = C.per_row_loss(market, test[market.target], labels, proba)

    naive_metrics, naive_detail = C.naive_scores(market, train, test)
    if market.classification:
        naive_proba = np.full(len(test), naive_detail["rate"], dtype=np.float64)
        naive_losses = C.per_row_loss(market, test[market.target],
                                      (naive_proba > 0.5).astype(int),
                                      naive_proba)
    else:
        naive_losses = C.per_row_loss(
            market, test[market.target],
            np.full(len(test), naive_detail["mean"], dtype=np.float64))

    elo_losses = None
    if market.key != "total":
        if market.classification:
            elo_proba = np.asarray(test[C.ELO_PROB_COLUMN[elo_variant]],
                                   dtype=np.float64)
            elo_losses = C.per_row_loss(market, test[market.target],
                                        (elo_proba > 0.5).astype(int),
                                        elo_proba)
        else:
            from sklearn.linear_model import LinearRegression
            home, away = C.ELO_COLUMNS[elo_variant]
            fit_x = (train[home] - train[away]).to_numpy().reshape(-1, 1)
            mapper = LinearRegression().fit(fit_x, train[market.target])
            x = (test[home] - test[away]).to_numpy().reshape(-1, 1)
            elo_losses = C.per_row_loss(market, test[market.target],
                                        mapper.predict(x))
    return metrics, {"naive": naive_losses, "elo_alone": elo_losses}, \
        model_losses, naive_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                       help="re-draw every interval, not only the triggered "
                            "ones")
    args = parser.parse_args()

    if not C.RECEIPT_PATH.exists():
        print(f"REFUSING: no {C.RECEIPT_PATH.name}; the test has not been "
              f"opened, so there is no interval to re-draw.")
        return 1
    receipt = json.loads(C.RECEIPT_PATH.read_text(encoding="utf-8"))
    span_start = receipt["span_start"]

    triggered = []
    for market in C.MARKETS:
        for key, interval in receipt["results"][market.key]["intervals"].items():
            if interval and (args.force or C.needs_redraw(market, interval)):
                triggered.append((market, key, interval))

    print("=" * 78)
    print("NFL PHASE 3 §5.3: SEED RE-DRAW")
    print("=" * 78)
    if not triggered:
        print("  No interval's nearer bound is inside the trigger "
              f"({C.NEARER_BOUND_TRIGGER}).")
        print("  Nothing to re-draw; the receipt's single draw stands. "
              "(--force re-draws anyway.)")
        return 0
    print(f"  {len(triggered)} interval(s) inside the trigger "
          f"{C.NEARER_BOUND_TRIGGER}")

    before = C.RECEIPT_PATH.read_bytes()
    for market in {m for m, _k, _i in triggered}:
        selected = receipt["selected"][market.key]
        metrics, baselines, model_losses, _naive = reproduce(
            market, span_start, selected)
        key = "log_loss" if market.classification else "mae"
        recorded = receipt["results"][market.key]["model"][key]
        print()
        print(f"  {market.label} ({selected})")
        if float(metrics[key]) != float(recorded):
            print(f"    REFUSING: reproduced {metrics[key]!r} against the "
                  f"receipt's {recorded!r}. A re-draw on numbers that do not "
                  f"reproduce would describe a different model.")
            return 1
        print(f"    reproduces the receipt's {key} at "
              f"{abs(float(metrics[key]) - float(recorded)):.1e} - re-drawing")

        for mkt, interval_key, interval in triggered:
            if mkt.key != market.key:
                continue
            baseline_name = ("naive" if interval_key.endswith("naive")
                             else "elo_alone")
            baseline_losses = baselines[baseline_name]
            excluding, bounds = 0, []
            for seed in C.REDRAW_SEEDS:
                drawn = C.paired_bootstrap(model_losses, baseline_losses,
                                           seed=seed)
                excluding += drawn["excludes_zero"]
                bounds.append(min(abs(drawn["lo"]), abs(drawn["hi"])))
            print(f"    {interval_key}")
            print(f"      one draw : {interval['difference']:+.4f} "
                  f"[{interval['lo']:+.4f}, {interval['hi']:+.4f}]")
            print(f"      {excluding} of {len(C.REDRAW_SEEDS)} seeds exclude "
                  f"zero; nearer bound mean {np.mean(bounds):.4f}, worst "
                  f"{min(bounds):.4f}")
            if excluding == len(C.REDRAW_SEEDS):
                print(f"      -> real, and very small. A different claim from "
                      f"one that clears zero comfortably.")
            elif excluding == 0:
                print(f"      -> no seed excludes zero: read it as a tie.")
            else:
                print(f"      -> SEED-DEPENDENT. Not a reliable difference.")

    if C.RECEIPT_PATH.read_bytes() != before:
        print("\nREFUSING: the receipt changed during the re-draw.")
        return 1
    print()
    print(f"  {C.RECEIPT_PATH.name} untouched; nothing selected, nothing "
          f"written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
