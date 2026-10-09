"""NFL phase 3: open the test seasons ONCE, behind a guard, and write a receipt.

2024 and 2025 are read here and nowhere else in this phase. The run refuses if a
receipt already exists, and names the file to delete rather than offering a flag
to bypass itself - the WNBA and G League precedent.

    python ml-training/nfl/test_nfl_models.py --i-am-ready-to-touch-test
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import nfl_common as C      # noqa: E402

EARLY_WEEKS = (1, 2, 3, 4)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.REPO,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:       # noqa: BLE001
        return "unknown"


def parse(name):
    family, elo_variant, feature_set = name.split("/")
    return feature_set, elo_variant, family


def evaluate_market(market, span_start, selected_name):
    """Fit on the whole training span, score the test seasons once."""
    feature_set, elo_variant, family = parse(selected_name)
    training = list(range(span_start, min(C.TEST_SEASONS)))
    frame = C.fold_frame(training)
    columns = C.feature_list(feature_set, elo_variant)

    train = C.rows_for(frame, training, market)
    test = C.rows_for(frame, C.TEST_SEASONS, market)

    model, setting, trees = C.fit_family(market, family, train, columns)
    labels, proba = C.predict(market, model, test, columns)
    metrics = C.score(market, test[market.target], labels, proba)
    # The bootstrap's per-row terms must reproduce this very number.
    C.assert_decomposes(market, test[market.target], labels, proba)
    model_losses = C.per_row_loss(market, test[market.target], labels, proba)

    out = {
        "selected": selected_name,
        "feature_set": feature_set,
        "elo_variant": elo_variant,
        "family": family,
        "features": columns,
        "setting": setting,
        "trees": trees,
        "train_rows": len(train),
        "test_rows": len(test),
        "model": metrics,
        "baselines": {},
        "intervals": {},
    }

    # naive
    naive_metrics, naive_detail = C.naive_scores(market, train, test)
    out["baselines"]["naive"] = {"metrics": naive_metrics,
                                "detail": naive_detail}
    if market.classification:
        rate = naive_detail["rate"]
        naive_proba = np.full(len(test), rate, dtype=np.float64)
        naive_losses = C.per_row_loss(market, test[market.target],
                                      (naive_proba > 0.5).astype(int),
                                      naive_proba)
    else:
        mean = naive_detail["mean"]
        naive_losses = C.per_row_loss(
            market, test[market.target],
            np.full(len(test), mean, dtype=np.float64))
    out["intervals"]["model_minus_naive"] = C.paired_bootstrap(
        model_losses, naive_losses)

    # Elo alone, both variants reported; the interval uses the model's own.
    for variant in C.ELO_VARIANTS:
        elo_metrics, elo_detail = C.elo_alone_scores(market, variant, train,
                                                     test)
        if elo_metrics is None:
            out["baselines"][f"elo_{variant}"] = None
            continue
        out["baselines"][f"elo_{variant}"] = {"metrics": elo_metrics,
                                              "detail": elo_detail}

    if market.key != "total":
        if market.classification:
            elo_proba = np.asarray(test[C.ELO_PROB_COLUMN[elo_variant_of(out)]],
                                   dtype=np.float64)
            elo_losses = C.per_row_loss(market, test[market.target],
                                        (elo_proba > 0.5).astype(int),
                                        elo_proba)
        else:
            from sklearn.linear_model import LinearRegression
            home, away = C.ELO_COLUMNS[elo_variant_of(out)]
            fit_x = (train[home] - train[away]).to_numpy().reshape(-1, 1)
            mapper = LinearRegression().fit(fit_x, train[market.target])
            x = (test[home] - test[away]).to_numpy().reshape(-1, 1)
            elo_losses = C.per_row_loss(market, test[market.target],
                                        mapper.predict(x))
        out["intervals"]["model_minus_elo_alone"] = C.paired_bootstrap(
            model_losses, elo_losses)
    else:
        out["intervals"]["model_minus_elo_alone"] = None

    # weeks 1-4, the slice served first
    early = test[test.WEEK.isin(EARLY_WEEKS)]
    early_labels, early_proba = C.predict(market, model, early, columns)
    out["weeks_1_4"] = {
        "games": len(early),
        "model": C.score(market, early[market.target], early_labels,
                         early_proba),
    }
    return out


def elo_variant_of(entry):
    return entry["elo_variant"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i-am-ready-to-touch-test", action="store_true")
    args = parser.parse_args()

    if not args.i_am_ready_to_touch_test:
        print("REFUSING: the test seasons are opened once. Pass "
              "--i-am-ready-to-touch-test when the selection is final.")
        return 1
    if C.RECEIPT_PATH.exists():
        print(f"REFUSING: {C.RECEIPT_PATH.name} already exists, so the test "
              f"has been opened.")
        print(f"  Delete {C.RECEIPT_PATH} deliberately if it must be reopened; "
              f"there is no flag for it.")
        return 1
    if not C.SELECTION_PATH.exists():
        print(f"REFUSING: no {C.SELECTION_PATH.name}; run the selection first.")
        return 1

    selection = json.loads(C.SELECTION_PATH.read_text(encoding="utf-8"))
    span_name = selection["chosen_span"]
    span_start = C.SPAN_CANDIDATES[span_name]

    print("=" * 78)
    print("NFL PHASE 3: THE TEST, ONCE")
    print(f"  span {span_name} -> training {span_start}..{min(C.TEST_SEASONS) - 1}")
    print(f"  test seasons {list(C.TEST_SEASONS)}")
    print("=" * 78)

    results, verdicts = {}, {}
    for market in C.MARKETS:
        selected = selection["stage_b"][market.key]["winner"]
        entry = evaluate_market(market, span_start, selected)
        results[market.key] = entry

        metric = "log_loss" if market.classification else "mae"
        label = "log loss" if market.classification else "MAE"
        print()
        print(f"  {market.label}  ({label}, {entry['test_rows']} test games)")
        print(f"    selected      {selected}")
        print(f"    model         {entry['model'][metric]:.4f}")
        print(f"    naive         {entry['baselines']['naive']['metrics'][metric]:.4f}")
        for variant in C.ELO_VARIANTS:
            base = entry["baselines"].get(f"elo_{variant}")
            if base is None:
                print(f"    elo {variant:<9} -   (Elo says nothing about the "
                      f"total; naive is its only baseline)")
            else:
                print(f"    elo {variant:<9} {base['metrics'][metric]:.4f}")
        if market.classification:
            print(f"    accuracy {entry['model']['accuracy']:.4f}   "
                  f"Brier {entry['model']['brier']:.4f}")
        else:
            print(f"    RMSE     {entry['model']['rmse']:.4f}")

        for key, interval in entry["intervals"].items():
            if interval is None:
                print(f"    {key:<24} -")
                continue
            flag = "EXCLUDES zero" if interval["excludes_zero"] else "spans zero"
            redraw = " <- NEARER BOUND CLOSE TO ZERO, re-draw" if \
                C.needs_redraw(market, interval) else ""
            print(f"    {key:<24} {interval['difference']:+.4f} "
                  f"95% CI [{interval['lo']:+.4f}, {interval['hi']:+.4f}]  "
                  f"{flag}{redraw}")

        print(f"    weeks 1-4     {entry['weeks_1_4']['model'][metric]:.4f} "
              f"over {entry['weeks_1_4']['games']} games")

        # §5.5's three outcomes
        elo_interval = entry["intervals"]["model_minus_elo_alone"]
        if elo_interval is None:
            verdict = "no Elo baseline; judged against naive only"
        elif not elo_interval["excludes_zero"]:
            verdict = "TIES Elo alone"
        elif elo_interval["difference"] < 0:
            verdict = "beats Elo alone"
        else:
            verdict = "LOSES to Elo alone - STOP AND REPORT"
        verdicts[market.key] = verdict
        print(f"    verdict       {verdict}")

    receipt = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "span": span_name,
        "span_start": span_start,
        "test_seasons": list(C.TEST_SEASONS),
        "selected": {m.key: results[m.key]["selected"] for m in C.MARKETS},
        "features": {m.key: results[m.key]["features"] for m in C.MARKETS},
        "results": results,
        "verdicts": verdicts,
        "winner_qualifier": C.WINNER_QUALIFIER,
    }
    C.RECEIPT_PATH.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=float),
        encoding="utf-8")
    print()
    print(f"wrote {C.RECEIPT_PATH.name}")
    losing = [k for k, v in verdicts.items() if v.startswith("LOSES")]
    if losing:
        print(f"  STOP: {losing} lost to Elo alone with the interval excluding "
              f"zero. Do not ship a model measurably worse than its own "
              f"baseline; the choice is the owner's.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
