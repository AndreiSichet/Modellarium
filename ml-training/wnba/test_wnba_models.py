"""Score the selected WNBA configurations on 2025-2026. ONCE.

Guarded two ways: an explicit flag must be passed, and a receipt is written
afterwards which makes a second run refuse. Debugging a script is exactly how a
test set quietly becomes a validation set, so the guard is mechanical rather
than a note in a docstring.

Run:  python test_wnba_models.py --i-am-ready-to-touch-test
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from wnba_common import (  # noqa: E402
    MODELS_DIR, TARGETS, TEST_SEASONS, TRAIN_SEASONS, VALIDATION_SEASONS,
    bootstrap_difference, feature_columns, load_dataset, load_long, section,
    with_fold_elo,
)
from select_wnba_models import expected_score, naive_prediction  # noqa: E402

RECEIPT = MODELS_DIR / "TEST_WAS_EVALUATED.json"
FIT_SEASONS = TRAIN_SEASONS + VALIDATION_SEASONS   # 2015-2024

EXPANSION_NAMES = ["Golden State", "Portland", "Toronto"]


def guard(force_second_look: bool) -> None:
    if RECEIPT.exists() and not force_second_look:
        receipt = json.loads(RECEIPT.read_text())
        print("REFUSING TO RUN.\n")
        print(f"  The test seasons were already scored at {receipt['when']}")
        print(f"  against this selection: {receipt['selection']}")
        print("\n  A second pass over the test set turns it into a validation")
        print("  set. If a genuinely new question needs asking, delete")
        print(f"  {RECEIPT.name} deliberately and record why - do not pass a")
        print("  flag to get around this.")
        raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i-am-ready-to-touch-test", action="store_true",
                        dest="ready")
    parser.add_argument("--force-second-look", action="store_true")
    args = parser.parse_args()

    if not args.ready:
        print(__doc__)
        print("Refusing: --i-am-ready-to-touch-test was not passed.")
        return 1
    guard(args.force_second_look)

    selection = json.loads((MODELS_DIR / "selection.json").read_text())["selection"]
    print(__doc__)
    section("THE SELECTION BEING TESTED (decided before this ran)")
    for target, choice in selection.items():
        print(f"  {target:<10} {choice['family']:<8} {choice['window']:<9} "
              f"validation {choice['validation_mean']:.4f} "
              f"(sd {choice['validation_sd']:.4f})")

    dataset = load_dataset()
    long_frame = load_long()

    # Elo refit on everything except the test seasons - the same rule the folds
    # used, applied to the final fit.
    fold_data, elo_info = with_fold_elo(dataset, long_frame, FIT_SEASONS)
    print(f"\n  Elo refit on {FIT_SEASONS[0]}-{FIT_SEASONS[-1]}: "
          f"K={elo_info['k']}, carryover={elo_info['carryover']:.3f}")

    section("HOME-WIN RATE, ALL SEASONS")
    rates = dataset.groupby("SEASON")["HOME_WIN"].agg(["mean", "size"])
    for season, row in rates.iterrows():
        tag = ("  <- test" if season in TEST_SEASONS else "")
        print(f"  {season}  {row['mean'] * 100:5.1f}%  "
              f"({int(row['size'])} games){tag}")

    expansion_ids = set()
    for name in EXPANSION_NAMES:
        for side in ("HOME", "AWAY"):
            hit = dataset[dataset[f"{side}_TEAM_NAME"].str.startswith(name)]
            expansion_ids.update(hit[f"{side}_TEAM_ID"].unique())
    print(f"\n  expansion franchises: {sorted(expansion_ids)} "
          f"({', '.join(EXPANSION_NAMES)})")

    results = {}
    for target, choice in selection.items():
        window = choice["window"]
        features = feature_columns(window)
        label = TARGETS[target]["label"]
        classification = target == "moneyline"

        usable = fold_data.dropna(subset=features)
        fit = usable[usable["SEASON"].isin(FIT_SEASONS)]
        test = usable[usable["SEASON"].isin(TEST_SEASONS)]

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000) if classification
            else LinearRegression())
        model.fit(fit[features], fit[label])
        predicted = (model.predict_proba(test[features])[:, 1] if classification
                     else model.predict(test[features]))
        baseline = naive_prediction(target, window, test, fit)

        truth = test[label].to_numpy()
        if classification:
            eps = 1e-15
            p, b = np.clip(predicted, eps, 1 - eps), np.clip(baseline, eps, 1 - eps)
            model_err = -(truth * np.log(p) + (1 - truth) * np.log(1 - p))
            base_err = -(truth * np.log(b) + (1 - truth) * np.log(1 - b))
            elo_p = np.clip(expected_score(test["HOME_TEAM_ELO"].to_numpy(),
                                           test["AWAY_TEAM_ELO"].to_numpy()),
                            eps, 1 - eps)
            elo_err = -(truth * np.log(elo_p) + (1 - truth) * np.log(1 - elo_p))
        else:
            model_err = np.abs(truth - predicted)
            base_err = np.abs(truth - baseline)
            elo_err = None

        is_expansion = (test["HOME_TEAM_ID"].isin(expansion_ids)
                        | test["AWAY_TEAM_ID"].isin(expansion_ids)).to_numpy()

        results[target] = {
            "window": window, "family": choice["family"],
            "n": len(test), "n_expansion": int(is_expansion.sum()),
            "model": float(model_err.mean()),
            "naive": float(base_err.mean()),
            "elo": None if elo_err is None else float(elo_err.mean()),
            "model_excl": float(model_err[~is_expansion].mean()),
            "naive_excl": float(base_err[~is_expansion].mean()),
            "model_only_exp": float(model_err[is_expansion].mean()),
            "errors": model_err, "base_errors": base_err,
            "mask": is_expansion,
        }

    section("TEST RESULTS - 2025-2026, SCORED ONCE")
    for target, r in results.items():
        metric = TARGETS[target]["metric"]
        mean, lo, hi, spans = bootstrap_difference(r["errors"], r["base_errors"])
        print(f"\n  {target.upper()}  ({metric}, {r['family']}/{r['window']}, "
              f"{r['n']} games)")
        print(f"    selected model        {r['model']:.4f}")
        print(f"    naive baseline        {r['naive']:.4f}")
        if r["elo"] is not None:
            print(f"    Elo alone             {r['elo']:.4f}")
        print(f"    model - naive         {mean:+.4f}   "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]")
        if spans:
            print("    -> THE INTERVAL SPANS ZERO. On ~600 test games the")
            print("       selected model does not separate from its baseline")
            print("       with confidence. That width is the finding.")
        else:
            print(f"    -> excludes zero: a real improvement of "
                  f"{abs(mean) / r['naive'] * 100:.1f}%")

    section("WITH AND WITHOUT EXPANSION-TEAM GAMES")
    print("""All three expansion franchises debut inside the test window, and phase 2
measured the league-mean starting Elo as optimistic for two of them (-129 and
-227 through their first season). Their games carry error no model here can
fix, so both numbers are shown rather than one.
""")
    print(f"  {'TARGET':<10}{'ALL':>10}{'EXCL EXP':>11}{'EXP ONLY':>11}"
          f"{'N EXP':>8}{'SHARE':>8}")
    print("  " + "-" * 58)
    for target, r in results.items():
        share = r["n_expansion"] / r["n"] * 100
        print(f"  {target:<10}{r['model']:>10.4f}{r['model_excl']:>11.4f}"
              f"{r['model_only_exp']:>11.4f}{r['n_expansion']:>8}{share:>7.0f}%")

    payload = {
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection": {t: f"{c['family']}/{c['window']}"
                      for t, c in selection.items()},
        "elo": elo_info,
        "results": {t: {k: v for k, v in r.items()
                        if k not in ("errors", "base_errors", "mask")}
                    for t, r in results.items()},
    }
    RECEIPT.write_text(json.dumps(payload, indent=2))
    (MODELS_DIR / "test_results.json").write_text(json.dumps(payload, indent=2))

    section("RECEIPT WRITTEN")
    print(f"  {RECEIPT.name} - a second run will now refuse.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
