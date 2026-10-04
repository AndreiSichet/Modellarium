"""Score the selected G League configurations on test. ONCE.

Requires --i-am-ready-to-touch-test and writes a receipt afterwards; a second
run refuses and names the file to delete, deliberately rather than offering a
flag to bypass it.

BOTH INTERVALS ARE REPORTED UP FRONT - model minus naive AND model minus Elo.
The WNBA phase reported only the first, then found Elo alone ahead of the
selected moneyline model on test, and needed a follow-up study to learn the
difference was noise. Computing both here means a reversal arrives with its
interval rather than as an unbounded point estimate.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gleague_common import (  # noqa: E402
    IDENTITY_PATH, MODELS_DIR, SPAN_CANDIDATES, TARGETS, TEST_SEASONS,
    all_seasons, bootstrap_difference, errors, feature_columns, fit_elo_on,
    load_dataset, load_long, score, section, training_seasons, usable,
    with_fold_elo)
from select_gleague_models import (  # noqa: E402
    fit_predict, naive_prediction, split_inner)

RECEIPT = MODELS_DIR / "TEST_WAS_EVALUATED.json"
SELECTION = MODELS_DIR / "selection.json"


def guard(ready: bool) -> None:
    if not ready:
        raise SystemExit(
            "refusing: test is scored once. Pass "
            "--i-am-ready-to-touch-test when the selection is final.")
    if RECEIPT.exists():
        raise SystemExit(
            f"refusing: test has already been evaluated.\n"
            f"  {RECEIPT}\n"
            f"Delete that file deliberately if a genuine re-evaluation is "
            f"intended. There is no flag for it.")


def home_win_rates(dataset: pd.DataFrame) -> None:
    """Per season, with the 2020-21 bubble called out.

    If home advantage is drifting, a single linear home term will misjudge
    recent seasons - which is a candidate explanation for any
    validation-to-test drop.
    """
    section("HOME-WIN RATE PER SEASON")
    rows = dataset[dataset["HOME_WIN"].notna()]
    rates = rows.groupby("SEASON")["HOME_WIN"].agg(["mean", "size"])

    for season, row in rates.iterrows():
        bar = "#" * int(round(row["mean"] * 40))
        note = ""
        if season == "2020-21":
            note = "  <- the 15-game BUBBLE, every game at one site"
        elif season in TEST_SEASONS:
            note = "  <- test"
        print(f"  {season:<9}{row['mean'] * 100:5.1f}%  "
              f"({int(row['size']):>4})  {bar}{note}")

    bubble = rates.loc["2020-21", "mean"] if "2020-21" in rates.index else None
    overall = rows["HOME_WIN"].mean()
    early = rates.loc[rates.index <= "2007-08", "mean"].mean()
    late = rates.loc[rates.index >= "2022-23", "mean"].mean()

    print(f"\n  overall {overall * 100:.1f}%   "
          f"first five seasons {early * 100:.1f}%   "
          f"last four {late * 100:.1f}%")
    if bubble is not None:
        print(f"  2020-21 bubble: {bubble * 100:.1f}%")
        print("""
  THE WNBA'S BUBBLE LANDED AT EXACTLY 50.0% - home advantage vanishing
  completely when every game is at a neutral site, which is about as clean a
  natural experiment as that data held. The G League's own bubble is reported
  above rather than assumed to repeat it; a 15-game season is a small sample
  and the two leagues ran their bubbles differently.""")
    drift = late - early
    print(f"\n  drift first-five to last-four: {drift * 100:+.1f} points - "
          f"{'declining' if drift < -0.01 else 'rising' if drift > 0.01 else 'flat'}")


def debut_franchises(identity: pd.DataFrame) -> dict:
    """Teams whose first season is a test season."""
    debuts = identity[identity["first_season"]
                      & identity["SEASON"].isin(TEST_SEASONS)]
    return debuts.groupby("SEASON")["TEAM_ID"].apply(list).to_dict()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--i-am-ready-to-touch-test", action="store_true",
                        dest="ready")
    args = parser.parse_args()
    guard(args.ready)

    print(__doc__)
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    span = selection["span"]

    dataset = load_dataset()
    long_frame = load_long()
    identity = pd.read_csv(IDENTITY_PATH, dtype={"TEAM_ID": "int64"})

    section("THE TEST EVALUATION - ONCE")
    training = training_seasons(dataset, span, TEST_SEASONS[0])
    print(f"  span      {span} (from {SPAN_CANDIDATES[span]})")
    print(f"  training  {training[0]} .. {training[-1]} "
          f"({len(training)} seasons)")
    print(f"  test      {', '.join(TEST_SEASONS)}")

    k, carryover = fit_elo_on(long_frame, identity, training)
    print(f"  Elo refit on those training seasons: K={k}, "
          f"carryover={carryover:.3f}")
    frame = with_fold_elo(dataset, long_frame, identity, k, carryover)

    home_win_rates(dataset)

    debuts = debut_franchises(identity)
    debut_ids = {t for ids in debuts.values() for t in ids}
    section("DEBUT-SEASON FRANCHISES IN THE TEST WINDOW")
    if debuts:
        for season, ids in debuts.items():
            print(f"  {season}: {len(ids)} franchise(s) {ids}")
    else:
        print("  none - no franchise debuts in either test season")
    print("""  Reported either way. Phase 2 fitted an expansion offset and found its
  interval spans zero, so no systematic effect is expected - and checking
  rules the factor out rather than assuming it away.""")

    results = {}
    for target, pick in selection["selected"].items():
        spec = TARGETS[target]
        window = pick["window"]
        family = pick["family"]
        columns = feature_columns(window)

        train = frame[frame["SEASON"].isin(training)]
        train = train[usable(train, window, target)]
        test = frame[frame["SEASON"].isin(TEST_SEASONS)]
        test = test[usable(test, window, target)]

        outer, inner = split_inner(train)
        predicted = fit_predict(family, spec["kind"], outer, test, columns,
                                spec["label"], inner)
        y = test[spec["label"]].to_numpy(dtype=float)
        naive = naive_prediction(target, window, train, test)

        model_errors = errors(spec["kind"], y, predicted)
        naive_errors = errors(spec["kind"], y, naive)

        entry = {
            "family": family, "window": window,
            "rows": int(len(test)),
            "model": score(spec["kind"], y, predicted),
            "naive": score(spec["kind"], y, naive),
            "validation_mean": pick["validation_mean"],
        }
        mean, low, high = bootstrap_difference(model_errors, naive_errors)
        entry["vs_naive"] = {"difference": mean, "low": low, "high": high}

        if spec["kind"] == "classification":
            elo = test["ELO_EXPECTED"].to_numpy(dtype=float)
            elo_errors = errors(spec["kind"], y, elo)
            entry["elo"] = score(spec["kind"], y, elo)
            mean, low, high = bootstrap_difference(model_errors, elo_errors)
            entry["vs_elo"] = {"difference": mean, "low": low, "high": high}

        # With and without debut-season franchises, on the same fitted model.
        if debut_ids:
            involves = (test["HOME_TEAM_ID"].isin(debut_ids)
                        | test["AWAY_TEAM_ID"].isin(debut_ids)).to_numpy()
            entry["excluding_debuts"] = score(
                spec["kind"], y[~involves], predicted[~involves])
            entry["debuts_only"] = (
                score(spec["kind"], y[involves], predicted[involves])
                if involves.any() else None)
            entry["debut_rows"] = int(involves.sum())

        results[target] = entry

    section("TEST RESULTS")
    print(f"  {'target':<11}{'config':<18}{'model':>9}{'naive':>9}"
          f"{'Elo':>9}{'vs val':>9}{'rows':>7}")
    print("  " + "-" * 72)
    for target, entry in results.items():
        drift = ((entry["model"] - entry["validation_mean"])
                 / entry["validation_mean"] * 100)
        print(f"  {target:<11}{entry['family'] + '/' + entry['window']:<18}"
              f"{entry['model']:>9.4f}{entry['naive']:>9.4f}"
              f"{entry.get('elo', float('nan')):>9.4f}"
              f"{drift:>+8.1f}%{entry['rows']:>7}")

    section("PAIRED BOOTSTRAP INTERVALS - BOTH COMPARISONS")
    print("  A negative difference means the model loses less than the "
          "comparison.\n  An interval spanning zero means the two are "
          "indistinguishable.\n")
    for target, entry in results.items():
        for name, key in (("naive", "vs_naive"), ("Elo alone", "vs_elo")):
            if key not in entry:
                continue
            d = entry[key]
            spans = d["low"] <= 0 <= d["high"]
            print(f"  {target:<11}model - {name:<10}"
                  f"{d['difference']:>+9.4f}  "
                  f"95% CI [{d['low']:+.4f}, {d['high']:+.4f}]  "
                  f"{'SPANS ZERO - indistinguishable' if spans else 'excludes zero'}")
    print("""
  Elo alone is a probability, so it is a comparison for the moneyline only.
  Deriving a margin from a rating difference would be inventing a baseline
  rather than using one - and §30 already measured the reverse direction on
  the NBA, where a probability derived from the margin model tied the
  dedicated classifier.""")

    if debut_ids:
        section("WITH AND WITHOUT DEBUT-SEASON FRANCHISES")
        print(f"  {'target':<11}{'all':>10}{'excl. debuts':>14}"
              f"{'debuts only':>13}{'debut rows':>12}")
        print("  " + "-" * 60)
        for target, entry in results.items():
            only = entry.get("debuts_only")
            print(f"  {target:<11}{entry['model']:>10.4f}"
                  f"{entry['excluding_debuts']:>14.4f}"
                  f"{only if only is None else f'{only:>13.4f}'}"
                  f"{entry['debut_rows']:>12}")

    section("THE EXPECTED RESULT, STATED BEFORE IT IS JUDGED")
    print("""  This should be the least predictable league of the three. Roster churn
  moves team composition mid-season more than anywhere else in this project -
  a two-way player can be recalled by an NBA parent club between games - so a
  model beating naive by less than the WNBA's did is the expected outcome
  rather than a failure. The WNBA's margins were 11.0% (moneyline), 14.6%
  (spread) and 3.8% (totals).""")
    for target, entry in results.items():
        margin = ((entry["naive"] - entry["model"])
                  / entry["naive"] * 100)
        print(f"    {target:<11}{margin:+6.1f}% better than naive")

    RECEIPT.write_text(json.dumps({
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "span": span,
        "test_seasons": TEST_SEASONS,
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {RECEIPT.name} - a second run will refuse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
