"""NFL phase 3 §8: report, decide nothing.

Four items the spec asks for and explicitly forbids acting on: the 2026 games
so far, how often the winner and margin models disagree, the winner model's
reliability bins, and home advantage per season.

TWO DIFFERENT MODELS ARE USED HERE, AND THE DISTINCTION IS THE POINT:

  * the 2026 scores come from the PRODUCTION artifacts, which is what 2026
    would actually be served by;
  * the agreement rate and the reliability bins come from the HELD-OUT fit the
    receipt recorded, because scoring the production artifacts on the test
    seasons would grade them on their own training data - §24 measured that
    trap at 9.58 MAE against an honest 10.74, and §29 met it from the other
    direction where it made production look excellent.

The held-out fit is reproduced here and checked against the receipt at float
EQUALITY before anything is reported, which is what makes these numbers belong
to the model the test scored rather than to a lookalike.

    python ml-training/nfl/report_nfl_extras.py
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ML = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ML))

import nfl_common as C                       # noqa: E402
import build_nfl_model_dataset as B          # noqa: E402
from calibration import calibration_report, format_bins   # noqa: E402

MANIFEST_PATH = C.MODELS_DIR / "manifest.json"


def parse(name):
    family, elo_variant, feature_set = name.split("/")
    return feature_set, elo_variant, family


def heldout_predictions(span_start, receipt):
    """Refit span_start..2023 and predict the test seasons, as the test did."""
    training = list(range(span_start, min(C.TEST_SEASONS)))
    frame = C.fold_frame(training)
    out = {}
    for market in C.MARKETS:
        selected = receipt["selected"][market.key]
        feature_set, elo_variant, family = parse(selected)
        columns = C.feature_list(feature_set, elo_variant)
        train = C.rows_for(frame, training, market)
        test = C.rows_for(frame, C.TEST_SEASONS, market)
        model, _setting, _trees = C.fit_family(market, family, train, columns)
        labels, proba = C.predict(market, model, test, columns)
        metrics = C.score(market, test[market.target], labels, proba)
        recorded = receipt["results"][market.key]["model"]
        key = "log_loss" if market.classification else "mae"
        if float(metrics[key]) != float(recorded[key]):
            raise SystemExit(
                f"REFUSING to report: the reproduced {market.key} {key} "
                f"{metrics[key]!r} is not the receipt's {recorded[key]!r}. "
                f"These extras would describe a different model than the test "
                f"scored.")
        out[market.key] = {"rows": test, "labels": labels, "proba": proba,
                           "metrics": metrics}
    return out


def production_on_2026(manifest, span_start):
    """Score the shipped artifacts on the games played since the last fit."""
    import pandas as pd
    configs = {}
    for name, entry in manifest["elo"].items():
        configs[name] = B.EloParams(
            k=entry["k"], carryover=entry["carryover"],
            home_advantage=entry["home_advantage"],
            mov=entry["margin_of_victory_multiplier"])
    rows, _history, _priors = B.build_rows(C.games(), configs)
    frame = pd.DataFrame(rows)
    after = manifest["fitted_through"]
    seasons = sorted({int(s) for s in frame.season.unique() if s > after})
    if not seasons:
        return None, seasons

    out = {}
    for market in C.MARKETS:
        entry = manifest["markets"][market.key]
        subset = C.rows_for(frame, seasons, market)
        if not len(subset):
            continue
        if entry["family"] == "linear":
            import joblib
            model = joblib.load(C.MODELS_DIR / entry["artifact"])
        else:
            from xgboost import XGBClassifier, XGBRegressor
            model = (XGBClassifier() if market.classification
                     else XGBRegressor())
            model.load_model(str(C.MODELS_DIR / entry["artifact"]))
        labels, proba = C.predict(market, model, subset, entry["features"])
        train_rows = C.rows_for(frame, entry["fitted_on_seasons"], market)
        naive, _detail = C.naive_scores(market, train_rows, subset)
        out[market.key] = {
            "games": len(subset),
            "model": C.score(market, subset[market.target], labels, proba),
            "naive": naive,
        }
    return out, seasons


def agreement(heldout):
    """How often the winner model and the sign of the margin model disagree.

    The two markets drop different rows - the winner drops the 13 tied games -
    so the comparison is on the INTERSECTION by game id rather than by
    position, which is the kind of join that silently misaligns otherwise.
    """
    winner = heldout["winner"]
    margin = heldout["margin"]
    w = dict(zip(winner["rows"].game_id, winner["proba"]))
    m = dict(zip(margin["rows"].game_id, margin["labels"]))
    shared = sorted(set(w) & set(m))
    disagree = [g for g in shared if (w[g] > 0.5) != (m[g] > 0)]
    near = [g for g in shared
            if (w[g] > 0.5) != (m[g] > 0) and abs(w[g] - 0.5) < 0.05]
    return {
        "shared_games": len(shared),
        "disagreements": len(disagree),
        "rate": len(disagree) / len(shared) if shared else float("nan"),
        "within_5pp_of_a_coin_flip": len(near),
    }


def home_advantage(games):
    by_season = {}
    for game in games:
        if game["home_result"] == "T":
            continue
        by_season.setdefault(game["season"], []).append(
            1.0 if game["home_result"] == "W" else 0.0)
    return {season: (statistics.fmean(v), len(v))
            for season, v in sorted(by_season.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    if not C.RECEIPT_PATH.exists():
        print(f"REFUSING: no {C.RECEIPT_PATH.name}. Items 2 and 3 are about "
              f"the test set, and the test is opened once by "
              f"test_nfl_models.py.")
        return 1
    receipt = json.loads(C.RECEIPT_PATH.read_text(encoding="utf-8"))
    span_start = receipt["span_start"]

    print("=" * 78)
    print("NFL PHASE 3 §8: REPORTED, NOT DECIDED")
    print("=" * 78)

    heldout = heldout_predictions(span_start, receipt)
    print(f"  the held-out fit reproduces the receipt exactly, so items 2 and "
          f"3 describe")
    print(f"  the model the test scored rather than the production refit")

    # ---------------------------------------------------- 1. 2026 so far
    print()
    print("  1. THE 2026 GAMES SO FAR - for interest, used for nothing")
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        scores, seasons = production_on_2026(manifest, span_start)
        if not scores:
            print(f"     no games after {manifest['fitted_through']} in the "
                  f"data, so there is nothing to score")
        else:
            print(f"     seasons {seasons}, scored with the SHIPPED artifacts")
            for market in C.MARKETS:
                entry = scores.get(market.key)
                if not entry:
                    continue
                key = "log_loss" if market.classification else "mae"
                print(f"     {market.label:<8} {entry['games']:>4} games  "
                      f"model {entry['model'][key]:.4f}   "
                      f"naive {entry['naive'][key]:.4f}")
            print(f"     {scores[C.MARKETS[0].key]['games']} games decide "
                  f"nothing, and they were in neither selection nor test")
    else:
        print(f"     no {MANIFEST_PATH.name} yet - run finalize_nfl_models.py")

    # ------------------------------------------- 2. agreement between markets
    print()
    print("  2. AGREEMENT BETWEEN MARKETS, on the test seasons")
    stats = agreement(heldout)
    print(f"     {stats['disagreements']} of {stats['shared_games']} shared "
          f"games ({stats['rate']:.2%}) have the winner model on one side and "
          f"the")
    print(f"     sign of the margin model on the other")
    print(f"     {stats['within_5pp_of_a_coin_flip']} of those sit within 5 "
          f"points of a coin flip on the winner, where a")
    print(f"     disagreement costs the page least")

    # ------------------------------------------------------- 3. calibration
    print()
    print("  3. RELIABILITY BINS for the winner model - report only")
    winner = heldout["winner"]
    y = np.asarray(winner["rows"][C.MARKETS[0].target], dtype=float)
    p = np.asarray(winner["proba"], dtype=np.float64)
    report = calibration_report(y, p)
    print(f"     {report['n']} games, base rate {report['base_rate']:.4f}, "
          f"Brier {report['brier']:.4f}")
    for strategy in ("uniform", "quantile"):
        block = report[strategy]
        print(f"     {strategy:<9} ECE {block['ece']:.4f}   "
              f"MCE {block['mce']:.4f}")
        # §29: MCE is a MAX over bins, so the smallest and unluckiest one owns
        # it. Three of four MCE figures in the NBA study were noise from bins
        # of five to seven games, which is why the bin count behind a thin MCE
        # is named rather than left to be looked up.
        thin = block["thin_bins"]
        if thin:
            print(f"               {len(thin)} THIN bin(s) "
                  + ", ".join(f"{b['lower']:.2f}-{b['upper']:.2f} n={b['count']}"
                              for b in thin)
                  + " - a thin bin can own the MCE outright")
    decomposition = report["decomposition"]
    print(f"     Murphy: reliability {decomposition['reliability']:.6f}, "
          f"resolution {decomposition['resolution']:.6f}, "
          f"uncertainty {decomposition['uncertainty']:.6f}")
    share = decomposition["reliability"] / (decomposition["reliability"]
                                            + decomposition["resolution"])
    print(f"             reliability is {share:.1%} of "
          f"reliability+resolution")
    for line in format_bins(report["uniform"]["bins"]).splitlines():
        print(f"     {line}")
    print(f"     Calibrators were tested and REJECTED for the NBA (§29: "
          f"reliability was 2.3% of")
    print(f"     reliability+resolution, and neither Platt nor isotonic "
          f"cleared the bar). Nothing is")
    print(f"     added here.")

    # -------------------------------------------- 4. home advantage over time
    print()
    print("  4. HOME ADVANTAGE PER SEASON")
    rates = home_advantage(C.games())
    seasons = sorted(rates)
    for season in seasons:
        rate, n = rates[season]
        role = B.split_role(season)
        bar = "#" * int(round((rate - 0.40) * 100))
        print(f"     {season}  {rate:.3f}  {n:>4} games  {role:<7} {bar}")
    first = [rates[s][0] for s in seasons[:3]]
    last = [rates[s][0] for s in seasons[-3:]]
    print(f"     first three seasons {statistics.fmean(first):.3f}, last three "
          f"{statistics.fmean(last):.3f} "
          f"({(statistics.fmean(last) - statistics.fmean(first)) * 100:+.1f} "
          f"points)")
    low = min(seasons, key=lambda s: rates[s][0])
    print(f"     lowest: {low} at {rates[low][0]:.3f} over {rates[low][1]} "
          f"games")
    return 0


if __name__ == "__main__":
    sys.exit(main())
