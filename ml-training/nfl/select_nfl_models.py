"""NFL phase 3 selection: Stage A then Stage B, on validation folds only.

Nothing here reads 2024 or 2025. `nfl_common.fold_elo` goes through phase 2's
`fit_elo`, which calls `refuse_test_rows`, so a test season reaching any fit
raises rather than being fitted on.

Writes `ml-training/models_nfl/selection.json`: every Stage A and Stage B
result, so the record is the measurements rather than the conclusion.

    python ml-training/nfl/select_nfl_models.py
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import nfl_common as C      # noqa: E402


def fold_scores(market, span_start, validation_season, feature_set,
                elo_variant, family, weeks=None):
    """One fold: fit on the span's training seasons, score the validation one."""
    training = [s for s in range(span_start, validation_season)]
    frame = C.fold_frame(training)
    columns = C.feature_list(feature_set, elo_variant)
    train = C.rows_for(frame, training, market)
    validation = C.rows_for(frame, [validation_season], market)
    if weeks is not None:
        validation = validation[validation.WEEK.isin(weeks)]

    model, setting, trees = C.fit_family(market, family, train, columns)
    labels, proba = C.predict(market, model, validation, columns)
    metrics = C.score(market, validation[market.target], labels, proba)
    return {
        "primary": C.primary(market, metrics),
        "metrics": metrics,
        "setting": setting,
        "trees": trees,
        "train_rows": len(train),
        "validation_rows": len(validation),
    }


def baseline_scores(market, span_start, validation_season):
    """Naive and both Elo variants, on the same rows the candidates score."""
    training = [s for s in range(span_start, validation_season)]
    frame = C.fold_frame(training)
    train = C.rows_for(frame, training, market)
    validation = C.rows_for(frame, [validation_season], market)

    out = {}
    metrics, detail = C.naive_scores(market, train, validation)
    out["naive"] = {"primary": C.primary(market, metrics),
                    "metrics": metrics, "detail": detail}
    for variant in C.ELO_VARIANTS:
        metrics, detail = C.elo_alone_scores(market, variant, train,
                                             validation)
        if metrics is None:
            out[f"elo_{variant}"] = None      # the total has no Elo baseline
        else:
            out[f"elo_{variant}"] = {"primary": C.primary(market, metrics),
                                     "metrics": metrics, "detail": detail}
    return out


# ------------------------------------------------------------------ Stage A

def stage_a(validation_seasons):
    print("=" * 78)
    print("STAGE A - the training span, at one fixed reference")
    print(f"  reference: {C.STAGE_A_REFERENCE}")
    print(f"  folds    : {list(validation_seasons)} "
          f"(the seasons EVERY span can serve)")
    print("=" * 78)

    reference = C.STAGE_A_REFERENCE
    results = {}
    for market in C.MARKETS:
        per_span = {}
        for name, start in C.SPAN_CANDIDATES.items():
            folds = []
            for season in validation_seasons:
                folds.append(fold_scores(
                    market, start, season, reference["features"],
                    reference["elo"], reference["family"]))
            per_span[name] = folds
        results[market.key] = per_span

        print()
        print(f"  {market.label} ({'log loss' if market.classification else 'MAE'})")
        print(f"    {'span':<7}{'mean':>9}{'sd':>9}{'train rows':>12}  per fold")
        scored = {}
        for name, folds in per_span.items():
            values = [f["primary"] for f in folds]
            scored[name] = (values,
                            (len(C.feature_list(reference["features"],
                                                reference["elo"])),
                             reference["family"], reference["features"]))
            print(f"    {name:<7}{statistics.fmean(values):>9.4f}"
                  f"{statistics.stdev(values):>9.4f}"
                  f"{folds[0]['train_rows']:>12}  "
                  + " ".join(f"{v:.4f}" for v in values))
        winner, tied = C.rank_with_ties(scored)
        print(f"    -> {winner}" + (f", tied with {tied}" if tied
                                    else " (nothing ties it)"))
        results[market.key] = {"folds": per_span, "winner": winner,
                               "tied_with": tied}

    # One span for all three markets.
    votes = [results[m.key]["winner"] for m in C.MARKETS]
    tally = {name: votes.count(name) for name in C.SPAN_CANDIDATES}
    best = max(tally, key=lambda n: (tally[n], -C.SPAN_CANDIDATES[n]))
    print()
    print(f"  per-market winners: {votes}")
    print(f"  chosen span       : {best} "
          f"({tally[best]} of {len(C.MARKETS)} markets)")
    if tally[best] < len(C.MARKETS):
        print(f"    the markets disagree, so this is the most-wins span with "
              f"more history as the tie-break")
    return best, results


# ------------------------------------------------------------------ Stage B

def stage_b(span_name, validation_seasons):
    start = C.SPAN_CANDIDATES[span_name]
    print()
    print("=" * 78)
    print(f"STAGE B - the model, at span {span_name}")
    print(f"  folds: {list(validation_seasons)}")
    print("=" * 78)

    out = {}
    for market in C.MARKETS:
        metric = "log loss" if market.classification else "MAE"
        print()
        print(f"  {market.label} ({metric})")

        baselines = {season: baseline_scores(market, start, season)
                     for season in validation_seasons}
        print(f"    {'baseline':<34}{'mean':>9}{'sd':>9}")
        baseline_means = {}
        for key in ("naive", "elo_plain", "elo_mov"):
            values = [baselines[s][key]["primary"] for s in validation_seasons
                      if baselines[s][key] is not None]
            if not values:
                print(f"    {key:<34}{'-':>9}  (no Elo baseline for the total)")
                continue
            baseline_means[key] = statistics.fmean(values)
            print(f"    {key:<34}{statistics.fmean(values):>9.4f}"
                  f"{statistics.stdev(values):>9.4f}")

        scored, detail = {}, {}
        for feature_set, elo_variant, family in C.candidates():
            name = f"{family}/{elo_variant}/{feature_set}"
            folds = [fold_scores(market, start, season, feature_set,
                                 elo_variant, family)
                     for season in validation_seasons]
            values = [f["primary"] for f in folds]
            columns = C.feature_list(feature_set, elo_variant)
            scored[name] = (values, (len(columns), family, feature_set))
            detail[name] = folds

        print()
        print(f"    {'candidate':<40}{'mean':>9}{'sd':>9}{'trees':>18}")
        for name in sorted(scored, key=lambda n: statistics.fmean(scored[n][0])):
            values = scored[name][0]
            trees = [f["trees"] for f in detail[name]]
            tree_text = ("-" if trees[0] is None
                         else "/".join(str(t) for t in trees))
            print(f"    {name:<40}{statistics.fmean(values):>9.4f}"
                  f"{statistics.stdev(values):>9.4f}{tree_text:>18}")

        winner, tied = C.rank_with_ties(scored)
        print(f"    -> {winner}")
        print(f"       ties with {len(tied)} of {len(scored) - 1} others"
              + (f": {tied}" if tied else ""))
        for key, mean in baseline_means.items():
            gap = statistics.fmean(scored[winner][0]) - mean
            print(f"       vs {key:<12} {gap:+.4f} "
                  f"({'better' if gap < 0 else 'worse'})")

        # weeks 1-4, for the selected candidate only
        feature_set, elo_variant, family = _parse(winner)
        early = [fold_scores(market, start, season, feature_set, elo_variant,
                             family, weeks=(1, 2, 3, 4))
                 for season in validation_seasons]
        early_values = [f["primary"] for f in early]
        print(f"       weeks 1-4 only: mean {statistics.fmean(early_values):.4f}"
              f" over {sum(f['validation_rows'] for f in early)} games")

        out[market.key] = {
            "winner": winner,
            "tied_with": tied,
            "baselines": {k: baseline_means.get(k) for k in
                          ("naive", "elo_plain", "elo_mov")},
            "candidates": {name: {"per_fold": scored[name][0],
                                  "mean": statistics.fmean(scored[name][0]),
                                  "sd": statistics.stdev(scored[name][0]),
                                  "folds": detail[name]}
                           for name in scored},
            "weeks_1_4": {"per_fold": early_values,
                          "mean": statistics.fmean(early_values)},
        }
    return out


def _parse(name):
    family, elo_variant, feature_set = name.split("/")
    return feature_set, elo_variant, family


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-only", action="store_true")
    args = parser.parse_args()

    stage_a_folds = C.common_folds()
    print("NFL PHASE 3: SELECTION")
    print(f"  validation seasons declared : {list(C.VALIDATION_SEASONS)}")
    print(f"  usable by every span        : {list(stage_a_folds)}")
    for name, start in C.SPAN_CANDIDATES.items():
        usable = [v for _t, v in C.folds_for(start)]
        print(f"    span {name:<5} {len(usable)} fold(s) {usable}")
    print("  SPAN 2018 CANNOT VALIDATE ON 2018 - it would train on nothing, so")
    print("  the spec's 'the same six folds for all three' is unsatisfiable and")
    print("  Stage A uses the five every span can serve. Stage B then uses all")
    print("  folds the CHOSEN span can serve.")

    span_name, stage_a_results = stage_a(stage_a_folds)
    if args.stage_a_only:
        return 0

    stage_b_folds = tuple(v for _t, v in
                          C.folds_for(C.SPAN_CANDIDATES[span_name]))
    stage_b_results = stage_b(span_name, stage_b_folds)

    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    C.SELECTION_PATH.write_text(json.dumps({
        "validation_seasons_declared": list(C.VALIDATION_SEASONS),
        "stage_a_folds": list(stage_a_folds),
        "stage_a_reference": C.STAGE_A_REFERENCE,
        "stage_a": stage_a_results,
        "chosen_span": span_name,
        "stage_b_folds": list(stage_b_folds),
        "stage_b": stage_b_results,
        "tie_rule": "a candidate beats another only if its mean validation "
                    "score is better by more than the fold-to-fold standard "
                    "deviation of the paired difference; ties break on fewer "
                    "features, then linear over trees, then lower serving cost",
    }, indent=2, sort_keys=True, default=float), encoding="utf-8")
    print()
    print(f"wrote {C.SELECTION_PATH.relative_to(C.REPO)}")

    print()
    print("SELECTED, for the test step to read:")
    for market in C.MARKETS:
        entry = stage_b_results[market.key]
        print(f"  {market.key:<8} {entry['winner']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
