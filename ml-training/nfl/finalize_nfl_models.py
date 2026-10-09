"""NFL phase 3 §6: refit the selected models on the whole span through 2025.

REFUSES unless `test_receipt.json` exists AND its configuration matches what
this script is about to fit. That is what makes "the test was opened once, for
these models" checkable rather than stated: a configuration changed after the
test would have to open the test again to get a matching receipt.

The Elo fitted here has NO HOLDOUT - it is fitted on seasons that include the
test ones, exactly like the model weights - and the manifest says so, because
the G League found the same thing one level down from the weights and a future
retrain gate has to refit rather than inherit.

    python ml-training/nfl/finalize_nfl_models.py
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import nfl_common as C                       # noqa: E402
import build_nfl_model_dataset as B          # noqa: E402

MANIFEST_PATH = C.MODELS_DIR / "manifest.json"
ATTRIBUTION = {
    "source": "English Wikipedia season and schedule pages",
    "licence": "CC BY-SA 4.0",
    "share_alike": "no NFL table is committed, raw or processed - the licence "
                   "is share-alike and this repository is not licensed to "
                   "redistribute a derived database",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(name):
    family, elo_variant, feature_set = name.split("/")
    return feature_set, elo_variant, family


def production_frame(span_start, last_season):
    """Every game's features under Elo fitted on span_start..last_season.

    The fit deliberately includes the test seasons - that is what §6 asks for -
    so it goes through the one named opt-out rather than through `fit_elo`,
    which refuses test rows and must keep refusing them for every selection
    step.
    """
    seasons = set(range(span_start, last_season + 1))
    rows = [g for g in C.games() if g["season"] in seasons]
    configs, elo_detail = {}, {}
    for name, mov in (("ELO", False), ("ELO_MOV", True)):
        params, loss, edges = B.fit_elo_including_test(
            rows, mov, who=f"production_fit_{name}")
        configs[name] = params
        elo_detail[name] = {
            "k": params.k,
            "carryover": params.carryover,
            "home_advantage": params.home_advantage,
            "margin_of_victory_multiplier": params.mov,
            "fitted_on_seasons": sorted(seasons),
            "training_log_loss": loss,
            "grid_edges": edges,
            "no_holdout": "fitted on seasons that include the test ones, so "
                          "this is a no-holdout quantity like the weights; a "
                          "retrain gate must refit K, carryover and home "
                          "advantage rather than inherit them",
        }
    import pandas as pd
    built, _history, _priors = B.build_rows(C.games(), configs)
    return pd.DataFrame(built), configs, elo_detail


def fit_market(market, frame, span_start, last_season, selected):
    feature_set, elo_variant, family = parse(selected)
    columns = C.feature_list(feature_set, elo_variant)
    seasons = list(range(span_start, last_season + 1))
    train = C.rows_for(frame, seasons, market)
    model, setting, trees = C.fit_family(market, family, train, columns)
    return model, {
        "selected": selected,
        "family": family,
        "elo_variant": elo_variant,
        "feature_set": feature_set,
        "features": columns,
        "setting": setting,
        "trees": trees,
        "fitted_on_seasons": seasons,
        "rows": len(train),
    }


def save(market, model, family):
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if family == "linear":
        import joblib
        path = C.MODELS_DIR / f"{market.key}.joblib"
        # The imputer and the scaler are INSIDE the Pipeline, so the artifact
        # carries the training medians with it and serving cannot apply a
        # different fill than the fit did.
        joblib.dump(model, path, compress=0)
    else:
        path = C.MODELS_DIR / f"{market.key}.json"
        model.save_model(str(path))
    return path


def check_receipt(span_name, span_start):
    if not C.RECEIPT_PATH.exists():
        return None, (f"REFUSING: no {C.RECEIPT_PATH.name}. The production fit "
                      f"is only defensible after the test was opened once for "
                      f"THESE models; run test_nfl_models.py first.")
    receipt = json.loads(C.RECEIPT_PATH.read_text(encoding="utf-8"))
    selection = json.loads(C.SELECTION_PATH.read_text(encoding="utf-8"))

    if receipt.get("span") != span_name:
        return None, (f"REFUSING: the receipt was written for span "
                      f"{receipt.get('span')}, this fit is span {span_name}.")
    about_to_fit = {m.key: selection["stage_b"][m.key]["winner"]
                    for m in C.MARKETS}
    if receipt.get("selected") != about_to_fit:
        return None, (f"REFUSING: the receipt's configuration is "
                      f"{receipt.get('selected')} and this fit would use "
                      f"{about_to_fit}. A configuration changed after the test "
                      f"was opened; it would have to be opened again.")
    for market in C.MARKETS:
        feature_set, elo_variant, _family = parse(about_to_fit[market.key])
        if receipt["features"][market.key] != C.feature_list(feature_set,
                                                             elo_variant):
            return None, (f"REFUSING: {market.key}'s feature list has changed "
                          f"since the receipt was written.")
    return receipt, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--last-season", type=int,
                       default=max(C.TEST_SEASONS))
    args = parser.parse_args()

    if not C.SELECTION_PATH.exists():
        print(f"REFUSING: no {C.SELECTION_PATH.name}; run the selection first.")
        return 1
    selection = json.loads(C.SELECTION_PATH.read_text(encoding="utf-8"))
    span_name = selection["chosen_span"]
    span_start = C.SPAN_CANDIDATES[span_name]

    receipt, refusal = check_receipt(span_name, span_start)
    if refusal:
        print(refusal)
        return 1

    print("=" * 78)
    print("NFL PHASE 3 §6: THE PRODUCTION FIT")
    print(f"  span {span_name} -> fitting {span_start}..{args.last_season}")
    print(f"  the 2026 games played so far are EXCLUDED: a production model "
          f"here is")
    print(f"  reproducible from completed seasons only")
    print(f"  receipt {C.RECEIPT_PATH.name} written {receipt['written_at']} at "
          f"commit {receipt['git_commit'][:12]}")
    print("=" * 78)

    frame, _configs, elo_detail = production_frame(span_start,
                                                   args.last_season)
    print()
    for name, detail in elo_detail.items():
        print(f"  {name:<8} K={detail['k']} carry={detail['carryover']:.3f} "
              f"home={detail['home_advantage']} "
              f"loss={detail['training_log_loss']:.6f}"
              + (f"  GRID EDGE {detail['grid_edges']}"
                 if detail["grid_edges"] else ""))

    markets, hashes = {}, {}
    print()
    for market in C.MARKETS:
        selected = selection["stage_b"][market.key]["winner"]
        model, entry = fit_market(market, frame, span_start, args.last_season,
                                  selected)
        path = save(market, model, entry["family"])
        digest = sha256(path)
        entry["artifact"] = path.name
        entry["sha256"] = digest
        hashes[path.name] = digest
        markets[market.key] = entry
        print(f"  {market.label:<8} {selected:<34} {entry['rows']:,} rows "
              f"-> {path.name}")
        print(f"           setting {entry['setting']}"
              + (f", {entry['trees']} trees" if entry["trees"] else "")
              + f", {len(entry['features'])} features")
        print(f"           sha256  {digest}")

    manifest = {
        "phase": "nfl-3-models",
        "written_at": datetime.now(timezone.utc).isoformat(),
        "league": "nfl",
        "span": span_name,
        "span_start": span_start,
        "fitted_through": args.last_season,
        "excluded": "the 2026 games played so far - a production model is "
                    "reproducible from completed seasons only, and 64 games "
                    "decide nothing",
        "markets": markets,
        "winner_qualifier": C.WINNER_QUALIFIER,
        "elo": elo_detail,
        "tie_rule": selection["tie_rule"],
        "tied_with": {m.key: selection["stage_b"][m.key]["tied_with"]
                      for m in C.MARKETS},
        "selection_is_a_pick_not_a_win": {
            m.key: (f"{len(selection['stage_b'][m.key]['tied_with'])} of "
                    f"{len(selection['stage_b'][m.key]['candidates']) - 1} "
                    f"other candidates tie it under the tie rule, so the "
                    f"shipped configuration is defensible rather than "
                    f"demonstrated")
            for m in C.MARKETS},
        "test": {m.key: {
            "scores": receipt["results"][m.key]["model"],
            "baselines": {k: (v["metrics"] if v else None) for k, v in
                          receipt["results"][m.key]["baselines"].items()},
            "intervals": receipt["results"][m.key]["intervals"],
            "weeks_1_4": receipt["results"][m.key]["weeks_1_4"],
            "verdict": receipt["verdicts"][m.key],
        } for m in C.MARKETS},
        "test_receipt": {"written_at": receipt["written_at"],
                         "git_commit": receipt["git_commit"],
                         "test_seasons": receipt["test_seasons"]},
        "no_holdout": "these artifacts are fitted through the test seasons and "
                      "so cannot be scored themselves; every number under "
                      "`test` belongs to the held-out fit the receipt records",
        "attribution": ATTRIBUTION,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=float),
        encoding="utf-8")
    print()
    print(f"  wrote {MANIFEST_PATH.name}")
    print()
    print("  ARTIFACT HASHES (a second run of this script must reproduce them)")
    for name, digest in sorted(hashes.items()):
        print(f"    {digest}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
