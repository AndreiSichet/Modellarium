"""Write the shipped G League artifacts and their manifest.

Each selected configuration is retrained on EVERY season in the chosen span,
so the shipped artifact has no holdout and cannot be evaluated itself - the
numbers in the manifest belong to the held-out versions that selection and
test produced. That is `finalize_models.py`'s choice for the NBA, and it is
also exactly what made the retrain gate misjudge production in §24, reading
9.58 MAE on spread against an honest 10.74. The warning is in the manifest so
a future G League retrain gate re-fits the architecture rather than scoring
these weights.

THE SCALER IS BUNDLED INTO EACH ARTIFACT as a Pipeline, so serving cannot
apply the wrong scaling - the linear models are meaningless without it.

MANIFEST NOTES ARE FOR ENGINEERS. The WNBA's model-selection caveat reached
the screen because it sat in a manifest and travelled through the API; nothing
here is written as though it will be displayed, and anything user-facing is
phase 5's explicit decision.
"""

import json
import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gleague_common import (  # noqa: E402
    BASELINE_RATING, CONTEXT_FEATURES, IDENTITY_PATH, MODELS_DIR,
    ROLLING_METRICS, SPAN_CANDIDATES, TARGETS, TEST_SEASONS,
    VALIDATION_FOLDS, feature_columns, fit_elo_on, load_dataset, load_long,
    section, span_seasons, usable, with_fold_elo)
from select_gleague_models import build_model  # noqa: E402


def apply_serving_tiebreak(selection: dict) -> dict:
    """Switch any CUP window to its CARRY equivalent, and say why.

    PHASE 3 SHIPPED A TIE ON EVERY TARGET, so nothing measurable distinguishes
    the eight candidates and the choice falls to a cost that selection never
    scored: a CUP window makes a served form feature depend on live Showcase
    Cup results, which is a second live dependency for no measured gain.

    The selection record is left alone - it is phase 3's honest account of
    what validation chose. This is a SERVING decision applied on top, recorded
    as its own manifest block so the two cannot be confused.

    What this costs in honesty, stated rather than hidden: test was scored on
    the CUP variant, so the shipped window's test numbers are its tied twin's.
    The manifest says so per target.
    """
    section("THE SERVING TIE-BREAK")
    out = {}
    for target, pick in selection["selected"].items():
        window = pick["window"]
        if not window.startswith("CUP"):
            print(f"  {target:<11}{window:<9}already a CARRY window, "
                  f"nothing to switch")
            continue

        carry = window.replace("CUP", "CARRY")
        runner = pick["runner_up"]
        if runner["window"] != carry or runner["family"] != pick["family"]:
            raise SystemExit(
                f"{target}: the CARRY equivalent {carry} is not the recorded "
                f"runner-up ({runner['family']}/{runner['window']}), so its "
                f"validation score is not available without re-selecting - "
                f"which this must not do.")

        out[target] = {
            "selected": window,
            "shipped": carry,
            "reason": ("every candidate tied on validation, so serving cost "
                       "decides: a CUP window would make a served form "
                       "feature depend on live Showcase Cup results for no "
                       "measured gain"),
            "validation_mean_selected": pick["validation_mean"],
            "validation_mean_shipped": runner["validation_mean"],
            "validation_gap_pct": (
                (runner["validation_mean"] - pick["validation_mean"])
                / pick["validation_mean"] * 100),
            "test_numbers_belong_to": window,
        }
        print(f"  {target:<11}{window} -> {carry}   validation "
              f"{pick['validation_mean']:.4f} -> "
              f"{runner['validation_mean']:.4f} "
              f"({out[target]['validation_gap_pct']:+.2f}%)")

    if out:
        print(f"""
  {len(out)} of {len(selection['selected'])} target(s) switched. The cost is
  {max(abs(o['validation_gap_pct']) for o in out.values()):.2f}% on validation at worst, inside the fold spread that made
  these ties in the first place - and the gain is that serving reads no Cup
  result to build a form feature.

  REST DAYS STILL NEED THE CUP, and that is a separate matter settled in
  phase 2: REST_DAYS was computed across the Cup boundary, so a team's first
  regular-season game reads rest from its last Cup game. Serving must do the
  same or that row gets a value the model never saw for it.""")
    return out


def main() -> int:
    print(__doc__)
    selection = json.loads(
        (MODELS_DIR / "selection.json").read_text(encoding="utf-8"))
    receipt_path = MODELS_DIR / "TEST_WAS_EVALUATED.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) \
        if receipt_path.exists() else {"results": {}}

    span = selection["span"]
    dataset = load_dataset()
    long_frame = load_long()
    identity = pd.read_csv(IDENTITY_PATH, dtype={"TEAM_ID": "int64"})

    seasons = span_seasons(dataset, span)
    section("RETRAINING ON EVERY SEASON IN THE SPAN")
    print(f"  span     {span} (from {SPAN_CANDIDATES[span]})")
    print(f"  seasons  {seasons[0]} .. {seasons[-1]} ({len(seasons)})")

    # Fitted on every season in the span, which is what serving will replay
    # under. Phase 2 fitted the expansion offset and found its interval spans
    # zero on 31 franchises, so it stays at the league mean.
    k, carryover = fit_elo_on(long_frame, identity, seasons)
    print(f"  Elo      K={k}, carryover={carryover:.3f}, "
          f"expansion offset 0 (the league mean)")
    frame = with_fold_elo(dataset, long_frame, identity, k, carryover)
    frame = frame[frame["SEASON"].isin(seasons)]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "league": "gleague",
        "built": date.today().isoformat(),
        "span": span,
        "span_start": SPAN_CANDIDATES[span],
        "seasons_trained_on": seasons,
        "elo": {"k": k, "carryover": carryover,
                "baseline_rating": BASELINE_RATING,
                "expansion_offset": 0.0,
                # Always the first season in the record, whatever the span: a
                # rating entering any season must reflect the history before
                # it.
                "replay_starts": sorted(dataset["SEASON"].unique())[0],
                "fitted_on": "every season in the span, test included",
                "per_fold_values": selection["elo_per_fold"]},
        "rolling_metrics": ROLLING_METRICS,
        "context_features": CONTEXT_FEATURES,
        "requires_complete_features": True,
        "targets": {},
        "engineering_notes": {
            "no_holdout": ("These artifacts are retrained on every season in "
                           "the span, so they cannot be evaluated "
                           "themselves. The validation and test numbers "
                           "below belong to the held-out versions. A future "
                           "retrain gate must RE-FIT this architecture on "
                           "the candidate's own training rows rather than "
                           "scoring these weights - see CLAUDE.md section "
                           "24."),
            "elo_is_also_no_holdout": ("K and carryover above are fitted "
                                       "on every season in the span, test "
                                       "included, so they are a no-holdout "
                                       "quantity like the model weights. "
                                       "They differ from the per-fold values "
                                       "recorded under elo.per_fold_values - "
                                       "every validation fold fitted "
                                       "carryover 0.200 and the full span "
                                       "fits 0.333, which is more data "
                                       "finding a different optimum rather "
                                       "than a defect. A retrain gate must "
                                       "refit these too."),
            "selection_was_a_tie": ("Every window and family combination "
                                    "landed inside the fold-to-fold spread "
                                    "on all three targets, so the selected "
                                    "configuration is not measurably better "
                                    "than the ones listed in tied_with. It "
                                    "is a defensible pick, not a "
                                    "demonstrated winner."),
            "audience": ("Written for engineers. Nothing in this file is "
                         "intended for display; any user-facing text is "
                         "phase 5's explicit decision."),
        },
    }

    overrides = apply_serving_tiebreak(selection)
    manifest["serving_tiebreak"] = overrides

    for target, pick in selection["selected"].items():
        spec = TARGETS[target]
        family = pick["family"]
        window = overrides.get(target, {}).get("shipped", pick["window"])
        columns = feature_columns(window)

        rows = frame[usable(frame, window, target)]
        model = build_model(spec["kind"], family)
        model.fit(rows[columns].to_numpy(dtype=float),
                  rows[spec["label"]].to_numpy(dtype=float))

        path = MODELS_DIR / f"{target}.joblib"
        joblib.dump(model, path)

        tested = receipt["results"].get(target, {})
        manifest["targets"][target] = {
            "artifact": path.name,
            "family": family,
            "window": window,
            "metric": spec["metric"],
            "label": spec["label"],
            "features": columns,
            "rows_fitted_on": int(len(rows)),
            "validation": {
                "folds": VALIDATION_FOLDS,
                "mean": pick["validation_mean"],
                "sd": pick["validation_sd"],
                "naive_mean": pick["naive_mean"],
                "elo_mean": pick["elo_mean"],
                "runner_up": pick["runner_up"],
                "gap_pct": pick["gap_pct"],
                "fold_spread_pct": pick["fold_spread_pct"],
                "cup_vs_carry_pct": pick["cup_vs_carry_pct"],
            },
            "tied_with": pick["tied_with"],
            "test": {
                "seasons": TEST_SEASONS,
                "model": tested.get("model"),
                "naive": tested.get("naive"),
                "elo": tested.get("elo"),
                "rows": tested.get("rows"),
                "vs_naive": tested.get("vs_naive"),
                "vs_elo": tested.get("vs_elo"),
            },
        }
        print(f"  {target:<11}{family}/{window:<9}fitted on "
              f"{len(rows):,} games -> {path.name}")

    path = MODELS_DIR / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    section("WRITTEN")
    for item in sorted(MODELS_DIR.iterdir()):
        print(f"  {item.name:<28}{item.stat().st_size:>9,} bytes")

    section("THE REGISTRY IS THE MANIFEST")
    print("""  The manifest's `targets` keys ARE the expected artifact set, as the WNBA's
  are. The window, feature list and Elo parameters here are all selection
  outputs, so a literal copy in the serving code would be a second place that
  has to agree with this phase - and a reselection would need a code change
  to match.""")
    expected = set(manifest["targets"])
    on_disk = {p.stem for p in MODELS_DIR.glob("*.joblib")}
    if on_disk != expected:
        raise SystemExit(f"models_gleague/ does not match manifest.json.\n"
                         f"  missing from disk: "
                         f"{sorted(expected - on_disk) or 'none'}\n"
                         f"  present but unregistered: "
                         f"{sorted(on_disk - expected) or 'none'}")
    print(f"\n  {len(on_disk)} artifact(s) on disk match the "
          f"{len(expected)} registered target(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
