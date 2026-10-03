"""Retrain the selected WNBA configurations on every season and write artifacts.

Ships what VALIDATION selected, not what the test numbers would prefer. The
test seasons were scored once, after selection was final, and re-selecting on
them would convert the test set into a second validation set - which is the
whole thing the walk-forward design and the test-once guard exist to prevent.
Where test disagrees with validation, that is recorded in the manifest as a
finding rather than acted on.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from wnba_common import (  # noqa: E402
    MODELS_DIR, TARGETS, TEST_SEASONS, TRAIN_SEASONS, VALIDATION_SEASONS,
    feature_columns, load_dataset, load_long, section, with_fold_elo,
)

ALL_SEASONS = TRAIN_SEASONS + VALIDATION_SEASONS + TEST_SEASONS


def main() -> int:
    print(__doc__)
    selection = json.loads((MODELS_DIR / "selection.json").read_text())["selection"]
    test_results = json.loads((MODELS_DIR / "test_results.json").read_text())

    dataset = load_dataset()
    long_frame = load_long()

    # Elo fitted on every season, because the shipped artifact is meant to
    # serve games after all of them. Recorded, since it differs from the
    # parameters any reported number was produced under.
    fold_data, elo_info = with_fold_elo(dataset, long_frame, ALL_SEASONS)
    section("ELO FOR THE SHIPPED ARTIFACTS")
    print(f"  fitted on {ALL_SEASONS[0]}-{ALL_SEASONS[-1]}: "
          f"K={elo_info['k']}, carryover={elo_info['carryover']:.3f}")
    print("  NOTE: the validation and test numbers were produced under")
    print("  per-fold and 2015-2024 fits respectively, not this one.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "league": "WNBA",
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seasons_trained_on": [ALL_SEASONS[0], ALL_SEASONS[-1]],
        "elo": elo_info,
        "no_holdout_warning": (
            "These artifacts are retrained on EVERY season, so they have no "
            "holdout and cannot themselves be evaluated - the validation and "
            "test numbers below belong to the held-out versions, not to these "
            "files. This is the same choice finalize_models.py makes for the "
            "NBA, and it is also exactly what made the NBA retrain gate "
            "misjudge production (CLAUDE.md s24): scoring a no-holdout model "
            "on a window inside its own training data read 9.58 MAE against "
            "an honest 10.74. Any future WNBA retrain gate must re-fit the "
            "architecture on the candidate's own training rows rather than "
            "score these weights."
        ),
        "targets": {},
    }

    section("ARTIFACTS")
    for target, choice in selection.items():
        window = choice["window"]
        features = feature_columns(window)
        label = TARGETS[target]["label"]
        classification = target == "moneyline"

        usable = fold_data.dropna(subset=features)
        print(f"\n  {target}: {choice['family']}/{window}, "
              f"{len(usable):,} of {len(fold_data):,} games usable "
              f"({len(usable) / len(fold_data) * 100:.1f}%)")

        # Scaler and estimator bundled, so serving cannot apply the wrong
        # scaling - the same reason the quarter/half models ship as pipelines.
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000) if classification
            else LinearRegression())
        model.fit(usable[features], usable[label])

        path = MODELS_DIR / f"{target}.joblib"
        joblib.dump(model, path)
        print(f"    wrote {path.name}")

        test = test_results["results"][target]
        entry = {
            "family": choice["family"],
            "window": window,
            "features": features,
            "label": label,
            "metric": TARGETS[target]["metric"],
            "requires_complete_features": True,
            "rows_trained_on": int(len(usable)),
            "validation": {
                "mean": choice["validation_mean"],
                "sd": choice["validation_sd"],
                "folds": VALIDATION_SEASONS,
                "runner_up": choice["runner_up"],
            },
            "test": {
                "seasons": TEST_SEASONS,
                "n": test["n"],
                "model": test["model"],
                "naive": test["naive"],
                "elo_alone": test["elo"],
                "excluding_expansion": test["model_excl"],
                "expansion_only": test["model_only_exp"],
            },
        }

        # Where test disagrees with validation, say so in the artifact itself.
        if target == "moneyline" and test["elo"] is not None and \
                test["elo"] < test["model"]:
            entry["caveat"] = (
                f"Elo alone scored {test['elo']:.4f} on test against this "
                f"model's {test['model']:.4f}, reversing the validation "
                "ordering where this model led Elo by 1.4%. NOT acted on: "
                "re-selecting on test would make the test set a validation "
                "set. Treat it as the first thing to re-examine if the "
                "moneyline market is revisited, ideally with more seasons."
            )
            print(f"    caveat recorded: Elo alone beat this model on test")

        manifest["targets"][target] = entry

    (MODELS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    section("MANIFEST")
    print(f"  wrote manifest.json")
    for target, entry in manifest["targets"].items():
        print(f"    {target:<10} {entry['family']}/{entry['window']}  "
              f"{len(entry['features'])} features, "
              f"{entry['rows_trained_on']:,} rows")
        if "caveat" in entry:
            print(f"               ^ carries a caveat")

    section("TRACKING")
    print("  models_wnba/ is left UNCOMMITTED, consistent with phase 1:")
    print("  nothing consumes these until serving exists in phase 4. When they")
    print("  are added to any COPY list, run `git check-ignore -v` in the same")
    print("  change - CLAUDE.md s4's standing rule.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
