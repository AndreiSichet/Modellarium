"""Retrain the seven production models on the complete dataset."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
from mlflow.models import infer_signature
from xgboost import XGBClassifier, XGBRegressor

from common import (
    TEST_SEASONS, TRACKING_URI, VALIDATION_SEASON, setup_mlflow,
)
from train_baseline import FEATURE_COLUMNS, REGRESSION_TARGETS, load_dataset, section
from train_moneyline_xgb import EXPERIMENT_NAME as MONEYLINE_EXPERIMENT
from train_moneyline_xgb import PARAMS as MONEYLINE_PARAMS
from train_moneyline_xgb import TARGET as MONEYLINE_TARGET
from train_regression_xgb import PARAMS as REGRESSION_PARAMS
from train_regression_xgb import experiment_name

MODELS_DIR = Path(__file__).resolve().parent / "models"
PRODUCTION_EXPERIMENT = "production"

# Read off held-out early stopping, never set by hand. Regenerated on
# 2026-10-06 when A7 corrected the team margin: the 38-feature counts were
# 131/117/133/119/49/131/132 and several moved a long way (totals 133 -> 102,
# ast_margin 131 -> 103), because changing a feature changes how soon the
# optimizer stops finding structure worth fitting. verify_tree_counts checks
# these against the MLflow runs they came from.
# REGENERATED 2026-10-24 for A3: 34 features and the GAME_ID season
# rule. The 38-feature post-A7 counts were
# 138/104/102/105/53/103/134 and every one of them moved, because
# changing the feature set and the season labels changes how soon the
# optimizer stops finding structure worth fitting. Read off the
# selection runs of 2026-10-24 12:54, which verify_tree_counts
# checks these against.
TREE_COUNTS = {
    "moneyline": 67,
    "spread": 78,
    "totals": 108,
    "reb_margin": 88,
    "reb_total": 31,
    "ast_margin": 68,
    "ast_total": 131,
}

ORIGINAL_MAX_DEPTH = 4
ORIGINAL_LEARNING_RATE = 0.05

TASKS = [(MONEYLINE_EXPERIMENT, "Moneyline", MONEYLINE_TARGET, True)] + [
    (experiment_name(label), label, target, False)
    for target, label, _stat, _combine in REGRESSION_TARGETS
]

def final_params(classification: bool, trees: int) -> dict:
    """Selection config with early stopping replaced by a fixed tree count."""
    source = MONEYLINE_PARAMS if classification else REGRESSION_PARAMS
    params = {k: v for k, v in source.items() if k != "early_stopping_rounds"}
    params["n_estimators"] = trees
    return params

def verify_tree_counts() -> None:
    """Check the frozen counts against the MLflow runs they came from."""
    section("TREE COUNT VERIFICATION")
    mlflow.set_tracking_uri(TRACKING_URI)

    checked = mismatched = 0
    for key, _label, _target, _classification in TASKS:
        try:
            runs = mlflow.search_runs(
                experiment_names=[key],
                filter_string=(
                    f"params.max_depth = '{ORIGINAL_MAX_DEPTH}' "
                    f"and params.learning_rate = '{ORIGINAL_LEARNING_RATE}'"
                ),
                order_by=["start_time DESC"],
                max_results=1,
            )
        except Exception as error:  # noqa: BLE001 - advisory check, never fatal
            print(f"  {key:<12} could not query MLflow ({type(error).__name__}) - skipped")
            continue

        if len(runs) == 0 or "metrics.best_iteration" not in runs.columns:
            print(f"  {key:<12} no original-config run recorded - skipped")
            continue

        recorded = int(runs["metrics.best_iteration"].iloc[0])
        frozen = TREE_COUNTS[key]
        checked += 1
        if recorded == frozen:
            print(f"  {key:<12} {frozen:>4} trees  OK")
        else:
            mismatched += 1
            print(f"  {key:<12} {frozen:>4} trees  MISMATCH - MLflow recorded {recorded}")

    if mismatched:
        print(
            f"\n{mismatched} of {checked} counts disagree with MLflow. The training "
            "scripts have\nmoved since these were frozen - reconcile before shipping "
            "these models."
        )
    elif checked:
        print(f"\nAll {checked} verifiable counts match their recorded runs.")

# --------------------------------------------------------------------------
# PROVENANCE
#
# TWO FILES, DELIBERATELY SEPARATE, because they answer different questions
# and one of them is read by a program.
#
#   trained_through.json  the RETRAIN MARKER. `continuous_retrain.py` reads it
#                         to learn how much data these weights have already
#                         seen, and refuses to guess when it is absent.
#                         Until 2026-10-24 finalize_models.py did not write
#                         it, so it had NEVER existed - and the weekly job
#                         skipped every week and reported green, because a
#                         missing marker made it return EXIT_OK with no
#                         candidates. Its own message said the situation
#                         "resolves itself after one run", but the skip
#                         happened before candidates were written, so it never
#                         could.
#
#   provenance.json       WHAT PRODUCED EACH ARTIFACT, for a person. Before
#                         this, the only record was the MLflow SQLite store -
#                         which is not committed, not backed up, and was the
#                         single reason the misattributed frozen numbers of
#                         2026-10-06 could be untangled at all.
#
# Merging them would mean one of the two audiences gets the wrong shape.
INPUT_TABLES = [
    "games_final.csv",
    "player_boxscores_with_rolling.csv",
    "team_advanced_rolling.csv",
    "quarter_half_raw.csv",
]

RAW_FOLDERS = ["player_boxscores", "quarter_scores", "team_advanced_stats"]

PROCESSED_DIR = (Path(__file__).resolve().parents[1] / "data-pipeline" /
                 "data" / "processed")
RAW_DIR = (Path(__file__).resolve().parents[1] / "data-pipeline" / "data" /
           "raw")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_commit() -> dict:
    """The commit these weights were produced from, and whether it was clean.

    THE DIRTY FLAG MATTERS AS MUCH AS THE HASH. Artifacts built from a working
    tree with uncommitted changes cannot be reproduced from the commit alone,
    and recording only the hash would claim they could.
    """
    root = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                              capture_output=True, text=True, timeout=60)
        # TRACKED FILES ONLY. `CLAUDE.md`, `NOTES_HISTORY.md` and every
        # `probes/` folder live in this working tree untracked by design and
        # are always present, so counting them would make `working_tree_clean`
        # permanently false and therefore meaningless.
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(root), capture_output=True, text=True, timeout=120)
        dirty = [line[3:].strip() for line in status.stdout.splitlines()
                 if line.strip()]

        # THIS RUN'S OWN OUTPUTS ARE SEPARATED OUT, because finalize_models.py
        # writes the very files it then hashes: the seven models and the two
        # provenance files are modified BY this run, so they appear dirty
        # until someone commits them. Lumping them in with a half-finished
        # edit somewhere else would hide the difference that matters - whether
        # the CODE and the INPUT TABLES were clean when the build ran.
        own = [path for path in dirty
               if path.startswith("ml-training/models/")]
        other = [path for path in dirty if path not in own]
        return {
            "commit": head.stdout.strip() or None,
            "working_tree_clean": not dirty,
            "tracked_dirty_total": len(dirty),
            "written_by_this_run": sorted(own),
            "other_tracked_dirty": sorted(other),
            "note": (
                "working_tree_clean counts TRACKED files only. "
                "written_by_this_run is this script's own output and is "
                "expected to be dirty until it is committed; "
                "other_tracked_dirty is what a reader should actually "
                "look at."
            ),
        }
    except Exception as error:                                  # noqa: BLE001
        return {"commit": None, "error": f"{type(error).__name__}: {error}"}


def raw_corpus_fingerprint(dataset) -> dict:
    """The raw corpus AT THE CUTOFF: per-folder counts and one combined hash.

    ONE COMBINED HASH RATHER THAN 39,000 INDIVIDUAL ONES. The question is "was
    the input corpus the same", and a single digest answers it in a line a
    person can compare by eye. The per-folder counts localise a mismatch
    without recomputing anything.

    AT THE CUTOFF, WHICH IS NOT THE SAME AS "WHAT IS ON DISK NOW", and the
    first version of this function got that wrong. The working copy's raw
    corpus keeps growing - the daily fetch adds this season's games - while
    these weights were fitted on games up to `data_cutoff`. Hashing the whole
    directory would have recorded a fingerprint that no rebuild of these
    models could ever reproduce, which is worse than recording nothing.

    So the corpus is restricted the way the build was:
      * a per-game file counts only if its GAME_ID is in the built dataset
      * a season CSV is hashed over its rows AT OR BEFORE the cutoff, in
        memory, rather than as it sits on disk

    That makes the digest reproducible from the full corpus plus the cutoff
    date, by anyone, later.

    THE LIST IS SORTED BEFORE HASHING, so filesystem enumeration order cannot
    change the answer - the same trap that makes a raw season CSV's md5 differ
    between two fetches of identical rows.
    """
    cutoff = pd.Timestamp(dataset["GAME_DATE"].max())
    in_scope = {str(g).zfill(10) for g in dataset["GAME_ID"]}

    entries, counts, skipped = [], {}, {}
    for folder in RAW_FOLDERS:
        directory = RAW_DIR / folder
        files = sorted(directory.glob("*.csv")) if directory.is_dir() else []
        kept = [f for f in files if f.stem.zfill(10) in in_scope]
        counts[folder] = len(kept)
        skipped[folder] = len(files) - len(kept)
        for path in kept:
            entries.append((f"{folder}/{path.name}", sha256_of(path)))

    season_files = sorted(RAW_DIR.glob("games_*.csv"))
    seasons_kept = 0
    for path in season_files:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        dates = pd.to_datetime(frame["GAME_DATE"])
        truncated = frame[dates <= cutoff]
        if truncated.empty:
            continue
        seasons_kept += 1
        digest = hashlib.sha256(
            truncated.to_csv(index=False).encode("utf-8")).hexdigest()
        entries.append((path.name, digest))
    counts["games_*.csv"] = seasons_kept
    skipped["games_*.csv"] = len(season_files) - seasons_kept

    combined = hashlib.sha256()
    for name, digest in sorted(entries):
        combined.update(name.encode("utf-8"))
        combined.update(digest.encode("ascii"))
    return {
        "scope": (f"raw files for the {len(in_scope):,} games in the dataset; "
                  f"season CSVs hashed over rows <= {cutoff.date()}"),
        "file_counts": counts,
        "files_after_cutoff_excluded": skipped,
        "files_hashed": len(entries),
        "combined_sha256": combined.hexdigest(),
    }


def selection_metrics() -> dict:
    """Each market's newest SELECTION run: its id, trees and held-out score.

    FROM THE SELECTION RUNS, NOT FROM THESE ARTIFACTS. The production models
    are trained on everything with no holdout, so they have no honest score of
    their own - scoring them on the test seasons would grade them on their own
    training data, which is the trap recorded in PROJECT_INSIGHTS 16.7.
    """
    import sqlite3

    database = Path(TRACKING_URI.replace("sqlite:///", ""))
    if not database.is_file():
        return {"error": f"no MLflow store at {database}"}

    con = sqlite3.connect(str(database))
    out = {}
    try:
        for key, _label, _target, _classification in TASKS:
            rows = list(con.execute(
                """select run_uuid from runs where name = ?
                   order by start_time desc limit 1""", (f"xgb-{key}",)))
            if not rows:
                out[key] = {"error": "no selection run found"}
                continue
            uid = rows[0][0]
            metrics = dict(con.execute(
                "select key, value from latest_metrics where run_uuid = ?",
                (uid,)))
            out[key] = {
                "mlflow_run_id": uid,
                "best_iteration": metrics.get("best_iteration"),
                "test_mae": metrics.get("test_mae"),
                "test_accuracy": metrics.get("test_accuracy"),
                "test_log_loss": metrics.get("test_log_loss"),
            }
    finally:
        con.close()
    return out


def write_trained_through(dataset) -> Path:
    """The retrain marker, in the format continuous_retrain.py reads.

    THE FORMAT IS NOT INVENTED HERE - it is `continuous_retrain`'s own
    `write_trained_through`: a `trained_through` date and a `games` count.
    Writing a different shape would leave the reader refusing to guess for a
    second reason.
    """
    body = {
        "trained_through": str(dataset["GAME_DATE"].max().date()),
        "games": int(len(dataset)),
    }
    path = MODELS_DIR / "trained_through.json"
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return path


def write_provenance(dataset, summary) -> Path:
    """What produced each artifact, for a person reading it a year later."""
    tables = {}
    for name in INPUT_TABLES:
        path = PROCESSED_DIR / name
        tables[name] = sha256_of(path) if path.is_file() else "ABSENT"

    dataset_path = PROCESSED_DIR / "model_dataset.csv"
    selection = selection_metrics()
    commit = code_commit()
    corpus = raw_corpus_fingerprint(dataset)
    cutoff = str(dataset["GAME_DATE"].max().date())

    models = {}
    for row in summary:
        key = row["key"]
        models[key] = {
            "file": row["file"],
            "sha256": sha256_of(MODELS_DIR / row["file"]),
            "trees": row["trees"],
            "parameters": row["parameters"],
            "n_features": len(FEATURE_COLUMNS),
            "split": {
                "trained_on": "every game in model_dataset.csv, no holdout",
                "selection_split": {
                    "train": f"SEASON < {VALIDATION_SEASON}",
                    "validation": VALIDATION_SEASON,
                    "test": list(TEST_SEASONS),
                },
            },
            "selection": selection.get(key, {}),
        }

    body = {
        "note": (
            "model_dataset.csv is NOT COMMITTED. It is rebuilt from the raw "
            "corpus by the build chain (build_games_table -> "
            "build_rolling_features -> build_rest_days -> build_elo_ratings "
            "-> build_player_rolling_minutes -> build_team_availability -> "
            "build_team_advanced_rolling -> build_quarter_half_raw -> "
            "build_quarter_half_rolling -> build_final_dataset). Its sha256 "
            "is recorded here BECAUSE it is not committed: that is the only "
            "way to tell later whether the table these weights were fitted "
            "on is the table a rebuild produces. Of the ten tables that "
            "chain writes, only the four under input_tables are committed."
        ),
        "written_at": datetime.now(timezone.utc).isoformat(),
        "data_cutoff": cutoff,
        "code": commit,
        "model_dataset": {
            "path": "data-pipeline/data/processed/model_dataset.csv",
            "sha256": (sha256_of(dataset_path) if dataset_path.is_file()
                       else "ABSENT"),
            "rows": int(len(dataset)),
            "committed": False,
        },
        "input_tables": tables,
        "raw_corpus": corpus,
        "feature_columns": list(FEATURE_COLUMNS),
        "models": models,
    }
    path = MODELS_DIR / "provenance.json"
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return path


def train_final(label: str, target: str, classification: bool, trees: int, df):
    params = final_params(classification, trees)
    estimator = XGBClassifier if classification else XGBRegressor

    model = estimator(**params)
    model.fit(df[FEATURE_COLUMNS], df[target], verbose=False)
    return model, params

def log_production_run(key, label, target, classification, model, params, path, sample, rows):
    with mlflow.start_run(run_name=key):
        mlflow.set_tags(
            {
                "stage": "final",
                "selection_experiment": key,
                "trained_on": "full dataset, no holdout",
                "evaluated": "no - no holdout remains by design",
            }
        )
        mlflow.log_params(params)
        mlflow.log_params(
            {
                "target": target,
                "label": label,
                "task": "classification" if classification else "regression",
                "n_features": len(FEATURE_COLUMNS),
                "training_rows": rows,
                "model_file": path.name,
            }
        )

        predictions = model.predict_proba(sample) if classification else model.predict(sample)
        mlflow.xgboost.log_model(
            model,
            name="model",
            signature=infer_signature(sample, predictions),
            input_example=sample,
        )
        mlflow.log_artifact(str(path))

def print_summary(rows, total_games):
    section(f"PRODUCTION MODELS (trained on all {total_games} games)")

    label_w = max(len(r["label"]) for r in rows)
    file_w = max(len(r["file"]) for r in rows)

    header = (
        f"{'TARGET':<{label_w}}  {'DEPTH':>5} {'LR':>5} {'TREES':>6}  "
        f"{'FILE':<{file_w}}  SAVED"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['label']:<{label_w}}  {r['max_depth']:>5} {r['learning_rate']:>5} "
            f"{r['trees']:>6}  {r['file']:<{file_w}}  {r['saved']}"
        )

    print(f"\n{len(rows)} models written to {MODELS_DIR}")
    print(f"Logged to MLflow experiment '{PRODUCTION_EXPERIMENT}' with stage=final.")
    print("No metrics logged - these models have no holdout set to score against.")

def main():
    section("DATA PREP")
    df = load_dataset()
    print(f"Loaded {len(df)} games - training on all of them, no split.")
    print("Incomplete rolling windows kept: XGBoost handles NaN natively.")

    verify_tree_counts()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    sample = df[FEATURE_COLUMNS].head(5)

    setup_mlflow(PRODUCTION_EXPERIMENT)

    section("TRAINING")
    summary = []
    for key, label, target, classification in TASKS:
        trees = TREE_COUNTS[key]
        model, params = train_final(label, target, classification, trees, df)

        path = MODELS_DIR / f"{key}.json"
        model.save_model(str(path))

        log_production_run(
            key, label, target, classification, model, params, path, sample, len(df)
        )

        print(f"  {label:<12} {trees:>4} trees -> {path.name}")
        summary.append(
            {
                "key": key,
                "parameters": {k: v for k, v in params.items()},
                "label": label,
                "max_depth": params["max_depth"],
                "learning_rate": params["learning_rate"],
                "trees": trees,
                "file": path.name,
                "saved": "OK" if path.exists() else "FAILED",
            }
        )

    print_summary(summary, len(df))

    section("PROVENANCE")
    marker = write_trained_through(df)
    provenance = write_provenance(df, summary)
    print(f"  {marker.name:<22} the retrain marker continuous_retrain.py reads")
    print(f"  {provenance.name:<22} what produced each artifact")
    print(f"  both are COMMITTED, and they answer different questions - see "
          f"the comment above INPUT_TABLES")
    for row in summary:
        digest = sha256_of(MODELS_DIR / row["file"])
        print(f"    {row['file']:<22} {digest[:16]}  {row['trees']:>4} trees")

if __name__ == "__main__":
    main()
