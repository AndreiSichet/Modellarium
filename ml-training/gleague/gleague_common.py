"""Shared setup for G League model selection - the split, the FIXED candidate
set, the stage order, and per-fold Elo.

EVERYTHING IN THIS MODULE IS DECLARED BEFORE ANYTHING IS SCORED. Phase 3 has a
third axis the WNBA's did not - how far back to train - and three axes chosen
at once against four folds is a large space. So the candidates are narrowed
here, the two stages are named here, and their ORDER is fixed here, so no
selection can be reordered after seeing a result.

THE DISRUPTED SEASONS STAY IN TRAINING AND ARE NEVER VALIDATION TARGETS.
2019-20 was cancelled mid-season and 2020-21 was a 15-game bubble. Scoring a
fold on either would judge every configuration on a season unlike any it will
be served on - but they are real basketball and still inform a rating and a
rolling window, so they contribute as history.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML = Path(__file__).resolve().parents[1]
PROJECT = ML.parent
GLEAGUE_PREP = PROJECT / "data-pipeline" / "gleague" / "preprocessing"
sys.path.insert(0, str(GLEAGUE_PREP))

# Imported rather than restated, so the serving replay and the selection
# cannot drift apart on the Elo formula or the new-franchise rule.
from build_gleague_elo import (  # noqa: E402
    BASELINE_RATING, CARRYOVER_GRID, K_GRID, expected_score, log_loss_on,
    new_franchises, prepare, run_elo)

PROCESSED = PROJECT / "data-pipeline" / "data" / "gleague" / "processed"
DATASET_PATH = PROCESSED / "gleague_model_dataset.csv"
REGULAR_PATH = PROCESSED / "gleague_games_final.csv"
IDENTITY_PATH = PROCESSED / "gleague_franchise_identity.csv"
MODELS_DIR = ML / "models_gleague"

# --------------------------------------------------------------- the split

VALIDATION_FOLDS = ["2018-19", "2021-22", "2022-23", "2023-24"]
TEST_SEASONS = ["2024-25", "2025-26"]

DISRUPTED_SEASONS = {
    "2019-20": "cancelled mid-season",
    "2020-21": "a 15-game bubble",
}

# -------------------------------------------------------- the candidates

# TWO SPANS, NOT THREE. The structural argument for a later start is real -
# the one-affiliate-per-team structure arrived only recently - but a third
# candidate between them would add a choice without adding a hypothesis.
SPAN_CANDIDATES = {
    "full": "2003-04",
    "recent": "2015-16",
}

# FOUR WINDOWS, NOT FIVE. ROLL5 is dropped: the WNBA found carried windows
# matched within-season accuracy while retaining far more games, and phase 2
# reproduced the retention gap here (86% against 95-97%). The question that
# matters is CARRY against CUP, and these four answer it at two lengths.
WINDOW_CANDIDATES = ["CARRY5", "CARRY10", "CUP5", "CUP10"]

MODEL_FAMILIES = ["linear", "xgboost"]

ROLLING_METRICS = ["WIN_PCT", "PTS", "PLUS_MINUS", "FG_PCT", "REB", "AST",
                   "TOV"]
CONTEXT_FEATURES = ["REST_DAYS", "IS_LONG_BREAK", "TEAM_ELO"]

# ------------------------------------------------------------- the stages

# THE ORDER IS THE POINT, and it is fixed here rather than emerging from the
# script's control flow. Stage A moves one axis with the other two held at a
# reference; stage B then searches the remaining two under the chosen span.
STAGE_ORDER = [
    "A: training span, at one fixed reference configuration",
    "B: window and model family, under the span A chose",
]

# Linear with CARRY5 - the WNBA's winner on two of three targets. Fixing it
# before stage B means the span is chosen without simultaneously searching
# windows and models.
STAGE_A_REFERENCE = {"window": "CARRY5", "family": "linear"}

TARGETS = {
    "moneyline": {"label": "HOME_WIN", "kind": "classification",
                  "metric": "log_loss"},
    "spread": {"label": "HOME_MARGIN", "kind": "regression",
               "metric": "mae"},
    "totals": {"label": "TOTAL_PTS", "kind": "regression", "metric": "mae"},
}

ADOPTION_NOTE = """A winner by a margin smaller than the fold-to-fold spread
is a tie and is called one."""


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def feature_columns(window: str) -> list:
    """The 20 columns one window candidate contributes, both sides."""
    columns = []
    for side in ("HOME", "AWAY"):
        columns += [f"{side}_{window}_{m}" for m in ROLLING_METRICS]
    for side in ("HOME", "AWAY"):
        columns += [f"{side}_{c}" for c in CONTEXT_FEATURES]
    return columns


def load_dataset() -> pd.DataFrame:
    frame = pd.read_csv(DATASET_PATH, dtype={"GAME_ID": str},
                        parse_dates=["GAME_DATE"])
    return frame.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def load_long() -> pd.DataFrame:
    frame = pd.read_csv(REGULAR_PATH, dtype={"GAME_ID": str},
                        parse_dates=["GAME_DATE"])
    return frame


def all_seasons(frame: pd.DataFrame) -> list:
    return sorted(frame["SEASON"].unique())


def span_seasons(frame: pd.DataFrame, span: str) -> list:
    """Seasons in a span, disrupted ones included - they are history."""
    start = SPAN_CANDIDATES[span]
    return [s for s in all_seasons(frame) if s >= start]


def training_seasons(frame: pd.DataFrame, span: str, before: str) -> list:
    """Everything in the span strictly before `before`."""
    return [s for s in span_seasons(frame, span) if s < before]


def folds(frame: pd.DataFrame, span: str) -> list:
    """One walk-forward fold per validation season."""
    out = []
    for season in VALIDATION_FOLDS:
        train = training_seasons(frame, span, season)
        if train:
            out.append({"validation": season, "training": train})
    return out


def fit_elo_on(long_frame: pd.DataFrame, identity: pd.DataFrame,
               training: list) -> tuple:
    """Grid-search K and carryover on `training` only.

    THE REPLAY ALWAYS STARTS AT THE FIRST SEASON IN THE RECORD, whatever span
    is being fitted. A rating entering a training season must reflect the
    history before it, so a span starting in 2015-16 still sees ratings shaped
    by 2003-2014.

    The expansion offset stays at the league mean: phase 2 fitted it and found
    its interval spans zero on 31 franchises, so refitting a quantity measured
    as indistinguishable from zero would add noise per fold rather than
    information.
    """
    games = prepare(long_frame)
    newcomers = new_franchises(identity, all_seasons(long_frame))
    target = set(training)

    best = None
    for k in K_GRID:
        for carryover in CARRYOVER_GRID:
            played = run_elo(games, k, carryover, 0.0, newcomers)
            loss = log_loss_on(played, target)
            if best is None or loss < best[2]:
                best = (k, carryover, loss)
    return best[0], best[1]


def elo_table(long_frame: pd.DataFrame, identity: pd.DataFrame,
              k: float, carryover: float) -> pd.DataFrame:
    games = prepare(long_frame)
    newcomers = new_franchises(identity, all_seasons(long_frame))
    played = run_elo(games, k, carryover, 0.0, newcomers)
    return played.set_index("GAME_ID")[
        ["HOME_TEAM_ELO", "AWAY_TEAM_ELO", "HOME_EXPECTED"]]


def with_fold_elo(dataset: pd.DataFrame, long_frame: pd.DataFrame,
                  identity: pd.DataFrame, k: float,
                  carryover: float) -> pd.DataFrame:
    """The dataset with its Elo columns replaced by this fold's refit."""
    table = elo_table(long_frame, identity, k, carryover)
    out = dataset.copy()
    out["HOME_TEAM_ELO"] = out["GAME_ID"].map(table["HOME_TEAM_ELO"])
    out["AWAY_TEAM_ELO"] = out["GAME_ID"].map(table["AWAY_TEAM_ELO"])
    out["ELO_EXPECTED"] = out["GAME_ID"].map(table["HOME_EXPECTED"])
    if out[["HOME_TEAM_ELO", "AWAY_TEAM_ELO", "ELO_EXPECTED"]].isna().any().any():
        raise SystemExit("a game with no refitted Elo rating")
    return out


def usable(frame: pd.DataFrame, window: str, target: str) -> pd.Series:
    """Rows a candidate can score: complete features and a known label."""
    columns = feature_columns(window)
    ok = frame[columns].notna().all(axis=1)
    return ok & frame[TARGETS[target]["label"]].notna()


def score(kind: str, y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if kind == "classification":
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return float(-np.mean(y_true * np.log(p)
                              + (1 - y_true) * np.log(1 - p)))
    return float(np.mean(np.abs(y_true - y_pred)))


def errors(kind: str, y_true, y_pred) -> np.ndarray:
    """Per-game loss, so a paired bootstrap can resample games."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if kind == "classification":
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    return np.abs(y_true - y_pred)


def bootstrap_difference(a_errors, b_errors, iterations=2000, seed=0):
    """Paired bootstrap on the per-game difference a - b."""
    a = np.asarray(a_errors, dtype=float)
    b = np.asarray(b_errors, dtype=float)
    if len(a) != len(b):
        raise SystemExit("paired bootstrap on unequal row counts")
    difference = a - b
    rng = np.random.default_rng(seed)
    draws = np.array([
        rng.choice(difference, size=len(difference), replace=True).mean()
        for _ in range(iterations)])
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(difference.mean()), float(low), float(high)
