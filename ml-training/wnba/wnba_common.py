"""Shared pieces for WNBA model selection: the split, the FIXED candidate set,
and per-fold Elo refitting.

THE CANDIDATE SET IS DECLARED HERE AND NOT ADDED TO AFTER RESULTS ARE SEEN.
Six window sets exist in the dataset; three are carried forward, chosen on
retention and on what question they answer, before anything was scored:

  ROLL5    within-season, the NBA's approach scaled down
  CARRY5   the same window carried across the season boundary
  CARRY10  longer memory, near-complete retention

ROLL10 is out on retention alone - phase 2 measured it losing 29% of games and
nearly half of 2020. ROLL3 is out because three games is noise. CARRY3 adds
nothing CARRY5 and CARRY10 do not already test. What these three answer is
whether carrying across seasons helps or hurts, which is the live question.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML = Path(__file__).resolve().parents[1]
PROJECT = ML.parent
WNBA_PREP = PROJECT / "data-pipeline" / "wnba" / "preprocessing"
sys.path.insert(0, str(WNBA_PREP))
from build_wnba_elo import (  # noqa: E402
    BASELINE_RATING, CARRYOVER_GRID, K_GRID, log_loss_on, prepare, run_elo,
)

PROCESSED = PROJECT / "data-pipeline" / "data" / "wnba" / "processed"
DATASET_PATH = PROCESSED / "wnba_model_dataset.csv"
LONG_PATH = PROCESSED / "wnba_games_final_features.csv"
MODELS_DIR = ML / "models_wnba"

TRAIN_SEASONS = [2015, 2016, 2017, 2018, 2019, 2020]
VALIDATION_SEASONS = [2021, 2022, 2023, 2024]   # walk-forward, one fold each
TEST_SEASONS = [2025, 2026]

EXPANSION_DEBUTS = {2025: ["Golden State"], 2026: ["Portland", "Toronto"]}

WINDOW_CANDIDATES = ["ROLL5", "CARRY5", "CARRY10"]

ROLLING_METRICS = ["WIN_PCT", "PTS", "PTS_ALLOWED", "PLUS_MINUS",
                   "FG_PCT", "REB", "AST", "TOV"]

# Per side, independent of which window set is chosen.
CONTEXT_FEATURES = ["REST_DAYS", "IS_BACK_TO_BACK", "IS_LONG_BREAK", "TEAM_ELO"]

TARGETS = {
    "moneyline": {"label": "HOME_WIN", "metric": "log_loss"},
    "spread": {"label": "HOME_MARGIN", "metric": "mae"},
    "totals": {"label": "TOTAL_PTS", "metric": "mae"},
}


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def feature_columns(window: str) -> list:
    """The feature list for one window candidate, 24 columns."""
    columns = []
    for side in ("HOME", "AWAY"):
        columns += [f"{side}_{window}_{m}" for m in ROLLING_METRICS]
    for side in ("HOME", "AWAY"):
        columns += [f"{side}_{c}" for c in CONTEXT_FEATURES]
    return columns


def load_dataset() -> pd.DataFrame:
    frame = pd.read_csv(DATASET_PATH, dtype={"GAME_ID": str})
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])
    for side in ("HOME", "AWAY"):
        for flag in ("IS_BACK_TO_BACK", "IS_LONG_BREAK"):
            frame[f"{side}_{flag}"] = frame[f"{side}_{flag}"].astype(float)
    return frame


def load_long() -> pd.DataFrame:
    frame = pd.read_csv(LONG_PATH, dtype={"GAME_ID": str})
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])
    return frame.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def fit_elo_on(long_frame: pd.DataFrame, training_seasons) -> tuple:
    """Fit (K, carryover) by minimising Elo's log loss on `training_seasons`.

    Phase 2 fitted these once, on whatever it called training. Under
    walk-forward "training" changes every fold, so a fold reusing phase 2's
    values would be using parameters informed by seasons it is about to be
    scored on. Refit per fold; the grid is small and the cost trivial.
    """
    games = prepare(long_frame)
    wanted = set(training_seasons)
    best = None
    for carryover in CARRYOVER_GRID:
        for k in K_GRID:
            _, _, expectations, _ = run_elo(games, k, carryover)
            loss = log_loss_on(expectations, wanted)
            if best is None or loss < best[0]:
                best = (loss, k, carryover)
    return best[1], best[2], best[0]


def elo_columns(long_frame: pd.DataFrame, k: float, carryover: float) -> pd.DataFrame:
    """Per-game HOME/AWAY Elo under the given parameters, indexed by GAME_ID."""
    games = prepare(long_frame)
    team_elo, _, _, _ = run_elo(games, k, carryover)
    elo = pd.Series(team_elo)
    rows = long_frame.assign(ELO=elo)
    home = rows[rows["IS_HOME"]].set_index("GAME_ID")["ELO"]
    away = rows[~rows["IS_HOME"]].set_index("GAME_ID")["ELO"]
    return pd.DataFrame({"HOME_TEAM_ELO": home, "AWAY_TEAM_ELO": away})


def with_fold_elo(dataset: pd.DataFrame, long_frame: pd.DataFrame,
                  training_seasons) -> tuple:
    """Dataset with Elo recomputed from parameters fitted on these seasons."""
    k, carryover, loss = fit_elo_on(long_frame, training_seasons)
    columns = elo_columns(long_frame, k, carryover)
    out = dataset.copy()
    out["HOME_TEAM_ELO"] = out["GAME_ID"].map(columns["HOME_TEAM_ELO"])
    out["AWAY_TEAM_ELO"] = out["GAME_ID"].map(columns["AWAY_TEAM_ELO"])
    if out[["HOME_TEAM_ELO", "AWAY_TEAM_ELO"]].isna().any().any():
        raise SystemExit("a game could not be assigned a fold Elo")
    return out, {"k": k, "carryover": carryover, "train_log_loss": loss}


def folds() -> list:
    """Walk-forward: each validation season trained on everything before it."""
    return [(VALIDATION_SEASONS[i],
             [s for s in range(TRAIN_SEASONS[0], VALIDATION_SEASONS[i])])
            for i in range(len(VALIDATION_SEASONS))]


def bootstrap_difference(a_errors, b_errors, iterations=2000, seed=0):
    """Paired bootstrap on (a - b). Both score the same rows, in order."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a_errors, float), np.asarray(b_errors, float)
    n = len(a)
    deltas = np.empty(iterations)
    for i in range(iterations):
        idx = rng.integers(0, n, n)
        deltas[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(deltas.mean()), float(lo), float(hi), bool(lo * hi <= 0)
