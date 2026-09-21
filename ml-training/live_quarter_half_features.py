"""Build a model-ready Q1/first-half feature row for a game that hasn't been"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from train_quarter_half_baseline import (
    CONTEXT_FEATURES,
    FEATURE_COLUMNS,
    QUARTER_HALF_FEATURES,
)

from live_features import current_elo, rest_features, season_of, team_history

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "data-pipeline" / "preprocessing"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from build_rolling_features import WINDOWS, derive_season  # noqa: E402
from build_quarter_half_rolling import (  # noqa: E402
    ROLLING_METRICS,
    attach_opponent,
    derive_metrics,
    load_raw,
    load_universe,
    reindex_to_universe,
)

MODELS_DIR = Path(__file__).resolve().parent / "models_quarter_half"
MANIFEST_PATH = MODELS_DIR / "manifest.json"

class InsufficientQuarterHalfHistory(RuntimeError):
    """A team has too few games this season for a complete trailing window."""

    def __init__(self, team_id: int, side: str, missing: list, played: int):
        self.team_id = team_id
        self.side = side
        self.missing = missing
        self.games_played = played
        super().__init__(
            f"{side} team {team_id} has played {played} game(s) with quarter "
            f"data this season - not enough for a complete window. "
            f"{len(missing)} feature(s) would be NaN, e.g. {missing[:3]}. "
            f"The Q1/1H models are linear and cannot score a partial row; "
            f"refuse the fixture rather than imputing."
        )

def load_quarter_half_history(quiet: bool = True) -> pd.DataFrame:
    """One row per team-game: the six derived metrics, with date and season."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer) if quiet else contextlib.nullcontext():
        raw = load_raw()
        paired = attach_opponent(raw)
        metrics = derive_metrics(paired)
        universe = load_universe()
        history = reindex_to_universe(universe, metrics)

    history["SEASON"] = derive_season(history["GAME_DATE"])
    return history.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(
        drop=True
    )

def team_quarter_half_rolling(history: pd.DataFrame, team_id: int,
                              game_date: pd.Timestamp, season: int) -> tuple:
    """Trailing Q1/1H means for one team. Returns (features, games_played)."""
    rows = history[
        (history["TEAM_ID"] == team_id)
        & (history["GAME_DATE"] < pd.Timestamp(game_date))
        & (history["SEASON"] == season)
    ].sort_values(["GAME_DATE", "GAME_ID"])

    features = {}
    for window in WINDOWS:
        recent = rows.tail(window)
        for metric in ROLLING_METRICS:
            complete = len(recent) == window and bool(recent[metric].notna().all())
            features[f"ROLL{window}_{metric}"] = (
                float(recent[metric].mean()) if complete else np.nan
            )
    return features, len(rows)

def get_live_quarter_half_features(
    home_team_id: int,
    away_team_id: int,
    game_date,
    quarter_half_history_df: pd.DataFrame,
    games_final_df: pd.DataFrame,
) -> pd.DataFrame:
    """The 30-column row the six Q1/1H models expect, for one unplayed game."""
    game_date = pd.Timestamp(game_date)
    season = season_of(game_date)

    row = {}
    for side, team_id in (("HOME", home_team_id), ("AWAY", away_team_id)):
        rolling, played = team_quarter_half_rolling(
            quarter_half_history_df, team_id, game_date, season
        )

        missing = [f"{side}_{name}" for name, value in rolling.items()
                   if pd.isna(value)]
        if missing:
            raise InsufficientQuarterHalfHistory(team_id, side, missing, played)

        for name, value in rolling.items():
            row[f"{side}_{name}"] = value

        history = team_history(games_final_df, team_id, game_date)
        rest = rest_features(history, game_date)
        row[f"{side}_TEAM_ELO"] = current_elo(history, season)
        row[f"{side}_REST_DAYS"] = rest["REST_DAYS"]
        row[f"{side}_IS_BACK_TO_BACK"] = rest["IS_BACK_TO_BACK"]

    frame = pd.DataFrame([row])[FEATURE_COLUMNS]

    still_missing = [c for c in FEATURE_COLUMNS if frame[c].isna().any()]
    if still_missing:
        raise InsufficientQuarterHalfHistory(
            home_team_id, "context for", still_missing, -1
        )

    return frame

def verify_manifest_agrees() -> bool:
    """The shipped manifest's column list must match the code's."""
    import json

    if not MANIFEST_PATH.exists():
        print(f"  manifest not found at {MANIFEST_PATH} - skipped")
        return True

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    shipped = manifest["feature_columns"]
    if shipped != FEATURE_COLUMNS:
        raise RuntimeError(
            "manifest.json's feature_columns disagrees with "
            "train_quarter_half_baseline.FEATURE_COLUMNS.\n"
            f"  manifest: {len(shipped)} columns\n"
            f"  code    : {len(FEATURE_COLUMNS)} columns\n"
            f"  first difference at index "
            f"{next(i for i, (a, b) in enumerate(zip(shipped, FEATURE_COLUMNS)) if a != b)}"
        )
    print(f"  manifest and code agree on all {len(FEATURE_COLUMNS)} columns, "
          f"in order.")
    return True

def load_games_final() -> pd.DataFrame:
    """Convenience loader, matching live_features.load_games_final."""
    from live_features import load_games_final as _load

    return _load()

if __name__ == "__main__":
    print("Quarter/half live features")
    print(f"  {len(QUARTER_HALF_FEATURES)} rolling + {len(CONTEXT_FEATURES)} "
          f"context = {len(FEATURE_COLUMNS)} columns")
    verify_manifest_agrees()
    history = load_quarter_half_history()
    print(f"  history loaded: {len(history):,} team-game rows, "
          f"{history['GAME_DATE'].min().date()} to "
          f"{history['GAME_DATE'].max().date()}")
