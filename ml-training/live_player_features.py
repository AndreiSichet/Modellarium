"""Build model-ready player-prop feature rows for a team's roster, for a game"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from live_features import current_elo, rest_features, season_of, team_history

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "data-pipeline" / "preprocessing"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from build_player_dataset import (  # noqa: E402
    FEATURE_COLUMNS,
    PLAYER_FEATURE_COLUMNS,
    TEAM_CONTEXT_COLUMNS,
)
from build_player_rolling_minutes import ROLLING_SOURCES, WINDOWS  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent / "models_player_props"
MANIFEST_PATH = MODELS_DIR / "manifest.json"

ROSTER_COLUMN = "ROLL10_MIN"

HISTORY_COLUMNS = (
    ["GAME_ID", "GAME_DATE", "SEASON", "TEAM_ID", "PLAYER_ID", "PLAYER_NAME",
     "MIN_NUMERIC", ROSTER_COLUMN]
    + list(dict.fromkeys(ROLLING_SOURCES))
)

OUTPUT_ID_COLUMNS = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "ROUTE",
                     "AVAILABILITY_KNOWN", "APPEARANCES_THIS_SEASON"]

def load_player_history() -> pd.DataFrame:
    """The player-game history, narrowed to what this module reads."""
    path = (Path(__file__).resolve().parents[1] / "data-pipeline" / "data"
            / "processed" / "player_boxscores_with_rolling.csv")
    history = pd.read_csv(path, usecols=HISTORY_COLUMNS, low_memory=False)
    history["GAME_DATE"] = pd.to_datetime(history["GAME_DATE"])
    return history.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(
        drop=True
    )

def load_routing_rule() -> list:
    """The columns the manifest says decide linear-vs-xgb."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} not found - run finalize_player_models.py first. "
            f"Routing cannot be guessed."
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return manifest["routing"]["decide_on"]

def route_for(features: dict, decide_on: list) -> str:
    """THE ROUTING RULE, applied to one player's feature dict."""
    complete = all(not pd.isna(features[column]) for column in decide_on)
    return "linear" if complete else "xgb"

def current_roster(player_history_df: pd.DataFrame, team_id: int,
                   game_date: pd.Timestamp, season: int) -> pd.DataFrame:
    """Players currently on this team with real playing-time history."""
    appearances = player_history_df[
        (player_history_df["TEAM_ID"] == team_id)
        & (player_history_df["GAME_DATE"] < pd.Timestamp(game_date))
        & player_history_df["MIN_NUMERIC"].notna()
    ]
    if appearances.empty:
        return appearances

    latest = appearances.sort_values(["GAME_DATE", "GAME_ID"]).drop_duplicates(
        subset="PLAYER_ID", keep="last")
    latest = latest[latest["SEASON"] == season]
    return latest[latest[ROSTER_COLUMN].notna()].reset_index(drop=True)

def absent_player_ids(injury_report, team_id: int) -> set:
    """PLAYER_IDs the report marks Out or Doubtful for this team."""
    from injury_availability import reconcile

    if injury_report is None or injury_report.players.empty:
        return set()

    reconciled = reconcile(injury_report.players)
    absent = reconciled[
        (reconciled["TEAM_ID"] == team_id)
        & reconciled["IS_ABSENT"]
        & reconciled["PLAYER_ID"].notna()
    ]
    return {int(pid) for pid in absent["PLAYER_ID"]}

def player_rolling(player_history_df: pd.DataFrame, player_id: int,
                   season: int, game_date: pd.Timestamp) -> tuple:
    """Trailing means over the player's last N APPEARANCES this season."""
    played = player_history_df[
        (player_history_df["PLAYER_ID"] == player_id)
        & (player_history_df["SEASON"] == season)
        & (player_history_df["GAME_DATE"] < pd.Timestamp(game_date))
        & player_history_df["MIN_NUMERIC"].notna()
    ].sort_values(["GAME_DATE", "GAME_ID"])

    features = {}
    for window in WINDOWS:
        recent = played.tail(window)
        complete = len(recent) == window
        for source, suffix in ROLLING_SOURCES.items():
            features[f"ROLL{window}_{suffix}"] = (
                float(recent[source].mean()) if complete else np.nan
            )
    return features, len(played)

def get_live_player_features(
    team_id: int,
    opponent_team_id: int,
    game_date,
    is_home: bool,
    player_history_df: pd.DataFrame,
    games_final_df: pd.DataFrame,
    injury_report=None,
    decide_on: list = None,
) -> pd.DataFrame:
    """One feature row per rostered, available player, tagged with its model."""
    game_date = pd.Timestamp(game_date)
    season = season_of(game_date)
    decide_on = decide_on if decide_on is not None else load_routing_rule()

    roster = current_roster(player_history_df, team_id, game_date, season)
    availability_known = injury_report is not None
    excluded = absent_player_ids(injury_report, team_id) if availability_known else set()

    own = team_history(games_final_df, team_id, game_date)
    opponent = team_history(games_final_df, opponent_team_id, game_date)
    rest = rest_features(own, game_date)
    context = {
        "IS_HOME": int(bool(is_home)),
        "REST_DAYS": rest["REST_DAYS"],
        "IS_BACK_TO_BACK": rest["IS_BACK_TO_BACK"],
        "TEAM_ELO": current_elo(own, season),
        "OPPONENT_ELO": current_elo(opponent, season),
    }

    rows = []
    for entry in roster.itertuples():
        player_id = int(entry.PLAYER_ID)
        if player_id in excluded:
            continue

        features, appearances = player_rolling(
            player_history_df, player_id, season, game_date
        )
        features.update(context)

        rows.append({
            "PLAYER_ID": player_id,
            "PLAYER_NAME": entry.PLAYER_NAME,
            "TEAM_ID": team_id,
            "ROUTE": route_for(features, decide_on),
            "AVAILABILITY_KNOWN": availability_known,
            "APPEARANCES_THIS_SEASON": appearances,
            **features,
        })

    if not rows:
        return pd.DataFrame(columns=OUTPUT_ID_COLUMNS + FEATURE_COLUMNS)

    frame = pd.DataFrame(rows)[OUTPUT_ID_COLUMNS + FEATURE_COLUMNS]
    return frame.sort_values(ROSTER_COLUMN if ROSTER_COLUMN in frame
                             else "ROLL10_MIN", ascending=False).reset_index(drop=True)

def describe(frame: pd.DataFrame) -> str:
    """One line a caller can surface, including the honesty caveat."""
    if frame.empty:
        return "no players with usable history"

    linear = int((frame["ROUTE"] == "linear").sum())
    xgb = int((frame["ROUTE"] == "xgb").sum())
    known = bool(frame["AVAILABILITY_KNOWN"].iloc[0])
    caveat = ("" if known else
              "  AVAILABILITY UNKNOWN - no injury report was available, so "
              "nobody has been excluded. This is not a clean bill of health.")
    return (f"{len(frame)} players ({linear} via linear, {xgb} via xgb)." + caveat)

if __name__ == "__main__":
    print("Live player-prop features")
    rule = load_routing_rule()
    print(f"  routing reads {len(rule)} columns: {rule[:3]} ...")
    print(f"  feature row is {len(FEATURE_COLUMNS)} columns "
          f"({len(PLAYER_FEATURE_COLUMNS)} rolling + "
          f"{len(TEAM_CONTEXT_COLUMNS)} context)")
