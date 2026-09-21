"""Prediction API for the seven basketball models."""

from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
import json
import sys

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from nba_api.stats.endpoints import scheduleleaguev2
from pydantic import BaseModel, ConfigDict, Field
from xgboost import XGBClassifier, XGBRegressor

ML_TRAINING_DIR = Path(__file__).resolve().parents[1] / "ml-training"
if str(ML_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(ML_TRAINING_DIR))

from common import FEATURE_COLUMNS  # noqa: E402
from live_features import (  # noqa: E402
    get_live_features,
    load_games_final,
    season_of,
)
from live_quarter_half_features import (  # noqa: E402
    InsufficientQuarterHalfHistory,
    get_live_quarter_half_features,
    load_quarter_half_history,
)
from live_player_features import (  # noqa: E402
    FEATURE_COLUMNS as FEATURE_COLUMNS_PLAYER,
    describe as describe_roster,
    get_live_player_features,
    load_player_history,
)

MODELS_DIR = ML_TRAINING_DIR / "models"
QH_MODELS_DIR = ML_TRAINING_DIR / "models_quarter_half"
PP_MODELS_DIR = ML_TRAINING_DIR / "models_player_props"

STALE_AFTER_DAYS = 2

MAX_DAYS_AHEAD = 1

SCHEDULE_DAYS_AHEAD_DEFAULT = 14
SCHEDULE_TIMEOUT_SECONDS = 45

REGULAR_SEASON_GAME_ID_DIGIT = "2"

GAME_STATUS_SCHEDULED = 1

SCHEDULE_CACHE_TTL_SECONDS = 6 * 60 * 60

MODEL_REGISTRY = [
    ("moneyline", "home_win_probability", True),
    ("spread", "home_margin", False),
    ("totals", "total_points", False),
    ("reb_margin", "rebound_margin", False),
    ("reb_total", "total_rebounds", False),
    ("ast_margin", "assist_margin", False),
    ("ast_total", "total_assists", False),
]

QH_MODEL_REGISTRY = [
    ("q1_spread", "q1_home_margin", False),
    ("q1_total", "q1_total_points", False),
    ("1h_spread", "half1_home_margin", False),
    ("1h_total", "half1_total_points", False),
    ("q1_winner", "q1_home_win_probability", True),
    ("1h_winner", "half1_home_win_probability", True),
]

CONDITIONAL_INTERPRETATION = "P(home leads | not tied)"

PP_TARGETS = ["PTS", "REB", "AST", "FG3M", "PRA"]
PP_ROUTES = ["linear", "xgb"]

PLAYER_PROPS_PER_TEAM = 10

class ScheduledGame(BaseModel):
    home_team_id: int
    away_team_id: int
    game_date: date

class PredictionRequest(BaseModel):
    home_team_id: int = Field(..., description="NBA team id of the home side")
    away_team_id: int = Field(..., description="NBA team id of the away side")
    game_date: date = Field(..., description="Tip-off date, YYYY-MM-DD")

class Predictions(BaseModel):
    home_win_probability: float
    home_margin: float
    total_points: float
    rebound_margin: float
    total_rebounds: float
    assist_margin: float
    total_assists: float

class PredictionResponse(BaseModel):
    home_team_id: int
    away_team_id: int
    game_date: date
    data_as_of: date
    stale: bool
    days_behind: int
    predictions: Predictions

class QuarterHalfPrediction(BaseModel):
    """One Q1/1H market."""

    market: str
    value: float
    confidence: str
    interpretation: str | None = None

class QuarterHalfResponse(BaseModel):
    home_team_id: int
    away_team_id: int
    game_date: date
    data_as_of: date
    stale: bool
    days_behind: int
    predictions: list[QuarterHalfPrediction]

class PlayerPrediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    player_id: int
    player_name: str
    model_used: str
    predictions: dict[str, float]

class TeamPlayerProps(BaseModel):
    team_id: int
    is_home: bool
    availability_known: bool
    availability_note: str | None = None
    players: list[PlayerPrediction]

class PlayerPropsResponse(BaseModel):
    home_team_id: int
    away_team_id: int
    game_date: date
    data_as_of: date
    stale: bool
    days_behind: int
    teams: list[TeamPlayerProps]

class ServiceState:
    """Everything loaded once at startup and shared across requests."""

    games_final_df: pd.DataFrame
    models: dict
    known_team_ids: set
    data_as_of: pd.Timestamp

    qh_models: dict
    qh_confidence: dict
    qh_history_df: pd.DataFrame

    pp_models: dict
    pp_routing_columns: list
    player_history_df: pd.DataFrame

state = ServiceState()

_schedule_cache: dict = {}

def load_models() -> dict:
    """Load all seven models as sklearn-API estimators."""
    on_disk = {path.stem for path in MODELS_DIR.glob("*.json")}
    expected = {key for key, _field, _clf in MODEL_REGISTRY}

    if on_disk != expected:
        raise RuntimeError(
            f"models/ does not match the registry.\n"
            f"  missing from disk: {sorted(expected - on_disk) or 'none'}\n"
            f"  present but unregistered: {sorted(on_disk - expected) or 'none'}"
        )

    models = {}
    for key, _field, classification in MODEL_REGISTRY:
        model = XGBClassifier() if classification else XGBRegressor()
        model.load_model(str(MODELS_DIR / f"{key}.json"))
        models[key] = model
    return models

def load_quarter_half_models() -> tuple:
    """The six Q1/1H Pipelines, plus each one's declared confidence."""
    on_disk = {path.stem for path in QH_MODELS_DIR.glob("*.joblib")}
    expected = {key for key, _field, _clf in QH_MODEL_REGISTRY}

    if on_disk != expected:
        raise RuntimeError(
            f"models_quarter_half/ does not match QH_MODEL_REGISTRY.\n"
            f"  missing from disk: {sorted(expected - on_disk) or 'none'}\n"
            f"  present but unregistered: {sorted(on_disk - expected) or 'none'}"
        )

    manifest_path = QH_MODELS_DIR / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"{manifest_path} is missing. Confidence labels and the feature "
            f"order live there; guessing either would be worse than failing."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    confidence = {entry["key"]: entry["confidence"] for entry in manifest["models"]}

    missing_confidence = expected - set(confidence)
    if missing_confidence:
        raise RuntimeError(
            f"manifest.json has no confidence entry for "
            f"{sorted(missing_confidence)}. Serving a prediction without its "
            f"confidence label is exactly what the field exists to prevent."
        )

    models = {key: joblib.load(QH_MODELS_DIR / f"{key}.joblib")
              for key, _field, _clf in QH_MODEL_REGISTRY}
    return models, confidence

def load_player_prop_models() -> tuple:
    """The ten player-prop artifacts and the routing rule that picks between."""
    expected = {f"{t.lower()}_{route}" for t in PP_TARGETS for route in PP_ROUTES}
    on_disk = ({path.stem for path in PP_MODELS_DIR.glob("*.joblib")}
               | {path.stem for path in PP_MODELS_DIR.glob("*.json")
                  if path.stem != "manifest"})

    if on_disk != expected:
        raise RuntimeError(
            f"models_player_props/ does not match the expected artifact set.\n"
            f"  missing from disk: {sorted(expected - on_disk) or 'none'}\n"
            f"  present but unregistered: {sorted(on_disk - expected) or 'none'}"
        )

    manifest_path = PP_MODELS_DIR / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"{manifest_path} is missing. It carries the routing rule, and a "
            f"hybrid with no router is not servable."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    routing_columns = manifest["routing"]["decide_on"]

    models = {}
    for target in PP_TARGETS:
        stem = target.lower()
        models[(target, "linear")] = joblib.load(
            PP_MODELS_DIR / f"{stem}_linear.joblib")
        booster = XGBRegressor()
        booster.load_model(str(PP_MODELS_DIR / f"{stem}_xgb.json"))
        models[(target, "xgb")] = booster
    return models, routing_columns

@asynccontextmanager
async def lifespan(_app: FastAPI):
    state.games_final_df = load_games_final()
    state.data_as_of = pd.Timestamp(state.games_final_df["GAME_DATE"].max())
    state.known_team_ids = set(state.games_final_df["TEAM_ID"].unique())
    state.models = load_models()

    state.qh_history_df = load_quarter_half_history()
    state.qh_models, state.qh_confidence = load_quarter_half_models()

    state.player_history_df = load_player_history()
    state.pp_models, state.pp_routing_columns = load_player_prop_models()

    print(
        f"Loaded {len(state.games_final_df)} team-game rows, "
        f"{len(state.known_team_ids)} teams, {len(state.models)} team models. "
        f"Data as of {state.data_as_of.date()}."
    )
    print(
        f"Quarter/half: {len(state.qh_models)} models, "
        f"{len(state.qh_history_df):,} team-game rows."
    )
    print(
        f"Player props: {len(state.pp_models)} artifacts, "
        f"{len(state.player_history_df):,} player-game rows, "
        f"routing on {len(state.pp_routing_columns)} columns."
    )
    yield

app = FastAPI(
    title="Basketball prediction service",
    description="Seven XGBoost models over engineered pre-game features.",
    version="1.0.0",
    lifespan=lifespan,
)

def freshness() -> tuple:
    """How far behind the underlying data is, as of right now."""
    days_behind = (pd.Timestamp(datetime.now().date()) - state.data_as_of).days
    return days_behind, days_behind > STALE_AFTER_DAYS

@app.get("/health")
def health():
    """Liveness plus freshness, so monitoring can alert on stale data."""
    days_behind, stale = freshness()
    return {
        "status": "ok",
        "models_loaded": {
            "team": len(state.models),
            "quarter_half": len(state.qh_models),
            "player_props": len(state.pp_models),
        },
        "data_as_of": state.data_as_of.date().isoformat(),
        "days_behind": days_behind,
        "stale": stale,
    }

def validate_matchup(request: "PredictionRequest") -> pd.Timestamp:
    """Checks every prediction endpoint must make. Returns the parsed date."""
    game_date = pd.Timestamp(request.game_date)

    if request.home_team_id == request.away_team_id:
        raise HTTPException(400, "home_team_id and away_team_id must differ.")

    unknown = [
        team_id
        for team_id in (request.home_team_id, request.away_team_id)
        if team_id not in state.known_team_ids
    ]
    if unknown:
        raise HTTPException(
            400,
            f"No history for team id(s) {unknown}. Features cannot be built for a "
            f"team with no completed games in the data.",
        )

    latest_allowed = state.data_as_of + pd.DateOffset(days=MAX_DAYS_AHEAD)
    if game_date > latest_allowed:
        raise HTTPException(
            400,
            f"game_date {request.game_date} is more than {MAX_DAYS_AHEAD} day past "
            f"the newest game in the data ({state.data_as_of.date()}). Rest-day "
            f"features would be computed against the wrong prior game. Latest "
            f"accepted date is {latest_allowed.date()}.",
        )

    return game_date

def season_string(today: date) -> str:
    """Season as ScheduleLeagueV2 wants it: "2026-27"."""
    start_year = season_of(pd.Timestamp(today))
    return f"{start_year}-{str(start_year + 1)[2:]}"

def fetch_schedule_frame(season: str) -> pd.DataFrame:
    """The season's full schedule, cached briefly."""
    cached = _schedule_cache.get(season)
    if cached is not None:
        age = (datetime.now() - cached["fetched_at"]).total_seconds()
        if age < SCHEDULE_CACHE_TTL_SECONDS:
            return cached["frame"]

    try:
        frame = scheduleleaguev2.ScheduleLeagueV2(
            season=season, league_id="00", timeout=SCHEDULE_TIMEOUT_SECONDS
        ).get_data_frames()[0]
    except Exception as error:
        raise HTTPException(
            502, f"Could not reach the NBA schedule API for season {season}: {error}"
        ) from error

    _schedule_cache[season] = {"fetched_at": datetime.now(), "frame": frame}
    return frame

@app.get("/schedule", response_model=list[ScheduledGame])
def schedule(
    days_ahead: int = Query(SCHEDULE_DAYS_AHEAD_DEFAULT, ge=1, le=365),
):
    """Upcoming regular-season fixtures, as candidates to display."""
    today = pd.Timestamp(datetime.now().date())
    frame = fetch_schedule_frame(season_string(today.date()))

    if frame.empty:
        return []

    frame = frame.assign(
        parsed_date=pd.to_datetime(frame["gameDate"], format="%m/%d/%Y %H:%M:%S"),
        padded_game_id=frame["gameId"].astype(str).str.zfill(10),
    )
    horizon = today + pd.DateOffset(days=days_ahead)

    upcoming = frame[
        (frame["parsed_date"] >= today)
        & (frame["parsed_date"] <= horizon)
        & (frame["gameStatus"] == GAME_STATUS_SCHEDULED)
        & (frame["padded_game_id"].str[2] == REGULAR_SEASON_GAME_ID_DIGIT)
    ].sort_values(["parsed_date", "padded_game_id"])

    return [
        ScheduledGame(
            home_team_id=int(row.homeTeam_teamId),
            away_team_id=int(row.awayTeam_teamId),
            game_date=row.parsed_date.date(),
        )
        for row in upcoming.itertuples()
    ]

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    game_date = validate_matchup(request)

    try:
        features = get_live_features(
            request.home_team_id,
            request.away_team_id,
            game_date,
            state.games_final_df,
        )
    except (ValueError, KeyError) as error:
        raise HTTPException(400, f"Could not build features: {error}") from error

    results = {}
    for key, field, classification in MODEL_REGISTRY:
        model = state.models[key]
        value = (
            model.predict_proba(features)[0, 1]
            if classification
            else model.predict(features)[0]
        )
        results[field] = float(value)

    days_behind, stale = freshness()
    return PredictionResponse(
        home_team_id=request.home_team_id,
        away_team_id=request.away_team_id,
        game_date=request.game_date,
        data_as_of=state.data_as_of.date(),
        stale=stale,
        days_behind=days_behind,
        predictions=Predictions(**results),
    )

@app.post("/predict/quarter-half", response_model=QuarterHalfResponse)
def predict_quarter_half(request: PredictionRequest):
    """Six Q1 / first-half markets for one fixture."""
    game_date = validate_matchup(request)

    try:
        features = get_live_quarter_half_features(
            request.home_team_id,
            request.away_team_id,
            game_date,
            state.qh_history_df,
            state.games_final_df,
        )
    except InsufficientQuarterHalfHistory as error:
        raise HTTPException(400, str(error)) from error
    except (ValueError, KeyError) as error:
        raise HTTPException(400, f"Could not build features: {error}") from error

    predictions = []
    for key, field, classification in QH_MODEL_REGISTRY:
        model = state.qh_models[key]
        value = (
            model.predict_proba(features)[0, 1]
            if classification
            else model.predict(features)[0]
        )
        predictions.append(
            QuarterHalfPrediction(
                market=field,
                value=float(value),
                confidence=state.qh_confidence[key],
                interpretation=CONDITIONAL_INTERPRETATION if classification else None,
            )
        )

    days_behind, stale = freshness()
    return QuarterHalfResponse(
        home_team_id=request.home_team_id,
        away_team_id=request.away_team_id,
        game_date=request.game_date,
        data_as_of=state.data_as_of.date(),
        stale=stale,
        days_behind=days_behind,
        predictions=predictions,
    )

def player_props_for_team(team_id: int, opponent_id: int, game_date, is_home: bool):
    """Top-N players for one side, each scored by the model its row routes to."""
    roster = get_live_player_features(
        team_id,
        opponent_id,
        game_date,
        is_home,
        state.player_history_df,
        state.games_final_df,
        injury_report=None,
        decide_on=state.pp_routing_columns,
    )

    availability_known = (
        bool(roster["AVAILABILITY_KNOWN"].iloc[0]) if not roster.empty else False
    )
    note = None if availability_known else describe_roster(roster)

    roster = roster.head(PLAYER_PROPS_PER_TEAM)

    players = []
    if not roster.empty:
        scores = {}
        for route in PP_ROUTES:
            block = roster[roster["ROUTE"] == route]
            if block.empty:
                continue
            matrix = block[FEATURE_COLUMNS_PLAYER]
            for target in PP_TARGETS:
                values = state.pp_models[(target, route)].predict(matrix)
                for player_id, value in zip(block["PLAYER_ID"], values):
                    scores.setdefault(int(player_id), {})[target] = float(value)

        for row in roster.itertuples():
            players.append(
                PlayerPrediction(
                    player_id=int(row.PLAYER_ID),
                    player_name=str(row.PLAYER_NAME),
                    model_used=row.ROUTE,
                    predictions=scores[int(row.PLAYER_ID)],
                )
            )

    return TeamPlayerProps(
        team_id=team_id,
        is_home=is_home,
        availability_known=availability_known,
        availability_note=note,
        players=players,
    )

@app.post("/predict/player-props", response_model=PlayerPropsResponse)
def predict_player_props(request: PredictionRequest):
    """Both teams' prop boards for one fixture, in one call."""
    game_date = validate_matchup(request)

    try:
        teams = [
            player_props_for_team(
                request.home_team_id, request.away_team_id, game_date, True
            ),
            player_props_for_team(
                request.away_team_id, request.home_team_id, game_date, False
            ),
        ]
    except (ValueError, KeyError) as error:
        raise HTTPException(400, f"Could not build features: {error}") from error

    days_behind, stale = freshness()
    return PlayerPropsResponse(
        home_team_id=request.home_team_id,
        away_team_id=request.away_team_id,
        game_date=request.game_date,
        data_as_of=state.data_as_of.date(),
        stale=stale,
        days_behind=days_behind,
        teams=teams,
    )
