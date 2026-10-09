"""Prediction API for the seven basketball models."""

from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
import json
import sys
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from nba_api.stats.endpoints import scheduleleaguev2
from pydantic import BaseModel, ConfigDict, Field
from xgboost import XGBClassifier, XGBRegressor

ML_TRAINING_DIR = Path(__file__).resolve().parents[1] / "ml-training"
_PIPELINE_DIR = Path(__file__).resolve().parents[1] / "data-pipeline"
for _directory in (ML_TRAINING_DIR, ML_TRAINING_DIR / "wnba",
                   ML_TRAINING_DIR / "gleague", ML_TRAINING_DIR / "nfl",
                   _PIPELINE_DIR / "nfl" / "preprocessing"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

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
from served_data import (  # noqa: E402
    require_data_root,
    snapshot_metadata,
    table_hashes,
)
from live_gleague_features import (  # noqa: E402
    NotScoreable as GleagueNotScoreable,
    get_live_features as get_live_gleague_features,
    load_state as load_gleague_state,
)
from live_nfl_features import (  # noqa: E402
    NotScoreable as NflNotScoreable,
    SOURCE_ATTRIBUTION as NFL_SOURCE,
    SyntheticDataRefused,
    get_live_features as get_live_nfl_features,
    kickoff_for as nfl_kickoff_for,
    load_state as load_nfl_state,
    predictable_fixtures as nfl_predictable_fixtures,
    upcoming as nfl_upcoming,
)
from live_wnba_features import (  # noqa: E402
    NotScoreable as WnbaNotScoreable,
    get_live_features as get_live_wnba_features,
    load_games as load_wnba_games,
    load_manifest as load_wnba_manifest,
)

MODELS_DIR = ML_TRAINING_DIR / "models"
QH_MODELS_DIR = ML_TRAINING_DIR / "models_quarter_half"
PP_MODELS_DIR = ML_TRAINING_DIR / "models_player_props"
WNBA_MODELS_DIR = ML_TRAINING_DIR / "models_wnba"
GLEAGUE_MODELS_DIR = ML_TRAINING_DIR / "models_gleague"
NFL_MODELS_DIR = ML_TRAINING_DIR / "models_nfl"

NEWLINE = chr(10)

STALE_AFTER_DAYS = 2

MAX_DAYS_AHEAD = 1

SCHEDULE_DAYS_AHEAD_DEFAULT = 14
SCHEDULE_TIMEOUT_SECONDS = 45

REGULAR_SEASON_GAME_ID_DIGIT = "2"

NBA_LEAGUE_ID = "00"
WNBA_LEAGUE_ID = "10"
GLEAGUE_LEAGUE_ID = "20"

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

class WnbaMarket(BaseModel):
    """One WNBA market, with the provenance a thin model needs carried.

    THE MANIFEST'S `caveat` IS DELIBERATELY NOT HERE. It records that bare Elo
    scored 0.6046 on the test seasons against the moneyline model's 0.6130 -
    which sounds like a weakness a client should see, and was served as one
    for a day. Putting a paired bootstrap on that difference gives
    +0.008301 with a 95% interval of [-0.005932, +0.022443], spanning zero on
    all ten seeds (ml-training/wnba/moneyline_vs_elo.py). So it is not a
    reliable difference, and shipping it over the wire invited every client to
    present a null result as a finding.

    It is NOT the same case as q1_winner's confidence label, which marks a
    market that is measurably weak - 0.5796 against a 0.5184 naive. The note
    stays in models_wnba/manifest.json, where a model-selection record written
    for engineers belongs.
    """

    value: float
    metric: str
    window: str

class WnbaResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    home_team_id: int
    away_team_id: int
    game_date: date
    data_as_of: date
    stale: bool
    days_behind: int
    season: int
    markets: dict[str, WnbaMarket]

class GleagueMarket(BaseModel):
    """One G League market, with the provenance a thin model needs carried.

    NO ENGINEERING NOTE IS CARRIED, and the manifest has one that invites it:
    it records that every candidate tied on validation, so the shipped
    configuration is "defensible, not demonstrated". That is a true and
    useful statement for whoever retrains this, and it is NOT a property of
    any one prediction - which is the test CLAUDE.md section 8 sets for what a
    response may carry. The WNBA phase shipped exactly this kind of note over
    the wire and had to withdraw it from three layers.

    `window` is carried because it varies per market by selection and a
    client comparing two markets should be able to see it. That is a fact
    about this response, not a caveat about the project.
    """

    value: float
    metric: str
    window: str


class GleagueResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    home_team_id: int
    away_team_id: int
    game_date: date
    data_as_of: date
    stale: bool
    days_behind: int
    season: str
    markets: dict[str, GleagueMarket]


class NflMarket(BaseModel):
    """One NFL market, with the provenance a thin model needs carried.

    `note` carries the manifest's per-market verdict - "TIES Elo alone" for
    the winner and margin markets. That is a borderline call against
    CLAUDE.md section 8's rule, so the reasoning is written down: the WNBA
    shipped a moneyline caveat over the wire, measured it, found the interval
    spanned zero and withdrew it from three layers. This is not that. A TIE
    against Elo alone is a measured, reported result in the receipt, and it
    describes WHAT THIS NUMBER IS - a model whose discrimination is not
    distinguishable from a one-feature formula's. A client showing a win
    probability without it is showing more confidence than was earned.

    `imputed` is a property of THIS response rather than of the project: it
    names the features the artifact's own imputer supplied. A season opener
    has no REST_DAYS because an offseason is not rest, and a client should be
    able to see that the row was completed rather than observed.
    """

    value: float
    metric: str
    model_used: str
    note: str | None = None
    imputed: list[str] = Field(default_factory=list)


class NflFixture(BaseModel):
    game_id: str
    season: int
    week: int
    game_date: date
    home_team_id: int
    away_team_id: int
    home: str
    away: str
    # ISO 8601 UTC, or null where the source writes TBD. Converted from the
    # zone the article writes, which is the home team's - and Arizona does not
    # observe daylight saving while its own article still labels the column
    # Mountain Time Zone, so the home franchise decides the zone rather than
    # the label alone.
    kickoff_utc: str | None
    kickoff_zone: str | None
    flex: bool
    neutral_site: bool
    # Under the DEPENDENCY rule, not a date rule: true once both teams'
    # previous games are in history.
    predictable: bool
    venue: str


class NflResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    home_team_id: int
    away_team_id: int
    game_date: date
    data_as_of: date
    stale: bool
    days_behind: int
    season: int
    week: int
    markets: dict[str, NflMarket]
    home_win_interpretation: str
    # THE LICENCE TRAVELS WITH THE PREDICTION. The NFL history is derived from
    # English Wikipedia and is CC BY-SA 4.0; phase 5 renders this, and it is on
    # the wire rather than only in a README so a client cannot show the number
    # without being able to show where it came from.
    source: str


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

    wnba_models: dict
    wnba_manifest: dict
    wnba_known_team_ids: set
    wnba_data_as_of: pd.Timestamp

    data_root: Path
    snapshot: dict

    gleague_state: dict
    gleague_models: dict
    gleague_manifest: dict
    gleague_known_team_ids: set
    gleague_data_as_of: pd.Timestamp

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

def load_wnba_models() -> tuple:
    """The WNBA Pipelines and their manifest, checked against each other.

    THE REGISTRY HERE IS THE MANIFEST, not a list in this file. The other
    three families hardcode their expected artifact set because their targets
    are fixed by the markets they price; the WNBA's window, feature list and
    Elo parameters are all selection outputs, so a literal copy of the target
    names would be a fourth place that has to agree with phase 3. Checking the
    directory against the manifest instead means the two cannot disagree
    silently - and the check is still strict in both directions.
    """
    manifest_path = WNBA_MODELS_DIR / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"{manifest_path} is missing. It carries the window choice, the "
            f"feature order and the Elo parameters, and a WNBA model cannot "
            f"be served without them."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected = set(manifest["targets"])
    on_disk = {path.stem for path in WNBA_MODELS_DIR.glob("*.joblib")}
    if on_disk != expected:
        raise RuntimeError(
            f"models_wnba/ does not match manifest.json.\n"
            f"  missing from disk: {sorted(expected - on_disk) or 'none'}\n"
            f"  present but unregistered: {sorted(on_disk - expected) or 'none'}"
        )

    models = {}
    for target in sorted(expected):
        models[target] = joblib.load(WNBA_MODELS_DIR / f"{target}.joblib")
    return models, manifest

def load_gleague_models() -> tuple:
    """The G League Pipelines and their manifest, checked against each other.

    THE REGISTRY IS THE MANIFEST, as it is for the WNBA and for the same
    reason: the window, feature order and Elo parameters are selection
    outputs, so a literal list here would be a fifth place that has to agree
    with phase 3.

    THIS CHECK NOW REFUSES THE WHOLE SERVICE FOR THREE LEAGUES. A broken G
    League artifact takes NBA and WNBA serving down with it at boot. That is
    deliberate - partial loading that serves some families silently is worse,
    because a client cannot tell a missing family from a league with no
    fixtures - and it is acceptable only because it fails at BUILD rather
    than on a live day. The cold start in the regression gate is what keeps
    that true.
    """
    manifest_path = GLEAGUE_MODELS_DIR / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"{manifest_path} is missing. It carries the window choice, the "
            f"feature order and the Elo parameters, and a G League model "
            f"cannot be served without them."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected = set(manifest["targets"])
    on_disk = {path.stem for path in GLEAGUE_MODELS_DIR.glob("*.joblib")}
    if on_disk != expected:
        raise RuntimeError(
            f"models_gleague/ does not match manifest.json.\n"
            f"  missing from disk: {sorted(expected - on_disk) or 'none'}\n"
            f"  present but unregistered: {sorted(on_disk - expected) or 'none'}"
        )

    models = {}
    for target in sorted(expected):
        entry = manifest["targets"][target]
        models[target] = joblib.load(GLEAGUE_MODELS_DIR / entry["artifact"])
    return models, manifest


def load_nfl_models() -> tuple:
    """The three NFL artifacts and their manifest, checked against each other.

    THE REGISTRY IS THE MANIFEST, as it is for the WNBA and the G League and
    for the same reason: the per-market feature lists, the families and the
    production Elo parameters are all selection outputs, so a literal list
    here would be a place that has to agree with phase 3 and could stop
    agreeing silently.

    THE FAMILIES ARE MIXED HERE, which is new. Two markets ship a linear
    Pipeline (.joblib) and one ships XGBoost (.json), so the check cannot glob
    a single extension the way the G League's does - it asks the manifest what
    each artifact is called and requires exactly that set on disk, in both
    directions.
    """
    manifest_path = NFL_MODELS_DIR / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"{manifest_path} is missing. It carries the per-market feature "
            f"lists and the production Elo parameters, and an NFL model "
            f"cannot be served without them."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected = {entry["artifact"] for entry in manifest["markets"].values()}
    on_disk = {path.name for path in NFL_MODELS_DIR.iterdir()
               if path.suffix in (".joblib", ".json")
               and path.name not in ("manifest.json", "selection.json")}
    if on_disk != expected:
        raise RuntimeError(
            "models_nfl/ does not match manifest.json." + NEWLINE
            + f"  missing from disk: {sorted(expected - on_disk) or 'none'}"
            + NEWLINE
            + f"  present but unregistered: "
              f"{sorted(on_disk - expected) or 'none'}"
        )

    models = {}
    for market, entry in sorted(manifest["markets"].items()):
        path = NFL_MODELS_DIR / entry["artifact"]
        if entry["family"] == "linear":
            models[market] = joblib.load(path)
        else:
            model = (XGBClassifier() if market == "winner" else XGBRegressor())
            model.load_model(str(path))
            models[market] = model
    return models, manifest


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # RESOLVED BEFORE ANY TABLE IS READ, so a missing volume is one clear
    # refusal rather than six FileNotFoundErrors in whichever order the
    # loaders happen to run. There is deliberately no fallback to a copy
    # inside the image - see served_data for why.
    state.data_root = require_data_root()
    state.snapshot = snapshot_metadata(state.data_root)

    # HASHED AT STARTUP, FROM THE FILES THIS PROCESS IS ABOUT TO READ. The
    # snapshot id is a timestamp, so it says the identity changed and nothing
    # about whether the contents are what they should be - a snapshot mutated in
    # place keeps its name. Measured: all six tables hash in 0.09s, so this is
    # not worth deferring or caching further.
    hash_started = time.monotonic()
    state.table_hashes = table_hashes(state.data_root)
    state.table_hash_seconds = round(time.monotonic() - hash_started, 3)

    state.games_final_df = load_games_final()
    state.data_as_of = pd.Timestamp(state.games_final_df["GAME_DATE"].max())
    state.known_team_ids = set(state.games_final_df["TEAM_ID"].unique())
    state.models = load_models()

    state.qh_history_df = load_quarter_half_history()
    state.qh_models, state.qh_confidence = load_quarter_half_models()

    state.player_history_df = load_player_history()
    state.pp_models, state.pp_routing_columns = load_player_prop_models()

    # The WNBA's own cutoff and own team ids. Not merged into the NBA's:
    # the two leagues' data ends on different dates, so one data_as_of would
    # be wrong for whichever league it did not come from, and MAX_DAYS_AHEAD
    # is computed against it.
    wnba_games = load_wnba_games()
    state.wnba_data_as_of = pd.Timestamp(wnba_games["GAME_DATE"].max())
    state.wnba_known_team_ids = set(wnba_games["TEAM_ID"].unique())
    state.wnba_models, state.wnba_manifest = load_wnba_models()

    state.gleague_state = load_gleague_state()
    state.gleague_data_as_of = state.gleague_state["data_as_of"]
    state.gleague_known_team_ids = state.gleague_state["known_team_ids"]
    state.gleague_models, state.gleague_manifest = load_gleague_models()

    # The NFL's own cutoff, own ids and own prediction rule. `load_nfl_state`
    # REFUSES a synthetic table set here unless MODELLARIUM_ALLOW_SYNTHETIC_NFL
    # is set, so a production boot cannot serve invented scores as results -
    # see live_nfl_features.refuse_synthetic.
    state.nfl_state = load_nfl_state()
    state.nfl_data_as_of = state.nfl_state["data_as_of"]
    state.nfl_known_team_ids = state.nfl_state["known_team_ids"]
    state.nfl_models, state.nfl_manifest = load_nfl_models()

    print(
        f"Serving data from {state.data_root} "
        f"(snapshot {state.snapshot.get('snapshot') or 'unstamped'})."
    )
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
    print(
        f"WNBA: {len(state.wnba_models)} models "
        f"({', '.join(sorted(state.wnba_models))}), "
        f"{len(state.wnba_known_team_ids)} teams, "
        f"data as of {state.wnba_data_as_of.date()}."
    )
    print(
        f"G League: {len(state.gleague_models)} models "
        f"({', '.join(sorted(state.gleague_models))}), "
        f"{len(state.gleague_known_team_ids)} teams, "
        f"data as of {state.gleague_data_as_of.date()}."
    )
    print(
        f"NFL: {len(state.nfl_models)} models "
        f"({', '.join(sorted(state.nfl_models))}), "
        f"{len(state.nfl_known_team_ids)} franchises, "
        f"{len(state.nfl_state['games']):,} games, "
        f"{len(nfl_predictable_fixtures(state.nfl_state))} of "
        f"{len(state.nfl_state['fixtures'])} fixtures predictable, "
        f"data as of {state.nfl_data_as_of.date()}."
        + ("  SYNTHETIC TABLES - CI only."
           if state.nfl_state.get("synthetic") else "")
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

def availability_block() -> dict:
    """Per-league availability state for /health. Cannot fail, cannot be slow.

    ADDITIVE. Every existing field of /health is untouched, because this
    endpoint already took the browse view down once by retyping models_loaded
    (CLAUDE.md section 21) and the lesson was to add rather than change.

    WRAPPED ENTIRELY, including the import: before 2026-10-07 the four
    availability features resolved to NaN whenever anything was wrong and
    /health said `ok` regardless, so the NBA could be served on 34 of 38
    features with nothing saying so. Reporting that must not introduce the
    failure it exists to describe - a health endpoint that 500s because it
    could not determine a sub-state is worse than the silence it replaces.
    """
    wnba_gleague = {
        "state": "not_applicable",
        "detail": "no availability features in this league's feature set",
    }
    try:
        sys.path.insert(0, str(ML_TRAINING_DIR))
        import injury_availability as availability

        from live_features import availability_is_required

        if not availability_is_required():
            nba = {"state": "not_applicable",
                   "detail": "FEATURE_COLUMNS carries no availability features"}
        else:
            nba = availability.availability_state(
                for_date=(state.data_as_of + pd.DateOffset(days=MAX_DAYS_AHEAD))
                .date().isoformat())
    except Exception as error:  # noqa: BLE001
        nba = {"state": "unknown",
               "detail": f"could not determine: {type(error).__name__}: {error}"}

    # The NFL's feature set carries no availability column either - phase 2's
    # manifest names quarterback availability as the largest known gap and
    # records that nothing player-level is in that phase. So the honest state
    # is not_applicable, not "none_published": nothing has been asked for.
    return {"nba": nba, "wnba": wnba_gleague, "gleague": wnba_gleague,
            "nfl": wnba_gleague}

@app.get("/health")
def health():
    """Liveness plus freshness, so monitoring can alert on stale data."""
    days_behind, stale = freshness()
    wnba_days_behind, wnba_stale = wnba_freshness()
    gleague_days_behind, gleague_stale = gleague_freshness()
    nfl_days_behind, nfl_stale = nfl_freshness()
    nfl_ready = len(nfl_predictable_fixtures(state.nfl_state))
    return {
        "status": "ok",
        "models_loaded": {
            "team": len(state.models),
            "quarter_half": len(state.qh_models),
            "player_props": len(state.pp_models),
            "wnba": len(state.wnba_models),
            "gleague": len(state.gleague_models),
            "nfl": len(state.nfl_models),
        },
        "data_as_of": state.data_as_of.date().isoformat(),
        "days_behind": days_behind,
        "stale": stale,
        # The WNBA's cutoff is a separate fact and gets separate fields. The
        # top-level data_as_of stays the NBA's, so every existing client keeps
        # reading what it always read - this endpoint already broke the browse
        # view once by retyping models_loaded (CLAUDE.md section 21), and the
        # lesson was to add rather than change.
        "wnba": {
            "data_as_of": state.wnba_data_as_of.date().isoformat(),
            "days_behind": wnba_days_behind,
            "stale": wnba_stale,
        },
        "gleague": {
            "data_as_of": state.gleague_data_as_of.date().isoformat(),
            "days_behind": gleague_days_behind,
            "stale": gleague_stale,
        },
        # The NFL's own cutoff, and the one field no other league has:
        # `predictable_fixtures`. The other three are served under
        # MAX_DAYS_AHEAD = 1, so "what is predictable" is a date a client can
        # compute from the cutoff. The NFL's rule is a DEPENDENCY, so it
        # cannot be - and a client that guessed cutoff + 1 would be wrong
        # about every fixture. Reported here so the answer comes from the
        # service that knows it.
        "nfl": {
            "data_as_of": state.nfl_data_as_of.date().isoformat(),
            "days_behind": nfl_days_behind,
            "stale": nfl_stale,
            "predictable_fixtures": nfl_ready,
            "fixtures": len(state.nfl_state["fixtures"]),
            "prediction_rule": "both teams' previous games in history",
            "source": NFL_SOURCE,
            "synthetic": bool(state.nfl_state.get("synthetic")),
        },
        # WHETHER AVAILABILITY IS ACTUALLY WORKING. Four of the NBA's 38
        # features come from the live injury report, and every way of not
        # having one used to resolve to NaN silently.
        "availability": availability_block(),
        # WHICH SNAPSHOT IS BEING SERVED. Without this, "the refresh ran" and
        # "the service picked it up" are two separate facts with one
        # observation between them - and the daily job's rollback path has
        # nothing to confirm itself against.
        "served_data": {
            "root": str(state.data_root),
            "snapshot": state.snapshot.get("snapshot"),
            "created": state.snapshot.get("created"),
            "source": state.snapshot.get("source"),
            # Present only when a snapshot's own stamp names a different id
            # than the directory it sits in - which happens when one is
            # copied by hand. Reported rather than hidden: a silent
            # disagreement here is the hazard this whole block exists for.
            "stamp_disagrees": state.snapshot.get("stamp_disagrees"),
            # THE CONTENTS, NOT JUST THE NAME. A gate comparing two captures
            # can only call a body difference attributable if the data under it
            # is the same, and the id alone cannot establish that.
            "table_sha256": state.table_hashes,
            "table_hash_seconds": state.table_hash_seconds,
        },
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

def gleague_schedule_seasons(today: date) -> list:
    """Which seasons to ask for upcoming G League fixtures, in order.

    A FIFTH QUESTION, AND THE EXISTING BOUNDARIES DO NOT ANSWER IT. The four
    on record map a game date to its season, or today to which seasons exist
    in the data. This asks which season's schedule holds fixtures that have
    not been played - and between March and November those are different
    seasons, because the G League season that "exists" by its November
    boundary is the one that finished in March.

    Measured rather than reasoned: on 2026-10-04 the 2025-26 schedule returns
    558 regular-season fixtures and 0 unplayed, while 2026-27 returns 527
    regular-season fixtures and all 527 unplayed. Asking only for the current
    season by the November boundary served an empty list while a full season
    sat one label away.

    DERIVED BY PROBING RATHER THAN BY A FIFTH CONSTANT. The caller takes the
    first season that yields fixtures, so the answer comes from the schedule
    itself and self-corrects across the boundary - no new calendar rule to get
    wrong, which is section 35's lesson. The cost is one extra upstream call
    in the pre-season window, and both are cached for six hours.
    """
    from fetch_gleague_games import current_season_start_year, season_label
    start = current_season_start_year(today)
    return [season_label(start), season_label(start + 1)]


def gleague_season_string(today: date) -> str:
    """The G League labels a season across two years: "2025-26".

    Taken from the G League fetcher's own boundary rather than restated. Its
    season tips off in NOVEMBER, which is neither the NBA's October nor the
    WNBA's May - four boundaries now exist in this project and they answer
    four different questions.
    """
    from fetch_gleague_games import current_season_start_year, season_label
    return season_label(current_season_start_year(today))


def wnba_season_string(today: date) -> str:
    """The WNBA labels a season by its single calendar year: "2026".

    Taken from the WNBA fetcher's own boundary rather than restated. A WNBA
    season runs May to October inside one year, so there is no hyphenated
    second year and no August rule - a different question from the NBA's, and
    the two must not be harmonised.
    """
    from fetch_wnba_games import current_season
    return str(current_season(today))

def fetch_schedule_frame(season: str, league_id: str = NBA_LEAGUE_ID) -> pd.DataFrame:
    """The season's full schedule, cached briefly.

    Keyed on (league, season) rather than season alone. The two leagues label
    seasons differently - "2026-27" against "2026" - so string keys happen not
    to collide today, and keying on the pair means that stays true by
    construction rather than by coincidence.
    """
    key = (league_id, season)
    cached = _schedule_cache.get(key)
    if cached is not None:
        age = (datetime.now() - cached["fetched_at"]).total_seconds()
        if age < SCHEDULE_CACHE_TTL_SECONDS:
            return cached["frame"]

    try:
        frames = scheduleleaguev2.ScheduleLeagueV2(
            season=season, league_id=league_id, timeout=SCHEDULE_TIMEOUT_SECONDS
        ).get_data_frames()
        # An unpublished season returns NO frames at all rather than an empty
        # one, so get_data_frames()[0] raises IndexError. Measured on WNBA
        # season 2027 in October 2026. Treated as "nothing scheduled yet",
        # which is an ordinary state of the world between seasons.
        frame = frames[0] if frames else pd.DataFrame()
    except IndexError:
        frame = pd.DataFrame()
    except Exception as error:
        raise HTTPException(
            502, f"Could not reach the schedule API for league {league_id} "
                 f"season {season}: {error}"
        ) from error

    _schedule_cache[key] = {"fetched_at": datetime.now(), "frame": frame}
    return frame

def upcoming_regular_season(frame: pd.DataFrame, days_ahead: int) -> list:
    """Unplayed regular-season fixtures inside the window.

    Shared by both leagues' schedule endpoints so the filter exists once. Two
    things in it are load-bearing and were each a real defect somewhere:

    The derived columns are assigned BEFORE filtering. Assigning a Series onto
    a zero-row frame reindexes the frame back up to the Series' index in
    pandas 2.3.3, so filter-then-assign turned "nothing scheduled" into a full
    frame of NaN and a 500 - and it failed only in the offseason.

    The type-digit filter is not decorative. Of the 377 rows the WNBA's 2026
    schedule returns, 18 are preseason, 27 playoff, one All-Star and one
    Commissioner's Cup; ALL 17 currently-unplayed games are playoffs. Without
    the digit filter those 17 would be offered as predictable by models that
    have never seen a playoff game.
    """
    if frame.empty:
        return []

    today = pd.Timestamp(datetime.now().date())
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

@app.get("/schedule", response_model=list[ScheduledGame])
def schedule(
    days_ahead: int = Query(SCHEDULE_DAYS_AHEAD_DEFAULT, ge=1, le=365),
):
    """Upcoming regular-season fixtures, as candidates to display."""
    today = pd.Timestamp(datetime.now().date())
    frame = fetch_schedule_frame(season_string(today.date()), NBA_LEAGUE_ID)
    return upcoming_regular_season(frame, days_ahead)

@app.get("/schedule/wnba", response_model=list[ScheduledGame])
def schedule_wnba(
    days_ahead: int = Query(SCHEDULE_DAYS_AHEAD_DEFAULT, ge=1, le=365),
):
    """Upcoming WNBA regular-season fixtures.

    A SIBLING ENDPOINT, not a league query parameter on /schedule. Two
    reasons: the season label has a different shape per league, and the NBA's
    response must stay byte-identical while a second league is added - a
    parameter with a default is one typo away from changing it.
    """
    frame = fetch_schedule_frame(
        wnba_season_string(datetime.now().date()), WNBA_LEAGUE_ID)
    return upcoming_regular_season(frame, days_ahead)

@app.get("/schedule/gleague", response_model=list[ScheduledGame])
def schedule_gleague(
    days_ahead: int = Query(SCHEDULE_DAYS_AHEAD_DEFAULT, ge=1, le=365),
):
    """Upcoming G League REGULAR-SEASON fixtures.

    A sibling endpoint for the same two reasons the WNBA's is: the season
    label has a different shape per league, and the NBA's response must stay
    byte-identical while a third league is added.

    THE REGULAR-SEASON FILTER IS LOAD-BEARING HERE, NOT DECORATIVE. The G
    League plays a Showcase Cup before its regular season, and those games
    carry type digit 5 - which in the NBA's scheme means the play-in. A
    cached Cup fixture would show as a predictable game that the backend then
    rejects, because no shipped model has seen one. `upcoming_regular_season`
    keeps only digit 2, the same filter the training pull used.
    """
    today = datetime.now().date()
    for season in gleague_schedule_seasons(today):
        frame = fetch_schedule_frame(season, GLEAGUE_LEAGUE_ID)
        fixtures = upcoming_regular_season(frame, days_ahead)
        if fixtures:
            return fixtures
    return []


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

def wnba_freshness() -> tuple:
    """The WNBA's own staleness, against its own cutoff."""
    days_behind = (pd.Timestamp(datetime.now().date()) - state.wnba_data_as_of).days
    return days_behind, days_behind > STALE_AFTER_DAYS

def validate_wnba_matchup(request: "PredictionRequest") -> pd.Timestamp:
    """The WNBA counterpart of validate_matchup, against WNBA state.

    A separate function rather than a parameter on the NBA one: the team-id
    universes are disjoint, so a shared validator taking a league flag would
    accept an NBA id for a WNBA fixture whenever the flag was wrong, and the
    error would surface much later as a feature row of NaN.
    """
    game_date = pd.Timestamp(request.game_date)

    if request.home_team_id == request.away_team_id:
        raise HTTPException(400, "home_team_id and away_team_id must differ.")

    unknown = [
        team_id
        for team_id in (request.home_team_id, request.away_team_id)
        if team_id not in state.wnba_known_team_ids
    ]
    if unknown:
        raise HTTPException(
            400,
            f"No WNBA history for team id(s) {unknown}. WNBA and NBA team ids "
            f"are different universes, so an NBA id reaches this endpoint as "
            f"an unknown team rather than as a wrong league.",
        )

    latest_allowed = state.wnba_data_as_of + pd.DateOffset(days=MAX_DAYS_AHEAD)
    if game_date > latest_allowed:
        raise HTTPException(
            400,
            f"game_date {request.game_date} is more than {MAX_DAYS_AHEAD} day "
            f"past the newest WNBA game in the data "
            f"({state.wnba_data_as_of.date()}). Rest-day features would be "
            f"computed against the wrong prior game. Latest accepted date is "
            f"{latest_allowed.date()}.",
        )

    return game_date

def gleague_freshness() -> tuple:
    """The G League's own staleness, against its own cutoff."""
    days_behind = (pd.Timestamp(datetime.now().date())
                   - state.gleague_data_as_of).days
    return days_behind, days_behind > STALE_AFTER_DAYS


def validate_gleague_matchup(request: "PredictionRequest") -> pd.Timestamp:
    """The G League counterpart, against G League state.

    A separate function rather than a league flag on the NBA one, for the
    reason the WNBA's records: the three id universes are disjoint, so a
    shared validator taking a flag would accept the wrong league's id
    whenever the flag was wrong, and the error would surface much later as a
    feature row of NaN.
    """
    game_date = pd.Timestamp(request.game_date)

    if request.home_team_id == request.away_team_id:
        raise HTTPException(400, "home_team_id and away_team_id must differ.")

    unknown = [
        team_id
        for team_id in (request.home_team_id, request.away_team_id)
        if team_id not in state.gleague_known_team_ids
    ]
    if unknown:
        raise HTTPException(
            400,
            f"No G League history for team id(s) {unknown}. The NBA, WNBA "
            f"and G League id universes are disjoint, so another league's id "
            f"reaches this endpoint as an unknown team rather than as a "
            f"wrong league.",
        )

    latest_allowed = state.gleague_data_as_of + pd.DateOffset(
        days=MAX_DAYS_AHEAD)
    if game_date > latest_allowed:
        raise HTTPException(
            400,
            f"game_date {request.game_date} is more than {MAX_DAYS_AHEAD} "
            f"day past the newest G League game in the data "
            f"({state.gleague_data_as_of.date()}). Rest-day features would "
            f"be computed against the wrong prior game, and in this league "
            f"that prior game may be a Showcase Cup game. Latest accepted "
            f"date is {latest_allowed.date()}.",
        )

    return game_date


@app.post("/predict/gleague", response_model=GleagueResponse)
def predict_gleague(request: PredictionRequest):
    """Three G League markets for one fixture: moneyline, spread and totals.

    Refuses rather than degrades when a feature row cannot be completed: the
    shipped models are linear Pipelines with requires_complete_features set,
    so unlike the NBA's XGBoost they cannot score a NaN at all.
    """
    game_date = validate_gleague_matchup(request)

    try:
        features = get_live_gleague_features(
            request.home_team_id, request.away_team_id, game_date,
            state.gleague_state,
        )
    except GleagueNotScoreable as error:
        raise HTTPException(400, str(error)) from error
    except (ValueError, KeyError) as error:
        raise HTTPException(
            400, f"Could not build features: {error}") from error

    markets = {}
    for target, model in state.gleague_models.items():
        entry = state.gleague_manifest["targets"][target]
        row = pd.DataFrame([features["rows"][target]],
                           columns=entry["features"])
        if entry["metric"] == "log_loss":
            value = float(model.predict_proba(row)[0][1])
        else:
            value = float(model.predict(row)[0])
        markets[target] = GleagueMarket(
            value=value,
            metric=entry["metric"],
            window=entry["window"],
        )

    days_behind, stale = gleague_freshness()
    return GleagueResponse(
        home_team_id=request.home_team_id,
        away_team_id=request.away_team_id,
        game_date=request.game_date,
        data_as_of=state.gleague_data_as_of.date(),
        stale=stale,
        days_behind=days_behind,
        season=features["season"],
        markets=markets,
    )


@app.post("/predict/wnba", response_model=WnbaResponse)
def predict_wnba(request: PredictionRequest):
    """Three WNBA markets for one fixture: moneyline, spread and totals.

    Refuses rather than degrades when a feature row cannot be completed. The
    WNBA models are linear Pipelines with requires_complete_features set, so
    unlike the NBA's XGBoost they cannot score a NaN at all - the same shape
    of refusal the quarter/half endpoint makes, and for the same reason.
    """
    game_date = validate_wnba_matchup(request)

    try:
        features = get_live_wnba_features(
            request.home_team_id, request.away_team_id, game_date
        )
    except WnbaNotScoreable as error:
        raise HTTPException(400, str(error)) from error
    except (ValueError, KeyError) as error:
        raise HTTPException(400, f"Could not build features: {error}") from error

    markets = {}
    for target, model in state.wnba_models.items():
        entry = state.wnba_manifest["targets"][target]
        row = pd.DataFrame([features["rows"][target]], columns=entry["features"])
        if entry["metric"] == "log_loss":
            value = float(model.predict_proba(row)[0][1])
        else:
            value = float(model.predict(row)[0])
        markets[target] = WnbaMarket(
            value=value,
            metric=entry["metric"],
            window=entry["window"],
        )

    days_behind, stale = wnba_freshness()
    return WnbaResponse(
        home_team_id=request.home_team_id,
        away_team_id=request.away_team_id,
        game_date=request.game_date,
        data_as_of=state.wnba_data_as_of.date(),
        stale=stale,
        days_behind=days_behind,
        season=features["season"],
        markets=markets,
    )


def nfl_freshness() -> tuple:
    """The NFL's own staleness, against its own cutoff."""
    days_behind = (pd.Timestamp(datetime.now().date())
                   - state.nfl_data_as_of).days
    return days_behind, days_behind > STALE_AFTER_DAYS


def validate_nfl_matchup(request: "PredictionRequest") -> pd.Timestamp:
    """The NFL counterpart, and the one that does NOT check a date window.

    THE OTHER THREE LEAGUES ENFORCE MAX_DAYS_AHEAD = 1 AND THIS ONE MUST NOT.
    That rule exists because their rest-day features are computed against the
    team's previous game, so a date more than a day past the cutoff would be
    scored against the wrong prior game. The NFL's features have the same
    property and a stronger guarantee: every one reads only each team's own
    earlier games, and Elo updates only on games already played, so a fixture
    is final as soon as both sides' previous games are recorded - whatever the
    date. Phase 2 measured that as 15 of 208 fixtures predictable at once, a
    whole week's slate, where a date rule yields a single day.

    So the window check is REPLACED rather than relaxed, and the replacement
    lives in `get_live_nfl_features`: it refuses a fixture whose teams have an
    earlier unplayed game, and names which side. Applying MAX_DAYS_AHEAD here
    as well would refuse 14 of those 15 fixtures for a reason that does not
    hold in this league.

    The three basketball leagues' validators are untouched by this, which the
    regression gate checks rather than this comment asserting it.
    """
    game_date = pd.Timestamp(request.game_date)

    if request.home_team_id == request.away_team_id:
        raise HTTPException(400, "home_team_id and away_team_id must differ.")

    unknown = [
        team_id
        for team_id in (request.home_team_id, request.away_team_id)
        if team_id not in state.nfl_known_team_ids
    ]
    if unknown:
        raise HTTPException(
            400,
            f"No NFL history for franchise id(s) {unknown}. The four leagues' "
            f"id universes are disjoint - the NFL's are "
            f"1613000001-1613000032, project-assigned because the source "
            f"carries no stable team id - so another league's id reaches this "
            f"endpoint as an unknown franchise rather than as a wrong league.",
        )

    return game_date


@app.get("/schedule/nfl", response_model=list[NflFixture])
def schedule_nfl(
    predictable_only: bool = Query(
        False,
        description="only fixtures whose features are already final",
    ),
):
    """Unplayed NFL fixtures, from the served table rather than a live API.

    NO UPSTREAM CALL AND NO CACHE, unlike the three basketball schedules. Those
    ask nba_api's ScheduleLeagueV2 and cache the frame for six hours. The NFL's
    fixtures arrive in the served snapshot, because the same Wikipedia articles
    that carry results carry the remaining schedule - so the fixture list is as
    fresh as the daily refresh and no fresher, which is exactly what
    `data_as_of` already says.
    """
    fixtures = nfl_upcoming(state.nfl_state)
    if predictable_only:
        fixtures = [row for row in fixtures if row["predictable"]]
    return [NflFixture(**row) for row in fixtures]


@app.post("/predict/nfl", response_model=NflResponse)
def predict_nfl(request: PredictionRequest):
    """Three NFL markets for one fixture: winner, margin and total.

    DOES NOT REFUSE AN INCOMPLETE FEATURE ROW, which is the opposite of the
    WNBA and G League endpoints and is a property of the artifacts rather than
    a looser standard. Their linear Pipelines carry no imputer, so a NaN
    cannot be scored at all. The NFL's two linear artifacts carry a
    SimpleImputer fitted on training rows only, with SEASON_OPENER riding
    along as the indicator that the value was supplied rather than observed -
    so a season opener scores exactly as it did in training. Which features
    were imputed is reported per market instead of being hidden.
    """
    validate_nfl_matchup(request)

    try:
        features = get_live_nfl_features(
            request.home_team_id, request.away_team_id,
            request.game_date, state.nfl_state,
        )
    except NflNotScoreable as error:
        raise HTTPException(400, str(error)) from error
    except (ValueError, KeyError) as error:
        raise HTTPException(
            400, f"Could not build features: {error}") from error

    markets = {}
    for market, model in state.nfl_models.items():
        entry = state.nfl_manifest["markets"][market]
        row = pd.DataFrame([features["rows"][market]],
                           columns=entry["features"])
        if market == "winner":
            value = float(model.predict_proba(row)[0][1])
            metric = "log_loss"
        else:
            value = float(model.predict(row)[0])
            metric = "mae"
        verdict = (state.nfl_manifest["test"].get(market) or {}).get("verdict")
        markets[market] = NflMarket(
            value=value,
            metric=metric,
            model_used=f"{entry['family']}/{entry['elo_variant']}/"
                       f"{entry['feature_set']}",
            note=verdict,
            imputed=features["missing"].get(market, []),
        )

    days_behind, stale = nfl_freshness()
    return NflResponse(
        home_team_id=request.home_team_id,
        away_team_id=request.away_team_id,
        game_date=request.game_date,
        data_as_of=state.nfl_data_as_of.date(),
        stale=stale,
        days_behind=days_behind,
        season=features["season"],
        week=features["week"],
        markets=markets,
        home_win_interpretation=state.nfl_manifest["winner_qualifier"],
        source=NFL_SOURCE,
    )
