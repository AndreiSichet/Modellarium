"""Serving-time features for an unplayed G League fixture.

EVERY PARAMETER COMES FROM manifest.json - the window, the Elo K and
carryover, the expansion offset, the feature list. Nothing is restated here,
so a reselection needs no change to this file and the two cannot drift.

THE FEATURES ARE REPLAYED FROM THE RAW TABLES, not read from a precomputed
column. Phase 2 wrote `gleague_rolling_features.csv` and `gleague_elo.csv`,
and both were built under parameters that are NOT the shipped ones: phase 2's
Elo diagnostic fitted carryover 0.200 on a training span, while the artifacts
were fitted under 0.333 on everything. A stored column beside a manifest
declaring different parameters is a drift surface with no guard on it; a raw
table cannot drift from itself.

REST DAYS COUNT SHOWCASE CUP GAMES, AND THAT IS WHY THE CUP TABLE SHIPS.
Phase 2 computed `REST_DAYS` across the Cup boundary, so a team's first
regular-season game of a Cup season reads rest from its last Cup game. If
serving used regular-season games only, that one game per team per season
would read as a season opener - a value the model never saw for that row.
Silent, and exactly the mismatch the offline comparison exists to catch.

FORM FEATURES USE REGULAR-SEASON GAMES ONLY. All three shipped windows are
CARRY10: phase 3's selection tied on every candidate, and the serving
tie-break switched its two CUP picks to their CARRY equivalents so no form
feature depends on a live Cup result. The Cup still reaches rest days, which
is a different column and a separate decision.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ML = HERE.parent
PROJECT = ML.parent
GLEAGUE_PREP = PROJECT / "data-pipeline" / "gleague" / "preprocessing"
GLEAGUE_INGEST = PROJECT / "data-pipeline" / "gleague" / "ingestion"
# The ingestion directory is on the path for fetch_gleague_games' season
# helpers, which app.py's /schedule/gleague needs. Same arrangement as the
# WNBA module's: the module that owns a league's layout is what puts that
# league's directories on the path, rather than app.py knowing three layouts.
for _path in (GLEAGUE_PREP, GLEAGUE_INGEST,
              PROJECT / "data-pipeline" / "ingestion"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import json  # noqa: E402

def processed_dir() -> Path:
    """Resolved at call time so DATA_DIR moves it onto the volume."""
    from served_data import data_root

    return data_root() / "gleague" / "processed"
MODELS_DIR = ML / "models_gleague"
MANIFEST_PATH = MODELS_DIR / "manifest.json"

REST_DAYS_CAP = 7
SEASON_START_MONTH = 11


class NotScoreable(Exception):
    """The fixture cannot be scored, with the reason named.

    An exception rather than a row of NaN: the shipped models are linear
    pipelines and cannot take a NaN at all, and a None gets treated as
    "empty, so nothing is wrong" - the reasoning that made
    NoReportAvailable an exception too.
    """


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def load_games() -> pd.DataFrame:
    frame = pd.read_csv(processed_dir() / "gleague_games_final.csv",
                        dtype={"GAME_ID": str},
                        parse_dates=["GAME_DATE"])
    frame["COMPETITION"] = "regular"
    return frame


def load_showcase() -> pd.DataFrame:
    path = processed_dir() / "gleague_showcase_games.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, dtype={"GAME_ID": str},
                        parse_dates=["GAME_DATE"])
    frame["COMPETITION"] = "showcase"
    return frame


def assert_margin_is_derived(frame: pd.DataFrame) -> None:
    """PLUS_MINUS must equal PTS minus the opponent's.

    The source column is wrong on 6.79% of regular-season games here and
    14.6% of Cup games - the worst of the three leagues - and the builder
    recomputes it. Serving asserts that rather than trusting it, because a
    rolling PLUS_MINUS built on the raw column would differ from what the
    models were fitted on.
    """
    opponent = frame.groupby("GAME_ID")["PTS"].transform(
        lambda s: s.values[::-1] if len(s) == 2 else pd.NA)
    if int(((frame["PTS"] - opponent) != frame["PLUS_MINUS"]).sum()):
        raise RuntimeError(
            "PLUS_MINUS is not the derived margin in the served table")


def season_of(game_date: pd.Timestamp) -> str:
    """A date to its G League season label.

    November, the boundary `fetch_gleague_games` uses - and deliberately NOT
    the NBA's August or October or the WNBA's May. There are four in this
    project and they answer four different questions.
    """
    year = (game_date.year if game_date.month >= SEASON_START_MONTH
            else game_date.year - 1)
    return f"{year}-{str(year + 1)[-2:].zfill(2)}"


def split_window(label: str) -> tuple:
    """'CARRY10' -> ('CARRY', 10). The prefix decides the scope."""
    for prefix in ("CARRY", "CUP", "ROLL"):
        if label.startswith(prefix):
            return prefix, int(label[len(prefix):])
    raise RuntimeError(f"unrecognised window label {label!r}")


def window_scope(prefix: str) -> dict:
    """What a window prefix means, read from one place.

    Stated as data rather than as `if prefix == "CARRY"` scattered through
    the module, so adding a prefix is one entry rather than a search.
    """
    return {
        "ROLL": {"crosses_seasons": False, "include_cup": False},
        "CARRY": {"crosses_seasons": True, "include_cup": False},
        "CUP": {"crosses_seasons": True, "include_cup": True},
    }[prefix]


def replay_elo(frame: pd.DataFrame, manifest: dict) -> dict:
    """Every team's current rating, replayed from the first season on record.

    Imported formula, not a restated one - and replayed in full rather than
    updated one step from a stored rating, because 9,628 games costs well
    under a second at load and a stored rating would have to agree with the
    manifest's parameters.
    """
    from build_gleague_elo import expected_score, prepare, run_elo
    from clean_gleague_rows import VALID_WL  # noqa: F401

    elo = manifest["elo"]
    games = prepare(frame)
    played = run_elo(games, elo["k"], elo["carryover"],
                     elo["expansion_offset"], newcomers={})

    # The rating each team carries AFTER its last game, plus the
    # between-season regression if the next fixture is in a new season.
    ratings = {}
    for row in played.itertuples(index=False):
        change = elo["k"] * (row.HOME_WIN - row.HOME_EXPECTED)
        ratings[row.HOME_TEAM_ID] = row.HOME_TEAM_ELO + change
        ratings[row.AWAY_TEAM_ID] = row.AWAY_TEAM_ELO - change

    last_season = {}
    for row in played.itertuples(index=False):
        last_season[row.HOME_TEAM_ID] = row.SEASON
        last_season[row.AWAY_TEAM_ID] = row.SEASON

    return {"ratings": ratings, "last_season": last_season,
            "baseline": elo["baseline_rating"],
            "carryover": elo["carryover"],
            "offset": elo["expansion_offset"],
            "expected_score": expected_score}


def elo_for(state: dict, team_id: int, season: str) -> float:
    """A team's rating entering `season`, regressed if the season is new."""
    if team_id not in state["ratings"]:
        return state["baseline"] + state["offset"]
    rating = state["ratings"][team_id]
    if state["last_season"].get(team_id) == season:
        return rating
    mean = float(np.mean(list(state["ratings"].values())))
    return mean + state["carryover"] * (rating - mean)


def team_history(frame: pd.DataFrame, team_id: int,
                 before: pd.Timestamp) -> pd.DataFrame:
    rows = frame[(frame["TEAM_ID"] == team_id) & (frame["GAME_DATE"] < before)]
    return rows.sort_values(["GAME_DATE", "GAME_ID"])


def rolling_for(history: pd.DataFrame, window: str, season: str,
                metrics: list) -> dict:
    """The trailing means a window contributes, or NaN where short.

    Mirrors `trailing_mean`'s shift-then-roll: the mean of the previous
    `size` games, excluding the fixture being predicted - which here is
    automatic, since `history` holds only earlier games.
    """
    prefix, size = split_window(window)
    scope = window_scope(prefix)

    rows = history
    if not scope["include_cup"]:
        rows = rows[rows["COMPETITION"] == "regular"]
    if not scope["crosses_seasons"]:
        rows = rows[rows["SEASON"] == season]

    out = {}
    recent = rows.tail(size)
    for metric in metrics:
        source = "WIN" if metric == "WIN_PCT" else metric
        out[metric] = (float(recent[source].mean())
                       if len(recent) == size else float("nan"))
    return out


def rest_for(history: pd.DataFrame, game_date: pd.Timestamp,
             season: str) -> dict:
    """Rest from the previous game of ANY competition, within the season.

    Season-scoped, like phase 2's: a seven-month off-season is not a week
    off, so a genuine season opener is NaN and the fixture is refused rather
    than scored on an invented value.
    """
    same_season = history[history["SEASON"] == season]
    if not len(same_season):
        return {"REST_DAYS": float("nan"), "IS_LONG_BREAK": float("nan")}
    previous = same_season.iloc[-1]["GAME_DATE"]
    raw = float((game_date - previous).days)
    return {"REST_DAYS": min(raw, float(REST_DAYS_CAP)),
            "IS_LONG_BREAK": float(raw > REST_DAYS_CAP)}


def team_features(frame: pd.DataFrame, state: dict, team_id: int,
                  game_date: pd.Timestamp, window: str,
                  metrics: list) -> dict:
    season = season_of(game_date)
    history = team_history(frame, team_id, game_date)
    out = rolling_for(history, window, season, metrics)
    out.update(rest_for(history, game_date, season))
    out["TEAM_ELO"] = elo_for(state, team_id, season)
    return out


def feature_row(entry: dict, home: dict, away: dict) -> tuple:
    """The model's feature vector, in the manifest's own column order.

    Built by NAME against `entry["features"]` rather than by position, so a
    reordering in the manifest cannot silently feed the model a permuted
    row.
    """
    values, missing = [], []
    for column in entry["features"]:
        side, _, rest = column.partition("_")
        source = home if side == "HOME" else away
        key = rest[len(entry["window"]) + 1:] if rest.startswith(
            entry["window"] + "_") else rest
        if key not in source:
            raise RuntimeError(f"no value computed for {column}")
        value = source[key]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            missing.append(column)
        values.append(value)
    return np.array([values], dtype=float), missing


def load_state() -> dict:
    regular = load_games()
    assert_margin_is_derived(regular)
    showcase = load_showcase()
    if len(showcase):
        assert_margin_is_derived(showcase)

    both = pd.concat([regular, showcase], ignore_index=True)
    both = both.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    manifest = load_manifest()
    return {"regular": regular, "all_games": both, "manifest": manifest,
            "elo": replay_elo(regular, manifest),
            "known_team_ids": set(regular["TEAM_ID"].unique()),
            "data_as_of": pd.Timestamp(regular["GAME_DATE"].max())}


def data_as_of() -> pd.Timestamp:
    return pd.Timestamp(load_games()["GAME_DATE"].max())


def get_live_features(home_team_id: int, away_team_id: int, game_date,
                      state: dict = None) -> dict:
    """One feature row per target, keyed by target name.

    Raises NotScoreable, naming the columns, when a window is incomplete -
    which is the honest answer for a linear pipeline that cannot take NaN.
    """
    state = state or load_state()
    manifest = state["manifest"]
    game_date = pd.Timestamp(game_date)
    metrics = manifest["rolling_metrics"]

    out = {}
    incomplete = {}
    for target, entry in manifest["targets"].items():
        window = entry["window"]
        home = team_features(state["all_games"], state["elo"], home_team_id,
                             game_date, window, metrics)
        away = team_features(state["all_games"], state["elo"], away_team_id,
                             game_date, window, metrics)
        row, missing = feature_row(entry, home, away)
        if missing:
            incomplete[target] = missing
        out[target] = row

    if incomplete and manifest.get("requires_complete_features"):
        detail = "; ".join(f"{t}: {', '.join(c)}"
                           for t, c in incomplete.items())
        raise NotScoreable(
            f"incomplete features for this fixture ({detail}). The shipped "
            f"models are linear pipelines and cannot score a partial row.")

    # The same shape live_wnba_features returns, so the two endpoints in
    # app.py differ only in which league they name.
    return {"rows": {t: row[0] for t, row in out.items()},
            "season": season_of(game_date)}
