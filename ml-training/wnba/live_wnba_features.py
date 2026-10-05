"""Pre-game WNBA features for a fixture that has not been played.

Everything here is REPLAYED from the raw long table under the parameters
recorded in models_wnba/manifest.json. Three consequences worth stating
because each one is a deliberate difference from the NBA's live_features.py:

  - No precomputed feature column is read. wnba_games_final_features.csv
    carries rolling, rest and Elo already, and it is deliberately absent from
    the inference image: a stored column fitted under one set of parameters
    sitting beside a manifest declaring another is a drift surface with no
    guard on it. The raw table cannot drift from itself.

  - The window, the feature list and the Elo parameters all come FROM the
    manifest. Phase 3 selected CARRY5 for moneyline and spread and CARRY10 for
    totals, so there is no single window to hardcode even if hardcoding were
    acceptable - and a reselection must not need a code change here.

  - Elo is a full replay rather than the NBA's one-step update from a stored
    rating, because there is no stored rating to update from. 5,310 games in a
    Python loop costs well under a second, and it happens once at load.

Formulas and constants are imported from the pipeline scripts that define
them, never restated - the one exception is the between-season carryover,
which lives inside a closure in run_elo and is marked where it is reused.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML = Path(__file__).resolve().parents[1]
PROJECT = ML.parent
WNBA_PREP = PROJECT / "data-pipeline" / "wnba" / "preprocessing"
WNBA_INGEST = PROJECT / "data-pipeline" / "wnba" / "ingestion"
for path in (WNBA_PREP, WNBA_INGEST, PROJECT / "data-pipeline" / "ingestion"):
    sys.path.insert(0, str(path))

from build_wnba_elo import (  # noqa: E402
    BASELINE_RATING, expected_score, prepare, run_elo,
)
from build_wnba_rest_days import (  # noqa: E402
    LONG_BREAK_DAYS, REST_DAYS_CAP, add_rest_days,
)
from build_wnba_rolling_features import (  # noqa: E402
    METRICS, SCOPES, TEAM_KEY, add_rolling,
)

MODELS_DIR = ML / "models_wnba"
MANIFEST_PATH = MODELS_DIR / "manifest.json"
def games_path() -> "Path":
    """Resolved at call time, so DATA_DIR moves it onto the volume.

    A function rather than the module-level constant it replaces: the
    constant was evaluated at import, which made the served path depend on
    whether the environment was set before this module was first imported.
    """
    from served_data import data_root

    return data_root() / "wnba" / "processed" / "wnba_games_final.csv"


GAMES_PATH = (PROJECT / "data-pipeline" / "data" / "wnba" / "processed"
              / "wnba_games_final.csv")

# The long table with the pipeline's own feature columns. Used ONLY by the
# verification harness, never by serving, and absent from the image.
PIPELINE_FEATURES_PATH = GAMES_PATH.with_name("wnba_games_final_features.csv")

WINDOW_PATTERN = re.compile(r"^([A-Z]+?)(\d+)$")


class NotScoreable(Exception):
    """A fixture whose required features cannot be completed.

    An exception rather than a row of NaN, for the same reason
    fetch_current_injury_report raises NoReportAvailable: the WNBA models are
    linear pipelines with requires_complete_features set, so they cannot take
    a NaN at all. Returning one would push the refusal to whoever forgot to
    check for it.
    """


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def split_window(label: str) -> tuple:
    """'CARRY5' -> ('CARRY', 5). Raises on anything the pipeline cannot build."""
    match = WINDOW_PATTERN.match(label)
    if not match or match.group(1) not in SCOPES:
        raise ValueError(f"unknown window label {label!r}; the pipeline builds "
                         f"prefixes {sorted(SCOPES)}")
    return match.group(1), int(match.group(2))


def crosses_seasons(prefix: str) -> bool:
    """Whether this scope carries history across the season boundary.

    Read off the pipeline's own grouping keys rather than asserted: ROLL groups
    on (team, season) and CARRY on team alone, so the absence of SEASON IS the
    definition of carrying. Restating 'CARRY crosses seasons' here would be a
    second copy of a fact that already exists one import away.
    """
    return "SEASON" not in SCOPES[prefix]


def required_windows(manifest: dict) -> list:
    """Every window any target asks for, deduplicated."""
    return sorted({entry["window"] for entry in manifest["targets"].values()})


def load_games() -> pd.DataFrame:
    frame = pd.read_csv(games_path(), dtype={"GAME_ID": str})
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])
    frame["WIN"] = (frame["WL"] == "W").astype(int)
    frame["PTS_ALLOWED"] = frame["PTS"] - frame["PLUS_MINUS"]
    assert_margin_is_derived(frame)
    return frame.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def assert_margin_is_derived(frame: pd.DataFrame) -> None:
    """PLUS_MINUS must be the margin derived from PTS, not the source column.

    Phase 1 found the source column wrong on ~1% of games, and PTS_ALLOWED
    above is computed from it, so a regression here would quietly corrupt a
    rolling input at serving time only. The pipeline asserts the same property
    on the same file; this is the serving side of that gate.
    """
    opponent = frame.groupby("GAME_ID")["PTS"].transform(
        lambda s: s.values[::-1] if len(s) == 2 else np.nan)
    off = int((frame["PLUS_MINUS"] - (frame["PTS"] - opponent)).abs().gt(0).sum())
    if off:
        raise RuntimeError(
            f"{off} rows where PLUS_MINUS is not PTS - opponent PTS. The "
            "phase-1 margin correction is missing from the served table, so "
            "PTS_ALLOWED and every rolling feature built on it would be wrong.")


def replay_elo(frame: pd.DataFrame, k: float, carryover: float) -> dict:
    """Final rating and last season per team, by replaying every game.

    run_elo's fourth return value is the ratings dict as it stands after the
    last game it processed, which is exactly the pre-game rating for a team's
    next fixture - subject to the between-season regression applied below.
    """
    _, _, _, ratings = run_elo(prepare(frame), k, carryover)
    last_season = frame.groupby(TEAM_KEY)["SEASON"].max().to_dict()
    return {"ratings": ratings, "last_season": last_season,
            "k": k, "carryover": carryover}


def elo_for(state: dict, team_id: int, season: int) -> float:
    """That team's rating going into a game in `season`."""
    rating = state["ratings"].get(team_id)
    if rating is None:
        # An unseen team starts at the league mean, matching run_elo's
        # rating_for. Stated as an assumption there, not a finding.
        return BASELINE_RATING
    if state["last_season"].get(team_id) != season:
        # THE ONE RESTATED FORMULA. run_elo applies this inside a closure that
        # cannot be imported; it is the same expression, and the verification
        # harness compares both sides across a real season boundary rather
        # than trusting that they agree.
        rating = BASELINE_RATING + (rating - BASELINE_RATING) * (1 - state["carryover"])
    return float(rating)


def team_history(frame: pd.DataFrame, team_id: int, before: pd.Timestamp) -> pd.DataFrame:
    """That team's games strictly before the given date, oldest first."""
    rows = frame[(frame[TEAM_KEY] == team_id) & (frame["GAME_DATE"] < before)]
    return rows.sort_values(["GAME_DATE", "GAME_ID"])


def rolling_for(history: pd.DataFrame, window_label: str, season: int) -> dict:
    """Trailing means for one window candidate, over the team's recent games.

    The pipeline produces these with trailing_mean, which shifts by one so a
    row never sees its own result. Here the fixture has no result to exclude,
    so the window is the team's most recent `size` games INCLUDING its latest -
    which is the same set of games the next played row's column would cover.
    """
    prefix, size = split_window(window_label)
    scope = history if crosses_seasons(prefix) else history[history["SEASON"] == season]
    recent = scope.tail(size)
    complete = len(recent) == size
    return {f"{window_label}_{label}":
            float(recent[source].mean()) if complete else np.nan
            for source, label in METRICS.items()}


def rest_for(history: pd.DataFrame, game_date: pd.Timestamp, season: int) -> dict:
    """Days since the team's last game THIS SEASON, capped, plus both flags.

    Season-scoped, matching add_rest_days. A season opener is NaN rather than
    the cap: a seven-month WNBA offseason is not seven days of rest, and those
    are different facts. It makes the opener unscoreable, which is correct and
    reported rather than filled.
    """
    in_season = history[history["SEASON"] == season]
    if in_season.empty:
        return {"REST_DAYS": np.nan, "IS_LONG_BREAK": 0.0,
                "IS_BACK_TO_BACK": 0.0}

    raw = (pd.Timestamp(game_date) - in_season["GAME_DATE"].iloc[-1]).days
    capped = min(raw, REST_DAYS_CAP)
    return {"REST_DAYS": float(capped),
            "IS_LONG_BREAK": float(raw > LONG_BREAK_DAYS),
            "IS_BACK_TO_BACK": float(capped == 1)}


def team_features(frame: pd.DataFrame, state: dict, team_id: int,
                  game_date: pd.Timestamp, windows) -> dict:
    """Every pre-game feature for one team, keyed by unprefixed name."""
    game_date = pd.Timestamp(game_date)
    season = season_of(game_date)
    history = team_history(frame, team_id, game_date)

    features = {}
    for window in windows:
        features.update(rolling_for(history, window, season))
    features.update(rest_for(history, game_date, season))
    features["TEAM_ELO"] = elo_for(state, team_id, season)
    return features


def season_of(game_date: pd.Timestamp) -> int:
    """Season label for one date, from the fetcher's own boundary.

    Imported lazily: fetch_wnba_games pulls in nba_api at module scope, and
    nothing in the serving path should make that a hard import just to do one
    piece of calendar arithmetic. It is the definition of the WNBA season
    boundary, so restating May here would be a second copy of it.
    """
    from fetch_wnba_games import current_season
    return int(current_season(pd.Timestamp(game_date).date()))


def feature_row(entry: dict, home: dict, away: dict) -> tuple:
    """One target's features in the manifest's declared order.

    Returns (values, missing). The order is the manifest's, not a regenerated
    one, so a column reordering in the pipeline cannot silently permute what
    is fed to a fitted pipeline - sklearn would accept the array regardless.
    """
    values, missing = [], []
    for column in entry["features"]:
        side, name = column.split("_", 1)
        source = home if side == "HOME" else away
        # NOT .get(): a feature this module never built and one that is NaN
        # because the window is short are different failures, and only the
        # second is a legitimate refusal. Collapsing them would let a manifest
        # naming a column nobody builds read as "team too early in season".
        if name not in source:
            raise KeyError(
                f"the manifest asks for {column}, which this module does not "
                f"build. Built names: {sorted(source)}")
        value = float(source[name])
        if np.isnan(value):
            missing.append(column)
        values.append(value)
    return values, missing


_STATE = None


def load_state():
    """The served table and the Elo replay, built once."""
    global _STATE
    if _STATE is None:
        manifest = load_manifest()
        frame = load_games()
        state = replay_elo(frame, manifest["elo"]["k"],
                           manifest["elo"]["carryover"])
        _STATE = (manifest, frame, state)
    return _STATE


def data_as_of() -> pd.Timestamp:
    _, frame, _ = load_state()
    return frame["GAME_DATE"].max()


def get_live_features(home_team_id: int, away_team_id: int, game_date) -> dict:
    """Per-target feature rows for one unplayed fixture.

    Raises NotScoreable if any target's features cannot be completed, naming
    the columns and the reason, because every WNBA target is a linear pipeline
    that cannot accept a NaN.
    """
    manifest, frame, state = load_state()
    windows = required_windows(manifest)

    home = team_features(frame, state, home_team_id, game_date, windows)
    away = team_features(frame, state, away_team_id, game_date, windows)

    rows, incomplete = {}, {}
    for target, entry in manifest["targets"].items():
        values, missing = feature_row(entry, home, away)
        rows[target] = values
        if missing:
            incomplete[target] = missing

    if incomplete:
        raise NotScoreable(
            "this fixture cannot be scored: "
            + "; ".join(f"{t} is missing {len(c)} feature(s) {c[:4]}"
                        for t, c in incomplete.items())
            + ". The WNBA models require a complete feature row, so a team "
              "early in its season - or in its first games as a franchise - "
              "has no window to score from yet.")

    return {"rows": rows, "home": home, "away": away,
            "season": season_of(pd.Timestamp(game_date)),
            "data_as_of": data_as_of()}
