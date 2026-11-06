r"""Serve the NHL from a snapshot: the dependency rule, the winner, the puck line.

ONE IMPLEMENTATION OF THE RULE, which is the NFL's design and the reason it
is repeated here rather than improved on. `predictable_fixtures` and the
refusal inside `get_live_features` call the SAME function, so a fixture that
`/schedule/nhl` marks `predictable: true` cannot be refused by `/predict/nhl`
and the reverse. A second rule engine for the schedule endpoint is how the
two drift.

THE PREDICTION RULE IS THE DEPENDENCY RULE AND `MAX_DAYS_AHEAD` MUST NOT BE
APPLIED. A fixture is predictable once BOTH teams' previous games are in
history. Measured on 2025-26 at an 08:30 UTC refresh: 1,292 of 1,312
predictable against the date rule's 1,242, median lead 38.5 h against 14.5 h,
and 71% of the season available the morning before against 0%. The date
rule's 0% is structural - it offers `data_as_of + 1`, which at a morning
refresh is the same calendar day.

SERVING EQUALS TRAINING BECAUSE IT CALLS THE TRAINING CODE. Every feature
comes from `nhl_features.row_features` and the Elo from
`nhl_features.elo_as_of`, which is what `verify_nhl_no_leakage` recomputes its
6,425 comparisons through. Nothing is reimplemented here - the module's job is
to decide WHICH rows to compute and to refuse the rest.

WHAT IS SERVED, from `models_nhl/manifest.json` rather than from a list here:

  winner      Elo alone. There is no artifact: the parameters ARE the model,
              because Elo won the validation selection outright. Its test
              interval against a constant INCLUDES ZERO, so the manifest's
              caveat travels with every answer.
  puck_line   home wins by 2+, `puck_line.joblib` over the three Elo columns.
  totals      NOT SERVED, with reason_code `no_skill_over_constant`.
"""
import datetime as dt
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ML_TRAINING = HERE.parent
PROJECT = ML_TRAINING.parent
for _path in (HERE, ML_TRAINING):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import nhl_features as F                                        # noqa: E402
from served_data import data_root                               # noqa: E402

MODELS_DIR = ML_TRAINING / "models_nhl"
MANIFEST_PATH = MODELS_DIR / "manifest.json"

GAMES_TABLE = Path("nhl") / "processed" / "nhl_games_final.csv"
FIXTURES_TABLE = Path("nhl") / "processed" / "nhl_fixtures.csv"
IDENTITY_TABLE = Path("nhl") / "processed" / "nhl_franchise_identity.csv"
QUARANTINE_TABLE = Path("nhl") / "processed" / "nhl_quarantine.csv"
STANDINGS_TABLE = Path("nhl") / "processed" / "nhl_standings_report.csv"

SOURCE_ATTRIBUTION = "English Wikipedia, CC BY-SA 4.0"

# Pacific is the latest zone an NHL game is played in, so a US date D has
# ended everywhere relevant at midnight Pacific on D+1 - 07:00 UTC in PDT and
# 08:00 UTC in PST.
#
# THE SAME RULE AS `parse_nhl_wikitext.date_ended_utc`, AND THE TWO ARE HELD
# TO EACH OTHER. The parser belongs to the data pipeline and this module to
# the serving image, which are different deployment units - the inference
# image copies `ml-training/nhl/` and not `data-pipeline/nhl/preprocessing/`.
# So the rule exists twice by necessity, and `verify_nhl_serving.py` asserts
# they agree across a date range spanning both DST boundaries rather than
# trusting that they do. Two copies of one rule with nothing holding them
# together is how a gate ends up grading the wrong thing.
PACIFIC = ZoneInfo("America/Los_Angeles")

# A fixture whose date has ended this long ago and still has no result is a
# MISSING RESULT, not a pending one. The NFL's analogue is its >5 h kickoff
# rule; here the finality instant is exact, so the only slack needed is for
# the editor who has not typed it in yet.
MISSING_RESULT_GRACE_HOURS = 6

STALE_AFTER_DAYS = 2


# ---------------------------------------------------------------- refusals
REFUSAL_SEASON_OPENER = "season_opener"
# A DISTINCT CODE, BECAUSE IT IS A DISTINCT SITUATION. The first version used
# `season_opener` for both "this team has never played" and "this team's
# earlier fixture has no result yet", and the second is not an opener at all -
# measured on 2026-10-31, Calgary had played twelve games and still blocked
# its 11-01 fixture because its 10-30 game was not yet recorded. A caller
# told `season_opener` about a team with twelve games would reasonably
# conclude the rule was broken.
REFUSAL_PREVIOUS_GAME_PENDING = "previous_game_pending"
REFUSAL_DATE_OUT_OF_RANGE = "date_out_of_range"
REFUSAL_UNKNOWN_TEAM = "unknown_team"
REFUSAL_SAME_TEAM = "same_team"
REFUSAL_NOT_SERVED = "nhl_not_served"
REFUSAL_UNCLASSIFIED = "unclassified"


class NotScoreable(RuntimeError):
    """A fixture this league cannot answer, with a structured code.

    THE CODE IS WHAT A CALLER KEYS ON AND THE MESSAGE IS FOR READING. The
    gate's false REGRESSED is the recorded reason those are two things: a
    human-readable reason that embeds a count turns a diff into a tripwire.
    """

    def __init__(self, code, message, **fields):
        super().__init__(message)
        self.code = code
        self.fields = fields


def date_ended_utc(date):
    """The instant date `date` has ended in the latest US zone."""
    midnight_pacific = dt.datetime.combine(
        date + dt.timedelta(days=1), dt.time(0, 0), tzinfo=PACIFIC)
    return midnight_pacific.astimezone(dt.timezone.utc)


# ------------------------------------------------------------------ state

def load_manifest(path=MANIFEST_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def is_served(root=None):
    """Whether this snapshot carries the NHL at all.

    THE TABLES ARE OPTIONAL, so "absent" is a state and not a fault - a new
    image on an old snapshot answers `not_yet_served` rather than refusing to
    boot. Checked on the GAMES table alone: without history there is nothing
    to replay and no rule to apply, so the other four cannot rescue it.
    """
    root = Path(root) if root is not None else data_root()
    return (root / GAMES_TABLE).is_file()


def load_state(root=None, manifest=None):
    """Everything serving needs, read once at boot.

    RAISES NOTHING WHEN THE NHL IS ABSENT - it returns a state that says so.
    A caller that asks for a prediction then gets `nhl_not_served`, which is
    a refusal with a code rather than a 500 from a missing file.
    """
    root = Path(root) if root is not None else data_root()
    manifest = manifest or load_manifest()
    elo = {"k": manifest["elo"]["k"],
           "home_advantage": manifest["elo"]["home_advantage"],
           "carryover": manifest["elo"]["carryover"]}

    if not (root / GAMES_TABLE).is_file():
        return {"served": False, "root": root, "manifest": manifest,
                "elo": elo, "games": None, "fixtures": None,
                "identity": {}, "quarantine": {"state": "absent"},
                "standings": [], "puck_line": None}

    games = F.load_games(root / GAMES_TABLE)
    corpus = F.as_fixtures(games)

    fixtures = None
    if (root / FIXTURES_TABLE).is_file():
        fixtures = pd.read_csv(root / FIXTURES_TABLE, low_memory=False)
        if len(fixtures):
            fixtures["GAME_DATE"] = (pd.to_datetime(fixtures["GAME_DATE"])
                                     .dt.date)
            # THE RELOCATION MAP APPLIES TO FIXTURES TOO. `load_games` does
            # it for history; a fixture naming the old franchise id would
            # otherwise look like an unknown team to a rule that reads the
            # new one.
            for column in ("HOME_ID", "AWAY_ID"):
                fixtures[column] = fixtures[column].replace(F.RELOCATIONS)

    identity = {}
    if (root / IDENTITY_TABLE).is_file():
        table = pd.read_csv(root / IDENTITY_TABLE)
        for row in table.itertuples():
            identity.setdefault(int(row.FRANCHISE_ID), []).append(
                {"abbrev": row.ABBREV,
                 "first_season": (None if pd.isna(row.FIRST_SEASON)
                                  else int(row.FIRST_SEASON)),
                 "last_season": (None if pd.isna(row.LAST_SEASON)
                                 else int(row.LAST_SEASON)),
                 "name": row.LINK_TARGET})

    puck_line = None
    target = manifest["targets"].get("puck_line") or {}
    if target.get("ships") and target.get("artifact"):
        path = MODELS_DIR / target["artifact"]
        puck_line = joblib.load(path)
        # THE FEATURE NAMES ARE VERIFIED, not assumed. The artifact is a
        # bare pipeline over a positional matrix, so a manifest whose
        # feature list drifted from the fitted order would score the right
        # numbers in the wrong columns and never say so.
        expected = target["features"]
        width = getattr(puck_line, "n_features_in_", None)
        if width is not None and width != len(expected):
            raise RuntimeError(
                f"{path.name} was fitted on {width} feature(s) and the "
                f"manifest declares {len(expected)}: {expected}. A "
                f"positional pipeline cannot detect this at predict time.")

    return {"served": True, "root": root, "manifest": manifest, "elo": elo,
            "games": games, "corpus": corpus, "fixtures": fixtures,
            "identity": identity,
            "quarantine": load_quarantine(root),
            "standings": load_standings(root),
            "puck_line": puck_line}


def load_quarantine(root):
    """The quarantine table as a reportable state.

    ABSENT IS NOT EMPTY, which is the NFL quarantine's recorded lesson: a
    snapshot built before the table existed has an UNKNOWN quarantine state,
    and reporting zero games for it would invent a fact about data nobody
    checked.
    """
    path = Path(root) / QUARANTINE_TABLE
    if not path.is_file():
        return {"state": "absent", "count": 0, "games": []}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {"state": "none", "count": 0, "games": []}
    table = pd.read_csv(path)
    return {"state": "present" if len(table) else "none",
            "count": int(len(table)),
            "games": table.to_dict("records")}


def load_standings(root):
    """Section 1's per-season standings report, or an empty list."""
    path = Path(root) / STANDINGS_TABLE
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return pd.read_csv(path).to_dict("records")


# -------------------------------------------------------- the one rule

def _team_dates(state):
    """{franchise id: sorted completed game dates}."""
    cached = state.get("_team_dates")
    if cached is None:
        cached = {}
        for team, frame in state["games"].groupby("TEAM_ID"):
            cached[int(team)] = sorted(frame["GAME_DATE"])
        state["_team_dates"] = cached
    return cached


def _rule_context(state):
    """The dependency rule's inputs, built once per loaded state.

    WHY THIS EXISTS, measured rather than assumed. `/health` reports the
    NHL's predictable count, so it walked every future fixture and asked
    `blocking_reason` about it - and `blocking_reason` rebuilt two
    snapshot-wide facts every single time. Profiled 2026-11-06 over 1,094
    future fixtures against snapshot 2026-11-05T0843:

        max(state["games"]["GAME_DATE"])   1,094 x 1.25 ms = 1.372 s
        _unplayed_before                   2,188 x 0.55 ms = 1.206 s

    That was 2.41 s of a 4.34 s `/health`, against the daily refresh's 5 s
    per-attempt timeout - so the refresh read a working service as "no
    snapshot" and rolled back a good swap.

    CACHED ON `state`, THE SAME WAY `_team_dates` ALREADY IS, and under the
    same assumption: a loaded state is immutable, and a refresh replaces the
    process rather than mutating what it holds.

    `unplayed` is a team -> sorted dates index of this team's fixtures with
    no result yet, which is what the rule actually asks for. Building it
    costs one pass over the fixtures frame instead of two boolean masks per
    side per fixture.
    """
    cached = state.get("_rule_context")
    if cached is not None:
        return cached

    played = _team_dates(state)
    games = state.get("games")
    context = {
        # ONE NUMBER FOR THE WHOLE SNAPSHOT. It was `max()` over 32,428 rows,
        # per fixture.
        "cutoff": (max(games["GAME_DATE"])
                   if games is not None and len(games) else None),
        "known": set(played),
        "unplayed": {},
    }

    fixtures = state.get("fixtures")
    if fixtures is not None and len(fixtures):
        index = {}
        for row in fixtures.itertuples():
            for team in (int(row.HOME_ID), int(row.AWAY_ID)):
                # A FIXTURE DATE THIS TEAM HAS ALREADY PLAYED IS NOT
                # PENDING, which is the test `_unplayed_before` applied.
                if row.GAME_DATE in played.get(team, ()):
                    continue
                index.setdefault(team, []).append(row.GAME_DATE)
        context["unplayed"] = {team: sorted(dates)
                               for team, dates in index.items()}

    state["_rule_context"] = context
    return context


def _unplayed_before(state, team_id, date):
    """This team's fixtures before `date` that have NO result yet.

    THE OTHER HALF OF THE DEPENDENCY RULE, and the NFL's `earlier_unplayed`
    is where this shape comes from. "Both teams' previous games are in
    history" is not "both teams have played at least once" - if a team's
    game two nights ago has no result, its previous game is NOT in history
    and the next one cannot be predicted from a complete window.

    READS THE INDEX rather than masking the fixtures frame. Same answer, and
    `verify_health_budget.py` asserts that on every fixture of the real
    corpus rather than trusting this comment.
    """
    return [d for d in _rule_context(state)["unplayed"].get(int(team_id), ())
            if d < date]


def blocking_reason(state, home_id, away_id, date):
    """None if this fixture is predictable, else (code, message, fields).

    THE SINGLE SOURCE FOR BOTH ENDPOINTS. `predictable_fixtures` asks this
    for every fixture and `get_live_features` asks it for the one it was
    given, so the schedule's flag and the prediction's refusal cannot
    disagree.
    """
    if not state.get("served"):
        return (REFUSAL_NOT_SERVED,
                "this snapshot carries no NHL tables, so the NHL is not "
                "served from it", {})
    if home_id == away_id:
        return (REFUSAL_SAME_TEAM,
                f"home and away are the same franchise ({home_id})", {})
    context = _rule_context(state)
    known = context["known"]
    for side, team in (("home", home_id), ("away", away_id)):
        if team not in known:
            return (REFUSAL_UNKNOWN_TEAM,
                    f"the {side} franchise {team} has no completed game in "
                    f"this snapshot", {"side": side, "franchise_id": team})

    cutoff = context["cutoff"]
    if date <= cutoff:
        return (REFUSAL_DATE_OUT_OF_RANGE,
                f"{date} is not after the last completed game "
                f"({cutoff}); the NHL predicts fixtures, not results it "
                f"already has",
                {"requested": str(date), "data_as_of": str(cutoff)})

    for side, team in (("home", home_id), ("away", away_id)):
        earlier = [d for d in _team_dates(state)[team] if d < date]
        if not earlier:
            return (REFUSAL_SEASON_OPENER,
                    f"the {side} side has no previous game before {date}, "
                    f"so its rolling window and rest cannot be built",
                    {"side": side, "franchise_id": team})
        pending = _unplayed_before(state, team, date)
        if pending:
            return (REFUSAL_PREVIOUS_GAME_PENDING,
                    f"the {side} side has {len(pending)} earlier fixture(s) "
                    f"with no result yet, the latest on "
                    f"{max(pending)}; its previous game is not in history",
                    {"side": side, "franchise_id": team,
                     "pending": [str(d) for d in pending]})
    return None


def predictable_fixtures(state, limit=None):
    """Every fixture the dependency rule can answer, with its reason if not.

    RETURNS EVERY FIXTURE, flagged - not just the predictable ones. The
    schedule endpoint needs the whole slate with a per-row flag, and
    returning only the answerable ones would make a caller unable to tell a
    blocked fixture from one that does not exist.
    """
    fixtures = state.get("fixtures")
    if fixtures is None or not len(fixtures):
        return []
    cutoff = _rule_context(state)["cutoff"] if state.get("served") else None
    out = []
    ordered = fixtures.sort_values(["GAME_DATE", "GAME_ID"])
    for row in ordered.itertuples():
        if cutoff is not None and row.GAME_DATE <= cutoff:
            continue
        blocked = blocking_reason(state, int(row.HOME_ID), int(row.AWAY_ID),
                                  row.GAME_DATE)
        out.append({
            "game_id": row.GAME_ID,
            "season": row.SEASON,
            "game_date": str(row.GAME_DATE),
            "home_id": int(row.HOME_ID),
            "away_id": int(row.AWAY_ID),
            "home": team_name(state, int(row.HOME_ID), row.SEASON),
            "away": team_name(state, int(row.AWAY_ID), row.SEASON),
            "predictable": blocked is None,
            "reason_code": None if blocked is None else blocked[0],
            "reason": None if blocked is None else blocked[1],
        })
        if limit is not None and len(out) >= limit:
            break
    return out


def team_name(state, franchise_id, season=None):
    """The franchise's name as of `season`, or its latest known one."""
    spans = state["identity"].get(int(franchise_id))
    if not spans:
        return None
    if season is None:
        return spans[-1]["name"]
    start = int(str(season)[:4])
    for span in spans:
        first = span["first_season"]
        last = span["last_season"]
        if (first is None or start >= first) and (last is None
                                                  or start <= last):
            return span["name"]
    return spans[-1]["name"]


def team_abbrev(state, franchise_id, season=None):
    spans = state["identity"].get(int(franchise_id))
    if not spans:
        return None
    if season is None:
        return spans[-1]["abbrev"]
    start = int(str(season)[:4])
    for span in spans:
        first, last = span["first_season"], span["last_season"]
        if (first is None or start >= first) and (last is None
                                                  or start <= last):
            return span["abbrev"]
    return spans[-1]["abbrev"]


# ----------------------------------------------------------- predictions

def season_for(state, date):
    """Which season label a date belongs to, from the corpus's own labels.

    DERIVED, NOT COMPUTED FROM A MONTH. `nhl_features` takes SEASON from the
    corpus rather than from a date rule, and a second rule here could
    disagree with it at exactly the boundary that matters. A fixture carries
    its own season label, so this is only the fallback for a constructed
    request.
    """
    fixtures = state.get("fixtures")
    if fixtures is not None and len(fixtures):
        match = fixtures[fixtures["GAME_DATE"] == date]
        if len(match):
            return match.iloc[0]["SEASON"]
    labels = F.season_order(state["games"])
    return labels[-1] if labels else None


def get_live_features(home_id, away_id, game_date, state):
    """One fixture's features and predictions, or NotScoreable with a code."""
    if isinstance(game_date, (pd.Timestamp, dt.datetime)):
        game_date = game_date.date()
    home_id, away_id = int(home_id), int(away_id)

    blocked = blocking_reason(state, home_id, away_id, game_date)
    if blocked is not None:
        code, message, fields = blocked
        raise NotScoreable(code, message, **fields)

    season = season_for(state, game_date)
    elo = state["elo"]
    ratings = F.elo_as_of(state["corpus"], game_date, season, elo["k"],
                          elo["home_advantage"], elo["carryover"])
    home_elo = ratings.get(home_id, F.ELO_BASELINE)
    away_elo = ratings.get(away_id, F.ELO_BASELINE)

    history = F.History(state["games"][
        state["games"]["GAME_DATE"] < game_date])
    row = F.row_features(history, home_id, away_id, game_date, season,
                         home_elo, away_elo, carry_rolling=True,
                         carry_rest=True)

    manifest = state["manifest"]
    predictions = {}
    markets = {}

    # ---- winner: Elo alone, no artifact -----------------------------
    winner = manifest["targets"]["winner"]
    probability = float(F.elo_win_probability([home_elo], [away_elo],
                                             elo["home_advantage"])[0])
    predictions["home_win_probability"] = probability
    markets["winner"] = {
        "served": True, "family": winner["family"],
        "caveat": winner.get("caveat", {}).get("short"),
    }

    # ---- puck line --------------------------------------------------
    puck = manifest["targets"].get("puck_line") or {}
    if state.get("puck_line") is not None:
        matrix = np.array([[row[name] for name in puck["features"]]],
                          dtype=float)
        predictions["home_wins_by_2_or_more"] = float(
            state["puck_line"].predict_proba(matrix)[0, 1])
        markets["puck_line"] = {"served": True, "family": puck["family"],
                                "definition": puck["label_definition"]}
    else:
        markets["puck_line"] = {"served": False,
                                "reason_code": "artifact_not_loaded"}

    # ---- totals: declared unserved ----------------------------------
    totals = manifest["targets"]["totals"]
    shape = totals.get("health_shape_for_phase_3") or {}
    markets["totals"] = {"served": False,
                         "reason_code": shape.get("reason_code"),
                         "detail": shape.get("detail")}

    return {
        "league": "nhl",
        "game_date": str(game_date),
        "season": season,
        "home_id": home_id, "away_id": away_id,
        "home": team_name(state, home_id, season),
        "away": team_name(state, away_id, season),
        "features": row,
        "predictions": predictions,
        "markets": markets,
        "source": SOURCE_ATTRIBUTION,
    }


# ------------------------------------------------------------ freshness

def missing_results(state, now=None):
    """Fixtures whose date has ENDED and which still carry no result.

    THE NHL'S ANALOGUE OF THE NFL's >5 h KICKOFF RULE, and the finality
    instant is exact here rather than approximate: a game's US date has
    ended at midnight Pacific on D+1. A grace period follows, because the
    editor who types the score in is not instantaneous.
    """
    if not state.get("served"):
        return []
    fixtures = state.get("fixtures")
    if fixtures is None or not len(fixtures):
        return []
    now = now or dt.datetime.now(dt.timezone.utc)
    grace = dt.timedelta(hours=MISSING_RESULT_GRACE_HOURS)
    played = set(state["corpus"]["GAME_ID"])
    out = []
    for row in fixtures.itertuples():
        if row.GAME_ID in played:
            continue
        if date_ended_utc(row.GAME_DATE) + grace > now:
            continue
        out.append({"game_id": row.GAME_ID,
                    "game_date": str(row.GAME_DATE),
                    "home_id": int(row.HOME_ID),
                    "away_id": int(row.AWAY_ID)})
    return out


def freshness(state, now=None):
    """(data_as_of, days_behind, stale) for the NHL."""
    if not state.get("served"):
        return None, None, None
    now = now or dt.datetime.now(dt.timezone.utc)
    cutoff = max(state["games"]["GAME_DATE"])
    days_behind = (now.date() - cutoff).days
    return cutoff, days_behind, days_behind > STALE_AFTER_DAYS


def standings_class(state):
    """The open season's standings class, or None.

    READS THE REPORT RATHER THAN RECOMPUTING IT. Section 1's classifier runs
    in the build, where it has the game logs; serving has the conclusion.
    """
    for row in state.get("standings") or []:
        if str(row.get("STATE", "")).upper() == "OPEN":
            return {"season": row.get("SEASON"),
                    "class": row.get("CLASS"),
                    "classes": str(row.get("CLASSES", "")).split("|"),
                    "detail": row.get("DETAIL"),
                    "conflicts": int(row.get("CONFLICTS") or 0),
                    "teams_flagged": int(row.get("TEAMS_FLAGGED") or 0)}
    return None
