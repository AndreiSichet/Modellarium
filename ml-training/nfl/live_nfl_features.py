"""Serving features for one NFL fixture, from the served history.

THE PHASE 2 FUNCTION, NOT A COPY. `features_for` is imported from
`build_nfl_model_dataset` and called exactly as the training walk called it:
with a history holding only games strictly before the fixture. There is no
second feature implementation in this repository for this league, which is
what makes "served equals trained" checkable rather than hoped for.

ELO IS REPLAYED, NEVER READ. The dataset carries HOME_ELO and HOME_ELO_MOV
columns, and they are the WRONG numbers for serving: phase 2 fitted
home_advantage=50 on training seasons only, while the shipped artifacts were
fitted under home_advantage=35 on everything through 2025. A precomputed
column sitting beside parameters it does not agree with is a drift surface
with no guard on it - which is why the WNBA and G League serving paths replay
too, and here the two fits demonstrably differ rather than merely might.

Replaying is affordable: measured at 15 ms for all 3,727 games, so a request
rebuilds the ledger from scratch rather than maintaining a cache whose
staleness would be a second thing to get wrong.

THE PREDICTION RULE IS A DEPENDENCY, NOT A DATE. The other three leagues serve
`MAX_DAYS_AHEAD = 1`, a rule about dates. Every feature here reads only each
team's own earlier games, and Elo updates only on games already played, so:

    a fixture is predictable once both teams' previous games are in history

A Sunday fixture's features are final as soon as both sides' prior games are
recorded, whatever happens on the Thursday in between. Phase 2 measured 15 of
208 unplayed 2026 fixtures predictable under this rule - a whole week's slate
at once, where a date rule yields a single day.

AND AN INCOMPLETE FEATURE ROW IS NOT REFUSED HERE, which is the opposite of
the WNBA and the G League. Their linear Pipelines carry no imputer, so a NaN
cannot be scored at all and the endpoint answers 400. The NFL's two linear
artifacts carry a `SimpleImputer` fitted on training rows only, with
`SEASON_OPENER` riding along as the indicator that the value was supplied
rather than observed - so a season opener, whose `REST_DAYS` is NaN because an
offseason is not rest, scores exactly as it did in training. `NotScoreable` is
raised for three things only: no such fixture, a date that is not the
scheduled one, and the dependency rule. Missing features are REPORTED in the
returned `missing` block rather than raised on, so a caller can see the row
was imputed without being denied the prediction the model was trained to give.
"""

import datetime
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ML_TRAINING = HERE.parent
PROJECT = ML_TRAINING.parent
NFL_PREPROCESSING = PROJECT / "data-pipeline" / "nfl" / "preprocessing"

for _directory in (ML_TRAINING, NFL_PREPROCESSING):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import pandas as pd  # noqa: E402

import build_nfl_model_dataset as B  # noqa: E402
import nfl_franchises as F  # noqa: E402
from parse_nfl_wikitext import ZONE_OVERRIDE, ZONES, kickoff_utc  # noqa: E402
from served_data import data_root  # noqa: E402

MODELS_DIR = ML_TRAINING / "models_nfl"
MANIFEST_PATH = MODELS_DIR / "manifest.json"

# Relative to the data root, so one DATA_DIR points at either the repo or a
# snapshot. Mirrors the layout `served_data.REQUIRED_FILES` declares.
GAMES_TABLE = Path("nfl") / "processed" / "nfl_games_final.csv"
FIXTURES_TABLE = Path("nfl") / "processed" / "nfl_fixtures.csv"
IDENTITY_TABLE = Path("nfl") / "processed" / "nfl_franchise_identity.csv"

# A synthetic table set exists for CI, because the real one is CC BY-SA and is
# never committed. Production must refuse it; see `refuse_synthetic`.
SYNTHETIC_MARKER = "nfl_synthetic.json"
ALLOW_SYNTHETIC_ENV = "MODELLARIUM_ALLOW_SYNTHETIC_NFL"

SOURCE_ATTRIBUTION = "English Wikipedia, CC BY-SA 4.0"


class NotScoreable(RuntimeError):
    """The fixture cannot be scored, with the reason naming what is missing.

    An exception rather than a row of NaN, for the reason `NoReportAvailable`
    is one: a None gets treated as "empty, so nothing is wrong", and two of
    the three shipped models are linear Pipelines that cannot take a NaN at
    all.
    """


class SyntheticDataRefused(RuntimeError):
    """A synthetic NFL table set reached a production boot."""


def refuse_synthetic(root: Path) -> dict:
    """Refuse a synthetic NFL table set unless CI has explicitly asked for it.

    THE REAL TABLES CANNOT BE COMMITTED - they are CC BY-SA and this
    repository is not licensed to redistribute a derived database - and CI
    boots the inference container against the committed data copy. So CI gets
    a deterministic synthetic set instead, and the guard runs in the
    direction that matters: production refuses it.

    The marker is a FILE beside the tables rather than a column inside one,
    because a column would have to be read, trusted and kept out of every
    feature list; a file is checked once at boot and cannot reach a model.
    """
    marker = root / "nfl" / "processed" / SYNTHETIC_MARKER
    if not marker.is_file():
        return {}

    import os

    meta = json.loads(marker.read_text(encoding="utf-8"))
    if os.environ.get(ALLOW_SYNTHETIC_ENV) == "1":
        return meta

    raise SyntheticDataRefused(
        f"{marker} marks this NFL table set as SYNTHETIC "
        f"({meta.get('generated_by', 'unknown generator')}), and "
        f"{ALLOW_SYNTHETIC_ENV} is not set to 1. Refusing to serve invented "
        f"scores as if they were results. Set that variable only in CI, where "
        f"the real tables cannot exist because they are CC BY-SA and are "
        f"never committed."
    )


def production_elo_configs(manifest: dict) -> dict:
    """The shipped Elo parameters, from the manifest rather than restated."""
    return {
        name: B.EloParams(
            k=entry["k"],
            carryover=entry["carryover"],
            home_advantage=entry["home_advantage"],
            mov=entry["margin_of_victory_multiplier"],
        )
        for name, entry in manifest["elo"].items()
    }


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    if not path.is_file():
        raise RuntimeError(
            f"{path} is missing. It carries the per-market feature lists and "
            f"the production Elo parameters, and an NFL model cannot be "
            f"served without them."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_state(root: Path = None, manifest: dict = None) -> dict:
    """Everything serving needs, read once at boot."""
    root = root or data_root()
    synthetic = refuse_synthetic(root)
    manifest = manifest or load_manifest()

    games = B.load_games(path=root / GAMES_TABLE)
    fixtures = pd.read_csv(root / FIXTURES_TABLE, parse_dates=["date"])
    identity = pd.read_csv(root / IDENTITY_TABLE)

    configs = production_elo_configs(manifest)
    data_as_of = max(game["date"] for game in games).date()

    return {
        "root": root,
        "manifest": manifest,
        "elo_configs": configs,
        "games": games,
        "fixtures": fixtures,
        "identity": identity,
        "data_as_of": pd.Timestamp(data_as_of),
        "known_team_ids": set(int(g) for g in identity["franchise_id"].unique()),
        "synthetic": synthetic,
    }


# =========================================================================
# THE DEPENDENCY RULE
# =========================================================================

def _ordered_fixtures(state: dict) -> pd.DataFrame:
    return state["fixtures"].sort_values(["date", "game_id"])


def earlier_unplayed(state: dict, franchise_id: int, before_key) -> list:
    """This team's unplayed fixtures that sit before `before_key`.

    `before_key` is the (date, game_id) tuple of the fixture being asked
    about, so the ordering is the same total order the training walk used -
    two fixtures on one date are broken by game id rather than left to the
    frame's row order.
    """
    out = []
    for _index, row in _ordered_fixtures(state).iterrows():
        if franchise_id not in (int(row["home_franchise_id"]),
                                int(row["away_franchise_id"])):
            continue
        key = (pd.Timestamp(row["date"]), str(row["game_id"]))
        if key < before_key:
            out.append(row)
    return out


def predictable_fixtures(state: dict) -> list:
    """Every unplayed fixture whose features are already final.

    Returned in schedule order, each with the pair and the date a request must
    use. `/schedule/nfl` reports this as a flag per fixture, and the gate
    derives its matchups from it rather than from `cutoff + 1 day`, which is
    not what this league's rule says.
    """
    out = []
    for _index, row in _ordered_fixtures(state).iterrows():
        key = (pd.Timestamp(row["date"]), str(row["game_id"]))
        blocked = []
        for side, column in (("home", "home_franchise_id"),
                             ("away", "away_franchise_id")):
            earlier = earlier_unplayed(state, int(row[column]), key)
            if earlier:
                blocked.append((side, earlier[0]))
        if not blocked:
            out.append(row)
    return out


def _blocking_reason(state: dict, fixture_row, key) -> str:
    """Which side's previous game is missing, named."""
    parts = []
    for side, column, abbr_column in (("home", "home_franchise_id", "home"),
                                      ("away", "away_franchise_id", "away")):
        earlier = earlier_unplayed(state, int(fixture_row[column]), key)
        if earlier:
            first = earlier[0]
            parts.append(
                f"{fixture_row[abbr_column]} (the {side} side) has not yet "
                f"played its week {int(first['week'])} fixture "
                f"{first['game_id']} on {pd.Timestamp(first['date']).date()}"
            )
    return "; ".join(parts)


# =========================================================================
# FEATURES
# =========================================================================

def _resolve_fixture(state: dict, home_id: int, away_id: int,
                     game_date: pd.Timestamp) -> tuple:
    """Find the fixture or the played game being asked about.

    Returns (fixture_dict, kind, row) where kind is "fixture" or "played".

    THE DATE IS PART OF THE KEY, AND THE FIRST VERSION LEFT IT OUT. Matching
    the fixtures table on the team pair alone looked sufficient - there is at
    most one unplayed fixture per ordered pair in a season - and it is wrong
    across seasons: the same two teams meet most years, so a request for a
    played 2014 game found the 2026 fixture for that pair and was refused with
    "scheduled for 2027-01-09, not 2014-09-11". The verifier's 25-game sample
    caught it on its first run. Played games are therefore resolved FIRST and
    on all three of (home, away, date); the pair-only lookup survives only to
    produce a useful message when a caller has the date wrong.

    A REQUEST FOR A GAME ALREADY PLAYED IS ANSWERED, NOT REFUSED, which is
    what the other three leagues do: their validators check the date is not
    beyond cutoff + 1 and never check that it is in the future, so a past
    matchup is re-scored from the history that preceded it. Doing the same
    here keeps one behaviour across four leagues, and it is also what makes
    the 25-game equality check in section 1.2 possible at all.
    """
    requested = pd.Timestamp(game_date).date()

    for game in state["games"]:
        if (game["home_franchise_id"] == home_id
                and game["away_franchise_id"] == away_id
                and game["date"].date() == requested):
            return (game, "played", None)

    unplayed = state["fixtures"]
    pair = unplayed[(unplayed.home_franchise_id == home_id)
                    & (unplayed.away_franchise_id == away_id)]
    exact = pair[pd.to_datetime(pair.date).dt.date == requested]

    if len(exact) == 1:
        row = exact.iloc[0]
        return ({
            "season": int(row["season"]),
            "week": int(row["week"]),
            "date": pd.Timestamp(row["date"]).to_pydatetime(),
            "neutral_site": int(row["neutral_site"]),
            "home_franchise_id": home_id,
            "away_franchise_id": away_id,
            "game_id": str(row["game_id"]),
        }, "fixture", row)

    if len(exact) > 1:
        raise NotScoreable(
            f"{len(exact)} unplayed fixtures match this pair on "
            f"{requested}, so the one being asked about is ambiguous: "
            f"{sorted(exact.game_id)}."
        )

    if len(pair):
        row = pair.iloc[0]
        scheduled = pd.Timestamp(row["date"]).date()
        raise NotScoreable(
            f"{row['away']} at {row['home']} is scheduled for {scheduled}, "
            f"not {requested}. REST_DAYS is computed from this date, so a "
            f"request must use the scheduled one."
            + (" This fixture is flexed and carries two candidate dates; the "
               "earlier is the one served."
               if int(row["flex_two_dates"]) else "")
        )

    raise NotScoreable(
        f"No scheduled fixture and no played game for franchise {home_id} at "
        f"home against {away_id} on {requested}. The NFL is served from its "
        f"own schedule rather than from any date the caller supplies, because "
        f"WEEK and NEUTRAL_SITE are features and neither can be derived from "
        f"a date."
    )


def _history_before(state: dict, fixture: dict) -> B.NflHistory:
    """Replay every played game strictly before this fixture.

    The same total order the training walk used - (date, game_id) - because
    `load_games` sorts on it and two games on one date would otherwise enter
    the ledger in whichever order the frame happened to hold.
    """
    key = (pd.Timestamp(fixture["date"]), str(fixture.get("game_id", "")))
    history = B.NflHistory(state["elo_configs"])
    for game in state["games"]:
        game_key = (pd.Timestamp(game["date"]), str(game["game_id"]))
        if game_key >= key:
            break
        history.add(game)
    return history


def get_live_features(home_id: int, away_id: int, game_date, state: dict) -> dict:
    """The per-market feature rows for one fixture, plus its context."""
    game_date = pd.Timestamp(game_date)
    fixture, kind, row = _resolve_fixture(state, home_id, away_id, game_date)

    if kind == "fixture":
        key = (pd.Timestamp(row["date"]), str(row["game_id"]))
        reason = _blocking_reason(state, row, key)
        if reason:
            raise NotScoreable(
                f"Not predictable yet: {reason}. An NFL fixture becomes "
                f"predictable once both teams' previous games are in history, "
                f"which is a dependency rather than a date - every feature "
                f"reads only each team's own earlier games."
            )

    history = _history_before(state, fixture)
    full = B.features_for(fixture, history)

    rows, missing = {}, {}
    for market, entry in state["manifest"]["markets"].items():
        values = {name: full[name] for name in entry["features"]}
        rows[market] = values
        absent = [name for name, value in values.items()
                  if value != value]  # NaN
        if absent:
            missing[market] = absent

    return {
        "rows": rows,
        "full": full,
        "season": int(fixture["season"]),
        "week": int(fixture["week"]),
        "neutral_site": int(fixture["neutral_site"]),
        "kind": kind,
        "history_games": history.count,
        "missing": missing,
    }


# =========================================================================
# THE SCHEDULE, AND THE KICKOFF CONVERSION
# =========================================================================

def _link_target(state: dict, franchise_id: int, season: int):
    """The season article this franchise's identity is keyed on.

    Needed because the Arizona exception is keyed on the link target in the
    parser, and that is the one place the exception should live.
    """
    table = F.franchise_table(season)
    for target, (fid, _abbr, _name) in table.items():
        if fid == franchise_id:
            return target
    return None


def kickoff_for(state: dict, row) -> tuple:
    """(iso 8601 UTC or None, the IANA zone used or None).

    THE CONVERSION IS THE PARSER'S, NOT A SECOND IMPLEMENTATION. `kickoff_utc`
    already carries the zone map and the Arizona exception, and it is the
    function that produced the five-hour in-progress guard phase 1 relies on.
    A copy here would be a second place for "Mountain means Denver except in
    Phoenix" to be got wrong.

    Arizona is the case worth naming: it does not observe daylight saving, and
    its own article still labels the column `Mountain Time Zone`. So the zone
    as written is not sufficient on its own and the home franchise decides.
    """
    zone_written = row.get("kickoff_zone_as_written")
    local = str(row.get("kickoff_local") or "")
    if "TBD" in local.upper():
        return None, None

    season = int(row["season"])
    target = _link_target(state, int(row["home_franchise_id"]), season)
    iana = ZONE_OVERRIDE.get(target) or ZONES.get(zone_written or "")
    moment = kickoff_utc(pd.Timestamp(row["date"]).date(), local,
                         zone_written, target)
    if moment is None:
        return None, iana
    return moment.astimezone(datetime.timezone.utc).isoformat(), iana


def upcoming(state: dict) -> list:
    """Unplayed fixtures, with the kickoff in UTC and the predictable flag."""
    predictable_ids = {str(row["game_id"]) for row in predictable_fixtures(state)}
    out = []
    for _index, row in _ordered_fixtures(state).iterrows():
        kickoff, zone = kickoff_for(state, row)
        out.append({
            "game_id": str(row["game_id"]),
            "season": int(row["season"]),
            "week": int(row["week"]),
            "game_date": pd.Timestamp(row["date"]).date(),
            "home_team_id": int(row["home_franchise_id"]),
            "away_team_id": int(row["away_franchise_id"]),
            "home": str(row["home"]),
            "away": str(row["away"]),
            "kickoff_utc": kickoff,
            "kickoff_zone": zone,
            # The source writes two dates for a flexed fixture and the earlier
            # one is served; reported rather than hidden, because a client
            # showing a date the league may still move should be able to say so.
            "flex": bool(int(row["flex_two_dates"])),
            "neutral_site": bool(int(row["neutral_site"])),
            "predictable": str(row["game_id"]) in predictable_ids,
            "venue": str(row["venue"]),
        })
    return out


def freshness(state: dict, today=None) -> tuple:
    today = pd.Timestamp(today or datetime.date.today())
    days_behind = (today - state["data_as_of"]).days
    return days_behind, days_behind


if __name__ == "__main__":
    state = load_state()
    print(f"root           {state['root']}")
    print(f"games          {len(state['games']):,}")
    print(f"fixtures       {len(state['fixtures'])}")
    print(f"data_as_of     {state['data_as_of'].date()}")
    print(f"teams          {len(state['known_team_ids'])}")
    ready = predictable_fixtures(state)
    print(f"predictable    {len(ready)} of {len(state['fixtures'])}")
    for row in ready:
        print(f"  week {int(row['week']):>2}  {row['away']:>3} at "
              f"{row['home']:<3}  {pd.Timestamp(row['date']).date()}")
