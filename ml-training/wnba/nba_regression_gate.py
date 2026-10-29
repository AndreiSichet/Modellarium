"""The regression gate: capture a context and bodies, then compare two captures.

A PAIRED INSTRUMENT, WHICH IS WHAT IT ALWAYS WAS AND NEVER SAID. Capture,
change code, re-check - minutes apart. It was used instead as a baseline that
persisted for weeks, and every time it went red on the clock rather than on the
code that mismatch was the cause: `daysBehind` twice, `schedule_count` once,
and - had it not been rewritten - `dataAsOf`, `stale` and the predictions
themselves on 21 October, when the data advances and injury reports resume.

So a capture now records the CONTEXT it was produced under, and a comparison
whose context differs returns CANNOT COMPARE instead of inventing a regression.

THE FOUR VERDICTS
    PASS            context identical, bodies identical          exit 0
    REGRESSED       context identical, bodies differ             exit 1
    ERROR           the gate itself could not run                exit 2
    CANNOT COMPARE  context differs - not attributable           exit 3

The verdict is per league, so a new NBA injury report cannot hide a real WNBA
comparison.

WHAT REPLACED THE PINNED NUMBER. The NBA reference probability used to be
asserted live. No fixed number can survive the data advancing, so it is kept
below as a dated record rather than a check: the bodies are compared against
another capture, not against a literal.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ML_TRAINING = HERE.parent
REPO = ML_TRAINING.parent

# Overridable so a rehearsal can point at a second stack serving a copy of the
# volume, which is how the data-advanced and mutated-snapshot cases are produced
# without touching what is actually being served.
BACKEND = os.environ.get("GATE_BACKEND", "http://localhost:8080")
INFERENCE = os.environ.get("GATE_INFERENCE", "http://localhost:8000")

# A DATED RECORD, NOT A CHECK, AND NOTHING IN THIS FILE READS IT.
#
# Checked 2026-10-29: `HISTORICAL_NBA_REFERENCE` appears exactly once in the
# whole repository - this assignment. No comparison uses it, so the withdrawn
# value it held until today was never grading anything; it was a number
# recorded under a name. Which also means "derive it from the same source the
# gate uses" has no source to point at: this gate is pairwise by design and
# has no live reference to compare against. So it stays a record, updated.
#
# WHAT IT RECORDS NOW. 0.48472610116004944 is the NBA reference probability
# for the gate's own first body - ATL (1610612737) vs BOS (1610612738), a
# CONSTRUCTED matchup from MATCHUPS below rather than a scheduled game - for
# game_date 2026-10-25 under data_as_of 2026-10-24 and snapshot
# 2026-10-25T0945, from the 34-feature models A3 shipped on 2026-10-25, with
# the injury report pinned to INJURY_LIVE_AS_OF=2026-10-25T05:00:00.
#
# WHY THE OLD ONE COULD NOT BE DEFENDED, kept here because the succession is
# the useful part: A3 dropped the four post-game availability features
# (38 -> 34) and derived SEASON from the GAME_ID rather than the date, which
# moved TEAM_ELO/OPPONENT_ELO on 14,592 rows. All seven team models are new
# weights on a new feature history, so the old value cannot be reproduced by
# the shipped models and is not a check that failed.
#
#   0.4191701412200928   ATL vs BOS 2026-04-13, snapshot 2026-10-06T2018,
#                        post-A7 38-feature models      WITHDRAWN 2026-10-25
#   0.42541608214378357  the 38-feature retrain of 2026-08-26   SUPERSEDED
#
# Not recorded with PowerShell's `-f` operator, ever: "{0}" -f $v prints 15
# significant digits and the result does not read back as the same double.
# Use .ToString('G17').
HISTORICAL_NBA_REFERENCE = 0.48472610116004944
HISTORICAL_NBA_REFERENCE_FIXTURE = (1610612737, 1610612738)
HISTORICAL_NBA_REFERENCE_DATE = "2026-10-25"
HISTORICAL_NBA_REFERENCE_SNAPSHOT = "2026-10-25T0945"

# TWO MATCHUPS PER LEAGUE, FOUR DISTINCT TEAMS. One was demonstrably blind:
# A7 corrected 135 games, and on 2026-04-13 not one of the 30 teams had a
# corrected game inside its trailing ten, so the single pinned matchup could not
# have told the corrected tables from the old ones. Two is better and is NOT a
# guarantee - nothing here establishes that two cover what one missed.
#
# Teams are fixed; only the DATE is derived, from each league's own cutoff.
MATCHUPS = {
    "nba": [(1610612737, 1610612738), (1610612744, 1610612747)],
    "wnba": [(1611661317, 1611661319), (1611661320, 1611661313)],
    # ADDED HERE. The G League has been verified by hand on every change since
    # phase 4 and was never in the gate, which is the one league whose serving
    # had no automated guard at all.
    "gleague": [(1612709919, 1612709923), (1612709890, 1612709905)],
    # THE NFL'S ARE DERIVED AT CAPTURE TIME AND CANNOT BE FIXED HERE. The other
    # three keep fixed teams and derive only the DATE, because any pair of
    # their teams is predictable on cutoff + 1. The NFL is served under a
    # dependency rule - both teams' previous games in history - so which pairs
    # are predictable changes every week, and 193 of 208 fixtures are refused
    # at any moment. A fixed pair here would go stale within days and the gate
    # would report a 400 as a regression.
    #
    # So `nfl` maps to None and `nfl_matchups()` asks the service, taking two
    # predictable fixtures with four distinct teams - the same "two matchups,
    # four teams" shape the other three have, for the reason recorded above it.
    "nfl": None,
}

PREDICT_PATH = {"nba": "/predict", "wnba": "/predict/wnba",
                "gleague": "/predict/gleague", "nfl": "/predict/nfl"}
BACKEND_PATH = {"nba": "/api/predictions", "wnba": "/api/predictions/wnba",
                "gleague": "/api/predictions/gleague",
                "nfl": "/api/predictions/nfl"}

# Per-call identifiers and timestamps. NOT dataAsOf and NOT stale: those are
# CONTEXT - they are exactly what tells you a baseline came from different data,
# so hiding them would restore the blindness this rewrite removes.
VOLATILE = {"predictedAt", "id", "gameId", "predicted_at"}

# Derived from today and so never equal across a date boundary. Reported with
# the context rather than diffed.
CONTEXT_ONLY = {"daysBehind", "days_behind"}

MIN_SCHEDULE_COUNT = 100

# Fields of the two health bodies that ARE the context, and so are compared as
# context rather than byte-for-byte here. Leaving them in would double-count: a
# table change would be reported once per league as CANNOT COMPARE and again as
# a health-body regression, and the second reading is simply wrong. What remains
# compared is the part the health check exists for - a field lost or retyped,
# which is how models_loaded becoming an int took the browse view down.
HEALTH_CONTEXT_FIELDS = {
    "availability", "served_data",
    # THE REFRESH BLOCK, WHICH MOVES ON EVERY CLOCK TICK. `hours_since` is a
    # float measured against now, so two captures minutes apart differ by
    # construction - diffing it would make the gate red on literally every
    # run, which is the loudest possible version of red-on-the-clock.
    #
    # STRIPPING IT HERE IS ONLY HALF THE JOB, and an earlier version of this
    # comment got that wrong: it said a change in this block would make the
    # league CANNOT COMPARE, but membership here removes a field from the BODY
    # diff and compares nothing. The block's IDENTITY is carried into the
    # per-league context by `refresh_identity`, which is what actually
    # produces that verdict; the clock half stays ignored.
    "refresh",
    "data_as_of", "dataAsOf", "stale",
    # THE BACKEND'S SCHEDULE-SYNC BLOCK, which carries two wall-clock stamps
    # and a per-league detail string. `lastAttemptAt` moves on every container
    # start and every six-hourly tick, so diffing it would make the gate red
    # whenever the pair straddles a restart - which is exactly what a before /
    # after pair around a rebuild IS. Classified rather than toleranced, for
    # the same reason the refresh block above is: the clock is not the code.
    #
    # Stripped rather than carried into the per-league context because it is
    # not per-league identity and it says nothing about which snapshot or
    # which models produced a body. What a lost field would break is the
    # backend's own shape, and _missing_or_changed still guards every field
    # that is NOT in this set.
    "scheduleSync",
    # THE NFL'S TWO MOVING COUNTS. `predictable_fixtures` falls and `fixtures`
    # falls as the season is played, both as a function of the data rather than
    # of any code - so they are context in exactly the way data_as_of is, and
    # diffing them would report a played game as a regression. They are still
    # compared as context, where a change makes the league CANNOT COMPARE, so
    # nothing here is unguarded.
    "predictable_fixtures", "predictableFixtures", "fixtures",
    # AND ITS MISSING-RESULT REPORT, for the same reason one step on. The
    # NFL's staleness is "a finished game has no result", so between a Sunday
    # kickoff and the next morning's refresh these are legitimately non-zero
    # and then legitimately zero again. Diffing them would make the gate go
    # red on a Sunday evening and green on Monday with no code having changed
    # - the fourth time the clock would have moved under this gate, after
    # daysBehind twice and schedule_count once.
    "missing_results", "missing_result_ids",
    # AND THE QUARANTINE, which moves on the same clock as the two above and
    # for a stronger reason: a game enters it when a Wikipedia editor saves
    # half an update and leaves it when they save the other half, neither of
    # which is a code change. CLASSIFIED, NOT TOLERANCED - the gate's own rule
    # is that a field that moves on the clock is named as context, so a change
    # makes the league CANNOT COMPARE rather than being absorbed by a
    # threshold. That is the instrument's fifth clock-not-code field.
    #
    # NOTE the shape under this key is also compared as context, which costs
    # the one thing a byte diff would have given: `state` flipping to
    # "unknown" - a snapshot predating the table - is seen as context too. The
    # boot assertion in CI is what covers that, not this.
    "quarantine",
}


def curl(method, url, body=None):
    """Bytes, decoded as UTF-8 explicitly.

    NOT text=True: subprocess would decode with the Windows ANSI codepage, and
    the player-props body carries names like 'Vit Krejci' with diacritics. That
    is the same trap CLAUDE.md s25 records, where Invoke-WebRequest decoded a
    UTF-8 body as Latin-1 and double-encoded it.
    """
    cmd = ["curl.exe", "-s", "-X", method, url]
    if body is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(body)]
    done = subprocess.run(cmd, capture_output=True)
    raw = done.stdout.decode("utf-8", errors="strict")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"__raw__": raw[:400]}


def refresh_identity(health: dict) -> dict:
    """The refresh block reduced to what identifies the RUN, not the clock.

    Returns {} when the field is absent, which is the pre-upgrade shape - an
    old image does not send it - so two captures across that boundary compare
    {} against a real dict and the league withholds judgement rather than
    silently passing.

    THE CLOCK FIELDS ARE LEFT OUT BY NAME. `hours_since` ticks, and `state`
    (ok/late/overdue) and `note` are both derived from it, so any two captures
    minutes apart differ in all three while describing the same run.
    """
    block = (health or {}).get("refresh")
    if not isinstance(block, dict):
        return {}
    return {
        "last_verdict_at": block.get("last_verdict_at"),
        "outcome": block.get("outcome"),
        "leagues": block.get("leagues"),
    }
    # `snapshot` IS DELIBERATELY NOT IN HERE, and it was until 2026-10-24.
    #
    # Including it made the refresh identity fire whenever the POINTER moved,
    # whether or not a refresh had run - and moving the pointer by hand is
    # exactly what a deploy does. A constructed pair with the snapshot id
    # changed and every league table identical came out CANNOT COMPARE for
    # the WNBA, the G League and the NFL, with the reason "a refresh ran
    # between the captures (advanced at t -> advanced at t)": the same
    # outcome at the same timestamp, which is the tell that nothing had run.
    #
    # The snapshot id is ALREADY judged, and judged better, by the table
    # rule: different id with identical hashes is comparable and says the id
    # moved. Carrying it here as well double-counted it and let the weaker
    # reading win. `last_verdict_at` still catches a real refresh, because a
    # refresh that ran cannot leave that timestamp unchanged.


def strip_volatile(value):
    """Recursively drop per-call fields so the rest can be compared exactly."""
    if isinstance(value, dict):
        return {k: strip_volatile(v) for k, v in value.items()
                if k not in VOLATILE and k not in CONTEXT_ONLY}
    if isinstance(value, list):
        return [strip_volatile(v) for v in value]
    return value


def model_hashes() -> dict:
    """sha256 per shipped model artifact, per league.

    FROM THE REPOSITORY, NOT FROM THE RUNNING IMAGE, and the distinction is
    deliberate. These are the files a build would bake in, so they answer "did
    the models change between these two captures". Hashing what the container
    holds would instead make a hand-swapped artifact look like a context change,
    and a swapped artifact is precisely the REGRESSED case the gate must catch.
    """
    directories = {
        "nba": ML_TRAINING / "models",
        "wnba": ML_TRAINING / "models_wnba",
        "gleague": ML_TRAINING / "models_gleague",
        "nfl": ML_TRAINING / "models_nfl",
    }
    out = {}
    for league, directory in directories.items():
        entries = {}
        if directory.is_dir():
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix in (".json", ".joblib"):
                    entries[path.name] = hashlib.sha256(
                        path.read_bytes()).hexdigest()
        out[league] = entries
    return out


def predictable_date(health: dict, league: str):
    """The one date MAX_DAYS_AHEAD permits for this league: cutoff + 1 day.

    DERIVED, NOT PINNED. A fixed date stops being predictable the moment the
    data advances past it - which on 21 October is every league at once.
    """
    import datetime as dt

    block = health if league == "nba" else health.get(league) or {}
    cutoff = block.get("data_as_of")
    if not cutoff:
        return None
    return (dt.date.fromisoformat(cutoff) + dt.timedelta(days=1)).isoformat()


def nfl_matchups(count=2) -> list:
    """Two predictable NFL fixtures with four distinct teams.

    FROM THE SERVICE, BECAUSE NOTHING ELSE KNOWS. The dependency rule is
    evaluated against the served history, so the only honest source for "which
    fixtures can be predicted right now" is the service doing the serving.
    Asking it is also what makes this gate survive the season advancing: next
    week's answer is a different four teams and the gate needs no edit.

    Returns [(home, away, date), ...], or [] when nothing is predictable -
    which is correct between seasons and is reported rather than failing.
    """
    fixtures = curl("GET", f"{INFERENCE}/schedule/nfl?predictable_only=true")
    if not isinstance(fixtures, list):
        return []

    chosen, used = [], set()
    for fixture in sorted(fixtures, key=lambda f: (f.get("game_date") or "",
                                                   f.get("game_id") or "")):
        home, away = fixture.get("home_team_id"), fixture.get("away_team_id")
        if home in used or away in used:
            continue
        chosen.append((home, away, fixture.get("game_date")))
        used.update((home, away))
        if len(chosen) == count:
            break
    return chosen


# Every differing field is printed. A CAP WOULD HAVE TO ANNOUNCE ITSELF, and
# the version before 2026-10-29 did not: `fields[:6]` on the CANNOT COMPARE
# branch and `fields[:8]` on REGRESSED, with a heading that counted BODIES
# (`len(differing)`) rather than fields. Across the A3 switch that made
# `total_rebounds` look unchanged when it had moved 0.2441 and 0.2745 - the
# figure was only recovered afterwards by recomputing it from the baseline.
#
# The REGRESSED branch was the worse of the two: with seven markets it could
# not truncate today, so a body with a ninth changed field would have hidden a
# real regression behind a silent cut. The misleading-green pattern in its
# purest form - green step, true fact, unreadable.
#
# There is no cap now. If one is ever wanted for readability it must print the
# count it dropped, which is what SHOW_ALL_FIELDS being absent would mean.
def print_fields(key: str, fields: list, limit: int = None) -> None:
    """Every field, or `limit` of them followed by how many were dropped."""
    shown = fields if limit is None else fields[:limit]
    for field in shown:
        print(f"             {key}  {field}")
    dropped = len(fields) - len(shown)
    if dropped:
        print(f"             {key}  ... +{dropped} more field(s) not shown")


def field_count(differing: list) -> int:
    """Total differing FIELDS across bodies, which is not len(differing)."""
    return sum(len(fields) for _key, fields in differing)


def backend_fixtures(schedule, league: str, date, count: int = 2) -> tuple:
    """Predictable fixtures the backend ALREADY HAS A ROW FOR.

    WHY THE BACKEND HALF CANNOT USE THE CONSTRUCTED PAIRS. MATCHUPS holds
    fixed team pairs so the inference half's bodies are stable from day to
    day, and ATL vs BOS was chosen precisely because it is not a scheduled
    game. Posting it to /api/predictions makes the backend run
    GameLookup.findOrCreateGame, which CREATES a `game` row - so every
    capture left two new rows in the production database, dated whatever
    cutoff+1 was that morning. Measured 2026-10-29: 17 such rows, and 1,677
    prediction rows hanging off them. A measuring tool was changing what it
    measured.

    So the backend half is pointed at rows that already exist. The source is
    the backend's OWN cached schedule - the same body this capture already
    fetched - which is by definition the set of `game` rows in the window. No
    row can be created by posting one of them.

    THE INFERENCE HALF IS UNCHANGED and keeps the constructed pairs. The two
    halves are diffed against their own baselines and never cross-compared,
    so they do not have to be the same fixture; what each one needs is to be
    the same fixture as its own baseline, which the context below enforces.

    PREDICTABILITY IS PER LEAGUE, as everywhere else in this project. The
    three basketball leagues permit exactly cutoff + 1, so `date` is matched
    exactly; the NFL's rule is a dependency on both teams' earlier games, so
    its own `predictable` flag is read instead. A fixture the backend would
    refuse is not a usable body.

    Returns (fixtures, reason). `fixtures` is a list of (home, away, date);
    `reason` is None when `count` were found and a sentence otherwise. An
    empty list with a reason is the WNBA out of season, and it must be
    REPORTED - see capture(), which records it in the context and prints it.
    """
    if not isinstance(schedule, list):
        return [], ("the backend's schedule could not be read, so no "
                    "existing fixture could be chosen")

    rows = [row for row in schedule
            if isinstance(row, dict) and row.get("league") == league]
    if not rows:
        return [], (f"the backend has no cached {league} fixture at all "
                    f"(its sync writes none out of season)")

    if league == "nfl":
        # ITS OWN FLAG, NOT A DATE. 193 of 208 fixtures are refused at any
        # moment under the dependency rule, and which ones changes weekly.
        usable = [row for row in rows if row.get("predictable") is True]
        why = "none of them is predictable yet under the dependency rule"
    else:
        usable = [row for row in rows if row.get("gameDate") == date]
        why = (f"none of its {len(rows)} cached fixture(s) falls on "
               f"{date}, the only date MAX_DAYS_AHEAD permits")

    if not usable:
        return [], f"the backend has {league} fixtures but {why}"

    # FOUR DISTINCT TEAMS, the same shape MATCHUPS has and for the same
    # reason: two bodies that share a team share whatever that team's
    # features did.
    chosen, used = [], set()
    for row in usable:
        home, away = row.get("homeTeamId"), row.get("awayTeamId")
        if home is None or away is None or home in used or away in used:
            continue
        chosen.append((home, away, row.get("gameDate")))
        used.update((home, away))
        if len(chosen) == count:
            break

    if not chosen:
        return [], (f"the backend's {league} fixtures could not yield a pair "
                    f"with distinct teams")
    if len(chosen) < count:
        return chosen, (f"only {len(chosen)} of {count} usable {league} "
                        f"fixture(s) with distinct teams")
    return chosen, None


def capture() -> dict:
    """Bodies plus the context they were produced under."""
    inference_health = curl("GET", f"{INFERENCE}/health")
    backend_health = curl("GET", f"{BACKEND}/api/health")
    schedule = curl("GET", f"{BACKEND}/api/games/schedule?daysAhead=120")

    served = inference_health.get("served_data") or {}
    availability = inference_health.get("availability") or {}
    models = model_hashes()

    context = {"captured_at": datetime.now(timezone.utc).isoformat(),
               "leagues": {}}
    bodies = {}

    for league, pairs in MATCHUPS.items():
        block = (inference_health if league == "nba"
                 else inference_health.get(league) or {})
        avail = availability.get(league) or {}

        # The NFL's pairs AND date come from its own schedule; the other three
        # keep fixed pairs and a date derived from their cutoff.
        if league == "nfl":
            derived = nfl_matchups()
            pairs = [(home, away) for home, away, _d in derived]
            dates = [d for _h, _a, d in derived]
        else:
            date = predictable_date(inference_health, league)
            pairs = list(pairs or [])
            dates = [date] * len(pairs)

        context["leagues"][league] = {
            "data_as_of": block.get("data_as_of"),
            "snapshot": served.get("snapshot"),
            "table_sha256": (served.get("table_sha256") or {}).get(league),
            "model_sha256": models.get(league),
            # Enough identity to tell whether two captures saw the SAME report.
            # The state alone is not enough: two different reports both read
            # `used`, and that would make a real body difference look
            # attributable to code.
            "availability_state": avail.get("state"),
            "availability_report": avail.get("report_timestamp"),
            # WHICH REFRESH RUN THE CAPTURE SAW, identity only.
            #
            # `refresh` was first added to HEALTH_CONTEXT_FIELDS alone, with a
            # comment claiming a change in it would make the league CANNOT
            # COMPARE. THAT CLAIM WAS WRONG: membership there only STRIPS a
            # field from the body diff, and nothing compared this one as
            # context - so a refresh running between the two captures was
            # invisible, which is a misleading green rather than a
            # classification.
            #
            # IDENTITY, NOT THE CLOCK. `hours_since` is a float against now,
            # and `state` and `note` both derive from it, so all three move
            # between any two captures and none may enter the comparison.
            # What goes in is which verdict was reached, when, on what
            # snapshot, and per league - things that change only when a
            # refresh actually ran.
            "refresh_verdict": refresh_identity(inference_health),
            "requested": [{"home": h, "away": a, "date": d}
                          for (h, a), d in zip(pairs, dates)],
        }

        # THE BACKEND HALF, ON ROWS THAT ALREADY EXIST, and decided before
        # the early return below so that a league with nothing predictable on
        # the inference side still records what happened to its backend half.
        # See backend_fixtures: posting a constructed pair here ran
        # GameLookup.findOrCreateGame and CREATED a `game` row every morning,
        # so the instrument was changing the database it measures.
        chosen, shortfall = backend_fixtures(
            schedule, league, dates[0] if dates else None)
        context["leagues"][league]["backend_fixtures"] = [
            {"home": h, "away": a, "date": d} for h, a, d in chosen]
        context["leagues"][league]["backend_skipped"] = shortfall
        if shortfall:
            # LOUDLY, AND IN THE CONTEXT. A body that silently stops being
            # captured is a guard that silently stops guarding - which is how
            # the WNBA's freshness block came to be missing in the first
            # place.
            print(f"  {league}: BACKEND HALF SKIPPED - {shortfall}")
        if not chosen:
            bodies[f"{league}.backend.none"] = {
                "__skipped__": shortfall or "no usable existing fixture"}
        for index, (home, away, date) in enumerate(chosen):
            bodies[f"{league}.backend.{index}"] = curl(
                "POST", f"{BACKEND}{BACKEND_PATH[league]}",
                {"homeTeamId": home, "awayTeamId": away, "gameDate": date})

        if not pairs:
            # Nothing predictable. True for the WNBA most of the year and for
            # the NFL between seasons, and reported rather than failed.
            bodies[f"{league}.inference.none"] = {
                "__skipped__": "no predictable matchup"}
            continue

        for index, ((home, away), date) in enumerate(zip(pairs, dates)):
            if date is None:
                bodies[f"{league}.inference.{index}"] = {
                    "__skipped__": "no cutoff in /health"}
                continue
            bodies[f"{league}.inference.{index}"] = curl(
                "POST", f"{INFERENCE}{PREDICT_PATH[league]}",
                {"home_team_id": home, "away_team_id": away,
                 "game_date": date})

    # Not league-scoped and not a prediction, so kept beside them: a count that
    # slides with the calendar, guarded against collapse rather than diffed.
    context["schedule_count"] = len(schedule) if isinstance(schedule, list) else None
    context["teams_count"] = len(curl("GET", f"{BACKEND}/api/teams") or [])
    bodies["inference_health"] = inference_health
    bodies["backend_health"] = backend_health

    return {"context": context, "bodies": bodies}


def league_comparability(before: dict, after: dict) -> tuple:
    """(comparable, reasons, notes) for one league's context.

    THE TABLE RULE, AND THE SECOND ROW IS THE ONE WORTH HAVING:
        same id,  same hash   -> comparable
        same id,  diff hash   -> REGRESSED, a snapshot mutated in place
        diff id,  same hash   -> comparable, and say the id moved
        diff id,  diff hash   -> CANNOT COMPARE, the data advanced
        table SET changed     -> CANNOT COMPARE, the instrument changed

    THE FIFTH ROW WAS ADDED AFTER THIS GATE CALLED A DEPLOY A REGRESSION.
    Comparing the two hash DICTS for equality cannot tell "a table's contents
    changed" from "a table was added to the set being hashed" - and the second
    is what happens the first time an image hashes a new optional table. On
    2026-10-22 the NFL came back REGRESSED with "a snapshot mutated in place"
    when nothing had been mutated at all: all three tables present in both
    captures were byte-identical and the only difference was a new key,
    nfl_quarantine.csv -> "absent", because the new image hashes an optional
    table that the served snapshot predates.
    Baseline and current snapshot ids were both 2026-10-22T0711.

    So mutation is judged on the tables present in BOTH captures, which keeps
    the detector's whole point intact - a changed table under an unchanged id
    is still REGRESSED - while a changed table SET withholds judgement instead
    of asserting a defect. CANNOT COMPARE is a verdict and not a failure; a
    false REGRESSED is worse than either, because the only way to clear it
    looks like recapturing the baseline.
    """
    reasons, notes = [], []
    mutated = False

    same_snapshot = before.get("snapshot") == after.get("snapshot")

    before_tables = before.get("table_sha256") or {}
    after_tables = after.get("table_sha256") or {}
    shared = set(before_tables) & set(after_tables)
    added = sorted(set(after_tables) - set(before_tables))
    removed = sorted(set(before_tables) - set(after_tables))
    shared_differ = sorted(key for key in shared
                           if before_tables[key] != after_tables[key])
    same_tables = not shared_differ and not added and not removed

    if same_snapshot and shared_differ:
        mutated = True
    elif not same_snapshot and shared_differ:
        reasons.append("tables changed (data advanced)")
    elif not same_snapshot and same_tables:
        notes.append(f"snapshot id moved {before.get('snapshot')} -> "
                     f"{after.get('snapshot')} with identical table contents")

    # A REASON, SO THE LEAGUE GOES CANNOT COMPARE RATHER THAN REGRESSED, and
    # named both ways round: a table appearing is an image that hashes more
    # than it used to, a table DISAPPEARING is the more alarming direction and
    # must never be quietly reported as a pass.
    if added or removed:
        parts = []
        if added:
            parts.append(f"now hashed: {', '.join(added)}")
        if removed:
            parts.append(f"no longer hashed: {', '.join(removed)}")
        reasons.append("the set of served tables changed (" +
                       "; ".join(parts) + ")")

    if before.get("data_as_of") != after.get("data_as_of"):
        reasons.append(f"data_as_of {before.get('data_as_of')} -> "
                       f"{after.get('data_as_of')}")
    if before.get("model_sha256") != after.get("model_sha256"):
        reasons.append("models changed")
    # WHICH EXISTING FIXTURE THE BACKEND HALF USED. It is chosen from the
    # backend's cached schedule at capture time, so it can move between two
    # captures for reasons that are not code: a fixture becoming predictable,
    # a played game leaving the window, the date rolling over. A body
    # difference on a different fixture is not attributable to a change, so
    # the league withholds instead of asserting one.
    if before.get("backend_fixtures") != after.get("backend_fixtures"):
        reasons.append(
            f"the backend half used different existing fixtures "
            f"({before.get('backend_fixtures')} -> "
            f"{after.get('backend_fixtures')})")
    # AND A HALF THAT WAS CAPTURED IN ONE AND SKIPPED IN THE OTHER. Not a
    # defect, and not comparable either: one side has a body the other has
    # not. Reported as its own reason rather than folded into the line above.
    if bool(before.get("backend_skipped")) != bool(after.get("backend_skipped")):
        reasons.append(
            f"the backend half was captured in one capture and skipped in "
            f"the other ({before.get('backend_skipped') or 'captured'} -> "
            f"{after.get('backend_skipped') or 'captured'})")
    if before.get("availability_state") != after.get("availability_state"):
        reasons.append(f"availability {before.get('availability_state')} -> "
                       f"{after.get('availability_state')}")
    elif before.get("availability_report") != after.get("availability_report"):
        reasons.append(f"a different injury report "
                       f"({before.get('availability_report')} -> "
                       f"{after.get('availability_report')})")
    # A REFRESH RAN BETWEEN THE TWO CAPTURES. Deliberately conservative: even
    # when the snapshot id and every table hash prove nothing moved - which is
    # what a `nothing-new` run looks like - the gate withholds rather than
    # attributing a body difference to the code. The reason names both
    # verdicts, so a `nothing-new` between captures is cheap to recognise and
    # re-run past.
    if before.get("refresh_verdict") != after.get("refresh_verdict"):
        was = before.get("refresh_verdict") or {}
        now = after.get("refresh_verdict") or {}
        reasons.append(
            f"a refresh ran between the captures "
            f"({was.get('outcome')} at {was.get('last_verdict_at')} -> "
            f"{now.get('outcome')} at {now.get('last_verdict_at')})")

    if before.get("requested") != after.get("requested"):
        reasons.append("the requested matchups or dates differ")

    return (not reasons and not mutated), reasons, notes, mutated


def diff_fields(before, after, path="") -> list:
    """Every leaf that differs, named by path, as information."""
    out = []
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            out += diff_fields(before.get(key), after.get(key),
                               f"{path}.{key}" if path else key)
    elif isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            out.append(f"{path}: {len(before)} items -> {len(after)} items")
        else:
            for i, (b, a) in enumerate(zip(before, after)):
                out += diff_fields(b, a, f"{path}[{i}]")
    elif before != after:
        out.append(f"{path}: {before!r} -> {after!r}")
    return out


EXIT_PASS, EXIT_REGRESSED, EXIT_ERROR, EXIT_CANNOT = 0, 1, 2, 3


def compare(before: dict, after: dict) -> int:
    """Four verdicts, decided per league."""
    if "context" not in before:
        print("CANNOT COMPARE - baseline predates context recording.")
        print("  This baseline has no `context` block, so there is no way to "
              "know what\n  data, models or injury report it was produced "
              "against. A field-by-field\n  red would be a guess dressed as a "
              "finding. Capture a new baseline\n  immediately before the "
              "change you want to check.")
        return EXIT_CANNOT

    print(f"baseline captured {before['context'].get('captured_at')}")
    print(f"current  captured {after['context'].get('captured_at')}\n")

    verdicts, regressions = {}, []

    for league in MATCHUPS:
        b = before["context"]["leagues"].get(league, {})
        a = after["context"]["leagues"].get(league, {})
        comparable, reasons, notes, mutated = league_comparability(b, a)

        keys = [k for k in after["bodies"] if k.startswith(f"{league}.")]
        differing = []
        for key in keys:
            fields = diff_fields(strip_volatile(before["bodies"].get(key)),
                                 strip_volatile(after["bodies"].get(key)))
            if fields:
                differing.append((key, fields))

        if mutated:
            verdicts[league] = "REGRESSED"
            print(f"  {league:<8} REGRESSED - the snapshot id is unchanged "
                  f"but its table contents are not.")
            print(f"           A snapshot mutated in place. The id cannot "
                  f"show this; the hash can.")
            regressions.append(league)
        elif not comparable:
            verdicts[league] = "CANNOT COMPARE"
            print(f"  {league:<8} CANNOT COMPARE - "
                  f"{'; '.join(reasons)}")
            print(f"           A body difference here is not attributable to "
                  f"a code change.")
            if differing:
                print(f"           {len(differing)} body/bodies and "
                      f"{field_count(differing)} field(s) differ, shown as "
                      f"information:")
                for key, fields in differing:
                    print_fields(key, fields)
        elif differing:
            verdicts[league] = "REGRESSED"
            print(f"  {league:<8} REGRESSED - context identical, "
                  f"{len(differing)} body/bodies and "
                  f"{field_count(differing)} field(s) differ:")
            for key, fields in differing:
                print_fields(key, fields)
            regressions.append(league)
        else:
            verdicts[league] = "PASS"
            print(f"  {league:<8} PASS - context identical, every body "
                  f"byte-identical")
        for note in notes:
            print(f"           note: {note}")

        # THE BACKEND HALF'S COVERAGE, STATED EVERY RUN. A skipped body is
        # reported whether or not the league passed, because "PASS - every
        # body byte-identical" over a body that was never captured is the
        # misleading green this gate exists to avoid.
        for side, label in ((b, "baseline"), (a, "current")):
            skipped = side.get("backend_skipped")
            if skipped:
                print(f"           backend half SKIPPED in the {label}: "
                      f"{skipped}")
        used = a.get("backend_fixtures")
        if used is not None and not a.get("backend_skipped"):
            print(f"           backend half used {len(used)} existing "
                  f"fixture(s): "
                  + ", ".join(f"{f['home']}v{f['away']}@{f['date']}"
                              for f in used))

    for name in ("inference_health", "backend_health"):
        b = strip_health_context(strip_volatile(before["bodies"].get(name)))
        a = strip_health_context(strip_volatile(after["bodies"].get(name)))
        added = _added_keys(b, a)
        missing = _missing_or_changed(b, a)
        if missing:
            print(f"  {name}: {len(missing)} baseline field(s) lost or "
                  f"changed: {missing[:6]}")
            regressions.append(name)
        elif added:
            print(f"  {name}: unchanged, {len(added)} field(s) added: {added}")
        else:
            print(f"  {name}: identical")

    sb = before["context"].get("schedule_count")
    sa = after["context"].get("schedule_count")
    print(f"  schedule_count {sb} -> {sa}  (slides with the calendar, "
          f"reported not diffed)")
    if sa is not None and sa < MIN_SCHEDULE_COUNT:
        print(f"           COLLAPSED below {MIN_SCHEDULE_COUNT}")
        regressions.append("schedule_count")
    if before["context"].get("teams_count") != after["context"].get("teams_count"):
        print(f"  teams_count {before['context'].get('teams_count')} -> "
              f"{after['context'].get('teams_count')}  CHANGED")
        regressions.append("teams_count")

    print()
    if regressions:
        print(f"REGRESSED: {sorted(set(regressions))}")
        return EXIT_REGRESSED
    if "CANNOT COMPARE" in verdicts.values():
        cannot = [k for k, v in verdicts.items() if v == "CANNOT COMPARE"]
        print(f"CANNOT COMPARE for {cannot}; "
              f"{[k for k, v in verdicts.items() if v == 'PASS']} passed.")
        return EXIT_CANNOT
    print("PASS - every league comparable and every body byte-identical.")
    return EXIT_PASS


def strip_health_context(value):
    """Drop the context-bearing fields of a health body, at any depth."""
    if isinstance(value, dict):
        return {k: strip_health_context(v) for k, v in value.items()
                if k not in HEALTH_CONTEXT_FIELDS}
    if isinstance(value, list):
        return [strip_health_context(v) for v in value]
    return value


def _added_keys(before, after, path="") -> list:
    out = []
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(after) - set(before)):
            out.append(f"{path}.{key}" if path else key)
        for key in sorted(set(before) & set(after)):
            out += _added_keys(before[key], after[key],
                              f"{path}.{key}" if path else key)
    return out


def _missing_or_changed(before, after, path="") -> list:
    """Every baseline field that is absent or different. Additions are allowed.

    The health bodies are compared this way because adding a league makes
    /health grow fields, and demanding byte equality there would fail on the
    intended change - the only way past which is to re-baseline, which is how a
    real regression gets waved through. models_loaded BECOMING a different type
    still fails, which is the outage this rule was written for.
    """
    out = []
    if isinstance(before, dict):
        if not isinstance(after, dict):
            return [path or "<root>"]
        for key, value in before.items():
            here = f"{path}.{key}" if path else key
            if key not in after:
                out.append(here)
            else:
                out += _missing_or_changed(value, after[key], here)
    elif before != after:
        out.append(path or "<root>")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    cap = sub.add_parser("capture", help="record bodies and context")
    cap.add_argument("path")

    cmp_ = sub.add_parser("compare", help="compare a capture against live")
    cmp_.add_argument("baseline")
    cmp_.add_argument("--against", help="a second capture file, not the "
                                        "live service")

    args = parser.parse_args()

    try:
        if args.command == "capture":
            Path(args.path).write_text(
                json.dumps(capture(), indent=2, sort_keys=True),
                encoding="utf-8")
            print(f"captured -> {args.path}")
            return EXIT_PASS

        if args.command == "compare":
            before = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
            after = (json.loads(Path(args.against).read_text(encoding="utf-8"))
                     if args.against else capture())
            return compare(before, after)

        # THE BARE RUN: capture twice, back to back, and compare the two. This
        # checks the service is deterministic and that a PASS is reachable at
        # all, and it cannot go stale, because it never reads a stored file.
        print("Bare run: capturing twice and comparing. No stored baseline is "
              "read,\nso this cannot go red because a file aged.\n")
        # WARM FIRST. The sidecar reports `never_fetched` until something has
        # asked it, which maps to `unknown` and is a genuinely different state
        # from `none_published` - so a cold first capture would make the pair
        # incomparable for a reason about this gate's own timing rather than
        # about the service. Not done inside capture(), which must report the
        # state it actually finds.
        curl("GET", f"{INFERENCE}/health")
        first = capture()
        second = capture()
        return compare(first, second)
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print("\nERROR - the gate did not reach a verdict.")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
