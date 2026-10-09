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

# A DATED RECORD, NOT A CHECK. 0.4191701412200928 was the NBA reference
# probability for ATL vs BOS on 2026-04-13, under snapshot 2026-10-06T2018 with
# no injury report published, from the models A7 shipped on 2026-10-06. Its
# predecessor, 0.42541608214378357, held from the 38-feature retrain on
# 2026-08-26. Both are historical: once reports resume, the same request
# against the same models returns something else, because four of the 38
# features are a function of today rather than of the fixture.
HISTORICAL_NBA_REFERENCE = 0.4191701412200928

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
    "data_as_of", "dataAsOf", "stale",
    # THE NFL'S TWO MOVING COUNTS. `predictable_fixtures` falls and `fixtures`
    # falls as the season is played, both as a function of the data rather than
    # of any code - so they are context in exactly the way data_as_of is, and
    # diffing them would report a played game as a regression. They are still
    # compared as context, where a change makes the league CANNOT COMPARE, so
    # nothing here is unguarded.
    "predictable_fixtures", "predictableFixtures", "fixtures",
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
            "requested": [{"home": h, "away": a, "date": d}
                          for (h, a), d in zip(pairs, dates)],
        }

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
            bodies[f"{league}.backend.{index}"] = curl(
                "POST", f"{BACKEND}{BACKEND_PATH[league]}",
                {"homeTeamId": home, "awayTeamId": away, "gameDate": date})

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
    """
    reasons, notes = [], []
    mutated = False

    same_snapshot = before.get("snapshot") == after.get("snapshot")
    same_tables = before.get("table_sha256") == after.get("table_sha256")

    if same_snapshot and not same_tables:
        mutated = True
    elif not same_snapshot and not same_tables:
        reasons.append("tables changed (data advanced)")
    elif not same_snapshot and same_tables:
        notes.append(f"snapshot id moved {before.get('snapshot')} -> "
                     f"{after.get('snapshot')} with identical table contents")

    if before.get("data_as_of") != after.get("data_as_of"):
        reasons.append(f"data_as_of {before.get('data_as_of')} -> "
                       f"{after.get('data_as_of')}")
    if before.get("model_sha256") != after.get("model_sha256"):
        reasons.append("models changed")
    if before.get("availability_state") != after.get("availability_state"):
        reasons.append(f"availability {before.get('availability_state')} -> "
                       f"{after.get('availability_state')}")
    elif before.get("availability_report") != after.get("availability_report"):
        reasons.append(f"a different injury report "
                       f"({before.get('availability_report')} -> "
                       f"{after.get('availability_report')})")
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
                print(f"           {len(differing)} body/bodies differ, "
                      f"shown as information:")
                for key, fields in differing:
                    for field in fields[:6]:
                        print(f"             {key}  {field}")
        elif differing:
            verdicts[league] = "REGRESSED"
            print(f"  {league:<8} REGRESSED - context identical, "
                  f"{len(differing)} body/bodies differ:")
            for key, fields in differing:
                for field in fields[:8]:
                    print(f"             {key}  {field}")
            regressions.append(league)
        else:
            verdicts[league] = "PASS"
            print(f"  {league:<8} PASS - context identical, every body "
                  f"byte-identical")
        for note in notes:
            print(f"           note: {note}")

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
