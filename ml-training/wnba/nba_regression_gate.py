"""The NBA regression gate for WNBA phase 4.

Phase 4 is the first WNBA phase that edits shared code - the inference service,
the backend, the schedule sync - so every step needs checking against what the
NBA served before any of it. Captured once with --capture, then rerun after
every step with no argument.

The pass condition is BYTE-IDENTICAL NBA response bodies, excluding only the
fields that legitimately differ per call: predictedAt, and row ids, which are
IDENTITY columns and advance on every append-only write.

    python nba_regression_gate.py --capture        # before touching anything
    python nba_regression_gate.py                  # after every step
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# THE DEFAULT POINTS AT THE NEWEST RECAPTURE, NOT THE OLDEST RECORD. Each phase
# captures its own baseline so no earlier one is overwritten, and the earlier
# files stay as dated records - but a bare run has to mean "against what is in
# production now". Left pointing at the WNBA-era file, the first bare run after
# A7 would have reported a regression on every NBA field, from a stale default
# rather than a real change.
BASELINE = HERE.parent / "gate_baseline_a7.json"

BACKEND = "http://localhost:8080"
INFERENCE = "http://localhost:8000"

# The reference matchup: ATL home vs BOS, the only date MAX_DAYS_AHEAD permits
# against a data_as_of of 2026-04-12.
MATCHUP = {"homeTeamId": 1610612737, "awayTeamId": 1610612738,
           "gameDate": "2026-04-13"}
# Moved by A7 on 2026-10-06, when the NBA stopped trusting LeagueGameFinder's
# margin column. The previous value, 0.42541608214378357, held from the
# 38-feature retrain on 2026-08-26 and is the number every study before A7 was
# measured against. This is the ONE sanctioned change to it.
REFERENCE_PROBABILITY = 0.4191701412200928

# The WNBA's pinned reference: Phoenix Mercury home vs Las Vegas Aces, the
# only date MAX_DAYS_AHEAD permits against a WNBA cutoff of 2026-09-24.
WNBA_MATCHUP = {"homeTeamId": 1611661317, "awayTeamId": 1611661319,
                "gameDate": "2026-09-25"}
WNBA_INFERENCE_BODY = {"home_team_id": 1611661317,
                       "away_team_id": 1611661319,
                       "game_date": "2026-09-25"}
WNBA_REFERENCE = {
    "moneyline": 0.18193019489666626,
    "spread": -8.403725674288566,
    "totals": 178.40689601339275,
}

# Legitimately per-call, so excluded from the diff rather than silently ignored.
#
# daysBehind/days_behind are in here because they are computed from
# datetime.now() against a fixed data cutoff, so they increment every midnight.
# That was missed on the first pass and the gate went red the first time this
# session crossed a date boundary: four "regressions" that were all
# 174 -> 175, with every prediction value bit-identical beside them. The fix is
# to classify the field, NOT to recapture the baseline - recapturing is how a
# real regression gets absorbed into a new "correct" state.
VOLATILE = {"predictedAt", "id", "gameId", "predicted_at",
            "daysBehind", "days_behind"}

# THE HEALTH BODIES ARE COMPARED AS A SUBSET, EVERYTHING ELSE BYTE-FOR-BYTE.
#
# Adding a league means /health reports a fourth model family and the WNBA's
# own cutoff, so demanding byte equality there would fail on the intended
# change and the only way past it would be to re-baseline - which is how a
# real regression gets waved through. The semantics that actually matter are
# "no field a client already reads has changed", so every baseline key must
# still be present and equal, and new keys are allowed and NAMED.
#
# This is the same direction CLAUDE.md section 25 settled on for the backend's
# Jackson config: a removed or retyped field is a break, an added one is
# backwards compatible. models_loaded gaining a key is the additive case;
# models_loaded BECOMING a different type is what took the browse view down
# in section 21, and a subset check still catches that.
#
# The prediction bodies stay strictly byte-identical. Nothing about serving a
# second league should change what the NBA returns.
ADDITIVE_ONLY = {"inference_health", "backend_health"}

# SLIDING, NOT VOLATILE-PER-FIELD: /api/games/schedule returns fixtures in a
# 120-day window measured FROM TODAY, so the count moves every midnight as the
# far edge advances into the season. Measured when it first differed:
# 711 -> 718, with exactly 7 fixtures on the new far edge (2027-02-01) and 0
# dropping off the near edge, so the arithmetic closed exactly and the change
# was the calendar rather than the code.
#
# Equality is therefore the wrong test, but dropping the check loses the half
# that mattered - a schedule endpoint returning nothing. So the count is
# reported rather than diffed, and a separate assertion fails on a collapse.
# Classified rather than re-baselined, for the same reason daysBehind was:
# recapturing is how a real regression gets absorbed into a new "correct"
# state.
SLIDING = {"schedule_count"}
MIN_SCHEDULE_COUNT = 100


def subset_match(before, after):
    """(every baseline field survived unchanged, the names that were added).

    Recurses, so a nested object gaining a field is tolerated while a nested
    field changing value is not.
    """
    if isinstance(before, dict):
        if not isinstance(after, dict):
            return False, set()
        added = set(after) - set(before)
        for key, value in before.items():
            if key not in after:
                return False, added
            ok, deeper = subset_match(value, after[key])
            if not ok:
                return False, added
            added |= {f"{key}.{name}" for name in deeper}
        return True, added
    return before == after, set()


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
                if k not in VOLATILE}
    if isinstance(value, list):
        return [strip_volatile(v) for v in value]
    return value


def probe():
    schedule = curl("GET", f"{BACKEND}/api/games/schedule?daysAhead=120")
    return {
        # THE WNBA IS GUARDED TOO, FROM PHASE 4 OF THE G LEAGUE ONWARD. This
        # file's name is historical - it guarded one league when it was
        # written. Both layers are probed because they can drift apart: the
        # WNBA phase caught the Java record silently dropping a freshness
        # block exactly that way, by the inference body gaining two fields
        # while the backend body gained one.
        "wnba_prediction": curl("POST", f"{BACKEND}/api/predictions/wnba",
                                WNBA_MATCHUP),
        "wnba_inference": curl("POST", f"{INFERENCE}/predict/wnba",
                               WNBA_INFERENCE_BODY),
        "predictions": curl("POST", f"{BACKEND}/api/predictions", MATCHUP),
        "quarter_half": curl("POST", f"{BACKEND}/api/predictions/quarter-half",
                             MATCHUP),
        "player_props": curl("POST", f"{BACKEND}/api/predictions/player-props",
                             MATCHUP),
        "backend_health": curl("GET", f"{BACKEND}/api/health"),
        "inference_health": curl("GET", f"{INFERENCE}/health"),
        "schedule_count": len(schedule) if isinstance(schedule, list) else -1,
        "teams_count": len(curl("GET", f"{BACKEND}/api/teams")),
    }


def reference_value(snapshot):
    try:
        return snapshot["predictions"]["latestPrediction"]["homeWinProbability"]
    except (KeyError, TypeError):
        return None


def wnba_reference_values(snapshot):
    """The WNBA's three pinned values, from the backend body.

    Read from the BACKEND rather than the inference body, because that is the
    layer a client sees and the one where the WNBA phase found a field
    silently dropped. The inference body is compared too, byte-for-byte, by
    the ordinary diff.
    """
    try:
        prediction = snapshot["wnba_prediction"]["prediction"]
        return {"moneyline": prediction["homeWinProbability"],
                "spread": prediction["homeMargin"],
                "totals": prediction["totalPoints"]}
    except (KeyError, TypeError):
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--baseline", default=str(BASELINE),
                        help="which baseline to compare against or write. "
                             "A phase uses its own file so an earlier "
                             "phase's record is never overwritten - the "
                             "refusal below only protects the file named "
                             "here.")
    parser.add_argument("--step", default="", help="label for this run")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    snapshot = probe()
    value = reference_value(snapshot)

    if args.capture:
        if baseline_path.exists():
            print(f"REFUSING: {baseline_path.name} already exists. The baseline is "
                  "the pre-change state and\nmust not be recaptured after a "
                  "change - that would make any regression invisible.")
            return 1
        baseline_path.write_text(json.dumps(strip_volatile(snapshot), indent=2,
                                       sort_keys=True))
        print(f"Captured baseline to {baseline_path.name}\n")
        print(f"  reference homeWinProbability : {value}")
        print(f"  matches the recorded value   : {value == REFERENCE_PROBABILITY}")
        print(f"  schedule fixtures            : {snapshot['schedule_count']}")
        print(f"  teams                        : {snapshot['teams_count']}")
        print(f"  inference models_loaded      : "
              f"{snapshot['inference_health'].get('models_loaded')}")
        return 0

    if not baseline_path.exists():
        print(f"No {baseline_path.name}. Run with --capture first.")
        return 1

    # Both sides are stripped HERE rather than relying on the baseline having
    # been stripped at capture time, so VOLATILE is authoritative now instead
    # of frozen into a file written earlier. Adding a field to that set then
    # takes effect against an existing baseline, which is what let
    # daysBehind be classified without recapturing.
    before = strip_volatile(json.loads(baseline_path.read_text()))
    after = strip_volatile(snapshot)

    label = f" [{args.step}]" if args.step else ""
    print(f"NBA REGRESSION GATE{label}")
    print("=" * 78)

    failures = []
    if value != REFERENCE_PROBABILITY:
        failures.append(f"reference value is {value}, recorded "
                        f"{REFERENCE_PROBABILITY}")
    print(f"  reference homeWinProbability : {value}  "
          f"{'OK' if value == REFERENCE_PROBABILITY else 'CHANGED'}")

    wnba = wnba_reference_values(snapshot)
    for market, expected in WNBA_REFERENCE.items():
        got = wnba.get(market)
        if got != expected:
            failures.append(f"WNBA {market} is {got}, recorded {expected}")
        print(f"  WNBA {market:<24}{got}  "
              f"{'OK' if got == expected else 'CHANGED'}")

    for key in sorted(before):
        if key in SLIDING:
            value = after.get(key)
            collapsed = not isinstance(value, int) or value < MIN_SCHEDULE_COUNT
            print(f"  {key:<28} {before[key]} -> {value}  "
                  f"(date-derived, not diffed)"
                  f"{'  COLLAPSED' if collapsed else ''}")
            if collapsed:
                failures.append(f"{key} collapsed to {value}")
            continue

        if key in ADDITIVE_ONLY:
            same, added = subset_match(before[key], after.get(key))
            note = "identical" if same else "DIFFERS"
            if same and added:
                note = f"unchanged, {len(added)} field(s) added: {sorted(added)}"
        else:
            same, note = before[key] == after.get(key), None
            note = note or ("identical" if same else "DIFFERS")
        print(f"  {key:<28} {note}")
        if not same:
            failures.append(key)

    if failures:
        print(f"\n  {len(failures)} REGRESSION(S): {failures}")
        for key in failures:
            if key in before and isinstance(before[key], dict):
                b, a = before[key], after.get(key, {})
                for field in sorted(set(b) | set(a)):
                    if b.get(field) != a.get(field):
                        print(f"    {key}.{field}: {b.get(field)!r} -> "
                              f"{a.get(field)!r}")
        return 1

    print("\n  PASS - every NBA body byte-identical, reference value unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
