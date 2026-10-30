r"""The gate's table rule: five rows, and the fifth one was bought dearly.

`league_comparability` decides whether two captures of one league can be
compared at all. Its table rule used to compare the two hash DICTS for
equality, which cannot tell these apart:

    a table's CONTENTS changed under an unchanged snapshot id
        -> a snapshot was mutated in place. REGRESSED, and rightly: the id
           cannot show this and the hash can.

    a table was ADDED to the set being hashed
        -> the image now hashes something it did not before. Nothing was
           mutated; the instrument changed.

On 2026-10-22 the second was reported as the first. The NFL came back
"REGRESSED - a snapshot mutated in place" on a deploy where all three tables
present in both captures were byte-identical and the only difference was a new
key, nfl_quarantine.csv -> "absent", for an optional table the served snapshot
predates. Both snapshot ids were 2026-10-22T0711.

A FALSE REGRESSED IS WORSE THAN A FALSE CANNOT COMPARE, because the only
obvious way to clear it is to recapture the baseline - which is the one thing
the gate's own rules forbid. So mutation is now judged on the tables present in
both captures, and a changed table SET withholds judgement instead.

This verifier tests the function directly. It is a pure function of two
dicts, so there is nothing to mock and nothing to warm.

    python ml-training/wnba/verify_gate_comparability.py
"""
import contextlib
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nba_regression_gate as G  # noqa: E402

RESULTS = []


def record(name, passed, detail):
    RESULTS.append((name, passed, detail))
    print("  %-4s %-58s %s" % ("PASS" if passed else "FAIL", name, detail))


def context(snapshot, tables, **rest):
    """One league's capture context, with everything else held equal.

    Every field league_comparability reads is set to the same value on both
    sides unless a case overrides it, so a verdict can only be attributable to
    the tables. A case that forgot one would otherwise pass for the wrong
    reason.
    """
    base = {
        "snapshot": snapshot,
        "table_sha256": tables,
        "data_as_of": "2026-10-21",
        "model_sha256": {"margin": "aaa"},
        "availability_state": "used",
        "availability_report": "2026-10-22T16:00:00",
        "requested": [("ATL", "BOS", "2026-10-22")],
    }
    base.update(rest)
    return base


THREE = {"a.csv": "1", "b.csv": "2", "c.csv": "3"}
FOUR = {"a.csv": "1", "b.csv": "2", "c.csv": "3", "d.csv": "absent"}
CHANGED = {"a.csv": "1", "b.csv": "CHANGED", "c.csv": "3"}


def verdict(before, after):
    """(comparable, reasons, notes, mutated) - the function's own 4-tuple.

    `mutated` is a SEPARATE flag rather than a value of `comparable`, which is
    the distinction this whole file turns on: the caller prints REGRESSED off
    `mutated` and CANNOT COMPARE off `comparable`. A harness that collapsed
    them could not tell the two verdicts apart - and it did not, at first.
    """
    return G.league_comparability(before, after)


REFRESH_RUN = {
    "last_verdict_at": "2026-10-22T07:27:44+00:00",
    "outcome": "advanced",
    "snapshot": "2026-10-22T0711",
    "leagues": {"nba": "advanced", "wnba": "unchanged",
                "gleague": "unchanged", "nfl": "unchanged"},
}


def health(**refresh) -> dict:
    """An inference /health body carrying only the refresh block that matters.

    Built rather than captured, because the two cases below differ in ONE
    field each and a live pair cannot be held still to that precision - which
    is the whole reason the gate needs the distinction in the first place.
    """
    block = dict(REFRESH_RUN)
    block.update(refresh)
    return {"refresh": block}


def refresh_block_rule() -> None:
    """The refresh block: the clock is ignored, the run identity is not.

    THE FIRST CASE IS THE ONE THAT WOULD HAVE BITTEN DAILY. Two captures are
    minutes apart by design - warm, capture, deploy, compare - so
    `hours_since` differs on every single pair the gate will ever see. If it
    entered the comparison the gate would be red or unusable always.

    THE SECOND IS THE ONE THAT WAS SILENTLY MISSING. `refresh` was put in
    HEALTH_CONTEXT_FIELDS, which only strips a field from the body diff; it
    compares nothing. So a refresh running between the two captures - new
    verdict, possibly a new snapshot swapped in underneath - produced no
    reason, no note and a clean PASS. A changed `last_verdict_at` has to
    withhold judgement instead.
    """
    # ---- identical except hours_since -> must PASS
    before = context("S1", THREE,
                     refresh_verdict=G.refresh_identity(health()))
    after = context("S1", THREE,
                    refresh_verdict=G.refresh_identity(health()))
    ok, reasons, _n, mut = verdict(before, after)
    record("refresh.hours_since 1.2 -> 1.4 only -> PASS",
           ok is True and not reasons and not mut,
           f"comparable={ok}, reasons={reasons}")

    # AND THE REASON IT PASSES IS THAT THE FIELD NEVER REACHES THE CONTEXT,
    # which is asserted rather than inferred from the verdict: a PASS would
    # look identical if `refresh` were dropped from /health altogether.
    ident = G.refresh_identity({"refresh": {
        "last_verdict_at": "2026-10-22T07:27:44+00:00",
        "outcome": "advanced", "snapshot": "2026-10-22T0711",
        "leagues": {}, "hours_since": 1.2, "state": "ok",
        "note": "last verdict 1.2h ago"}})
    record("the clock fields are absent from the captured identity",
           "hours_since" not in ident and "state" not in ident
           and "note" not in ident and ident["last_verdict_at"],
           f"identity keys: {sorted(ident)}")

    # ---- differing in the verdict timestamp -> must CANNOT COMPARE
    ok, reasons, _n, mut = verdict(
        context("S1", THREE, refresh_verdict=G.refresh_identity(health())),
        context("S1", THREE, refresh_verdict=G.refresh_identity(
            health(last_verdict_at="2026-10-22T13:10:36+00:00"))))
    record("refresh last-verdict timestamp differs -> CANNOT COMPARE",
           ok is False and mut is False
           and any("a refresh ran between the captures" in r
                   for r in reasons),
           f"comparable={ok}, mutated={mut}, "
           f"reason={reasons[0][:70] if reasons else '-'}")
    record("it is CANNOT COMPARE and NOT mutated - no defect is asserted",
           mut is False, f"mutated={mut}")

    # A `nothing-new` run between captures moves the verdict and nothing else.
    # Still withheld, deliberately, and the reason names both outcomes so it
    # is cheap to recognise.
    ok, reasons, _n, _m = verdict(
        context("S1", THREE, refresh_verdict=G.refresh_identity(health())),
        context("S1", THREE, refresh_verdict=G.refresh_identity(health(
            outcome="nothing-new",
            last_verdict_at="2026-10-22T13:10:36+00:00"))))
    record("a `nothing-new` run between captures is named in the reason",
           ok is False and any("advanced at" in r and "nothing-new at" in r
                               for r in reasons),
           f"reason={reasons[0][:76] if reasons else '-'}")

    # ---- the pre-upgrade boundary: an old image sends no refresh block.
    ok, reasons, _n, _m = verdict(
        context("S1", THREE, refresh_verdict=G.refresh_identity({})),
        context("S1", THREE, refresh_verdict=G.refresh_identity(health())))
    record("no refresh block -> a real one: withheld, not passed",
           ok is False and any("a refresh ran" in r for r in reasons),
           f"comparable={ok}, {{}} vs a real block")

    # ---- PLANT: let the clock back into the identity and the first case,
    # which is the pair every real run produces, must go red.
    original = G.refresh_identity

    def with_clock(h):
        block = (h or {}).get("refresh")
        if not isinstance(block, dict):
            return {}
        out = dict(original(h))
        out["hours_since"] = block.get("hours_since")
        return out

    G.refresh_identity = with_clock
    try:
        ok, reasons, _n, _m = verdict(
            context("S1", THREE, refresh_verdict=G.refresh_identity(
                {"refresh": dict(REFRESH_RUN, hours_since=1.2)})),
            context("S1", THREE, refresh_verdict=G.refresh_identity(
                {"refresh": dict(REFRESH_RUN, hours_since=1.4)})))
        record("PLANT: with hours_since in the identity, the clock pair "
               "goes red",
               ok is False,
               "red, so excluding it is what makes case 1 pass"
               if ok is False else "STILL PASSES - the exclusion is not "
                                   "what is doing the work")
    finally:
        G.refresh_identity = original

    # ---- and the guard against the guard
    ok, reasons, _n, _m = verdict(
        context("S1", THREE, refresh_verdict=G.refresh_identity(
            {"refresh": dict(REFRESH_RUN, hours_since=1.2)})),
        context("S1", THREE, refresh_verdict=G.refresh_identity(
            {"refresh": dict(REFRESH_RUN, hours_since=1.4)})))
    record("PLANT reverted: the clock pair passes again",
           ok is True and not reasons,
           "green, so the red above was the plant and not the harness")

    # ---- THE DEPLOY'S OWN PAIR, which the gate got wrong until 2026-10-24.
    # A deploy moves the pointer BY HAND: the snapshot id differs between the
    # two captures while the WNBA, G League and NFL tables are byte-identical
    # and no refresh has run. Those three leagues MUST pass - the switch
    # runbook treats anything else as fatal - and they came out CANNOT
    # COMPARE, because the refresh identity carried the snapshot id and fired
    # on the pointer move alone. The reason string was its own tell: "a
    # refresh ran between the captures (advanced at t -> advanced at t)".
    run = {"last_verdict_at": "2026-10-25T06:14:00+00:00",
           "outcome": "advanced", "leagues": {}}
    ok, reasons, notes, mut = verdict(
        context("2026-10-24T0803", THREE, refresh_verdict=dict(run)),
        context("2026-10-25T0901", THREE, refresh_verdict=dict(run)))
    record("the DEPLOY pair: id moves, tables identical -> PASS",
           ok is True and not reasons and not mut,
           f"comparable={ok}, reasons={reasons}")
    record("the deploy pair still says the id moved",
           any("snapshot id moved" in n for n in notes),
           notes[0][:66] if notes else "no note")

    ok, reasons, _n, _m = verdict(
        context("2026-10-24T0803", THREE,
                refresh_verdict=dict(run, last_verdict_at="t1")),
        context("2026-10-25T0901", THREE,
                refresh_verdict=dict(run, last_verdict_at="t2")))
    record("control: a real refresh between captures is still caught",
           ok is False and any("a refresh ran" in r for r in reasons),
           reasons[0][:66] if reasons else "NOT CAUGHT")

    # PLANT: put the snapshot id back and the deploy pair must go red again,
    # which is what proves the exclusion is load-bearing rather than tidy.
    original = G.refresh_identity

    def with_snapshot(h):
        block = (h or {}).get("refresh")
        if not isinstance(block, dict):
            return {}
        out = dict(original(h))
        out["snapshot"] = block.get("snapshot")
        return out

    G.refresh_identity = with_snapshot
    try:
        a = G.refresh_identity({"refresh": dict(run, snapshot="S1")})
        b = G.refresh_identity({"refresh": dict(run, snapshot="S2")})
        ok, reasons, _n, _m = verdict(
            context("2026-10-24T0803", THREE, refresh_verdict=a),
            context("2026-10-25T0901", THREE, refresh_verdict=b))
        record("PLANT: with the snapshot id back in, the deploy pair goes red",
               ok is False,
               "red, so excluding it is what makes the deploy comparable"
               if ok is False else "STILL PASSES - exclusion does nothing")
    finally:
        G.refresh_identity = original


# ==========================================================================
# the difference list is COMPLETE, and the heading counts fields
# ==========================================================================

def nine_field_bodies():
    """A capture pair whose one body differs in exactly nine fields.

    NINE, BECAUSE THE TWO OLD CAPS WERE SIX AND EIGHT. A body with nine
    changed fields is the smallest one that both branches would have
    truncated, and the REGRESSED branch's cut at eight is the dangerous
    one - seven markets cannot reach it today, so the defect was invisible
    and would have surfaced only once a model family grew.
    """
    fields = {f"metric_{i}": float(i) for i in range(9)}
    moved = {name: value + 1.0 for name, value in fields.items()}
    tables = {"games_final.csv": "aaa"}
    before = {
        "context": {"leagues": {"nba": context("2026-10-29T1027", tables)},
                    "schedule_count": 1311, "teams_count": 30},
        "bodies": {"nba.inference.0": dict(fields),
                   "inference_health": health(), "backend_health": {}},
    }
    after = {
        "context": {"leagues": {"nba": context("2026-10-29T1027", tables)},
                    "schedule_count": 1311, "teams_count": 30},
        "bodies": {"nba.inference.0": dict(moved),
                   "inference_health": health(), "backend_health": {}},
    }
    return before, after


def every_differing_field_is_printed() -> None:
    before, after = nine_field_bodies()

    printed, code = run_compare(before, after)

    record("nine changed fields -> REGRESSED, not a silent pass",
           code == G.EXIT_REGRESSED, f"exit {code}")
    shown = [line for line in printed.splitlines()
             if "nba.inference.0  metric_" in line]
    record("all NINE fields are printed, not six and not eight",
           len(shown) == 9, f"{len(shown)} field line(s) printed")
    record("the heading counts FIELDS as well as bodies",
           "9 field(s) differ" in printed,
           [line.strip() for line in printed.splitlines()
            if "field(s) differ" in line][:1])
    record("and no field is dropped without saying so",
           "more field(s) not shown" not in printed,
           "there is no cap to announce")


def every_field_is_printed_on_cannot_compare() -> None:
    """THE OTHER BRANCH, which is where the A3 switch lost total_rebounds.

    A changed snapshot id with changed tables makes the league CANNOT
    COMPARE, and the bodies are then shown as information - which is exactly
    the branch that was capped at six.
    """
    before, after = nine_field_bodies()
    after["context"]["leagues"]["nba"] = context(
        "2026-10-29T2113", {"games_final.csv": "bbb"})

    printed, code = run_compare(before, after)

    record("a CANNOT COMPARE pair still lists every field",
           len([line for line in printed.splitlines()
                if "nba.inference.0  metric_" in line]) == 9,
           f"exit {code}, branch = "
           + ("CANNOT COMPARE" if code == G.EXIT_CANNOT else f"exit {code}"))
    record("its heading counts fields too",
           "9 field(s) differ" in printed,
           [line.strip() for line in printed.splitlines()
            if "field(s) differ" in line][:1])


def plant_field_cap() -> None:
    """Put the caps back, and both completeness checks must go red."""
    real = G.print_fields

    def capped(key, fields, limit=None):
        """The defect as it shipped: six per body, silently."""
        for field in fields[:6]:
            print(f"             {key}  {field}")

    before, after = nine_field_bodies()
    G.print_fields = capped
    try:
        printed, _code = run_compare(before, after)
    finally:
        G.print_fields = real

    shown = len([line for line in printed.splitlines()
                 if "nba.inference.0  metric_" in line])
    if shown == 9:
        raise AssertionError(
            "the plant printed nine fields anyway, so the check is not "
            "reading print_fields")
    record("PLANT the six-field cap is back -> only 6 of 9 printed (red)",
           shown == 6, f"{shown} field line(s) with the cap reinstated")

    # AND A CAP THAT DOES ANNOUNCE ITSELF IS ACCEPTABLE, which is what the
    # helper's `limit` exists for. This is the shape a future readability cap
    # must have: the count it dropped, printed.
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        real("k", [f"f{i}" for i in range(9)], limit=4)
    text = buffer.getvalue()
    record("an announced cap prints what it dropped",
           "+5 more field(s) not shown" in text,
           text.strip().splitlines()[-1].strip())


# ==========================================================================
# the recorded NBA reference is the post-A3 one
# ==========================================================================

WITHDRAWN_REFERENCES = (0.4191701412200928, 0.42541608214378357)


def recorded_reference() -> None:
    """A dated record, asserted so a silent revert is impossible.

    NOT A COMPARISON AGAINST A LIVE PREDICTION, and it cannot be: checked
    2026-10-29, `HISTORICAL_NBA_REFERENCE` is read nowhere in the repository,
    and this gate has no live reference to derive one from - it compares two
    captures by design. So what is testable is that the recorded number is
    the current one and not a withdrawn one, which is the failure that
    actually happened: the A3 switch rebaselined the reference on 2026-10-25
    and this constant still held the pre-A3 value four days later.
    """
    record("the recorded reference is the post-A3 value",
           G.HISTORICAL_NBA_REFERENCE == 0.48472610116004944,
           repr(G.HISTORICAL_NBA_REFERENCE))
    record("and is not either withdrawn value",
           G.HISTORICAL_NBA_REFERENCE not in WITHDRAWN_REFERENCES,
           f"withdrawn: {WITHDRAWN_REFERENCES}")

    # THE FIXTURE IT BELONGS TO, which a bare float cannot carry. ATL vs BOS
    # is MATCHUPS["nba"][0], and that is the point of the pairing: the
    # reference is the gate's own first body, not whatever the schedule
    # happened to order first.
    record("the fixture is the gate's own first NBA matchup",
           G.HISTORICAL_NBA_REFERENCE_FIXTURE == G.MATCHUPS["nba"][0],
           f"{G.HISTORICAL_NBA_REFERENCE_FIXTURE} vs "
           f"{G.MATCHUPS['nba'][0]}")
    record("the date and snapshot are recorded beside it",
           G.HISTORICAL_NBA_REFERENCE_DATE == "2026-10-25"
           and G.HISTORICAL_NBA_REFERENCE_SNAPSHOT == "2026-10-25T0945",
           f"{G.HISTORICAL_NBA_REFERENCE_DATE} / "
           f"{G.HISTORICAL_NBA_REFERENCE_SNAPSHOT}")

    # 17 SIGNIFICANT DIGITS SURVIVED. A value recorded through PowerShell's
    # -f operator loses the last digits and does not read back as the same
    # double, which is the whole point of recording it.
    record("the value round-trips at 17 significant digits",
           float(repr(G.HISTORICAL_NBA_REFERENCE))
           == G.HISTORICAL_NBA_REFERENCE
           and len(repr(G.HISTORICAL_NBA_REFERENCE).split(".")[1]) >= 16,
           repr(G.HISTORICAL_NBA_REFERENCE))


def plant_withdrawn_reference() -> None:
    """Put the withdrawn value back; the check must go red."""
    real = G.HISTORICAL_NBA_REFERENCE
    G.HISTORICAL_NBA_REFERENCE = 0.4191701412200928
    try:
        caught = (G.HISTORICAL_NBA_REFERENCE == 0.48472610116004944)
        withdrawn = (G.HISTORICAL_NBA_REFERENCE in WITHDRAWN_REFERENCES)
    finally:
        G.HISTORICAL_NBA_REFERENCE = real
    record("PLANT the withdrawn pre-A3 value is back -> caught (red)",
           caught is False and withdrawn is True,
           "0.4191701412200928 fails both assertions")


# ==========================================================================
# the backend half uses existing fixtures, and says so when it cannot
# ==========================================================================

def fixture(league, home, away, date, predictable=None):
    """One row in the shape /api/games/schedule returns."""
    row = {"league": league, "homeTeamId": home, "awayTeamId": away,
           "gameDate": date}
    if predictable is not None:
        row["predictable"] = predictable
    return row


def backend_half_picks_existing_fixtures() -> None:
    schedule = [
        fixture("nba", 1610612757, 1610612763, "2026-10-29"),
        fixture("nba", 1610612737, 1610612739, "2026-10-29"),
        fixture("nba", 1610612740, 1610612758, "2026-10-29"),
        fixture("nba", 1610612737, 1610612744, "2026-11-04"),
    ]

    chosen, skipped = G.backend_fixtures(schedule, "nba", "2026-10-29")
    record("two fixtures are chosen from the cached schedule",
           len(chosen) == 2 and skipped is None,
           f"{chosen}, skip={skipped}")
    teams = {t for home, away, _d in chosen for t in (home, away)}
    record("with four distinct teams, like MATCHUPS",
           len(teams) == 4, f"{sorted(teams)}")
    record("and only on the one permitted date",
           all(d == "2026-10-29" for _h, _a, d in chosen),
           "cutoff + 1, which is all MAX_DAYS_AHEAD allows")

    # THE WNBA OUT OF SEASON: its sync caches nothing at all.
    chosen, skipped = G.backend_fixtures(schedule, "wnba", "2026-09-25")
    record("a league with NO cached fixture gets its own CODE",
           chosen == [] and skipped
           and skipped["code"] == G.SKIP_NO_CACHED_FIXTURES,
           (skipped or {}).get("code"))
    record("and a readable reason beside it",
           bool(skipped) and "no cached wnba fixture" in skipped["reason"],
           (skipped or {}).get("reason", "")[:68])

    # THE G LEAGUE OUT OF SEASON: fixtures exist, none on the one date.
    gleague = [fixture("gleague", 1612709890, 1612709905, "2027-01-08")]
    chosen, skipped = G.backend_fixtures(gleague, "gleague", "2026-03-29")
    record("fixtures but none on the date is a DIFFERENT code",
           chosen == [] and skipped
           and skipped["code"] == G.SKIP_NONE_ON_RULE_DATE,
           (skipped or {}).get("code"))
    record("and the COUNT lives only in the reason, never in the code",
           "none of its 1 cached" in skipped["reason"]
           and not any(ch.isdigit() for ch in skipped["code"]),
           f"code={skipped['code']!r} carries no number")

    # THE NFL READS ITS OWN FLAG, NOT THE DATE. Its rule is a dependency on
    # both teams' earlier games, so a date filter would refuse everything.
    nfl = [fixture("nfl", 1613000012, 1613000005, "2026-10-29", True),
           fixture("nfl", 1613000011, 1613000018, "2026-11-01", True),
           fixture("nfl", 1613000001, 1613000002, "2026-12-20", False)]
    chosen, skipped = G.backend_fixtures(nfl, "nfl", "1999-01-01")
    record("the NFL picks by its predictable FLAG, ignoring the date",
           len(chosen) == 2 and skipped is None
           and all(d != "1999-01-01" for _h, _a, d in chosen),
           f"{chosen}")

    unpredictable = [fixture("nfl", 1613000001, 1613000002, "2026-12-20",
                             False)]
    chosen, skipped = G.backend_fixtures(unpredictable, "nfl", None)
    record("an NFL week with nothing predictable yet has its own code",
           chosen == [] and skipped
           and skipped["code"] == G.SKIP_NONE_PREDICTABLE
           and "dependency rule" in skipped["reason"],
           (skipped or {}).get("code"))

    # A SHORTFALL IS NOT A SILENT SUCCESS. One usable fixture where two were
    # asked for still returns a reason alongside what it found.
    one = [fixture("nba", 1610612737, 1610612739, "2026-10-29")]
    chosen, skipped = G.backend_fixtures(one, "nba", "2026-10-29")
    record("finding ONE where two were asked for is reported too",
           len(chosen) == 1 and skipped
           and skipped["code"] == G.SKIP_FEWER_THAN_REQUESTED
           and "only 1 of 2" in skipped["reason"],
           (skipped or {}).get("code"))

    # An unreadable schedule is an unknown, not an empty list.
    chosen, skipped = G.backend_fixtures(None, "nba", "2026-10-29")
    record("an unreadable schedule says so rather than reading as empty",
           chosen == [] and skipped
           and skipped["code"] == G.SKIP_SCHEDULE_UNREADABLE,
           (skipped or {}).get("code"))

    # EVERY CODE IS A BARE WORD. A code with a number in it would be the
    # original defect wearing a new name.
    codes = [value for name, value in vars(G).items()
             if name.startswith("SKIP_")]
    record("every SKIP_* code is a stable bare word with no digits",
           len(codes) >= 6
           and all(isinstance(c, str) and c and not any(ch.isdigit()
                                                        for ch in c)
                   for c in codes),
           f"{sorted(codes)}")


def skipped_half_reaches_the_verdict() -> None:
    """A half captured on one side and skipped on the other is not a pass."""
    tables = {"games_final.csv": "aaa"}

    def side(code, fixtures, reason=None):
        block = context("2026-10-29T1027", tables)
        block["backend_skip_code"] = code
        block["backend_skip_reason"] = reason
        block["backend_fixtures"] = fixtures
        bodies = {"inference_health": health(), "backend_health": {}}
        if code:
            bodies["nba.backend.none"] = {"__skipped__": code}
        return {"context": {"leagues": {"nba": block},
                            "schedule_count": 1311, "teams_count": 30},
                "bodies": bodies}

    used = [{"home": 1, "away": 2, "date": "2026-10-29"}]
    before = side(None, used)
    after = side(G.SKIP_NO_CACHED_FIXTURES, [],
                 "the backend has no cached nba fixture at all")

    ok, reasons, _notes, mutated = G.league_comparability(
        before["context"]["leagues"]["nba"], after["context"]["leagues"]["nba"])
    record("captured then skipped -> CANNOT COMPARE, not PASS",
           ok is False and mutated is False,
           f"comparable={ok}, mutated={mutated}")
    record("and the reason NAMES the change in availability",
           any("availability changed" in r for r in reasons),
           [r[:72] for r in reasons][:2])

    # A DIFFERENT FIXTURE IS ALSO NOT COMPARABLE. The choice comes from the
    # live schedule, so it moves when a game is played or a week flexes -
    # neither of which is a code change.
    moved = side(None, [{"home": 3, "away": 4, "date": "2026-10-30"}])
    ok, reasons, _notes, mutated = G.league_comparability(
        before["context"]["leagues"]["nba"], moved["context"]["leagues"]["nba"])
    record("a different existing fixture -> CANNOT COMPARE",
           ok is False and mutated is False
           and any("different existing fixtures" in r for r in reasons),
           [r[:72] for r in reasons][:1])

    # AND THE SAME FIXTURE IS STILL COMPARABLE, or the rule above would make
    # every pair incomparable and the gate could never say PASS.
    ok, reasons, _notes, mutated = G.league_comparability(
        before["context"]["leagues"]["nba"],
        side(None, used)["context"]["leagues"]["nba"])
    record("the same fixture on both sides is still comparable",
           ok is True and not reasons, f"comparable={ok}")

    # ------------------------------------------------------------------
    # DEPLOY C'S FALSE REGRESSED, AS A CASE. Two captures that differ ONLY
    # in the cached fixture count must PASS. On 2026-10-30 this came back
    # REGRESSED for the G League: the skip body held the prose, the backend
    # restart synced 8 fixtures between the captures, and "none of its 349
    # cached fixture(s)" became "357".
    # ------------------------------------------------------------------
    counted_349 = side(G.SKIP_NONE_ON_RULE_DATE, [],
                       "the backend has gleague fixtures but none of its 349 "
                       "cached fixture(s) falls on 2026-03-29, the only date "
                       "MAX_DAYS_AHEAD permits")
    counted_357 = side(G.SKIP_NONE_ON_RULE_DATE, [],
                       "the backend has gleague fixtures but none of its 357 "
                       "cached fixture(s) falls on 2026-03-29, the only date "
                       "MAX_DAYS_AHEAD permits")
    ok, reasons, _notes, mutated = G.league_comparability(
        counted_349["context"]["leagues"]["nba"],
        counted_357["context"]["leagues"]["nba"])
    record("349 -> 357 cached fixtures: still COMPARABLE",
           ok is True and not reasons and mutated is False,
           f"comparable={ok}, reasons={reasons}")

    printed, code = run_compare(counted_349, counted_357)
    record("and the whole pair PASSES - the count is not a regression",
           code == G.EXIT_PASS,
           f"exit {code}: "
           + ("PASS" if code == G.EXIT_PASS else "NOT PASS"))
    record("the differing count is still PRINTED, from the reason",
           "357" in printed and "349" in printed,
           "both counts readable in the output, neither compared")

    # AND TWO DIFFERENT CODES ARE NOT COMPARABLE, which is the other half:
    # the half was unavailable for two different reasons, so its bodies are
    # not comparable even though both are absent.
    other_code = side(G.SKIP_NO_CACHED_FIXTURES, [],
                      "the backend has no cached gleague fixture at all")
    ok, reasons, _notes, mutated = G.league_comparability(
        counted_349["context"]["leagues"]["nba"],
        other_code["context"]["leagues"]["nba"])
    record("a DIFFERENT skip code -> CANNOT COMPARE, not PASS",
           ok is False and mutated is False
           and any("availability changed" in r for r in reasons),
           [r[:72] for r in reasons][:1])
    _printed, code = run_compare(counted_349, other_code)
    record("and the pair's verdict is CANNOT COMPARE",
           code == G.EXIT_CANNOT, f"exit {code}")

    # SKIPPED ON BOTH SIDES IS COMPARABLE TOO: out of season that is the
    # steady state, and a gate that withheld on it could never pass the WNBA.
    both = side(G.SKIP_NO_CACHED_FIXTURES, [],
                "the backend has no cached nba fixture at all")
    ok, reasons, _notes, _m = G.league_comparability(
        both["context"]["leagues"]["nba"], both["context"]["leagues"]["nba"])
    record("skipped on BOTH sides is comparable, not withheld",
           ok is True and not reasons,
           "out of season this is the steady state")

    # IT IS SAID OUT LOUD, EVERY RUN. The context carrying it is half the
    # requirement; compare() printing it is the other half.
    printed, _code = run_compare(before, after)
    record("compare() prints the skip for the side that skipped",
           "backend half SKIPPED in the current" in printed
           and G.SKIP_NO_CACHED_FIXTURES in printed,
           [line.strip() for line in printed.splitlines()
            if "SKIPPED" in line][:1])
    printed, _code = run_compare(side(None, used), side(None, used))
    record("and names the fixtures it used when it did not skip",
           "backend half used 1 existing fixture" in printed,
           [line.strip() for line in printed.splitlines()
            if "backend half used" in line][:1])


def plant_skip_is_silent() -> None:
    """Drop the two keys from the compared context, and the mismatch passes."""
    tables = {"games_final.csv": "aaa"}

    def side(code, fixtures):
        block = context("2026-10-29T1027", tables)
        block["backend_skip_code"] = code
        block["backend_skip_reason"] = "prose with 349 in it"
        block["backend_fixtures"] = fixtures
        return block

    before = side(None, [{"home": 1, "away": 2, "date": "2026-10-29"}])
    after = side(G.SKIP_NO_CACHED_FIXTURES, [])

    stripped_before = {k: v for k, v in before.items()
                       if not k.startswith("backend_")}
    stripped_after = {k: v for k, v in after.items()
                      if not k.startswith("backend_")}
    ok, reasons, _notes, _m = G.league_comparability(stripped_before,
                                                     stripped_after)
    if ok is False:
        raise AssertionError(
            "the stripped pair was still incomparable, so the check is not "
            "reading the backend_ keys")
    record("PLANT the backend keys out of the context -> the mismatch "
           "PASSES silently",
           ok is True and not reasons,
           "so carrying them is what makes a skipped half visible")


def plant_reason_in_the_compared_body() -> None:
    """Put the prose back where the code is, and deploy C's red returns.

    TWO PLANTS, because the defect had two halves: the compared BODY held the
    prose, and the compared CONTEXT would have too if the code had not been
    separated out. Each is restored on its own, so each assertion is shown to
    be reading the thing it names.
    """
    tables = {"games_final.csv": "aaa"}

    def side(reason, in_body):
        block = context("2026-10-29T1027", tables)
        block["backend_skip_code"] = G.SKIP_NONE_ON_RULE_DATE
        block["backend_skip_reason"] = reason
        block["backend_fixtures"] = []
        bodies = {"inference_health": health(), "backend_health": {}}
        bodies["nba.backend.none"] = {"__skipped__": in_body}
        return {"context": {"leagues": {"nba": block},
                            "schedule_count": 1311, "teams_count": 30},
                "bodies": bodies}

    prose_349 = "none of its 349 cached fixture(s) falls on 2026-03-29"
    prose_357 = "none of its 357 cached fixture(s) falls on 2026-03-29"

    # (a) THE PROSE IN THE BODY, which is what shipped. The counts differ, the
    #     context is identical, so the league reports REGRESSED.
    _printed, code = run_compare(side(prose_349, prose_349),
                                 side(prose_357, prose_357))
    if code == G.EXIT_PASS:
        raise AssertionError(
            "the plant still passed, so the check is not reading the body")
    record("PLANT the prose back in the compared body -> REGRESSED (red)",
           code == G.EXIT_REGRESSED,
           f"exit {code}, which is the false red deploy C hit")

    # (b) THE CONTROL, same pair with the code in the body: PASS. Without
    #     this the plant above could be red for any reason at all.
    _printed, code = run_compare(side(prose_349, G.SKIP_NONE_ON_RULE_DATE),
                                 side(prose_357, G.SKIP_NONE_ON_RULE_DATE))
    record("control: the same pair with the CODE in the body -> PASS",
           code == G.EXIT_PASS,
           f"exit {code}, so the red above is the prose and nothing else")

    # (c) THE REASON MUST NOT BE NAMED BY league_comparability. Read out of
    #     the source rather than inferred from behaviour: a future edit that
    #     adds it would reintroduce the defect for every league at once.
    # THE SLICE HAS TO STOP AT THE NEXT def, and the first version of this
    # check did not: it ran to `strip_health_context`, which is 150 lines
    # further down, so it swallowed `compare()` - where the reason is
    # legitimately PRINTED - and failed for the opposite of a defect.
    source = Path(G.__file__).read_text(encoding="utf-8")
    start = source.index("def league_comparability")
    end = source.index(chr(10) + "def ", start + 1)
    # COMMENTS STRIPPED, because the function's own comment says that
    # backend_skip_reason is deliberately not compared - and a check a
    # comment can turn red is a check about prose rather than about code.
    # That is what the first version of this did.
    body = chr(10).join(
        line for line in source[start:end].splitlines()
        if not line.strip().startswith("#"))
    record("league_comparability never names backend_skip_reason",
           "backend_skip_reason" not in body
           and "backend_skip_code" in body,
           "the code is compared; the prose is not reachable from here")


def run_compare(before, after):
    """Drive the gate's own `compare` on two in-memory captures.

    THE REAL FUNCTION, NOT A REIMPLEMENTATION. `compare` is a pure function
    of two capture dicts, so there is nothing to stub: stdout is redirected
    and the verdict is the one the gate would have reached.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = G.compare(before, after)
    return buffer.getvalue(), code


def main() -> int:
    print("THE GATE'S TABLE RULE")
    print()

    # ---- row 1: same id, same hash -> comparable
    ok, reasons, _n, mut = verdict(context("S1", THREE),
                                   context("S1", THREE))
    record("same id, same tables -> comparable",
           ok is True and not reasons and not mut,
           f"comparable={ok}, mutated={mut}, reasons={reasons}")

    # ---- row 2: THE ONE WORTH HAVING. Still has to bite.
    ok, reasons, _n, mut = verdict(context("S1", THREE),
                                   context("S1", CHANGED))
    record("same id, a SHARED table's hash changed -> MUTATED",
           mut is True and ok is False,
           f"mutated={mut}, comparable={ok} - the detector still bites")

    # ---- row 3: diff id, same hash -> comparable, and say so
    ok, reasons, notes, mut = verdict(context("S1", THREE),
                                      context("S2", THREE))
    record("different id, identical tables -> comparable with a note",
           ok is True and not reasons and not mut
           and any("snapshot id moved" in n for n in notes),
           f"comparable={ok}, notes={notes[:1]}")

    # ---- row 4: diff id, diff hash -> cannot compare, data advanced
    ok, reasons, _n, mut = verdict(context("S1", THREE),
                                   context("S2", CHANGED))
    record("different id, different tables -> cannot compare, NOT mutated",
           ok is False and mut is False
           and any("data advanced" in r for r in reasons),
           f"comparable={ok}, mutated={mut}, reasons={reasons}")

    # ---- row 5: THE DEFECT. A grown table set is NOT a mutation.
    ok, reasons, _n, mut = verdict(context("S1", THREE),
                                   context("S1", FOUR))
    record("same id, a table ADDED to the set -> NOT mutated",
           mut is False,
           f"mutated={mut} - a newly hashed table is not a mutation")
    record("the added table makes the league CANNOT COMPARE, with a reason",
           ok is False and any("set of served tables changed" in r
                               for r in reasons),
           f"reasons={reasons}")
    record("the reason NAMES the table that appeared",
           any("d.csv" in r for r in reasons), f"{reasons}")

    # ---- row 5, the alarming direction. A table VANISHING must never pass.
    ok, reasons, _n, mut = verdict(context("S1", FOUR),
                                   context("S1", THREE))
    record("a table REMOVED from the set -> cannot compare, named",
           ok is False and mut is False
           and any("no longer hashed" in r for r in reasons)
           and any("d.csv" in r for r in reasons),
           f"comparable={ok}, mutated={mut}, reasons={reasons}")

    # ---- the two must not mask each other: a set change AND a real mutation
    # under one id has to come out as the mutation, because that is the
    # finding. If the set change won, a mutated table could be hidden by
    # adding one.
    ok, reasons, _n, mut = verdict(
        context("S1", THREE),
        context("S1", {"a.csv": "1", "b.csv": "CHANGED", "c.csv": "3",
                       "d.csv": "absent"}))
    record("a set change CANNOT MASK a mutation under the same id",
           mut is True, f"mutated={mut} - the mutation still wins")

    # ---- the live shape, exactly as observed on 2026-10-22
    live_before = {"nfl_games_final.csv": "326117072bdfd0ec",
                   "nfl_fixtures.csv": "9b28ee06f6703419",
                   "nfl_franchise_identity.csv": "0f0a5402b8fb3fdb"}
    live_after = dict(live_before, **{"nfl_quarantine.csv": "absent"})
    ok, reasons, _n, mut = verdict(context("2026-10-22T0711", live_before),
                                   context("2026-10-22T0711", live_after))
    record("the real 2026-10-22 NFL deploy no longer reads as REGRESSED",
           mut is False and ok is False
           and any("nfl_quarantine.csv" in r for r in reasons),
           f"mutated={mut}, comparable={ok}, "
           f"reason={reasons[0][:66] if reasons else '-'}")

    print()
    refresh_block_rule()
    print()
    every_differing_field_is_printed()
    every_field_is_printed_on_cannot_compare()
    plant_field_cap()
    print()
    recorded_reference()
    plant_withdrawn_reference()
    print()
    backend_half_picks_existing_fixtures()
    skipped_half_reaches_the_verdict()
    plant_skip_is_silent()
    plant_reason_in_the_compared_body()

    passed = sum(1 for _n, ok_, _d in RESULTS if ok_)
    print()
    print(f"  {passed} of {len(RESULTS)} checks passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:                                  # noqa: BLE001
        print(f"\nCRASHED before a verdict: {type(error).__name__}: {error}")
        raise SystemExit(2)
