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
