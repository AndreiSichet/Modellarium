r"""/health must be cheap, and cheap must not mean different.

WHY THIS FILE EXISTS. On 2026-11-06 the daily refresh swapped in a good
snapshot, restarted inference, asked `/health` twenty times, read `{}` every
time, rolled the swap back and printed *"THE SERVICE IS STILL NOT SERVING.
Manual attention needed."* The service was serving: the container logs show
both boots completing, every league loaded and `/health` answering 200. The
whole failure was that `/health` cost 4.34 s against a 5 s per-attempt
timeout, and `TimeoutError` is an `OSError`, which the probe caught and
flattened to `{}`.

So two things are asserted here and they are not the same thing:

  THE BUDGET - `/health`'s snapshot-derived work stays well inside the
  refresh's own per-attempt timeout. Both numbers come from
  `served_data`; neither is written again here.

  THE EQUIVALENCE - the indexes that made it cheap did not change a single
  answer. This is the half that matters more. A faster dependency rule that
  moved one fixture from predictable to blocked would be a far worse bug
  than the one being fixed, and it would be invisible: nobody reads 1,094
  flags. So the OLD implementation is loaded from git and run beside the new
  one over the whole real corpus, field by field.

Exit codes follow the project contract: 0 verdict reached, 1 a check failed,
2 crashed before comparing.

    python ml-training\verify_health_budget.py
    python ml-training\verify_health_budget.py --plant slow-health
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _part in ("ml-training", "ml-training/nhl", "ml-training/nfl",
              "data-pipeline"):
    sys.path.insert(0, str(ROOT / _part))

import served_data as S                              # noqa: E402

EXIT_OK, EXIT_FAILED, EXIT_CRASHED = 0, 1, 2

# A PLANT HAS TO CHANGE SOMETHING. An earlier draft listed `slow-health`
# and `no-snapshot-field` here; both are exercised unconditionally by
# `probe_reasons`, so selecting them changed nothing and the run reported
# 29 of 29 under a banner saying checks were expected to fail. The guard at
# the bottom of `main` caught it and returned 2 - which is the
# `verify_nhl_selection.py` lesson arriving a second time. A case that is
# always tested is a CHECK, not a plant.
PLANTS = ("collapse-reasons", "slow-handler", "revert-index")

# Fields this working tree adds to the NHL slate that HEAD does not have.
# EMPTY IS THE NORMAL STATE.
ADDED_ON_PURPOSE: list[str] = []

PASSED: list = []
FAILED: list = []


def record(name: str, passed: bool, detail: str = "") -> None:
    (PASSED if passed else FAILED).append(name)
    print(f"  {'PASS' if passed else 'FAIL'} {name:<60} {detail}")


# ------------------------------------------------------ the old implementation
WROTE: list = []


def load_from_git(relative: str, module_name: str, work: Path):
    """Import a module as it stands at HEAD, beside the working copy's.

    FROM GIT, NOT FROM A HAND-WRITTEN COPY. The question is whether the
    change in the working tree altered an answer, so the reference has to be
    the committed code and nothing else.

    WRITTEN BESIDE THE ORIGINAL RATHER THAN INTO A TEMP DIRECTORY, and that
    is not tidiness: these modules resolve their model directory and their
    data root RELATIVE TO THEIR OWN FILE, so a copy under %TEMP% went
    looking for a `models_nhl` beside itself and crashed. The copy is
    removed by `cleanup()` from a `finally`, and the pre-push gate refuses a
    dirty tree if one ever survives.
    """
    done = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8")
    if done.returncode != 0:
        raise RuntimeError(f"git show HEAD:{relative} failed: {done.stderr}")
    path = ROOT / Path(relative).parent / (module_name + ".py")
    path.write_text(done.stdout, encoding="utf-8")
    WROTE.append(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def cleanup() -> None:
    """Remove every HEAD copy this run wrote into the tree."""
    for path in WROTE:
        try:
            path.unlink()
        except OSError:
            pass
        for stale in (path.parent / "__pycache__").glob(path.stem + ".*"):
            try:
                stale.unlink()
            except OSError:
                pass
    WROTE.clear()


# --------------------------------------------------------------- equivalence
def equivalence(work: Path) -> None:
    """The indexes changed no answer, over the whole real corpus."""
    import live_nfl_features as nfl_new
    import live_nhl_features as nhl_new

    nhl_old = load_from_git("ml-training/nhl/live_nhl_features.py",
                            "_old_live_nhl", work)
    nfl_old = load_from_git("ml-training/nfl/live_nfl_features.py",
                            "_old_live_nfl", work)

    manifest = json.loads(
        (ROOT / "ml-training" / "models_nhl" / "manifest.json")
        .read_text(encoding="utf-8"))

    # SEPARATE STATES. The new code caches its indexes ON the state, so one
    # shared state would let the new run's cache be visible to the old one -
    # harmless here, but the kind of shortcut that makes a comparison prove
    # nothing.
    nhl_state_new = nhl_new.load_state(manifest=manifest)
    nhl_state_old = nhl_old.load_state(manifest=manifest)
    nfl_state_new = nfl_new.load_state()
    nfl_state_old = nfl_old.load_state()

    record("the NHL corpus under test is the real one, and not empty",
           bool(nhl_state_new.get("served"))
           and len(nhl_state_new["games"]) > 10_000,
           f"served={nhl_state_new.get('served')}, "
           f"{len(nhl_state_new['games'])} team-game rows, "
           f"{len(nhl_state_new['fixtures'])} fixtures")

    slate_new = nhl_new.predictable_fixtures(nhl_state_new)
    slate_old = nhl_old.predictable_fixtures(nhl_state_old)
    record("NHL predictable_fixtures: same number of rows",
           len(slate_new) == len(slate_old),
           f"{len(slate_new)} vs {len(slate_old)}")
    # ADDED IS NOT CHANGED, which is the distinction the gate's own health
    # comparison makes and the one this check needed once a field was added
    # on purpose. Every field HEAD had must still say the same thing; a
    # field only the new code has is reported separately and is allowed.
    added = sorted(set().union(*(set(r) for r in slate_new))
                   - set().union(*(set(r) for r in slate_old)))         if slate_new and slate_old else []
    differing = [(a.get("game_id"), key, a.get(key), b.get(key))
                 for a, b in zip(slate_new, slate_old)
                 for key in sorted(set(b))
                 if a.get(key) != b.get(key)]
    record("NHL predictable_fixtures: fields HEAD had are unchanged",
           not differing and len(slate_new) > 100,
           f"{len(differing)} differing field(s) over {len(slate_new)} rows"
           + (f"; first {differing[0]}" if differing else ""))
    # THE ALLOWANCE IS A DECLARED LIST, NOT A SENTENCE ABOUT "THIS PHASE".
    # This read `added == ["blocked_by"]` and went RED the moment that field
    # was committed: the comparison is against `git show HEAD`, so once HEAD
    # carries the field there is nothing new to find and `added` is empty.
    # Same way `verify_quarantine_compatibility` went un-runnable when its
    # change landed - a check written about a change expires when the change
    # does, and expires GREEN-looking in the worst cases.
    #
    # The durable rule is that this body does not grow by accident. Put a
    # name in ADDED_ON_PURPOSE only while the change adding it is
    # uncommitted, and remove it in the commit that lands the field.
    record("no field is added to the slate except a declared one",
           added == ADDED_ON_PURPOSE,
           f"added {added}, declared {ADDED_ON_PURPOSE}")

    # AND THE FLAG ITSELF, counted - so a run where every row became blocked
    # cannot pass by being uniformly wrong in both.
    predictable = sum(1 for row in slate_new if row["predictable"])
    record("NHL predictable count is non-trivial in BOTH directions",
           0 < predictable < len(slate_new),
           f"{predictable} predictable of {len(slate_new)}")

    # blocking_reason, asked about every fixture rather than the slate's rows
    mismatch = 0
    checked = 0
    for row in nhl_state_new["fixtures"].itertuples():
        home, away, date = int(row.HOME_ID), int(row.AWAY_ID), row.GAME_DATE
        a = nhl_new.blocking_reason(nhl_state_new, home, away, date)
        b = nhl_old.blocking_reason(nhl_state_old, home, away, date)
        checked += 1
        if a != b:
            mismatch += 1
    record("NHL blocking_reason: identical for every fixture, code and text",
           mismatch == 0 and checked > 1000,
           f"{checked} fixtures compared, {mismatch} differing")

    # _unplayed_before, the function that changed most
    unplayed_mismatch = 0
    unplayed_checked = 0
    nonempty = 0
    teams = sorted(nhl_new._team_dates(nhl_state_new))
    for row in nhl_state_new["fixtures"].itertuples():
        for team in (int(row.HOME_ID), int(row.AWAY_ID)):
            a = nhl_new._unplayed_before(nhl_state_new, team, row.GAME_DATE)
            b = nhl_old._unplayed_before(nhl_state_old, team, row.GAME_DATE)
            unplayed_checked += 1
            if sorted(a) != sorted(b):
                unplayed_mismatch += 1
            if a:
                nonempty += 1
    record("NHL _unplayed_before: identical on every (team, fixture) pair",
           unplayed_mismatch == 0 and unplayed_checked > 2000,
           f"{unplayed_checked} pairs, {unplayed_mismatch} differing")
    record("and it is not vacuous - most pairs return a NON-EMPTY list",
           nonempty > unplayed_checked // 2,
           f"{nonempty} of {unplayed_checked} non-empty")
    record("every NHL franchise is in the index",
           len(teams) == 32, f"{len(teams)} franchises")

    # ---- the NFL half
    nfl_new_rows = nfl_new.predictable_fixtures(nfl_state_new)
    nfl_old_rows = nfl_old.predictable_fixtures(nfl_state_old)
    record("NFL predictable_fixtures: same rows, same order",
           [str(r["game_id"]) for r in nfl_new_rows]
           == [str(r["game_id"]) for r in nfl_old_rows],
           f"{len(nfl_new_rows)} vs {len(nfl_old_rows)} fixture(s)")

    nfl_mismatch, nfl_checked, nfl_nonempty = 0, 0, 0
    import pandas as pd
    for _position, row in nfl_state_new["fixtures"].iterrows():
        key = (pd.Timestamp(row["date"]), str(row["game_id"]))
        for column in ("home_franchise_id", "away_franchise_id"):
            team = int(row[column])
            a = nfl_new.earlier_unplayed(nfl_state_new, team, key)
            b = nfl_old.earlier_unplayed(nfl_state_old, team, key)
            nfl_checked += 1
            if ([str(r["game_id"]) for r in a]
                    != [str(r["game_id"]) for r in b]):
                nfl_mismatch += 1
            if a:
                nfl_nonempty += 1
    record("NFL earlier_unplayed: identical ids IN ORDER on every pair",
           nfl_mismatch == 0 and nfl_checked > 100,
           f"{nfl_checked} pairs, {nfl_mismatch} differing")
    record("and the NFL comparison is not vacuous either",
           nfl_nonempty > 0,
           f"{nfl_nonempty} of {nfl_checked} non-empty")

    return nhl_state_new, nfl_state_new, nhl_new, nfl_new


# -------------------------------------------------------------------- budget
def blocked_by_invariants(nhl_state, nhl) -> None:
    """`blocked_by` says the same thing as `reason_code`, in data.

    THE TWO ARE COMPUTED INDEPENDENTLY AND THAT IS THE POINT.
    `predictable_fixtures` asks `blocking_reason` for the verdict and
    `blocking_details` for the per-side detail; neither reads the other, so
    nothing can make them agree except both being right. These checks are
    what notices if they stop.

    IT COSTS A SECOND WALK OF THE SAME CONDITIONS - measured 2026-11-06, the
    snapshot-derived work went from 68 ms to 174 ms, 17.4% of the budget.
    Kept rather than fused, because fusing them would make the detail the
    source of the verdict and there would be nothing left to cross-check.
    """
    slate = nhl.predictable_fixtures(nhl_state)
    record("the NHL slate is the real one and non-trivial",
           len(slate) > 500, f"{len(slate)} row(s)")

    per_side = {"season_opener", "previous_game_pending"}

    predictable = [r for r in slate if r["predictable"]]
    blocked = [r for r in slate if not r["predictable"]]
    record("every PREDICTABLE row has an empty blocked_by",
           all(r["blocked_by"] == [] for r in predictable),
           f"{len(predictable)} predictable row(s)")
    record("every row blocked for a PER-SIDE reason names at least one side",
           all(r["blocked_by"] for r in blocked
               if r["reason_code"] in per_side),
           f"{sum(1 for r in blocked if r['reason_code'] in per_side)} "
           f"per-side row(s)")
    record("and a WHOLE-FIXTURE reason names none, because it has no side",
           all(r["blocked_by"] == [] for r in blocked
               if r["reason_code"] not in per_side),
           f"{sum(1 for r in blocked if r['reason_code'] not in per_side)} "
           f"whole-fixture row(s)")

    # THE CODE ON EACH BLOCKER MATCHES THE ROW'S, because the row's code is
    # whichever side `blocking_reason` reached first - so every blocker must
    # carry a code the row could have reported.
    mismatched = [r for r in blocked
                  if r["blocked_by"]
                  and r["reason_code"] not in {b["code"]
                                               for b in r["blocked_by"]}]
    record("the row's reason_code is one of its blockers' codes",
           not mismatched,
           f"{len(mismatched)} mismatch(es)"
           + (f", first {mismatched[0]['game_id']}" if mismatched else ""))

    sides = {tuple(b["side"] for b in r["blocked_by"]) for r in blocked
             if r["blocked_by"]}
    record("sides are home, away or both - in that order, never repeated",
           sides <= {("home",), ("away",), ("home", "away")},
           f"{sorted(sides)}")

    # `pending` SEPARATES THE TWO SENTENCES. "Waiting for X's result from
    # 5 Nov" needs a date; "first game of the season" must not pretend to
    # have one.
    pending_ok = all(
        b["pending"] for r in blocked for b in r["blocked_by"]
        if b["code"] == "previous_game_pending")
    opener_ok = all(
        b["pending"] == [] for r in blocked for b in r["blocked_by"]
        if b["code"] == "season_opener")
    record("a previous_game_pending blocker carries dates", pending_ok, "")
    record("a season_opener blocker carries NONE - it waits for nothing",
           opener_ok, "")

    both = sum(1 for r in blocked if len(r["blocked_by"]) == 2)
    record("BOTH sides is the common case, which is why it is a list",
           both > len(blocked) // 2,
           f"{both} of {len(blocked)} blocked on both sides")

    # ORDERED, so a client rendering them produces a stable sentence.
    unsorted = [r for r in blocked
                for b in r["blocked_by"]
                if b["pending"] != sorted(b["pending"])]
    record("each blocker's pending dates are sorted", not unsorted,
           f"{len(unsorted)} unsorted")


def plant_blocked_by_names_one_side(nhl_state, nhl) -> None:
    """PLANT: truncate the blocker list and watch the invariants go red.

    This is the shape `blocking_reason` alone would have produced - it
    short-circuits on the first side it finds - and it is why `blocked_by`
    exists as a separate list rather than a rendering of the code.
    """
    original = nhl.blocking_details

    def truncated(state, home_id, away_id, date):
        return original(state, home_id, away_id, date)[:1]

    nhl.blocking_details = truncated
    try:
        slate = nhl.predictable_fixtures(nhl_state)
        both = sum(1 for r in slate if len(r["blocked_by"]) == 2)
    finally:
        nhl.blocking_details = original

    record("PLANT only the first side -> no row blocks on both",
           both == 0,
           "which the 'BOTH sides is the common case' check refuses")

    after = nhl.predictable_fixtures(nhl_state)
    record("PLANT guard reverting restores the two-sided rows",
           sum(1 for r in after if len(r["blocked_by"]) == 2) > 0, "")


def budget(nhl_state, nfl_state, nhl, nfl, slow: bool = False) -> None:
    """The snapshot-derived work stays inside served_data's budget."""
    def walk():
        nhl.predictable_fixtures(nhl_state)
        nhl.missing_results(nhl_state)
        nhl.standings_class(nhl_state)
        nfl.predictable_fixtures(nfl_state)

    if slow:
        # PLANT: a handler that does the same work and then sleeps. The
        # budget check must notice, or it is measuring nothing.
        original = nhl.predictable_fixtures

        def slowed(state, *args, **kwargs):
            time.sleep(S.HEALTH_BUDGET_SECONDS * 1.5)
            return original(state, *args, **kwargs)
        nhl.predictable_fixtures = slowed

    best = None
    for _ in range(3):
        started = time.perf_counter()
        walk()
        elapsed = time.perf_counter() - started
        best = elapsed if best is None else min(best, elapsed)

    record(f"the budget is DERIVED from the timeout, not written twice",
           S.HEALTH_BUDGET_SECONDS == S.HEALTH_TIMEOUT_SECONDS / 5,
           f"timeout {S.HEALTH_TIMEOUT_SECONDS}s, budget "
           f"{S.HEALTH_BUDGET_SECONDS}s")
    record("/health's snapshot-derived work is inside the budget",
           best < S.HEALTH_BUDGET_SECONDS,
           f"{best * 1000:.0f} ms against a "
           f"{S.HEALTH_BUDGET_SECONDS * 1000:.0f} ms budget")
    record("and inside the refresh's per-attempt timeout with room to spare",
           best < S.HEALTH_TIMEOUT_SECONDS / 4,
           f"{best * 1000:.0f} ms against "
           f"{S.HEALTH_TIMEOUT_SECONDS * 1000:.0f} ms")


# ------------------------------------------------------------- the probe half
class Handler(BaseHTTPRequestHandler):
    delay = 0.0
    body: dict = {}

    def do_GET(self):                                # noqa: N802
        time.sleep(self.delay)
        payload = json.dumps(self.body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):                    # silence
        return


class Quiet(HTTPServer):
    """An HTTPServer that does not print a traceback for an abandoned read.

    The slow-/health checks let `urllib` give up at the timeout, so the
    handler's `wfile.write` then fails with `ConnectionAbortedError` - which
    is the test working. Left alone, `socketserver` prints a full traceback
    for each one, and a verifier whose expected path prints tracebacks is a
    verifier whose UNEXPECTED ones are invisible.
    """

    def handle_error(self, request, client_address):
        return


def serve(delay: float, body: dict):
    Handler.delay = delay
    Handler.body = body
    server = Quiet(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}/health"


def probe_reasons(plant: str | None) -> None:
    """probe_health names each failure, and the rollback says which."""
    import daily_refresh as refresh

    if plant == "collapse-reasons":
        # PLANT: the 2026-11-06 state, reconstructed. One word for "it
        # answered too slowly" and "nothing is listening" is precisely what
        # made the refresh report a healthy service as dead.
        refresh.PROBE_TIMED_OUT = refresh.PROBE_NOT_LISTENING

    # 1. CONFIRMED
    server, url = serve(0.0, {"served_data": {"snapshot": "SNAP-1"}})
    try:
        probe = refresh.probe_health(expected="SNAP-1", attempts=2, delay=0,
                                     url=url)
        record("a fast /health with the right snapshot -> confirmed",
               probe.reason == refresh.PROBE_CONFIRMED
               and probe.snapshot == "SNAP-1",
               f"{probe!r}")
    finally:
        server.shutdown()

    # 2. TIMED OUT - the 2026-11-06 failure, reproduced UNCONDITIONALLY.
    # Not behind a plant flag: this is the situation that cost a good swap,
    # so it is checked on every run.
    server, url = serve(S.HEALTH_TIMEOUT_SECONDS + 1,
                        {"served_data": {"snapshot": "SNAP-1"}})
    try:
        probe = refresh.probe_health(expected="SNAP-1", attempts=2, delay=0,
                                     url=url)
        record("a /health slower than the timeout -> timed_out, DISTINCTLY",
               probe.reason == refresh.PROBE_TIMED_OUT,
               f"reason {probe.reason!r}, slowest {probe.slowest:.2f}s")
        record("and timed_out does NOT read as 'not serving'",
               probe.reason != refresh.PROBE_NOT_LISTENING
               and "listening" in probe.detail,
               probe.detail[:70])
    finally:
        server.shutdown()

    # 3. ANSWERED WITHOUT A SNAPSHOT
    server, url = serve(0.0, {"served_data": {}})
    try:
        probe = refresh.probe_health(expected="SNAP-1", attempts=2, delay=0,
                                     url=url)
        record("a /health with no served_data.snapshot -> no_snapshot_field",
               probe.reason == refresh.PROBE_NO_SNAPSHOT_FIELD,
               f"reason {probe.reason!r}")
    finally:
        server.shutdown()

    # 4. ANSWERED WITH A DIFFERENT SNAPSHOT
    server, url = serve(0.0, {"served_data": {"snapshot": "SNAP-OLD"}})
    try:
        probe = refresh.probe_health(expected="SNAP-NEW", attempts=2, delay=0,
                                     url=url)
        record("a /health serving a different snapshot -> "
               "different_snapshot",
               probe.reason == refresh.PROBE_DIFFERENT_SNAPSHOT
               and probe.snapshot == "SNAP-OLD",
               f"reason {probe.reason!r}, saw {probe.snapshot!r}")
        record("and it counts as LISTENING, because it answered",
               probe.listening is True, f"listening={probe.listening}")
    finally:
        server.shutdown()

    # 5. NOTHING LISTENING - the only case that means what the old message said
    probe = refresh.probe_health(expected="SNAP-1", attempts=2, delay=0,
                                 url="http://127.0.0.1:9/health")
    record("nothing listening -> not_listening, and NOT listening",
           probe.reason == refresh.PROBE_NOT_LISTENING
           and probe.listening is False,
           f"reason {probe.reason!r}")
    record("A TIMEOUT COUNTS AS LISTENING, which is the whole distinction",
           refresh.HealthProbe(None, refresh.PROBE_TIMED_OUT, "", 1, 1.0)
           .listening is True,
           "the connection was accepted; only a refusal means nothing is "
           "there")

    record("the five reasons are five DIFFERENT words",
           len({refresh.PROBE_CONFIRMED, refresh.PROBE_TIMED_OUT,
                refresh.PROBE_NO_SNAPSHOT_FIELD,
                refresh.PROBE_DIFFERENT_SNAPSHOT,
                refresh.PROBE_NOT_LISTENING}) == 5,
           "one word for two states is how this bug happened")
    record("and a timed-out probe is still LISTENING, not dead",
           refresh.HealthProbe(None, refresh.PROBE_TIMED_OUT, "", 1, 1.0)
           .listening is True,
           "collapsing this onto not_listening is plant 'collapse-reasons'")


def rollback_message(plant: str | None) -> None:
    """The rollback prints the reason, and only cries wolf when it should."""
    import daily_refresh as refresh

    work = Path(tempfile.mkdtemp(prefix="health_budget_"))
    try:
        volume = work / "vol"
        (volume / "snapshots" / "NEW").mkdir(parents=True)
        (volume / "snapshots" / "OLD").mkdir(parents=True)
        refresh.write_pointer(volume, "OLD")

        restarts = []

        def fake_restart(project):
            restarts.append(project)

        def timing_out(expected=None, **kwargs):
            return refresh.HealthProbe(
                None, refresh.PROBE_TIMED_OUT,
                f"no response within {S.HEALTH_TIMEOUT_SECONDS}s; the "
                f"service accepted the connection, so it is listening and "
                f"slow rather than absent", 20, 120.3, 5.01)

        buffer = io.StringIO()
        stdout = sys.stdout
        sys.stdout = buffer
        try:
            # (volume, PREVIOUS, IDENTIFIER) - an earlier draft of this
            # test had the two names the other way round and asserted
            # against a rollback that had restored the wrong one.
            code = refresh.swap_and_verify(
                volume, "OLD", "NEW", restart=True,
                restart_fn=fake_restart, health_fn=timing_out)
        finally:
            sys.stdout = stdout
        printed = buffer.getvalue()

        record("a timing-out /health still rolls the pointer back",
               code == refresh.EXIT_FAILED
               and refresh.read_pointer(volume) == "OLD",
               f"exit {code}, pointer {refresh.read_pointer(volume)!r}")
        record("the rollback NAMES the reason as a word",
               "[timed_out]" in printed, "")
        record("it says the service is listening, not dead",
               "THIS IS NOT A DEAD SERVICE" in printed, "")
        record("and it does NOT print 'STILL NOT SERVING' for a slow one",
               "STILL NOT SERVING" not in printed,
               "that line was false on 2026-11-06")
        record("it restarted twice - swap, then rollback",
               len(restarts) == 2, f"{len(restarts)} restart(s)")

        # CONTROL: nothing listening at all MUST still say so.
        refresh.write_pointer(volume, "OLD")

        def not_listening(expected=None, **kwargs):
            return refresh.HealthProbe(None, refresh.PROBE_NOT_LISTENING,
                                       "ConnectionRefusedError", 20, 60.0)

        buffer = io.StringIO()
        sys.stdout = buffer
        try:
            refresh.swap_and_verify(volume, "OLD", "NEW", restart=True,
                                    restart_fn=fake_restart,
                                    health_fn=not_listening)
        finally:
            sys.stdout = stdout
        record("CONTROL a dead service DOES print 'STILL NOT SERVING'",
               "STILL NOT SERVING" in buffer.getvalue(),
               "the loud line is kept for the case it describes")
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plant", choices=PLANTS)
    args = parser.parse_args()

    if args.plant:
        print(f"PLANT '{args.plant}' - checks are expected to FAIL\n")

    work = Path(tempfile.mkdtemp(prefix="health_budget_old_"))
    try:
        if args.plant == "revert-index":
            # PLANT: run the budget against the COMMITTED implementation,
            # which is the slow one. The budget check must go red, or it is
            # not measuring the thing the fix changed.
            print("  (budget measured against HEAD's implementation)\n")
            nhl = load_from_git("ml-training/nhl/live_nhl_features.py",
                                "_rev_live_nhl", work)
            nfl = load_from_git("ml-training/nfl/live_nfl_features.py",
                                "_rev_live_nfl", work)
            manifest = json.loads(
                (ROOT / "ml-training" / "models_nhl" / "manifest.json")
                .read_text(encoding="utf-8"))
            nhl_state = nhl.load_state(manifest=manifest)
            nfl_state = nfl.load_state()
        else:
            print("EQUIVALENCE - the indexes changed no answer\n")
            nhl_state, nfl_state, nhl, nfl = equivalence(work)

        if args.plant != "revert-index":
            print("\nBLOCKED_BY\n")
            blocked_by_invariants(nhl_state, nhl)
            plant_blocked_by_names_one_side(nhl_state, nhl)

        print("\nBUDGET\n")
        budget(nhl_state, nfl_state, nhl, nfl,
               slow=(args.plant == "slow-handler"))

        print("\nTHE PROBE'S REASONS\n")
        probe_reasons(args.plant)

        print("\nTHE ROLLBACK MESSAGE\n")
        rollback_message(args.plant)
    except Exception as error:                        # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"\nCRASHED before a verdict: {type(error).__name__}: {error}")
        return EXIT_CRASHED
    finally:
        import shutil
        cleanup()
        shutil.rmtree(work, ignore_errors=True)

    print(f"\n  {len(PASSED)} of {len(PASSED) + len(FAILED)} checks passed")
    if FAILED:
        for name in FAILED:
            print(f"  FAILED: {name}")
        return EXIT_FAILED
    if args.plant:
        print("\n  THE PLANT CHANGED NOTHING - that is a harness bug, "
              "not a passing check.")
        return EXIT_CRASHED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
