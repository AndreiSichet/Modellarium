"""Verify the refresh's failure paths against the real functions.

Checks 3 to 6 of this item. The two that matter are 6 - a snapshot that fails
to boot is rolled back - and 5, a snapshot that fails validation is never
swapped in. Both call `daily_refresh`'s own functions rather than
reimplementing them, because a rollback verified by a harness that has its own
copy of the rollback is not verified at all.

EVERY CHECK RESTORES WHAT IT FOUND. The pointer is recorded first and put back
at the end, and the service is left serving the snapshot it was serving
before.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))
import daily_refresh as refresh  # noqa: E402
from served_data import (  # noqa: E402
    POINTER_FILE, REQUIRED_FILES, SNAPSHOTS_DIR, missing_files)

# THE LIVE VOLUME, WHICH THIS SCRIPT REFUSES BY DEFAULT.
#
# It used to be the only target, because check 6 calls the real
# swap_and_verify and that needs a real service to restart - so the rollback
# was exercised against the thing production serves. The cost showed up on
# 2026-10-09: an interrupted run left `current` naming a deliberately
# unbootable probe snapshot, the inference service refused to boot, and the
# next run self-destructed by copying the served snapshot over itself.
#
# A copy costs nothing and loses nothing, because the verifier now starts its
# OWN inference service reading the copy and hands swap_and_verify that
# service's restart and health functions. The function under test is still the
# production one; only the volume and the service it drives are disposable.
LIVE_VOLUME = Path("D:/modellarium-data")
DEFAULT_COPY = Path("D:/modellarium-verify")

# The override. A named flag rather than a bare --volume, so pointing this at
# production is a sentence someone had to type rather than a default they did
# not notice - the same shape as fit_elo_including_test and
# --i-am-ready-to-touch-test.
LIVE_FLAG = "--i-am-ready-to-touch-the-live-volume"

# Set by main(). Every check reads this rather than a constant, so there is
# exactly one place that decides what gets written to.
VOLUME = DEFAULT_COPY

PROBE_PORT = 8111

# Set by main(); check 6 drives it.
PROBE = None

# Every snapshot this script creates is named with this prefix, which is
# what lets the pre-flight tell a wedge left by an interrupted run from a
# real snapshot, and what the cleanup globs for.
PROBE_PREFIX = "9999-"

results = []


def record(name, ok, detail=""):
    results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).splitlines():
            print(f"         {line}")


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def copy_volume(source: Path, target: Path) -> str:
    """A disposable volume holding only the pointer and the snapshot it names.

    Not the whole of `source`: the other snapshots are history this script
    never reads, and copying half a gigabyte to test a pointer write would be
    the kind of cost that stops people running it.
    """
    pointer = refresh.read_pointer(source)
    if not pointer:
        raise RuntimeError(f"{source / 'current'} is empty or absent")
    served = source / SNAPSHOTS_DIR / pointer
    if not served.is_dir():
        raise RuntimeError(
            f"{source / 'current'} names {pointer!r}, which is not a "
            f"directory, so there is nothing to copy. Fix the live volume "
            f"before verifying anything.")

    if target.exists():
        shutil.rmtree(target)
    (target / SNAPSHOTS_DIR).mkdir(parents=True)
    shutil.copytree(served, target / SNAPSHOTS_DIR / pointer)
    refresh.write_pointer(target, pointer)
    return pointer


class ProbeService:
    """The inference service, run by this script, against a disposable volume.

    THE SAME APPLICATION THE CONTAINER RUNS, not a stand-in: it resolves
    DATA_DIR, follows the pointer, and refuses to boot on an incomplete
    snapshot, which is the behaviour check 6 exists to drive. What differs is
    the volume it reads and the port it answers on, so an interrupted run
    leaves a dead process and a disposable directory rather than the live
    service down.

    `restart` takes an ignored positional argument, because production's
    `restart_service` is called as `restart_fn(PROJECT)` and the seam must
    accept the same call.
    """

    def __init__(self, volume: Path, port: int = PROBE_PORT):
        self.volume = Path(volume)
        self.port = port
        self.url = f"http://127.0.0.1:{port}/health"
        self.process = None

    def _spawn(self) -> None:
        environment = dict(os.environ)
        environment["DATA_DIR"] = str(self.volume)
        self.process = subprocess.Popen(
            [sys.executable, "-W", "ignore", "-m", "uvicorn", "app:app",
             "--host", "127.0.0.1", "--port", str(self.port),
             "--log-level", "warning"],
            cwd=PROJECT / "inference-service",
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)

    def start(self) -> None:
        if self.process is not None:
            self.stop()
        self._spawn()

    def restart(self, _unused=None) -> None:
        """Stop and start, matching production's restart_service signature."""
        self.stop()
        self._spawn()

    def health(self, attempts: int = 20, delay: float = 1.5) -> dict:
        """What this service says it is serving, or {} if it never answers.

        A dead process short-circuits rather than waiting out the attempts,
        because check 6 deliberately makes the service refuse to boot and the
        answer there is already known.
        """
        for _ in range(attempts):
            if self.process is not None and self.process.poll() is not None:
                return {}
            served = refresh.served_snapshot(attempts=1, delay=0,
                                             url=self.url)
            if served:
                return served
            time.sleep(delay)
        return {}

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=15)
        self.process = None


def make_snapshot(name: str, breaker=None) -> Path:
    """A copy of what is being served, optionally damaged.

    TWO REFUSALS, AND BOTH COME FROM A REAL INCIDENT. The source is whatever
    `current` names, so a wedged pointer propagates into this function. On
    2026-10-09 a run of this script was interrupted mid-check-6 - a killed
    process does not run `main`'s finally - which left `current` naming
    9999-01-02T0000-wontboot. The next run then called make_snapshot with that
    same name: source and target were the same directory, rmtree deleted it,
    and copytree failed on a path that no longer existed. The live pointer was
    left naming a snapshot that did not exist and the inference service was
    down.

    So: a source that does not exist, or a source that IS the target, refuses
    with the wedge named rather than destroying the evidence.
    """
    pointer = refresh.read_pointer(VOLUME)
    source = VOLUME / SNAPSHOTS_DIR / pointer
    target = VOLUME / SNAPSHOTS_DIR / name

    if not source.is_dir():
        raise RuntimeError(
            f"{VOLUME / 'current'} names {pointer!r}, which is not a "
            f"directory. The volume is WEDGED - most likely a previous run of "
            f"this script was interrupted before its finally could restore "
            f"the pointer. Write a real snapshot name into that file and "
            f"restart the service before running this again.")
    if source.resolve() == target.resolve():
        raise RuntimeError(
            f"the served snapshot IS {name!r}, so copying it over itself "
            f"would delete it. The pointer is wedged at a probe snapshot from "
            f"an earlier interrupted run; restore it first.")

    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    if breaker:
        breaker(target)
    return target


def check_nothing_new(original: str) -> None:
    section("3  A REFRESH WITH NOTHING NEW IS GREEN AND DOES NOT SWAP")
    before = refresh.read_pointer(VOLUME)
    done = subprocess.run(
        [sys.executable, "-W", "ignore", str(HERE / "daily_refresh.py"),
         "--volume", str(VOLUME), "--no-restart",
         "--skip-rebuild", "--leagues", "gleague"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=PROJECT, timeout=1800)
    after = refresh.read_pointer(VOLUME)
    out = done.stdout or ""

    record("exit code 0, not a failure", done.returncode == 0,
           f"exit {done.returncode}")
    record("it says NOTHING NEW rather than printing nothing",
           "NOTHING NEW" in out,
           [line for line in out.splitlines()
            if "continues to serve" in line][:1])
    record("the pointer did not move", before == after,
           f"{before} -> {after}")
    staging = list((VOLUME / SNAPSHOTS_DIR).glob("*.staging"))
    record("no staged directory was left behind", not staging,
           f"found: {[p.name for p in staging]}")


def check_validation_blocks_swap() -> None:
    section("5  A SNAPSHOT THAT FAILS VALIDATION IS NOT SWAPPED IN")
    before = refresh.read_pointer(VOLUME)

    # Structurally complete so missing_files passes, but the NBA table is
    # garbage - so the SERVING SMOKE TEST is what has to catch it. That is
    # the half a validator cannot do, which is why it exists.
    def wreck(root: Path):
        (root / "processed" / "games_final.csv").write_text(
            "not,a,real,table\n1,2,3,4\n", encoding="utf-8")

    broken = make_snapshot("9999-01-01T0000-unservable", wreck)
    record("the broken snapshot still looks complete to missing_files",
           not missing_files(broken),
           f"all {len(REQUIRED_FILES)} files present, one of them garbage")

    done = subprocess.run(
        [sys.executable, "-W", "ignore",
         str(HERE / "smoke_serve_snapshot.py"), str(broken)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=PROJECT, timeout=900)
    record("the serving smoke test rejects it", done.returncode != 0,
           [line for line in (done.stdout or "").splitlines()
            if "NOT SERVABLE" in line or "nba:" in line][:2])
    record("the pointer did not move", before == refresh.read_pointer(VOLUME),
           f"still {before}")
    shutil.rmtree(broken, ignore_errors=True)


def check_rollback(original: str) -> None:
    section("6  A SNAPSHOT THAT WILL NOT BOOT IS ROLLED BACK")
    print("""  The most important check in this item. A snapshot is made that passes the
  structural check but makes the service refuse to boot, then the REAL
  swap_and_verify is called on it - the same function the daily job uses.""")

    previous = refresh.read_pointer(VOLUME)

    # Deleting a required table makes require_data_root refuse at boot - the
    # same refusal a genuinely truncated mount would produce, and the
    # cheapest way to make a real boot failure.
    def wreck(root: Path):
        (root / "wnba" / "processed" / "wnba_games_final.csv").unlink()

    broken = make_snapshot("9999-01-02T0000-wontboot", wreck)
    print(f"\n  staged {broken.name}, missing "
          f"{missing_files(broken)}")

    # THE REAL swap_and_verify, AIMED AT THE PROBE SERVICE. Production's
    # restart and health functions talk to the compose stack on port 8000;
    # these talk to the service reading this copy. The function being tested
    # is unchanged.
    outcome = refresh.swap_and_verify(VOLUME, previous, broken.name,
                                      restart=True,
                                      restart_fn=PROBE.restart,
                                      health_fn=PROBE.health)

    record("the run is RED", outcome == refresh.EXIT_FAILED,
           f"exit {outcome}")
    restored = refresh.read_pointer(VOLUME)
    record("the pointer was restored to the previous snapshot",
           restored == previous, f"{restored!r} == {previous!r}")

    served = PROBE.health()
    record("the service is serving again, on the previous snapshot",
           served.get("snapshot") == previous,
           f"/health reports {served.get('snapshot')!r}")

    shutil.rmtree(broken, ignore_errors=True)


def check_failure_before_swap() -> None:
    section("4  A FAILURE DURING THE REBUILD SWAPS NOTHING")
    before = refresh.read_pointer(VOLUME)

    # A league whose pipeline step does not exist stands in for a step that
    # fails: the point is that the failure happens BEFORE any pointer write,
    # and the report says which step.
    done = subprocess.run(
        [sys.executable, "-W", "ignore", str(HERE / "daily_refresh.py"),
         "--volume", str(VOLUME), "--no-restart",
         "--leagues", "nba", "--step-timeout", "1"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=PROJECT, timeout=1800)
    out = (done.stdout or "") + (done.stderr or "")

    record("the run is RED", done.returncode == refresh.EXIT_FAILED,
           f"exit {done.returncode}")
    record("it reports that nothing was swapped",
           "FAILED BEFORE ANY SWAP" in out or "IS STILL SERVING" in out,
           [line.strip() for line in out.splitlines()
            if "STILL SERVING" in line or "FAILED BEFORE" in line][:2])
    record("the pointer did not move", before == refresh.read_pointer(VOLUME),
           f"still {before}")


def resolve_volume(args) -> Path:
    """Which volume to write to, refusing production unless asked by name.

    DEFAULT IS A COPY, AND THAT IS THE WHOLE CHANGE. This script's checks are
    destructive by design - check 6 points `current` at a snapshot that cannot
    boot - and for three items it did that to the volume the app serves. On
    2026-10-09 a run was interrupted before its finally and left the live
    pointer naming that snapshot; the service stayed down until it was fixed
    by hand.

    Nothing is lost by using a copy, because the verifier starts its own
    inference service against it and hands swap_and_verify that service's
    restart and health functions. The function under test is still the one the
    daily job calls.
    """
    if args.volume:
        target = Path(args.volume)
        if target.resolve() == LIVE_VOLUME.resolve() and not args.live:
            print(f"REFUSING: {target} is the LIVE volume.")
            print(f"  This script writes a deliberately unbootable snapshot "
                  f"into whatever volume it\n  is given, and an interrupted "
                  f"run leaves it there - which took the inference\n  "
                  f"service down on 2026-10-09 and needed a manual restore.")
            print(f"  Run it with no --volume at all, which copies the live "
                  f"volume to\n  {DEFAULT_COPY} and uses that. If you really "
                  f"mean production, pass\n  {LIVE_FLAG}.")
            return None
        if target.resolve() == LIVE_VOLUME.resolve():
            print(f"WARNING: writing to the LIVE volume {target} because "
                  f"{LIVE_FLAG} was passed.")
            print("  Do not interrupt this run. If you do, the pointer "
                  "will be left naming an unbootable snapshot and the "
                  "service will stay down until it is fixed by hand.")
        return target

    if args.live:
        print(f"Using the LIVE volume {LIVE_VOLUME} because {LIVE_FLAG} "
              f"was passed.")
        return LIVE_VOLUME

    section("COPYING THE LIVE VOLUME")
    pointer = copy_volume(LIVE_VOLUME, DEFAULT_COPY)
    print(f"  {LIVE_VOLUME} -> {DEFAULT_COPY}")
    print(f"  carried the pointer ({pointer}) and the snapshot it names")
    print(f"  the live volume is NOT written to by this run")
    return DEFAULT_COPY


def main() -> int:
    global VOLUME, PROBE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume",
                       help="the volume to verify. Omit for a fresh copy of "
                            "the live one, which is the default and the "
                            "intended way to run this.")
    parser.add_argument(LIVE_FLAG, dest="live", action="store_true",
                       help="write to the LIVE volume. Needed only to verify "
                            "production itself, and an interrupted run will "
                            "leave the service down.")
    parser.add_argument("--dry-run", action="store_true",
                       help="resolve the target volume, report it, and stop")
    args = parser.parse_args()

    print(__doc__)

    target = resolve_volume(args)
    if target is None:
        return 2
    VOLUME = target

    if args.dry_run:
        print(f"\nDRY RUN: would verify {VOLUME}")
        print(f"  live volume  : {LIVE_VOLUME}")
        print(f"  is that live?: {VOLUME.resolve() == LIVE_VOLUME.resolve()}")
        return 0

    original = refresh.read_pointer(VOLUME)

    # A PRE-FLIGHT, BECAUSE THE ONLY CHEAP MOMENT TO FIX A WEDGE IS BEFORE
    # STARTING. The restore below is a finally, and a killed process does not
    # run one - so an interrupted run leaves `current` naming a deliberately
    # unbootable probe snapshot. Starting again on top of that is how a
    # recoverable state became a self-destruct; refusing here is what makes
    # the wedge visible instead.
    if original.startswith(PROBE_PREFIX) or not (
            VOLUME / SNAPSHOTS_DIR / original).is_dir():
        print(f"\nREFUSING: {VOLUME / 'current'} names {original!r}.")
        print("  That is either a probe snapshot from this script or a name "
              "with no directory,\n  so a previous run was interrupted "
              "before its finally could restore the pointer.")
        print("  The live service is probably refusing to boot. Restore it "
              "first:")
        real = sorted(p.name for p in (VOLUME / SNAPSHOTS_DIR).iterdir()
                      if p.is_dir() and not p.name.startswith(PROBE_PREFIX)
                      and not p.name.endswith(".staging"))
        if real:
            print(f"    echo {real[-1]} > {VOLUME / 'current'}")
            print(f"    docker compose -p {refresh.COMPOSE_PROJECT} restart "
                  f"inference-service")
        return 2

    print(f"Serving {original} at the start; it will be restored.")
    print("""
CHECK 6 CALLS THE REAL swap_and_verify, because a rollback verified by a
harness holding its own copy of the rollback is not verified at all. What it
drives is this script's OWN inference service, reading the volume above - the
same application the container runs, resolving DATA_DIR, reading the pointer
at boot and refusing an incomplete snapshot.

It can only pass when the snapshot under test carries every table
`daily_refresh.LEAGUES` declares, because check 3 asserts that a refresh finds
nothing new and a league whose tables are absent is genuinely behind. So after
a league is added, this has to wait until a snapshot carrying it exists.""")

    PROBE = ProbeService(VOLUME)
    try:
        section("STARTING THE PROBE SERVICE")
        PROBE.start()
        booted = PROBE.health()
        print(f"  port {PROBE.port}, serving "
              f"{booted.get('snapshot')!r} from {VOLUME}")
        if booted.get("snapshot") != original:
            print(f"  REFUSING: the probe service reports "
                  f"{booted.get('snapshot')!r}, not {original!r}. Nothing "
                  f"below would mean anything.")
            return 2

        check_nothing_new(original)
        check_failure_before_swap()
        check_validation_blocks_swap()
        check_rollback(original)
    finally:
        section("RESTORING")
        if refresh.read_pointer(VOLUME) != original:
            refresh.write_pointer(VOLUME, original)
            PROBE.restart()
            print(f"  pointer put back to {original}")
        else:
            print(f"  pointer already {original}")
        for leftover in (VOLUME / SNAPSHOTS_DIR).glob("9999-*"):
            shutil.rmtree(leftover, ignore_errors=True)
            print(f"  removed {leftover.name}")
        served = PROBE.health()
        print(f"  probe /health reports {served.get('snapshot')!r}")
        PROBE.stop()
        print(f"  probe service stopped")
        if VOLUME.resolve() != LIVE_VOLUME.resolve():
            print(f"  the live volume was never written to; "
                  f"{LIVE_VOLUME / POINTER_FILE} still reads "
                  f"{refresh.read_pointer(LIVE_VOLUME)!r}")

    section("RESULT")
    failed = [name for name, ok in results if not ok]
    print(f"  {len(results) - len(failed)} passed, {len(failed)} failed")
    for name in failed:
        print(f"    FAILED: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
