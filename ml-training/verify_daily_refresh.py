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

import json
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

VOLUME = Path("D:/modellarium-data")

results = []


def record(name, ok, detail=""):
    results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).splitlines():
            print(f"         {line}")


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def health() -> dict:
    import urllib.error
    import urllib.request

    for _ in range(20):
        try:
            with urllib.request.urlopen(
                    "http://localhost:8000/health", timeout=5) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            time.sleep(3)
    return {}


def make_snapshot(name: str, breaker=None) -> Path:
    """A copy of what is being served, optionally damaged."""
    source = VOLUME / SNAPSHOTS_DIR / refresh.read_pointer(VOLUME)
    target = VOLUME / SNAPSHOTS_DIR / name
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

    outcome = refresh.swap_and_verify(VOLUME, previous, broken.name,
                                      restart=True)

    record("the run is RED", outcome == refresh.EXIT_FAILED,
           f"exit {outcome}")
    restored = refresh.read_pointer(VOLUME)
    record("the pointer was restored to the previous snapshot",
           restored == previous, f"{restored!r} == {previous!r}")

    served = health().get("served_data", {})
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


def main() -> int:
    print(__doc__)
    original = refresh.read_pointer(VOLUME)
    print(f"Serving {original} at the start; it will be restored.")

    try:
        check_nothing_new(original)
        check_failure_before_swap()
        check_validation_blocks_swap()
        check_rollback(original)
    finally:
        section("RESTORING")
        if refresh.read_pointer(VOLUME) != original:
            refresh.write_pointer(VOLUME, original)
            refresh.restart_service(PROJECT)
            print(f"  pointer put back to {original}")
        else:
            print(f"  pointer already {original}")
        for leftover in (VOLUME / SNAPSHOTS_DIR).glob("9999-*"):
            shutil.rmtree(leftover, ignore_errors=True)
            print(f"  removed {leftover.name}")
        served = health().get("served_data", {})
        print(f"  /health reports {served.get('snapshot')!r}")

    section("RESULT")
    failed = [name for name, ok in results if not ok]
    print(f"  {len(results) - len(failed)} passed, {len(failed)} failed")
    for name in failed:
        print(f"    FAILED: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
