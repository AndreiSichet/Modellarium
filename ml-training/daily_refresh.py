"""Build a new served-data snapshot, validate it, and swap it in.

Three outcomes, reported distinctly, and the exit code carries them - the
lesson the weekly retrain learned the hard way, where a crash and a gate
refusal shared exit 1 and the wrapper reported a crash as "the guard worked":

    0  ADVANCED      new games found, snapshot swapped, data_as_of moved
    0  NOTHING NEW   no games since the last snapshot. Off-day, All-Star
                     break, or a league's offseason. NOT a failure.
    1  FAILED        a pipeline step, validation or the swap failed. The
                     PREVIOUS snapshot keeps serving.

A job that reports the same thing whether it did something or nothing is a
job nobody can read, so "advanced" and "nothing new" are both green and say
which.

THE SWAP IS A POINTER WRITE, AND THAT IS WHY IT IS SAFE ON WINDOWS. File
operations crossing the Docker Desktop bind-mount boundary are not guaranteed
atomic, so nothing here depends on atomically replacing 80 MB of tables: the
snapshot is built under its own name, validated in place, and only then does
`current` - a few bytes naming it - get rewritten. The service reads the
pointer once, at boot, after that write.

ROLLBACK IS THE SAME WRITE IN REVERSE. If the service fails to come up on the
new snapshot, `current` goes back to the previous one and the service is
restarted again. That is the one path that decides whether a bad day can take
the app down, so it is tested rather than argued.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))
from served_data import (  # noqa: E402
    POINTER_FILE, REPO_DATA_ROOT, REQUIRED_FILES, SNAPSHOTS_DIR,
    missing_files, resolve_root, write_snapshot_metadata)

DEFAULT_VOLUME = Path("D:/modellarium-data")
KEEP_SNAPSHOTS = 7

EXIT_OK = 0
EXIT_FAILED = 1

# Per league: the pipeline steps that rebuild its history, in order, and the
# tables the snapshot takes from the result.
#
# ONLY THE TABLES THE SERVICE READS ARE COPIED. The pipeline writes a dozen
# intermediate files; a snapshot carrying them would invite something to
# start reading one, and then the snapshot's contract would be "whatever the
# pipeline happened to write".
LEAGUES = {
    "nba": {
        "steps": [
            PROJECT / "data-pipeline" / "ingestion" / "fetch_games.py",
            PROJECT / "data-pipeline" / "preprocessing" / "validate_games.py",
            PROJECT / "data-pipeline" / "preprocessing" / "build_games_table.py",
            PROJECT / "data-pipeline" / "preprocessing" / "build_rolling_features.py",
            PROJECT / "data-pipeline" / "preprocessing" / "build_rest_days.py",
            PROJECT / "data-pipeline" / "preprocessing" / "build_elo_ratings.py",
            PROJECT / "data-pipeline" / "ingestion" / "fetch_player_boxscores.py",
            PROJECT / "data-pipeline" / "preprocessing" / "build_player_rolling_minutes.py",
            PROJECT / "data-pipeline" / "ingestion" / "fetch_quarter_scores.py",
            PROJECT / "data-pipeline" / "preprocessing" / "build_quarter_half_raw.py",
        ],
        "tables": [
            Path("processed") / "games_final.csv",
            Path("processed") / "player_boxscores_with_rolling.csv",
            Path("processed") / "quarter_half_raw.csv",
        ],
        "cutoff_table": Path("processed") / "games_final.csv",
    },
    "wnba": {
        "steps": [
            PROJECT / "data-pipeline" / "wnba" / "ingestion" / "fetch_wnba_games.py",
            PROJECT / "data-pipeline" / "wnba" / "preprocessing" / "validate_wnba_games.py",
            PROJECT / "data-pipeline" / "wnba" / "preprocessing" / "build_wnba_games_table.py",
        ],
        "tables": [Path("wnba") / "processed" / "wnba_games_final.csv"],
        "cutoff_table": Path("wnba") / "processed" / "wnba_games_final.csv",
    },
    "gleague": {
        "steps": [
            PROJECT / "data-pipeline" / "gleague" / "ingestion" / "fetch_gleague_games.py",
            PROJECT / "data-pipeline" / "gleague" / "preprocessing" / "validate_gleague_games.py",
            PROJECT / "data-pipeline" / "gleague" / "preprocessing" / "build_gleague_games_table.py",
        ],
        "tables": [
            Path("gleague") / "processed" / "gleague_games_final.csv",
            Path("gleague") / "processed" / "gleague_showcase_games.csv",
        ],
        "cutoff_table": Path("gleague") / "processed" / "gleague_games_final.csv",
    },
}


LOCK_FILE = "refresh.lock"

# A refresh that overran this long is assumed dead rather than holding the
# lock forever. Matches the workflow's own timeout-minutes, so a run killed by
# Actions cannot block the next morning's.
LOCK_STALE_AFTER_SECONDS = 4 * 60 * 60


def acquire_lock(volume: Path) -> Path:
    """Refuse to run while another refresh holds the volume.

    FOUND BY A TEST, NOT BY REVIEW. Two refreshes ran concurrently - one of
    them a process a kill had failed to reach - and they raced on the
    pointer: the first swapped to its snapshot, the second swapped to its own
    0.75 seconds later, and the first then saw /health reporting a snapshot
    it had not written and rolled back. The rollback behaved correctly in a
    situation nobody had designed for, which is the good news; the bad news
    is that two runs could interleave at all.

    The workflow also sets a `concurrency` group, which covers the scheduled
    and dispatched runs. This covers a run started by hand, which is how it
    happened.
    """
    lock = volume / LOCK_FILE
    if lock.is_file():
        age = time.time() - lock.stat().st_mtime
        held = lock.read_text(encoding="utf-8").strip()
        if age < LOCK_STALE_AFTER_SECONDS:
            raise SystemExit(
                f"another refresh holds {lock}:\n  {held}\n"
                f"  held for {age / 60:.0f} minute(s).\n"
                f"Two refreshes racing on the pointer is how one ends up "
                f"rolling back a snapshot the other wrote. Wait, or delete "
                f"the lock if that process is gone.")
        print(f"  taking over a lock held for {age / 3600:.1f}h by:\n"
              f"    {held}")

    lock.write_text(
        f"pid {os.getpid()} on {platform.node()} "
        f"since {datetime.now(timezone.utc).isoformat()}",
        encoding="utf-8")
    return lock


def release_lock(lock: Path) -> None:
    lock.unlink(missing_ok=True)


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def newest_date(root: Path, relative: Path) -> str:
    import pandas as pd

    frame = pd.read_csv(root / relative, usecols=["GAME_DATE"])
    return str(pd.to_datetime(frame["GAME_DATE"]).max().date())


def cutoffs(root: Path) -> dict:
    return {league: newest_date(root, spec["cutoff_table"])
            for league, spec in LEAGUES.items()}


def run_step(script: Path, timeout: int) -> None:
    print(f"    {script.name} ...", end=" ", flush=True)
    started = time.monotonic()
    done = subprocess.run([sys.executable, "-W", "ignore", str(script)],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, cwd=PROJECT)
    elapsed = time.monotonic() - started
    if done.returncode != 0:
        print(f"FAILED ({elapsed:.0f}s)")
        tail = (done.stdout or "")[-1500:] + (done.stderr or "")[-1500:]
        raise RuntimeError(
            f"{script.name} exited {done.returncode}\n{tail}")
    print(f"ok ({elapsed:.0f}s)")


def rebuild(leagues: list, timeout: int) -> None:
    section("REBUILDING, PER LEAGUE")
    for league in leagues:
        print(f"  {league}")
        for script in LEAGUES[league]["steps"]:
            if not script.is_file():
                raise RuntimeError(f"{script} does not exist")
            run_step(script, timeout)


def stage(volume: Path, identifier: str) -> Path:
    """Copy the service's tables out of the repo into a staged snapshot."""
    section("STAGING")
    staging = volume / SNAPSHOTS_DIR / f"{identifier}.staging"
    if staging.exists():
        shutil.rmtree(staging)

    for relative in REQUIRED_FILES:
        source = REPO_DATA_ROOT / relative
        if not source.is_file():
            raise RuntimeError(f"the rebuild did not produce {relative}")
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"  {str(relative):<56}{destination.stat().st_size:>12,}")
    return staging


def validate(staging: Path) -> dict:
    """The validators, then a serving smoke test per league.

    THE SMOKE TEST IS THE HALF THAT MATTERS. A validator checks the table's
    own shape; it cannot tell whether live features COMPUTE from it. A
    snapshot that validates and then cannot be served is exactly the failure
    the pointer design exists to avoid swapping in.
    """
    section("VALIDATING THE STAGED SNAPSHOT")

    missing = missing_files(staging)
    if missing:
        raise RuntimeError(f"staged snapshot is missing {missing}")
    print(f"  all {len(REQUIRED_FILES)} required tables present")

    found = cutoffs(staging)
    for league, date in found.items():
        print(f"  {league:<9}data_as_of {date}")

    section("SERVING SMOKE TEST AGAINST THE STAGED SNAPSHOT")
    probe = HERE / "smoke_serve_snapshot.py"
    done = subprocess.run(
        [sys.executable, "-W", "ignore", str(probe), str(staging)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=PROJECT, timeout=900)
    print((done.stdout or "").rstrip())
    if done.returncode != 0:
        raise RuntimeError(
            "the staged snapshot cannot be served:\n"
            + (done.stderr or "")[-1200:])
    return found


def read_pointer(volume: Path) -> str:
    pointer = volume / POINTER_FILE
    return pointer.read_text(encoding="utf-8").strip() \
        if pointer.is_file() else ""


def write_pointer(volume: Path, identifier: str) -> None:
    (volume / POINTER_FILE).write_text(identifier, encoding="utf-8")


# The compose project is named after the DIRECTORY it was brought up from,
# and this job does not run from that directory. The stack lives in the
# Desktop checkout (project `basketball-predictor`); the refresh runs from the
# runner's working copy, which is called `Modellarium` - so a bare
# `docker compose restart` there would target a project that does not exist
# and silently restart nothing. Addressed by name instead.
COMPOSE_PROJECT = "basketball-predictor"


def restart_service(_unused: Path = None, project: str = COMPOSE_PROJECT,
                    timeout: int = 240) -> None:
    subprocess.run(
        ["docker", "compose", "-p", project, "restart", "inference-service"],
        capture_output=True, text=True, timeout=timeout)


def served_snapshot(attempts: int = 20, delay: float = 3.0) -> dict:
    """What /health says it is serving, once it answers."""
    import urllib.error
    import urllib.request

    for _ in range(attempts):
        try:
            with urllib.request.urlopen(
                    "http://localhost:8000/health", timeout=5) as response:
                return json.loads(response.read()).get("served_data", {})
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            time.sleep(delay)
    return {}


def prune(volume: Path, keep: int) -> None:
    snapshots = sorted(
        (p for p in (volume / SNAPSHOTS_DIR).glob("*")
         if p.is_dir() and not p.name.endswith(".staging")),
        key=lambda p: p.name)
    doomed = snapshots[:-keep] if len(snapshots) > keep else []
    current = read_pointer(volume)
    for path in doomed:
        if path.name == current:
            continue  # never delete what is being served
        shutil.rmtree(path, ignore_errors=True)
        print(f"  pruned {path.name}")
    print(f"  {len(snapshots) - len(doomed)} snapshot(s) kept "
          f"(limit {keep})")


def swap_and_verify(volume: Path, previous: str, identifier: str,
                    restart: bool = True) -> int:
    """Point `current` at `identifier`, restart, and roll back if it fails.

    EXTRACTED SO THE ROLLBACK IS TESTED BY THE CODE PRODUCTION RUNS. This is
    the one path that decides whether a bad day can take the app down, and a
    rollback verified by a harness that reimplements it is not verified at
    all - the same reason the retrain harness invokes continuous_retrain.py
    as a subprocess rather than calling into a copy of its logic.
    """
    section("SWAP")
    print(f"  {previous} -> {identifier}")
    write_pointer(volume, identifier)
    confirmed = read_pointer(volume)
    print(f"  pointer now reads {confirmed!r}")

    # The pointer write IS the swap, so it is read back rather than assumed.
    # Windows does not guarantee atomicity for file operations crossing the
    # Docker Desktop bind-mount boundary; nothing here needs atomicity, but a
    # write that did not land must not be mistaken for one that did.
    if confirmed != identifier:
        write_pointer(volume, previous)
        print(f"  POINTER WRITE DID NOT TAKE. Restored {previous}.")
        return EXIT_FAILED

    if not restart:
        print("  restart skipped; the service was left alone")
        return EXIT_OK

    print("  restarting inference-service ...")
    restart_service(PROJECT)
    serving = served_snapshot()

    if serving.get("snapshot") == identifier:
        print(f"  /health confirms snapshot {serving.get('snapshot')}")
        return EXIT_OK

    section("THE SERVICE DID NOT COME UP - ROLLING BACK")
    print(f"  /health reported {serving.get('snapshot')!r}, "
          f"expected {identifier!r}")
    write_pointer(volume, previous)
    print(f"  pointer restored to {previous}")
    restart_service(PROJECT)
    recovered = served_snapshot()
    print(f"  after restart /health reports {recovered.get('snapshot')!r}")
    if recovered.get("snapshot") == previous:
        print("  the previous snapshot is serving again")
    else:
        print("  THE SERVICE IS STILL NOT SERVING. Manual attention needed.")
    return EXIT_FAILED


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", default=str(DEFAULT_VOLUME))
    parser.add_argument("--leagues", default="nba,wnba,gleague")
    parser.add_argument("--step-timeout", type=int, default=5400)
    parser.add_argument("--skip-rebuild", action="store_true",
                        help="stage and swap whatever the repo already "
                             "holds; for exercising the swap itself")
    parser.add_argument("--outcome-file",
                        help="write 'advanced', 'nothing-new' or 'failed' "
                             "here. The exit code cannot carry the "
                             "distinction - advanced and nothing-new are "
                             "both green, which is the point - and a "
                             "workflow summary that greps stdout breaks "
                             "the first time the wording changes.")
    parser.add_argument("--compose-project",
                        default=COMPOSE_PROJECT,
                        help="the compose project holding the "
                             "running stack. Named rather than "
                             "inferred from a directory, "
                             "because this job does not run "
                             "from the stack's checkout.")
    parser.add_argument("--no-restart", action="store_true",
                        help="swap the pointer but leave the service alone")
    args = parser.parse_args()

    volume = Path(args.volume)
    leagues = [name.strip() for name in args.leagues.split(",")
               if name.strip()]
    unknown = [name for name in leagues if name not in LEAGUES]
    if unknown:
        raise SystemExit(f"unknown league(s) {unknown}")

    def record_outcome(name: str) -> None:
        """Which of the three outcomes this was.

        The exit code cannot carry it: `advanced` and `nothing-new` are both
        green, which is the whole point of distinguishing them. A workflow
        summary that greps stdout instead would break the first time the
        wording changed, so the job reads this file.
        """
        if args.outcome_file:
            Path(args.outcome_file).write_text(name, encoding="utf-8")
        print(f"\n  outcome: {name}")

    print(__doc__)

    lock = acquire_lock(volume)
    try:
        return _run(args, volume, leagues, record_outcome)
    finally:
        release_lock(lock)


def _run(args, volume, leagues, record_outcome) -> int:
    section("BEFORE")
    previous = read_pointer(volume)
    if not previous:
        raise SystemExit(
            f"{volume / POINTER_FILE} does not exist. Seed the volume first "
            f"with seed_served_volume.py - this job swaps snapshots and does "
            f"not create the first one.")
    previous_root = resolve_root(volume)
    before = cutoffs(previous_root)
    print(f"  serving  {previous}")
    for league, date in before.items():
        print(f"  {league:<9}data_as_of {date}")

    identifier = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M")

    try:
        if args.skip_rebuild:
            section("REBUILD SKIPPED")
            print("  staging whatever this checkout already holds")
        else:
            rebuild(leagues, args.step_timeout)

        staging = stage(volume, identifier)
        after = validate(staging)
    except Exception as error:
        section("FAILED BEFORE ANY SWAP")
        print(f"  {type(error).__name__}: {error}")
        print(f"\n  {previous} IS STILL SERVING. Nothing was swapped.")
        record_outcome("failed")
        return EXIT_FAILED

    advanced = {league: (before[league], after[league])
                for league in after if after[league] != before[league]}

    section("WHAT CHANGED")
    for league in after:
        mark = "ADVANCED" if league in advanced else "unchanged"
        print(f"  {league:<9}{before[league]} -> {after[league]}   {mark}")

    if not advanced:
        section("NOTHING NEW")
        print(f"""  No league has a game later than the snapshot already being served, so
  there is nothing to swap in. An off-day, an All-Star break, or simply
  every league between seasons - which is the state today.

  This is GREEN and not a failure. The staged snapshot is discarded rather
  than kept, because an identical snapshot under a new name would make the
  retention window shorter for no gain.""")
        shutil.rmtree(staging, ignore_errors=True)
        print(f"\n  {previous} continues to serve.")
        record_outcome("nothing-new")
        return EXIT_OK

    final = volume / SNAPSHOTS_DIR / identifier
    write_snapshot_metadata(staging, identifier,
                            source="daily_refresh", per_league=after)
    staging.rename(final)

    swapped = swap_and_verify(volume, previous, identifier,
                              restart=not args.no_restart)
    if swapped == EXIT_OK and not args.no_restart:
        section("ADVANCED")
        for league, (was, now) in advanced.items():
            print(f"  {league:<9}{was} -> {now}")
        prune(volume, KEEP_SNAPSHOTS)
    record_outcome("advanced" if swapped == EXIT_OK else "failed")
    return swapped


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001
        print(f"\nUNHANDLED: {type(error).__name__}: {error}")
        sys.exit(EXIT_FAILED)
