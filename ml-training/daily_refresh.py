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
import hashlib
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
import pandas as pd  # noqa: E402
import drift_report  # noqa: E402
from served_data import (  # noqa: E402
    OPTIONAL_FILES, POINTER_FILE, REFRESH_LATE_HOURS, REPO_DATA_ROOT,
    REQUIRED_FILES, SNAPSHOTS_DIR, missing_files, read_last_refresh,
    resolve_root, write_last_refresh, write_snapshot_metadata)

DEFAULT_VOLUME = Path("D:/modellarium-data")
KEEP_SNAPSHOTS = 7

NEWLINE = chr(10)

EXIT_OK = 0
EXIT_FAILED = 1
# SWAPPED, BUT AT LEAST ONE LEAGUE WAS CARRIED FORWARD. The healthy leagues
# have advanced and are being served; the failed one is still serving
# yesterday's tables. The run ends RED so the failure is seen - the workflow
# reads the outcome FILE rather than this code, and `partial` is red there -
# but the distinction matters when reading a log months later.
#
# 3 RATHER THAN 2. Every verifier in this project uses 2 for "crashed before
# reaching a verdict", and a partial refresh very much reached one.
EXIT_PARTIAL = 3

# THE ONE LINE OF A SUCCESSFUL STEP THAT SURVIVES. run_step
# captures stdout and throws it away, which is right for 19 steps
# of progress chatter and wrong for the fetchers' request
# counters: a run that was throttled looked identical to one that
# was not, and the 429 count for the first NFL fetch is gone for
# good. A fetcher prints its summary behind this prefix and
# nothing else is kept. fetch_nfl_seasons.py prints it; the five
# nba_api fetchers have no counters to print.
FETCH_STATS_PREFIX = "FETCH-STATS:"

# Per league: the pipeline steps that rebuild its history, in order, and the
# tables the snapshot takes from the result.
#
# ONLY THE TABLES THE SERVICE READS ARE COPIED. The pipeline writes a dozen
# intermediate files; a snapshot carrying them would invite something to
# start reading one, and then the snapshot's contract would be "whatever the
# pipeline happened to write".
LEAGUES = {
    "nba": {
        # THE ONLY LEAGUE WITH AVAILABILITY FEATURES, declared rather than
        # assumed. availability_lines() reads this instead of naming the
        # leagues that lack them, which is how the NFL ended up with no line
        # at all: the other three were spelled out and a fourth was simply
        # not in the list.
        "availability": True,
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
    # THE NFL LEARNS THIS IN THE SAME COMMIT THAT MAKES SERVING REQUIRE IT,
    # and the ordering is not a preference. The moment the inference service
    # requires the three NFL tables, a refresh that does not build them
    # produces a snapshot the service refuses - so validation fails, nothing
    # swaps, and the job goes red every morning until this entry exists.
    # Shipping serving first and the refresh later would have been a week of
    # red mornings for a reason nobody would have had to live with.
    #
    # THREE STEPS, AND THE FETCHER IS ALREADY THE SHAPE THIS NEEDS. Phase 1
    # built it with revision-id change detection first - one cheap request per
    # season tells it which articles moved - and completed seasons frozen by
    # revision id. So a quiet day costs a handful of cheap requests and no
    # content fetch at all, which is the rule the first daily-refresh failure
    # taught the other three leagues retroactively and this one had from the
    # start.
    "nfl": {
        "steps": [
            PROJECT / "data-pipeline" / "nfl" / "ingestion" / "fetch_nfl_seasons.py",
            PROJECT / "data-pipeline" / "nfl" / "preprocessing" / "validate_nfl_games.py",
            PROJECT / "data-pipeline" / "nfl" / "preprocessing" / "build_nfl_tables.py",
        ],
        "tables": [
            Path("nfl") / "processed" / "nfl_games_final.csv",
            Path("nfl") / "processed" / "nfl_fixtures.csv",
            Path("nfl") / "processed" / "nfl_franchise_identity.csv",
        ],
        "cutoff_table": Path("nfl") / "processed" / "nfl_games_final.csv",
        # The NFL's table names its date column `date`, lower case, where the
        # three nba_api leagues all write GAME_DATE. Declared rather than
        # guessed, because a missing column here would surface as a KeyError
        # inside the cutoff report rather than as "the NFL is out of date".
        "date_column": "date",
        # WRITTEN EVERY RUN AND ABSENT FROM EVERY SNAPSHOT BUILT BEFORE IT.
        # Declared as optional so that neither direction of the upgrade
        # breaks: a snapshot without it still validates and can still be
        # carried, and a snapshot with it still boots an image that has never
        # heard of it. See served_data.OPTIONAL_FILES for why that matters on
        # the day this ships - the snapshot a rollback would land on is one of
        # the old ones.
        "optional_tables": [
            Path("nfl") / "processed" / "nfl_quarantine.csv",
        ],
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


def newest_date(root: Path, relative: Path, column: str = "GAME_DATE"):
    """The newest date in a cutoff table, or None if the table is not there.

    NONE RATHER THAN A RAISE, AND THE FIRST NFL RUN IS WHY. This is called on
    the PREVIOUS snapshot as well as the staged one, and the first run after a
    league is added necessarily reads a previous snapshot that has none of its
    tables - so raising here aborts the very run that would add them. It did:
    the phase 4 rehearsal failed with FileNotFoundError on
    nfl/processed/nfl_games_final.csv before staging anything, which would have
    been the real switch too.

    A missing table is only legitimate on the PREVIOUS side. The staged
    snapshot is checked by missing_files() and then by the smoke test, both of
    which refuse an absent table outright, so nothing can reach the pointer on
    the strength of this None.
    """
    import pandas as pd

    path = root / relative
    if not path.is_file():
        return None
    frame = pd.read_csv(path, usecols=[column])
    return str(pd.to_datetime(frame[column]).max().date())


def cutoffs(root: Path) -> dict:
    return {league: newest_date(root, spec["cutoff_table"],
                                spec.get("date_column", "GAME_DATE"))
            for league, spec in LEAGUES.items()}


def run_step(script: Path, timeout: int, env: dict = None) -> list:
    print(f"    {script.name} ...", end=" ", flush=True)
    started = time.monotonic()
    done = subprocess.run([sys.executable, "-W", "ignore", str(script)],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, cwd=PROJECT,
                          env=env)
    elapsed = time.monotonic() - started
    if done.returncode != 0:
        print(f"FAILED ({elapsed:.0f}s)")
        tail = (done.stdout or "")[-1500:] + (done.stderr or "")[-1500:]
        raise RuntimeError(
            f"{script.name} exited {done.returncode}\n{tail}")
    print(f"ok ({elapsed:.0f}s)")

    # Only lines behind the marker, and only from a fetch step. Everything
    # else a successful step wrote stays discarded.
    surfaced = []
    if script.name.startswith("fetch_"):
        for line in (done.stdout or "").splitlines():
            stripped = line.strip()
            if stripped.startswith(FETCH_STATS_PREFIX):
                surfaced.append(
                    stripped[len(FETCH_STATS_PREFIX):].strip())
        for line in surfaced:
            print(f"      {script.name}: {line}")
    return surfaced


def rebuild(leagues: list, timeout: int, full_refetch: bool = False) -> list:
    section("REBUILDING, PER LEAGUE")

    env = None
    if full_refetch:
        # The fetchers read this rather than taking a flag, because the steps
        # are run as subprocesses by path and there is no argument list to
        # thread an option through.
        env = dict(os.environ, MODELLARIUM_FULL_REFETCH="1")
        print("  MODELLARIUM_FULL_REFETCH=1 - every season will be "
              "re-fetched, including completed ones")

    collected, fetch_steps = [], 0
    for league in leagues:
        print(f"  {league}")
        for script in LEAGUES[league]["steps"]:
            if not script.is_file():
                raise RuntimeError(f"{script} does not exist")
            lines = run_step(script, timeout, env=env)
            if script.name.startswith("fetch_"):
                fetch_steps += 1
            collected.extend(f"{script.name}: {line}" for line in lines)

    # A COUNT, SO A VANISHING LINE IS VISIBLE. Printing a note for each
    # fetcher without counters would be five lines of noise every morning;
    # printing nothing would let the one line that does exist disappear
    # silently the day its marker breaks, which is the failure mode this
    # whole change exists to remove.
    print()
    print(f"  {len(collected)} of {fetch_steps} fetch step(s) reported "
          f"request counters")
    return collected


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


HEALTH_URL = "http://localhost:8000/health"


def served_snapshot(attempts: int = 20, delay: float = 3.0,
                    url: str = HEALTH_URL) -> dict:
    """What /health says it is serving, once it answers."""
    import urllib.error
    import urllib.request

    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return json.loads(response.read()).get("served_data", {})
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            time.sleep(delay)
    return {}


# A staging directory this old cannot belong to a live run: the longest real
# refresh measured is 18.0 minutes and the lock itself goes stale at 4 hours,
# so 6 hours is past anything that could still be working. Deliberately not
# "older than a day", which CLAUDE.md suggested - a cancelled 07:00 run would
# then survive until the next morning's, which is the one run whose log nobody
# reads twice.
STAGING_STALE_AFTER_HOURS = 6


def directory_size(path: Path) -> int:
    """Bytes under `path`, counting files only, errors ignored."""
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def sweep_staging(volume: Path, identifier: str,
                  max_age_hours: float = STAGING_STALE_AFTER_HOURS) -> list:
    """Remove orphaned `*.staging` directories. Called INSIDE the lock.

    NOTHING HAS EVER REMOVED THESE. `prune` skips them by name, and
    `stage_per_league` removes only the staging directory of the run it is in
    - whose identifier is minute-granular and never recurs. So a run cancelled
    between two leagues leaves its partial snapshot on the volume for good:
    `2026-10-24T1331.staging`, 84.9 MB, the remains of a server-side
    cancellation, sat there until it was deleted by hand on 2026-10-25.

    THREE THINGS IT MUST NOT TOUCH, and each is checked rather than assumed:

    * a completed snapshot. Only names ending `.staging` are considered, so a
      real snapshot is out of scope by construction - including one that
      happens to be NAMED like a staging directory, because the suffix test
      is on the name and a completed snapshot never carries it.
    * the directory `current` points at. Checked explicitly even though the
      pointer can only name a completed snapshot, because "can only" is the
      kind of claim that stops being true quietly, and the cost of the check
      is one string comparison.
    * THIS run's staging directory. At the moment this is called it does not
      exist yet, so the age filter alone would do - but relying on that would
      make the sweep unsafe the first time anything is reordered, and the
      identifier is already in hand.

    Returns what it removed, so the caller can report it and a test can
    assert on it rather than on stdout.
    """
    root = volume / SNAPSHOTS_DIR
    if not root.is_dir():
        return []
    current = read_pointer(volume)
    cutoff = time.time() - max_age_hours * 3600.0
    removed = []

    for path in sorted(root.glob("*.staging")):
        if not path.is_dir():
            continue
        if path.name == f"{identifier}.staging":
            continue
        if path.name == current or path.stem == current:
            print(f"  kept {path.name}: the pointer names it", flush=True)
            continue
        try:
            age_seconds = time.time() - path.stat().st_mtime
        except OSError as error:
            print(f"  kept {path.name}: cannot read its age "
                  f"({type(error).__name__})", flush=True)
            continue
        if path.stat().st_mtime > cutoff:
            print(f"  kept {path.name}: only {age_seconds / 3600:.1f}h old, "
                  f"under {max_age_hours}h - a run may still be using it",
                  flush=True)
            continue

        size = directory_size(path)
        shutil.rmtree(path, ignore_errors=True)
        if path.exists():
            print(f"  WARNING: could not remove {path.name}", flush=True)
            continue
        removed.append((path.name, size, age_seconds))
        print(f"  swept {path.name}: {size / 1_048_576:.1f} MB, "
              f"{age_seconds / 3600:.1f}h old", flush=True)

    if not removed:
        print("  no orphaned staging directory to sweep", flush=True)
    return removed


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
                    restart: bool = True,
                    restart_fn=None, health_fn=None) -> int:
    """Point `current` at `identifier`, restart, and roll back if it fails.

    EXTRACTED SO THE ROLLBACK IS TESTED BY THE CODE PRODUCTION RUNS. This is
    the one path that decides whether a bad day can take the app down, and a
    rollback verified by a harness that reimplements it is not verified at
    all - the same reason the retrain harness invokes continuous_retrain.py
    as a subprocess rather than calling into a copy of its logic.

    `restart_fn` and `health_fn` DEFAULT TO PRODUCTION and the daily job
    passes neither, so nothing about the live path changes. They exist so the
    verifier can aim this same function at a service reading a COPY of the
    volume: before them, the only way to exercise the rollback was against the
    live volume and the live service, and an interrupted run then left the
    live pointer naming a deliberately unbootable snapshot. A seam here is
    cheaper than that, and it keeps the function under test the real one.
    """
    restart_fn = restart_fn or restart_service
    health_fn = health_fn or served_snapshot
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
    restart_fn(PROJECT)
    serving = health_fn()

    if serving.get("snapshot") == identifier:
        print(f"  /health confirms snapshot {serving.get('snapshot')}")
        return EXIT_OK

    section("THE SERVICE DID NOT COME UP - ROLLING BACK")
    print(f"  /health reported {serving.get('snapshot')!r}, "
          f"expected {identifier!r}")
    write_pointer(volume, previous)
    print(f"  pointer restored to {previous}")
    restart_fn(PROJECT)
    recovered = health_fn()
    print(f"  after restart /health reports {recovered.get('snapshot')!r}")
    if recovered.get("snapshot") == previous:
        print("  the previous snapshot is serving again")
    else:
        print("  THE SERVICE IS STILL NOT SERVING. Manual attention needed.")
    return EXIT_FAILED


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", default=str(DEFAULT_VOLUME))
    # DERIVED, NEVER A LITERAL. A hand-written list silently drops a
    # league the moment one is added: the NFL was in LEAGUES and absent
    # from this default, and daily-refresh.yml passes no --leagues at
    # all, so every scheduled run would have rebuilt three leagues and
    # then staged the fourth's tables from whatever the repo happened
    # to hold - stale on a machine that had them, a hard failure on a
    # clean checkout. Same shape as the hardcoded season list and the
    # hardcoded row counts: an expectation with no expiry inside
    # something built to run unattended.
    parser.add_argument("--leagues", default=",".join(LEAGUES))
    parser.add_argument("--step-timeout", type=int, default=5400)
    parser.add_argument("--full-refetch", action="store_true",
                        help="re-fetch EVERY season, including completed "
                             "ones, and report how the result differs from "
                             "what is being served. SWAPS NOTHING. The "
                             "daily path reuses completed seasons, so this "
                             "is the only way to see the source correcting "
                             "old data - and whether to adopt such a change "
                             "is a decision, not something a morning job "
                             "should take on its own.")
    parser.add_argument(
        "--force-swap", action="store_true",
        help="Swap the staged snapshot in even when no league's cutoff "
             "advanced. For a correction that changes the CONTENT of the "
             "served tables without changing any date - A7's margin fix is "
             "the first - where an ordinary run correctly reports nothing-new "
             "and correctly swaps nothing. Same validation, same smoke test, "
             "same lock, same rollback; its own outcome word, because "
             "borrowing 'advanced' would claim new games arrived.")
    parser.add_argument("--skip-rebuild", action="store_true",
                        help="stage and swap whatever the repo already "
                             "holds; for exercising the swap itself")
    parser.add_argument("--stats-file",
                        help="write the fetch steps' request counters here, "
                             "one per line, for the job summary to read. "
                             "Same shape as --outcome-file: the workflow "
                             "cannot grep a step's stdout, because run_step "
                             "captures and discards it.")
    parser.add_argument(
        "--summary-file",
        help="append the per-league outcome table here as markdown, for the "
             "workflow summary. The job already prints it to the step log, "
             "but a 12-minute log is not where anyone looks on a green "
             "morning - and the `partial` summary text promised a table that "
             "was never rendered.")
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

    # BEFORE ANYTHING ELSE WRITES OR READS EITHER FILE. The summary is
    # emptied so only this run reaches GitHub, and the previous run's
    # timestamp is read while it is still the PREVIOUS one - record_outcome
    # overwrites it on every terminal path.
    start_summary(args)
    write_summary(args, previous_run_lines(volume))

    leagues = [name.strip() for name in args.leagues.split(",")
               if name.strip()]
    unknown = [name for name in leagues if name not in LEAGUES]
    if unknown:
        raise SystemExit(f"unknown league(s) {unknown}")

    def record_outcome(name: str, snapshot: str = None,
                       per_league: dict = None) -> None:
        """Which of the three outcomes this was.

        The exit code cannot carry it: `advanced` and `nothing-new` are both
        green, which is the whole point of distinguishing them. A workflow
        summary that greps stdout instead would break the first time the
        wording changed, so the job reads this file.

        ALSO WRITES THE VOLUME'S OWN RECORD, and here rather than at the swap,
        because this is the only point every terminal path passes through. A
        record written at the swap would be absent on exactly the runs that
        make the question worth asking - `nothing-new` swaps nothing, and a
        job that silently stopped being scheduled looks identical to a quiet
        season unless the quiet runs leave a mark. See
        served_data.LAST_REFRESH_FILE.

        THE RECORD IS NEVER ALLOWED TO FAIL THE RUN. It is an observation
        about the job rather than part of the swap, so a read-only volume or a
        permissions problem must not turn a successful refresh into a failed
        one - that would be the instrument breaking the thing it measures. It
        warns loudly instead, which is the one case in this file where a
        warning is the right instrument rather than a misleading green.
        """
        if args.outcome_file:
            Path(args.outcome_file).write_text(name, encoding="utf-8")
        try:
            write_last_refresh(volume, name, snapshot, per_league)
        except OSError as error:
            print(f"  WARNING: the run could not be recorded at "
                  f"{volume}: {type(error).__name__}: {error}")
        print(f"\n  outcome: {name}")

    print(__doc__)

    lock = acquire_lock(volume)
    try:
        return _run(args, volume, leagues, record_outcome)
    finally:
        release_lock(lock)


def _drift_report(args, volume, leagues, served_root, identifier,
                  record_outcome) -> int:
    """Re-fetch everything, report how it differs from what is served, swap
    nothing.

    THE DAILY PATH REUSES COMPLETED SEASONS, which is what makes historical
    rows fixed between snapshots - and the cost is that the job stops
    noticing if the source corrects old data. This is where that becomes
    visible, on demand, as a report rather than a gate.

    It stages and validates exactly as the real path does, so the comparison
    is between two snapshots rather than between a snapshot and a working
    tree. The staging directory is removed afterwards: an unswapped snapshot
    left on the volume would compete with real ones for the retention
    window.

    SCOPED TO THE SEASON-LEVEL GAME TABLES, deliberately. Player box scores
    and quarter scores resume per game by file existence - checked, not
    assumed: both call `if out_path.exists(): return "skipped"` - so a full
    re-fetch does not re-pull them and there is nothing of theirs to
    compare.
    """
    section("FULL RE-FETCH DRIFT REPORT - NOTHING WILL BE SWAPPED")
    print(f"  served snapshot: {served_root.name}")

    staging = None
    try:
        rebuild(leagues, args.step_timeout, full_refetch=True)
        staging = stage(volume, identifier + "-driftcheck")
        validate(staging)
    except Exception as error:
        section("THE RE-FETCH OR ITS VALIDATION FAILED")
        print(f"  {type(error).__name__}: {error}")
        print(f"{NEWLINE}  Nothing was swapped - this mode never swaps.")
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        record_outcome("drift-report-failed")
        return EXIT_FAILED

    section("DIFFERENCES AGAINST THE SERVED SNAPSHOT")
    any_difference = False
    for league in leagues:
        relative = LEAGUES[league]["cutoff_table"]
        fresh = pd.read_csv(staging / relative, low_memory=False)
        served = pd.read_csv(served_root / relative, low_memory=False)
        if drift_report.render(league, drift_report.compare(fresh, served)):
            any_difference = True

    section("WHAT THIS MEANS")
    if any_difference:
        print("""  The source now answers differently for at least one completed season.
  NOTHING HAS BEEN ADOPTED. Deciding whether to take a historical change is
  a judgement about which version is right, and this job does not make it.

  To adopt a change: re-fetch the affected season by hand, rebuild, and let
  the next ordinary refresh snapshot the result.""")
    else:
        print("""  Every re-fetched row matches the snapshot being served, so freezing
  completed seasons is costing nothing today.""")

    shutil.rmtree(staging, ignore_errors=True)
    print(f"{NEWLINE}  staged copy removed. "
          f"{read_pointer(volume)} is serving, untouched.")
    record_outcome("drift-report")
    return EXIT_OK


def availability_lines() -> list:
    """One line per league on whether availability is actually working.

    NEVER RAISES, AND NEVER FAILS THE REFRESH. An injury-service outage is no
    reason to withhold fresh game data - the two are independent, and treating
    them as one would mean a sidecar restart costing a day of predictions. But
    it must be impossible to miss: before this, an unreachable sidecar meant the
    NBA was served on 34 of 38 features with every signal saying healthy.
    """
    try:
        sys.path.insert(0, str(HERE))
        import injury_availability as availability

        state = availability.availability_state()
        nba = state.get("state")
        detail = (state.get("detail") or "").strip()
        loud = nba in ("unreachable", "source_failed")

        with_features = [name for name, config in LEAGUES.items()
                         if config.get("availability")]
        if with_features != ["nba"]:
            # injury_availability.availability_state() answers for the NBA
            # alone. If another league ever declares availability features it
            # needs its own state, and silently reusing the NBA's would be the
            # one-date-under-three-leagues bug again.
            return [f"  availability declared for {with_features} but only the "
                    f"NBA has a state to report - this needs a per-league "
                    f"lookup before it can be trusted"]

        lines = []
        if loud:
            lines.append(f"  {'nba':<9}AVAILABILITY {str(nba).upper()}")
        else:
            lines.append(f"  {'nba':<9}availability {nba}")
        if detail:
            lines.append(f"           {detail[:150]}")
        if loud:
            lines.append("           The four availability features resolve to NaN, so the")
            lines.append("           NBA is served on 34 of 38 features. Game data is")
            lines.append("           unaffected; this does NOT fail the refresh.")
        for name, config in LEAGUES.items():
            if config.get("availability"):
                continue
            lines.append(f"  {name:<9}availability not_applicable "
                         f"(no such features)")
        return lines
    except Exception as error:  # noqa: BLE001
        return [f"  availability state could not be determined: "
                f"{type(error).__name__}: {error}"]

# --------------------------------------------------------------------------
# PER-LEAGUE ISOLATION
#
# A league that fails keeps serving its previous tables while the others
# advance. Before this, every league's steps ran inside one try block and one
# handler printed FAILED BEFORE ANY SWAP, so a bad NFL edit - and the NFL is
# read from Wikipedia pages anyone can edit - froze the NBA mid-season.
#
# The unit of success is the LEAGUE, never the table. The NBA's three tables
# are built by ten steps that feed each other; a snapshot carrying a fresh
# games_final.csv beside a carried quarter_half_raw.csv would be a mixture no
# validator checks for and no model was trained on.
OUTCOME_ADVANCED = "advanced"
OUTCOME_UNCHANGED = "unchanged"
OUTCOME_CARRIED = "carried"
OUTCOME_SKIPPED = "skipped"


def league_tables(league: str) -> list:
    """The tables this league contributes to a snapshot, from LEAGUES.

    REQUIRED ONLY. Every existing caller means "the tables whose absence is a
    failure", and widening this to include the optional ones would have
    changed all of them silently - stage_league_from_repo would raise on a
    league that simply had nothing to quarantine.
    """
    return list(LEAGUES[league]["tables"])


def league_optional_tables(league: str) -> list:
    """Tables this league MAY contribute. Absence is a fact, not a failure.

    Separate from league_tables for the reason that function's docstring
    gives. The split is also what makes the two compatibility directions
    testable on their own: an old snapshot has none of these, a new one has
    them, and both must swap, carry and serve.
    """
    return list(LEAGUES[league].get("optional_tables") or [])


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def league_hashes(root: Path, league: str) -> dict:
    """{relative path as text: sha256} for one league's tables under root.

    Optional tables are included WHEN PRESENT, because the carried-unchanged
    assertion has to cover them: a quarantine file that changed between being
    copied and being swapped is as much of a mixture as a games table that
    did. An absent one is omitted rather than recorded as None - None here
    means "required and missing", which is a different thing and is what the
    staging code raises on.
    """
    out = {}
    for relative in league_tables(league):
        path = root / relative
        out[relative.as_posix()] = sha256_of(path) if path.is_file() else None
    for relative in league_optional_tables(league):
        path = root / relative
        if path.is_file():
            out[relative.as_posix()] = sha256_of(path)
    return out


def stage_league_from_repo(league: str, staging: Path) -> None:
    """Copy this league's freshly built tables out of the repo."""
    for relative in league_tables(league):
        source = REPO_DATA_ROOT / relative
        if not source.is_file():
            raise RuntimeError(
                f"the {league} rebuild did not produce {relative}")
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for relative in league_optional_tables(league):
        source = REPO_DATA_ROOT / relative
        if not source.is_file():
            # NOT A FAILURE. build_nfl_tables.py writes an empty quarantine
            # file every run, so in practice this is only reached by an older
            # build script - and a snapshot without the table is servable by
            # construction. Raising here would turn a tolerated absence into
            # the league being carried for a reason that is not a problem.
            continue
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def carry_league(league: str, staging: Path, previous_root: Path) -> dict:
    """Copy this league's tables from the snapshot being served, byte for byte.

    Returns the hashes copied, so the caller can assert them again after
    everything else has been staged - a carried table that changed between
    being copied and being swapped is the one failure this cannot tolerate.
    """
    for relative in league_tables(league):
        source = previous_root / relative
        if not source.is_file():
            # NOT RECOVERABLE BY CARRYING. If the snapshot being served does
            # not have the table either, there is nothing to carry forward and
            # the staged snapshot would be unservable.
            raise RuntimeError(
                f"cannot carry {league}: the snapshot being served has no "
                f"{relative}")
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for relative in league_optional_tables(league):
        source = previous_root / relative
        if not source.is_file():
            # CARRYING AN OLD SNAPSHOT FORWARD. The snapshot being served may
            # predate this table entirely; the carried league then has no
            # quarantine file and /health reports it absent rather than
            # clean. Copying nothing is the honest outcome - inventing an
            # empty file here would assert that nobody checked found nothing.
            continue
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return league_hashes(previous_root, league)


def carried_from_for(league: str, previous_root: Path, previous_id: str) -> str:
    """Which snapshot a carried league's tables ORIGINALLY came from.

    A league carried for several days keeps its first `carried_from`, so the
    age of the data it is serving stays visible instead of resetting to
    yesterday every morning.
    """
    try:
        meta = json.loads(
            (previous_root / "SNAPSHOT.json").read_text(encoding="utf-8"))
    except Exception:                                           # noqa: BLE001
        return previous_id
    entry = (meta.get("leagues") or {}).get(league) or {}
    return entry.get("carried_from") or previous_id


def rebuild_league(league: str, timeout: int, env: dict = None) -> list:
    """Run one league's steps. Raises on the first failure."""
    collected = []
    print(f"  {league}")
    for script in LEAGUES[league]["steps"]:
        if not script.is_file():
            # GLOBAL, not league-local, and deliberately raised as such by the
            # caller: a missing step script means a broken checkout, and
            # carrying one league would hide a bad deploy behind a partial
            # green.
            raise MissingStepScript(f"{script} does not exist")
        collected.extend(
            f"{script.name}: {line}" for line in run_step(script, timeout,
                                                          env=env))
    return collected


class MissingStepScript(RuntimeError):
    """A step script is absent - a broken checkout, so a GLOBAL failure."""


def describe_failure(error: BaseException) -> str:
    r"""A carried league's reason, written so an operator can act on it.

    THIS USED TO BE `f"{type(error).__name__}: {error}"` TRUNCATED TO 160
    CHARACTERS, and for the one failure that actually happens it told you
    nothing. `subprocess.TimeoutExpired`'s str() begins with the whole command
    list - interpreter path, flags, script path - so 160 characters ran out
    halfway through the absolute path and the message read:

        TimeoutExpired: Command '['C:\Users\...\python.exe', '-W',
        'ignore', 'C:\Users\User\Desktop\basketball-pr

    Neither WHICH script timed out nor AFTER HOW LONG survived the cut, and
    those are the only two things worth knowing. Observed in the A3 switch
    rehearsal on 2026-10-24, where the G League was carried and the run
    reported `partial` without saying what had hung.

    So the two cases that carry structured information are named, and
    everything else keeps the old generic form - still one line, but no longer
    cut mid-word.
    """
    if isinstance(error, subprocess.TimeoutExpired):
        command = error.cmd if isinstance(error.cmd, (list, tuple)) else [error.cmd]
        script = next((Path(str(part)).name for part in reversed(command)
                       if str(part).endswith(".py")), "the step")
        return (f"TimeoutExpired: {script} did not finish within "
                f"{error.timeout:g}s")
    if isinstance(error, subprocess.CalledProcessError):
        command = error.cmd if isinstance(error.cmd, (list, tuple)) else [error.cmd]
        script = next((Path(str(part)).name for part in reversed(command)
                       if str(part).endswith(".py")), "the step")
        return f"CalledProcessError: {script} exited {error.returncode}"
    text = f"{type(error).__name__}: {error}".splitlines()[0]
    return text if len(text) <= 200 else text[:197] + "..."


def stage_per_league(args, volume, leagues, previous, previous_root,
                     identifier) -> tuple:
    """Build or carry every league into one staged snapshot.

    Returns (staging, per_league, carried_hashes, fetch_stats) or raises
    SystemExit-worthy RuntimeError for a GLOBAL failure. A league-local
    failure never raises out of here: it is carried and recorded.
    """
    staging = volume / SNAPSHOTS_DIR / f"{identifier}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    env = None
    if getattr(args, "full_refetch", False):
        env = dict(os.environ, MODELLARIUM_FULL_REFETCH="1")

    per_league, carried_hashes, fetch_stats = {}, {}, []

    section("REBUILDING AND STAGING, PER LEAGUE")
    for league in leagues:
        try:
            if args.skip_rebuild:
                print(f"  {league}: rebuild skipped, staging what this "
                      f"checkout holds")
            else:
                fetch_stats.extend(
                    rebuild_league(league, args.step_timeout, env=env))
            stage_league_from_repo(league, staging)
            per_league[league] = {"outcome": None, "reason": None,
                                  "carried_from": None}
            print(f"    {league} staged fresh")
        except MissingStepScript:
            raise
        except Exception as error:                              # noqa: BLE001
            reason = describe_failure(error)
            print(f"    {league} FAILED: {reason}")
            # THE SHORT REASON TRAVELS; THE FULL TEXT STILL GETS PRINTED.
            # `reason` has to fit a markdown cell, so it is one line - but
            # `run_step` puts the failing step's stdout and stderr tail into
            # the exception, and that is the only diagnostic there is. The
            # first version of this printed `reason` alone and threw the tail
            # away, which was noticed the first time a real fetch failed.
            detail = str(error).strip()
            if detail and detail != reason:
                for line in detail.splitlines():
                    print(f"      | {line}")
            carried_hashes[league] = carry_league(league, staging,
                                                  previous_root)
            origin = carried_from_for(league, previous_root, previous)
            per_league[league] = {"outcome": OUTCOME_CARRIED,
                                  "reason": reason, "carried_from": origin}
            print(f"    {league} CARRIED from {origin}")

    # A LEAGUE NOT SELECTED FOR THIS RUN STILL HAS TO BE IN THE SNAPSHOT, or
    # the service refuses to boot on a missing table. --leagues is a debugging
    # convenience, not a way to produce a smaller snapshot.
    for league in LEAGUES:
        if league in per_league:
            continue
        carried_hashes[league] = carry_league(league, staging, previous_root)
        per_league[league] = {
            "outcome": OUTCOME_SKIPPED,
            "reason": "not selected for this run",
            "carried_from": carried_from_for(league, previous_root, previous)}
        print(f"    {league} SKIPPED, carried from "
              f"{per_league[league]['carried_from']}")

    return staging, per_league, carried_hashes, fetch_stats


def smoke_per_league(staging: Path) -> dict:
    """Per-league servability of the staged snapshot."""
    probe = HERE / "smoke_serve_snapshot.py"
    results_path = staging.parent / f"{staging.name}.smoke.json"
    done = subprocess.run(
        [sys.executable, "-W", "ignore", str(probe), str(staging),
         "--json", str(results_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=PROJECT, timeout=900)
    print((done.stdout or "").rstrip())

    if not results_path.is_file():
        # NO STRUCTURED RESULT MEANS NO ATTRIBUTION, so this cannot be turned
        # into a per-league decision and is global by definition.
        raise RuntimeError(
            "the smoke test wrote no per-league result"
            + (f": {(done.stderr or '')[-800:]}" if done.returncode else ""))
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    results_path.unlink(missing_ok=True)
    return payload.get("leagues") or {}


def carry_unservable_leagues(staging, per_league, carried_hashes, previous,
                             previous_root) -> list:
    """Smoke-test the staged snapshot; carry whichever leagues cannot serve.

    At most two smoke runs. The carried tables were servable yesterday, so if
    the second run still fails, no servable snapshot can be assembled from
    this material and the run is a global failure.
    """
    section("SERVING SMOKE TEST AGAINST THE STAGED SNAPSHOT, PER LEAGUE")
    results = smoke_per_league(staging)
    failing = sorted(name for name, entry in results.items()
                     if not entry.get("ok", True))
    if not failing:
        return []

    newly_carried = []
    for league in failing:
        if league not in LEAGUES:
            raise RuntimeError(
                f"the smoke test reported league {league!r}, which is not in "
                f"LEAGUES - refusing to guess what to carry")
        if per_league.get(league, {}).get("outcome") in (OUTCOME_CARRIED,
                                                         OUTCOME_SKIPPED):
            raise RuntimeError(
                f"{league} was already carried from {previous} and STILL "
                f"cannot serve: {results[league].get('failure')}")
        reason = f"smoke test: {results[league].get('failure')}"
        print(f"  {league} cannot serve from the staged snapshot - carrying it")
        carried_hashes[league] = carry_league(league, staging, previous_root)
        per_league[league] = {
            "outcome": OUTCOME_CARRIED, "reason": reason,
            "carried_from": carried_from_for(league, previous_root, previous)}
        newly_carried.append(league)

    print("  re-running the smoke test with those leagues carried")
    results = smoke_per_league(staging)
    still = sorted(name for name, entry in results.items()
                   if not entry.get("ok", True))
    if still:
        raise RuntimeError(
            f"even with {newly_carried} carried, these leagues cannot serve: "
            f"{still}")
    return newly_carried


def assert_carried_unchanged(staging, carried_hashes, previous_root) -> int:
    """Every carried table must still be byte-identical to its source.

    Checked immediately before the swap rather than at copy time: the point is
    that nothing modified a carried table while the rest of the snapshot was
    being assembled.
    """
    checked = 0
    for league, expected in sorted(carried_hashes.items()):
        actual = league_hashes(staging, league)
        source = league_hashes(previous_root, league)
        for relative, digest in expected.items():
            checked += 1
            if actual.get(relative) != digest or source.get(relative) != digest:
                raise RuntimeError(
                    f"carried table {relative} for {league} is not identical "
                    f"to the snapshot it came from (staged "
                    f"{str(actual.get(relative))[:12]}, source "
                    f"{str(source.get(relative))[:12]}, copied "
                    f"{str(digest)[:12]})")
    return checked


def classify_outcomes(per_league, before, after) -> None:
    """Fill in advanced / unchanged for the leagues that built."""
    for league, entry in per_league.items():
        if entry["outcome"] is not None:
            continue
        was, now = before.get(league), after.get(league)
        entry["data_as_of"] = now
        entry["outcome"] = (OUTCOME_ADVANCED if was != now
                            else OUTCOME_UNCHANGED)
    for league, entry in per_league.items():
        entry.setdefault("data_as_of", after.get(league))


def outcome_table(per_league, before) -> list:
    """The per-league summary, as markdown rows."""
    lines = ["| league | outcome | data_as_of | carried_from | reason |",
             "|---|---|---|---|---|"]
    for league in sorted(per_league):
        entry = per_league[league]
        reason = (entry.get("reason") or "").splitlines()
        reason = (reason[0][:120] if reason else "")
        lines.append(
            f"| {league} | {entry['outcome']} | "
            f"{entry.get('data_as_of') or '-'} | "
            f"{entry.get('carried_from') or '-'} | {reason or '-'} |")
    return lines


def start_summary(args) -> None:
    """Empty the summary file, so only THIS run's content reaches GitHub.

    MEASURED 2026-10-29: the summary file on the runner held **eleven**
    per-league tables. `write_summary` opens it with mode "a", the workflow
    passes a fixed relative path (`refresh-summary.md`) and then pipes the
    whole file into $GITHUB_STEP_SUMMARY - and nothing ever removed it,
    because the runner's checkout uses `clean: false` so that the ignored
    `data/raw/` corpus survives. So every run appended to every previous run's
    content and Monday's job summary showed Friday's, Saturday's twice,
    Sunday's and Monday's tables, newest last.

    TRUNCATED HERE RATHER THAN IN THE WORKFLOW, which was the other option.
    A `Remove-Item` step would fix the scheduled job and leave the defect in
    place for every local and rehearsal invocation - and the rehearsal is
    exactly where a second run's table appearing under the first would be
    read as the harness having done something. One place, however the script
    is invoked.

    NOT ALLOWED TO FAIL THE RUN, for the same reason the outcome record is
    not: this is presentation, and an unwritable path must not turn a good
    refresh into a failed one. It warns, which is the one kind of place in
    this file where a warning is the right instrument.
    """
    if not args.summary_file:
        return
    try:
        with open(args.summary_file, "w", encoding="utf-8"):
            pass
    except OSError as error:
        print(f"WARNING: could not empty {args.summary_file}: "
              f"{type(error).__name__}: {error}", flush=True)


def previous_run_lines(volume: Path) -> list:
    """How long since the previous run, read BEFORE this one overwrites it.

    READ AT STARTUP, WHICH IS THE WHOLE POINT. `record_outcome` rewrites
    last_refresh.json on every terminal path, so by the time the summary is
    assembled the previous run's timestamp is gone. This is called before any
    step runs.

    THE SAME INSTRUMENT /health USES, deliberately. `read_last_refresh` owns
    the thresholds (late at REFRESH_LATE_HOURS, overdue at
    REFRESH_OVERDUE_HOURS) and the derivation, so the summary and /health
    cannot disagree about whether a gap is late - two places computing the
    same judgement is how they end up saying different things on the same
    morning.

    BUT THE WARNING IS KEYED ON THE HOURS, NOT ON `state`. `read_last_refresh`
    returns `failed` in preference to `late`, short-circuiting before the
    hours are examined - so a previous run that failed 27 hours ago reports
    `failed` and would, if the warning keyed on the state, print no late
    warning at all. Both facts are worth printing and they are printed
    separately.

    AN ABSENT RECORD IS AN UNKNOWN, NOT A ZERO. "0.0 h since the previous
    run" on a volume that has never been refreshed is the conflation this
    project has four recorded bugs from, so it says so in words.
    """
    record = read_last_refresh(volume)
    state = record.get("state")
    hours = record.get("hours_since")
    outcome = record.get("outcome")
    stamp = record.get("last_verdict_at")

    if state == "not_a_volume":
        return ["_No previous run recorded: this DATA_DIR is a plain data "
                "root rather than a volume, so no refresh job maintains a "
                "record here._"]
    if hours is None:
        return [f"_No previous run recorded (`{state}`), so the gap since "
                f"the last refresh is **unknown** rather than zero._"]

    lines = [f"_Previous run: **{hours:.1f} h** ago "
             f"(`{outcome}`, {stamp})._"]
    if hours >= REFRESH_LATE_HOURS:
        lines.append("")
        lines.append(
            f"> **Late: {hours:.1f} h since the previous run, past the "
            f"{REFRESH_LATE_HOURS} h threshold.** The cron QUEUES at 07:00 "
            f"and this runner starts when the host wakes, so a gap of around "
            f"half a day is ordinary here; this one is longer than that.")
    if outcome in ("failed", "partial", "drift-report-failed"):
        lines.append("")
        lines.append(
            f"> **The previous run ended `{outcome}`.** Whatever is being "
            f"served may be older than the timestamp above.")
    return lines


def write_summary(args, lines) -> None:
    """Append markdown to the workflow summary, if one was asked for.

    A BLANK LINE FIRST, ALWAYS, and it is not cosmetic. GitHub renders a table
    as a table only when a blank line separates it from the text above; put a
    `|` row straight after a paragraph and it is rendered as literal pipe
    characters. The existing summary had exactly that defect - the per-league
    availability rows were emitted after the fetch-counter heading, orphaned
    from their table header, and showed as raw text.
    """
    if not args.summary_file:
        return
    with open(args.summary_file, "a", encoding="utf-8") as handle:
        handle.write(NEWLINE)
        for line in lines:
            handle.write(line + NEWLINE)


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

    # INSIDE THE LOCK, which main() holds for the whole of _run, and before
    # anything stages: a sweep racing a concurrent run is exactly how you
    # delete a staging directory someone is writing into.
    section("SWEEP ORPHANED STAGING")
    sweep_staging(volume, identifier)

    if args.full_refetch:
        return _drift_report(args, volume, leagues, previous_root,
                             identifier, record_outcome)

    try:
        staging, per_league, carried_hashes, fetch_stats = stage_per_league(
            args, volume, leagues, previous, previous_root, identifier)

        if args.stats_file:
            Path(args.stats_file).write_text(
                chr(10).join(fetch_stats), encoding="utf-8")

        section("VALIDATING THE STAGED SNAPSHOT")
        missing = missing_files(staging)
        if missing:
            raise RuntimeError(f"staged snapshot is missing {missing}")
        print(f"  all {len(REQUIRED_FILES)} required tables present")

        after = cutoffs(staging)
        for league, date in after.items():
            print(f"  {league:<9}data_as_of {date}")

        carry_unservable_leagues(staging, per_league, carried_hashes,
                                 previous, previous_root)
        after = cutoffs(staging)

        checked = assert_carried_unchanged(staging, carried_hashes,
                                           previous_root)
        if carried_hashes:
            print(f"  {checked} carried table(s) verified byte-identical to "
                  f"{previous}")
    except Exception as error:                                  # noqa: BLE001
        section("FAILED BEFORE ANY SWAP")
        print(f"  {type(error).__name__}: {error}")
        print(f"{chr(10)}  {previous} IS STILL SERVING. Nothing was swapped.")
        record_outcome("failed")
        return EXIT_FAILED

    classify_outcomes(per_league, before, after)
    advanced = {league: (before.get(league), after.get(league))
                for league, entry in per_league.items()
                if entry["outcome"] == OUTCOME_ADVANCED}
    carried = sorted(league for league, entry in per_league.items()
                     if entry["outcome"] == OUTCOME_CARRIED)

    section("WHAT CHANGED, PER LEAGUE")
    table = outcome_table(per_league, before)
    for line in table:
        print("  " + line)
    write_summary(args, ["### Per league", ""] + table)

    print()
    for line in availability_lines():
        print(line)

    if not advanced and not carried and not args.force_swap:
        section("NOTHING NEW")
        print("""  No league has a game later than the snapshot already being serving, and
  none failed, so there is nothing to swap in. An off-day, a break, or every
  league between seasons.

  This is GREEN and not a failure. The staged snapshot is discarded rather
  than kept, because an identical snapshot under a new name would make the
  retention window shorter for no gain.""")
        shutil.rmtree(staging, ignore_errors=True)
        print(f"{chr(10)}  {previous} continues to serve.")
        # THE PREVIOUS SNAPSHOT IS THE ONE BEING SERVED, so that is the id
        # recorded - not the staged identifier, which was just discarded.
        record_outcome("nothing-new", previous, per_league)
        return EXIT_OK

    if not advanced and carried and not args.force_swap:
        # NOTHING GAINED AND SOMETHING LOST. Swapping would replace the served
        # snapshot with one that is identical except for a new name, while
        # recording a failure - so the pointer stays and the run goes red.
        section("NO LEAGUE ADVANCED AND AT LEAST ONE FAILED")
        print(f"  carried: {', '.join(carried)}")
        print(f"  Nothing newer exists to serve, so the pointer is NOT moved "
              f"and {previous} continues to serve.")
        shutil.rmtree(staging, ignore_errors=True)
        record_outcome("failed", previous, per_league)
        return EXIT_FAILED

    final = volume / SNAPSHOTS_DIR / identifier
    write_snapshot_metadata(staging, identifier, source="daily_refresh",
                            per_league=after, leagues=per_league)
    staging.rename(final)

    swapped = swap_and_verify(volume, previous, identifier,
                              restart=not args.no_restart)
    if swapped == EXIT_OK and not args.no_restart:
        if advanced:
            section("ADVANCED")
            for league, (was, now) in advanced.items():
                print(f"  {league:<9}{was} -> {now}")
        if carried:
            section("CARRIED")
            for league in carried:
                entry = per_league[league]
                print(f"  {league:<9}serving {entry['carried_from']}: "
                      f"{(entry.get('reason') or '').splitlines()[0][:120]}")
        prune(volume, KEEP_SNAPSHOTS)

    if swapped != EXIT_OK:
        # THE POINTER HAS BEEN ROLLED BACK by swap_and_verify, so what is
        # being served is the previous snapshot and not the one just written.
        record_outcome("failed", previous, per_league)
        return swapped
    if carried:
        record_outcome("partial", identifier, per_league)
        return EXIT_PARTIAL
    record_outcome("advanced" if advanced else "forced", identifier,
                   per_league)
    return EXIT_OK


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001
        print(f"\nUNHANDLED: {type(error).__name__}: {error}")
        sys.exit(EXIT_FAILED)
