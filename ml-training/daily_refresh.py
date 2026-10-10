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
import pandas as pd  # noqa: E402
import drift_report  # noqa: E402
from served_data import (  # noqa: E402
    POINTER_FILE, REPO_DATA_ROOT, REQUIRED_FILES, SNAPSHOTS_DIR,
    missing_files, resolve_root, write_snapshot_metadata)

DEFAULT_VOLUME = Path("D:/modellarium-data")
KEEP_SNAPSHOTS = 7

NEWLINE = chr(10)

EXIT_OK = 0
EXIT_FAILED = 1

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

    if args.full_refetch:
        return _drift_report(args, volume, leagues, previous_root,
                             identifier, record_outcome)

    try:
        fetch_stats = []
        if args.skip_rebuild:
            section("REBUILD SKIPPED")
            print("  staging whatever this checkout already holds")
        else:
            fetch_stats = rebuild(leagues, args.step_timeout)

        # WRITTEN HERE, NOT AT THE END. A run that fails during validation or
        # the swap still made the requests, so its counters are exactly the
        # ones worth reading - writing them only on success would lose them
        # on the morning they matter most.
        if args.stats_file:
            Path(args.stats_file).write_text(
                chr(10).join(fetch_stats), encoding="utf-8")

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
        if before.get(league) is None:
            # A league the previous snapshot did not carry at all. ADDED
            # rather than ADVANCED, because no date moved - a table appeared.
            print(f"  {league:<9}absent -> {after[league]}   ADDED "
                  f"(not in the previous snapshot)")
            continue
        mark = "ADVANCED" if league in advanced else "unchanged"
        print(f"  {league:<9}{before[league]} -> {after[league]}   {mark}")

    print()
    for line in availability_lines():
        print(line)

    if not advanced and args.force_swap:
        section("FORCED SWAP")
        print("""  No league's cutoff moved, so an ordinary run would discard this snapshot
  and keep serving the previous one. --force-swap was passed, which says the
  CONTENT changed without any date changing - a correction rather than new
  games. It has been validated and smoke-tested exactly as any other snapshot;
  only the decision to swap differs.""")

    if not advanced and not args.force_swap:
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
        if advanced:
            section("ADVANCED")
            for league, (was, now) in advanced.items():
                print(f"  {league:<9}{was} -> {now}")
        else:
            section("FORCED")
            print("  content changed, no cutoff moved. Every league's "
                  "data_as_of is unchanged by design.")
        prune(volume, KEEP_SNAPSHOTS)
    if swapped != EXIT_OK:
        record_outcome("failed")
    else:
        record_outcome("advanced" if advanced else "forced")
    return swapped


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001
        print(f"\nUNHANDLED: {type(error).__name__}: {error}")
        sys.exit(EXIT_FAILED)
