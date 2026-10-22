r"""Both directions of the quarantine-table upgrade must work.

THE TABLE IS OPTIONAL, AND THIS IS THE CHECK THAT THE OPTIONALITY IS REAL
rather than only declared. Two directions, and they fail differently:

  OLD SNAPSHOT, NEW IMAGE   the seven snapshots already on the volume have no
                            nfl_quarantine.csv. The new image must boot
                            against them and report the quarantine state as
                            UNKNOWN - not as zero games quarantined, which
                            would be a fact about data nobody checked. This
                            direction is the one that matters on the day the
                            change ships: it is where a rollback lands.

  NEW SNAPSHOT, OLD IMAGE   a refresh that writes the table must produce a
                            snapshot the PREVIOUS image still boots, or the
                            rollback path has a one-way door in it. Tested
                            against a real checkout of HEAD, not a simulation
                            of one - the point is code that has never heard of
                            the table, and a monkeypatched copy of the new
                            code is not that.

Also checked: the refresh can CARRY either shape forward. A carried old
snapshot must not grow an invented empty table, and a carried new one must
arrive byte-identical.

READ-ONLY with respect to the served volume. Everything is built in --work,
which defaults to a scratch directory, and the git worktree it creates for the
old image is removed in a `finally`.

    python ml-training/verify_quarantine_compatibility.py
"""
import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPO_DATA = PROJECT / "data-pipeline" / "data"
QUARANTINE_REL = Path("nfl") / "processed" / "nfl_quarantine.csv"

RESULTS = []
BOOT_TIMEOUT_SECONDS = 180


class HarnessFailure(RuntimeError):
    """The harness could not set up the thing it meant to test."""


def record(name, passed, detail):
    RESULTS.append((name, passed, detail))
    print("  %-4s %-54s %s" % ("PASS" if passed else "FAIL", name, detail))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def boot_and_get_health(tree: Path, data_dir: Path, label: str) -> tuple:
    """Boot one checkout's inference service against one data root.

    Returns (health_body_or_None, tail_of_log). A refusal is a RESULT here,
    not an error - one of the plants is a boot that must fail - so this
    reports rather than raises.

    THE INTERPRETER IS NAMED, not found on PATH: this project's own runner
    notes record that a PATH-resolved python is how a job ends up using a
    different environment than the one it was verified in.
    """
    python = PROJECT / ".venv" / "Scripts" / "python.exe"
    if not python.is_file():
        raise HarnessFailure(f"no interpreter at {python}")

    port = free_port()
    env = dict(os.environ)
    env["DATA_DIR"] = str(data_dir)
    # The synthetic guard must not fire - these are the real repo tables.
    env.pop("MODELLARIUM_ALLOW_SYNTHETIC_NFL", None)
    # The sidecar is not part of this question. Availability resolving to
    # `never_fetched` is fine; what is being tested is whether the service
    # boots and what it says about the NFL quarantine.
    log = data_dir.parent / f"boot-{label}.log"

    with log.open("wb") as handle:
        process = subprocess.Popen(
            [str(python), "-W", "ignore", "-m", "uvicorn", "app:app",
             "--port", str(port)],
            cwd=str(tree / "inference-service"), env=env,
            stdout=handle, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + BOOT_TIMEOUT_SECONDS
        while time.time() < deadline:
            if process.poll() is not None:
                # REFUSED TO BOOT. Reported as such, with the log tail, since
                # one plant requires exactly this.
                return None, tail(log)
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=5) as body:
                    return json.loads(body.read().decode("utf-8")), tail(log)
            except (urllib.error.URLError, OSError, TimeoutError):
                time.sleep(1)
        return None, "TIMED OUT waiting for /health\n" + tail(log)
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()


def tail(path: Path, lines: int = 6) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(no log)"
    return " | ".join(text.strip().splitlines()[-lines:])


def build_snapshots(work: Path) -> tuple:
    """A NEW-shape and an OLD-shape snapshot, from the repo data copy.

    The OLD one is the NEW one with the quarantine table DELETED rather than a
    separate build, so the two differ in exactly that file and nothing else -
    any difference in the results is attributable to the table alone.
    """
    sys.path.insert(0, str(HERE))
    from served_data import REQUIRED_FILES

    new_root = work / "snapshot-new"
    old_root = work / "snapshot-old"
    for root in (new_root, old_root):
        if root.exists():
            shutil.rmtree(root)
        for relative in REQUIRED_FILES:
            source = REPO_DATA / relative
            if not source.is_file():
                raise HarnessFailure(
                    f"the repo data copy has no {relative}; this verifier "
                    f"needs a complete table set to copy from")
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    source = REPO_DATA / QUARANTINE_REL
    if not source.is_file():
        raise HarnessFailure(
            f"the repo data copy has no {QUARANTINE_REL} - run "
            f"build_nfl_tables.py first, or this verifier would be testing "
            f"two identical snapshots and passing vacuously")
    shutil.copy2(source, new_root / QUARANTINE_REL)

    if (old_root / QUARANTINE_REL).exists():
        raise HarnessFailure("the OLD snapshot grew a quarantine table")
    return new_root, old_root


def old_checkout(work: Path) -> Path:
    """A real checkout of HEAD, source only.

    SOURCE ONLY, and that is not an optimisation. A full checkout would pull
    the committed tables through Git LFS, and the one thing this must not do is
    make the old image read a data root that is not the one being tested.
    `--no-checkout` plus a sparse set that excludes data-pipeline/data, with
    the LFS smudge filter off.
    """
    tree = work / "head-checkout"
    if tree.exists():
        shutil.rmtree(tree, ignore_errors=True)
    subprocess.run(["git", "worktree", "prune"], cwd=str(PROJECT), check=False,
                   capture_output=True)

    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    for command in (
            ["git", "worktree", "add", "--no-checkout", str(tree), "HEAD"],
            ["git", "-C", str(tree), "sparse-checkout", "set", "--no-cone",
             "/*", "!/data-pipeline/data"],
            ["git", "-C", str(tree), "checkout"]):
        done = subprocess.run(command, cwd=str(PROJECT), env=env,
                              capture_output=True, text=True)
        if done.returncode != 0:
            raise HarnessFailure(
                f"{' '.join(command[:3])} failed: {done.stderr.strip()[:300]}")

    app = tree / "inference-service" / "app.py"
    served = tree / "ml-training" / "served_data.py"
    if not app.is_file() or not served.is_file():
        raise HarnessFailure("the HEAD checkout is missing serving sources")

    # THE HARNESS GUARD. If HEAD already knows about the quarantine table -
    # because the change was committed since this was written - then this is
    # not an old image and the direction it tests is untested. That is a
    # harness bug and must not look like a pass.
    for path in (app, served, tree / "ml-training" / "nfl" /
                 "live_nfl_features.py"):
        if "quarantine" in path.read_text(encoding="utf-8", errors="replace"):
            raise HarnessFailure(
                f"HEAD's {path.name} already mentions the quarantine table, "
                f"so this is not an OLD image. Point --old-ref at the commit "
                f"before the change.")
    return tree


def drop_old_checkout(tree: Path) -> None:
    subprocess.run(["git", "worktree", "remove", "--force", str(tree)],
                   cwd=str(PROJECT), capture_output=True)
    subprocess.run(["git", "worktree", "prune"], cwd=str(PROJECT),
                   capture_output=True)
    if tree.exists():
        shutil.rmtree(tree, ignore_errors=True)


def nfl_block(health: dict) -> dict:
    return (health or {}).get("nfl") or {}


def nfl_hashes(health: dict) -> dict:
    served = (health or {}).get("served_data") or {}
    return (served.get("table_sha256") or {}).get("nfl") or {}


# ==========================================================================
# the two directions
# ==========================================================================

def direction_old_snapshot_new_image(old_root: Path) -> None:
    health, log = boot_and_get_health(PROJECT, old_root, "old-snapshot")
    record("OLD snapshot, NEW image: the service boots",
           health is not None, "booted" if health else f"REFUSED: {log}")
    if health is None:
        return

    record("OLD snapshot, NEW image: models_loaded is unchanged",
           health.get("models_loaded") == {
               "team": 7, "quarter_half": 6, "player_props": 10,
               "wnba": 3, "gleague": 3, "nfl": 3},
           str(health.get("models_loaded")))

    quarantine = nfl_block(health).get("quarantine") or {}
    record("OLD snapshot: the state is UNKNOWN, not zero",
           quarantine.get("state") == "unknown",
           f"state={quarantine.get('state')!r}, "
           f"count={quarantine.get('count')!r}")
    record("OLD snapshot: the note says the snapshot predates the table",
           "predates" in str(quarantine.get("note", "")),
           str(quarantine.get("note"))[:60])
    record("OLD snapshot: the hash is reported ABSENT, not omitted",
           nfl_hashes(health).get("nfl_quarantine.csv") == "absent",
           f"nfl_quarantine.csv -> "
           f"{nfl_hashes(health).get('nfl_quarantine.csv')!r}")
    record("OLD snapshot: the three required NFL tables still hash",
           all(len(str(nfl_hashes(health).get(name, ""))) == 64
               for name in ("nfl_games_final.csv", "nfl_fixtures.csv",
                            "nfl_franchise_identity.csv")),
           f"{len(nfl_hashes(health))} NFL entries")


def direction_new_snapshot_old_image(tree: Path, new_root: Path) -> None:
    health, log = boot_and_get_health(tree, new_root, "old-image")
    record("NEW snapshot, OLD image: the service boots",
           health is not None, "booted" if health else f"REFUSED: {log}")
    if health is None:
        return

    record("NEW snapshot, OLD image: models_loaded is unchanged",
           health.get("models_loaded") == {
               "team": 7, "quarter_half": 6, "player_props": 10,
               "wnba": 3, "gleague": 3, "nfl": 3},
           str(health.get("models_loaded")))
    record("NEW snapshot, OLD image: it serves the NFL block",
           bool(nfl_block(health).get("data_as_of")),
           f"data_as_of={nfl_block(health).get('data_as_of')}, "
           f"fixtures={nfl_block(health).get('fixtures')}")
    # THE OLD IMAGE SAYS NOTHING ABOUT THE TABLE, and that is the correct
    # behaviour rather than a gap: it cannot report a field it has never heard
    # of. What matters is that the extra file in the snapshot did not stop it.
    record("NEW snapshot, OLD image: it reports no quarantine field",
           "quarantine" not in nfl_block(health),
           "absent from the NFL block, as an old image must")


def carrying_either_shape(work: Path, new_root: Path, old_root: Path) -> None:
    """The refresh must be able to carry a league forward from either shape."""
    sys.path.insert(0, str(HERE))
    import daily_refresh as D

    record("the NFL declares the table as OPTIONAL, not required",
           QUARANTINE_REL not in D.league_tables("nfl")
           and QUARANTINE_REL in D.league_optional_tables("nfl"),
           f"required={len(D.league_tables('nfl'))}, "
           f"optional={[p.name for p in D.league_optional_tables('nfl')]}")

    for label, source in (("NEW", new_root), ("OLD", old_root)):
        staging = work / f"carry-from-{label.lower()}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        hashes = D.carry_league("nfl", staging, source)

        present = (staging / QUARANTINE_REL).is_file()
        if label == "NEW":
            record("carrying a NEW snapshot copies the table byte for byte",
                   present and sha256_of(staging / QUARANTINE_REL)
                   == sha256_of(source / QUARANTINE_REL),
                   f"sha256 {sha256_of(staging / QUARANTINE_REL)[:12]}")
            record("carrying a NEW snapshot records its hash for the "
                   "pre-swap assertion",
                   QUARANTINE_REL.as_posix() in hashes,
                   f"{len(hashes)} hashes recorded")
        else:
            record("carrying an OLD snapshot invents no empty table",
                   not present,
                   "no nfl_quarantine.csv, as the source had none")
            record("carrying an OLD snapshot records no hash for it",
                   QUARANTINE_REL.as_posix() not in hashes,
                   f"{len(hashes)} hashes recorded, none for the table")

        # Every required table must still be there either way - a carry that
        # tolerated the optional table by tolerating everything would pass the
        # checks above and be useless.
        missing = [r for r in D.league_tables("nfl")
                   if not (staging / r).is_file()]
        record(f"carrying {label}: all three required tables arrive",
               not missing, f"missing {missing}" if missing else "3 of 3")


# ==========================================================================
# plants
# ==========================================================================

def plant_required_would_refuse(work: Path, old_root: Path) -> None:
    """Make the optional table REQUIRED and the old snapshot must refuse.

    THIS IS WHAT PROVES THE OPTIONALITY IS LOAD-BEARING. Without it, every
    check above would pass just as well if the table were required and the old
    snapshot happened to contain one - and the whole point of the mechanism is
    that the old snapshots do NOT contain one. The plant is applied to a COPY
    of the tree, never to the working tree.
    """
    tree = work / "plant-required"
    if tree.exists():
        shutil.rmtree(tree, ignore_errors=True)
    tree.mkdir(parents=True)
    for part in ("inference-service", "ml-training", "data-pipeline"):
        shutil.copytree(PROJECT / part, tree / part,
                        ignore=shutil.ignore_patterns(
                            "data", "probes", "__pycache__", "mlruns",
                            "*.log"))

    served = tree / "ml-training" / "served_data.py"
    text = served.read_text(encoding="utf-8")
    marker = "OPTIONAL_FILES = [\n"
    if marker not in text:
        raise HarnessFailure("OPTIONAL_FILES is not where the plant expects")
    planted = text.replace(
        marker, "OPTIONAL_FILES = []\nUNUSED_OPTIONAL = [\n", 1)
    planted = planted.replace(
        '    Path("nfl") / "processed" / "nfl_franchise_identity.csv",\n]',
        '    Path("nfl") / "processed" / "nfl_franchise_identity.csv",\n'
        '    Path("nfl") / "processed" / "nfl_quarantine.csv",\n]', 1)
    if planted == text:
        raise HarnessFailure("the plant changed nothing")
    served.write_text(planted, encoding="utf-8", newline="\n")

    health, log = boot_and_get_health(tree, old_root, "plant-required")
    record("PLANT: with the table REQUIRED, the OLD snapshot is refused",
           health is None,
           "refused, so optionality is what let it boot" if health is None
           else "BOOTED ANYWAY - the mechanism is not load-bearing")
    record("PLANT: it refuses by NAMING the missing table",
           health is None and "nfl_quarantine" in log,
           next((part.strip() for part in log.split("|")
                 if "nfl_quarantine" in part), log[-120:]))
    shutil.rmtree(tree, ignore_errors=True)


def plant_noop(work: Path, old_root: Path) -> None:
    """A plant that changes nothing must leave the result green.

    The guard against the guard: if the copy-the-tree-and-edit-it machinery
    above were broken - wrong path, edit silently not applied - the refusal it
    produced would be a harness artefact rather than the plant. This runs the
    same machinery with a comment-only edit, and the boot must succeed.
    """
    tree = work / "plant-noop"
    if tree.exists():
        shutil.rmtree(tree, ignore_errors=True)
    tree.mkdir(parents=True)
    for part in ("inference-service", "ml-training", "data-pipeline"):
        shutil.copytree(PROJECT / part, tree / part,
                        ignore=shutil.ignore_patterns(
                            "data", "probes", "__pycache__", "mlruns",
                            "*.log"))
    served = tree / "ml-training" / "served_data.py"
    text = served.read_text(encoding="utf-8")
    served.write_text(text.replace("OPTIONAL_FILES = [",
                                   "# no-op plant\nOPTIONAL_FILES = [", 1),
                      encoding="utf-8", newline="\n")

    health, _log = boot_and_get_health(tree, old_root, "plant-noop")
    quarantine = nfl_block(health).get("quarantine") or {}
    record("PLANT (no-op): the same machinery still boots and reports unknown",
           health is not None and quarantine.get("state") == "unknown",
           "green, so the refusal above was the plant and not the harness"
           if health is not None else "REFUSED - the harness is broken")
    shutil.rmtree(tree, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path,
                        help="scratch directory; a temporary one by default")
    parser.add_argument("--skip-old-image", action="store_true",
                        help="skip the HEAD checkout half (needs git)")
    args = parser.parse_args()

    work = args.work or Path(tempfile.mkdtemp(prefix="quarantine-compat-"))
    work.mkdir(parents=True, exist_ok=True)

    print("NFL QUARANTINE TABLE: BOTH COMPATIBILITY DIRECTIONS")
    print(f"  work     : {work}")
    print(f"  repo data: {REPO_DATA}")
    print()

    new_root, old_root = build_snapshots(work)
    print(f"  NEW snapshot has {QUARANTINE_REL.name}: "
          f"{(new_root / QUARANTINE_REL).is_file()}")
    print(f"  OLD snapshot has {QUARANTINE_REL.name}: "
          f"{(old_root / QUARANTINE_REL).is_file()}")
    print()

    direction_old_snapshot_new_image(old_root)
    print()

    if args.skip_old_image:
        print("  (old-image direction SKIPPED by request)")
    else:
        tree = old_checkout(work)
        try:
            direction_new_snapshot_old_image(tree, new_root)
        finally:
            drop_old_checkout(tree)
    print()

    carrying_either_shape(work, new_root, old_root)
    print()

    plant_required_would_refuse(work, old_root)
    plant_noop(work, old_root)

    passed = sum(1 for _n, ok, _d in RESULTS if ok)
    print()
    print(f"  {passed} of {len(RESULTS)} checks passed")
    if args.work is None:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except HarnessFailure as error:
        print(f"\nHARNESS FAILURE, no verdict reached: {error}")
        raise SystemExit(2)
    except Exception as error:                                  # noqa: BLE001
        print(f"\nCRASHED before a verdict: {type(error).__name__}: {error}")
        raise SystemExit(2)
