r"""Per-league isolation: one league's failure must not stop the others.

NEVER TOUCHES THE LIVE VOLUME. Every run here uses a scratch volume built by
copying a snapshot, and a scratch "repo" root so a fresh stage can be made to
succeed or fail per league without fetching anything. The live volume's path is
refused outright.

The shape of each case: point `REPO_DATA_ROOT` at a scratch directory holding
copies of the snapshot's own tables, so "freshly built" and "carried" are the
same bytes and the only variable is which leagues can be staged. Then remove or
corrupt one league's table in the scratch repo to make exactly that league
fail.

    python ml-training/verify_refresh_isolation.py
    python ml-training/verify_refresh_isolation.py --volume D:\modellarium-rehearsal
"""
import argparse
import io
import json
import shutil
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import daily_refresh as D                                       # noqa: E402
from served_data import POINTER_FILE, SNAPSHOTS_DIR             # noqa: E402

LIVE_VOLUME = Path(r"D:\modellarium-data")
DEFAULT_SCRATCH = Path(r"D:\modellarium-rehearsal")
RESULTS = []


class PlantDidNothing(RuntimeError):
    """A plant that changes no behaviour is a harness bug, not a pass."""


class RealSubprocessRefused(BaseException):
    """A verifier tried to run a real pipeline step. Named, and refused.

    DERIVES FROM BaseException, NOT Exception, AND THAT IS THE WHOLE POINT.
    The first version inherited from RuntimeError and the guard looked like it
    worked - no subprocess started - but `stage_per_league` catches
    `Exception` to carry a failing league, so the refusal was absorbed and
    reported as "nba CARRIED". The step was prevented and nothing said so,
    which is the misleading green this project keeps removing.

    The alternative was to teach `stage_per_league` to re-raise it, next to
    MissingStepScript. Rejected: production code should not have to know the
    name of a test harness's exception to behave correctly. BaseException is
    how KeyboardInterrupt passes through the same handler, for the same
    reason.
    """


def refusing_run_step(script, timeout, env=None):
    """Stand-in for `daily_refresh.run_step` that never starts a subprocess.

    WHY THIS EXISTS, FROM A REAL INCIDENT. On 2026-10-22 at 17:19 this
    verifier's missing-step-script case set `skip_rebuild = False` so the step
    list would be read - and because the leagues run in order, that executed
    the REAL NBA pipeline before reaching the planted league. It ran FOUR times before being
    noticed - once per run of this suite while the bug was present - making
    about 72 requests to the NBA's API and rewriting three tracked tables in
    the working tree each time. It was caught only because an unrelated verifier's derived pair
    count moved from 414 to 278 and that owed an explanation.
    A verifier that mutates the repository to test a code path is worse than
    the bug it is checking for.

    Installed for the WHOLE of this suite, so no future case can reintroduce
    it by accident: every path that reaches a league's pipeline step raises
    here instead, naming the script it tried to run.

    DELIBERATELY NOT THE SMOKE TEST. `smoke_per_league` starts its own
    subprocess, that subprocess only READS a staged snapshot and writes one
    JSON file beside it, and the isolation cases need it to run for real -
    carrying an unservable league is one of the behaviours under test. The
    dangerous subprocesses are the fetch-and-build steps, and those all go
    through `run_step`.
    """
    raise RealSubprocessRefused(
        f"this verifier tried to run {getattr(script, 'name', script)} as a "
        f"real subprocess. Pipeline steps fetch from live sources and write "
        f"into data-pipeline/data/, so a verifier must never start one: drive "
        f"the behaviour you need directly, or use skip_rebuild.")


def record(name, passed, detail):
    RESULTS.append((name, passed, detail))
    print("  %-4s %-58s %s" % ("PASS" if passed else "FAIL", name, detail))


def fake_args(volume):
    return SimpleNamespace(
        volume=str(volume), skip_rebuild=True, no_restart=True,
        force_swap=False, full_refetch=False, step_timeout=900,
        stats_file=None, outcome_file=None, leagues="")


def run_refresh(volume, scratch_repo, leagues=None):
    """Run _run against the scratch volume. Returns (code, outcome, output)."""
    outcome = {}

    def record_outcome(name):
        outcome["name"] = name

    original_repo = D.REPO_DATA_ROOT
    D.REPO_DATA_ROOT = scratch_repo
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            code = D._run(fake_args(volume), volume,
                          leagues or list(D.LEAGUES), record_outcome)
    finally:
        D.REPO_DATA_ROOT = original_repo
    return code, outcome.get("name"), buffer.getvalue()


def prepare(volume):
    """A scratch repo holding copies of the served snapshot's tables."""
    pointer = (volume / POINTER_FILE).read_text(encoding="utf-8").strip()
    snapshot = volume / SNAPSHOTS_DIR / pointer
    scratch_repo = volume / "scratch-repo"
    if scratch_repo.exists():
        shutil.rmtree(scratch_repo)
    for relative in D.REQUIRED_FILES:
        source = snapshot / relative
        target = scratch_repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return pointer, snapshot, scratch_repo


def snapshots_in(volume):
    return sorted(p.name for p in (volume / SNAPSHOTS_DIR).iterdir()
                  if p.is_dir() and not p.name.endswith(".staging"))


def league_of_table(relative):
    for league in D.LEAGUES:
        if relative in D.league_tables(league):
            return league
    return None


def said(output: str, league: str, phrase: str) -> bool:
    """Did the run print `<league> <phrase>` for THAT league?

    LINE-WISE AND TOKEN-EXACT, because a plain substring test is wrong here:
    "nba" is a suffix of "wnba", so `"nba staged fresh" in output` is true
    whenever the WNBA staged fresh. That read a correct NBA carry as a
    failure the first time this suite ran - the harness was wrong, not the
    subject.
    """
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped.startswith(league):
            continue
        rest = stripped[len(league):]
        if rest[:1] not in (" ", ":"):
            continue
        if phrase in rest:
            return True
    return False


def case_control(volume, scratch_repo, pointer):
    code, outcome, out = run_refresh(volume, scratch_repo)
    record("control: nothing newer, nothing failed -> nothing-new",
           code == D.EXIT_OK and outcome == "nothing-new",
           f"exit {code}, outcome {outcome!r}")
    record("control: the pointer did not move",
           (volume / POINTER_FILE).read_text(encoding="utf-8").strip() == pointer,
           f"pointer {pointer}")
    return out


def case_one_league_fails(volume, scratch_repo, pointer, snapshot, league):
    """Remove one league's tables from the scratch repo so only it fails."""
    tables = D.league_tables(league)
    backups = {}
    for relative in tables:
        path = scratch_repo / relative
        backups[relative] = path.read_bytes()
        path.unlink()
    if any((scratch_repo / r).exists() for r in tables):
        raise PlantDidNothing(f"{league}'s tables were not removed")
    try:
        code, outcome, out = run_refresh(volume, scratch_repo)
    finally:
        for relative, blob in backups.items():
            (scratch_repo / relative).write_bytes(blob)

    carried = said(out, league, "CARRIED from")
    others_fresh = all(said(out, other, "staged fresh")
                       for other in D.LEAGUES if other != league)
    # Nothing advanced (the scratch repo is a copy of the snapshot), so the
    # rule is: no league advanced and one failed -> no swap, red.
    record(f"plant: {league} cannot stage -> carried, others staged fresh",
           carried and others_fresh, f"carried={carried} others={others_fresh}")
    record(f"plant: {league} failed and nothing advanced -> red, no swap",
           code == D.EXIT_FAILED and outcome == "failed"
           and (volume / POINTER_FILE).read_text(encoding="utf-8").strip() == pointer,
           f"exit {code}, outcome {outcome!r}")
    return out


def case_nba_unit(volume, scratch_repo, pointer):
    """One of the NBA's three tables missing must carry ALL three."""
    relative = D.league_tables("nba")[1]
    path = scratch_repo / relative
    blob = path.read_bytes()
    path.unlink()
    try:
        code, outcome, out = run_refresh(volume, scratch_repo)
    finally:
        path.write_bytes(blob)
    record("plant: one NBA table missing -> the whole NBA unit carried",
           said(out, "nba", "CARRIED from")
           and not said(out, "nba", "staged fresh"),
           f"exit {code}, outcome {outcome!r}")
    return out


def case_every_league_fails(volume, scratch_repo, pointer):
    shutil.move(str(scratch_repo), str(scratch_repo) + ".hidden")
    try:
        code, outcome, out = run_refresh(volume, scratch_repo)
    finally:
        shutil.move(str(scratch_repo) + ".hidden", str(scratch_repo))
    every = all(said(out, league, "CARRIED from") for league in D.LEAGUES)
    record("plant: every league fails -> no swap, red, previous serves",
           every and code == D.EXIT_FAILED
           and (volume / POINTER_FILE).read_text(encoding="utf-8").strip() == pointer,
           f"exit {code}, outcome {outcome!r}, all carried={every}")
    return out


def advance_wnba(scratch_repo) -> bytes:
    """Give the WNBA a later game so exactly one league can advance."""
    import pandas as pd

    relative = D.league_tables("wnba")[0]
    path = scratch_repo / relative
    original = path.read_bytes()
    frame = pd.read_csv(path)
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])
    last = frame["GAME_DATE"].max()
    tail = frame[frame["GAME_DATE"] == last].copy()
    tail["GAME_DATE"] = last + pd.Timedelta(days=7)
    tail["GAME_ID"] = tail["GAME_ID"].astype(str) + "9"
    pd.concat([frame, tail], ignore_index=True).to_csv(path, index=False)
    return original


def case_partial(volume, scratch_repo, pointer, snapshot):
    """One league advances while another is carried -> swap, EXIT_PARTIAL."""
    original = advance_wnba(scratch_repo)
    nfl_tables = D.league_tables("nfl")
    backups = {r: (scratch_repo / r).read_bytes() for r in nfl_tables}
    for relative in nfl_tables:
        (scratch_repo / relative).unlink()

    before_hashes = D.league_hashes(snapshot, "nfl")
    try:
        code, outcome, out = run_refresh(volume, scratch_repo)
    finally:
        (scratch_repo / D.league_tables("wnba")[0]).write_bytes(original)
        for relative, blob in backups.items():
            (scratch_repo / relative).write_bytes(blob)

    moved = (volume / POINTER_FILE).read_text(encoding="utf-8").strip()
    record("plant: one league advances, one carried -> swapped, EXIT_PARTIAL",
           code == D.EXIT_PARTIAL and outcome == "partial" and moved != pointer,
           f"exit {code}, outcome {outcome!r}, pointer {moved}")
    record("plant: the advancing league advanced and the other was carried",
           said(out, "wnba", "staged fresh") and said(out, "nfl", "CARRIED"),
           "per-league outcomes as expected")

    new_snapshot = volume / SNAPSHOTS_DIR / moved
    after_hashes = D.league_hashes(new_snapshot, "nfl")
    record("plant: the carried league's tables are byte-identical",
           after_hashes == before_hashes,
           f"{len(before_hashes)} table(s) compared")

    meta = json.loads((new_snapshot / "SNAPSHOT.json").read_text(
        encoding="utf-8"))
    entry = (meta.get("leagues") or {}).get("nfl") or {}
    record("plant: the manifest records the outcome and carried_from",
           entry.get("outcome") == D.OUTCOME_CARRIED
           and entry.get("carried_from") == pointer
           and bool(entry.get("reason")),
           f"outcome={entry.get('outcome')} from={entry.get('carried_from')}")

    # Put the scratch volume back so later cases see the original pointer.
    (volume / POINTER_FILE).write_text(pointer, encoding="utf-8")
    shutil.rmtree(new_snapshot, ignore_errors=True)
    return out


def case_real_step_refused(volume, scratch_repo):
    """A case that tries to run a real step must fail loudly, not run it."""
    first = D.LEAGUES["nba"]["steps"][0]
    raised = None
    try:
        with redirect_stdout(io.StringIO()):
            D.rebuild_league("nba", timeout=5)
    except RealSubprocessRefused as error:
        raised = str(error)
    except Exception as error:                                  # noqa: BLE001
        raised = f"WRONG TYPE: {type(error).__name__}: {error}"
    record("plant: trying to run a real step is refused, naming the script",
           raised is not None and not raised.startswith("WRONG TYPE")
           and first.name in raised,
           f"-> {first.name} named: "
           f"{bool(raised and first.name in str(raised))}")

    # And the whole-suite install must be the thing doing it, not a local
    # patch: reverting it has to restore the real binding.
    if D.run_step is not refusing_run_step:
        raise PlantDidNothing(
            "the refusing step runner was not installed for the suite")
    record("the refusing runner is installed suite-wide", True,
           "daily_refresh.run_step is the refusing stand-in")


def case_missing_step_script(volume, scratch_repo, pointer):
    """A missing step script is GLOBAL, not a carried league.

    TESTED AGAINST `rebuild_league` DIRECTLY, never through `_run` with
    rebuilds enabled. The first version of this case set
    `args.skip_rebuild = False` so the step list would actually be read - and
    because the leagues run in order, that executed the REAL NBA pipeline
    before ever reaching the WNBA's planted list: it fetched from nba_api and
    rewrote three tracked tables in the working tree. A verifier that mutates
    the repo to test a code path is worse than the bug it is checking for.
    """
    original = D.LEAGUES["wnba"]["steps"]
    D.LEAGUES["wnba"]["steps"] = [Path("does-not-exist-anywhere.py")]
    if D.LEAGUES["wnba"]["steps"] == original:
        raise PlantDidNothing("the step list was not replaced")
    raised = None
    try:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            D.rebuild_league("wnba", timeout=5)
    except D.MissingStepScript as error:
        raised = f"MissingStepScript: {error}"
    except Exception as error:                                  # noqa: BLE001
        raised = f"WRONG TYPE: {type(error).__name__}: {error}"
    finally:
        D.LEAGUES["wnba"]["steps"] = original

    record("plant: a missing step script raises MissingStepScript (global)",
           raised is not None and raised.startswith("MissingStepScript"),
           f"-> {raised}")

    # And it must NOT be caught as a league-local failure: stage_per_league
    # re-raises it rather than carrying the league.
    source = io.open(Path(D.__file__), encoding="utf-8").read()
    body = source[source.index("def stage_per_league"):
                  source.index("def smoke_per_league")]
    record("plant: stage_per_league re-raises it instead of carrying",
           "except MissingStepScript:" in body and "raise" in body,
           "re-raise present")


def case_carried_table_altered(volume, scratch_repo, pointer, snapshot):
    """A carried table changed before the swap must stop the swap."""
    carried = {"wnba": D.league_hashes(snapshot, "wnba")}
    staging = volume / SNAPSHOTS_DIR / "altered.staging"
    if staging.exists():
        shutil.rmtree(staging)
    for relative in D.league_tables("wnba"):
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot / relative, target)
    clean = True
    try:
        D.assert_carried_unchanged(staging, carried, snapshot)
    except Exception:                                           # noqa: BLE001
        clean = False
    record("control: an untouched carried table passes the hash assertion",
           clean, "no exception")

    victim = staging / D.league_tables("wnba")[0]
    victim.write_bytes(victim.read_bytes() + b"\n# altered\n")
    stopped = False
    try:
        D.assert_carried_unchanged(staging, carried, snapshot)
    except Exception as error:                                  # noqa: BLE001
        stopped = "not identical" in str(error)
    shutil.rmtree(staging, ignore_errors=True)
    record("plant: a carried table altered before the swap stops the swap",
           stopped, f"-> {stopped}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", default=str(DEFAULT_SCRATCH))
    args = parser.parse_args(argv)
    volume = Path(args.volume)

    if volume.resolve() == LIVE_VOLUME.resolve():
        print(f"REFUSING: {volume} is the live volume. This verifier builds "
              f"and swaps snapshots; point it at a scratch copy.")
        return 2
    if not (volume / POINTER_FILE).is_file():
        print(f"{volume / POINTER_FILE} does not exist - seed a scratch "
              f"volume by copying a snapshot and writing its name into "
              f"`current`.")
        return 2

    print("PER-LEAGUE ISOLATION")
    print(f"  scratch volume: {volume}")
    # INSTALLED BEFORE ANY CASE RUNS. See refusing_run_step for the incident
    # this prevents.
    real_run_step = D.run_step
    D.run_step = refusing_run_step
    print("  pipeline steps: REFUSED (no real subprocess can start)")
    pointer, snapshot, scratch_repo = prepare(volume)
    print(f"  serving       : {pointer}")
    print(f"  scratch repo  : {scratch_repo}")
    print()

    case_control(volume, scratch_repo, pointer)
    case_one_league_fails(volume, scratch_repo, pointer, snapshot, "nfl")
    case_one_league_fails(volume, scratch_repo, pointer, snapshot, "wnba")
    case_nba_unit(volume, scratch_repo, pointer)
    case_partial(volume, scratch_repo, pointer, snapshot)
    case_every_league_fails(volume, scratch_repo, pointer)
    case_real_step_refused(volume, scratch_repo)
    case_missing_step_script(volume, scratch_repo, pointer)
    case_carried_table_altered(volume, scratch_repo, pointer, snapshot)

    D.run_step = real_run_step
    if D.run_step is refusing_run_step:
        raise PlantDidNothing("the real step runner was not restored")

    shutil.rmtree(scratch_repo, ignore_errors=True)
    passed = sum(1 for _n, ok, _d in RESULTS if ok)
    print()
    print(f"  {passed} of {len(RESULTS)} checks passed")
    print(f"  snapshots on the scratch volume now: {snapshots_in(volume)}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:                                  # noqa: BLE001
        print(f"\nCRASHED before a verdict: {type(error).__name__}: {error}")
        raise SystemExit(2)
