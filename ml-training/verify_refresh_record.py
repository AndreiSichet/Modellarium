r"""The refresh records that it ran, and the summary renders as markdown.

TWO THINGS NOTHING ELSE CHECKS.

1. WHETHER THE JOB IS STILL RUNNING. Every freshness field in /health
   describes the DATA, and all of them read identically whether the refresh
   ran this morning and found nothing or stopped being scheduled three weeks
   ago. Between seasons that silence lasts months. So the volume carries a
   record of the last verdict, written on EVERY terminal outcome including the
   green quiet one, and /health reads the state of it.

2. WHETHER THE SUMMARY IS A TABLE OR A ROW OF PIPE CHARACTERS. GitHub renders
   a markdown table only when a blank line separates it from the text above.
   The availability rows were emitted after the fetch-counter heading, with no
   header row and no blank line, and showed as literal text - the one row that
   says the NBA is serving on 34 of 38 features. A misleading green: the step
   was green, the fact was there, and nobody could read it.

READ-ONLY with respect to the served volume. Every volume it touches is built
under --work.

    python ml-training/verify_refresh_record.py
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
WORKFLOW = PROJECT / ".github" / "workflows" / "daily-refresh.yml"

sys.path.insert(0, str(HERE))
import daily_refresh as D  # noqa: E402
import served_data as S  # noqa: E402

NEWLINE = chr(10)

RESULTS = []


class PlantDidNothing(RuntimeError):
    """A plant that changes no behaviour is a harness bug, not a pass."""


def record(name, passed, detail):
    RESULTS.append((name, passed, detail))
    print("  %-4s %-58s %s" % ("PASS" if passed else "FAIL", name, detail))


# ==========================================================================
# 1. the record, and the states read off it
# ==========================================================================

def fake_volume(work: Path, name: str) -> Path:
    """A directory that `resolve_root` recognises as a volume.

    A volume is "a root with a `current` pointer", so that is what is built -
    not a mock. read_last_refresh branches on the pointer's existence to tell
    a volume from a plain data root, and a stub that skipped the pointer would
    be testing a different function than the one that ships.
    """
    volume = work / name
    snapshot = volume / S.SNAPSHOTS_DIR / "2026-10-22T0711"
    snapshot.mkdir(parents=True, exist_ok=True)
    (volume / S.POINTER_FILE).write_text("2026-10-22T0711", encoding="utf-8")
    return volume


def states(work: Path) -> None:
    volume = fake_volume(work, "vol-states")
    now = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)

    record("a volume with no record reads UNKNOWN, not ok",
           S.read_last_refresh(volume, now)["state"] == "unknown",
           S.read_last_refresh(volume, now)["state"])

    # NOT the same answer, and the distinction is the point: CI serves a plain
    # data root where no refresh job was ever meant to write.
    plain = work / "plain-root"
    plain.mkdir(parents=True, exist_ok=True)
    record("a plain data root reads NOT_A_VOLUME, a different answer",
           S.read_last_refresh(plain, now)["state"] == "not_a_volume",
           S.read_last_refresh(plain, now)["state"])

    # THE THRESHOLDS, EXERCISED FROM THE CONSTANTS rather than from 26 and 40
    # written again here - a literal would stop tracking the constant the
    # moment someone moved it, which is the whole failure mode of a frozen
    # number with no date beside it.
    late = S.REFRESH_LATE_HOURS
    overdue = S.REFRESH_OVERDUE_HOURS
    for hours, expected in ((0.5, "ok"), (late - 1, "ok"), (late, "late"),
                            (overdue - 1, "late"), (overdue, "overdue"),
                            (overdue + 48, "overdue")):
        S.write_last_refresh(volume, "nothing-new", "2026-10-22T0711", {})
        body = json.loads(
            (volume / S.LAST_REFRESH_FILE).read_text(encoding="utf-8"))
        body["finished_at"] = (now - timedelta(hours=hours)).isoformat()
        (volume / S.LAST_REFRESH_FILE).write_text(
            json.dumps(body), encoding="utf-8")
        got = S.read_last_refresh(volume, now)
        record(f"{hours:g}h since the last verdict reads {expected.upper()}",
               got["state"] == expected,
               f"state={got['state']}, hours_since={got['hours_since']}")

    # A FAILING JOB IS NOT A FRESH ONE. A refresh that runs and fails every
    # morning has a current timestamp; collapsing the two into one "fresh"
    # boolean would report it as healthy.
    for outcome in ("failed", "partial", "drift-report-failed"):
        S.write_last_refresh(volume, outcome, "2026-10-22T0711", {})
        got = S.read_last_refresh(volume, now)
        record(f"a recent {outcome!r} verdict reads FAILED, not ok",
               got["state"] == "failed", f"state={got['state']}")

    # A record whose timestamp is garbage must not be read as a time.
    (volume / S.LAST_REFRESH_FILE).write_text(
        json.dumps({"finished_at": "not a timestamp", "outcome": "advanced"}),
        encoding="utf-8")
    record("an unparseable record reads UNKNOWN rather than crashing",
           S.read_last_refresh(volume, now)["state"] == "unknown",
           S.read_last_refresh(volume, now)["state"])

    S.write_last_refresh(volume, "advanced", "2026-10-22T1200",
                         {"nba": {"outcome": "advanced"},
                          "nfl": {"outcome": "carried"}})
    got = S.read_last_refresh(volume, now)
    record("the record carries the per-league outcomes",
           got["leagues"] == {"nba": "advanced", "nfl": "carried"},
           str(got["leagues"]))


def every_terminal_path_records(work: Path) -> None:
    """Every outcome the refresh can reach must leave a mark.

    READ OUT OF THE SOURCE, not out of a list written here. The point of the
    check is that no terminal path was missed, and a hand-maintained list of
    paths is exactly the thing that goes stale when a fifth outcome is added -
    it would then agree with itself and prove nothing.
    """
    source = (HERE / "daily_refresh.py").read_text(encoding="utf-8")
    calls = set(re.findall(r'record_outcome\(\s*"([a-z-]+)"', source))
    calls |= {"advanced", "forced"} if "advanced\" if advanced" in source \
        else set()
    record("every outcome in the source goes through record_outcome",
           len(calls) >= 5,
           f"{len(calls)} distinct: {sorted(calls)}")

    body = re.search(r"def record_outcome\(.*?\n(?=\n    print\(__doc__)",
                     source, re.S)
    record("record_outcome is the one place the record is written",
           body is not None and "write_last_refresh(" in body.group(0)
           and source.count("write_last_refresh(") == 1,
           f"{source.count('write_last_refresh(')} call site(s), "
           f"inside record_outcome: "
           f"{body is not None and 'write_last_refresh(' in body.group(0)}")

    # AND IT CANNOT FAIL THE RUN. A read-only volume must not turn a
    # successful refresh into a failed one - the instrument must not break the
    # thing it measures.
    record("the write is guarded so it cannot fail the run",
           body is not None and "except OSError" in body.group(0),
           "wrapped in try/except OSError, warning loudly")

    # The quiet green outcome is the one that matters most, so it is named
    # rather than left to the count above.
    record("`nothing-new` records too - the quiet run is the point",
           re.search(r'record_outcome\("nothing-new",\s*previous',
                     source) is not None,
           "records the PREVIOUS snapshot, since nothing was swapped")


# ==========================================================================
# 2. the summary's markdown structure
# ==========================================================================

EMIT = re.compile(r'^\s*(.*?)\s*\|\s*Out-File \$env:GITHUB_STEP_SUMMARY')


def emitted_lines(text: str) -> list:
    """The sequence of things appended to the step summary, in order.

    CONDITIONALS ARE TREATED AS ALWAYS TAKEN, which is the right
    over-approximation here: the step is a linear sequence of appends, and a
    branch that emits a table row has to be correct whether or not its
    neighbour ran. A literal becomes its text; anything else - a piped
    Get-Content, a variable - becomes OPAQUE, which breaks table continuity,
    because a table cannot be assumed to continue across content this check
    cannot see.
    """
    out = []
    for line in text.splitlines():
        match = EMIT.match(line)
        if not match:
            continue
        payload = match.group(1).strip()
        if payload.startswith('"') and payload.endswith('"'):
            out.append(payload[1:-1])
        else:
            out.append(None)
    return out


def table_rows_are_in_tables(lines) -> list:
    """Every `|` row must sit in a table that opened correctly.

    A table opens with a header row preceded by a blank line, and its second
    row must be the `|---|` separator. Anything else and GitHub prints the
    pipes.
    """
    problems = []
    previous = ""
    in_table = False
    index_in_table = 0
    for index, line in enumerate(lines):
        if line is None:
            in_table = False
            previous = None
            continue
        stripped = line.strip()
        if stripped.startswith("|"):
            if not in_table:
                if previous != "":
                    problems.append(
                        (index, f"table opens after {previous!r:.60} with no "
                                f"blank line"))
                in_table = True
                index_in_table = 0
            else:
                index_in_table += 1
                if index_in_table == 1 and not re.fullmatch(
                        r"\|(\s*:?-+:?\s*\|)+", stripped):
                    problems.append(
                        (index, f"second row is not a |---| separator: "
                                f"{stripped!r:.60}"))
        else:
            in_table = False
        previous = stripped
    return problems


def summary_structure(workflow_text: str = None) -> None:
    text = workflow_text if workflow_text is not None \
        else WORKFLOW.read_text(encoding="utf-8")
    lines = emitted_lines(text)
    record("the summary step emits lines this check can read",
           len(lines) > 20, f"{len(lines)} appended line(s)")

    problems = table_rows_are_in_tables(lines)
    record("every table row sits in a table that opened with a blank line",
           not problems,
           "clean" if not problems else f"{len(problems)}: {problems[:3]}")

    rows = [line for line in lines if line and "availability |" in line]
    record("the availability rows are emitted (they are the point of the fix)",
           bool(rows), f"{len(rows)} row(s)")

    record("the per-league table is rendered into the summary",
           "refresh-summary.md" in text,
           "refresh-summary.md is read into GITHUB_STEP_SUMMARY")
    record("the refresh state reaches the summary",
           "refresh state |" in text and "$health.refresh" in text,
           "a `refresh state` row, from /health")
    record("the quarantine list reaches the summary",
           "$health.nfl.quarantine" in text
           and "games set aside" in text.lower(),
           "its own table, with its own blank line and header")


def plant_summary_structure() -> None:
    """Re-introduce the orphaned rows and the check must go red.

    THE DEFECT THIS WAS WRITTEN FOR, put back: availability rows appended
    after the fetch-counter heading, with no header and no blank line.
    """
    text = WORKFLOW.read_text(encoding="utf-8")
    anchor = ('            foreach ($lg in $leagues) {\n'
              '              "| $($lg.ToUpper()) availability |')
    if anchor not in text:
        raise PlantDidNothing(
            "the availability loop is not where the plant expects it; the "
            "plant would have tested nothing")
    # Move the loop to the end, after the heading-and-bullets section, which
    # is exactly where it used to be.
    planted = text.replace(
        '            $avail = $health.availability.nba.state\n',
        '            $avail = $health.availability.nba.state\n'
        '            foreach ($lg in $leagues) {\n'
        '              "| $($lg.ToUpper()) availability | '
        '$($health.availability.$lg.state) |" | Out-File '
        '$env:GITHUB_STEP_SUMMARY -Append -Encoding utf8\n'
        '            }\n', 1)
    if planted == text:
        raise PlantDidNothing("the plant changed nothing")

    problems = table_rows_are_in_tables(emitted_lines(planted))
    record("PLANT: the orphaned availability rows are caught",
           bool(problems),
           f"{len(problems)} problem(s): {problems[0][1] if problems else '-'}"
           if problems else "NOT CAUGHT - the check is asleep")

    # And the guard against the guard.
    noop = text.replace("### Fetch request counters",
                        "### Fetch request counters", 1)
    record("PLANT (no-op): an unchanged workflow is still clean",
           not table_rows_are_in_tables(emitted_lines(noop)),
           "green, so the red above was the plant and not the checker")


def plant_missing_blank_line(work: Path) -> None:
    """write_summary's leading blank line is what makes its table a table."""
    import daily_refresh as D

    path = work / "summary.md"
    args = argparse.Namespace(summary_file=str(path))
    path.write_text("Some prose about the run.\n", encoding="utf-8")
    D.write_summary(args, ["| league | outcome |", "|---|---|",
                           "| nba | advanced |"])
    good = path.read_text(encoding="utf-8").splitlines()
    record("write_summary separates its table from the text above",
           not table_rows_are_in_tables(good),
           f"{len(good)} line(s), table opens after a blank line")

    # The same content WITHOUT the blank line must be caught, or the check
    # above passed for no reason.
    bad = ["Some prose about the run.", "| league | outcome |", "|---|---|",
           "| nba | advanced |"]
    record("PLANT: the same table with no blank line is caught",
           bool(table_rows_are_in_tables(bad)),
           str(table_rows_are_in_tables(bad)[:1]))


# ==========================================================================
# 3. the summary holds THIS run only, and says how long since the last one
# ==========================================================================

def _args(summary_file=None):
    """The one attribute start_summary and write_summary actually read."""
    return argparse.Namespace(summary_file=summary_file)


def summary_holds_one_run(work: Path) -> None:
    """Two runs in a row must leave one table, not two.

    THE DEFECT WAS AN ABSENT TRUNCATION, NOT A BAD APPEND. write_summary has
    always opened the file with mode "a", which is right - it is called
    several times per run and each call adds a section. What was missing was
    anything emptying the file BETWEEN runs, and the runner's checkout uses
    `clean: false` so the file survives. Measured on the runner 2026-10-29:
    ELEVEN accumulated per-league tables in 3,016 bytes.
    """
    path = work / "summary-two-runs.md"

    for run in ("first", "second"):
        D.start_summary(_args(path))
        D.write_summary(_args(path), ["### Per league", "",
                                      "| league | outcome |", "|---|---|",
                                      f"| nba | {run} |"])

    text = path.read_text(encoding="utf-8")
    record("two consecutive runs leave ONE per-league table",
           text.count("### Per league") == 1,
           f"{text.count('### Per league')} table(s) in the file")
    record("and it is the SECOND run's, not the first's",
           "second" in text and "first" not in text,
           "only the latest run's rows survive")

    # An absent --summary-file must be a no-op rather than a crash: the local
    # and rehearsal invocations pass none.
    D.start_summary(_args(None))
    D.write_summary(_args(None), ["anything"])
    record("no --summary-file is a no-op, not a crash", True,
           "start_summary and write_summary both return early")

    # An unwritable path must warn, not fail the refresh.
    try:
        D.start_summary(_args(work / "no-such-dir" / "s.md"))
        survived = True
    except Exception:                                           # noqa: BLE001
        survived = False
    record("an unwritable summary path warns and does not raise",
           survived, "presentation must not fail the swap")


def plant_summary_truncation(work: Path) -> None:
    """Put the absent truncation back, and the one-table check goes red."""
    path = work / "summary-plant.md"
    real = D.start_summary

    def appending_start(args):
        """The defect exactly as it shipped: nothing empties the file."""
        return None

    D.start_summary = appending_start
    try:
        for run in ("first", "second"):
            D.start_summary(_args(path))
            D.write_summary(_args(path), ["### Per league", "",
                                          f"| nba | {run} |"])
        tables = path.read_text(encoding="utf-8").count("### Per league")
    finally:
        D.start_summary = real

    if tables == 1:
        raise PlantDidNothing(
            "the plant left one table, so the check cannot be reading the "
            "truncation")
    record("PLANT no truncation -> two tables accumulate (red)",
           tables == 2, f"{tables} tables with start_summary neutered")


def hours_since_line(work: Path) -> None:
    """The line spec 2 asked for, and the warning threshold.

    READ BEFORE THE RUN OVERWRITES IT. record_outcome rewrites
    last_refresh.json on every terminal path, so this is only answerable at
    startup - which is where previous_run_lines is called from.
    """
    volume = fake_volume(work, "vol-hours")
    late = S.REFRESH_LATE_HOURS

    def at(hours, outcome="advanced"):
        S.write_last_refresh(volume, outcome, "2026-10-22T0711", {})
        body = json.loads(
            (volume / S.LAST_REFRESH_FILE).read_text(encoding="utf-8"))
        body["finished_at"] = (
            datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        (volume / S.LAST_REFRESH_FILE).write_text(
            json.dumps(body), encoding="utf-8")
        return NEWLINE.join(D.previous_run_lines(volume))

    # NO RECORD AT ALL. An unknown, never a zero - the conflation this
    # project has four recorded bugs from.
    missing = NEWLINE.join(D.previous_run_lines(volume))
    record("no previous record says so in words, and does not crash",
           "No previous run recorded" in missing and "unknown" in missing,
           missing.strip()[:80])
    record("and it does NOT print a 0.0 h gap",
           "0.0 h" not in missing, "an absent record is not a fresh one")

    under = at(late - 1)
    record(f"{late - 1}h prints the line and no warning",
           f"{late - 1:.1f} h" in under and "Late:" not in under,
           under.strip()[:80])

    over = at(late + 1)
    record(f"{late + 1}h prints the line AND the late warning",
           f"{late + 1:.1f} h" in over and "Late:" in over
           and str(late) in over,
           over.strip()[:80])

    # EXACTLY AT THE THRESHOLD warns, matching read_last_refresh's own `>=`.
    record(f"exactly {late}h warns, like /health's own boundary",
           "Late:" in at(late), f"{late}h is late, not ok")

    # A PREVIOUS RUN THAT FAILED, 27h ago. read_last_refresh returns `failed`
    # in preference to `late`, short-circuiting before the hours are looked
    # at - so a warning keyed on `state` would print nothing here.
    both = at(late + 1, outcome="failed")
    record("a failed run 27h ago prints BOTH warnings, not just one",
           "Late:" in both and "ended `failed`" in both,
           "the hours and the outcome are separate facts")

    # A plain data root is a different answer from a volume with no record.
    plain = work / "plain-for-hours"
    plain.mkdir(parents=True, exist_ok=True)
    text = NEWLINE.join(D.previous_run_lines(plain))
    record("a plain data root says NOT a volume, a different answer",
           "plain data root" in text, text.strip()[:80])


def plant_hours_threshold(work: Path) -> None:
    """Key the warning on `state` instead of the hours, and lose a case."""
    volume = fake_volume(work, "vol-hours-plant")
    late = S.REFRESH_LATE_HOURS
    S.write_last_refresh(volume, "failed", "2026-10-22T0711", {})
    body = json.loads(
        (volume / S.LAST_REFRESH_FILE).read_text(encoding="utf-8"))
    body["finished_at"] = (datetime.now(timezone.utc)
                           - timedelta(hours=late + 1)).isoformat()
    (volume / S.LAST_REFRESH_FILE).write_text(
        json.dumps(body), encoding="utf-8")

    real = D.read_last_refresh

    def state_only(vol, now=None):
        """As if the warning were keyed on `state`, which hides this case."""
        got = real(vol, now)
        got["hours_since"] = 0.0           # what keying on `state` amounts to
        return got

    D.read_last_refresh = state_only
    try:
        text = NEWLINE.join(D.previous_run_lines(volume))
    finally:
        D.read_last_refresh = real

    if "Late:" in text:
        raise PlantDidNothing("the plant still warned, so the check is not "
                              "reading the hours")
    record("PLANT warning keyed on state -> a late 'failed' run is silent",
           "Late:" not in text and "ended `failed`" in text,
           "the outcome is reported and the 27h gap is not")


# ==========================================================================
# 4. orphaned staging directories are swept, and only those
# ==========================================================================

def staging_sweep(work: Path) -> None:
    """Old orphans go; everything else stays.

    SEVEN ASSERTIONS AND FIVE OF THEM ARE ABOUT WHAT MUST NOT BE REMOVED. A
    sweep that only proves it deletes something is the dangerous half of the
    test: this runs inside the refresh lock, on the volume the app serves.
    """
    volume = fake_volume(work, "vol-sweep")
    snapshots = volume / S.SNAPSHOTS_DIR
    old = time.time() - 48 * 3600
    fresh = time.time() - 10 * 60

    def make(name, mtime):
        path = snapshots / name / "processed"
        path.mkdir(parents=True, exist_ok=True)
        (path / "games_final.csv").write_text("x" * 2048, encoding="utf-8")
        os.utime(snapshots / name, (mtime, mtime))
        return snapshots / name

    stale = make("2026-10-24T1331.staging", old)
    young = make("2026-10-29T1800.staging", fresh)
    completed = make("2026-10-22T0711", old)          # the pointer's target
    looks_like = make("2026-10-20T0900", old)         # a completed snapshot
    mine = make("2026-10-29T1900.staging", old)       # THIS run's

    removed = D.sweep_staging(volume, "2026-10-29T1900")
    names = [name for name, _size, _age in removed]

    record("a 2-day-old orphan is removed",
           not stale.exists() and "2026-10-24T1331.staging" in names,
           f"swept {names}")
    record("and its size is reported, not just its name",
           bool(removed) and removed[0][1] > 2000,
           f"{removed[0][1] if removed else 0} bytes")
    record("a 10-minute-old staging directory SURVIVES",
           young.exists(), "a run may still be writing into it")
    record("the pointer's target survives",
           completed.exists(), "never delete what is being served")
    record("a completed snapshot survives",
           looks_like.exists(), "only *.staging is in scope")
    record("THIS run's staging directory survives even when old",
           mine.exists(), "excluded by identifier, not only by age")
    record("exactly one directory was removed",
           len(removed) == 1, f"{len(removed)} removed")

    # A volume with no staging directories at all must be a no-op.
    bare = work / "bare-volume"
    (bare / S.SNAPSHOTS_DIR).mkdir(parents=True, exist_ok=True)
    (bare / S.POINTER_FILE).write_text("nothing", encoding="utf-8")
    record("a volume with no staging directories is a no-op",
           D.sweep_staging(bare, "x") == [], "returns an empty list")


def plant_staging_sweep(work: Path) -> None:
    """Two plants: sweep too much, and sweep what is being served."""
    # (a) DROP THE AGE FILTER. The young directory must then be removed, and
    #     the survival assertion above is what would go red.
    volume = fake_volume(work, "vol-sweep-plant-a")
    young = volume / S.SNAPSHOTS_DIR / "2026-10-29T1800.staging"
    young.mkdir(parents=True, exist_ok=True)
    mtime = time.time() - 10 * 60
    os.utime(young, (mtime, mtime))

    removed = D.sweep_staging(volume, "other", max_age_hours=0)
    if young.exists():
        raise PlantDidNothing("a 0-hour age limit left the young directory "
                              "in place, so the age filter is not what the "
                              "check is reading")
    record("PLANT age limit 0 -> a 10-minute-old staging dir is swept (red)",
           not young.exists() and len(removed) == 1,
           "the survival check reads the age filter")

    # (b) POINT THE POINTER AT A STAGING DIRECTORY, with the age filter wide
    #     open. It must survive on the pointer check alone.
    volume = fake_volume(work, "vol-sweep-plant-b")
    served = volume / S.SNAPSHOTS_DIR / "2026-10-24T1331.staging"
    served.mkdir(parents=True, exist_ok=True)
    old = time.time() - 48 * 3600
    os.utime(served, (old, old))
    (volume / S.POINTER_FILE).write_text("2026-10-24T1331.staging",
                                         encoding="utf-8")
    removed = D.sweep_staging(volume, "other", max_age_hours=0)
    record("PLANT the pointer names a staging dir -> it survives anyway",
           served.exists() and removed == [],
           "the pointer check is independent of the age filter")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path)
    args = parser.parse_args()
    work = args.work or Path(tempfile.mkdtemp(prefix="refresh-record-"))
    work.mkdir(parents=True, exist_ok=True)

    print("THE REFRESH'S OWN RECORD, AND THE SUMMARY'S MARKDOWN")
    print(f"  work     : {work}")
    print(f"  late/overdue thresholds: {S.REFRESH_LATE_HOURS}h / "
          f"{S.REFRESH_OVERDUE_HOURS}h")
    print()

    states(work)
    print()
    every_terminal_path_records(work)
    print()
    summary_structure()
    print()
    plant_summary_structure()
    plant_missing_blank_line(work)
    print()
    summary_holds_one_run(work)
    plant_summary_truncation(work)
    print()
    hours_since_line(work)
    plant_hours_threshold(work)
    print()
    staging_sweep(work)
    plant_staging_sweep(work)

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
    except Exception as error:                                  # noqa: BLE001
        print(f"\nCRASHED before a verdict: {type(error).__name__}: {error}")
        raise SystemExit(2)
