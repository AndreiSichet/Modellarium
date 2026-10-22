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
import re
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
WORKFLOW = PROJECT / ".github" / "workflows" / "daily-refresh.yml"

sys.path.insert(0, str(HERE))
import served_data as S  # noqa: E402

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
