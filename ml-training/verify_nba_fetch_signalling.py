r"""Every NBA acquisition step's failure must reach a non-zero exit.

Until 2026-10-22 four of the NBA's step scripts could not fail the daily
refresh. `validate_games.py` printed `RESULT: FAIL` and exited 0;
`fetch_games.py` caught each season and exited 0; `fetch_player_boxscores.py`
and `fetch_quarter_scores.py` counted failures, wrote them to a log and
returned None. `run_step` discards a non-fetch step's stdout, so the messages
were invisible as well as harmless. The WNBA, G League and NFL were all built
later with `sys.exit(main())` and an explicit failure return; the NBA's
scripts predate that discipline.

This verifier plants each defect the new signalling exists for and checks it
goes red. Nothing here touches the network or the live volume: the
`validate_games` plants run against a scratch raw directory, and the two
fetchers' guards were extracted into pure functions so they can be exercised
without fetching 13,000 games.

Exit codes: 0 every plant behaved, 1 one did not, 2 crashed before a verdict.

    python ml-training/verify_nba_fetch_signalling.py
"""
import importlib
import io
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
for directory in (PROJECT / "data-pipeline" / "preprocessing",
                  PROJECT / "data-pipeline" / "ingestion"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import pandas as pd                                            # noqa: E402

import fetch_games as FG                                        # noqa: E402
import fetch_player_boxscores as BOX                            # noqa: E402
import fetch_quarter_scores as QS                               # noqa: E402
import validate_games as VG                                     # noqa: E402

RAW = PROJECT / "data-pipeline" / "data" / "raw"
RESULTS = []


class PlantDidNothing(RuntimeError):
    """A plant that changes no behaviour is a harness bug, not a pass."""


def record(name, passed, detail):
    RESULTS.append((name, passed, detail))
    print("  %-4s %-54s %s" % ("PASS" if passed else "FAIL", name, detail))


def quiet(function, *args, **kwargs):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        value = function(*args, **kwargs)
    return value, buffer.getvalue()


# --------------------------------------------------- validate_games.py
def with_scratch_raw(seasons_present, corrupt=None):
    """Run validate_games.main() against a scratch copy of the raw corpus."""
    with tempfile.TemporaryDirectory() as temporary:
        scratch = Path(temporary)
        for season in seasons_present:
            source = RAW / f"games_{season}.csv"
            if not source.is_file():
                continue
            target = scratch / source.name
            shutil.copy2(source, target)
            if corrupt == season:
                frame = pd.read_csv(target)
                # A real validator failure: blank out a critical column.
                frame.loc[frame.index[:5], "WL"] = None
                frame.to_csv(target, index=False)
        original = VG.RAW_DATA_DIR
        VG.RAW_DATA_DIR = scratch
        try:
            return quiet(VG.main)
        finally:
            VG.RAW_DATA_DIR = original


def check_validate_games():
    print("\n  validate_games.py")
    scope = FG.seasons_through()
    on_disk = [s for s in scope if (RAW / f"games_{s}.csv").is_file()]

    # CONTROL: every season in scope present and clean -> 0. This checkout is
    # missing the current season, so the control supplies the scope itself.
    original_scope = VG.seasons_in_scope
    VG.seasons_in_scope = lambda: list(on_disk)
    try:
        code, _out = with_scratch_raw(on_disk)
        record("control: all in-scope seasons present and clean",
               code == 0, f"exit {code}, expected 0")

        code, out = with_scratch_raw(on_disk, corrupt=on_disk[-1])
        record("plant: one season fails validation",
               code == 1 and "Failed seasons" in out,
               f"exit {code}, expected 1")

        code, out = with_scratch_raw(on_disk[:-1])
        record("plant: a season in scope has no file",
               code == 1 and "in scope but no file" in out,
               f"exit {code}, expected 1")

        code, out = with_scratch_raw([])
        record("plant: no season files at all",
               code == 1 and "no games_" in out,
               f"exit {code}, expected 1")
    finally:
        VG.seasons_in_scope = original_scope

    # The scope itself failing to derive must refuse rather than guess.
    def boom():
        raise RuntimeError("simulated scope failure")
    VG.seasons_in_scope = boom
    try:
        code, out = with_scratch_raw(on_disk)
        record("plant: the seasons-in-scope list cannot be derived",
               code == 1 and "cannot derive" in out,
               f"exit {code}, expected 1")
    finally:
        VG.seasons_in_scope = original_scope


# ------------------------------------------------------ the schedule gate
def check_schedule_gate():
    print("\n  fetch_games.py - the schedule gate on an empty fetch")
    started, why = FG.season_not_started("2026-27")
    record("plant: schedule shows games played, fetch returned none",
           started is False and "STARTED" in why, f"-> {started}")

    future, why = FG.season_not_started("2027-28")
    record("control: unpublished season, nothing scheduled yet",
           future is True, f"-> {future}")

    before, why = FG.season_not_started(
        "2026-27", today=pd.Timestamp("2026-10-01"))
    record("control: before the first game, empty fetch is legitimate",
           before is True and "still ahead" in why, f"-> {before}")

    real = FG.first_regular_season_game

    def unreadable(season):
        raise ConnectionError("simulated network failure")
    FG.first_regular_season_game = unreadable
    try:
        unknown, why = FG.season_not_started("2026-27")
        record("plant: the schedule cannot be read at all",
               unknown is False and "could not be read" in why,
               f"-> {unknown}")
    finally:
        FG.first_regular_season_game = real

    # NO-OP GUARD: putting the real function back must restore the verdict.
    if FG.first_regular_season_game is not real:
        raise PlantDidNothing("the schedule plant was not reverted")
    again, _why = FG.season_not_started("2027-28")
    record("no-op guard: reverting the plant restores the verdict",
           again is True, f"-> {again}")


# ------------------------------------------- the known-unfetchable list
def check_known_list():
    print("\n  fetch_quarter_scores.py - KNOWN_UNFETCHABLE_QUARTER_SCORES")
    corpus = QS.load_game_ids()
    problems, _out = quiet(QS.check_known_list, corpus)
    record("control: the authored list is sound and evidenced",
           not problems, f"{len(problems)} problem(s)")

    original = dict(QS.KNOWN_UNFETCHABLE_QUARTER_SCORES)
    try:
        QS.KNOWN_UNFETCHABLE_QUARTER_SCORES["0029900001"] = {
            "date": "1999-01-01", "teams": "invented",
            "errors": ("invented",), "failed_attempts": 1}
        if QS.KNOWN_UNFETCHABLE_QUARTER_SCORES == original:
            raise PlantDidNothing("the list was not modified")
        problems, _out = quiet(QS.check_known_list, corpus)
        record("plant: a game listed with no logged failure",
               any("no evidence" in p or "nowhere in" in p for p in problems),
               f"{len(problems)} problem(s)")

        QS.KNOWN_UNFETCHABLE_QUARTER_SCORES.clear()
        QS.KNOWN_UNFETCHABLE_QUARTER_SCORES.update(original)
        first = sorted(original)[0]
        QS.KNOWN_UNFETCHABLE_QUARTER_SCORES[first] = dict(
            original[first], failed_attempts=None)
        problems, _out = quiet(QS.check_known_list, corpus)
        record("plant: an entry missing its attempt count",
               any("without failed_attempts" in p for p in problems),
               f"{len(problems)} problem(s)")

        QS.KNOWN_UNFETCHABLE_QUARTER_SCORES.clear()
        QS.KNOWN_UNFETCHABLE_QUARTER_SCORES.update(original)
        problems, _out = quiet(QS.check_known_list, [])
        record("plant: a listed game that is not in the corpus",
               any("not in" in p for p in problems),
               f"{len(problems)} problem(s)")
    finally:
        QS.KNOWN_UNFETCHABLE_QUARTER_SCORES.clear()
        QS.KNOWN_UNFETCHABLE_QUARTER_SCORES.update(original)

    problems, _out = quiet(QS.check_known_list, corpus)
    record("after reverting: the list is sound again", not problems,
           f"{len(problems)} problem(s)")


# ------------------------------------------------------ the run verdict
def check_run_verdict():
    print("\n  the post-run verdict, both fetchers")
    listed = sorted(QS.KNOWN_UNFETCHABLE_QUARTER_SCORES)

    code, lines = QS.post_run_verdict(listed, present=lambda gid: False)
    record("control: only known-unfetchable games failed",
           code == 0, f"exit {code}, expected 0")

    code, lines = QS.post_run_verdict(listed + ["0022600099"],
                                      present=lambda gid: False)
    record("plant: one unlisted game failed this run",
           code == 1 and any("NOT on" in line for line in lines),
           f"exit {code}, expected 1")

    code, lines = QS.post_run_verdict([], present=lambda gid: True)
    record("plant: a listed game now fetches (reverse guard)",
           code == 1 and any("NOW FETCH" in line for line in lines),
           f"exit {code}, expected 1")

    code, lines = QS.post_run_verdict([], present=lambda gid: False)
    record("control: nothing failed and nothing recovered",
           code == 0, f"exit {code}, expected 0")

    code, lines = BOX.post_run_verdict([])
    record("control: boxscores, nothing failed", code == 0,
           f"exit {code}, expected 0")

    code, lines = BOX.post_run_verdict(["0022600099"])
    record("plant: boxscores, one game failed (list is empty by design)",
           code == 1, f"exit {code}, expected 1")


def main() -> int:
    print("NBA ACQUISITION FAILURE SIGNALLING")
    print(f"  raw corpus: {RAW}")
    check_validate_games()
    check_schedule_gate()
    check_known_list()
    check_run_verdict()

    passed = sum(1 for _n, ok, _d in RESULTS if ok)
    print()
    print(f"  {passed} of {len(RESULTS)} checks passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:                                  # noqa: BLE001
        print(f"\nCRASHED before a verdict: {type(error).__name__}: {error}")
        raise SystemExit(2)
