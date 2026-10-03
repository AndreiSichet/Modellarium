"""Negative-test the WNBA validator. A guard never seen to fail is unverified.

Corrupts COPIES in a temp directory and repoints the validator's RAW_DIR at
them - the real season files are never touched, the same discipline the
quarter/half failures-log harness uses.
"""

import importlib
import io
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PIPELINE = HERE.parents[1]
REAL_RAW = PIPELINE / "data" / "wnba" / "raw"

TARGET_SEASON = "2023"
results = []


def run_validator(raw_dir: Path):
    """Run the validator against raw_dir, returning (exit_code, output)."""
    import validate_wnba_games as v
    importlib.reload(v)
    v.RAW_DIR = raw_dir
    v.failures.clear()
    v.notes.clear()
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = v.main()
    return code, buffer.getvalue(), list(v.failures)


def staged(corrupt=None) -> Path:
    """A temp copy of every season file, optionally with one corrupted."""
    tmp = Path(tempfile.mkdtemp(prefix="wnba_validator_"))
    for src in sorted(REAL_RAW.glob("wnba_games_*.csv")):
        shutil.copy2(src, tmp / src.name)
    if corrupt:
        path = tmp / f"wnba_games_{TARGET_SEASON}.csv"
        frame = pd.read_csv(path, dtype={"GAME_ID": str})
        frame = corrupt(frame)
        frame.to_csv(path, index=False, encoding="utf-8")
    return tmp


def case(name, corrupt, expect_failure, expect_text=None):
    tmp = staged(corrupt)
    try:
        code, output, failures = run_validator(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    failed = code != 0
    ok = failed == expect_failure
    if ok and expect_text:
        ok = any(expect_text in f for f in failures)

    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"           exit {code}, {len(failures)} failed check(s)")
    for f in failures[:3]:
        print(f"           - {f}")
    if not ok:
        print(f"           EXPECTED {'failure' if expect_failure else 'pass'}"
              + (f" mentioning {expect_text!r}" if expect_text else ""))
    results.append(ok)
    return ok


def drop_one_team_row(frame):
    """Remove one of a game's two rows - an unpaired game."""
    victim = frame["GAME_ID"].iloc[0]
    idx = frame.index[frame["GAME_ID"] == victim][0]
    return frame.drop(index=idx)


def swap_one_games_scores(frame):
    """Give the win to the team that scored fewer points."""
    victim = frame["GAME_ID"].iloc[0]
    rows = frame.index[frame["GAME_ID"] == victim]
    a, b = rows[0], rows[1]
    pts_a, pts_b = frame.at[a, "PTS"], frame.at[b, "PTS"]
    frame.at[a, "PTS"], frame.at[b, "PTS"] = pts_b, pts_a
    return frame


def null_a_required_field(frame):
    frame.at[frame.index[5], "PTS"] = None
    return frame


def break_home_away(frame):
    """Make both rows of a game say 'vs.' - two home teams."""
    victim = frame["GAME_ID"].iloc[0]
    rows = frame.index[frame["GAME_ID"] == victim]
    for i in rows:
        frame.at[i, "MATCHUP"] = frame.at[rows[0], "MATCHUP"]
    return frame


def unexplained_shortfall(frame):
    """Remove games so the schedule is non-uniform in a way that does NOT
    reconcile as cancellations - three teams short by differing amounts."""
    victims = []
    for team in frame["TEAM_ID"].unique()[:3]:
        team_games = frame.loc[frame["TEAM_ID"] == team, "GAME_ID"].unique()
        victims.extend(team_games[:2])
    # Drop ONE row of each, so the games become unpaired AND teams uneven.
    drop_idx = []
    for gid in set(victims):
        drop_idx.append(frame.index[frame["GAME_ID"] == gid][0])
    return frame.drop(index=drop_idx)


def main():
    print(__doc__)
    print(f"Real files stay untouched: {REAL_RAW}\n")

    print("0. CONTROL - unmodified copies must PASS")
    case("unmodified copy of all 12 seasons", None, expect_failure=False)

    print("\n1. THE TWO CORRUPTIONS THE SPEC NAMES")
    case("one team-row dropped from a game", drop_one_team_row,
         expect_failure=True, expect_text="exactly 2 rows")
    case("one game's scores swapped (winner now has fewer points)",
         swap_one_games_scores, expect_failure=True,
         expect_text="WL agrees with PTS")

    print("\n2. THE OTHER FATAL CHECKS, ALSO EXERCISED")
    case("a null in a required field", null_a_required_field,
         expect_failure=True, expect_text="no nulls")
    case("two home teams in one game", break_home_away,
         expect_failure=True, expect_text="one home team")
    case("a shortfall that does NOT reconcile as cancellations",
         unexplained_shortfall, expect_failure=True)

    print(f"\n{'=' * 78}")
    print(f"  {sum(results)} of {len(results)} negative tests behaved correctly")
    print("=" * 78)

    still_there = len(sorted(REAL_RAW.glob("wnba_games_*.csv")))
    print(f"  real season files still present: {still_there}")
    return 0 if all(results) and still_there == 12 else 1


if __name__ == "__main__":
    sys.exit(main())
