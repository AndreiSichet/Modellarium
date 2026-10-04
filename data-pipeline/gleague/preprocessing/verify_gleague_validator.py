"""Negative-test the G League validator and the completeness mechanism.

A guard that has never been seen to fail has not been verified. Every test
here corrupts a COPY of one season and requires the validator to go red; the
real files in data/gleague/raw are never written to, which is asserted at the
end by comparing their hashes against hashes taken before anything ran.

THE CONTROL IS NOT OPTIONAL. Before any corruption, the unmodified copies are
validated and must pass - otherwise a test that goes red proves nothing, since
it could have been red already. Three of this project's guards turned out to be
vacuous and all three were caught by a control rather than by review.

THE COMPLETENESS MECHANISM IS TESTED IN BOTH DIRECTIONS, which is the part
most worth testing because it is new. The spec asked for a named waiver
(TRUNCATED_SEASONS) with a stale-waiver guard; measurement replaced it with a
derived three-way verdict, so what needs proving is that the verdict is read
off the distribution every time rather than remembered:

  forward   a season that currently reconciles stops reconciling when a game
            is removed - the rule is actually being applied
  backward  a season currently judged unbalanced is NOT judged unbalanced once
            its distribution no longer looks unbalanced - the verdict cannot
            be a hidden list of season names
"""

import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parents[1]
sys.path.insert(0, str(HERE))
from clean_gleague_rows import schedule_shape  # noqa: E402

REAL_RAW = PIPELINE / "data" / "gleague" / "raw"
VALIDATOR = HERE / "validate_gleague_games.py"

results = []


def record(name, ok, detail=""):
    results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in detail.splitlines():
            print(f"         {line}")


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def hash_tree(directory: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.glob("*.csv"))}


def run_validator(raw_dir: Path) -> tuple:
    """Run the real validator against raw_dir, as a subprocess.

    A subprocess rather than an import, so the module's own RAW_DIR default and
    its exit code are both exercised - the same reason the retrain harness
    invokes continuous_retrain.py rather than calling into it.
    """
    source = VALIDATOR.read_text(encoding="utf-8")
    patched = source.replace(
        'RAW_DIR = PIPELINE / "data" / "gleague" / "raw"',
        f'RAW_DIR = Path(r"{raw_dir}")')
    if patched == source:
        raise SystemExit("could not repoint the validator's RAW_DIR - its "
                         "declaration has changed, so this harness would be "
                         "testing the real files")

    temp_script = raw_dir.parent / "_validator_under_test.py"
    temp_script.write_text(patched, encoding="utf-8")
    shutil.copy(HERE / "clean_gleague_rows.py",
                temp_script.parent / "clean_gleague_rows.py")

    proc = subprocess.run(
        [sys.executable, str(temp_script)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def fresh_copy(workspace: Path, seasons=None) -> Path:
    """A copy of the raw corpus, or of just the named seasons."""
    target = workspace / "raw"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for path in REAL_RAW.glob("*.csv"):
        if seasons is None or any(f"_{s}.csv" in path.name for s in seasons):
            shutil.copy(path, target / path.name)
    return target


def pick_clean_game(frame: pd.DataFrame) -> str:
    """A game with two rows, both WL present, scores unequal - so corrupting
    it is unambiguous and the validator has something to contradict."""
    for game_id, group in frame.groupby("GAME_ID"):
        if (len(group) == 2
                and group["WL"].isin({"W", "L"}).all()
                and group["PTS"].nunique() == 2
                and group["TEAM_ID"].notna().all()):
            return game_id
    raise SystemExit("no suitable game found to corrupt")


def main() -> int:
    print(__doc__)
    before = hash_tree(REAL_RAW)
    print(f"Hashed {len(before)} real raw file(s) before starting.")

    workspace = Path(tempfile.mkdtemp(prefix="gleague_negtest_"))
    season = "2024-25"
    target_file = f"gleague_regular_{season}.csv"

    try:
        # ---------------------------------------------------------- control
        # THE WHOLE CORPUS IS COPIED, NOT ONE SEASON, and the control is what
        # established that. Copying 2024-25 alone made the control fail -
        # correctly, because the staleness guard requires the three named
        # unusable games to still be present, and they live in 2004-05 and
        # 2005-06. A one-season harness would have reported every corruption
        # test as "caught" while the real cause was the missing exclusions.
        section("CONTROL - unmodified copies must pass")
        raw = fresh_copy(workspace)
        code, out = run_validator(raw)
        record("unmodified copy of the whole corpus validates cleanly", code == 0,
               "" if code == 0 else out[-1200:])
        if code != 0:
            print("\nThe control failed, so no corruption test below could "
                  "mean anything. Stopping.")
            return 1

        # ------------------------------------------------- 1 dropped row
        section("TEST 1 - drop one team-row from a game")
        raw = fresh_copy(workspace)
        path = raw / target_file
        frame = pd.read_csv(path, dtype={"GAME_ID": str})
        victim = pick_clean_game(frame)
        dropped = frame[~((frame["GAME_ID"] == victim)
                          & (frame.index == frame[
                              frame["GAME_ID"] == victim].index[0]))]
        dropped.to_csv(path, index=False, encoding="utf-8")
        code, out = run_validator(raw)
        caught = code != 0
        record(f"a one-sided game ({victim}) is caught", caught,
               extract_failures(out) if caught else out[-1500:])

        # -------------------------------------------------- 2 swapped scores
        section("TEST 2 - swap one game's scores")
        raw = fresh_copy(workspace)
        path = raw / target_file
        frame = pd.read_csv(path, dtype={"GAME_ID": str})
        victim = pick_clean_game(frame)
        rows = frame.index[frame["GAME_ID"] == victim].tolist()
        a, b = frame.at[rows[0], "PTS"], frame.at[rows[1], "PTS"]
        frame.at[rows[0], "PTS"], frame.at[rows[1], "PTS"] = b, a
        frame.to_csv(path, index=False, encoding="utf-8")
        code, out = run_validator(raw)
        caught = code != 0
        record(f"swapped scores on {victim} ({a} <-> {b}) are caught", caught,
               extract_failures(out) if caught else out[-1500:])

        # ----------------------------------------------------- 3 corrupt WL
        section("TEST 3 - corrupt one game's WL")
        raw = fresh_copy(workspace)
        path = raw / target_file
        frame = pd.read_csv(path, dtype={"GAME_ID": str})
        victim = pick_clean_game(frame)
        rows = frame.index[frame["GAME_ID"] == victim].tolist()
        # Both sides marked W: contradicts PTS whichever way the scores read,
        # and is not the 'O'/'T' shape the cleaner already normalises away.
        frame.loc[rows, "WL"] = "W"
        frame.to_csv(path, index=False, encoding="utf-8")
        code, out = run_validator(raw)
        caught = code != 0
        record(f"both sides of {victim} marked 'W' is caught", caught,
               extract_failures(out) if caught else out[-1500:])

        # ------------------------------------- 4 the completeness mechanism
        section("TEST 4 - the completeness mechanism, BOTH directions")

        uniform_season = None
        unbalanced_season = None
        for path in sorted(REAL_RAW.glob("gleague_regular_*.csv")):
            label = path.stem.split("_")[-1]
            frame = pd.read_csv(path, dtype={"GAME_ID": str})
            frame = frame[frame["TEAM_ID"].notna()]
            shape = schedule_shape(frame)
            if shape["verdict"] == "uniform" and uniform_season is None:
                uniform_season = (label, frame, shape)
            if shape["verdict"] == "unbalanced" and unbalanced_season is None:
                unbalanced_season = (label, frame, shape)

        # 4a FORWARD: a reconciling season must stop reconciling.
        label, frame, shape = uniform_season
        record(f"4a baseline: {label} is '{shape['verdict']}' and "
               f"reconciles ({shape['detail']})",
               shape["verdict"] == "uniform" and shape["ok"] is True)

        victim = pick_clean_game(frame)
        holed = frame[frame["GAME_ID"] != victim]
        after = schedule_shape(holed)
        # Removing a whole game leaves two teams one short of the mode, so the
        # verdict moves to short-only and must then reconcile at 2/2 == 1.
        # What matters is that the rule NOTICED - the count changed with it.
        noticed = (after["verdict"] != "uniform"
                   and after["games"] == shape["games"] - 1)
        record(f"4a forward: removing game {victim} moves {label} from "
               f"'uniform' to '{after['verdict']}' ({after['detail']})",
               noticed)

        # And a hole the rule CANNOT excuse: drop one game and also one
        # team-row of another, so the shortfall no longer halves evenly.
        extra = pick_clean_game(holed)
        idx = holed.index[holed["GAME_ID"] == extra].tolist()[0]
        lopsided = holed.drop(index=idx)
        broken = schedule_shape(lopsided)
        record("4a forward: a shortfall that does not halve evenly is "
               f"rejected ({broken['verdict']}: {broken['detail']})",
               broken["ok"] is False or broken["verdict"] == "unbalanced")

        # 4b BACKWARD: the verdict must be derived, not a remembered list.
        label, frame, shape = unbalanced_season
        record(f"4b baseline: {label} is 'unbalanced' "
               f"(min {shape['min']}, mode {shape['modal']}, "
               f"max {shape['max']})", shape["verdict"] == "unbalanced")

        # Trim every team back to the modal count. Same season name, same
        # file, a distribution that is no longer unbalanced - so if the
        # verdict is read off the data it must change.
        keep = []
        counts = {}
        modal = shape["modal"]
        for game_id, group in frame.groupby("GAME_ID", sort=True):
            ids = group["TEAM_ID"].tolist()
            if all(counts.get(t, 0) < modal for t in ids):
                keep.append(game_id)
                for t in ids:
                    counts[t] = counts.get(t, 0) + 1
        trimmed = frame[frame["GAME_ID"].isin(keep)]
        after = schedule_shape(trimmed)
        record(f"4b backward: with every team at or under the mode, {label} "
               f"is no longer 'unbalanced' (now '{after['verdict']}')",
               after["verdict"] != "unbalanced",
               f"min {after['min']}, mode {after['modal']}, "
               f"max {after['max']} - the verdict follows the distribution, "
               f"so it cannot be a list of season names")

        # ------------------------------ 6 a game duplicated under a new id
        section("TEST 6 - the same game duplicated under a fresh GAME_ID")
        print("The hole the derived schedule verdict opened: a duplicate "
              "looks like a")
        print("team playing more than the mode, which 'unbalanced' now "
              "permits, and both")
        print("copies pass the exactly-twice check.\n")

        raw = fresh_copy(workspace)
        path = raw / target_file
        frame = pd.read_csv(path, dtype={"GAME_ID": str})
        victim = pick_clean_game(frame)
        copy = frame[frame["GAME_ID"] == victim].copy()
        # A fresh id in the same season, same type digit, so nothing else
        # can object to it: only the date-and-pair key can.
        taken = set(frame["GAME_ID"])
        # A season's ids are contiguous, so a small offset is always taken.
        # The offset must keep the id 10 digits with '2' third, or the
        # type-digit check would fire instead and prove the wrong thing.
        new_id = next(
            candidate for n in range(1, 9000)
            if (candidate := str(int(victim) + 9000 + n)) not in taken
            and len(candidate) == 10 and candidate[2] == "2")
        copy["GAME_ID"] = new_id
        doubled = pd.concat([frame, copy], ignore_index=True)
        doubled.to_csv(path, index=False, encoding="utf-8")
        code, out = run_validator(raw)
        caught = code != 0 and "share a date and a team pair" in out
        record(f"game {victim} duplicated as {new_id} is caught", caught,
               extract_failures(out) if caught else out[-900:])

        # --------------------------- 5 the exclusion set, BOTH directions
        section("TEST 5 - the named exclusion set, BOTH directions")
        print("The spec asked for a waiver negative-tested in both "
              "directions. The waiver")
        print("turned out to belong on GAMES rather than on seasons, so "
              "both directions")
        print("are tested here.\n")

        # 5a A listed game that has vanished from the corpus must fail, or a
        #    dead entry could excuse a game that no longer exists.
        raw = fresh_copy(workspace)
        victim_id = "2020400101"
        path = raw / "gleague_regular_2004-05.csv"
        frame = pd.read_csv(path, dtype={"GAME_ID": str})
        frame[frame["GAME_ID"] != victim_id].to_csv(
            path, index=False, encoding="utf-8")
        code, out = run_validator(raw)
        caught = code != 0 and "still in the corpus" in out
        record(f"5a a listed game ({victim_id}) missing from the corpus "
               f"is caught", caught,
               extract_failures(out) if caught else out[-900:])

        # 5b A listed game that has stopped being unusable must fail - the
        #    entry is then stale and is excusing nothing.
        raw = fresh_copy(workspace)
        path = raw / "gleague_regular_2004-05.csv"
        frame = pd.read_csv(path, dtype={"GAME_ID": str})
        rows = frame.index[frame["GAME_ID"] == victim_id].tolist()
        # Repair it: make WL agree with PTS, so the cleaner keeps the game.
        hi = max(rows, key=lambda i: frame.at[i, "PTS"])
        for i in rows:
            frame.at[i, "WL"] = "W" if i == hi else "L"
        frame.to_csv(path, index=False, encoding="utf-8")
        code, out = run_validator(raw)
        caught = code != 0 and "still unusable" in out
        record(f"5b a listed game ({victim_id}) that is no longer unusable "
               f"is caught", caught,
               extract_failures(out) if caught else out[-900:])

        # ------------------------------------------------ the real files
        section("THE REAL RAW FILES WERE NEVER WRITTEN TO")
        after_hashes = hash_tree(REAL_RAW)
        changed = [n for n in before
                   if after_hashes.get(n) != before[n]]
        record(f"all {len(before)} raw file hashes unchanged",
               not changed and set(after_hashes) == set(before),
               f"changed: {changed}" if changed else "")

    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    section("RESULT")
    failed = [n for n, ok in results if not ok]
    print(f"  {len(results) - len(failed)} passed, {len(failed)} failed")
    for name in failed:
        print(f"    FAILED: {name}")
    return 1 if failed else 0


def extract_failures(out: str) -> str:
    """The validator's own failure lines, so a pass shows WHY it was caught
    rather than only that the exit code was non-zero."""
    lines = [ln.strip() for ln in out.splitlines()
             if "FAIL" in ln or "checks passed" in ln]
    return "\n".join(lines[-6:]) if lines else out[-400:]


if __name__ == "__main__":
    sys.exit(main())
