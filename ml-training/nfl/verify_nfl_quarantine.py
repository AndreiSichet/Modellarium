r"""An open-season disagreement is quarantined; a completed one still fails.

READ-ONLY with respect to the served volume and the repository's TRACKED
files. It does edit one raw wikitext article in place to plant a
disagreement, and restores the exact bytes in a `finally` - the NFL corpus is
gitignored (CC BY-SA, never committed), so nothing it touches can reach a
commit, and the restore is verified by sha256 before the suite reports.

The plants are the rows the refresh-isolation spec asks for:

  * open-season copies disagree       -> quarantined, build still exits 0
  * the game is excluded from nfl_games_final.csv
  * that disagreement resolved        -> leaves quarantine, reported
  * COMPLETED-season copies disagree  -> validation still fails (unchanged)

    python ml-training/nfl/verify_nfl_quarantine.py
"""
import hashlib
import io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
NFL_PRE = PROJECT / "data-pipeline" / "nfl" / "preprocessing"
RAW = PROJECT / "data-pipeline" / "data" / "nfl" / "raw"
PROCESSED = PROJECT / "data-pipeline" / "data" / "nfl" / "processed"
for directory in (NFL_PRE,):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

RESULTS = []

# The planted game, chosen because both copies are plain and unambiguous:
# New England at Seattle, 2026-09-09, SEA 13-10. The away copy is edited.
PLANT_ARTICLE = RAW / "2026" / "New_England_Patriots.wiki"
PLANT_FROM = "'''L''' 10–13"
PLANT_TO = "'''L''' 10–14"

# THE SAME DEFECT IN A COMPLETED SEASON, which must still be fatal. 2025 ended
# in February 2026, so its articles have settled and a disagreement is a real
# defect rather than a half-finished edit. The pair is this plant's whole
# point: the two runs differ ONLY in which season the game is in.
#
# Chosen so the plant breaks the PAIRING check alone. It moves
# `points_against` on New England's copy, which no other check reads - the
# quarter-sum check compares each side's own quarters against its own
# `points_for`, and the Record column does not depend on the margin - so a
# failure here cannot be the plant having broken something else.
COMPLETED_ARTICLE = RAW / "2025" / "New_England_Patriots.wiki"
COMPLETED_FROM = "'''L''' 31–35"
COMPLETED_TO = "'''L''' 31–36"


class PlantDidNothing(RuntimeError):
    """A plant that changes no behaviour is a harness bug, not a pass."""


def record(name, passed, detail):
    RESULTS.append((name, passed, detail))
    print("  %-4s %-56s %s" % ("PASS" if passed else "FAIL", name, detail))


def sha256_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_builder():
    done = subprocess.run(
        [sys.executable, "-W", "ignore",
         str(NFL_PRE / "build_nfl_tables.py")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=PROJECT, timeout=1800)
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def run_validator():
    done = subprocess.run(
        [sys.executable, "-W", "ignore",
         str(NFL_PRE / "validate_nfl_games.py")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=PROJECT, timeout=1800)
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def quarantine_rows():
    path = PROCESSED / "nfl_quarantine.csv"
    if not path.is_file():
        return None
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def games_final_rows():
    path = PROCESSED / "nfl_games_final.csv"
    return sum(1 for _ in path.open(encoding="utf-8")) - 1


def completed_season_still_fails() -> None:
    """The same disagreement, one season earlier, must still fail validation.

    This is the half of the rule that is easy to lose: relaxing strictness for
    the open season is only safe if it did not quietly relax it everywhere.
    The 8-of-9 run that preceded this function is why it exists - the first
    implementation keyed on `F.open_seasons()`, which includes the PREVIOUS
    season, so a disagreement in the finished 2025 season would have been set
    aside instead of failing. Only a plant in a completed season catches that.
    """
    original = COMPLETED_ARTICLE.read_bytes()
    before_hash = sha256_of(COMPLETED_ARTICLE)
    try:
        text = original.decode("utf-8")
        if COMPLETED_FROM not in text:
            raise PlantDidNothing(
                f"{COMPLETED_FROM!r} is not in {COMPLETED_ARTICLE.name}; the "
                f"article changed and the plant would have tested nothing")
        COMPLETED_ARTICLE.write_bytes(
            text.replace(COMPLETED_FROM, COMPLETED_TO, 1).encode("utf-8"))
        if sha256_of(COMPLETED_ARTICLE) == before_hash:
            raise PlantDidNothing("the plant changed no bytes")

        code, out = run_validator()
        record("completed-season plant: validation FAILS",
               code != 0, f"exit {code}, expected non-zero")
        # WHICH SIDE IS HOME IS NOT ASSUMED. The message reads
        # "scores {home.pf}-{home.pa} vs {away.pa}-{away.pf}", so whether New
        # England is the home or the away copy decides the order, and
        # asserting a fixed order would be guessing at the fixture. What the
        # plant has to prove is that PAIRING is the check that failed and that
        # the planted number reached its message.
        pairing = next((line for line in out.splitlines()
                        if "2025 pairing" in line), "")
        record("completed-season plant: it fails on PAIRING, nothing else",
               bool(pairing) and "36" in pairing and "1 unlisted" in pairing,
               pairing.strip() or "no pairing line")
        record("completed-season plant: it is NOT called quarantined",
               "QUARANTINED" not in out,
               "no quarantine note, as a settled season requires")
    finally:
        COMPLETED_ARTICLE.write_bytes(original)

    record("the completed-season article is restored byte for byte",
           sha256_of(COMPLETED_ARTICLE) == before_hash,
           f"sha256 {before_hash[:12]}")
    code, out = run_validator()
    record("after revert: validation is clean again",
           code == 0 and "0 failed" in out,
           f"exit {code}, 0 failed: {'0 failed' in out}")


def main() -> int:
    print("NFL OPEN-SEASON QUARANTINE")
    print(f"  corpus   : {RAW}")
    print(f"  article  : {PLANT_ARTICLE.name}")

    original = PLANT_ARTICLE.read_bytes()
    before_hash = sha256_of(PLANT_ARTICLE)

    # ---- control
    code, out = run_builder()
    base_rows = games_final_rows()
    base_quarantine = quarantine_rows()
    record("control: clean corpus builds, nothing quarantined",
           code == 0 and base_quarantine == [],
           f"exit {code}, {base_rows} rows, "
           f"{len(base_quarantine or [])} quarantined")

    try:
        text = original.decode("utf-8")
        if PLANT_FROM not in text:
            raise PlantDidNothing(
                f"{PLANT_FROM!r} is not in {PLANT_ARTICLE.name}; the article "
                f"changed and the plant would have tested nothing")
        PLANT_ARTICLE.write_bytes(
            text.replace(PLANT_FROM, PLANT_TO, 1).encode("utf-8"))
        if sha256_of(PLANT_ARTICLE) == before_hash:
            raise PlantDidNothing("the plant changed no bytes")

        code, out = run_builder()
        rows = quarantine_rows()
        record("plant: open-season copies disagree -> build still exits 0",
               code == 0, f"exit {code}, expected 0")
        record("plant: the game is quarantined with both copies and a reason",
               rows is not None and len(rows) == 1
               and "scores disagree" in rows[0]["reason"]
               and rows[0]["copy_a"] and rows[0]["copy_b"],
               f"{len(rows or [])} row(s): "
               f"{(rows or [{}])[0].get('reason', '')[:46]}")
        record("plant: it names both source articles",
               bool(rows) and "Seahawks" in rows[0]["copy_a"]
               and "Patriots" in rows[0]["copy_b"],
               "copy_a / copy_b name the articles")
        record("plant: the game is excluded from nfl_games_final.csv",
               games_final_rows() == base_rows - 2,
               f"{games_final_rows()} rows, was {base_rows} (one game = 2)")

        # ---- the validator, which runs BEFORE the build in the refresh and
        # therefore has to reach the same verdict without reading the
        # quarantine file.
        vcode, vout = run_validator()
        record("plant: the validator tolerates it and exits 0",
               vcode == 0, f"exit {vcode}, expected 0")
        record("plant: the validator NAMES it as quarantined, advisory",
               "QUARANTINED (open season, advisory)" in vout
               and "scores" in vout,
               "note present" if "QUARANTINED" in vout else "NO NOTE")
        record("plant: the validator still reports 0 failed checks",
               "0 failed" in vout,
               next((line.strip() for line in vout.splitlines()
                     if "checks," in line), "no summary line"))

        # ---- the rule behind the behaviour: the set is derived rather than
        # listed, so it moves when the calendar does.
        import nfl_franchises as F
        eligible = {F.current_season()}
        record("quarantine is eligible for the CURRENT season only",
               2026 in eligible and 2025 not in eligible,
               f"eligible: {sorted(eligible)}; "
               f"open_seasons() is {sorted(F.open_seasons())} and is the "
               f"FETCHER's question, not this one")
    finally:
        PLANT_ARTICLE.write_bytes(original)

    record("the planted article is restored byte for byte",
           sha256_of(PLANT_ARTICLE) == before_hash,
           f"sha256 {before_hash[:12]}")

    # ---- leaving quarantine, on the next build with the corpus clean
    code, out = run_builder()
    rows = quarantine_rows()
    record("plant resolved: the game leaves quarantine and it is reported",
           code == 0 and rows == [] and "LEFT QUARANTINE" in out,
           f"exit {code}, {len(rows or [])} quarantined, "
           f"movement reported: {'LEFT QUARANTINE' in out}")
    record("after revert: nfl_games_final.csv is back to its row count",
           games_final_rows() == base_rows,
           f"{games_final_rows()} rows, baseline {base_rows}")

    print()
    completed_season_still_fails()

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
