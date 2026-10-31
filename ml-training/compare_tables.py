r"""Byte-compare two trees of served tables, and say what differs.

WHY THIS EXISTS. The switch runbook copies a snapshot twice - once from the
live volume into staging, once from staging into the live volume - and both
copies are made by `robocopy`, which reports success for a partial copy in
several of its exit codes. `robocopy`'s own status answers "did the copy
tool fail"; it does not answer "do these two directories now hold the same
bytes", and that second question is the one the switch depends on. Nothing
in this repository answered it, so the runbook's own instructions named a
script that did not exist.

THE SET DIFFERENCE IS FATAL BY DEFAULT, AND THAT IS DELIBERATE. A flag that
has to be remembered to make a check strict is a check that goes vacuous the
first time somebody forgets it - `verify_nhl_selection.py`'s check 3 examined
nothing and reported "0 violations" for exactly that shape. So:

    (default)                 a differing file OR a differing file set -> 1
    --require-same-file-set   the same thing, said out loud at the call site
    --allow-set-difference    the opt-out, for comparing an OLD snapshot with
                              a NEW one, where the NHL's five optional tables
                              legitimately appear on one side only

COMPARING NOTHING IS NOT A PASS. Two empty trees, or a pattern that matches
no file, exits 2 - crashed before reaching a verdict - because "0 files
differ" out of 0 files compared is the most misleading green available.

RAW BYTES, NO NEWLINE NORMALISATION. These files are written by one program
on one machine; a snapshot whose line endings changed is a snapshot that
changed. (Normalising is right when comparing a working file with `git show`,
which is a different question - see CLAUDE.md's note on md5 and CRLF.)

Exit codes follow the project contract: 0 a verdict was reached and the trees
match, 1 they do not, 2 never reached a verdict.

    python ml-training\compare_tables.py A B --require-same-file-set
    python ml-training\compare_tables.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

EXIT_MATCH = 0
EXIT_DIFFER = 1
EXIT_NO_VERDICT = 2

READ_CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(READ_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root: Path, pattern: str = "*") -> dict:
    """{relative posix path: sha256} for every matching file under root.

    RECURSIVE ON PURPOSE. A snapshot is `processed/`, `wnba/`, `gleague/`,
    `nfl/`, `nhl/` and `SNAPSHOT.json`, so a top-level-only walk would
    compare one file and report a match.
    """
    out = {}
    for path in sorted(root.rglob(pattern)):
        if path.is_file():
            out[path.relative_to(root).as_posix()] = sha256_of(path)
    return out


def compare(before: dict, after: dict) -> dict:
    """Pure: two {path: hash} maps in, the three ways they can disagree out."""
    shared = sorted(set(before) & set(after))
    return {
        "compared": len(shared),
        "differing": [p for p in shared if before[p] != after[p]],
        "only_in_a": sorted(set(before) - set(after)),
        "only_in_b": sorted(set(after) - set(before)),
    }


def report(result: dict, a: Path, b: Path, allow_set_difference: bool) -> int:
    print(f"A  {a}")
    print(f"B  {b}")
    print(f"   {result['compared']} file(s) present in both")

    if not result["compared"] and not result["only_in_a"] \
            and not result["only_in_b"]:
        print("\nREFUSED: nothing to compare. Two empty trees, or a pattern "
              "that matched no file.")
        print("         '0 files differ' out of 0 files is not a pass.")
        return EXIT_NO_VERDICT

    failed = False

    if result["differing"]:
        failed = True
        print(f"\nDIFFERING CONTENT - {len(result['differing'])} file(s):")
        for path in result["differing"]:
            print(f"  {path}")

    for key, label in (("only_in_a", "only in A"), ("only_in_b", "only in B")):
        if result[key]:
            print(f"\n{label.upper()} - {len(result[key])} file(s):")
            for path in result[key]:
                print(f"  {path}")

    set_differs = bool(result["only_in_a"] or result["only_in_b"])
    if set_differs and not allow_set_difference:
        failed = True
        print("\nThe file SETS differ. That is fatal unless you meant it: "
              "pass --allow-set-difference")
        print("to compare an old snapshot with a new one, where the NHL's "
              "five optional tables")
        print("appear on one side only.")
    elif set_differs:
        print("\n(set difference ALLOWED by --allow-set-difference)")

    if failed:
        print("\nVERDICT: DIFFER")
        return EXIT_DIFFER
    print(f"\nVERDICT: MATCH - {result['compared']} file(s), byte-identical")
    return EXIT_MATCH


# --------------------------------------------------------------- self-test
PASSED = []
FAILED = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name)
    print(f"  {'PASS' if ok else 'FAIL'} {name:<62} {detail}")


def write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def self_test() -> int:
    """Plant every defect this script exists to catch and watch it go red.

    A check that has never failed has not been verified, and twice in this
    project the "finding" turned out to be the harness. So the controls
    matter as much as the plants: a pair that must match, and a pair whose
    files are the same SIZE but different content, which is what catches a
    comparison that is really comparing sizes.
    """
    print("compare_tables.py self-test\n")
    work = Path(tempfile.mkdtemp(prefix="compare_tables_"))
    try:
        a, b = work / "a", work / "b"
        for root in (a, b):
            write(root, "processed/games_final.csv", "GAME_ID,PTS\n1,100\n")
            write(root, "nhl/processed/nhl_games_final.csv", "x\n1\n")
            write(root, "SNAPSHOT.json", '{"stamp": "t"}\n')

        # CONTROL: identical trees match, and compared THREE files - the
        # count is asserted because a recursive walk that silently became
        # top-level-only would still report a match.
        result = compare(hash_tree(a), hash_tree(b))
        check("CONTROL identical trees -> MATCH",
              not result["differing"] and not result["only_in_a"]
              and not result["only_in_b"],
              f"{result['compared']} file(s) compared")
        check("CONTROL the walk is RECURSIVE - nested files are compared",
              result["compared"] == 3,
              f"compared {result['compared']}, expected 3")

        # PLANT (a): one byte, in a NESTED file.
        write(b, "nhl/processed/nhl_games_final.csv", "x\n2\n")
        result = compare(hash_tree(a), hash_tree(b))
        check("PLANT one byte in a nested file -> DIFFER, and it is named",
              result["differing"] == ["nhl/processed/nhl_games_final.csv"],
              f"differing {result['differing']}")
        write(b, "nhl/processed/nhl_games_final.csv", "x\n1\n")

        # PLANT (b): SAME LENGTH, different content. This is the one that
        # catches a comparison that is really comparing sizes or mtimes.
        write(b, "SNAPSHOT.json", '{"stamp": "u"}\n')
        same_size = ((a / "SNAPSHOT.json").stat().st_size
                     == (b / "SNAPSHOT.json").stat().st_size)
        result = compare(hash_tree(a), hash_tree(b))
        check("PLANT same SIZE, different content -> DIFFER",
              same_size and result["differing"] == ["SNAPSHOT.json"],
              f"sizes equal: {same_size}; differing {result['differing']}")
        write(b, "SNAPSHOT.json", '{"stamp": "t"}\n')

        # PLANT (c): a file only on one side, each way round.
        write(b, "nhl/processed/nhl_fixtures.csv", "f\n")
        result = compare(hash_tree(a), hash_tree(b))
        strict = report(result, a, b, allow_set_difference=False)
        lenient = report(result, a, b, allow_set_difference=True)
        check("PLANT a file only in B -> DIFFER by default",
              strict == EXIT_DIFFER
              and result["only_in_b"] == ["nhl/processed/nhl_fixtures.csv"],
              f"exit {strict}, only_in_b {result['only_in_b']}")
        check("and --allow-set-difference lets THAT, and only that, through",
              lenient == EXIT_MATCH,
              f"exit {lenient}")
        (b / "nhl/processed/nhl_fixtures.csv").unlink()

        write(a, "nhl/processed/nhl_quarantine.csv", "q\n")
        result = compare(hash_tree(a), hash_tree(b))
        check("PLANT a file only in A -> DIFFER, named the other way round",
              result["only_in_a"] == ["nhl/processed/nhl_quarantine.csv"]
              and report(result, a, b, False) == EXIT_DIFFER,
              f"only_in_a {result['only_in_a']}")
        (a / "nhl/processed/nhl_quarantine.csv").unlink()

        # PLANT (d): content differs AND the set differs. The opt-out must
        # not forgive the content - it forgives the SET only.
        write(b, "processed/games_final.csv", "GAME_ID,PTS\n1,101\n")
        write(b, "extra.csv", "e\n")
        result = compare(hash_tree(a), hash_tree(b))
        check("PLANT --allow-set-difference does NOT forgive CONTENT",
              report(result, a, b, allow_set_difference=True) == EXIT_DIFFER,
              f"differing {result['differing']}")
        write(b, "processed/games_final.csv", "GAME_ID,PTS\n1,100\n")
        (b / "extra.csv").unlink()

        # PLANT (e): nothing to compare. The vacuous-check guard.
        empty_a, empty_b = work / "empty_a", work / "empty_b"
        empty_a.mkdir()
        empty_b.mkdir()
        result = compare(hash_tree(empty_a), hash_tree(empty_b))
        check("PLANT two EMPTY trees -> exit 2, never reached a verdict",
              report(result, empty_a, empty_b, False) == EXIT_NO_VERDICT,
              "0 of 0 differing is not a pass")

        # And a pattern that matches nothing is the same mistake wearing a
        # flag: it would otherwise report MATCH on two populated trees.
        result = compare(hash_tree(a, "*.parquet"), hash_tree(b, "*.parquet"))
        check("PLANT a --pattern that matches NOTHING -> exit 2 as well",
              report(result, a, b, False) == EXIT_NO_VERDICT,
              "populated trees, no matching file")

        # FINAL CONTROL: back to where we started, so the plants left nothing
        # behind. A plant that does not revert makes every later check lie.
        result = compare(hash_tree(a), hash_tree(b))
        check("CONTROL the plants all reverted -> MATCH again",
              report(result, a, b, False) == EXIT_MATCH,
              f"{result['compared']} file(s)")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"\n  {len(PASSED)} of {len(PASSED) + len(FAILED)} checks passed")
    if FAILED:
        for name in FAILED:
            print(f"  FAILED: {name}")
        return EXIT_DIFFER
    return EXIT_MATCH


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a", nargs="?", help="the first tree")
    parser.add_argument("b", nargs="?", help="the second tree")
    parser.add_argument("--pattern", default="*",
                        help="glob for the files to compare (default: all)")
    parser.add_argument(
        "--require-same-file-set", action="store_true",
        help="the DEFAULT, spelled out at the call site. Accepted so a "
             "runbook can say what it means; it changes nothing.")
    parser.add_argument(
        "--allow-set-difference", action="store_true",
        help="a file present on one side only is reported, not fatal. For "
             "comparing an OLD snapshot with a NEW one.")
    parser.add_argument("--self-test", action="store_true",
                        help="plant every defect and watch it go red")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if not args.a or not args.b:
        parser.print_usage()
        print("\nREFUSED: two trees are required (or --self-test).")
        return EXIT_NO_VERDICT
    if args.require_same_file_set and args.allow_set_difference:
        print("REFUSED: --require-same-file-set and --allow-set-difference "
              "contradict each other.")
        return EXIT_NO_VERDICT

    a, b = Path(args.a), Path(args.b)
    for root in (a, b):
        if not root.is_dir():
            print(f"REFUSED: not a directory: {root}")
            return EXIT_NO_VERDICT

    return report(compare(hash_tree(a, args.pattern),
                          hash_tree(b, args.pattern)),
                  a, b, args.allow_set_difference)


if __name__ == "__main__":
    sys.exit(main())
