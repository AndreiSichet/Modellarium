r"""The pre-push gate: five things that must be true before `git push`.

WHY THIS EXISTS. Three of the five are failures this project has already had,
and the fourth is the standing rule that has bitten five times. They have
nothing in common except *when* they are cheap to catch - after the work is
committed and before it leaves the machine - which is why they are one script
rather than five.

  1. THE WORKING TREE IS CLEAN. Not tidiness: `verify_daily_refresh.py`'s
     check 7 used to run the real pipeline and rewrite three tracked NBA
     tables in the working copy, twice on 2026-10-22. The standing rule
     against `git add -A` is what kept it out of a commit, and "the rule is
     what saved us" is the argument that preceded the 17:19 accident.

  2. NO UNTRACKED INTERMEDIATES UNDER `data-pipeline/data/processed/`.
     The same rebuild also rewrote the files that are NOT tracked, and
     `git checkout --` can only restore tracked ones. On 2026-10-24 the
     untracked intermediates sat at 26,426 rows while the four tracked
     tables sat at 26,398 - a +28-row split - and `git status` reported the
     tree clean throughout, because the whole difference was in ignored
     files. The repo copy was internally inconsistent for two days and no
     instrument could see it. They are all regenerable, so the answer is to
     have none of them rather than to try to prove they agree.

  3. NO CC BY-SA TABLE IS TRACKED. The NFL's and the NHL's corpora are
     CC BY-SA 4.0 and may never be committed, raw or processed. This is a
     licence boundary, not a tidiness one, and the last moment to check it
     is the push.

  4. EVERY DOCKERFILE `COPY` SOURCE IS TRACKED. `docker build` reads the
     filesystem and does not care what is committed; `actions/checkout` is
     the only thing that sees only what is. So a `COPY` of a path that
     exists locally and is not committed builds green here and red in CI -
     five times so far.

  5. THE NEVER-COMMITTED FILES ARE STILL NOT COMMITTED. `CLAUDE.md`,
     `NOTES_HISTORY.md` and every `probes/` folder.

EVERY CHECK COUNTS WHAT IT EXAMINED. `verify_nhl_selection.py`'s check 3
reported "0 violation(s)" while examining nothing, because the key it read
was absent - the `all_match`-computed-and-never-acted-on pattern. So a check
here that examines zero things is a FAILURE, not a pass.

Exit codes follow the project contract: 0 clean, 1 refused, 2 crashed before
reaching a verdict.

    python ml-training\check_tree_clean_before_push.py
    python ml-training\check_tree_clean_before_push.py --self-test
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

EXIT_CLEAN = 0
EXIT_REFUSED = 1
EXIT_NO_VERDICT = 2

PROCESSED = "data-pipeline/data/processed/"

# THE LICENCE BOUNDARY, AS PATHS. One prefix per league whose source is
# CC BY-SA, plus the wikitext that the NFL's and NHL's parsers read.
LICENSED_PREFIXES = ("data-pipeline/data/nfl/", "data-pipeline/data/nhl/")
LICENSED_SUFFIXES = (".wikitext",)

NEVER_COMMITTED = ("CLAUDE.md", "NOTES_HISTORY.md")
NEVER_COMMITTED_DIRS = ("probes/",)

DOCKERFILES = ("inference-service/Dockerfile",)


# ------------------------------------------------------------------ pure
def porcelain_violations(porcelain: str) -> list:
    """Every entry `git status --porcelain` printed. Clean means none.

    DELIBERATELY INCLUDES UNTRACKED-AND-NOT-IGNORED FILES. A new file that
    nothing ignores is a file somebody is about to forget, which is how
    `models/*.json` came to be missing from the commit at all.
    """
    return [line for line in porcelain.splitlines() if line.strip()]


def untracked_intermediates(present: list, tracked: set) -> list:
    """Files under data-pipeline/data/processed/ that git does not track."""
    return sorted(p for p in present
                  if p.startswith(PROCESSED) and p not in tracked)


def licensed_tables_tracked(tracked: set) -> list:
    """Tracked paths that fall inside the CC BY-SA boundary."""
    return sorted(p for p in tracked
                  if p.startswith(LICENSED_PREFIXES)
                  or p.endswith(LICENSED_SUFFIXES))


def never_committed_tracked(tracked: set) -> list:
    """Tracked paths that the standing rule says are never committed."""
    out = []
    for path in tracked:
        name = path.rsplit("/", 1)[-1]
        if name in NEVER_COMMITTED:
            out.append(path)
        elif any(part in NEVER_COMMITTED_DIRS
                 for part in (seg + "/" for seg in path.split("/")[:-1])):
            out.append(path)
    return sorted(out)


def copy_sources(dockerfile_text: str) -> list:
    """The SOURCE paths of every `COPY` in a Dockerfile, in order.

    Continuations are joined first: the NHL's own COPY line is a two-line
    one, so a parser that read single lines would miss `nhl_features.py` -
    which is the file whose absence from a commit this check exists to find.
    The LAST token of a COPY is its destination and is not a source.
    """
    joined = re.sub(r"\\\s*\n\s*", " ", dockerfile_text)
    sources = []
    for line in joined.splitlines():
        line = line.strip()
        if not line.upper().startswith("COPY "):
            continue
        tokens = [t for t in line.split()[1:] if not t.startswith("--")]
        if len(tokens) < 2:
            continue
        sources.extend(tokens[:-1])
    return sources


def untracked_copy_sources(sources: list, tracked: set,
                           exists) -> tuple:
    """(missing, untracked) among a Dockerfile's COPY sources.

    A directory source counts as tracked when git tracks anything beneath
    it, which is what `COPY ml-training/models_nhl/` actually needs.
    """
    missing, untracked = [], []
    for source in sources:
        rel = source.rstrip("/")
        if not exists(rel):
            missing.append(source)
            continue
        if source.endswith("/"):
            prefix = rel + "/"
            if not any(p.startswith(prefix) for p in tracked):
                untracked.append(source)
        elif rel not in tracked:
            untracked.append(source)
    return missing, untracked


# ------------------------------------------------------------------ live
def git(*args: str, root: Path) -> str:
    done = subprocess.run(("git",) + args, cwd=root, capture_output=True,
                          text=True)
    if done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} exited "
                           f"{done.returncode}: {done.stderr.strip()}")
    return done.stdout


REFUSALS: list = []


def say(ok: bool, name: str, examined: int, detail: str = "") -> None:
    if examined == 0:
        REFUSALS.append(f"{name}: examined NOTHING")
        print(f"  REFUSED {name}")
        print(f"          examined 0 things - a check that looks at nothing "
              f"is not a check")
        return
    if not ok:
        REFUSALS.append(name)
    print(f"  {'OK     ' if ok else 'REFUSED'} {name}  "
          f"[{examined} examined]")
    if detail:
        for line in detail.splitlines():
            print(f"          {line}")


def run(root: Path) -> int:
    print(f"pre-push gate on {root}\n")

    porcelain = git("status", "--porcelain", root=root)
    tracked = set(git("ls-files", root=root).splitlines())
    # IGNORED FILES TOO: that is the whole point of check 2, and
    # --others --ignored is the only way to see them.
    others = set(git("ls-files", "--others", "--ignored", "--exclude-standard",
                     root=root).splitlines())
    present_in_processed = sorted(
        p for p in (tracked | others) if p.startswith(PROCESSED))

    dirty = porcelain_violations(porcelain)
    say(not dirty, "1. the working tree is clean",
        # A CLEAN TREE PRINTS NOTHING, so there is nothing to count from the
        # output itself - the thing examined is the repository.
        examined=1,
        detail="\n".join(dirty[:20]) if dirty else "")

    intermediates = untracked_intermediates(present_in_processed, tracked)
    say(not intermediates,
        "2. no untracked intermediates under data-pipeline/data/processed/",
        examined=len(present_in_processed),
        detail=("\n".join(intermediates)
                + "\n-> delete them; the next pipeline run regenerates them"
                if intermediates else ""))

    licensed = licensed_tables_tracked(tracked)
    say(not licensed, "3. no CC BY-SA table is tracked",
        examined=len(tracked),
        detail="\n".join(licensed))

    examined_sources = 0
    copy_problems = []
    for dockerfile in DOCKERFILES:
        path = root / dockerfile
        if not path.is_file():
            copy_problems.append(f"{dockerfile}: not found")
            continue
        sources = copy_sources(path.read_text(encoding="utf-8"))
        examined_sources += len(sources)
        missing, untracked = untracked_copy_sources(
            sources, tracked, lambda rel: (root / rel).exists())
        copy_problems += [f"{dockerfile}: COPY source does not exist: {m}"
                          for m in missing]
        copy_problems += [f"{dockerfile}: COPY source NOT TRACKED: {u}"
                          for u in untracked]
    say(not copy_problems, "4. every Dockerfile COPY source is tracked",
        examined=examined_sources,
        detail="\n".join(copy_problems))

    forbidden = never_committed_tracked(tracked)
    say(not forbidden, "5. the never-committed files are not committed",
        examined=len(tracked),
        detail="\n".join(forbidden))

    print()
    if REFUSALS:
        print(f"REFUSED - {len(REFUSALS)} of 5 checks. Do not push.")
        for name in REFUSALS:
            print(f"  {name}")
        return EXIT_REFUSED
    print("CLEAN - 5 of 5 checks. Safe to push.")
    return EXIT_CLEAN


# --------------------------------------------------------------- self-test
PASSED: list = []
FAILED: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name)
    print(f"  {'PASS' if ok else 'FAIL'} {name:<64} {detail}")


DOCKERFILE_SAMPLE = """FROM python:3.14-slim
COPY inference-service/requirements.txt ./inference-service/requirements.txt
COPY ml-training/nhl/live_nhl_features.py ml-training/nhl/nhl_features.py \\
     ./ml-training/nhl/
COPY ml-training/models_nhl/ ./ml-training/models_nhl/
"""


def self_test() -> int:
    """Plant each of the five, against a control that passes.

    PURE INPUTS, NOT THE REAL REPOSITORY. Planting on the real tree would
    mean dirtying it, and a verifier must not change the state it measures -
    the lesson from the isolation suite's advisory case, which swapped a
    scratch volume's pointer and made every later assertion compare against
    a stale value.
    """
    print("check_tree_clean_before_push.py self-test\n")

    # ---- 1. the working tree -------------------------------------------
    check("CONTROL a clean porcelain -> no violations",
          porcelain_violations("") == []
          and porcelain_violations("\n") == [],
          "empty and blank-line output both read clean")
    for planted, label in ((" M data-pipeline/data/processed/games_final.csv",
                            "a MODIFIED tracked table"),
                           ("?? ml-training/stray.py", "an UNTRACKED file"),
                           ("A  backend/src/Foo.java", "a STAGED file")):
        check(f"PLANT {label} -> refused",
              porcelain_violations(planted) == [planted],
              repr(planted))

    # ---- 2. the intermediates ------------------------------------------
    tracked = {PROCESSED + "games_final.csv",
               PROCESSED + "quarter_half_raw.csv"}
    present = sorted(tracked)
    check("CONTROL only tracked tables in processed/ -> none",
          untracked_intermediates(present, tracked) == [],
          f"{len(present)} examined")
    planted = sorted(tracked | {PROCESSED + "model_dataset.csv",
                                PROCESSED + "games_with_rolling.csv"})
    check("PLANT two untracked intermediates -> both named",
          untracked_intermediates(planted, tracked)
          == [PROCESSED + "games_with_rolling.csv",
              PROCESSED + "model_dataset.csv"],
          "the +28-row split of 2026-10-24 lived in files like these")
    check("PLANT and a file OUTSIDE processed/ is not its business",
          untracked_intermediates(
              present + ["data-pipeline/data/raw/games_2026.csv"],
              tracked) == [],
          "raw/ is ignored wholesale and always has been")

    # ---- 3. the licence boundary ---------------------------------------
    clean_tracked = {"ml-training/nhl/nhl_features.py",
                     "ml-training/models_nhl/manifest.json",
                     "data-pipeline/data/processed/games_final.csv"}
    check("CONTROL NHL CODE and MODELS are tracked, and that is correct",
          licensed_tables_tracked(clean_tracked) == [],
          "the boundary is data-pipeline/data/<league>/, not the league")
    for planted, label in (
            ("data-pipeline/data/nhl/processed/nhl_games_final.csv",
             "an NHL processed table"),
            ("data-pipeline/data/nfl/raw/nfl_2026.csv", "an NFL raw table"),
            ("data-pipeline/data/nhl/raw/2026-27.wikitext", "wikitext")):
        check(f"PLANT {label} tracked -> refused",
              licensed_tables_tracked(clean_tracked | {planted}) == [planted],
              planted)

    # ---- 4. the COPY list ----------------------------------------------
    sources = copy_sources(DOCKERFILE_SAMPLE)
    check("the COPY parser JOINS continuations",
          "ml-training/nhl/nhl_features.py" in sources,
          f"{len(sources)} source(s): {sources}")
    check("and drops the DESTINATION, which is not a source",
          "./ml-training/nhl/" not in sources
          and "./ml-training/models_nhl/" not in sources,
          "the last token of a COPY is where it goes")
    check("CONTROL every source tracked -> no problem",
          untracked_copy_sources(sources, set(sources) | {
              "ml-training/models_nhl/manifest.json"},
              lambda rel: True) == ([], []),
          "a directory source counts via anything beneath it")
    partial = set(sources) - {"ml-training/nhl/nhl_features.py"}
    partial |= {"ml-training/models_nhl/manifest.json"}
    missing, untracked = untracked_copy_sources(
        sources, partial, lambda rel: True)
    check("PLANT a COPY source that EXISTS but is not tracked -> refused",
          untracked == ["ml-training/nhl/nhl_features.py"] and not missing,
          "this is the failure that has bitten five times")
    # AN EXACT ABSENCE, NOT A SUBSTRING ONE. The first version of this plant
    # used `"nhl_features" not in rel`, which also matches
    # live_nhl_features.py - so TWO sources went missing where the assertion
    # named one, and the check went red against working code. The harness
    # was wrong and the check was right, which is the better way round.
    gone = "ml-training/nhl/nhl_features.py"
    missing, untracked = untracked_copy_sources(
        sources, set(sources), lambda rel: rel != gone)
    check("PLANT a COPY source that does not exist at all -> refused",
          missing == [gone] and not untracked,
          "a typo in a COPY path fails the build, not this gate - but say so")
    missing, untracked = untracked_copy_sources(
        ["ml-training/models_nhl/"], {"ml-training/models_nhl"},
        lambda rel: True)
    check("PLANT a tracked FILE where a DIRECTORY source was named -> refused",
          untracked == ["ml-training/models_nhl/"],
          "COPY of a directory needs something tracked INSIDE it")

    # ---- 5. the never-committed files ----------------------------------
    check("CONTROL the standing-rule files absent from ls-files -> none",
          never_committed_tracked({"ml-training/common.py",
                                   "README.md"}) == [],
          "")
    for planted in ("CLAUDE.md", "NOTES_HISTORY.md",
                    "ml-training/probes/A3_SWITCH_RUNBOOK.md",
                    "data-pipeline/probes/scratch.py"):
        check(f"PLANT {planted} tracked -> refused",
              never_committed_tracked({"README.md", planted}) == [planted],
              planted)
    check("PLANT a path merely CONTAINING the word probes is not caught",
          never_committed_tracked({"ml-training/probes_helper.py"}) == [],
          "the rule is the DIRECTORY, and a substring test would overreach")

    # ---- the vacuous-check guard ---------------------------------------
    REFUSALS.clear()
    say(True, "a check that examined nothing", examined=0)
    check("GUARD a check examining ZERO things is REFUSED, not passed",
          len(REFUSALS) == 1 and "examined NOTHING" in REFUSALS[0],
          REFUSALS[0] if REFUSALS else "")
    REFUSALS.clear()

    print(f"\n  {len(PASSED)} of {len(PASSED) + len(FAILED)} checks passed")
    if FAILED:
        for name in FAILED:
            print(f"  FAILED: {name}")
        return EXIT_REFUSED
    return EXIT_CLEAN


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=None,
                        help="repository root (default: this file's repo)")
    parser.add_argument("--self-test", action="store_true",
                        help="plant each of the five and watch it go red")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    root = Path(args.root) if args.root else Path(__file__).resolve().parent.parent
    try:
        return run(root)
    except RuntimeError as error:
        print(f"CRASHED before a verdict: {error}")
        return EXIT_NO_VERDICT


if __name__ == "__main__":
    sys.exit(main())
