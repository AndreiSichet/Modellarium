r"""A promotion whose provenance does not describe the promoted weights fails.

THE HAZARD THIS EXISTS FOR WAS CREATED BY ADDING provenance.json.

`finalize_models.py` writes `ml-training/models/provenance.json`. The weekly
retrain never runs it - it writes candidates into `models_candidate/` instead -
and the promotion step copies those candidates over the shipped models with
`Copy-Item *.json`. So a promoted pull request would have carried NEW WEIGHTS
with a provenance.json describing the PREVIOUS ones, and committed the pair.

Every number in such a file looks right. None of it is about the files beside
it. That is the most expensive shape of misleading green this project has:
a record that is wrong in a way no reader can see.

So `continuous_retrain.py` now writes a provenance file beside its candidates,
and this script refuses a promotion where the two have come apart. The
workflow runs it after copying the candidates and before `git add`, so the
failure happens before anything is committed rather than after it is merged.

    python ml-training/verify_promotion_provenance.py                 # models/
    python ml-training/verify_promotion_provenance.py --dir <path>
    python ml-training/verify_promotion_provenance.py --self-test     # plants
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
PROVENANCE = "provenance.json"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(directory: Path, verbose: bool = True) -> list:
    """Problems found, empty if the provenance describes what is there.

    FOUR WAYS THEY CAN COME APART, and each is reported separately because
    they mean different things:

      * no provenance at all        - nothing to check against
      * a model with no entry       - a file was added without a record
      * an entry with no model      - a record for a file that is not here
      * a hash that does not match  - the record is about different weights

    The last is the one the hazard above produces, and the only one a reader
    could not spot by eye.
    """
    problems = []
    path = directory / PROVENANCE

    if not path.is_file():
        return [f"no {PROVENANCE} in {directory} - there is nothing to "
                f"check the weights against, so a promotion from here would "
                f"ship an unrecorded set"]

    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [f"{PROVENANCE} is unreadable: {type(error).__name__}: {error}"]

    recorded = body.get("models") or {}
    on_disk = {p.name for p in directory.glob("*.json")
               if p.name not in (PROVENANCE, "trained_through.json")}
    claimed = {entry.get("file") for entry in recorded.values()}

    for name in sorted(on_disk - claimed):
        problems.append(f"{name} is present with no entry in {PROVENANCE}")
    for name in sorted(claimed - on_disk):
        problems.append(f"{PROVENANCE} records {name}, which is not here")

    for key, entry in sorted(recorded.items()):
        name = entry.get("file")
        if name not in on_disk:
            continue
        want = entry.get("sha256")
        got = sha256_of(directory / name)
        if want != got:
            problems.append(
                f"{name}: {PROVENANCE} records {str(want)[:16]} but the file "
                f"is {got[:16]} - the record describes DIFFERENT weights")

    if verbose and not problems:
        print(f"  {len(recorded)} model(s) in {directory.name}, every sha256 "
              f"matches {PROVENANCE}")
        print(f"  commit {(body.get('code') or {}).get('commit', '?')}")
        print(f"  gate   {(body.get('gate') or {}).get('verdict', 'n/a')}")
    return problems


def self_test() -> int:
    """Control, then two plants. A check that has never failed is unverified."""
    results = []

    def record(name, passed, detail):
        results.append(passed)
        print("  %-4s %-52s %s" % ("PASS" if passed else "FAIL", name, detail))

    source = MODELS_DIR
    if not (source / PROVENANCE).is_file():
        print(f"  cannot self-test: no {PROVENANCE} in {source}. Run "
              f"finalize_models.py first.")
        return 2

    work = Path(tempfile.mkdtemp(prefix="promo-prov-"))
    try:
        good = work / "good"
        shutil.copytree(source, good)
        problems = check(good, verbose=False)
        record("control: a freshly finalised models dir is consistent",
               not problems, "clean" if not problems else str(problems[:2]))

        # PLANT 1: the stale provenance. A model is replaced by different
        # weights - exactly what Copy-Item does on a promotion - while the
        # provenance stays as it was.
        stale = work / "stale"
        shutil.copytree(source, stale)
        victim = stale / "spread.json"
        body = victim.read_bytes()
        victim.write_bytes(body + b"\n")          # same model, different bytes
        problems = check(stale, verbose=False)
        caught = any("DIFFERENT weights" in p for p in problems)
        record("PLANT: weights changed, provenance not -> refused",
               caught,
               next((p for p in problems if "DIFFERENT" in p), "NOT CAUGHT")
               [:90])

        # PLANT 2: no provenance at all, which is what a promotion looked like
        # before this existed.
        bare = work / "bare"
        shutil.copytree(source, bare)
        (bare / PROVENANCE).unlink()
        problems = check(bare, verbose=False)
        record("PLANT: no provenance at all -> refused",
               any("nothing to" in p for p in problems),
               problems[0][:90] if problems else "NOT CAUGHT")

        # And the guard against the guard: the untouched copy must still pass
        # after all that, or the harness is what produced the reds.
        problems = check(good, verbose=False)
        record("PLANT reverted: the untouched copy still passes",
               not problems, "clean")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    passed = sum(1 for r in results if r)
    print()
    print(f"  {passed} of {len(results)} checks passed")
    return 0 if passed == len(results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    print(f"PROMOTION PROVENANCE CHECK: {args.dir}")
    problems = check(args.dir)
    if problems:
        print()
        print("  REFUSING: the provenance does not describe these weights.")
        for problem in problems:
            print(f"    - {problem}")
        print()
        print("  Nothing has been committed. A promotion must carry a "
              "provenance")
        print("  file written for the weights beside it - see "
              "continuous_retrain.write_candidate_provenance.")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:                                  # noqa: BLE001
        print(f"\nCRASHED before a verdict: {type(error).__name__}: {error}")
        raise SystemExit(2)
