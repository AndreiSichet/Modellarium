r"""Every key `predictable_fixtures` produces reaches the wire.

WHY THIS EXISTS, AND IT IS NOT A HYPOTHETICAL. `blocked_by` was added to
`live_nhl_features.predictable_fixtures`, to the backend's
`InferenceScheduledGame`, to `ScheduledGameDto`, to V7's `game.blocked_by`
column and to the frontend's sentence builder. Every one of those was tested
and every test passed - and the field never left the inference service,
because `app.py`'s `schedule_nhl` builds its response model FIELD BY FIELD
and simply did not mention it. The producer was tested against itself and the
consumer was tested against a FIXTURE FILE WRITTEN BY HAND, so both ends were
green while the route between them carried nothing. The reader saw rows with
no explanation and no error anywhere.

THE LESSON IS ABOUT THE SHAPE OF THE TEST, not about the field. A hand-written
fixture can only ever prove that the consumer parses what its author believed
the producer sends. It cannot notice that the producer has stopped sending it,
and it cannot notice that the producer never sent it. So this check derives
the expected key set FROM THE PRODUCER and asserts the route carries it - the
one direction a fixture file can never cover.

HOW IT CHECKS. Two independent readings, because each misses what the other
catches:

  1. STATIC. The keys of the dict literal `predictable_fixtures` returns, the
     fields `NhlFixture` declares, and the keyword arguments `schedule_nhl`
     actually passes when it constructs one. A field declared but not passed
     is exactly the defect that shipped, and only the third of those three
     can see it.

  2. RUNTIME, when the served tables are on this machine. `predictable_fixtures`
     is called for real and its keys are compared with the static reading, so
     a key built dynamically (or one that stopped being produced) cannot hide
     behind the source. When the tables are absent this is REPORTED as not run
     rather than passed - an unrun check that reports itself green is the
     misleading green this project keeps catching.

RENAMES ARE DECLARED, NEVER INFERRED. The producer says `home_id` where the
wire says `home_team_id`. A checker that guessed at that would also "guess"
its way past a genuinely missing field, so the map below is explicit and an
unmapped producer key is a FAILURE. Adding a key to the producer therefore
forces a decision here: carry it, or say in one line why it stays internal.

Exit codes follow the project's contract: 0 verdict reached, 1 refused,
2 crashed before comparing.
"""
from __future__ import annotations

import argparse
import ast
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "inference-service" / "app.py"
PRODUCER = ROOT / "ml-training" / "nhl" / "live_nhl_features.py"

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str) -> None:
    print(f"  {'PASS' if passed else 'FAIL'} {name}  {detail}")
    RESULTS.append((name, passed, detail))


# A producer key whose wire name differs, with the reason it differs.
RENAMES = {
    # The wire says which ENTITY the id belongs to; inside the pipeline the
    # frame's own columns are HOME_ID/AWAY_ID and the row mirrors them.
    "home_id": "home_team_id",
    "away_id": "away_team_id",
}

# A producer key deliberately not on the wire needs its reason HERE, so that
# "not carried" is always a decision somebody wrote down.
INTERNAL: dict[str, str] = {}


class _Finder(ast.NodeVisitor):
    """The three source facts, read without importing anything."""

    def __init__(self) -> None:
        self.produced: set[str] = set()
        self.declared: set[str] = set()
        self.passed: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        if node.name == "predictable_fixtures":
            for inner in ast.walk(node):
                if isinstance(inner, ast.Dict):
                    keys = [k.value for k in inner.keys
                            if isinstance(k, ast.Constant)
                            and isinstance(k.value, str)]
                    # The ROW dict is the one carrying game_id; a smaller
                    # dict literal inside the function (a blocker, say) is
                    # not the row and must not widen the expectation.
                    if "game_id" in keys:
                        self.produced.update(keys)
        if node.name == "schedule_nhl":
            for inner in ast.walk(node):
                if (isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Name)
                        and inner.func.id == "NhlFixture"):
                    self.passed.update(
                        kw.arg for kw in inner.keywords if kw.arg)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        if node.name == "NhlFixture":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                        stmt.target, ast.Name):
                    self.declared.add(stmt.target.id)
        self.generic_visit(node)


def read_sources() -> _Finder:
    finder = _Finder()
    for path in (PRODUCER, APP):
        finder.visit(ast.parse(io.open(path, encoding="utf-8").read()))
    return finder


def wire_name(key: str) -> str:
    return RENAMES.get(key, key)


def static_checks(finder: _Finder) -> None:
    record("the producer's row keys were found in the source",
           len(finder.produced) >= 8, f"{len(finder.produced)} keys")
    record("NhlFixture's fields were found in the source",
           len(finder.declared) >= 8, f"{len(finder.declared)} fields")
    record("the route's NhlFixture(...) call was found",
           len(finder.passed) >= 8, f"{len(finder.passed)} arguments")

    unmapped = {k for k in finder.produced
                if wire_name(k) not in finder.declared
                and k not in INTERNAL}
    record("every key the producer emits is DECLARED on the wire model, "
           "or named internal with a reason",
           not unmapped, f"missing: {sorted(unmapped) or 'none'}")

    not_passed = {k for k in finder.produced
                  if k not in INTERNAL
                  and wire_name(k) in finder.declared
                  and wire_name(k) not in finder.passed}
    record("and the ROUTE actually passes each one - the half a fixture "
           "file can never check",
           not not_passed, f"declared but not passed: "
                           f"{sorted(not_passed) or 'none'}")

    record("blocked_by specifically, since this is the field that shipped "
           "missing",
           "blocked_by" in finder.declared and "blocked_by" in finder.passed,
           f"declared={'blocked_by' in finder.declared}, "
           f"passed={'blocked_by' in finder.passed}")

    stale = {k for k in RENAMES if k not in finder.produced}
    record("no rename in the map names a key the producer no longer emits",
           not stale, f"stale: {sorted(stale) or 'none'}")


def runtime_checks(finder: _Finder, data_dir: str | None) -> None:
    """The producer, run for real, against the static reading."""
    if not data_dir:
        record("RUNTIME the producer's real keys match the source reading",
               False,
               "NOT RUN - no served NHL tables given. Pass --data-dir; an "
               "unrun check must not read as a pass")
        return

    os.environ["DATA_DIR"] = data_dir
    for part in ("ml-training", "ml-training/nhl", "data-pipeline"):
        sys.path.insert(0, str(ROOT / part))
    import json

    import live_nhl_features as nhl

    manifest = json.loads(
        (ROOT / "ml-training" / "models_nhl" / "manifest.json")
        .read_text(encoding="utf-8"))
    state = nhl.load_state(manifest=manifest)
    if not state.get("served"):
        record("RUNTIME the producer's real keys match the source reading",
               False, "NOT RUN - that root serves no NHL tables")
        return

    rows = nhl.predictable_fixtures(state)
    record("RUNTIME the producer returned rows to read keys from",
           bool(rows), f"{len(rows)} fixtures")
    if not rows:
        return

    real = set().union(*(set(row) for row in rows))
    record("RUNTIME the real key set equals the one read from the source",
           real == finder.produced,
           f"only at runtime: {sorted(real - finder.produced) or 'none'}; "
           f"only in source: {sorted(finder.produced - real) or 'none'}")

    blocked = [row for row in rows if not row["predictable"]]
    record("RUNTIME a blocked fixture really carries a non-empty blocked_by",
           bool(blocked) and all(row["blocked_by"] for row in blocked[:50]),
           f"{len(blocked)} blocked; first="
           f"{blocked[0]['blocked_by'] if blocked else '-'}")

    both = sum(1 for row in blocked if len(row["blocked_by"]) == 2)
    record("RUNTIME most blocked fixtures are blocked on BOTH sides, which "
           "is why the list exists rather than one reason",
           not blocked or both > len(blocked) / 2,
           f"{both} of {len(blocked)} blocked on both sides")


def plant(finder: _Finder, which: str) -> None:
    """Make each check fail once, by removing what it exists to require."""
    if which == "route":
        finder.passed.discard("blocked_by")
    elif which == "model":
        finder.declared.discard("blocked_by")
    elif which == "rename":
        finder.produced.add("a_key_nothing_carries")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR"),
                        help="a served root or volume, for the runtime half")
    parser.add_argument("--plant", choices=("route", "model", "rename"),
                        help="break one thing and watch its check go red")
    args = parser.parse_args()

    print("THE NHL SCHEDULE'S KEYS REACH THE WIRE")
    print()
    if args.plant:
        print(f"  PLANTED: {args.plant} - checks are expected to FAIL")
        print()

    finder = read_sources()
    if args.plant:
        plant(finder, args.plant)

    static_checks(finder)
    print()
    runtime_checks(finder, args.data_dir)

    passed = sum(1 for _n, ok, _d in RESULTS if ok)
    print()
    print(f"  {passed} of {len(RESULTS)} checks passed")

    if args.plant:
        # A PLANT THAT CHANGES NOTHING IS A HARNESS BUG, not a pass.
        if passed == len(RESULTS):
            print(f"\n  PLANT '{args.plant}' left every check green - the "
                  f"harness is not testing what it claims")
            return 2
        return 0
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:                                  # noqa: BLE001
        print(f"\nCRASHED before a verdict: {type(error).__name__}: {error}")
        raise SystemExit(2)
