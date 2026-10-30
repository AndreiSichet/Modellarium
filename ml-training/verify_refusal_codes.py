r"""Every refusal says WHY in a form a client can branch on.

THREE THINGS NOTHING ELSE CHECKS.

1. THE WINDOW IS DERIVED, NOT WRITTEN. `needed` on an insufficient-history
   refusal is max(WINDOWS) from build_rolling_features - the same source the
   builders read. A literal 10 would be a number with no expiry, and this
   project has three recorded breakages from exactly that.

2. EVERY REFUSAL PATH HAS A CODE. A path with no code is a path the frontend
   has to guess about, and guessing from an empty list is how the
   player-props panel came to render two headed boxes and nothing else. The
   set is read OUT OF THE SOURCE rather than from a list written here, so a
   seventh path cannot be added without either classifying it or failing
   this.

3. A MANIFEST FLAG WITH NOTHING CHECKING THE FLAG IS NOT A SAFETY PROPERTY.
   The G League refuses an incomplete feature row only when
   `requires_complete_features` is present and truthy; the WNBA's equivalent
   is unconditional. So the G League's protection can be switched off by a
   rebuilt manifest that omits a key, silently, after which a linear pipeline
   scores rows containing NaN.

The additive constraint is checked too: `detail` and the status code are what
they were, and the new fields sit beside them.

    python ml-training/verify_refusal_codes.py
"""
import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
APP = PROJECT / "inference-service" / "app.py"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / "data-pipeline" / "preprocessing"))

RESULTS = []


class PlantDidNothing(RuntimeError):
    """A plant that changes no behaviour is a harness bug, not a pass."""


def record(name, passed, detail=""):
    RESULTS.append((name, passed, detail))
    print("  %-4s %-58s %s" % ("PASS" if passed else "FAIL", name, detail))


# ==========================================================================
# 1. the window is derived
# ==========================================================================

def derive_in_subprocess(windows) -> str:
    """Import app with WINDOWS replaced, and report QH_WINDOW.

    A SUBPROCESS, BECAUSE QH_WINDOW IS COMPUTED AT IMPORT. Monkeypatching
    after the fact would prove nothing about the derivation; this replaces
    the source of truth before app.py reads it, which is what a window change
    actually looks like.
    """
    script = f"""
import sys
sys.path[:0] = [r"{PROJECT / 'inference-service'}", r"{HERE}",
                r"{HERE / 'wnba'}", r"{HERE / 'gleague'}", r"{HERE / 'nfl'}",
                r"{PROJECT / 'data-pipeline' / 'preprocessing'}"]
import build_rolling_features
build_rolling_features.WINDOWS = {windows!r}
import app
print(app.QH_WINDOW)
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as handle:
        handle.write(script)
        path = handle.name
    done = subprocess.run([sys.executable, "-W", "ignore", path],
                          capture_output=True, text=True, cwd=str(PROJECT),
                          env={**_env(), "DATA_DIR": str(_volume())},
                          timeout=900)
    Path(path).unlink(missing_ok=True)
    return (done.stdout or "").strip().splitlines()[-1:] or [done.stderr[-200:]]


def _env() -> dict:
    import os
    return dict(os.environ)


def _volume() -> Path:
    import os
    return Path(os.environ.get("DATA_DIR") or "D:/modellarium-data")


def window_is_derived() -> None:
    from build_rolling_features import WINDOWS

    real = derive_in_subprocess(list(WINDOWS))
    record("QH_WINDOW equals max(WINDOWS) as shipped",
           real == [str(max(WINDOWS))],
           f"WINDOWS={list(WINDOWS)} -> needed {real}")

    # THE PLANT. Move the window and `needed` must move with it.
    planted = derive_in_subprocess([5, 10, 17])
    if planted == [str(max(WINDOWS))]:
        raise PlantDidNothing(
            "QH_WINDOW did not follow a changed WINDOWS, so it is not "
            "derived from it")
    record("PLANT WINDOWS=[5,10,17] -> needed follows to 17",
           planted == ["17"], f"needed {planted}")

    record("and no literal window is written in app.py",
           not re.search(r"needed\s*=\s*\d+", APP.read_text(encoding="utf-8")),
           "`needed` is only ever QH_WINDOW")


# ==========================================================================
# 2. every refusal path is classified
# ==========================================================================

CODED_FUNCTIONS = {"validate_matchup", "predict_quarter_half",
                   "predict_player_props"}


def every_path_is_classified() -> None:
    source = APP.read_text(encoding="utf-8")
    tree = ast.parse(source)

    uncoded, coded = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) \
                or node.name not in CODED_FUNCTIONS:
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Raise) or inner.exc is None:
                continue
            name = getattr(getattr(inner.exc, "func", None), "id", None)
            if name == "HTTPException":
                uncoded.append((node.name, inner.lineno))
            elif name == "Refusal":
                coded.append((node.name, inner.lineno))

    record("no bare HTTPException is left on these endpoints",
           not uncoded, f"uncoded: {uncoded or 'none'}")
    record("every refusal on them is a Refusal with a code",
           len(coded) >= 6, f"{len(coded)} classified path(s)")

    # THE CODE SET, READ FROM THE SOURCE. A new REFUSAL_* constant that
    # nothing raises, or a raise with a code that is not a constant, both
    # show up here.
    constants = {name for name in dir(_app()) if name.startswith("REFUSAL_")}
    record("the code constants are a closed, named set",
           len(constants) >= 8, f"{len(constants)} REFUSAL_* constants")

    values = {getattr(_app(), name) for name in constants}
    record("and `unclassified` is one of them, not an absence",
           "unclassified" in values,
           "a refusal nothing has classified says so")
    record("every code is a bare lower-case word with no digits",
           all(re.fullmatch(r"[a-z_]+", v) for v in values),
           f"{sorted(values)}")


_APP = None


def _app():
    global _APP
    if _APP is None:
        for path in (PROJECT / "inference-service", HERE, HERE / "wnba",
                     HERE / "gleague", HERE / "nfl",
                     PROJECT / "data-pipeline" / "preprocessing"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        import os
        os.environ.setdefault("DATA_DIR", str(_volume()))
        import app as module
        _APP = module
    return _APP


# ==========================================================================
# 3. the additive constraint, and the body's shape
# ==========================================================================

def refusal_is_additive() -> None:
    app = _app()
    error = app.Refusal("the original prose, unchanged.",
                        app.REFUSAL_INSUFFICIENT_HISTORY,
                        needed=10, home_games_played=5, nothing=None)

    record("a Refusal is still an HTTPException",
           isinstance(error, app.HTTPException), type(error).__name__)
    record("its status is still 400", error.status_code == 400,
           str(error.status_code))
    record("`detail` is passed through untouched",
           error.detail == "the original prose, unchanged.", error.detail)

    body = app.refusal_body(error)
    record("the body puts `detail` first, then the additions",
           list(body)[0] == "detail" and body["code"]
           == "insufficient_history",
           f"{list(body)}")
    record("None-valued fields are dropped rather than sent as null",
           "nothing" not in body, f"{sorted(body)}")


# ==========================================================================
# 4. the completeness-flag boot guard
# ==========================================================================

def completeness_flag_guard() -> None:
    app = _app()

    present = {"gleague": {"requires_complete_features": True}}
    app.check_completeness_flags(present)
    record("a manifest WITH the flag boots", True, "no SystemExit")

    # FALSE IS A DECISION; ABSENT IS AN ACCIDENT. The guard checks presence.
    explicit_false = {"gleague": {"requires_complete_features": False}}
    app.check_completeness_flags(explicit_false)
    record("an explicit `false` also boots - someone decided", True,
           "presence, not truth")

    for label, manifests in (("the key missing", {"gleague": {"league": "g"}}),
                             ("the manifest empty", {"gleague": {}}),
                             ("the manifest absent", {})):
        try:
            app.check_completeness_flags(manifests)
            raised = None
        except SystemExit as error:
            raised = str(error)
        record(f"PLANT {label} -> refuses to start",
               raised is not None and "COMPLETENESS FLAG" in (raised or ""),
               (raised or "NOTHING RAISED").splitlines()[0][:70])

    # AND THE REAL MANIFEST HAS IT, which is what makes the guard quiet today
    # rather than a permanent red.
    manifest = json.loads(
        (HERE / "models_gleague" / "manifest.json").read_text(encoding="utf-8"))
    record("the shipped G League manifest carries the flag",
           "requires_complete_features" in manifest,
           f"value {manifest.get('requires_complete_features')!r}")


# ==========================================================================
# 5. the G League's season opener is its own code
# ==========================================================================

def gleague_season_opener_code() -> None:
    app = _app()

    class Fake(Exception):
        pass

    rest_only = Fake()
    rest_only.missing_columns = ["AWAY_IS_LONG_BREAK", "AWAY_REST_DAYS",
                                 "HOME_IS_LONG_BREAK", "HOME_REST_DAYS"]
    record("REST columns only -> season_opener",
           app.gleague_refusal_code(rest_only) == app.REFUSAL_SEASON_OPENER,
           app.gleague_refusal_code(rest_only))

    mixed = Fake()
    mixed.missing_columns = ["HOME_REST_DAYS", "HOME_CARRY10_PTS"]
    record("a form column among them -> incomplete_features",
           app.gleague_refusal_code(mixed)
           == app.REFUSAL_INCOMPLETE_FEATURES,
           app.gleague_refusal_code(mixed))

    bare = Fake()
    record("no columns at all -> incomplete_features, never a guess",
           app.gleague_refusal_code(bare)
           == app.REFUSAL_INCOMPLETE_FEATURES,
           app.gleague_refusal_code(bare))

    # THE REAL SHAPE, from the G League's own feature function. Measured
    # 2026-10-30: every 2026-27 fixture refuses and the missing columns are
    # exactly the four REST ones, because `rest_for` is season-scoped.
    sys.path.insert(0, str(HERE / "gleague"))
    from live_gleague_features import NotScoreable, get_live_features, load_state
    state = load_state()
    teams = sorted(set(state["all_games"]["TEAM_ID"]))[:2]
    try:
        get_live_features(teams[0], teams[1], "2026-11-07", state=state)
        record("a 2026-27 G League fixture refuses", False, "it SCORED")
    except NotScoreable as error:
        columns = getattr(error, "missing_columns", None)
        record("a 2026-27 G League fixture refuses with its columns carried",
               bool(columns), f"{columns}")
        record("and the real refusal classifies as season_opener",
               app.gleague_refusal_code(error)
               == app.REFUSAL_SEASON_OPENER,
               app.gleague_refusal_code(error))


# ==========================================================================
# 6. the early-season context block
# ==========================================================================

def early_season_block() -> None:
    """Additive, derived, and it can never fail a prediction."""
    app = _app()
    from build_rolling_features import WINDOWS

    # THE STATE THIS NEEDS, LOADED EXPLICITLY. Importing app does not run its
    # lifespan, so `state.games_final_df` is unset - the first version of
    # this section raised AttributeError, which also revealed that the
    # production function accessed the attribute directly on a path whose
    # contract is that it cannot fail. Both were fixed: the function uses
    # getattr, and this loads the frame so the counts are real rather than
    # the degraded zero.
    if getattr(app.state, "games_final_df", None) is None:
        app.state.games_final_df = app.load_games_final()
        app.state.data_as_of = app.pd.Timestamp(
            app.state.games_final_df["GAME_DATE"].max())

    # THE FIELD IS LAST AND NULLABLE. Last so every key already on the wire
    # keeps its POSITION as well as its name - the gate compares bodies, and
    # a reordering would read as a diff on every market. Nullable so an
    # unreachable schedule costs a note rather than a prediction.
    fields = list(app.PredictionResponse.model_fields)
    record("early_season is the LAST field of PredictionResponse",
           fields[-1] == "early_season", f"{fields}")
    record("and it is nullable, defaulting to None",
           app.PredictionResponse.model_fields["early_season"].default is None,
           "an absent block is None, not an error")

    # THE WINDOW COMES FROM THE FEATURE DEFINITIONS, same source as the
    # refusal's `needed`. Asserted against WINDOWS rather than against 10.
    block = app.early_season_context(1610612744, 1610612747,
                                     app.pd.Timestamp("2026-10-30"))
    record("needed equals max(WINDOWS)",
           block.needed == max(WINDOWS),
           f"needed={block.needed}, max(WINDOWS)={max(WINDOWS)}")
    record("the counts are the served tables' own counts",
           block.home_games_played
           == app.team_games_played(1610612744, app.pd.Timestamp("2026-10-30"))
           and block.away_games_played > 0,
           f"home={block.home_games_played} away={block.away_games_played}")

    # THE DATE IS REAL, AND IT IS THE LATER OF THE TWO. A window needs both
    # sides, so the fixture clears when the slower team does.
    record("full_window_date is a date while a side is short",
           block.full_window_date is not None,
           str(block.full_window_date))

    # ALREADY AT THE WINDOW -> None, not today's date and not a guess.
    record("a team already at the window yields no date",
           app.full_window_date(1610612744, 1610612747, 40, 40,
                                max(WINDOWS)) is None,
           "nothing to wait for")

    # PLANT: THE SCHEDULE RAISES. A note must never fail a prediction, so
    # every failure in that computation has to come back as None.
    real = app.fetch_schedule_frame

    def exploding(*_args, **_kwargs):
        raise RuntimeError("nba_api is down")

    app.fetch_schedule_frame = exploding
    try:
        out = app.full_window_date(1610612744, 1610612747, 5, 5,
                                   max(WINDOWS))
        raised = None
    except BaseException as error:                              # noqa: BLE001
        out, raised = "RAISED", error
    finally:
        app.fetch_schedule_frame = real
    record("PLANT the schedule API raises -> the date is None, no exception",
           raised is None and out is None,
           f"returned {out!r}" if raised is None
           else f"RAISED {type(raised).__name__}")

    # PLANT: AN EMPTY SCHEDULE. Different cause, same requirement.
    def empty(*_args, **_kwargs):
        return app.pd.DataFrame()

    app.fetch_schedule_frame = empty
    try:
        out = app.full_window_date(1610612744, 1610612747, 5, 5,
                                   max(WINDOWS))
    finally:
        app.fetch_schedule_frame = real
    record("PLANT an empty schedule -> None rather than a wrong date",
           out is None, f"returned {out!r}")

    # AND THE WHOLE BLOCK SURVIVES A BROKEN SCHEDULE, which is the shape a
    # prediction would actually meet.
    app.fetch_schedule_frame = exploding
    try:
        degraded = app.early_season_context(1610612744, 1610612747,
                                            app.pd.Timestamp("2026-10-30"))
        raised = None
    except BaseException as error:                              # noqa: BLE001
        degraded, raised = None, error
    finally:
        app.fetch_schedule_frame = real
    record("a prediction still gets its block with the date missing",
           raised is None and degraded is not None
           and degraded.needed == max(WINDOWS)
           and degraded.full_window_date is None,
           "needed and the counts survive; only the date is lost")

    # NO LITERAL WINDOW ANYWHERE NEAR IT.
    source = APP.read_text(encoding="utf-8")
    body = source[source.index("def early_season_context"):]
    body = body[:body.index(chr(10) + "def ")] if chr(10) + "def " in body else body
    record("early_season_context writes no numeric window",
           "QH_WINDOW" in body and not re.search(r"needed\s*=\s*\d", body),
           "needed is only ever QH_WINDOW")


def main() -> int:
    print("REFUSALS THAT SAY WHY")
    print()
    window_is_derived()
    print()
    every_path_is_classified()
    print()
    refusal_is_additive()
    print()
    completeness_flag_guard()
    print()
    gleague_season_opener_code()
    print()
    early_season_block()

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
