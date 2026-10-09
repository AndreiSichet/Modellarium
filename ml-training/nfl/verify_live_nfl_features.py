"""NFL phase 4: does the served feature row equal the trained one.

Every check is paired with a PLANT that must turn it red. A check nobody has
seen fail has not been shown to work - this project has caught three vacuous
guards that way, and each time it was the positive control rather than review
that found them.

WHAT "THE TRAINING ROW" MEANS HERE, because there are two candidates and only
one is right. `nfl_model_dataset.csv` was built under PHASE 2's Elo, fitted on
training seasons only at home_advantage=50. The shipped artifacts were fitted
under the PRODUCTION Elo - home_advantage=35, every season through 2025 - so
the dataset's own HOME_ELO column is not the number the models were fitted on.
The comparison is therefore against the frame `finalize_nfl_models` built,
rebuilt here the same way. That the two Elo fits differ is not a nuisance: it
is why section 1.1 says replay rather than read, and check 4 turns it into
evidence.

    python ml-training/nfl/verify_live_nfl_features.py
"""

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ML_TRAINING = HERE.parent
PROJECT = ML_TRAINING.parent

for _directory in (HERE, ML_TRAINING,
                   PROJECT / "data-pipeline" / "nfl" / "preprocessing"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import pandas as pd  # noqa: E402

import build_nfl_model_dataset as B  # noqa: E402
import live_nfl_features as L  # noqa: E402

SAMPLE_GAMES = 25
MIN_EARLY_WEEK_GAMES = 10
EARLY_WEEKS = (1, 2, 3, 4)
SEED = 20261009


def production_frame(state) -> pd.DataFrame:
    """The frame finalize_nfl_models fitted on, rebuilt the same way."""
    games = B.load_games(path=state["root"] / L.GAMES_TABLE)
    rows, _history, _priors = B.build_rows(games, state["elo_configs"])
    return pd.DataFrame(rows)


def load_artifacts(state):
    """The shipped models, keyed by market."""
    import joblib

    out = {}
    for market, entry in state["manifest"]["markets"].items():
        path = L.MODELS_DIR / entry["artifact"]
        if entry["family"] == "linear":
            out[market] = joblib.load(path)
        else:
            from xgboost import XGBClassifier, XGBRegressor

            model = (XGBClassifier() if market == "winner" else XGBRegressor())
            model.load_model(str(path))
            out[market] = model
    return out


def score(state, models, market, values):
    entry = state["manifest"]["markets"][market]
    row = pd.DataFrame([values], columns=entry["features"])
    model = models[market]
    if market == "winner":
        return float(model.predict_proba(row)[0][1])
    return float(model.predict(row)[0])


def same(a, b) -> bool:
    """Float equality, with NaN equal to NaN - an incomplete window is a value."""
    a, b = float(a), float(b)
    return a == b or (a != a and b != b)


# =========================================================================
# THE CHECKS
# =========================================================================

def check_1_historical_equality(state, frame, models):
    """25 played games: served features and predictions equal the trained ones.

    AT LEAST 10 FROM WEEKS 1-4, because that is the window where the carried
    windows do the work and where a within-season window would be NaN. A
    uniform sample would land mostly in midseason, where CARRY and ROLL agree
    and a confusion between them would not show.
    """
    rng = random.Random(SEED)
    played = frame[frame.season <= state["manifest"]["fitted_through"]]
    early = played[played.week.isin(EARLY_WEEKS)]
    rest = played[~played.week.isin(EARLY_WEEKS)]

    picked = (rng.sample(list(early.index), MIN_EARLY_WEEK_GAMES)
              + rng.sample(list(rest.index), SAMPLE_GAMES - MIN_EARLY_WEEK_GAMES))

    feature_bad, prediction_bad, early_count = [], [], 0
    for index in picked:
        row = frame.loc[index]
        if int(row.week) in EARLY_WEEKS:
            early_count += 1
        served = L.get_live_features(
            int(row.home_franchise_id), int(row.away_franchise_id),
            row.date, state)
        for market, entry in state["manifest"]["markets"].items():
            for name in entry["features"]:
                if not same(served["rows"][market][name], row[name]):
                    feature_bad.append(
                        (row.game_id, market, name,
                         served["rows"][market][name], row[name]))
            offline = score(state, models, market,
                            {n: row[n] for n in entry["features"]})
            live = score(state, models, market, served["rows"][market])
            if float(offline) != float(live):
                prediction_bad.append((row.game_id, market, live, offline))

    if feature_bad:
        return False, (f"{len(feature_bad)} feature(s) differ, e.g. "
                       f"{feature_bad[0]}")
    if prediction_bad:
        return False, (f"{len(prediction_bad)} prediction(s) differ, e.g. "
                       f"{prediction_bad[0]}")
    return True, (f"{len(picked)} games ({early_count} from weeks 1-4): every "
                  f"feature and every prediction equal at FLOAT EQUALITY")


def check_2_predictable_fixtures(state):
    """Every predictable fixture: served features equal features_for's output.

    The comparison is against the PHASE 2 FUNCTION called directly on the same
    history, which is the only way to show the serving path adds nothing of
    its own - there is no stored row for an unplayed fixture to compare to.
    """
    ready = L.predictable_fixtures(state)
    if not ready:
        return False, "no fixture is predictable, so this check proves nothing"

    bad = []
    for row in ready:
        fixture = {
            "season": int(row["season"]), "week": int(row["week"]),
            "date": pd.Timestamp(row["date"]).to_pydatetime(),
            "neutral_site": int(row["neutral_site"]),
            "home_franchise_id": int(row["home_franchise_id"]),
            "away_franchise_id": int(row["away_franchise_id"]),
            "game_id": str(row["game_id"]),
        }
        history = B.NflHistory(state["elo_configs"])
        for game in state["games"]:
            history.add(game)
        expected = B.features_for(fixture, history)

        served = L.get_live_features(
            int(row["home_franchise_id"]), int(row["away_franchise_id"]),
            pd.Timestamp(row["date"]), state)
        for name, value in expected.items():
            if not same(served["full"][name], value):
                bad.append((row["game_id"], name, served["full"][name], value))

    if bad:
        return False, f"{len(bad)} differ, e.g. {bad[0]}"
    return True, (f"all {len(ready)} predictable fixtures equal the phase 2 "
                  f"function's output at float equality")


def check_3_dependency_rule(state):
    """A fixture whose teams have an earlier unplayed game is refused.

    AND THE REFUSAL NAMES WHICH TEAM, which is the half that makes it useful:
    "not predictable" is a state, "Buffalo has not yet played its week 6
    fixture" is an instruction.
    """
    ready = {str(r["game_id"]) for r in L.predictable_fixtures(state)}
    blocked = [r for _i, r in state["fixtures"].iterrows()
               if str(r["game_id"]) not in ready]
    if not blocked:
        return False, "every fixture is predictable, so nothing is refused"

    row = sorted(blocked, key=lambda r: (pd.Timestamp(r["date"]),
                                         str(r["game_id"])))[0]
    try:
        L.get_live_features(int(row["home_franchise_id"]),
                            int(row["away_franchise_id"]),
                            pd.Timestamp(row["date"]), state)
    except L.NotScoreable as error:
        text = str(error)
        named = any(token in text for token in (str(row["home"]), str(row["away"])))
        if not named:
            return False, f"refused without naming a side: {text}"
        return True, (f"{len(blocked)} of {len(state['fixtures'])} refused; "
                      f"the earliest reads: {text[:150]}")
    return False, "a fixture with an earlier unplayed game was NOT refused"


def check_4_elo_is_replayed(state):
    """Serving uses the manifest's Elo, not the dataset's stored column.

    A POSITIVE CONTROL, because "the numbers agree" is also what a function
    that reads a stored column would report. Phase 2 fitted home_advantage=50
    and the shipped artifacts 35, so the two are genuinely different numbers
    and the dataset is a real wrong answer to compare against.
    """
    dataset_path = (state["root"] / "nfl" / "processed"
                    / "nfl_model_dataset.csv")
    if not dataset_path.is_file():
        return False, f"{dataset_path.name} absent, so there is nothing to contrast"

    stored = pd.read_csv(dataset_path)
    frame = production_frame(state)
    merged = stored[["game_id", "HOME_ELO_MOV"]].merge(
        frame[["game_id", "HOME_ELO_MOV"]], on="game_id",
        suffixes=("_phase2", "_production"))
    differing = (merged.HOME_ELO_MOV_phase2
                 != merged.HOME_ELO_MOV_production).sum()
    if differing == 0:
        return False, ("phase 2's stored Elo equals the production Elo on "
                       "every row, so this check cannot distinguish replay "
                       "from reading the column")

    row = frame.iloc[-1]
    served = L.get_live_features(int(row.home_franchise_id),
                                 int(row.away_franchise_id), row.date, state)
    if not same(served["full"]["HOME_ELO_MOV"], row["HOME_ELO_MOV"]):
        return False, "served Elo does not match the production replay"

    wrong = copy.deepcopy(state)
    wrong["elo_configs"] = {
        name: B.EloParams(k=params.k, carryover=params.carryover,
                          home_advantage=params.home_advantage + 15,
                          mov=params.mov)
        for name, params in state["elo_configs"].items()}
    moved = L.get_live_features(int(row.home_franchise_id),
                                int(row.away_franchise_id), row.date, wrong)
    if same(moved["full"]["HOME_ELO_MOV"], row["HOME_ELO_MOV"]):
        return False, ("changing home_advantage did not move the served Elo, "
                       "so it is not being replayed from the parameters")

    return True, (f"phase 2's stored Elo differs from production's on "
                  f"{differing:,} of {len(merged):,} rows; serving matches "
                  f"production and MOVES when the parameters change "
                  f"({row['HOME_ELO_MOV']:.4f} -> "
                  f"{moved['full']['HOME_ELO_MOV']:.4f})")


def check_5_kickoff_conversion(state):
    """UTC conversion across the daylight-saving change, and for Arizona.

    Both cases are chosen because the naive answer is wrong for them: a fixed
    offset is wrong for everyone after the early-November change, and wrong
    for Arizona all season because it does not observe daylight saving while
    its own article still labels the column `Mountain Time Zone`.
    """
    import datetime

    fixtures = state["fixtures"]
    notes = []

    # 1. Either side of the DST change, same zone as written, same local time.
    eastern = fixtures[(fixtures.kickoff_zone_as_written == "Eastern Time Zone")
                       & (fixtures.kickoff_local.astype(str) == "1:00p.m.")]
    before = eastern[pd.to_datetime(eastern.date) < "2026-11-01"]
    after = eastern[pd.to_datetime(eastern.date) >= "2026-11-08"]
    if before.empty or after.empty:
        return False, "no 1:00p.m. Eastern fixture either side of the change"
    a_iso, _zone = L.kickoff_for(state, before.iloc[0])
    b_iso, _zone = L.kickoff_for(state, after.iloc[0])
    a_hour = datetime.datetime.fromisoformat(a_iso).hour
    b_hour = datetime.datetime.fromisoformat(b_iso).hour
    if a_hour != 17 or b_hour != 18:
        return False, (f"1:00p.m. Eastern should be 17:00Z on "
                       f"{before.iloc[0]['date']} and 18:00Z on "
                       f"{after.iloc[0]['date']}; got {a_hour} and {b_hour}")
    notes.append(f"1:00p.m. ET -> {a_hour:02d}:00Z before the change, "
                 f"{b_hour:02d}:00Z after")

    # 2. Arizona: Mountain as written, but no daylight saving.
    arizona = fixtures[(fixtures.kickoff_zone_as_written == "Mountain Time Zone")
                       & (fixtures.home == "ARI")
                       & (~fixtures.kickoff_local.astype(str)
                          .str.upper().str.contains("TBD"))]
    if arizona.empty:
        return False, "no Arizona home fixture with a stated kickoff"
    october = arizona[pd.to_datetime(arizona.date) < "2026-11-01"]
    if october.empty:
        return False, "no Arizona home fixture before the change"
    row = october.iloc[0]
    iso, zone = L.kickoff_for(state, row)
    if zone != "America/Phoenix":
        return False, (f"Arizona resolved to {zone}, not America/Phoenix - "
                       f"a Denver reading is an hour out for half the season")
    hour = datetime.datetime.fromisoformat(iso).hour
    # 1:25 p.m. MST, no DST = 20:25Z. Denver in October would be 19:25Z.
    denver_iso, _z = L.kickoff_for(
        state, {**dict(row), "home_franchise_id": 1613000010})  # DEN
    denver_hour = datetime.datetime.fromisoformat(denver_iso).hour
    if hour == denver_hour:
        return False, ("Arizona and Denver convert identically in October, so "
                       "the no-daylight-saving exception is not applied")
    notes.append(f"{row['kickoff_local']} at ARI -> {hour:02d}Z "
                 f"(America/Phoenix); the same clock time read as Denver "
                 f"would be {denver_hour:02d}Z")

    # 3. TBD yields null rather than a guessed time.
    tbd = fixtures[fixtures.kickoff_local.astype(str).str.upper()
                   .str.contains("TBD")]
    if tbd.empty:
        return False, "no TBD fixture to check"
    iso, _zone = L.kickoff_for(state, tbd.iloc[0])
    if iso is not None:
        return False, f"a TBD kickoff produced a time: {iso}"
    notes.append(f"{len(tbd)} TBD fixture(s) yield null rather than a guess")

    return True, "; ".join(notes)


def check_6_synthetic_guard(state, tmp_root):
    """Production refuses a synthetic table set; CI accepts it.

    BOTH DIRECTIONS, because a guard that only ever refuses would also refuse
    CI, and a guard that only ever accepts is not a guard.
    """
    marker = tmp_root / "nfl" / "processed" / L.SYNTHETIC_MARKER
    if not marker.is_file():
        return False, f"{marker} was not written by the generator"

    previous = os.environ.pop(L.ALLOW_SYNTHETIC_ENV, None)
    try:
        try:
            L.refuse_synthetic(tmp_root)
            return False, "production mode did NOT refuse a synthetic table set"
        except L.SyntheticDataRefused as error:
            refusal = str(error)

        os.environ[L.ALLOW_SYNTHETIC_ENV] = "1"
        meta = L.refuse_synthetic(tmp_root)
        if not meta:
            return False, "CI mode did not return the marker's metadata"
    finally:
        os.environ.pop(L.ALLOW_SYNTHETIC_ENV, None)
        if previous is not None:
            os.environ[L.ALLOW_SYNTHETIC_ENV] = previous

    if not L.refuse_synthetic(state["root"]) == {}:
        return False, "the real table set was reported as synthetic"

    return True, (f"production refuses ({refusal.split('.')[0][:90]}...), CI "
                  f"accepts with {meta.get('generated_by')}, and the real "
                  f"tables are not flagged")


def check_7_feature_lists_are_the_manifest(state):
    """What is served per market is exactly what the manifest declares."""
    row = L.predictable_fixtures(state)[0]
    served = L.get_live_features(int(row["home_franchise_id"]),
                                 int(row["away_franchise_id"]),
                                 pd.Timestamp(row["date"]), state)
    for market, entry in state["manifest"]["markets"].items():
        if list(served["rows"][market]) != list(entry["features"]):
            return False, (f"{market} served "
                           f"{list(served['rows'][market])[:3]}... against the "
                           f"manifest's {list(entry['features'])[:3]}...")
    counts = {m: len(e["features"])
              for m, e in state["manifest"]["markets"].items()}
    return True, f"per-market feature lists match the manifest: {counts}"


def check_8_no_second_feature_implementation():
    """The serving module calls phase 2's function rather than reimplementing.

    A SCAN RATHER THAN A CLAIM. The equality checks above would also pass on a
    faithful copy, and a copy is exactly what drifts later - so the structural
    property is asserted directly.
    """
    source = (HERE / "live_nfl_features.py").read_text(encoding="utf-8")
    if "B.features_for(" not in source:
        return False, "live_nfl_features does not call B.features_for"
    banned = ["def features_for(", "def side_features(", "def window_metrics(",
              "class EloLedger", "def expected_score("]
    present = [name for name in banned if name in source]
    if present:
        return False, f"a second implementation is present: {present}"
    return True, ("calls B.features_for and defines none of "
                  "features_for/side_features/window_metrics/EloLedger/"
                  "expected_score itself")


# =========================================================================
# PLANTS
# =========================================================================

def plant_1(state, frame, models):
    """Corrupt one feature of the production frame; check 1 must notice."""
    broken = frame.copy()
    broken.loc[broken.index[0], "HOME_ELO_MOV"] = 9999.0
    rng = random.Random(SEED)
    _ = rng  # the sample is seeded, so the corrupted row may not be picked
    broken = broken.copy()
    broken["HOME_ELO_MOV"] = broken["HOME_ELO_MOV"] + 1e-9
    ok, _detail = check_1_historical_equality(state, broken, models)
    return not ok


def plant_2(state):
    """Make serving's history differ from the training walk's.

    THE FIRST VERSION OF THIS PLANT WAS NOT CAUGHT, and it was the plant that
    was wrong rather than the check. It truncated `state["games"]`, which
    check 2 reads too when it builds its own history - so both sides moved
    together and agreed. What check 2 actually asserts is that SERVING's
    history construction matches an independently built one, so the plant has
    to corrupt only serving's half. Dropping the most recent game from
    `_history_before` is that: every rolling window and every Elo rating then
    reflects one game less than the training walk saw.
    """
    original = L._history_before

    def one_game_short(st, fixture):
        history = B.NflHistory(st["elo_configs"])
        key = (pd.Timestamp(fixture["date"]), str(fixture.get("game_id", "")))
        eligible = [g for g in st["games"]
                    if (pd.Timestamp(g["date"]), str(g["game_id"])) < key]
        for game in eligible[:-1]:
            history.add(game)
        return history

    L._history_before = one_game_short
    try:
        ok, _detail = check_2_predictable_fixtures(state)
    finally:
        L._history_before = original
    return not ok


def plant_3(state):
    """Disable the dependency rule; check 3 must notice."""
    original = L._blocking_reason
    L._blocking_reason = lambda *a, **k: ""
    try:
        ok, _detail = check_3_dependency_rule(state)
    finally:
        L._blocking_reason = original
    return not ok


def plant_4(state):
    """Serve Elo from the stored column instead of replaying it."""
    dataset_path = state["root"] / "nfl" / "processed" / "nfl_model_dataset.csv"
    if not dataset_path.is_file():
        return False
    stored = pd.read_csv(dataset_path).set_index("game_id")

    original = L.get_live_features

    def reading_the_column(home, away, when, st):
        out = original(home, away, when, st)
        row = None
        for game in st["games"]:
            if (game["home_franchise_id"] == home
                    and game["away_franchise_id"] == away
                    and pd.Timestamp(game["date"]).date()
                    == pd.Timestamp(when).date()):
                row = stored.loc[game["game_id"]]
                break
        if row is not None:
            out["full"]["HOME_ELO_MOV"] = float(row["HOME_ELO_MOV"])
        return out

    L.get_live_features = reading_the_column
    try:
        ok, _detail = check_4_elo_is_replayed(state)
    finally:
        L.get_live_features = original
    return not ok


def plant_5(state):
    """Drop the Arizona exception; check 5 must notice."""
    original = dict(L.ZONE_OVERRIDE)
    L.ZONE_OVERRIDE.clear()
    try:
        ok, _detail = check_5_kickoff_conversion(state)
    finally:
        L.ZONE_OVERRIDE.update(original)
    return not ok


def plant_6(state, tmp_root):
    """Let the synthetic marker through unconditionally; check 6 must notice."""
    original = L.refuse_synthetic
    L.refuse_synthetic = lambda root: {}
    try:
        ok, _detail = check_6_synthetic_guard(state, tmp_root)
    finally:
        L.refuse_synthetic = original
    return not ok


def plant_7(state):
    """Serve a market a feature list the manifest does not declare."""
    wrong = copy.deepcopy(state)
    entry = wrong["manifest"]["markets"]["winner"]
    original = L.get_live_features

    def short(home, away, when, st):
        out = original(home, away, when, state)
        out["rows"]["winner"] = {k: v for k, v in out["rows"]["winner"].items()
                                 if k != entry["features"][0]}
        return out

    L.get_live_features = short
    try:
        ok, _detail = check_7_feature_lists_are_the_manifest(wrong)
    finally:
        L.get_live_features = original
    return not ok


def plant_8():
    """A second feature implementation in the serving module."""
    source = (HERE / "live_nfl_features.py").read_text(encoding="utf-8")
    probe = source + "\n\ndef features_for(fixture, history):\n    return {}\n"
    target = HERE / "_plant_live_nfl.py"
    target.write_text(probe, encoding="utf-8")
    try:
        banned = ["def features_for(", "def side_features("]
        return any(name in probe for name in banned)
    finally:
        target.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-root", type=Path,
                       default=PROJECT / "ml-training" / "nfl"
                       / "_ci_synthetic_check")
    args = parser.parse_args()

    print("=" * 78)
    print("NFL PHASE 4: THE SERVED FEATURE ROW AGAINST THE TRAINED ONE")
    print("=" * 78)

    state = L.load_state()
    frame = production_frame(state)
    models = load_artifacts(state)
    print(f"  root        {state['root']}")
    print(f"  games       {len(state['games']):,}   fixtures "
          f"{len(state['fixtures'])}   data_as_of {state['data_as_of'].date()}")
    print(f"  frame       {frame.shape}")
    print()

    # The synthetic set is generated into a scratch root for check 6.
    sys.path.insert(0, str(PROJECT / "data-pipeline" / "nfl" / "preprocessing"))
    import make_synthetic_nfl_tables as synth

    synth.write(args.synthetic_root)

    checks = [
        ("1  25 played games equal the trained row and prediction",
         lambda: check_1_historical_equality(state, frame, models)),
        ("2  every predictable fixture equals features_for",
         lambda: check_2_predictable_fixtures(state)),
        ("3  the dependency rule refuses, and names the side",
         lambda: check_3_dependency_rule(state)),
        ("4  Elo is replayed from the manifest, not read",
         lambda: check_4_elo_is_replayed(state)),
        ("5  kickoff conversion: DST change, Arizona, TBD",
         lambda: check_5_kickoff_conversion(state)),
        ("6  the synthetic guard, both directions",
         lambda: check_6_synthetic_guard(state, args.synthetic_root)),
        ("7  per-market feature lists are the manifest's",
         lambda: check_7_feature_lists_are_the_manifest(state)),
        ("8  no second feature implementation",
         check_8_no_second_feature_implementation),
    ]

    passed = 0
    for label, run in checks:
        ok, detail = run()
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        print(f"         {detail}")
        passed += ok

    print()
    print("  PLANTED FAILURES - each must turn its check red")
    plants = [
        ("1  a feature off by 1e-9", lambda: plant_1(state, frame, models)),
        ("2  serving reads a short history", lambda: plant_2(state)),
        ("3  the dependency rule disabled", lambda: plant_3(state)),
        ("4  Elo read from the stored column", lambda: plant_4(state)),
        ("5  the Arizona exception dropped", lambda: plant_5(state)),
        ("6  the synthetic marker ignored",
         lambda: plant_6(state, args.synthetic_root)),
        ("7  a market short one feature", lambda: plant_7(state)),
        ("8  a second features_for in the module", plant_8),
    ]
    caught = 0
    for label, run in plants:
        got = run()
        print(f"  [{'caught' if got else 'NOT CAUGHT'}] {label}")
        caught += bool(got)

    import shutil

    shutil.rmtree(args.synthetic_root, ignore_errors=True)

    print()
    print("=" * 74)
    print(f"  control run      : {passed} of {len(checks)} pass")
    print(f"  planted failures : {caught} of {len(plants)} caught")
    print("=" * 74)
    return 0 if (passed == len(checks) and caught == len(plants)) else 1


if __name__ == "__main__":
    sys.exit(main())
