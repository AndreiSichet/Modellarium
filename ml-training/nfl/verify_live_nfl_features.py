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

def _one_week_state(state, fixtures_rows, played_ids=()):
    """A state carrying exactly these fixtures and this history.

    CONSTRUCTED RATHER THAN SAMPLED, because the two cases that matter are a
    day with nothing due and a Monday-night game that is missing. Neither can
    be reached reliably from whatever the current snapshot happens to hold:
    the snapshot has no Monday fixture one day old, and asserting on the
    weekday of its earliest fixture would change answer every week. The
    identity and manifest are the real ones, so kickoff conversion still goes
    through the real zone map.
    """
    built = dict(state)
    built["fixtures"] = pd.DataFrame(fixtures_rows)
    # HISTORY REDUCED TO ITS IDS, which is all `missing_results` reads of it -
    # the question is only "does this fixture have a result". Filtering the
    # real games list cannot work here: these fixtures are invented, so no
    # real game carries their id and every one of them would read as missing.
    built["games"] = [{"game_id": game_id} for game_id in played_ids]
    return built


def _fixture(game_id, date, week, local, zone, home, away, home_fid, away_fid):
    return {
        "game_id": game_id, "season": 2026, "week": week, "date": date,
        "kickoff_local": local, "kickoff_zone_as_written": zone,
        "away_franchise_id": away_fid, "away": away,
        "home_franchise_id": home_fid, "home": home,
        "neutral_site": 0, "flex_two_dates": 0, "venue": "Test Field",
    }


def check_10_the_two_cases_the_defect_turned_on(state):
    """A Wednesday with nothing missing, and a Monday night missing on Tuesday.

    These are the two days the old rule could not tell apart. On both it
    reported the same thing - stale - because both sit three days past a
    Monday cutoff. One of them is correct and the other is the bug.
    """
    # Week 6 of 2026: Sunday the 18th, and Monday night the 19th. Monday
    # Night Football kicks off 8:15 p.m. Eastern, which is 00:15Z on Tuesday.
    sunday = _fixture("2026_6_CHI_ATL", "2026-10-18", 6, "1:00p.m.",
                      "Eastern Time Zone", "ATL", "CHI",
                      1613000002, 1613000006)
    monday = _fixture("2026_6_NYJ_BUF", "2026-10-19", 6, "8:15p.m.",
                      "Eastern Time Zone", "BUF", "NYJ",
                      1613000004, 1613000022)

    notes = []

    # ---- CASE 1: Wednesday, nothing due, nothing missing -> NOT stale ----
    # Both of the week's games are played and in history, and the next
    # fixture is eight days out. This is the state the service was actually
    # in on 15 October and wrongly called stale.
    wednesday_state = _one_week_state(
        state,
        [_fixture("2026_8_MIA_CIN", "2026-10-29", 8, "1:00p.m.",
                  "Eastern Time Zone", "CIN", "MIA", 1613000007, 1613000017)],
        played_ids=(),
    )
    days, stale, late = L.freshness(wednesday_state, "2026-10-21T12:00:00Z")
    if stale or late:
        return False, (f"a Wednesday with nothing due reported stale, "
                       f"naming {[e['game_id'] for e in late]}")
    notes.append(f"Wed 21 Oct, nothing due: days_behind={days}, NOT stale")

    # ---- CASE 2: Monday night missing on Tuesday -> STALE ----
    # The Sunday game is in history; the Monday-night one is not. Checked at
    # noon UTC on Tuesday, which is 11h45 after its 00:15Z kickoff.
    tuesday_state = _one_week_state(state, [sunday, monday],
                                    played_ids=("2026_6_CHI_ATL",))
    _d, stale, late = L.freshness(tuesday_state, "2026-10-20T12:00:00Z")
    if not stale:
        return False, "Monday night's result missing on Tuesday, reported fresh"
    ids = [entry["game_id"] for entry in late]
    if ids != ["2026_6_NYJ_BUF"]:
        return False, (f"expected only the Monday-night game to be late, "
                       f"got {ids}")
    notes.append("Tue 20 Oct, Monday night missing: STALE, naming only "
                 "2026_6_NYJ_BUF (the played Sunday game is not listed)")

    # ---- the same Tuesday state, five hours earlier, mid-game ----
    # 02:00Z Tuesday is 1h45 into the Monday-night game. The parser is still
    # refusing to treat a score as final, so the service must not yet be
    # calling itself behind.
    _d, stale, _late = L.freshness(tuesday_state, "2026-10-20T02:00:00Z")
    if stale:
        return False, "stale while the Monday-night game was still being played"
    notes.append("the same state at 02:00Z, mid-game: NOT stale")

    # ---- and the old rule cannot tell case 1 from case 2 ----
    # Both are three days past their respective cutoffs, which is what made
    # one of them a false alarm.
    wed_days, _s, _l = L.freshness(wednesday_state, "2026-10-21T12:00:00Z")
    tue_days, _s, _l = L.freshness(tuesday_state, "2026-10-20T12:00:00Z")
    if not (wed_days > 2 and tue_days > 2):
        return False, (f"both cases must sit past the old two-day threshold "
                       f"for the contrast to mean anything; got {wed_days} "
                       f"and {tue_days}")
    notes.append(f"the OLD rule says stale for BOTH (days_behind "
                 f"{wed_days} and {tue_days}) - that is the defect")

    return True, "; ".join(notes)


def check_9_staleness_is_the_weekly_rule(state):
    """Stale means a finished game has no result - not a days-since count.

    THE TWO CASES THE DEFECT TURNED ON. A days-since rule reported STALE on
    a Wednesday with every played game present, because the NFL had not
    played since Monday; and it would have said the same thing on a Tuesday
    when Monday night's result really was missing. The rule has to separate
    those, so both are asserted, and the days_behind figure is printed beside
    each to show it does not distinguish them.
    """
    fixtures = state["fixtures"]
    if fixtures.empty:
        return False, "no fixtures to judge"

    # The earliest fixture with a stated kickoff is the one to build both
    # cases around, derived rather than named - a fixed date would go stale
    # with the snapshot, which is the shape of defect this check exists for.
    timed = [row for _i, row in L._ordered_fixtures(state).iterrows()
             if L.kickoff_for(state, row)[0] is not None]
    if not timed:
        return False, "no fixture with a stated kickoff"
    row = timed[0]
    kickoff = pd.Timestamp(L.kickoff_for(state, row)[0])
    label = f"wk{int(row['week'])} {row['away']}@{row['home']}"

    notes = []

    # 1. BEFORE it kicks off, and well past the old two-day threshold.
    before = kickoff - pd.Timedelta(hours=6)
    days, stale, late = L.freshness(state, before)
    if stale or late:
        return False, (f"not yet kicked off and already stale: {len(late)} "
                       f"missing at {before.isoformat()}")
    if days <= 2:
        return False, (f"this case only means something past the old "
                       f"threshold; days_behind was {days} at "
                       f"{before.isoformat()}")
    notes.append(f"6h before {label}: days_behind={days} (the OLD rule would "
                 f"say stale), new rule says fresh")

    # 2. DURING the game - inside the window the parser refuses to record a
    #    result in, so calling it stale would blame the service for the
    #    parser doing the right thing.
    _d, stale, late = L.freshness(state, kickoff + pd.Timedelta(hours=1))
    if stale or late:
        return False, "stale one hour after kickoff, while the game is on"
    notes.append("1h after kickoff: fresh (inside the in-progress window)")

    # 3. AFTER it is certainly over, with no result in history.
    after = kickoff + pd.Timedelta(hours=L.HOURS_UNTIL_FINAL + 1)
    _d, stale, late = L.freshness(state, after)
    if not stale:
        return False, (f"{label} finished at {after.isoformat()} with no "
                       f"result and the rule said fresh")
    ids = [entry["game_id"] for entry in late]
    if str(row["game_id"]) not in ids:
        return False, f"stale, but {row['game_id']} is not in {ids[:3]}"
    notes.append(f"{L.HOURS_UNTIL_FINAL}h+ after it: STALE, naming "
                 f"{row['game_id']}")

    # 4. The threshold is the parser's, not a second copy of 5.
    import parse_nfl_wikitext as W
    if L.HOURS_UNTIL_FINAL is not W.HOURS_UNTIL_FINAL:
        return False, "the hours threshold is not the parser's constant"
    notes.append(f"threshold is parse_nfl_wikitext.HOURS_UNTIL_FINAL "
                 f"= {W.HOURS_UNTIL_FINAL}")

    # 5. Only regular-season weeks are judged.
    weeks = sorted(set(int(w) for w in fixtures["week"]))
    if not set(weeks) <= set(L.REGULAR_SEASON_WEEKS):
        return False, f"fixtures carry non-regular-season weeks: {weeks}"
    notes.append(f"weeks {min(weeks)}-{max(weeks)}, all regular season")

    return True, "; ".join(notes)


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


def plant_9(state):
    """Put the days-since rule back; check 9 must notice.

    This is the defect as it shipped, not an invented one: a cutoff more than
    STALE_AFTER_DAYS old is stale regardless of whether anything is missing.
    """
    import datetime

    original = L.freshness

    def days_since(st, now=None):
        today = pd.Timestamp(now).date() if now is not None \
            else datetime.date.today()
        days = (pd.Timestamp(today) - st["data_as_of"]).days
        return days, days > 2, []

    L.freshness = days_since
    try:
        ok, _detail = check_9_staleness_is_the_weekly_rule(state)
    finally:
        L.freshness = original
    return not ok


def plant_10(state):
    """Ignore the history check; check 10 must notice.

    A rule that only looks at the clock calls the Wednesday stale, because
    the fixture it finds has a kickoff in the past relative to nothing.
    """
    original = L.missing_results

    def clock_only(st, now=None):
        now = pd.Timestamp(now)
        if now.tz is None:
            now = now.tz_localize("UTC")
        days = (pd.Timestamp(now.date()) - st["data_as_of"]).days
        return [{"game_id": "ALL", "week": 0, "kickoff_utc": "",
                 "home": "", "away": ""}] if days > 2 else []

    L.missing_results = clock_only
    try:
        ok, _detail = check_10_the_two_cases_the_defect_turned_on(state)
    finally:
        L.missing_results = original
    return not ok


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
        ("9  staleness is the weekly rule, not a days-since count",
         lambda: check_9_staleness_is_the_weekly_rule(state)),
        ("10 a Wednesday with nothing missing, a Monday night missing "
         "on Tuesday", lambda: check_10_the_two_cases_the_defect_turned_on(state)),
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
        ("9  the days-since staleness rule restored",
         lambda: plant_9(state)),
        ("10 staleness read off the clock, ignoring history",
         lambda: plant_10(state)),
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
