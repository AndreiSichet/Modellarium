"""NFL phase 2 verification: eight checks, each made to fail before it is
trusted, plus a control run of all eight with nothing planted.

A check that has never been seen to fail has not been verified - this project
has caught three vacuous guards that way (an unshifted rolling mean that passed
its own leak test, a positive control demanding every feature respond at one
probe, and a query-count guard blinded by a first-level cache). So every check
here is paired with a plant that must make it go red, and the plant is reported
beside the pass.

Run from the project root, after the builder:
    python ml-training/nfl/verify_nfl_features.py
"""
import argparse
import collections
import copy
import json
import math
import random
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PREP = REPO / "data-pipeline" / "nfl" / "preprocessing"
sys.path.insert(0, str(PREP))

import build_nfl_model_dataset as B      # noqa: E402

TOLERANCE = 0.0          # float EQUALITY: the same code on the same inputs
SAMPLE_GAMES = 50
SEED = 20261008


class Ctx:
    """Loaded once and shared; checks copy what they mutate."""

    def __init__(self):
        self.games = B.load_games()
        self.manifest = json.loads(
            B.MANIFEST_PATH.read_text(encoding="utf-8"))
        self.elo_configs = {
            name: B.EloParams(
                k=entry["k"], carryover=entry["carryover"],
                home_advantage=entry["home_advantage"],
                mov=entry["margin_of_victory_multiplier"])
            for name, entry in self.manifest["elo"].items()
        }
        # `float_precision="round_trip"` IS LOAD-BEARING, AND IT COST A FALSE
        # FAILURE. pandas' default CSV float converter is fast and NOT
        # round-trip exact: the file holds 24.333333333333332 and the default
        # parser returns 24.33333333333333, a different float. Check 4 compares
        # at float EQUALITY, so without this the whole dataset reads as wrong by
        # ~1e-13 while the pipeline is correct - the same shape as the float32
        # ULP noise that produced 313 phantom inversions in the calibration
        # study. The tolerance stays 0; the reader is what was wrong.
        self.frame = pd.read_csv(B.OUT_PATH, float_precision="round_trip")
        self.by_id = {row["game_id"]: row
                      for row in self.frame.to_dict("records")}
        self.rows, _, self.priors = B.build_rows(self.games, self.elo_configs)
        self.rows_by_id = {r["game_id"]: r for r in self.rows}
        self.train = [g for g in self.games
                      if B.split_role(g["season"]) == "train"]


# =====================================================================
# 1. the lag guard: a window is the mean over games STRICTLY BEFORE
# =====================================================================

def hand_window(games, franchise_id, target, name):
    """Recompute one window independently of `side_features`."""
    size, scope = B.WINDOWS[name]
    prior = [g for g in games
             if (g["date"], g["game_id"]) < (target["date"], target["game_id"])
             and franchise_id in (g["home_franchise_id"],
                                  g["away_franchise_id"])]
    if scope == "season":
        prior = [g for g in prior if g["season"] == target["season"]]
    if len(prior) < size:
        return None
    entries = []
    for g in prior[-size:]:
        home = g["home_franchise_id"] == franchise_id
        entries.append({
            "pf": g["home_points"] if home else g["away_points"],
            "pa": g["away_points"] if home else g["home_points"],
            "result": g["home_result"] if home else g["away_result"],
        })
    return B.window_metrics(entries)


def check_window_lag(ctx, rows=None):
    rows = rows or ctx.rows_by_id
    rng = random.Random(SEED)
    sample = rng.sample([g for g in ctx.games if g["season"] >= 2014],
                        SAMPLE_GAMES)
    worst, checked, mismatches = 0.0, 0, []
    for game in sample:
        row = rows[game["game_id"]]
        for prefix, key in (("HOME", "home_franchise_id"),
                            ("AWAY", "away_franchise_id")):
            for name in B.WINDOWS:
                expected = hand_window(ctx.games, game[key], game, name)
                for metric in B.METRICS:
                    got = row[f"{prefix}_{name}_{metric}"]
                    if expected is None:
                        if not (isinstance(got, float) and math.isnan(got)):
                            mismatches.append(
                                (game["game_id"], prefix, name, metric,
                                 "expected NaN", got))
                        continue
                    checked += 1
                    delta = abs(expected[metric] - got)
                    worst = max(worst, delta)
                    if delta > TOLERANCE:
                        mismatches.append((game["game_id"], prefix, name,
                                           metric, expected[metric], got))
    return (not mismatches,
            f"{checked} window values on {SAMPLE_GAMES} games hand-recomputed, "
            f"largest difference {worst:.3e}, {len(mismatches)} mismatch(es)"
            + (f" e.g. {mismatches[0]}" if mismatches else ""))


def plant_window_lag(ctx):
    """A leaky history: append the result BEFORE reading the features."""
    history = B.NflHistory(ctx.elo_configs)
    leaky = {}
    for game in ctx.games:
        history.add(game)                      # <- the leak
        leaky[game["game_id"]] = B.features_for(game, history)
    return lambda: check_window_lag(ctx, leaky)


# =====================================================================
# 2a. corrupting a LATER game must not move the target (+ positive control)
# =====================================================================

def rebuild_through(games, target, elo_configs):
    history = B.NflHistory(elo_configs)
    for game in games:
        if (game["date"], game["game_id"]) >= (target["date"],
                                               target["game_id"]):
            break
        history.add(game)
    return B.features_for(target, history)


def pick_probe(ctx, season=2018, min_prior=9):
    """A target deep enough into a season that every window is live."""
    counts = collections.Counter()
    for game in ctx.games:
        for key in ("home_franchise_id", "away_franchise_id"):
            counts[(game["season"], game[key])] += 1
    for game in ctx.games:
        if game["season"] != season:
            continue
        prior = ctx.priors[game["game_id"]]
        if all(in_season >= min_prior for _, in_season in prior.values()):
            return game
    raise RuntimeError("no probe found")


def corrupt(games, predicate, points=999):
    out = []
    for game in games:
        if predicate(game):
            game = dict(game)
            game["home_points"] = points
            game["away_points"] = 0
            game["home_result"] = "W"
            game["away_result"] = "L"
            game["stored_margin"] = points
        out.append(game)
    return out


def check_leak_future(ctx):
    target = pick_probe(ctx)
    base = rebuild_through(ctx.games, target, ctx.elo_configs)
    later = corrupt(ctx.games, lambda g: (g["date"], g["game_id"])
                    > (target["date"], target["game_id"]))
    after = rebuild_through(later, target, ctx.elo_configs)
    moved = [k for k in base if _differs(base[k], after[k])]
    # POSITIVE CONTROL. Without it the negative test passes identically on a
    # feature that reads nothing at all.
    earlier = corrupt(ctx.games, lambda g: (g["date"], g["game_id"])
                      < (target["date"], target["game_id"])
                      and g["season"] == target["season"])
    control = rebuild_through(earlier, target, ctx.elo_configs)
    responded = [k for k in base if _differs(base[k], control[k])]
    return (not moved and bool(responded),
            f"probe {target['game_id']}: corrupting every LATER game moved "
            f"{len(moved)} feature(s); the control corrupting EARLIER games "
            f"moved {len(responded)}")


def plant_leak_future(ctx):
    """Read the features one game too late, so a later game reaches them."""
    target = pick_probe(ctx)

    def leaky():
        base = _one_past(ctx.games, target, ctx.elo_configs)
        later = corrupt(ctx.games, lambda g: (g["date"], g["game_id"])
                        > (target["date"], target["game_id"]))
        after = _one_past(later, target, ctx.elo_configs)
        moved = [k for k in base if _differs(base[k], after[k])]
        return (not moved,
                f"planted one-game-late read: corrupting later games moved "
                f"{len(moved)} feature(s)")
    return leaky


def _one_past(games, target, elo_configs):
    """History that also holds the target teams' NEXT game - a deliberate leak.

    IT HAS TO BE ONE OF THE TARGET'S OWN TEAMS. A first version admitted simply
    the next game on the calendar, and the plant was NOT CAUGHT - correctly, for
    an uninteresting reason: that game involved two other franchises, so it
    touched neither side's windows nor either side's rating, and there was
    nothing for the check to see. A leak that leaks nothing is not a leak.
    """
    history = B.NflHistory(elo_configs)
    ordered = sorted(games, key=lambda g: (g["date"], g["game_id"]))
    index = next(i for i, g in enumerate(ordered)
                 if g["game_id"] == target["game_id"])
    sides = {target["home_franchise_id"], target["away_franchise_id"]}
    leaked = next((g for g in ordered[index + 1:]
                   if sides & {g["home_franchise_id"],
                               g["away_franchise_id"]}), None)
    if leaked is None:
        raise RuntimeError("no later game involves either side")
    for game in ordered[:index]:
        history.add(game)
    history.add(leaked)
    return B.features_for(target, history)


def _differs(a, b):
    if isinstance(a, float) and math.isnan(a):
        return not (isinstance(b, float) and math.isnan(b))
    if isinstance(b, float) and math.isnan(b):
        return True
    return abs(a - b) > TOLERANCE


# =====================================================================
# 2b. corrupting ONLY the previous season: ROLL still, CARRY moves
# =====================================================================

def pick_season_opener(ctx, season=2018):
    """A team's FIRST game of a season, where CARRY draws wholly on the last.

    PLACED DELIBERATELY. The G League phase learned that all five of its probes
    landed in the first season of the corpus, where a carried window has nothing
    to carry and behaves exactly like a within-season one - so five passes
    established no lookahead and said nothing about reach. A season opener in a
    later season is the one place the two kinds must differ.
    """
    for game in ctx.games:
        if game["season"] != season:
            continue
        prior = ctx.priors[game["game_id"]]
        if all(in_season == 0 and overall >= 8
               for overall, in_season in prior.values()):
            return game
    raise RuntimeError("no season-opening probe found")


def check_leak_previous_season(ctx):
    target = pick_season_opener(ctx)
    base = rebuild_through(ctx.games, target, ctx.elo_configs)
    bumped = corrupt(ctx.games,
                     lambda g: g["season"] == target["season"] - 1)
    after = rebuild_through(bumped, target, ctx.elo_configs)

    roll_moved, carry_moved, roll_nan = [], [], []
    for name, (_size, scope) in B.WINDOWS.items():
        for prefix in ("HOME", "AWAY"):
            for metric in B.METRICS:
                key = f"{prefix}_{name}_{metric}"
                if scope == "season":
                    if not (isinstance(base[key], float)
                            and math.isnan(base[key])):
                        roll_nan.append(key)
                    if _differs(base[key], after[key]):
                        roll_moved.append(key)
                elif _differs(base[key], after[key]):
                    carry_moved.append(key)
    ok = not roll_moved and not roll_nan and bool(carry_moved)
    return (ok,
            f"probe {target['game_id']} (a season opener): corrupting only "
            f"{target['season'] - 1} moved {len(carry_moved)} CARRY value(s) "
            f"and {len(roll_moved)} ROLL value(s); every ROLL value is NaN "
            f"there as it must be ({len(roll_nan)} that were not)")


def plant_leak_previous_season(ctx):
    """Make every window ignore the season boundary, so ROLL must go red.

    The seam is `window_pool`, NOT `WINDOWS`. A first version rewrote WINDOWS so
    every scope read "carry", and the plant was NOT CAUGHT - because the check
    reads the same table to decide which values are supposed to be
    within-season, so rewriting it removed the ROLL assertions instead of
    breaking them. The plant must change the implementation while leaving the
    check's idea of the contract alone.
    """
    def leaky():
        real = B.window_pool
        B.window_pool = lambda entries, in_season, scope: entries
        try:
            return check_leak_previous_season(ctx)
        finally:
            B.window_pool = real
    return leaky


# =====================================================================
# 3. the two Elo controls
# =====================================================================

def check_elo_posttraining(ctx):
    """Corrupting every game AFTER the training seasons must not move the fit."""
    base, _, _ = B.fit_elo(ctx.train, mov=False, who="control base")
    last_train = max(g["season"] for g in ctx.train)
    swapped = corrupt(ctx.games, lambda g: g["season"] > last_train)
    after_train = [g for g in swapped if B.split_role(g["season"]) == "train"]
    after, _, _ = B.fit_elo(after_train, mov=False, who="control after")
    return (base == after,
            f"{sum(1 for g in ctx.games if g['season'] > last_train)} "
            f"post-training game(s) corrupted; fit "
            f"{'unchanged' if base == after else f'MOVED {base} -> {after}'} "
            f"(K={base.k}, carryover={base.carryover:.3f}, "
            f"home={base.home_advantage})")


def check_elo_training(ctx):
    """Randomising HALF the training results must move the fitted K.

    Half rather than all: flipping every result is self-inverting - Elo learns
    exactly mirrored ratings, predicts the mirrored outcomes equally well, and
    the loss surface is identical. The WNBA phase's control failed on exactly
    that.
    """
    base, _, _ = B.fit_elo(ctx.train, mov=False, who="control base")
    rng = random.Random(SEED)
    scrambled = []
    for game in ctx.train:
        if rng.random() < 0.5:
            game = dict(game)
            home_win = rng.random() < 0.5
            game["home_points"] = 24 if home_win else 17
            game["away_points"] = 17 if home_win else 24
            game["home_result"] = "W" if home_win else "L"
            game["away_result"] = "L" if home_win else "W"
            game["stored_margin"] = game["home_points"] - game["away_points"]
        scrambled.append(game)
    after, _, _ = B.fit_elo(scrambled, mov=False, who="control scrambled")
    direction = ("slower, as less signal predicts" if after.k < base.k
                 else "FASTER, which less signal does not predict")
    return (after.k != base.k,
            f"half of {len(ctx.train)} training results randomised: "
            f"K {base.k} -> {after.k} ({direction}), "
            f"carryover {base.carryover:.3f} -> {after.carryover:.3f}, "
            f"home {base.home_advantage} -> {after.home_advantage}")


def plant_elo_posttraining(ctx):
    """Widen the fit to the served season, so corrupting it DOES move the fit.

    No unguarded code path is added for this: the served season is not the test
    season, so `refuse_test_rows` has nothing to say about it and the plant is
    simply a wider input - which is the mistake the check exists to detect.
    """
    def leaky():
        base, _, _ = B.fit_elo(ctx.train, mov=False, who="plant base")
        last_train = max(g["season"] for g in ctx.train)
        swapped = corrupt(ctx.games, lambda g: g["season"] > last_train)
        allowed = [g for g in swapped
                   if B.split_role(g["season"]) != "test"]
        after, _, _ = B.fit_elo(allowed, mov=False, who="plant wide")
        return (base == after,
                f"planted a wider fit (training PLUS the corrupted served "
                f"season): fit "
                f"{'unchanged' if base == after else f'MOVED to K={after.k}'}")
    return leaky


def plant_elo_training(ctx):
    """A fit that ignores its data cannot move when the data is destroyed.

    `elo_log_loss` is replaced by a constant, so the grid search has nothing to
    discriminate on and always returns its first point. If check 3b still
    passed against that, it would not be testing responsiveness at all.
    """
    def leaky():
        real = B.elo_log_loss
        B.elo_log_loss = lambda *a, **k: 0.5
        try:
            return check_elo_training(ctx)
        finally:
            B.elo_log_loss = real
    return leaky


# =====================================================================
# 4. the feature function reproduces the stored row
# =====================================================================

def check_feature_function(ctx):
    """THE section-1 proof: history rebuilt from scratch, float equality."""
    rng = random.Random(SEED + 1)
    sample = rng.sample(ctx.games, SAMPLE_GAMES)
    columns = B.feature_columns(list(ctx.elo_configs))
    worst, mismatches, compared = 0.0, [], 0
    for game in sample:
        fresh = rebuild_through(ctx.games, game, ctx.elo_configs)
        stored = ctx.by_id[game["game_id"]]
        for column in columns:
            got, want = stored[column], fresh[column]
            if isinstance(want, float) and math.isnan(want):
                if not pd.isna(got):
                    mismatches.append((game["game_id"], column, "NaN", got))
                continue
            if pd.isna(got):
                mismatches.append((game["game_id"], column, want, "NaN"))
                continue
            compared += 1
            delta = abs(float(got) - float(want))
            worst = max(worst, delta)
            if delta > TOLERANCE:
                mismatches.append((game["game_id"], column, want, got))
    return (not mismatches,
            f"{SAMPLE_GAMES} games, {compared} values, history rebuilt from "
            f"scratch for each: largest difference {worst:.3e}, "
            f"{len(mismatches)} mismatch(es)"
            + (f" e.g. {mismatches[0]}" if mismatches else ""))


def plant_feature_function(ctx):
    """Perturb one stored value the check will actually look at.

    The game is drawn from the SAME seeded sample the check uses - perturbing an
    arbitrary row would leave the plant passing for the uninteresting reason
    that the check never read it.
    """
    def leaky():
        rng = random.Random(SEED + 1)
        target = rng.sample(ctx.games, SAMPLE_GAMES)[0]["game_id"]
        saved = ctx.by_id[target]
        perturbed = saved.copy()
        perturbed["HOME_ELO"] = float(saved["HOME_ELO"]) + 1e-9
        ctx.by_id[target] = perturbed
        try:
            return check_feature_function(ctx)
        finally:
            ctx.by_id[target] = saved
    return leaky


# =====================================================================
# 5. the test-set firewall
# =====================================================================

def check_test_firewall(ctx):
    test_games = [g for g in ctx.games
                  if B.split_role(g["season"]) == "test"]
    one = [test_games[0]]
    try:
        B.fit_elo(one, mov=False, who="firewall check")
    except B.TestSeasonLeak as error:
        return True, (f"a single {one[0]['season']} row raises "
                      f"TestSeasonLeak: {str(error).splitlines()[0][:70]}")
    return False, "fit_elo accepted a test-season row"


def plant_test_firewall(ctx):
    """Neutralise `refuse_test_rows` itself, so the check must go red.

    This is what makes the firewall load-bearing rather than incidental: with
    the guard disabled the same call completes, so the guard is demonstrably
    the thing that stops it.
    """
    def leaky():
        real = B.refuse_test_rows
        B.refuse_test_rows = lambda games, who, **kwargs: games
        try:
            return check_test_firewall(ctx)
        finally:
            B.refuse_test_rows = real
    return leaky


# =====================================================================
# 6, 7, 8
# =====================================================================

def check_ties(ctx):
    frame = ctx.frame
    ties = frame[frame.IS_TIE == 1]
    blank = ties["HOME_WIN"].isna().all()
    zero = (ties["HOME_MARGIN"] == 0).all()
    totals = ties["TOTAL_PTS"].notna().all()
    decided_blank = frame[(frame.IS_TIE == 0) & frame.HOME_WIN.isna()]
    return (blank and zero and totals and decided_blank.empty,
            f"{len(ties)} tie(s): HOME_WIN blank {blank}, HOME_MARGIN all 0 "
            f"{zero}, TOTAL_PTS kept {totals}; "
            f"{len(decided_blank)} decided game(s) with a blank winner")


def plant_ties(ctx):
    def leaky():
        frame = ctx.frame
        saved = frame["HOME_WIN"].copy()
        index = frame.index[frame.IS_TIE == 1][0]
        frame.loc[index, "HOME_WIN"] = 1
        try:
            return check_ties(ctx)
        finally:
            frame["HOME_WIN"] = saved
    return leaky


def check_neutral_sites(ctx):
    """Home advantage is not applied at a neutral site."""
    frame = ctx.frame
    neutral = frame[frame.NEUTRAL_SITE == 1]
    params = ctx.elo_configs["ELO"]
    bad = []
    for _, row in neutral.iterrows():
        expected = B.expected_score(row["HOME_ELO"], row["AWAY_ELO"])
        if abs(expected - row["ELO_PROB"]) > 1e-12:
            bad.append((row["game_id"], expected, row["ELO_PROB"]))
    # and the complement: a normal game DOES carry the edge
    normal = frame[frame.NEUTRAL_SITE == 0].iloc[0]
    with_edge = B.expected_score(normal["HOME_ELO"] + params.home_advantage,
                                 normal["AWAY_ELO"])
    edge_applied = abs(with_edge - normal["ELO_PROB"]) < 1e-12
    return (not bad and edge_applied and len(neutral) > 0,
            f"{len(neutral)} neutral-site game(s), {len(bad)} carrying the "
            f"+{params.home_advantage} edge they must not; a normal game does "
            f"carry it: {edge_applied}")


def plant_neutral_sites(ctx):
    def leaky():
        frame = ctx.frame
        saved = frame["NEUTRAL_SITE"].copy()
        # mark a normal game neutral: it still carries the edge, so the check
        # must notice the edge where there should be none
        frame.loc[frame.index[frame.NEUTRAL_SITE == 0][0],
                  "NEUTRAL_SITE"] = 1
        try:
            return check_neutral_sites(ctx)
        finally:
            frame["NEUTRAL_SITE"] = saved
    return leaky


def check_unrelated_games_cannot_reach(ctx, sample_size=12):
    """Section 4's claim, which the dependency rule for phase 4 rests on.

    A fixture's features are final once both sides' prior games are recorded,
    WHATEVER happens in between - so corrupting every game that kicks off
    between both sides' last game and the fixture, involving neither side, must
    move nothing. The positive control corrupts a side's OWN last game and must
    move something, or the test is vacuous.
    """
    ordered = sorted(ctx.games, key=lambda g: (g["date"], g["game_id"]))
    candidates = []
    for index, game in enumerate(ordered):
        if game["season"] < 2015 or game["week"] < 8:
            continue
        sides = {game["home_franchise_id"], game["away_franchise_id"]}
        own = [i for i in range(index)
               if sides & {ordered[i]["home_franchise_id"],
                           ordered[i]["away_franchise_id"]}]
        if not own:
            continue
        last_own = max(own)
        between = [i for i in range(last_own + 1, index)
                   if not (sides & {ordered[i]["home_franchise_id"],
                                    ordered[i]["away_franchise_id"]})]
        if between:
            candidates.append((game, between, last_own))

    rng = random.Random(SEED + 2)
    sample = rng.sample(candidates, min(sample_size, len(candidates)))
    moved, control_moved, corrupted_total = 0, 0, 0
    for game, between, last_own in sample:
        base = rebuild_through(ordered, game, ctx.elo_configs)
        ids = {ordered[i]["game_id"] for i in between}
        corrupted_total += len(ids)
        after = rebuild_through(
            corrupt(ordered, lambda g: g["game_id"] in ids), game,
            ctx.elo_configs)
        if any(_differs(base[k], after[k]) for k in base):
            moved += 1
        own_id = ordered[last_own]["game_id"]
        control = rebuild_through(
            corrupt(ordered, lambda g: g["game_id"] == own_id), game,
            ctx.elo_configs)
        if any(_differs(base[k], control[k]) for k in base):
            control_moved += 1
    return (moved == 0 and control_moved == len(sample),
            f"{len(sample)} fixtures, {corrupted_total} unrelated game(s) "
            f"corrupted: {moved} fixture(s) moved; the control corrupting each "
            f"side's OWN last game moved {control_moved} of {len(sample)}")


def plant_unrelated_games_cannot_reach(ctx):
    """Mis-key the game log so every team's window sees every game."""
    def leaky():
        real = B.NflHistory.team_log
        B.NflHistory.team_log = lambda self, fid: [
            entry for log in self.log.values() for entry in log]
        try:
            return check_unrelated_games_cannot_reach(ctx, sample_size=4)
        finally:
            B.NflHistory.team_log = real
    return leaky


def check_coverage_and_nulls(ctx):
    ok1, detail1 = B.check_key_coverage(ctx.rows, ctx.games)
    ok2, detail2 = B.check_all_null_rows(
        ctx.rows, list(ctx.elo_configs), ctx.priors)
    return ok1 and ok2, f"{detail1}; {detail2}"


def plant_coverage_and_nulls(ctx):
    def leaky():
        dropped = ctx.rows[:-1]
        ok1, detail1 = B.check_key_coverage(dropped, ctx.games)
        return ok1, f"planted a missing row: {detail1}"
    return leaky


# =====================================================================

CHECKS = [
    ("1  window lag, hand-recomputed", check_window_lag, plant_window_lag),
    ("2a a later game cannot reach back", check_leak_future,
     plant_leak_future),
    ("2b ROLL stays in season, CARRY crosses", check_leak_previous_season,
     plant_leak_previous_season),
    ("3a Elo ignores post-training rows", check_elo_posttraining,
     plant_elo_posttraining),
    ("3b Elo responds to training signal", check_elo_training,
     plant_elo_training),
    ("4  the feature function reproduces the row", check_feature_function,
     plant_feature_function),
    ("5  the test-set firewall", check_test_firewall, plant_test_firewall),
    ("6  ties", check_ties, plant_ties),
    ("7  neutral sites", check_neutral_sites, plant_neutral_sites),
    ("8  key coverage and the all-null rule", check_coverage_and_nulls,
     plant_coverage_and_nulls),
    # Beyond the spec's eight: section 4 asks for the dependency claim to be
    # SHOWN from the data, and a claim phase 4 will rest a serving rule on
    # belongs in the suite rather than in a one-off script.
    ("9  an unrelated game cannot reach a fixture",
     check_unrelated_games_cannot_reach, plant_unrelated_games_cannot_reach),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-only", action="store_true")
    args = parser.parse_args()

    ctx = Ctx()
    print("=" * 74)
    print("NFL PHASE 2 VERIFICATION")
    print("=" * 74)
    print(f"  {len(ctx.games):,} games, {len(ctx.frame):,} dataset rows, "
          f"{len(B.feature_columns(list(ctx.elo_configs)))} features")
    print(f"  Elo read from the manifest: "
          + "; ".join(f"{n} K={p.k} carry={p.carryover:.3f} "
                      f"home={p.home_advantage}"
                      for n, p in ctx.elo_configs.items()))

    print()
    print("THE CONTROL RUN: all checks, nothing planted")
    failed = 0
    for label, check, _plant in CHECKS:
        ok, detail = check(ctx)
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        print(f"         {detail}")
        failed += not ok

    if args.control_only:
        print(f"\n{'control clean' if not failed else f'{failed} FAILED'}")
        return 1 if failed else 0

    print()
    print("EACH CHECK MADE TO FAIL, so none of them is vacuous")
    not_caught = 0
    for label, _check, plant in CHECKS:
        runner = plant(ctx)
        ok, detail = runner()
        caught = not ok
        print(f"  [{'caught' if caught else 'NOT CAUGHT'}] {label}")
        print(f"         {detail}")
        not_caught += not caught

    print()
    print("=" * 74)
    print(f"  control run      : {len(CHECKS) - failed} of {len(CHECKS)} pass")
    print(f"  planted failures : {len(CHECKS) - not_caught} of {len(CHECKS)} "
          f"caught")
    print("=" * 74)
    return 1 if (failed or not_caught) else 0


if __name__ == "__main__":
    sys.exit(main())
