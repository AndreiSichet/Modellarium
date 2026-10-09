"""NFL phase 3 verification: twelve checks, every one of them BEFORE the test.

Phase 3's one-shot step is §5, so everything that could make the selection
meaningless has to be established while the test seasons are still closed. Each
check is paired with a plant that must turn it red, because this project has
caught four vacuous guards - an unshifted rolling mean that passed its own leak
test, a positive control demanding every feature respond at one probe, a
query-count guard blinded by a first-level cache, and a G League lag probe that
passed in the one season where it could not distinguish anything.

    python ml-training/nfl/verify_nfl_selection.py
    python ml-training/nfl/verify_nfl_selection.py --control-only
"""
import argparse
import copy
import json
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ML = HERE.parent
REPO = ML.parent
sys.path.insert(0, str(HERE))

import nfl_common as C                       # noqa: E402
import build_nfl_model_dataset as B          # noqa: E402

SELECT_SCRIPT = HERE / "select_nfl_models.py"
TEST_SCRIPT = HERE / "test_nfl_models.py"
PHASE2_VERIFIER = HERE / "verify_nfl_features.py"

# The names a selection step may read and must not write.
FROZEN_NAMES = (
    "TEST_SEASONS", "VALIDATION_SEASONS", "SPAN_CANDIDATES",
    "STAGE_A_REFERENCE", "FEATURE_SETS", "ELO_VARIANTS", "FAMILIES",
    "LINEAR_C_GRID", "RIDGE_ALPHA_GRID", "XGB_GRID", "XGB_BASE",
    "MARKETS", "NEARER_BOUND_TRIGGER", "BOOTSTRAP_RESAMPLES",
)
MUTATORS = ("append", "extend", "insert", "update", "pop", "clear",
            "setdefault", "remove", "add", "discard", "__setitem__")
SEED = 20261008


class Ctx:
    """One fold's data, built once; the expensive thing here is Elo."""

    def __init__(self):
        self.games = B.load_games()
        self.span = 2012
        self.probe_validation = 2022
        self.training = list(range(self.span, self.probe_validation))
        self.frame = C.fold_frame(self.training)
        self.market = C.MARKETS[1]          # margin: a regression target
        self.train = C.rows_for(self.frame, self.training, self.market)
        self.validation = C.rows_for(self.frame, [self.probe_validation],
                                     self.market)
        self.columns = C.feature_list("elo_context_carry5", "plain")


# =====================================================================
# 1. the candidate set is fixed, and a selection step cannot widen it
# =====================================================================

def mutation_hits(source, names=FROZEN_NAMES):
    """Assignments and mutating calls only - a READ must not register.

    The G League's first version of this scan searched for `SPAN_CANDIDATES[`
    and fired on `SPAN_CANDIDATES[span]`, which is a lookup. So every pattern
    here requires either an `=` that is not `==` or a named mutating method.
    """
    hits = []
    for name in names:
        for pattern in (
            rf"\b{name}\s*(?:\[[^\]]*\])?\s*(?<![=!<>])=(?!=)",
            rf"\b{name}\s*(?:\+|-|\*|\|)=",
            rf"\b{name}\s*\.\s*(?:{'|'.join(MUTATORS)})\s*\(",
        ):
            for match in re.finditer(pattern, source):
                line = source[:match.start()].count("\n") + 1
                hits.append(f"{name} line {line}")
    return hits


def check_candidate_set(ctx):
    declared = {
        "spans": sorted(C.SPAN_CANDIDATES),
        "feature sets": sorted(C.FEATURE_SETS),
        "elo variants": list(C.ELO_VARIANTS),
        "families": list(C.FAMILIES),
    }
    expected = len(C.FEATURE_SETS) * len(C.ELO_VARIANTS) * len(C.FAMILIES)
    if len(C.candidates()) != expected:
        return False, f"candidates() gives {len(C.candidates())}, not {expected}"
    hits = []
    for path in (SELECT_SCRIPT, TEST_SCRIPT):
        hits += [f"{path.name}: {h}"
                 for h in mutation_hits(path.read_text(encoding="utf-8"))]
    if hits:
        return False, f"a frozen name is written: {hits}"
    return True, (f"{expected} candidates from {declared}; no frozen name is "
                  f"assigned or mutated in either script")


def plant_candidate_set(ctx):
    source = SELECT_SCRIPT.read_text(encoding="utf-8")
    tampered = source + '\nC.FEATURE_SETS["cheat"] = ["ROLL5"]\n'

    def leaky():
        hits = mutation_hits(tampered)
        if hits:
            return False, f"the scan found the planted write: {hits}"
        return True, "the planted write was NOT found"
    return leaky


# =====================================================================
# 2. Stage B structurally cannot run before Stage A
# =====================================================================

def check_stage_order(ctx):
    import inspect
    import select_nfl_models as S

    signature = inspect.signature(S.stage_b)
    required = [p for p in signature.parameters.values()
                if p.default is inspect.Parameter.empty]
    if not required or required[0].name != "span_name":
        return False, (f"stage_b's first required parameter is "
                       f"{[p.name for p in required]}, not span_name - it could "
                       f"run without Stage A's answer")
    body = inspect.getsource(S.main)
    a, b = body.find("stage_a("), body.find("stage_b(")
    if a < 0 or b < 0 or a > b:
        return False, f"main calls stage_a at {a} and stage_b at {b}"
    return True, (f"stage_b(span_name=...) takes the span as a required "
                  f"argument, and main calls stage_a before stage_b")


def plant_stage_order(ctx):
    import inspect
    import select_nfl_models as S
    body = inspect.getsource(S.main)
    flipped = body.replace("stage_a(", "ZZZ(").replace("stage_b(", "stage_a(")

    def leaky():
        a, b = flipped.find("stage_a("), flipped.find("stage_b(")
        if a < 0 or b < 0 or a > b:
            return False, (f"the order check rejects a main that calls them "
                           f"out of order (stage_a at {a}, stage_b at {b})")
        return True, "an out-of-order main was accepted"
    return leaky


# =====================================================================
# 3. no private metric code - E4's one definition per metric
# =====================================================================

BANNED = (
    (r"from\s+sklearn\.metrics\s+import", "sklearn.metrics import"),
    (r"sklearn\.metrics\.", "sklearn.metrics use"),
    (r"def\s+(log_loss|mean_absolute_error|brier_score_loss|accuracy_score)\b",
     "a private metric definition"),
)


def metric_hits(source):
    return [label for pattern, label in BANNED if re.search(pattern, source)]


def check_no_private_metrics(ctx):
    # THIS FILE IS EXCLUDED, and the exclusion is the honest answer rather than
    # a loophole: the scan's own patterns are string literals here, so scanning
    # itself reported a private metric in the verifier. The scripts that matter
    # are the ones that compute a reported number.
    scripts = [p for p in sorted(HERE.glob("*.py"))
               if p.name != Path(__file__).name]
    hits = []
    for path in scripts:
        hits += [f"{path.name}: {h}"
                 for h in metric_hits(path.read_text(encoding="utf-8"))]
    if hits:
        return False, f"private metric code: {hits}"
    source = (HERE / "nfl_common.py").read_text(encoding="utf-8")
    if "from model_evaluation import" not in source:
        return False, "nfl_common does not import the shared scorer"
    # per_row_loss computes log terms, which is the DECOMPOSITION rather than a
    # second definition - check 8 is what ties it to the shared scorer.
    if "def assert_decomposes" not in source:
        return False, ("per_row_loss exists with no assert_decomposes tying it "
                       "to model_evaluation")
    return True, (f"{len(scripts)} script(s) scanned, no sklearn.metrics and no "
                  f"private metric; every score goes through model_evaluation, "
                  f"and per_row_loss is bound to it by assert_decomposes")


def plant_no_private_metrics(ctx):
    tampered = ("from sklearn.metrics import log_loss\n"
                + (HERE / "nfl_common.py").read_text(encoding="utf-8"))

    def leaky():
        hits = metric_hits(tampered)
        if hits:
            return False, f"the scan found the planted import: {hits}"
        return True, "a planted sklearn.metrics import was NOT found"
    return leaky


# =====================================================================
# 4. the test-set firewall
# =====================================================================

def check_test_firewall(ctx):
    test = set(C.TEST_SEASONS)
    for name, start in C.SPAN_CANDIDATES.items():
        for training, validation in C.folds_for(start):
            if set(training) & test:
                return False, f"span {name} fold {validation} trains on test"
            if validation in test:
                return False, f"span {name} validates on test season {validation}"
    if set(C.VALIDATION_SEASONS) & test:
        return False, "a validation season is a test season"

    # The fitting function itself refuses, rather than the caller remembering.
    leaked = [g for g in ctx.games if g["season"] == min(C.TEST_SEASONS)][:40]
    try:
        B.fit_elo(leaked, False, who="verify_firewall_probe")
    except B.TestSeasonLeak:
        pass
    else:
        return False, "fit_elo accepted test-season rows"

    rows = C.rows_for(ctx.frame, ctx.training, ctx.market)
    if set(rows.season.unique()) & test:
        return False, "a fold's training rows contain a test season"

    # The one deliberate opt-out must have exactly one caller: the production
    # fit, which is DEFINED as training through the last completed season. A
    # second caller would be the firewall quietly becoming optional.
    callers = []
    for path in sorted(HERE.glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(),
                                 1):
            if "fit_elo_including_test" in line and not line.strip().startswith(
                    "#"):
                callers.append(f"{path.name}:{n}")
    expected_caller = "finalize_nfl_models.py"
    stray = [c for c in callers if not c.startswith(expected_caller)]
    if stray:
        return False, (f"fit_elo_including_test is reached from {stray} as well "
                       f"as the production fit")
    return True, (f"no fold of any span touches {sorted(test)}; fit_elo raises "
                  f"TestSeasonLeak on a test row; {len(rows):,} training rows "
                  f"span {min(rows.season)}..{max(rows.season)}; the opt-out "
                  f"fit_elo_including_test is reached only from "
                  f"{sorted(set(c.split(':')[0] for c in callers)) or ['nothing yet']}")


def plant_test_firewall(ctx):
    real = B.refuse_test_rows

    def leaky():
        B.refuse_test_rows = lambda games, who, **kw: games
        try:
            ok, detail = check_test_firewall(ctx)
        finally:
            B.refuse_test_rows = real
        if not ok:
            return False, f"with the firewall disabled the check fails: {detail}"
        return True, "the check passed with refuse_test_rows neutered"
    return leaky


# =====================================================================
# 5. Elo is refitted per fold, and it moves with the training signal
# =====================================================================

def fit_for(training, games=None, mov=False):
    rows = [g for g in (games or Ctx.games_cache)
            if g["season"] in set(training)]
    params, loss, _edges = B.fit_elo(rows, mov, who="verify_elo_probe")
    return params, loss


def swap_result(game):
    """Swap a game's OUTCOME, not just its points.

    THIS COST A FALSE FAILURE AND AN UNCAUGHT PLANT, and it is the fourth time
    this project has met the shape. The first version swapped `home_points` and
    `away_points` only - but `elo_log_loss` and `EloLedger.update` read
    `home_result`, and the points reach the fit through the MOV multiplier
    alone. So for the PLAIN variant the corruption touched nothing the fit
    reads: 5b passed because the value could not move, and 5c failed because it
    demanded a move that nothing could produce. The guard was corrupting a
    field adjacent to the one that matters.
    """
    game = dict(game)
    game["home_points"], game["away_points"] = (game["away_points"],
                                                game["home_points"])
    game["home_result"], game["away_result"] = (game["away_result"],
                                                game["home_result"])
    return game


def check_elo_refit(ctx):
    triples = {}
    for training, validation in C.folds_for(ctx.span):
        params, _loss = fit_for(training, ctx.games)
        triples[validation] = (params.k, round(params.carryover, 3),
                               params.home_advantage)
    distinct = sorted(set(triples.values()))
    text = "; ".join(f"{v}:K={t[0]},carry={t[1]},home={t[2]}"
                     for v, t in sorted(triples.items()))
    # The G League got ONE distinct pair across four folds and had to say so:
    # fold-to-fold variation is evidence only when it exists, and when it does
    # not the whole burden falls on the two controls below.
    note = (f"{len(distinct)} distinct triple(s) across {len(triples)} folds"
            + ("" if len(distinct) > 1 else
               " - a single fit reused would look IDENTICAL, so this says "
               "nothing and the controls carry the whole burden"))
    return True, f"{note}. {text}"


def check_elo_posttraining(ctx):
    """Corrupting everything at or after the validation season changes nothing."""
    training = list(range(ctx.span, ctx.probe_validation))
    base, _ = fit_for(training, ctx.games)
    corrupted, touched = [], 0
    for game in ctx.games:
        if game["season"] >= ctx.probe_validation:
            game = swap_result(game)
            touched += 1
        corrupted.append(game)
    after, _ = fit_for(training, corrupted)
    if after != base:
        return False, f"the fit moved: {base} -> {after}"
    return True, (f"swapped the scores of {touched:,} rows at or after "
                  f"{ctx.probe_validation}; fit unchanged at K={base.k}, "
                  f"carry={base.carryover:.3f}, home={base.home_advantage}")


def check_elo_training(ctx):
    """Randomising HALF the training results moves the fit.

    Half rather than all: flipping every result is SELF-INVERTING - Elo learns
    mirrored ratings, predicts the mirrored outcomes exactly as well, and the
    loss surface is identical. That is the control the WNBA phase got wrong.
    """
    training = list(range(ctx.span, ctx.probe_validation))
    base, _ = fit_for(training, ctx.games)
    rng = random.Random(SEED)
    corrupted, touched = [], 0
    for game in ctx.games:
        if game["season"] in set(training) and rng.random() < 0.5:
            game = swap_result(game)
            touched += 1
        corrupted.append(game)
    after, _ = fit_for(training, corrupted)
    if after == base:
        return False, (f"randomising {touched:,} training rows left the fit at "
                       f"{base}")
    direction = ("slower, as less signal predicts" if after.k < base.k
                 else "faster, which is NOT what less signal predicts")
    return True, (f"randomised {touched:,} of the training rows; K moved "
                  f"{base.k} -> {after.k} ({direction}), carry "
                  f"{base.carryover:.3f} -> {after.carryover:.3f}")


def plant_elo_posttraining(ctx):
    def leaky():
        # A fit that DID read post-training rows would move under the same
        # corruption. Widening the span to include the validation season is
        # what demonstrates the check's zero is agreement rather than a number
        # nothing can shift - the positive control the WNBA phase needed for
        # its exact 0.000e+00.
        wide = list(range(ctx.span, ctx.probe_validation + 1))
        corrupted = [swap_result(g) if g["season"] >= ctx.probe_validation
                     else g for g in ctx.games]
        clean_wide, _ = fit_for(wide, ctx.games)
        dirty_wide, _ = fit_for(wide, corrupted)
        if clean_wide != dirty_wide:
            return False, (f"a fit that DOES read post-training rows moves "
                           f"under the same corruption: {clean_wide} -> "
                           f"{dirty_wide}, so the check's zero is agreement "
                           f"rather than a value nothing can shift")
        return True, (f"widening the fit to {wide[-1]} did not move it either, "
                      f"so the negative test is vacuous")
    return leaky


def plant_elo_training(ctx):
    real = B.fit_elo

    def leaky():
        B.fit_elo = lambda rows, mov, who="x": (
            B.EloParams(20, 1.0 / 3.0, 50, mov), 0.0, [])
        try:
            ok, detail = check_elo_training(ctx)
        finally:
            B.fit_elo = real
        if not ok:
            return False, f"a constant fit is caught: {detail}"
        return True, "a fit that ignores its input passed the control"
    return leaky


# =====================================================================
# 6. Stage A scores identical rows across the three spans
# =====================================================================

def check_common_rows(ctx):
    season = C.common_folds()[0]
    per_span = {}
    for name, start in C.SPAN_CANDIDATES.items():
        training = [s for s in range(start, season)]
        frame = C.fold_frame(training)
        rows = C.rows_for(frame, [season], ctx.market)
        per_span[name] = set(rows.game_id)
    sizes = {n: len(ids) for n, ids in per_span.items()}
    if len(set(map(frozenset, per_span.values()))) != 1:
        return False, f"the spans score different rows: {sizes}"
    return True, (f"validation season {season}: all {len(per_span)} spans score "
                  f"the identical {sizes[list(sizes)[0]]} games, so Stage A's "
                  f"means compare spans rather than row sets")


def plant_common_rows(ctx):
    real = C.rows_for

    def leaky():
        state = {"n": 0}

        def biased(frame, seasons, market):
            out = real(frame, seasons, market)
            state["n"] += 1
            return out.iloc[1:] if state["n"] == 2 else out

        C.rows_for = biased
        try:
            ok, detail = check_common_rows(ctx)
        finally:
            C.rows_for = real
        if not ok:
            return False, f"one span losing a row is caught: {detail}"
        return True, "one span scoring fewer rows was accepted"
    return leaky


# =====================================================================
# 7. the imputer and the scaler are fitted on training rows only
# =====================================================================

def check_fitted_on_training_only(ctx):
    model, _setting, _trees = C.fit_family(ctx.market, "linear", ctx.train,
                                           ctx.columns)
    imputer = model.named_steps["impute"]
    scaler = model.named_steps["scale"]
    train_median = ctx.train[ctx.columns].median().to_numpy(dtype=float)
    gap = float(np.nanmax(np.abs(imputer.statistics_ - train_median)))
    if gap != 0.0:
        return False, (f"the imputer's medians differ from the training rows' "
                       f"by {gap:.3e}")

    wider = ctx.frame[ctx.frame.season.isin(
        set(ctx.training) | {ctx.probe_validation})]
    wider_median = wider[ctx.columns].median().to_numpy(dtype=float)
    spread = float(np.nanmax(np.abs(train_median - wider_median)))
    if spread == 0.0:
        return False, ("the training median equals the train+validation "
                       "median, so this check cannot distinguish them")
    train_mean = ctx.train[ctx.columns].fillna(
        ctx.train[ctx.columns].median()).to_numpy(dtype=float).mean(axis=0)
    mean_gap = float(np.nanmax(np.abs(scaler.mean_ - train_mean)))
    if mean_gap > 1e-9:
        return False, f"the scaler's means are not the training rows': {mean_gap:.3e}"
    return True, (f"imputer medians equal the training rows' at 0.0e+00 and "
                  f"differ from the train+validation medians by up to {spread:.4f}; "
                  f"scaler means match at {mean_gap:.1e}")


def plant_fitted_on_training_only(ctx):
    def leaky():
        wider = ctx.frame[ctx.frame.season.isin(
            set(ctx.training) | {ctx.probe_validation})]
        wider_rows = C.rows_for(ctx.frame,
                                set(ctx.training) | {ctx.probe_validation},
                                ctx.market)
        model, _s, _t = C.fit_family(ctx.market, "linear", wider_rows,
                                     ctx.columns)
        train_median = ctx.train[ctx.columns].median().to_numpy(dtype=float)
        gap = float(np.nanmax(np.abs(
            model.named_steps["impute"].statistics_ - train_median)))
        if gap != 0.0:
            return False, (f"a pipeline fitted on train+validation no longer "
                           f"matches the training medians (gap {gap:.4f}), so "
                           f"the check is sensitive to what it was fitted on")
        return True, "fitting on the wider rows changed nothing"
    return leaky


# =====================================================================
# 8. the bootstrap measures the reported metric, and it is PAIRED
# =====================================================================

def check_decomposition(ctx):
    """BOTH FAMILIES, not just linear - and that omission is why the verifier
    missed a real bug.

    The first version fitted `linear` for all three markets, so it could not
    reach the path that actually broke: XGBoost returns float32, and
    `regression_metrics` averages at whatever width it is handed, which put the
    margin market's aggregate 3.8e-07 from the per-row mean and tripped
    `assert_decomposes` at its 1e-12 tolerance. A smoke test of the one-shot
    test step found it; this check could not have. Covering both families is
    the fix, because the selection is free to pick either.
    """
    columns = C.feature_list("elo_context", "plain")
    detail = []
    for market in C.MARKETS:
        train = C.rows_for(ctx.frame, ctx.training, market)
        validation = C.rows_for(ctx.frame, [ctx.probe_validation], market)
        for family in C.FAMILIES:
            model, _s, _t = C.fit_family(market, family, train, columns)
            labels, proba = C.predict(market, model, validation, columns)
            value = C.assert_decomposes(market, validation[market.target],
                                        labels, proba)
            detail.append(f"{market.key}/{family} {value:.6f}")
    return True, (f"the per-row mean reproduces model_evaluation's own number "
                  f"for all {len(detail)} market/family pairs, float32 "
                  f"predictions included: " + ", ".join(detail))


def check_pairing(ctx):
    rng = np.random.default_rng(SEED)
    model_losses = rng.normal(10.0, 3.0, 400)
    baseline_losses = model_losses + rng.normal(0.2, 0.05, 400)
    paired = C.paired_bootstrap(model_losses, baseline_losses)
    shuffled = C.paired_bootstrap(model_losses,
                                  rng.permutation(baseline_losses))
    paired_width = paired["hi"] - paired["lo"]
    shuffled_width = shuffled["hi"] - shuffled["lo"]
    if paired_width >= shuffled_width:
        return False, (f"the paired interval is no narrower than an unpaired "
                       f"one ({paired_width:.4f} vs {shuffled_width:.4f}), so "
                       f"the pairing is not doing anything")
    return True, (f"pairing is load-bearing: width {paired_width:.4f} paired "
                  f"against {shuffled_width:.4f} when the baseline losses are "
                  f"detached from their games")


def plant_decomposition(ctx):
    real = C.per_row_loss

    def leaky():
        C.per_row_loss = lambda market, y, p, proba=None: np.square(
            np.asarray(y, dtype=float) - np.asarray(p, dtype=float))
        try:
            ok, detail = check_decomposition(ctx)
        except AssertionError as exc:
            return False, f"assert_decomposes raised: {str(exc)[:120]}"
        finally:
            C.per_row_loss = real
        return True, f"a squared-error decomposition was accepted: {detail}"
    return leaky


def plant_pairing(ctx):
    real = C.paired_bootstrap

    def leaky():
        # Sorting both sides destroys the pairing DETERMINISTICALLY, so the two
        # calls inside the check get the same treatment and their widths match.
        # The first version permuted inside both calls, which left one of them
        # narrower by luck and the check passed - an uncaught plant.
        def unpaired(model_losses, baseline_losses, seed=C.SEED,
                     resamples=C.BOOTSTRAP_RESAMPLES):
            return real(np.sort(np.asarray(model_losses, dtype=float)),
                        np.sort(np.asarray(baseline_losses, dtype=float)),
                        seed=seed, resamples=resamples)
        C.paired_bootstrap = unpaired
        try:
            ok, detail = check_pairing(ctx)
        finally:
            C.paired_bootstrap = real
        if not ok:
            return False, f"an unpaired bootstrap is caught: {detail}"
        return True, "an unpaired bootstrap passed the pairing check"
    return leaky


# =====================================================================
# 9. the tie rule, and whether the tie-break is total
# =====================================================================

def check_tie_rule(ctx):
    a = [1.000, 1.002, 0.998, 1.001, 0.999, 1.000]
    b = [1.010, 0.990, 1.030, 0.970, 1.020, 0.980]
    if C.beats(a, b):
        return False, ("a 0.0 mean gap with a large paired spread counted as a "
                       "win")
    clear_a = [1.00, 1.01, 0.99, 1.00, 1.01, 0.99]
    clear_b = [1.20, 1.21, 1.19, 1.20, 1.21, 1.19]
    if not C.beats(clear_a, clear_b):
        return False, "a 0.20 gap with a 0.000 paired spread counted as a tie"

    # The spec's three keys are NOT total; the mean is the fourth, named rather
    # than left to the stability of `sorted`.
    # The per-fold scores need a REAL paired spread, and SHARING one wobble
    # pattern does not give one: if two candidates differ by a constant then
    # their paired difference is constant, stdev is 0.0, and `beats` succeeds on
    # any better mean. Two versions of this check failed that way before the
    # patterns were made independent - the synthetic was wrong, not the rule.
    vectors = {
        "linear/mov/elo_context": ([1.00, 1.04, 0.96, 1.02, 0.98, 1.00],
                                   (13, "linear", "elo_context")),
        "linear/plain/elo_context": ([1.03, 0.98, 1.02, 0.97, 1.04, 0.97],
                                     (13, "linear", "elo_context")),
        "xgboost/mov/elo_context": ([0.97, 1.02, 0.99, 1.03, 0.96, 1.02],
                                    (13, "xgboost", "elo_context")),
        "linear/mov/elo_context_carry5": ([1.01, 0.96, 1.03, 0.97, 1.02, 0.99],
                                          (21, "linear",
                                           "elo_context_carry5")),
    }
    lowest = min(vectors, key=lambda n: statistics.fmean(vectors[n][0]))
    not_tied = [n for n in vectors if n != lowest
                and (C.beats(vectors[n][0], vectors[lowest][0])
                     or C.beats(vectors[lowest][0], vectors[n][0]))]
    if not_tied:
        return False, (f"the synthetic has no pool to break: {not_tied} are "
                       f"separated from {lowest} rather than tied with it")

    winner, tied = C.rank_with_ties(vectors)
    if winner != "linear/mov/elo_context":
        return False, (f"the tie-break chose {winner}: it should prefer fewer "
                       f"features, then linear, then the better mean")
    if len(tied) != 3:
        return False, f"expected 3 tied, got {len(tied)}: {tied}"

    # The fourth key must be what decides between the two that share all three
    # of the spec's keys - swapping their means must swap the winner, or the
    # outcome is really coming from sort order.
    swapped = dict(vectors)
    swapped["linear/mov/elo_context"] = (
        vectors["linear/plain/elo_context"][0],
        vectors["linear/mov/elo_context"][1])
    swapped["linear/plain/elo_context"] = (
        vectors["linear/mov/elo_context"][0],
        vectors["linear/plain/elo_context"][1])
    other, _ = C.rank_with_ties(swapped)
    if other != "linear/plain/elo_context":
        return False, (f"swapping the two means left the winner at {other}, so "
                       f"the fourth key is not what decides between them")
    # THE POOL MUST BE THE MAXIMAL SET. `beats` is not transitive, so a pool
    # defined as "tied with one reference candidate" can admit a candidate that
    # something else demonstrably beats. That was a real defect: on the winner
    # market three candidates were reported as tied while the shipped model or
    # another candidate beat them.
    triangle = {
        "A": ([1.00, 1.01, 0.99, 1.00, 1.01, 0.99], (13, "linear", "elo_context")),
        "B": ([1.08, 1.09, 1.07, 1.08, 1.09, 1.07], (13, "linear", "elo_context")),
        "C": ([1.20, 0.90, 1.30, 0.85, 1.25, 0.95], (13, "linear", "elo_context")),
    }
    if not C.beats(triangle["A"][0], triangle["B"][0]):
        return False, "the triangle synthetic does not have A beating B"
    _w, tied = C.rank_with_ties(triangle)
    if "B" in tied:
        return False, ("B is reported as tied although A beats it - the pool "
                       "is not the maximal set")
    return True, (f"a 0.000 mean gap with a 0.020 paired spread ties; a 0.200 "
                  f"gap with a 0.010 spread wins; among 4 mutual ties the break "
                  f"picks {winner} (fewest features, then linear, then the "
                  f"better mean) and swapping the two means swaps the winner; "
                  f"and a candidate something beats is kept OUT of the pool "
                  f"even when it ties the reference")


def plant_tie_rule(ctx):
    real_beats = C.beats

    def leaky():
        C.beats = lambda challenger, incumbent: (
            statistics.fmean(challenger) < statistics.fmean(incumbent))
        try:
            ok, detail = check_tie_rule(ctx)
        finally:
            C.beats = real_beats
        if not ok:
            return False, (f"a rule that ignores the paired spread is caught: "
                           f"{detail}")
        return True, "a bare argmin passed the tie rule check"
    return leaky


# =====================================================================
# 10. ROLL5 is in no candidate
# =====================================================================

def check_roll5_absent(ctx):
    for feature_set, windows in C.FEATURE_SETS.items():
        if "ROLL5" in windows:
            return False, f"{feature_set} includes ROLL5"
    offending = []
    for feature_set, elo, _family in C.candidates():
        bad = [c for c in C.feature_list(feature_set, elo)
               if c.startswith("HOME_ROLL5") or c.startswith("AWAY_ROLL5")]
        if bad:
            offending.append(f"{feature_set}/{elo}: {bad[:2]}")
    if offending:
        return False, f"a candidate carries ROLL5 columns: {offending}"
    return True, (f"none of {len(C.candidates())} candidates carries a ROLL5 "
                  f"column - phase 2 measured its weeks 1-4 retention at 0.0%, "
                  f"and weeks 1-4 is when the season is served")


def plant_roll5_absent(ctx):
    real = dict(C.FEATURE_SETS)

    def leaky():
        C.FEATURE_SETS["cheat"] = ["ROLL5"]
        try:
            ok, detail = check_roll5_absent(ctx)
        finally:
            C.FEATURE_SETS.clear()
            C.FEATURE_SETS.update(real)
        if not ok:
            return False, f"a ROLL5 feature set is caught: {detail}"
        return True, "a ROLL5 feature set was accepted"
    return leaky


# =====================================================================
# 11. the test-once guard refuses in both modes
# =====================================================================

def check_test_once_guard(ctx):
    import test_nfl_models as T

    real_receipt = C.RECEIPT_PATH
    # What must survive the probe is the REAL receipt, byte for byte. The
    # probe's own scratch file is planted on purpose, so its existence proves
    # nothing; an earlier version asserted on the restored C.RECEIPT_PATH and
    # then on the scratch, and both were the wrong file.
    before = (real_receipt.read_bytes() if real_receipt.exists() else None)
    scratch = HERE / "_verify_probe_receipt.json"
    real_argv = sys.argv
    outcomes = {}
    try:
        C.RECEIPT_PATH = scratch
        if scratch.exists():
            scratch.unlink()

        sys.argv = ["test_nfl_models.py"]
        outcomes["no flag"] = T.main()

        scratch.write_text('{"probe": true}', encoding="utf-8")
        sys.argv = ["test_nfl_models.py", "--i-am-ready-to-touch-test"]
        outcomes["flag + receipt"] = T.main()
    finally:
        sys.argv = real_argv
        left_behind = scratch.exists()
        if left_behind:
            scratch.unlink()
        C.RECEIPT_PATH = real_receipt

    if outcomes["no flag"] == 0:
        return False, "it ran without --i-am-ready-to-touch-test"
    if outcomes["flag + receipt"] == 0:
        return False, "it ran with the flag while a receipt existed"
    if scratch.exists():
        return False, f"the probe's {scratch.name} survived cleanup"
    after = (real_receipt.read_bytes() if real_receipt.exists() else None)
    if after != before:
        return False, (f"{real_receipt.name} changed during the probe "
                       f"({'absent' if before is None else len(before)} -> "
                       f"{'absent' if after is None else len(after)} bytes)")
    state = ("absent, so the test has not been opened" if after is None
             else f"present and byte-identical ({len(after):,} bytes)")
    return True, (f"refuses with no flag (exit {outcomes['no flag']}) and "
                  f"refuses WITH the flag when a receipt exists "
                  f"(exit {outcomes['flag + receipt']}); the real "
                  f"{real_receipt.name} is {state}, and no test row was read "
                  f"on either path")


def plant_test_once_guard(ctx):
    import test_nfl_models as T

    def leaky():
        # With MARKETS empty the loop body never runs, so nothing can reach a
        # test row - which is what lets this plant show the guards were the
        # reason for the two refusals without opening the test.
        real_markets, real_receipt = C.MARKETS, C.RECEIPT_PATH
        scratch = HERE / "_verify_probe_receipt.json"
        real_argv = sys.argv
        try:
            C.MARKETS = ()
            C.RECEIPT_PATH = scratch
            if scratch.exists():
                scratch.unlink()
            sys.argv = ["test_nfl_models.py", "--i-am-ready-to-touch-test"]
            code = T.main()
            wrote = scratch.exists()
        finally:
            sys.argv = real_argv
            if scratch.exists():
                scratch.unlink()
            C.MARKETS, C.RECEIPT_PATH = real_markets, real_receipt
        if code == 0 and wrote:
            return False, ("with the flag set and no receipt it proceeds and "
                           "writes one (exit 0), so the two refusals above came "
                           "from the guards rather than from an always-refusing "
                           "script")
        return True, (f"even with the flag and no receipt it refused "
                      f"(exit {code}), so the check cannot tell a guard from a "
                      f"script that never runs")
    return leaky


# =====================================================================
# 12. phase 2's guards still pass
# =====================================================================

def check_phase2(ctx):
    result = subprocess.run(
        [sys.executable, str(PHASE2_VERIFIER), "--control-only"],
        cwd=REPO, capture_output=True, text=True)
    tail = [l for l in result.stdout.splitlines() if l.strip()][-1:]
    if result.returncode != 0:
        return False, f"phase 2's verifier exits {result.returncode}: {tail}"
    passes = result.stdout.count("[PASS]")
    return True, (f"phase 2's control run still passes ({passes} checks, exit "
                  f"0) - phase 3 changed none of its features")


def plant_phase2(ctx):
    def leaky():
        result = subprocess.run(
            [sys.executable, str(PHASE2_VERIFIER), "--control-only",
             "--no-such-flag"], cwd=REPO, capture_output=True, text=True)
        if result.returncode != 0:
            return False, (f"a broken invocation is reported rather than read "
                           f"as a pass (exit {result.returncode})")
        return True, "a broken invocation was read as a pass"
    return leaky


# =====================================================================
# 13. splitting fit_elo changed nothing phase 2 recorded
# =====================================================================

def check_phase2_elo_unchanged(ctx):
    """Phase 3 split `fit_elo` into a guard plus an unguarded core, so the
    production fit could name its opt-out instead of weakening the firewall by
    a keyword. A refactor of a FITTING function has to be shown inert, and the
    manifest is the record to show it against."""
    manifest = json.loads(B.MANIFEST_PATH.read_text(encoding="utf-8"))
    training = [g for g in ctx.games
                if B.split_role(g["season"]) == "train"]
    detail = []
    for name, mov in (("ELO", False), ("ELO_MOV", True)):
        recorded = manifest["elo"][name]
        params, _loss, _edges = B.fit_elo(training, mov,
                                          who="verify_refactor_probe")
        if (params.k != recorded["k"]
                or abs(params.carryover - recorded["carryover"]) > 1e-12
                or params.home_advantage != recorded["home_advantage"]
                or params.mov != recorded["margin_of_victory_multiplier"]):
            return False, (f"{name} refits to {params} against the manifest's "
                           f"{recorded}")
        detail.append(f"{name} K={params.k} carry={params.carryover:.3f} "
                      f"home={params.home_advantage}")
    return True, (f"fit_elo on phase 2's {len(training):,} training games "
                  f"reproduces the manifest exactly: " + "; ".join(detail))


def plant_phase2_elo_unchanged(ctx):
    real = B._fit_elo_unguarded

    def leaky():
        B._fit_elo_unguarded = lambda rows, mov, who: (
            B.EloParams(8, 0.0, 0, mov), 0.0, [])
        try:
            ok, detail = check_phase2_elo_unchanged(ctx)
        finally:
            B._fit_elo_unguarded = real
        if not ok:
            return False, f"a changed core is caught: {detail}"
        return True, "a core returning the wrong parameters was accepted"
    return leaky


CHECKS = [
    ("1  the candidate set is fixed and unwritable", check_candidate_set,
     plant_candidate_set),
    ("2  Stage B cannot run before Stage A", check_stage_order,
     plant_stage_order),
    ("3  no private metric code (E4)", check_no_private_metrics,
     plant_no_private_metrics),
    ("4  the test-set firewall", check_test_firewall, plant_test_firewall),
    # 5a has NO plant, and that is stated rather than faked: it reports the
    # distinct fitted triples and cannot fail, so pairing it with a plant would
    # manufacture a "caught" line that means nothing. The evidence for the
    # refit being live is 5b and 5c.
    ("5a Elo refit per fold, REPORT ONLY", check_elo_refit, None),
    ("5b Elo ignores post-training rows", check_elo_posttraining,
     plant_elo_posttraining),
    ("5c Elo responds to training signal", check_elo_training,
     plant_elo_training),
    ("6  Stage A scores identical rows across spans", check_common_rows,
     plant_common_rows),
    ("7  imputer and scaler fitted on training rows only",
     check_fitted_on_training_only, plant_fitted_on_training_only),
    ("8a the bootstrap measures the reported metric", check_decomposition,
     plant_decomposition),
    ("8b the bootstrap is paired", check_pairing, plant_pairing),
    ("9  the tie rule needs more than the paired spread", check_tie_rule,
     plant_tie_rule),
    ("10 ROLL5 is in no candidate", check_roll5_absent, plant_roll5_absent),
    ("11 the test-once guard refuses both ways", check_test_once_guard,
     plant_test_once_guard),
    ("12 phase 2's guards still pass", check_phase2, plant_phase2),
    ("13 splitting fit_elo changed nothing", check_phase2_elo_unchanged,
     plant_phase2_elo_unchanged),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-only", action="store_true")
    args = parser.parse_args()

    ctx = Ctx()
    Ctx.games_cache = ctx.games
    print("=" * 74)
    print("NFL PHASE 3 VERIFICATION - every check BEFORE the test is opened")
    print("=" * 74)
    print(f"  {len(ctx.games):,} games; probe fold = span {ctx.span}, "
          f"validation {ctx.probe_validation} "
          f"({len(ctx.train):,} train / {len(ctx.validation)} validation rows)")
    print(f"  test seasons {list(C.TEST_SEASONS)}, closed until "
          f"{TEST_SCRIPT.name} is run with its flag")

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
    not_caught, plantable = 0, 0
    for label, _check, plant in CHECKS:
        if plant is None:
            print(f"  [ -    ] {label}")
            print(f"         no plant: this one reports and cannot fail")
            continue
        plantable += 1
        ok, detail = plant(ctx)()
        caught = not ok
        print(f"  [{'caught' if caught else 'NOT CAUGHT'}] {label}")
        print(f"         {detail}")
        not_caught += not caught

    print()
    print("=" * 74)
    print(f"  control run      : {len(CHECKS) - failed} of {len(CHECKS)} pass")
    print(f"  planted failures : {plantable - not_caught} of {plantable} "
          f"caught ({len(CHECKS) - plantable} check reports only)")
    print("=" * 74)
    return 1 if (failed or not_caught) else 0


if __name__ == "__main__":
    sys.exit(main())
