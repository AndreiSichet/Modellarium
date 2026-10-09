"""NFL phase 3: the split, the FIXED candidate set, the tie rule, the folds.

Everything a selection step is allowed to choose between is declared here, as a
module constant, before any score exists. `verify_nfl_selection.py` scans the
selection script for any assignment to these names, so the candidate set cannot
be widened after seeing a result.

ELO IS REFITTED INSIDE EVERY FOLD, which is why a fold cannot simply read the
dataset's Elo columns: those were fitted on 2012-2023 and a fold validating on
2019 may not see 2019 onwards. Each fold refits K, carryover and home advantage
on its own training seasons and rebuilds the feature rows through phase 2's own
`features_for`, so the columns a fold scores are the columns phase 4 would serve
under that fold's parameters.
"""
import collections
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ML = HERE.parent
REPO = ML.parent
sys.path.insert(0, str(ML))
sys.path.insert(0, str(REPO / "data-pipeline" / "nfl" / "preprocessing"))

import build_nfl_model_dataset as B          # noqa: E402
from model_evaluation import (               # noqa: E402
    classification_metrics,
    regression_metrics,
)

MODELS_DIR = ML / "models_nfl"
RECEIPT_PATH = HERE / "test_receipt.json"
SELECTION_PATH = MODELS_DIR / "selection.json"

# ---------------------------------------------------------------- the split
# Phase 2 declared these before any feature was built and they are imported
# rather than restated, so the two phases cannot disagree about the firewall.
TEST_SEASONS = B.TEST_SEASONS                      # (2024, 2025)
VALIDATION_SEASONS = (2018, 2019, 2020, 2021, 2022, 2023)
SPAN_CANDIDATES = {"2012": 2012, "2015": 2015, "2018": 2018}
STAGE_A_REFERENCE = {"features": "elo_context_carry5", "elo": "plain",
                     "family": "linear"}

# ---------------------------------------------------------------- markets
Market = collections.namedtuple("Market", "key label target classification")
MARKETS = (
    Market("winner", "Winner", "HOME_WIN", True),
    Market("margin", "Margin", "HOME_MARGIN", False),
    Market("total", "Total", "TOTAL_PTS", False),
)

# A served winner probability means P(home wins | not tied): the label is blank
# on the 13 tied games, so they are dropped from fitting and from scoring.
WINNER_QUALIFIER = "P(home wins | not tied)"

# ---------------------------------------------------------------- features
CONTEXT = ["NEUTRAL_SITE", "DIVISION_GAME", "WEEK"]
REST = [f"{side}_{name}" for side in ("HOME", "AWAY")
        for name in ("REST_DAYS", "SHORT_WEEK", "OFF_BYE", "SEASON_OPENER")]


def window_columns(window):
    return [f"{side}_{window}_{metric}"
            for side in ("HOME", "AWAY") for metric in B.METRICS]


# THE ELO BLOCK IS THE TWO RATINGS, NOT THE PROBABILITY. `ELO_PROB` is a
# deterministic function of the two ratings and the home advantage, so adding it
# would be exact collinearity for the linear family; a model given both ratings
# can form the difference itself. The probability is reserved for the Elo-alone
# BASELINE, where it is the whole prediction.
ELO_COLUMNS = {
    "plain": ["HOME_ELO", "AWAY_ELO"],
    "mov": ["HOME_ELO_MOV", "AWAY_ELO_MOV"],
}
ELO_PROB_COLUMN = {"plain": "ELO_PROB", "mov": "ELO_MOV_PROB"}

# ROLL5 IS DELIBERATELY NOT A CANDIDATE. Phase 2 measured its retention in
# weeks 1-4 at 0.0% - five prior in-season games cannot exist before week 6,
# which is exactly when the season is served.
FEATURE_SETS = {
    "elo_context": [],
    "elo_context_carry5": ["CARRY5"],
    "elo_context_carry8": ["CARRY8"],
    "elo_context_carry5_roll3": ["CARRY5", "ROLL3"],
}
ELO_VARIANTS = ("plain", "mov")
FAMILIES = ("linear", "xgboost")


def feature_list(feature_set, elo_variant):
    columns = list(ELO_COLUMNS[elo_variant]) + list(CONTEXT) + list(REST)
    for window in FEATURE_SETS[feature_set]:
        columns += window_columns(window)
    return columns


def candidates():
    """Every Stage B combination, in a fixed order."""
    return [(fs, elo, family)
            for fs in FEATURE_SETS
            for elo in ELO_VARIANTS
            for family in FAMILIES]


# ---------------------------------------------------------------- the folds

def folds_for(span_start, validation_seasons=VALIDATION_SEASONS):
    """(training seasons, validation season) for every usable fold.

    A FOLD WITH NO TRAINING SEASON IS NOT USABLE, and the spec's "the same six
    folds for all three spans" is therefore unsatisfiable: span 2018 validating
    on 2018 would train on nothing. Such folds are dropped here and the caller
    reports it, rather than being handed an empty training set.
    """
    out = []
    for season in validation_seasons:
        training = [s for s in range(span_start, season)]
        if training:
            out.append((training, season))
    return out


def common_folds(spans=None, validation_seasons=VALIDATION_SEASONS):
    """The validation seasons EVERY span can serve.

    Stage A compares spans by their mean across folds, so they have to be the
    same folds - comparing means over different fold sets compares the fold sets
    too, which is the unfairness the WNBA phase caught and had to redo.
    """
    spans = spans or list(SPAN_CANDIDATES.values())
    usable = [{season for _train, season in folds_for(start)} for start in spans]
    return tuple(sorted(set.intersection(*usable)))


# ---------------------------------------------------------------- fold data

_GAMES = None
_FRAME_CACHE = {}
_ELO_CACHE = {}


def games():
    global _GAMES
    if _GAMES is None:
        _GAMES = B.load_games()
    return _GAMES


def fold_elo(training_seasons, mov):
    """Elo parameters fitted on these seasons only. Cached; test rows refused.

    The replay always starts at the first season in the data, whatever the span:
    a rating entering any training season must reflect everything before it. Only
    the FIT is restricted to the span's training seasons.
    """
    key = (tuple(training_seasons), mov)
    if key not in _ELO_CACHE:
        rows = [g for g in games() if g["season"] in set(training_seasons)]
        params, loss, edges = B.fit_elo(
            rows, mov, who=f"fold_elo{sorted(set(training_seasons))[-1]}")
        _ELO_CACHE[key] = (params, loss, edges)
    return _ELO_CACHE[key]


def fold_frame(training_seasons):
    """Every game's features under Elo refitted on these training seasons."""
    key = tuple(training_seasons)
    if key not in _FRAME_CACHE:
        configs = {}
        for name, mov in (("ELO", False), ("ELO_MOV", True)):
            params, _loss, _edges = fold_elo(training_seasons, mov)
            configs[name] = params
        rows, _history, _priors = B.build_rows(games(), configs)
        _FRAME_CACHE[key] = pd.DataFrame(rows)
    return _FRAME_CACHE[key]


def rows_for(frame, seasons, market):
    """The scoreable rows of a market: ties are dropped for the winner.

    The cast matters. `HOME_WIN` is blank on the 13 tied games, so the column
    arrives as object or float with NaN; once the ties are gone the remaining
    values are a clean 0/1 and sklearn needs to be told so, or it reads the
    target as continuous and refuses to fit a classifier.
    """
    subset = frame[frame.season.isin(set(seasons))]
    if market.classification:
        subset = subset[subset.IS_TIE == 0].copy()
        subset[market.target] = subset[market.target].astype(int)
    return subset


# ---------------------------------------------------------------- the tie rule

def beats(challenger, incumbent):
    """Per-fold scores, lower is better. True only if the gap clears the spread.

    THE RULE, WRITTEN BEFORE ANY RESULT: a candidate beats another only if its
    mean validation score is better by more than the fold-to-fold standard
    deviation of the PAIRED difference. Anything else is a tie, which makes
    "defensible rather than demonstrated" an outcome the record has to state
    instead of an argmin hiding it.
    """
    diffs = [c - i for c, i in zip(challenger, incumbent)]
    if len(diffs) < 2:
        return statistics.fmean(diffs) < 0
    return statistics.fmean(diffs) < -statistics.stdev(diffs)


def tie_break_key(entry):
    """Fewer features, then linear over trees, then lower serving cost.

    Serving cost here is whether the feature set needs a form window at all: a
    set that is Elo plus context asks phase 4 to replay no rolling window.
    """
    feature_count, family, feature_set = entry
    return (feature_count,
            0 if family == "linear" else 1,
            0 if not FEATURE_SETS[feature_set] else 1)


def rank_with_ties(scored):
    """scored: {name: (per_fold_scores, metadata)} -> (winner, tied_with).

    The pool is every candidate NOTHING beats; the tie-break ranks that pool
    and the rest are reported as tied with the pick.
    """
    names = sorted(scored, key=lambda n: statistics.fmean(scored[n][0]))

    # THE POOL IS EVERY CANDIDATE NOTHING BEATS, not everything tied with the
    # single lowest-mean one. The first version used the latter, and that was
    # an implementation choice of mine rather than anything §3 asks for - §3
    # defines `beats` and says nothing about how the tie-break's pool is
    # formed. It was wrong, and the winner market shows why: three candidates
    # were listed as tied while something in the set demonstrably beat them,
    # including `linear/plain/elo_context`, which the SHIPPED model itself
    # beats. A pool defined against one arbitrary reference inherits that
    # reference's blind spots, because `beats` is not transitive - it compares
    # a paired mean against a paired spread, so A can beat B, fail to beat C,
    # and C and B be indistinguishable.
    #
    # Being maximal is the property that actually matters for shipping: a
    # candidate is eligible unless some other candidate is demonstrably better
    # than it. Measured: this changes the reported tie set on the winner market
    # (16 -> 13) and changes NO shipped configuration in any of the three, so
    # the test stays closed and the receipt still matches.
    pool = [n for n in names
            if not any(beats(scored[o][0], scored[n][0])
                       for o in names if o != n)]
    # Among the pool, apply the tie-break - and then the mean,
    # because THE SPEC'S THREE KEYS ARE NOT TOTAL. Measured on this phase's own
    # results: 2 candidates share the best key on the winner, 2 on the margin
    # and 4 on the total. Without a fourth key the choice would fall to the
    # stability of `sorted` over a list that happens to be ordered by mean,
    # which is an implementation detail rather than a rule. Naming the mean as
    # the fourth key changes no selection - verified by re-running and diffing
    # selection.json - and removes the reliance on sort stability.
    ranked = sorted(pool, key=lambda n: tie_break_key(scored[n][1])
                    + (statistics.fmean(scored[n][0]),))
    return ranked[0], [n for n in pool if n != ranked[0]]


# ---------------------------------------------------------------- scoring

def score(market, y_true, predictions, proba=None):
    """Every metric in this phase comes from model_evaluation, never from here.

    E4 put the one definition of each metric in that module - float64 before the
    log, which is why a private log_loss in an NFL script would quietly disagree
    in the eighth decimal. `verify_nfl_selection.py` greps for one.
    """
    if market.classification:
        return classification_metrics(y_true, proba, predictions)
    return regression_metrics(y_true, predictions)


def primary(market, metrics):
    return metrics["log_loss"] if market.classification else metrics["mae"]


# ---------------------------------------------------------------- families
# Small grids, fixed here before any score exists. With roughly 2,500 training
# games there is no room for a wide search, and a wide one on this little data
# would select noise.
LINEAR_C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
RIDGE_ALPHA_GRID = (0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
XGB_GRID = (
    {"max_depth": 2, "learning_rate": 0.05},
    {"max_depth": 3, "learning_rate": 0.05},
    {"max_depth": 2, "learning_rate": 0.10},
    {"max_depth": 4, "learning_rate": 0.05},
)
XGB_BASE = {
    "n_estimators": 2000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "early_stopping_rounds": 50,
    "random_state": 0,
    "n_jobs": 4,
}
INNER_VALIDATION_FRACTION = 0.2
SEED = 20261008


def inner_split(frame):
    """The last fifth of the training rows by date, as an inner validation set.

    BY ROWS RATHER THAN BY SEASON, deliberately: carving off the last training
    SEASON is what the WNBA and G League phases did, and it cannot work here
    because span 2018 reaches folds with a single training season. Splitting on
    date keeps the time order - nothing in the inner validation set precedes
    anything it tunes - and is available at every fold size.
    """
    ordered = frame.sort_values(["date", "game_id"])
    cut = int(len(ordered) * (1.0 - INNER_VALIDATION_FRACTION))
    cut = max(1, min(cut, len(ordered) - 1))
    return ordered.iloc[:cut], ordered.iloc[cut:]


def _linear_pipeline(market, strength):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # THE IMPUTER IS PART OF THE ARTIFACT, and it has to be. REST_DAYS is NaN on
    # every season opener - an offseason is not rest - and a linear model cannot
    # take a NaN. Imputing inside the pipeline means the median is fitted on
    # training rows only, so there is no leakage, and SEASON_OPENER rides along
    # as the indicator saying the value was supplied rather than observed.
    # XGBoost gets the raw NaN instead: it learns a default direction, and
    # imputing for it would throw that away.
    estimator = (LogisticRegression(C=strength, max_iter=5000,
                                    random_state=SEED)
                 if market.classification
                 else Ridge(alpha=strength, random_state=SEED))
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def _xgb(market, params):
    from xgboost import XGBClassifier, XGBRegressor
    kwargs = dict(XGB_BASE)
    kwargs.update(params)
    if market.classification:
        return XGBClassifier(eval_metric="logloss", **kwargs)
    return XGBRegressor(eval_metric="mae", **kwargs)


def predict(market, model, frame, columns):
    """FLOAT64 BEFORE ANY SCORE, on both paths, and the regression one earned it.

    XGBoost returns float32. `model_evaluation.classification_metrics` already
    widens, with a comment saying E4 found the two halves of this project
    disagreeing in the eighth decimal - but `regression_metrics` does NOT, so a
    float32 prediction array is averaged at float32 width. Measured here on the
    margin market: MAE 10.45529842376709 against 10.455298046417097, a gap of
    3.8e-07 that broke `assert_decomposes` at its 1e-12 tolerance. The same
    float32 ULP trap as §29's 313 phantom inversions, in a new place.

    THE ROOT FIX BELONGS IN `regression_metrics`, and it is deliberately not
    made there: that module is how the NBA retrain gate and two other leagues
    score six XGBoost regression targets, and §9 of this phase forbids touching
    their paths. Widening it would move their recorded MAEs in the seventh
    decimal - almost certainly harmless against a 2% gate, but "almost
    certainly" is a claim that needs measuring across three leagues, which is
    not this phase. Recorded as a finding; fixed here where it is this league's
    to fix.
    """
    x = frame[columns]
    if market.classification:
        proba = np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
        return model.predict(x), proba
    return np.asarray(model.predict(x), dtype=np.float64), None


def fit_family(market, family, train, columns):
    """Fit one family, choosing its regularisation inside the training rows.

    Returns (fitted model, chosen setting, trees or None).
    """
    inner_train, inner_validation = inner_split(train)
    y = market.target

    if family == "linear":
        grid = LINEAR_C_GRID if market.classification else RIDGE_ALPHA_GRID
        best, best_loss = None, float("inf")
        for strength in grid:
            model = _linear_pipeline(market, strength)
            model.fit(inner_train[columns], inner_train[y])
            labels, proba = predict(market, model, inner_validation, columns)
            loss = primary(market, score(market, inner_validation[y],
                                         labels, proba))
            if loss < best_loss:
                best, best_loss = strength, loss
        model = _linear_pipeline(market, best)
        model.fit(train[columns], train[y])
        return model, {"strength": best}, None

    best, best_loss, best_trees = None, float("inf"), None
    for params in XGB_GRID:
        model = _xgb(market, params)
        model.fit(inner_train[columns], inner_train[y],
                  eval_set=[(inner_validation[columns], inner_validation[y])],
                  verbose=False)
        labels, proba = predict(market, model, inner_validation, columns)
        loss = primary(market, score(market, inner_validation[y],
                                     labels, proba))
        if loss < best_loss:
            best, best_loss = params, loss
            best_trees = int(model.best_iteration) + 1
    # Refit on all the training rows with the tree count the inner split chose,
    # the two-stage fit `finalize_models.py` uses: early stopping finds the
    # count, then the shipped model is fit with that count fixed. Scoring a
    # model saved straight from early stopping overcounts its rounds.
    final = dict(XGB_BASE)
    final.pop("early_stopping_rounds")
    final.update(best)
    final["n_estimators"] = best_trees
    from xgboost import XGBClassifier, XGBRegressor
    model = (XGBClassifier(eval_metric="logloss", **final)
             if market.classification
             else XGBRegressor(eval_metric="mae", **final))
    model.fit(train[columns], train[y], verbose=False)
    return model, dict(best), best_trees


# ---------------------------------------------------------------- baselines

def naive_scores(market, train, evaluate_on):
    """Winner: the training home-win rate. Margin/total: the training mean."""
    y = market.target
    if market.classification:
        rate = float(train[y].astype(float).mean())
        proba = np.full(len(evaluate_on), rate, dtype=np.float64)
        labels = (proba > 0.5).astype(int)
        return score(market, evaluate_on[y], labels, proba), {"rate": rate}
    mean = float(train[y].mean())
    return (score(market, evaluate_on[y],
                  np.full(len(evaluate_on), mean, dtype=np.float64)),
            {"mean": mean})


def elo_alone_scores(market, elo_variant, train, evaluate_on):
    """Elo's own prediction, with no other feature.

    Winner: the ledger's probability, which already carries home advantage and
    drops it at a neutral site. Margin: a one-variable map from the rating
    difference, fitted on the training rows. Total: Elo says nothing about how
    many points are scored, so there is no Elo baseline and naive is the only
    one - which the caller reports rather than silently omitting.
    """
    y = market.target
    if market.classification:
        proba = np.asarray(evaluate_on[ELO_PROB_COLUMN[elo_variant]],
                           dtype=np.float64)
        labels = (proba > 0.5).astype(int)
        return score(market, evaluate_on[y], labels, proba), {}
    if market.key == "total":
        return None, {}
    from sklearn.linear_model import LinearRegression
    home, away = ELO_COLUMNS[elo_variant]
    fit_x = (train[home] - train[away]).to_numpy().reshape(-1, 1)
    model = LinearRegression().fit(fit_x, train[y])
    x = (evaluate_on[home] - evaluate_on[away]).to_numpy().reshape(-1, 1)
    return (score(market, evaluate_on[y], model.predict(x)),
            {"slope": float(model.coef_[0]),
             "intercept": float(model.intercept_)})


# ---------------------------------------------------------------- bootstrap
BOOTSTRAP_RESAMPLES = 10_000
# §5.3's trigger for a seed re-draw: a nearer bound this close to zero is not
# the same claim as one that clears it, which §40 established for the NBA and
# the G League then had to act on at a bound of -0.0004.
NEARER_BOUND_TRIGGER = {"log_loss": 0.0005, "mae": 0.02}
REDRAW_SEEDS = tuple(range(10))


def per_row_loss(market, y_true, predictions, proba=None):
    """The per-game terms whose MEAN is the metric, for the paired bootstrap.

    NOT A SECOND DEFINITION OF THE METRIC. A bootstrap over games needs the
    decomposition rather than the aggregate, and `assert_decomposes` requires
    the mean of these terms to equal `model_evaluation`'s own number - so if the
    two ever disagree this raises instead of quietly reporting a different
    quantity than the headline score.
    """
    y = np.asarray(y_true, dtype=np.float64)
    if market.classification:
        p = np.clip(np.asarray(proba, dtype=np.float64), 1e-15, 1 - 1e-15)
        return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return np.abs(y - np.asarray(predictions, dtype=np.float64))


def assert_decomposes(market, y_true, predictions, proba=None, tol=1e-12):
    """The mean of the per-row terms IS the shared scorer's metric."""
    metrics = score(market, y_true, predictions, proba)
    mine = float(np.mean(per_row_loss(market, y_true, predictions, proba)))
    theirs = primary(market, metrics)
    if abs(mine - theirs) > tol:
        raise AssertionError(
            f"{market.key}: per-row mean {mine!r} does not reproduce "
            f"model_evaluation's {theirs!r} (difference {mine - theirs:.3e}). "
            f"The bootstrap would be measuring a different quantity than the "
            f"reported score.")
    return theirs


def paired_bootstrap(model_losses, baseline_losses, seed=SEED,
                     resamples=BOOTSTRAP_RESAMPLES):
    """95% interval on mean(model) - mean(baseline), resampling GAMES.

    Paired: one game contributes both its losses to the same resample, so the
    interval is on the difference rather than on two independent means.
    """
    difference = (np.asarray(model_losses, dtype=np.float64)
                  - np.asarray(baseline_losses, dtype=np.float64))
    rng = np.random.default_rng(seed)
    n = len(difference)
    draws = rng.integers(0, n, size=(resamples, n))
    means = difference[draws].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"difference": float(difference.mean()),
            "lo": float(lo), "hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "games": n, "resamples": resamples, "seed": seed}


def needs_redraw(market, interval):
    """Is the nearer bound close enough to zero that one draw is not enough?"""
    if not interval["excludes_zero"]:
        return False
    nearer = min(abs(interval["lo"]), abs(interval["hi"]))
    key = "log_loss" if market.classification else "mae"
    return nearer < NEARER_BOUND_TRIGGER[key]
