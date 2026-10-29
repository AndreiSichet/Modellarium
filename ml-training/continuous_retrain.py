r"""Retrain the seven team-level models on current data, and refuse to promote
a set that is worse than what is already serving.

THE COMPARISON, AND WHY IT IS SHAPED THIS WAY

    production = the SHIPPED models, scored as-is. Not refitted.
    test       = the most recent 500 games, every one of them strictly after
                 production's `trained_through`, so neither side has seen any
                 of them.
    candidate  = finalize_models.py's own procedure - early stopping on the
                 last 500 rows to choose a tree count, then a refit on all of
                 the candidate's rows at that count - applied to every game
                 before the test window.
    verdict    = no market more than 2% worse, AND every market keeping at
                 least half of production's advantage over its naive baseline.

Both sides are therefore built the same way and differ only in their data
cutoff, which is the one thing a retrain changes.

WHAT THIS REPLACED, AND WHY THE OLD RULE WAS A LOTTERY RATHER THAN A BIAS

The old gate kept the candidate's early-stopped model and compared it with
production's architecture refitted at its full frozen tree count. That was
recorded as a systematic bias against the candidate. Replaying six past
cutoffs on 2026-10-26 showed something less tidy and more awkward: the old
rule refused **once** in six, on totals, not on moneyline, and its control -
a candidate trained on production's own rows - **promoted**, at +0.25%.

The real mechanism is that moneyline's early-stopping point is unstable. It
lands anywhere from 21 to 329 trees depending on which 500 games happen to
form the validation window, while production's count is a frozen constant. So
the old comparison was a lottery between a noisy number and a fixed one, and
on the data of October 2026 it happened to sit at the bad end - candidate 21
trees against production's 67, measured twice at about +6.2% and +6.5%. It
would not have refused every Monday; it would have refused unpredictably,
which is worse, because a verdict nobody can anticipate is a verdict nobody
can learn from.

Making both sides fully identical - same procedure AND same rows - was tried
and rejected: it compares a model with itself, exactly 0.00% on every market
with bit-identical predictions, so it promotes unconditionally.
`probes/symmetric_rule_is_vacuous.py` is that measurement.

WHY 500 GAMES, AND WHY THE JOB WILL SAY "NOT ENOUGH" UNTIL JANUARY

A paired bootstrap over the same six cutoffs gives the standard deviation of
this gate's own change% as a function of the window: 1.59 points at 100 games,
1.00 at 300, 0.63 at 500, all on moneyline, which is the noisiest market by
about 1.4x. Turned into the question that matters - how often would a
candidate IDENTICAL to production be refused by sampling noise alone - that is
26.5% at 100 games, 3.3% at 300, and 0.1% at 500.

500 is therefore the smallest window that can tell a 2% change from noise, and
the test window must lie entirely after production's cutoff. With 100 new
training games on top, 600 games must exist after the cutoff before this job
will train. At the NBA's ~7.1-7.4 games a day that is reached around
**2027-01-11** for a production cutoff of 2026-04-12 - so the Mondays from
2026-11-02 to 2027-01-04 report "not enough new games (N of 600)", which is
green and says so. The consequence is deliberate: this gate can act about
twice a season, not weekly. A smaller window would not be a faster gate, it
would be a gate whose verdicts are mostly noise.

TWO THINGS THIS GATE CANNOT SEE, MEASURED AND ACCEPTED

  1. A candidate trained on materially staler data. A plant that truncated the
     candidate's training rows a FULL SEASON earlier than production's cutoff
     was PROMOTED, every market between -0.96% and +0.34% - inside the noise.
     A year of extra history is worth nothing measurable here, which is
     consistent with the project's information-ceiling finding and with the
     rolling features being within-season. `trained_through` and the
     new-games count are the only defence against stale training data; this
     comparison is not one.
  2. A market where production itself barely beats no skill. That is what the
     naive guard is for - see `naive_guard` - but where production has no edge
     at all the guard cannot bite either, and the table says so rather than
     implying a check happened.

Exit codes: 0 a verdict was reached (including "not enough new games"),
1 the gate refused, 2 no comparison was reached at all.
"""

import argparse
import hashlib
import io
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from common import FEATURE_COLUMNS
from finalize_models import (
    INPUT_TABLES, PROCESSED_DIR, code_commit, final_params,
    raw_corpus_fingerprint, sha256_of,
)
from model_evaluation import TASKS, evaluate, load_model, primary
from train_baseline import (
    REGRESSION_TARGETS, load_dataset, naive_prediction, section,
)
from train_moneyline_xgb import PARAMS as MONEYLINE_PARAMS
from train_regression_xgb import PARAMS as REGRESSION_PARAMS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = PROJECT_ROOT / "data-pipeline"

# Imported rather than restated, so the season the summary compares against is
# the same one the fetcher actually uses.
if str(PIPELINE / "ingestion") not in sys.path:
    sys.path.insert(0, str(PIPELINE / "ingestion"))
from fetch_games import current_season_start_year, season_label  # noqa: E402
GAMES_FINAL = PIPELINE / "data" / "processed" / "games_final.csv"
# The same file as git knows it, for reading the COMMITTED version.
GAMES_FINAL_IN_REPO = "data-pipeline/data/processed/games_final.csv"

PRODUCTION_DIR = Path(__file__).resolve().parent / "models"
CANDIDATE_DIR = Path(__file__).resolve().parent / "models_candidate"

TRAINED_THROUGH = "trained_through.json"

# Each market's no-skill baseline, DERIVED from the same list the training
# scripts and the recorded baselines use, so the gate's naive figure cannot
# drift from the one in the frozen tables. Moneyline carries no
# (stat, combine) pair because its no-skill forecast is a base rate, not a
# rolling average - see `naive_metric`.
NAIVE_BY_KEY = {"moneyline": (None, None)}
NAIVE_BY_KEY.update({
    label.lower().replace(" ", "_"): (stat, combine)
    for _target, label, stat, combine in REGRESSION_TARGETS
})

REFRESH_STEPS = [
    PIPELINE / "ingestion" / "fetch_games.py",
    PIPELINE / "ingestion" / "fetch_player_boxscores.py",
    PIPELINE / "preprocessing" / "validate_games.py",
    PIPELINE / "preprocessing" / "build_games_table.py",
    PIPELINE / "preprocessing" / "build_rolling_features.py",
    PIPELINE / "preprocessing" / "build_rest_days.py",
    PIPELINE / "preprocessing" / "build_elo_ratings.py",
    PIPELINE / "preprocessing" / "build_player_rolling_minutes.py",
    PIPELINE / "preprocessing" / "build_team_availability.py",
    PIPELINE / "ingestion" / "fetch_quarter_scores.py",
    PIPELINE / "preprocessing" / "build_quarter_half_raw.py",
    PIPELINE / "preprocessing" / "build_quarter_half_rolling.py",
    PIPELINE / "preprocessing" / "build_final_dataset.py",
]

# THE TEST WINDOW IS 500 GAMES BECAUSE 500 IS WHAT RESOLVES THE TOLERANCE,
# and that was measured rather than chosen. A paired bootstrap over six
# replayed cutoffs (`probes/gate_replay.py`, 2026-10-26) gives the standard
# deviation of the gate's own change% as a function of the window:
#
#   window   moneyline sd   P(a candidate IDENTICAL to production is refused)
#      100          1.59                                              26.5%
#      300          1.00                                               3.3%
#      500          0.63                                               0.1%
#
# At 300 games one run in thirty would refuse a copy of production, and the
# widest 95% band is nearly twice the 2% tolerance. Moneyline drives it - a
# log loss over 500 binary outcomes is the noisiest of the seven markets, at
# about 1.4x the next worst. A SMALLER WINDOW WOULD NOT BE A FASTER GATE, it
# would be a gate whose verdicts are mostly noise.
DEFAULT_TEST_GAMES = 500

DEFAULT_TOLERANCE_PCT = 2.0

# NEW TRAINING GAMES, NOT THE THRESHOLD. This is how much data the candidate
# must have that production has not seen; the number of games that must exist
# after production's cutoff before the job will train is
# `games_required()` = test window + this = 600. The old name was
# `DEFAULT_MIN_NEW_GAMES = 100`, which read as "100 games and we go" and was
# the threshold the workflow passed - so the two numbers were conflated in
# every message. Nothing says 100 where 600 is meant any more.
MIN_NEW_TRAINING_GAMES = 100

# The candidate's tree count is chosen by early stopping against the last
# VALIDATION_GAMES of its own training rows, which is finalize_models.py's
# procedure transposed onto a rolling cutoff.
VALIDATION_GAMES = 500

# A market's model must keep at least this share of production's advantage
# over its naive baseline. See `naive_guard`.
NAIVE_IMPROVEMENT_FRACTION = 0.5

BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261026
EPS = 1e-15


def games_required(test_games: int) -> int:
    """Games needed after production's cutoff before the job will train.

    The test window must be entirely unseen by BOTH sides, and production is
    used as-is, so every test game has to postdate production's cutoff. On top
    of that the candidate needs training data production has not seen, or it
    is the same model. Hence window + new training games, derived in one place
    so no message can quote half of it.
    """
    return test_games + MIN_NEW_TRAINING_GAMES

# A crash and a refusal must not share a code: one is the absence of a
# measurement, the other is a measurement.
EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_ERROR = 2

def summarise(source) -> dict:
    """The one reader. `source` is anything pandas.read_csv accepts."""
    frame = pd.read_csv(source, usecols=["GAME_ID", "GAME_DATE", "SEASON"])
    return {
        "rows": len(frame),
        "latest": str(pd.to_datetime(frame["GAME_DATE"]).max().date()),
        "season": int(frame["SEASON"].max()),
    }

def snapshot(path: Path) -> dict:
    """Enough of games_final.csv to tell whether anything new arrived."""
    if not path.exists():
        return {"rows": 0, "latest": None}
    return summarise(path)

def committed_snapshot():
    """The same summary for the COMMITTED file, never the working tree.

    `git show HEAD:<path>` deliberately: reading the working-tree file a second
    time would compare it against itself and report agreement forever, which is
    the one way this check could look correct while measuring nothing.

    Returns None if the committed version cannot be read - no git, no commit,
    or a file that is not a readable CSV. Reported, never fatal.
    """
    try:
        result = subprocess.run(
            ["git", "show", f"HEAD:{GAMES_FINAL_IN_REPO}"],
            cwd=PROJECT_ROOT, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    if not result.stdout.strip():
        return None
    try:
        return summarise(io.BytesIO(result.stdout))
    except (ValueError, KeyError):
        return None

def report_refresh_disposition(after: dict, refreshed: bool) -> None:
    """Say what the rebuild produced, and what became of it.

    The job rebuilds games_final.csv BEFORE the new-data check, but the
    workflow's `git add` for it sits inside the promoted-only step. So on any
    skip the refreshed data is built on the runner and dropped.

    OPERATOR-FACING. The discard is the designed behaviour; the problem is that
    it reads identically to a week where the refresh found nothing. Deliberately
    not a failure and not a ::warning:: - a warning that fires most weeks is one
    people stop reading.
    """
    served = committed_snapshot()
    built = "rebuilt by this run" if refreshed else "working tree (not rebuilt)"

    print(f"\n  {built:<26}: {after['rows']:,} team-game rows, "
          f"latest {after['latest']}")

    if served is None:
        print("  committed (what is served): could not be read via "
              "`git show HEAD:`.")
        print("  -> cannot say whether this run produced anything new.")
        return

    print(f"  {'committed (served)':<26}: {served['rows']:,} team-game rows, "
          f"latest {served['latest']}")

    gained = after["rows"] - served["rows"]
    if gained <= 0 and after["latest"] == served["latest"]:
        print("  -> identical. Nothing new was found, so nothing is being "
              "discarded.")
        return

    print(f"  -> THIS RUN BUILT {gained:,} ROW(S) THAT ARE BEING DISCARDED.")
    print("     games_final.csv is committed only on a promotion, so on a skip")
    print("     the rebuild stays on the runner and never reaches the served")
    print("     image. Advancing served data is a manual pipeline run plus an")
    print("     image rebuild.")

def report_season_coverage(after: dict) -> None:
    """Say whether the data covers the season the calendar says is underway.

    OPERATOR-FACING, not correctness. Without it a quiet week and a season the
    fetcher is not looking at produce byte-identical output - "nothing worth
    retraining on" - and the second stays invisible for months. The fetcher's
    season list used to be hardcoded, which is exactly how that happens.
    """
    calendar_season = current_season_start_year()
    data_season = after.get("season")
    if data_season is None:
        return

    print(f"\n  newest season in {GAMES_FINAL.name}: {season_label(data_season)}")
    print(f"  season underway by the calendar : {season_label(calendar_season)}")

    if data_season >= calendar_season:
        print("  -> the data covers the current season. A genuinely quiet week.")
        return

    print(f"  -> THE DATA IS {calendar_season - data_season} SEASON(S) BEHIND "
          "THE CALENDAR.")
    print("     Either that season has not tipped off yet (legitimate in early")
    print("     October), or the pipeline is not fetching it. Check")
    print("     fetch_games.seasons_through() before reading this as quiet.")


def read_trained_through(directory: Path):
    """What data the models in this directory were trained on, or None."""
    path = directory / TRAINED_THROUGH
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def write_trained_through(directory: Path, rows: pd.DataFrame) -> None:
    """The marker describes WHAT THESE WEIGHTS WERE FITTED ON, nothing wider.

    `rows` is the candidate's own training set - every game before the test
    window - not the whole dataset. It used to be the dataset, which was a
    quiet lie once the gate held a window back: the marker would have claimed
    the models had seen games they were deliberately never shown, and the next
    run would have computed "new games since" from a cutoff 500 games too
    late, and under-counted its own new data every time.
    """
    (directory / TRAINED_THROUGH).write_text(
        json.dumps({
            "trained_through": str(rows["GAME_DATE"].max().date()),
            "games": int(len(rows)),
        }, indent=2),
        encoding="utf-8",
    )

def refresh_data() -> None:
    """Rerun ingestion and preprocessing, in order, stopping on failure."""
    section("REFRESHING DATA")
    for step in REFRESH_STEPS:
        # flush=True, OR THE LOG READS IN THE WRONG ORDER. Each step is a
        # subprocess that inherits this process's stdout and writes to it
        # directly, while this print sits in Python's buffer until the buffer
        # fills. Redirected to a file - which is how both the workflow and any
        # local capture run it - the headers therefore land long after the
        # output they label, and a 2026-10-29 chain run put all thirteen of
        # them AFTER every step's output. The header is the only thing that
        # says which step produced what, so it has to be on the stream before
        # the child starts writing.
        print(f"\n--- {step.name}", flush=True)
        result = subprocess.run(
            [sys.executable, str(step)], cwd=str(PROJECT_ROOT), check=False
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{step.name} failed with exit code {result.returncode}. "
                f"Stopping: every later step reads what this one writes, so "
                f"continuing would train on a half-built dataset."
            )

def cutoff_split(dataset, cutoff, test_games: int) -> tuple:
    """The most recent `test_games` games as test; everything before as the
    candidate's rows. EVERY TEST GAME MUST POSTDATE `cutoff`.

    `cutoff` is production's `trained_through`. Production is compared AS-IS -
    not refitted - so the only honest test window is one it has never seen,
    and the check below is an assertion rather than a filter: if the window
    reaches back past the cutoff the caller has already decided to train on
    too little data, and silently shrinking the window would hide that.
    """
    ordered = dataset.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    test = ordered.iloc[-test_games:]
    rows = ordered.iloc[:-test_games]

    oldest_test = test["GAME_DATE"].min()
    if oldest_test <= cutoff:
        raise RuntimeError(
            f"the {test_games}-game test window starts {oldest_test.date()}, "
            f"on or before production's cutoff {pd.Timestamp(cutoff).date()}, "
            f"so production has already seen part of it. "
            f"{games_required(test_games):,} games must exist after the "
            f"cutoff before this comparison means anything."
        )
    if len(rows) < VALIDATION_GAMES * 3:
        raise RuntimeError(
            f"{len(rows):,} training games is too few to carve a "
            f"{VALIDATION_GAMES}-game validation window out of."
        )

    def span(frame):
        return (f"{frame['GAME_DATE'].min().date()} to "
                f"{frame['GAME_DATE'].max().date()}")

    print(f"  candidate rows {len(rows):>6,} games   {span(rows)}")
    print(f"    of which validation {VALIDATION_GAMES:>4,} games   "
          f"{span(rows.iloc[-VALIDATION_GAMES:])}")
    print(f"  test           {len(test):>6,} games   {span(test)}   "
          f"<- the gate window, all after {pd.Timestamp(cutoff).date()}")
    return rows, test


def train_candidate(target: str, classification: bool, rows):
    """finalize_models.py's procedure, on the candidate's rows.

    Early stopping against the last `VALIDATION_GAMES` picks a tree count,
    then the model is REFIT on all of `rows` at that count - which is what
    `finalize_models.train_final` does, and what makes the candidate the same
    kind of object as the shipped models it is compared against.

    The old version kept the early-stopped model and never refit, while
    production's side was refitted at its full frozen count. That difference
    decided the verdict, and not in a fixed direction: moneyline's stopping
    point is unstable, landing anywhere from 21 to 329 trees depending on
    which games form the validation window, so comparing it against a frozen
    constant was a lottery rather than a systematic bias. See the module
    docstring.
    """
    train = rows.iloc[:-VALIDATION_GAMES]
    validation = rows.iloc[-VALIDATION_GAMES:]

    params = MONEYLINE_PARAMS if classification else REGRESSION_PARAMS
    probe = (XGBClassifier if classification else XGBRegressor)(**params)
    probe.fit(train[FEATURE_COLUMNS], train[target],
              eval_set=[(validation[FEATURE_COLUMNS], validation[target])],
              verbose=False)
    trees = int(probe.best_iteration)

    model = (XGBClassifier if classification else XGBRegressor)(
        **final_params(classification, trees))
    model.fit(rows[FEATURE_COLUMNS], rows[target], verbose=False)
    return model, trees


def per_game_losses(model, frame, target: str, classification: bool):
    """Each game's own contribution to the market's primary metric.

    float64 BEFORE reducing - the project's standing rule. XGBoost returns
    float32 and summing a few hundred log terms at that width lands
    measurably away from the float64 answer.
    """
    x = frame[FEATURE_COLUMNS]
    y = frame[target].to_numpy(dtype="float64")
    if classification:
        p = np.clip(model.predict_proba(x)[:, 1].astype("float64"),
                    EPS, 1.0 - EPS)
        return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return np.abs(y - model.predict(x).astype("float64"))


def change_of(candidate_losses, production_losses, index=None) -> float:
    """The gate's number, on one set of games."""
    if index is not None:
        candidate_losses = candidate_losses[index]
        production_losses = production_losses[index]
    production = production_losses.mean()
    return (candidate_losses.mean() - production) / production * 100.0


def bootstrap_band(candidate_losses, production_losses, rng,
                   draws: int = BOOTSTRAP_DRAWS) -> tuple:
    """A 95% band for the change%, by resampling THE SAME games for both.

    The pairing is the point: an unpaired bootstrap would add the two sides'
    independent sampling noise and report a band far wider than the
    comparison actually carries. Measured on real windows the band is about
    +/-1.6 points at 500 games, which is why 500 is the window.

    THIS DOES NOT CHANGE THE VERDICT - that is still the point estimate
    against the tolerance. It is printed so that a +2.3% refusal with a
    +/-1.3 band cannot be read like a +2.3% refusal with a +/-0.2 one.
    """
    n = len(candidate_losses)
    out = np.empty(draws, dtype="float64")
    for i in range(draws):
        idx = rng.integers(0, n, size=n)
        out[i] = change_of(candidate_losses, production_losses, idx)
    return (float(np.percentile(out, 2.5)),
            float(np.percentile(out, 97.5)),
            float(out.std(ddof=1)))


def naive_metric(rows, test, target: str, classification: bool,
                 stat: str, combine: str) -> float:
    """The no-skill score on the test window, in the market's own metric.

    Regression reuses `train_baseline.naive_prediction` - each team's own
    ROLL10 average, differenced or summed - so the gate's baseline is the
    same object the recorded baselines were measured with.

    THE BASELINE IS THE BEST CONSTANT PREDICTOR, NOT THE ROLL10 ONE, AND THAT
    WAS FORCED BY MEASUREMENT. The first version used
    `train_baseline.naive_prediction` - each team's own ROLL10 average - to
    match the recorded baselines. On a real gate window that returns **NaN**
    for every regression market, because the window straddles a season
    opener and the first games of a season have no ROLL10 columns yet. A NaN
    baseline made `production_gain` NaN, which made the guard inert on all
    six regressions: it looked like a check and was doing nothing. Measured
    2026-10-26 on the 2025-03-10..2025-11-21 window.

    So the reference is the best *constant* predictor, which is always
    computable and is the textbook no-skill model: the training rows' median
    for an MAE (the median, not the mean, is what minimises absolute error)
    and their base rate for a log loss. It is taken from the CANDIDATE'S
    TRAINING ROWS, never from the test window - a constant fitted on the test
    games would be peeking and would make the baseline unbeatably good.

    `roll10_metric` still reports the ROLL10 figure as context where it is
    finite, because that is the number the recorded baselines use.
    """
    y = test[target].to_numpy(dtype="float64")
    if classification:
        rate = float(rows[target].mean())
        rate = min(max(rate, EPS), 1.0 - EPS)
        value = float(-(y * np.log(rate)
                        + (1.0 - y) * np.log(1.0 - rate)).mean())
    else:
        value = float(np.abs(y - float(rows[target].median())).mean())

    # AN UNKNOWN BASELINE IS NOT A SATISFIED GUARD. If this is ever not a
    # finite number the guard cannot say anything, and saying nothing quietly
    # is how the ROLL10 version passed six markets without checking them.
    if not np.isfinite(value):
        raise RuntimeError(
            f"the no-skill baseline for {target} came out {value!r}, so the "
            f"guard cannot be evaluated. Refusing to treat an unknown "
            f"baseline as a passed check."
        )
    return value


def roll10_metric(test, target: str, classification: bool,
                  stat: str, combine: str):
    """The project's recorded ROLL10 baseline, or None where it is undefined.

    Context for the table only - never the guard's reference. It is undefined
    on any window containing a season's opening games, which a rolling gate
    window routinely does.
    """
    if classification:
        return None
    predicted = naive_prediction(test, stat, combine).to_numpy(dtype="float64")
    value = float(np.abs(test[target].to_numpy(dtype="float64")
                         - predicted).mean())
    return value if np.isfinite(value) else None


def bootstrap_gain_low(production_losses, naive: float, rng,
                       draws: int = BOOTSTRAP_DRAWS) -> float:
    """The 2.5th percentile of production's OWN edge over the naive baseline.

    The guard needs to know whether production has an edge at all before it
    can demand a share of it. This resamples the same games and returns the
    low end of that edge's 95% band; at or below zero, production is not
    measurably better than no skill on this window.
    """
    out = np.empty(draws, dtype="float64")
    n = len(production_losses)
    for i in range(draws):
        idx = rng.integers(0, n, size=n)
        out[i] = naive - production_losses[idx].mean()
    return float(np.percentile(out, 2.5))


def naive_guard(candidate: float, production: float, naive: float,
                production_gain_low: float,
                fraction: float = NAIVE_IMPROVEMENT_FRACTION) -> dict:
    """Did the candidate keep enough of production's edge over no skill?

    WHY A SECOND TEST AT ALL. The tolerance compares the candidate with
    production and nothing else, so it cannot see a market where production
    itself is barely better than guessing - and in such a market a candidate
    with NO signal scores almost the same as one with all of it. Measured
    2026-10-26: a candidate trained on SHUFFLED targets came out only +0.41%
    on reb_total and passed the tolerance, because reb_total's real model
    beats its naive baseline by so little that destroying it costs almost
    nothing.

    So each market must also keep at least `fraction` of production's
    advantage over its own naive baseline, both measured on the same test
    window.

    AND ON THREE MARKETS IT STILL CANNOT HELP, WHICH IS WORTH KNOWING BEFORE
    RELYING ON IT. On a 500-game window `reb_margin`, `reb_total` and
    `ast_margin` have no edge over a constant predictor that is
    distinguishable from zero, so the guard is `informative: false` there and
    the tolerance alone governs them - a mean-predictor candidate passes
    `reb_total` at about +1.21%. That does not let a bad set through, because
    all seven markets ship together and the other six block it, but the guard
    is not a per-market safety net on the low-skill markets and must not be
    described as one.

    THE DEGENERATE CASE MUST PASS, AND GETTING THAT WRONG FAILED THE CONTROL.
    If production has no edge of its own (`production_gain <= 0`, i.e. the
    shipped model is no better than the naive baseline on this window) then
    half of it is a NEGATIVE floor, and `candidate_gain >= 0.5 *
    production_gain` is then *harder* to satisfy the closer the candidate is
    to production. The first version applied that literally and refused a
    candidate that WAS production, byte for byte, with a change of exactly
    0.000000% - caught by `verify_retrain_gate.py`'s control on the first run.

    So where production has no edge the guard cannot be a floor and does not
    pretend to be one: it passes, `informative` is False, and the table says
    the guard is not a check on that market this week. The tolerance still
    governs it.

    AND "NO EDGE" MEANS "NO *MEASURABLE* EDGE", WHICH IS THE THIRD VERSION OF
    THIS FUNCTION. Requiring only `production_gain > 0` made `reb_total` the
    new systematic blocker - the very pathology this gate was rebuilt to
    remove from moneyline. Replayed at six cutoffs on 2026-10-26, two refused
    the whole set on `reb_total` alone at changes of **+0.42%** and **+0.21%**,
    deep inside the tolerance, because production's entire edge over a
    constant there was **0.016-0.025 MAE** and half of that is a threshold
    about ten times finer than the sampling noise: the kept-fraction's 95%
    band spanned roughly [-3, +2]. The guard was deciding coin flips.

    So the guard acts only where production's own edge is distinguishable
    from zero - `production_gain_low`, the low end of that edge's 95%
    bootstrap band, must be above zero. A no-signal candidate still fails,
    because where production has a real edge a mean-predictor keeps none of
    it; a sub-1% wobble on an immeasurable edge no longer refuses anything.
    """
    production_gain = naive - production
    candidate_gain = naive - candidate
    informative = production_gain > 0 and production_gain_low > 0
    required = fraction * production_gain
    return {
        "naive": naive,
        "production_gain": production_gain,
        "production_gain_low_95": production_gain_low,
        "candidate_gain": candidate_gain,
        "required_gain": required if informative else None,
        "kept_fraction": (candidate_gain / production_gain
                          if informative else None),
        "informative": bool(informative),
        "passed": bool(candidate_gain >= required) if informative else True,
    }


def run_gate(rows, test, production_dir, tolerance_pct: float) -> list:
    """Train each candidate, score the SHIPPED models as-is, compare.

    PRODUCTION IS NOT REFITTED. It is the artifact that is actually serving,
    scored on games it has never seen. That is the design: both sides are
    built by the same procedure and the only difference between them is the
    data cutoff, which is the thing a retrain is supposed to change.

    Equalising the rows as well would make the two sides THE SAME MODEL -
    measured, exactly 0.00% on every market with bit-identical predictions
    (`probes/symmetric_rule_is_vacuous.py`) - so that rule promotes
    unconditionally and is no gate at all.
    """
    results = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for key, label, target, classification in TASKS:
        candidate, trees = train_candidate(target, classification, rows)
        shipped = load_model(key, classification, production_dir)

        candidate_metrics = evaluate(candidate, test, target, classification)
        production_metrics = evaluate(shipped, test, target, classification)
        cand = primary(candidate_metrics, classification)
        prod = primary(production_metrics, classification)

        candidate_losses = per_game_losses(candidate, test, target,
                                           classification)
        production_losses = per_game_losses(shipped, test, target,
                                            classification)
        change_pct = change_of(candidate_losses, production_losses)
        low, high, sd = bootstrap_band(candidate_losses, production_losses, rng)

        stat, combine = NAIVE_BY_KEY[key]
        naive = naive_metric(rows, test, target, classification, stat, combine)
        guard = naive_guard(cand, prod, naive,
                            bootstrap_gain_low(production_losses, naive, rng))
        guard["roll10_baseline"] = roll10_metric(test, target, classification,
                                                 stat, combine)

        results.append({
            "key": key, "label": label, "target": target,
            "classification": classification,
            "candidate": cand, "production": prod,
            "change_pct": change_pct,
            "band_low": low, "band_high": high, "band_sd": sd,
            "tolerance_passed": bool(change_pct <= tolerance_pct),
            "naive": guard,
            "passed": bool(change_pct <= tolerance_pct and guard["passed"]),
            "candidate_trees": trees,
            "production_trees": shipped.get_booster().num_boosted_rounds(),
            "candidate_metrics": candidate_metrics,
            "production_metrics": production_metrics,
            "model": candidate,
        })

    return results


ECE_WARN_MULTIPLE = 2.0

def calibration_note(results) -> None:
    """Report calibration. It informs the verdict; it does not decide it.

    Brier and ECE are shown, not gated on. The gate's tolerance is 2%, a number
    chosen against observed month-to-month movement in log loss and MAE; there
    is no equivalent observation for ECE, because no real retrain has run yet.
    A threshold invented now would be a guess wearing a number's clothes, and
    the failure mode is the expensive direction - refusing a good candidate.

    So this warns and lets the log-loss verdict stand. Revisit once several
    real retrains have recorded ECE, and set the bar from that spread.
    """
    classified = [r for r in results if r["classification"]]
    if not classified:
        return

    print()
    for r in classified:
        c, p = r["candidate_metrics"], r["production_metrics"]
        if p["ece"] > 0 and c["ece"] > p["ece"] * ECE_WARN_MULTIPLE:
            state = ("PASSED on log loss" if r["passed"]
                     else "already FAILED on log loss")
            print(f"  WARNING: {r['label']} {state}, and its ECE is "
                  f"{c['ece'] / p['ece']:.1f}x production's "
                  f"({c['ece']:.4f} against {p['ece']:.4f}).")
            if r["passed"]:
                print("  The ranking held up but the probabilities did not.")
                print("  Look at the reliability curve before merging -")
                print("  measure_calibration.py prints it.")
            else:
                print("  Both are worse, so this adds to the refusal rather")
                print("  than complicating it.")
        else:
            print(f"  Calibration: {r['label']} ECE {c['ece']:.4f} against "
                  f"production's {p['ece']:.4f} - no warning.")
    print(f"  ECE is not a gate criterion yet; see calibration_note.")

def print_table(results, tolerance_pct: float) -> None:
    section(f"PROMOTION GATE (a candidate may be up to {tolerance_pct:.1f}% worse)")
    print("PRODUCTION is the SHIPPED model, scored as-is on games it has never")
    print("seen - not a refit. The candidate was built by finalize_models.py's")
    print("own procedure on every game before the window, so the two sides")
    print("differ only in their data cutoff.\n")

    width = max(len(r["label"]) for r in results)
    print(f"{'TARGET':<{width}}  {'METRIC':<9}{'PRODUCTION':>11}{'CANDIDATE':>11}"
          f"{'CHANGE':>9}{'95% BAND':>18}{'TREES c/p':>11}"
          f"{'NAIVE':>9}{'KEPT':>8}   {'VERDICT':<22}")
    print("-" * (width + 101))
    for r in results:
        metric = "log loss" if r["classification"] else "MAE"
        trees = f"{r['candidate_trees']}/{r['production_trees']}"
        band = f"[{r['band_low']:+.2f}, {r['band_high']:+.2f}]"
        guard = r["naive"]
        kept = ("-" if guard["kept_fraction"] is None
                else f"{guard['kept_fraction'] * 100:.0f}%")
        if r["passed"]:
            verdict = "pass"
        elif not r["tolerance_passed"] and not guard["passed"]:
            verdict = "FAIL - both tests"
        elif not r["tolerance_passed"]:
            verdict = "FAIL - regressed"
        else:
            verdict = "FAIL - naive guard"
        print(f"{r['label']:<{width}}  {metric:<9}{r['production']:>11.4f}"
              f"{r['candidate']:>11.4f}{r['change_pct']:>+8.2f}%{band:>18}"
              f"{trees:>11}{guard['naive']:>9.4f}{kept:>8}   {verdict:<22}")

    calibration_note(results)

    print("\n(negative change = the candidate is better)")
    print()
    print("THE VERDICT IS THE POINT ESTIMATE AGAINST THE TOLERANCE. The 95%")
    print("band is a paired bootstrap over the same games and is printed, not")
    print("applied - it is there so a refusal at the edge of the tolerance is")
    print("not read as decisively as one well past it. At this window size the")
    print("band is normally about +/-1.6 points on moneyline and +/-1.3 on the")
    print("rest; a band much wider than that means the window is too small and")
    print("the verdict is mostly sampling noise.")
    print()
    print("NAIVE is each market's no-skill score on the same window: the best")
    print("CONSTANT predictor, fitted on the training rows - their median for")
    print("an MAE, their base rate for moneyline. It is not the ROLL10 naive")
    print("of the recorded baselines, which comes out NaN on any window that")
    print("includes a season's opening games and so cannot be a guard at all.")
    print("KEPT is how much of production's")
    print("advantage over that baseline the candidate still has. A market that")
    print("keeps less than "
          f"{NAIVE_IMPROVEMENT_FRACTION * 100:.0f}% of it fails even if it is "
          "inside the")
    print("tolerance, because in a market where production barely beats the")
    print("baseline a model with NO signal is also inside the tolerance.")
    degenerate = [r["label"] for r in results if not r["naive"]["informative"]]
    if degenerate:
        print()
        print(f"  NOTE: production has no edge over the naive baseline on "
              f"{', '.join(degenerate)},")
        print("  so the naive guard cannot bite there. KEPT reads '-'. That is")
        print("  reported rather than hidden: the guard is not a check on those")
        print("  markets this week.")

def promotion_hash_mismatch(output_dir: Path, evaluated: dict) -> list:
    """Artifacts on disk that no longer hash to the model the gate graded.

    The hazard is narrow and real: the verdict is reached on an in-memory
    model, and what gets promoted is a file. Anything that rewrites the file
    between those two moments - a stale artifact from a previous run that was
    not cleaned, a half-finished write, a hand edit - would ship weights no
    verdict describes. Comparing the digest of what was written against the
    digest taken at evaluation time closes that gap, and a plant in
    `verify_retrain_gate.py` proves it closes.
    """
    return sorted(key for key, digest in evaluated.items()
                  if not (output_dir / f"{key}.json").is_file()
                  or sha256_of(output_dir / f"{key}.json") != digest)


def _verdict_word(row) -> str:
    """One word per market, naming WHICH test failed."""
    if row["passed"]:
        return "pass"
    if not row["tolerance_passed"] and not row["naive"]["passed"]:
        return "FAIL - regressed and below the naive guard"
    if not row["tolerance_passed"]:
        return "FAIL - regressed"
    return "FAIL - below the naive guard"


def write_candidate_provenance(output_dir: Path, dataset, results,
                               args) -> Path:
    """What produced these candidates, written beside them.

    THE PROMOTED WEIGHTS MUST CARRY THEIR OWN RECORD. Before this, only
    `finalize_models.py` wrote a provenance file, and the weekly job never
    runs it - so a promotion would have copied new weights into
    `ml-training/models/` on top of a provenance.json describing the PREVIOUS
    ones, and committed the pair. New weights with an old record is a
    misleading green of the most expensive kind: every number in it looks
    right and none of it is about the files beside it.
    `verify_promotion_provenance.py` is the guard that makes that impossible;
    this function is what lets the guard pass honestly.

    THERE IS NO MLFLOW RUN ID, and that is recorded rather than faked. This
    script logs nothing to MLflow - `run_gate` trains candidates in-process -
    so there is no run to point at. A `gate_run_id` is generated here from the
    clock and the inputs so two retrains can be told apart, and the absence of
    an MLflow run is stated so nobody goes looking for one.
    """
    started = datetime.now(timezone.utc)
    dataset_path = PROCESSED_DIR / "model_dataset.csv"
    dataset_hash = (sha256_of(dataset_path) if dataset_path.is_file()
                    else "ABSENT")
    gate_run_id = hashlib.sha256(
        (started.isoformat() + dataset_hash).encode("utf-8")).hexdigest()[:16]

    tables = {}
    for name in INPUT_TABLES:
        path = PROCESSED_DIR / name
        tables[name] = sha256_of(path) if path.is_file() else "ABSENT"

    models = {}
    for row in results:
        path = output_dir / f"{row['key']}.json"
        models[row["key"]] = {
            "file": path.name,
            "sha256": sha256_of(path) if path.is_file() else "ABSENT",
            "trees": row["candidate_trees"],
            "production_trees": row["production_trees"],
            "metric": "log_loss" if row["classification"] else "mae",
            "candidate": row["candidate"],
            "production_as_shipped": row["production"],
            "change_pct": row["change_pct"],
            "change_pct_band_95": [row["band_low"], row["band_high"]],
            "change_pct_bootstrap_sd": row["band_sd"],
            "tolerance_passed": row["tolerance_passed"],
            "naive_guard": row["naive"],
            "verdict": _verdict_word(row),
            "candidate_metrics": {k: v for k, v in
                                  row["candidate_metrics"].items()},
            "production_metrics": {k: v for k, v in
                                   row["production_metrics"].items()},
        }

    body = {
        "note": (
            "Written by continuous_retrain.py for the CANDIDATES in this "
            "directory. model_dataset.csv is not committed; it is rebuilt "
            "from the raw corpus by the build chain, which is why its sha256 "
            "is recorded here. There is no MLflow run id because this script "
            "does not log to MLflow - see gate_run_id."
        ),
        "written_at": started.isoformat(),
        "gate_run_id": gate_run_id,
        "mlflow_run_id": None,
        "source": "continuous_retrain.py",
        "gate": {
            "rule": (
                "production = the shipped models as-is, no refit; candidate = "
                "finalize_models.py's procedure (early stopping on the last "
                f"{VALIDATION_GAMES} rows for a tree count, then a refit on "
                "all candidate rows at that count) applied to every game "
                "before the test window; test = the most recent "
                f"{args.test_games} games, all strictly after production's "
                "trained_through. Verdict = the point estimate against the "
                "tolerance, AND each market keeping at least "
                f"{NAIVE_IMPROVEMENT_FRACTION:.0%} of production's advantage "
                "over its naive baseline."
            ),
            "validation_games": VALIDATION_GAMES,
            "naive_improvement_fraction": NAIVE_IMPROVEMENT_FRACTION,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "tolerance_pct": args.tolerance_pct,
            "test_games": args.test_games,
            "min_new_training_games": args.min_new_training_games,
            "games_required_after_cutoff": games_required(args.test_games),
            "production_dir": str(args.production_dir),
            "verdict": ("all passed"
                        if all(r["passed"] for r in results)
                        else "REJECTED"),
            "passed": sum(1 for r in results if r["passed"]),
            "of": len(results),
        },
        "data_cutoff": str(dataset["GAME_DATE"].max().date()),
        "code": code_commit(),
        "model_dataset": {
            "path": "data-pipeline/data/processed/model_dataset.csv",
            "sha256": dataset_hash,
            "rows": int(len(dataset)),
            "committed": False,
        },
        "input_tables": tables,
        "raw_corpus": raw_corpus_fingerprint(dataset),
        "feature_columns": list(FEATURE_COLUMNS),
        "models": models,
    }
    path = output_dir / "provenance.json"
    path.write_text(json.dumps(body, indent=2, default=str) + chr(10),
                    encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-refresh", action="store_true",
                        help="do not rerun ingestion/preprocessing")
    parser.add_argument("--production-dir", type=Path, default=PRODUCTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=CANDIDATE_DIR)
    parser.add_argument("--test-games", type=int, default=DEFAULT_TEST_GAMES)
    parser.add_argument("--tolerance-pct", type=float, default=DEFAULT_TOLERANCE_PCT)
    parser.add_argument("--min-new-training-games", type=int,
                        default=MIN_NEW_TRAINING_GAMES,
                        dest="min_new_training_games",
                        help="games the candidate must have "
                             "that production has not seen; "
                             "the number that must exist "
                             "after the cutoff is this plus "
                             "the test window")
    parser.add_argument("--force", action="store_true",
                        help="retrain regardless of how little new data exists")
    args = parser.parse_args()

    section("NEW DATA CHECK")
    before = snapshot(GAMES_FINAL)
    print(f"  before refresh: {before['rows']:,} team-game rows, "
          f"latest {before['latest']}")

    if args.skip_refresh:
        print("  --skip-refresh: pipeline not rerun.")
    else:
        refresh_data()

    after = snapshot(GAMES_FINAL)
    print(f"\n  after refresh:  {after['rows']:,} team-game rows, "
          f"latest {after['latest']}")

    section("DATA PREP")
    dataset = load_dataset()
    print(f"Loaded {len(dataset):,} games.")

    marker = read_trained_through(args.production_dir)
    if marker is None:
        print(f"\n  No {TRAINED_THROUGH} beside the production models, so how "
              f"much data\n  they have already seen is unknown. Not guessing.")
        if not args.force:
            # RED, NOT GREEN, AND THIS USED TO RETURN EXIT_OK.
            #
            # A missing marker means the job cannot tell how much data the
            # shipped weights have already seen, so it refuses - which is
            # right. What was wrong is that it refused QUIETLY: EXIT_OK with
            # no candidates is classified by the workflow as `nothing`, the
            # same word a genuinely quiet week gets, and the run went green.
            # The weekly job had therefore been a no-op from the day the
            # models were finalised until 2026-10-24, because
            # finalize_models.py did not write the marker and only a
            # candidate-producing run did.
            #
            # The old message claimed this "resolves itself after one run".
            # It could not: the skip happens BEFORE candidates are written, so
            # no marker produced no candidate produced no marker - a loop that
            # reported success every Monday.
            #
            # EXIT_ERROR rather than EXIT_REFUSED, because the gate did not
            # refuse a candidate: no comparison was ever reached, which is
            # exactly what exit 2 means in this project's contract.
            print("  REFUSING, and this run is RED. Nothing was compared, so "
                  "there is no\n  verdict to report. The marker is written by "
                  "finalize_models.py beside\n  the models it produces - if "
                  "it is missing, the shipped artifacts were\n  not produced "
                  "by a run that recorded itself.")
            print("  Pass --force to retrain anyway.")
            report_refresh_disposition(after, not args.skip_refresh)
            report_season_coverage(after)
            return EXIT_ERROR
        print("  --force given: continuing.")
    else:
        cutoff = pd.Timestamp(marker["trained_through"])
        new_games = int((dataset["GAME_DATE"] > cutoff).sum())
        print(f"\n  production trained through {marker['trained_through']} "
              f"({marker['games']:,} games)")
        print(f"  genuinely new games since then: {new_games:,}")

        required = games_required(args.test_games)
        print(f"  needed before a comparison means anything: {required:,} "
              f"({args.test_games:,} test window + "
              f"{args.min_new_training_games:,} new training games)")

        if new_games < required and not args.force:
            print(f"\n  NOT ENOUGH NEW GAMES: {new_games:,} of {required:,}.")
            print("  The test window must lie entirely after production's")
            print("  cutoff, because production is compared as-is, and the")
            print("  candidate needs rows production has not seen or it is the")
            print("  same model. Neither is true yet.")
            print("  Pass --force to retrain anyway.")
            report_refresh_disposition(after, not args.skip_refresh)
            report_season_coverage(after)
            return EXIT_OK
        if new_games < required:
            print(f"\n  Only {new_games:,} of {required:,} new games, but "
                  f"--force was given. The window below may overlap")
            print("  production's training data, in which case the split "
                  "refuses.")

    print()
    rows, test = cutoff_split(dataset, cutoff, args.test_games)

    section("TRAINING CANDIDATES AND SCORING THE SHIPPED MODELS")
    print(f"Shipped models read from {args.production_dir}, scored AS-IS.")
    print("The candidate is built by finalize_models.py's procedure on the")
    print("rows above; neither side has seen the test window.")
    results = run_gate(rows, test, args.production_dir, args.tolerance_pct)

    print_table(results, args.tolerance_pct)

    passed = [r for r in results if r["passed"]]
    section("RESULT")
    print(f"{len(passed)} of {len(results)} targets passed the gate.")

    if len(passed) != len(results):
        failed = ", ".join(r["label"] for r in results if not r["passed"])
        print(f"\nNOT PROMOTING ANY MODEL. Regressed: {failed}")
        print("All seven ship together - they share a feature set and a")
        print("dataset, so promoting a subset would leave the served models")
        print("trained on different data as each other.")
        report_refresh_disposition(after, not args.skip_refresh)
        return EXIT_REFUSED

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.output_dir.glob("*.json"):
        stale.unlink()

    # THE PROMOTED ARTIFACT IS THE EVALUATED ONE. `r["model"]` is the object
    # the table above scored - trained on `rows`, which excludes the test
    # window. It is NOT refitted to include the window before shipping, which
    # would promote weights no verdict was ever reached about: the gate would
    # have graded one model and shipped another, and the 500 games it held
    # back would be the only games the shipped model had seen that the graded
    # one had not. The hash recorded here is what `verify_promotion_provenance`
    # and the check below compare against.
    evaluated_hashes = {}
    for r in results:
        path = args.output_dir / f"{r['key']}.json"
        r["model"].save_model(str(path))
        evaluated_hashes[r["key"]] = sha256_of(path)

    write_trained_through(args.output_dir, rows)

    mismatched = promotion_hash_mismatch(args.output_dir, evaluated_hashes)
    if mismatched:
        print(f"\n  REFUSING TO PROMOTE: {len(mismatched)} artifact(s) on disk "
              f"no longer hash to the")
        print(f"  evaluated candidate: {', '.join(sorted(mismatched))}.")
        print("  Something rewrote them between evaluation and promotion, so")
        print("  the verdict above does not describe these files.")
        return EXIT_ERROR
    provenance = write_candidate_provenance(args.output_dir, dataset,
                                            results, args)

    print(f"\nAll passed. {len(results)} candidate models written to "
          f"{args.output_dir}")
    print(f"A {TRAINED_THROUGH} marker was written alongside them, so the next")
    print("run can tell how much data these have already seen.")
    print(f"{provenance.name} records what produced them - the gate verdict, "
          f"the table")
    print("hashes and the commit. verify_promotion_provenance.py refuses a")
    print("promotion whose provenance does not describe the weights beside it.")
    print("ml-training/models/ is UNTOUCHED. Promoting these is a separate,")
    print("human decision - a pull request once the scheduled job exists.")
    return EXIT_OK

if __name__ == "__main__":
    try:
        code = main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        print()
        print("=" * 78)
        print("CRASHED BEFORE REACHING A VERDICT")
        print("=" * 78)
        print("This is NOT a gate refusal. No comparison was made, so nothing")
        print("above says anything about whether a retrained model would be")
        print("better or worse than production.")
        print()
        print(f"Exiting {EXIT_ERROR} so the caller can tell the two "
              f"apart. A real refusal exits {EXIT_REFUSED}.")
        code = EXIT_ERROR
    raise SystemExit(code)
