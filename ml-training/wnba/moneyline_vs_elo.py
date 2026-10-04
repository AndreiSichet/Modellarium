"""Is the WNBA moneyline model reliably worse than Elo alone on test?

Phase 3 reported two point estimates from the test seasons - the selected
model at 0.6130 log loss and bare Elo at 0.6046 - and recorded the gap as a
caveat on the artifact. It never put an interval on it. That interval is what
decides whether anything user-facing is warranted: a difference whose interval
spans zero is not a finding to show a reader.

THIS IS NOT A SECOND PASS OVER THE TEST SET IN THE SENSE THE GUARD FORBIDS.
test_wnba_models.py's receipt exists to stop the test seasons being used to
re-choose a model. Nothing here changes the selection - the shipped artifact
is unaffected whatever this prints. It attaches an uncertainty interval to a
comparison that was already made and already reported, which is a property of
that same single evaluation rather than a new question asked of the data.

The recorded means are reproduced first, and the script refuses to report an
interval if they do not match: an interval computed over a differently-fitted
model would be an interval about the wrong thing.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from wnba_common import (  # noqa: E402
    MODELS_DIR, TARGETS, TEST_SEASONS, TRAIN_SEASONS, VALIDATION_SEASONS,
    feature_columns, load_dataset, load_long, section, with_fold_elo,
)
from select_wnba_models import expected_score  # noqa: E402

FIT_SEASONS = TRAIN_SEASONS + VALIDATION_SEASONS
EPS = 1e-15
TOLERANCE = 1e-9
SEEDS = range(10)


def log_loss_per_game(truth, probability):
    p = np.clip(probability, EPS, 1 - EPS)
    return -(truth * np.log(p) + (1 - truth) * np.log(1 - p))


def paired_bootstrap(a_errors, b_errors, seed, iterations=2000):
    """Paired bootstrap on (a - b). Both score the same rows, in order."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a_errors, float), np.asarray(b_errors, float)
    n = len(a)

    deltas = np.empty(iterations)
    for i in range(iterations):
        idx = rng.integers(0, n, n)
        deltas[i] = a[idx].mean() - b[idx].mean()

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(deltas.mean()), float(lo), float(hi), bool(lo * hi <= 0)


def main() -> int:
    print(__doc__)

    selection = json.loads((MODELS_DIR / "selection.json").read_text())["selection"]
    recorded = json.loads((MODELS_DIR / "test_results.json").read_text())["results"]
    choice = selection["moneyline"]
    window = choice["window"]
    features = feature_columns(window)
    label = TARGETS["moneyline"]["label"]

    section("REPRODUCING THE RECORDED TEST NUMBERS FIRST")
    print(f"  selection: {choice['family']}/{window}")

    dataset = load_dataset()
    long_frame = load_long()
    fold_data, elo_info = with_fold_elo(dataset, long_frame, FIT_SEASONS)
    print(f"  Elo refit on {FIT_SEASONS[0]}-{FIT_SEASONS[-1]}: "
          f"K={elo_info['k']}, carryover={elo_info['carryover']:.3f}")

    usable = fold_data.dropna(subset=features)
    fit = usable[usable["SEASON"].isin(FIT_SEASONS)]
    test = usable[usable["SEASON"].isin(TEST_SEASONS)]

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(fit[features], fit[label])

    truth = test[label].to_numpy()
    model_err = log_loss_per_game(
        truth, model.predict_proba(test[features])[:, 1])
    elo_err = log_loss_per_game(
        truth, expected_score(test["HOME_TEAM_ELO"].to_numpy(),
                              test["AWAY_TEAM_ELO"].to_numpy()))

    want = recorded["moneyline"]
    checks = [
        ("games", len(test), want["n"]),
        ("model log loss", float(model_err.mean()), want["model"]),
        ("Elo log loss", float(elo_err.mean()), want["elo"]),
    ]

    ok = True
    for name, got, expected in checks:
        same = abs(got - expected) < TOLERANCE if isinstance(got, float) else got == expected
        ok &= same
        print(f"  {name:<16} {got!r:<22} recorded {expected!r:<22} "
              f"{'match' if same else 'MISMATCH'}")

    if not ok:
        print("\n  REFUSING to report an interval: this is not the evaluation "
              "phase 3 recorded,\n  so an interval from it would be about a "
              "different model.")
        return 1

    section("THE INTERVAL ON model - Elo alone")
    print("""Positive means the model loses MORE than Elo, so a positive interval that
excludes zero is the model being reliably worse. Paired, because both score
the same games in the same order.
""")

    mean, lo, hi, spans = paired_bootstrap(model_err, elo_err, seed=0)
    print(f"  model    {model_err.mean():.6f}")
    print(f"  Elo      {elo_err.mean():.6f}")
    print(f"  model - Elo  {mean:+.6f}   95% CI [{lo:+.6f}, {hi:+.6f}]")
    print(f"  spans zero: {spans}")

    # Any interval whose nearer bound sits close to zero is re-drawn across
    # seeds before "excludes zero" is claimed - the method lesson from the
    # absence-aggregation study, where a bound of -0.0011 needed ten seeds.
    section("SEED STABILITY")
    verdicts = []
    for seed in SEEDS:
        _, s_lo, s_hi, s_spans = paired_bootstrap(model_err, elo_err, seed=seed)
        verdicts.append(s_spans)
        print(f"  seed {seed}: [{s_lo:+.6f}, {s_hi:+.6f}]  "
              f"{'spans zero' if s_spans else 'EXCLUDES zero'}")

    spanning = sum(verdicts)
    print(f"\n  {spanning} of {len(verdicts)} seeds span zero")

    section("VERDICT")
    if spanning == len(verdicts):
        print("""  THE INTERVAL SPANS ZERO ON EVERY SEED.

  Elo's lower point estimate is not a reliable difference on 589 test games.
  So there is no measured finding here to put in front of a reader, and
  nothing user-facing is warranted. The gap stays recorded in manifest.json,
  which is where a model-selection note written for engineers belongs.""")
    elif spanning == 0:
        print("""  THE INTERVAL EXCLUDES ZERO ON EVERY SEED.

  The model is reliably worse than a single-feature closed-form formula on
  the test seasons. That is a real weakness in the market and warrants a
  visible, machine-readable label - the same treatment q1_winner gets - not
  prose.""")
    else:
        print(f"""  SEED-DEPENDENT: {spanning} of {len(verdicts)} seeds span zero.

  Not a reliable difference. Treated as spanning zero, because a conclusion
  that depends on the resampling seed is not a conclusion.""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
