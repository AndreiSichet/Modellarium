r"""A calibrator for the NHL's Elo winner. Chosen on validation; test is description.

WHY THIS IS WORTH TRYING HERE, more clearly than for the NFL. Elo alone has
**no calibration step at all** - it is a raw logistic of a rating difference
with a fitted home advantage, and nobody ever fitted it to produce calibrated
probabilities. Its test ECE is 0.042365 against the NBA's 0.0215, and its
worst bins are [0.4, 0.5) predicting 0.4557 against 0.5342 observed and
[0.7, 0.8) predicting 0.7366 against 0.6294.

THE DECISION RULE, STATED BEFORE THE NUMBERS so it cannot be chosen to fit
them. A calibrator ships only if, on the 2023-24 VALIDATION season:

  1. the paired 95% interval for the **ECE difference** EXCLUDES ZERO on the
     improving side - calibration is the thing being bought, so it is the
     thing that has to be established; and
  2. the paired 95% interval for the **log-loss difference** does NOT exclude
     zero on the worsening side - a calibrator that tidies the bins while
     making the probabilities worse overall is a bad trade, and the NFL's
     calibrator entry predicts exactly that outcome ("expect the tail bins
     to kill it").

If neither candidate clears both, **the winner ships UNCALIBRATED and that is
recorded** rather than quietly retried with a third family.

OUT-OF-FOLD MEANS LEAVE-ONE-SEASON-OUT ACROSS THE TRAINING SEASONS. For each
training season, the calibrator is fitted on the other nine and applied to
that one; the concatenation is an honest picture of what calibration does to
data the calibrator has not seen. Elo's own predictions are pre-game by
construction so there is nothing to leak from the model side - the fold
structure is for the CALIBRATOR, which is a fit like any other.

ISOTONIC OUTPUT IS CLIPPED and the reason is the unknown-is-never-a-zero rule
in its probabilistic form: an isotonic bin containing no wins will happily
return exactly 0.0, which asserts certainty from a handful of games and makes
log loss explode on the first counterexample. The clip is stated in the
record rather than hidden in a default.

    python ml-training/nhl/calibrate_nhl_winner.py --table <path> \
        --elo-json <path> --json out.json
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import nhl_features as F                                        # noqa: E402
from nhl_features import refuse_undeclared_seasons              # noqa: E402
from nhl_baselines import accuracy, brier, log_loss             # noqa: E402

MODELS_DIR = PROJECT / "ml-training" / "models_nhl"
CALIBRATION_BINS = 10
BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261031

# An isotonic bin with no wins returns exactly 0.0. See the module docstring.
ISOTONIC_CLIP = (0.001, 0.999)
LOGIT_CLIP = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), LOGIT_CLIP, 1.0 - LOGIT_CLIP)
    return np.log(p / (1.0 - p))


def reliability(actual, predicted, bins=CALIBRATION_BINS):
    """Reliability table, ECE, and the Murphy decomposition.

    THE SAME FUNCTION SHAPE AS finalize_nhl_models.reliability, deliberately
    recomputed here rather than imported, because this script must be able to
    report a BEFORE table that matches the frozen one exactly - and the check
    that it does is check 1 below. If it were imported, agreement would prove
    nothing about the frozen record.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(predicted, edges[1:-1], right=False),
                    0, bins - 1)
    overall = float(actual.mean())
    rows, ece, rel, res = [], 0.0, 0.0, 0.0
    for b in range(bins):
        pick = index == b
        n = int(pick.sum())
        if not n:
            rows.append({"bin": f"[{edges[b]:.1f}, {edges[b + 1]:.1f})",
                         "n": 0, "mean_predicted": None, "observed": None,
                         "gap": None})
            continue
        mean_predicted = float(predicted[pick].mean())
        observed = float(actual[pick].mean())
        weight = n / len(actual)
        ece += weight * abs(mean_predicted - observed)
        rel += weight * (mean_predicted - observed) ** 2
        res += weight * (observed - overall) ** 2
        rows.append({"bin": f"[{edges[b]:.1f}, {edges[b + 1]:.1f})", "n": n,
                     "mean_predicted": mean_predicted, "observed": observed,
                     "gap": mean_predicted - observed})
    return {"bins": rows, "ece": float(ece), "reliability": float(rel),
            "resolution": float(res),
            "reliability_share": (float(rel / (rel + res)) if (rel + res)
                                  else None),
            "base_rate": overall}


def expected_calibration_error(actual, predicted, bins=CALIBRATION_BINS):
    """ECE alone, for the bootstrap's inner loop."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(predicted, edges[1:-1], right=False),
                    0, bins - 1)
    total = 0.0
    for b in range(bins):
        pick = index == b
        n = int(pick.sum())
        if n:
            total += (n / len(actual)) * abs(predicted[pick].mean()
                                             - actual[pick].mean())
    return float(total)


# ==========================================================================
# the two families
# ==========================================================================

class Platt:
    """Logistic regression on the LOGIT of the raw probability.

    PLATT SCALING ON THE SCORE, NOT ON THE PROBABILITY. Fitting a logistic
    on p itself would be a logistic of a logistic and would fight its own
    link function; the standard form maps the score, and for Elo the score
    IS logit(p) up to a constant - the rating difference over 400 ln 10. So
    this is the textbook object, not an approximation of it.
    """

    name = "platt"

    def __init__(self):
        self.model = LogisticRegression(max_iter=5000)

    def fit(self, predicted, actual):
        self.model.fit(logit(predicted).reshape(-1, 1),
                       np.asarray(actual, dtype=float))
        return self

    def apply(self, predicted):
        return self.model.predict_proba(
            logit(predicted).reshape(-1, 1))[:, 1]


class Isotonic:
    """Monotone step function, clipped away from 0 and 1."""

    name = "isotonic"

    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip",
                                        y_min=ISOTONIC_CLIP[0],
                                        y_max=ISOTONIC_CLIP[1])

    def fit(self, predicted, actual):
        self.model.fit(np.asarray(predicted, dtype=float),
                       np.asarray(actual, dtype=float))
        return self

    def apply(self, predicted):
        return np.clip(self.model.predict(
            np.asarray(predicted, dtype=float)), *ISOTONIC_CLIP)


FAMILIES = (Platt, Isotonic)


def null_ece(probabilities, rng, draws=4000, bins=CALIBRATION_BINS):
    """ECE a PERFECTLY CALIBRATED predictor would show on this many rows.

    THE QUESTION NOBODY ASKS AND SHOULD. ECE averages absolute gaps, so
    sampling noise can only push it UP - it is biased upward and the bias
    depends on n. Comparing a 12,036-row training ECE with a 2,624-row test
    ECE is therefore comparing two different quantities, and "the test ECE is
    eight times the training ECE" can be true of a predictor with no
    miscalibration at all.

    This draws outcomes FROM the model's own probabilities, so by
    construction the predictor is perfectly calibrated, and the spread of the
    resulting ECE is the null. An observed ECE inside it establishes nothing;
    one outside it is real.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    out = np.empty(draws, dtype=float)
    for i in range(draws):
        drawn = (rng.random(len(probabilities)) < probabilities).astype(float)
        out[i] = expected_calibration_error(drawn, probabilities, bins)
    return out


def paired_interval(actual, before, after, metric, rng):
    """95% interval for metric(after) - metric(before), paired by row."""
    n = len(actual)
    draws = rng.integers(0, n, size=(BOOTSTRAP, n))
    deltas = np.empty(BOOTSTRAP, dtype=float)
    for i in range(BOOTSTRAP):
        pick = draws[i]
        deltas[i] = (metric(actual[pick], after[pick])
                     - metric(actual[pick], before[pick]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"observed": float(metric(actual, after)
                              - metric(actual, before)),
            "sd": float(deltas.std()), "ci_low": float(lo),
            "ci_high": float(hi),
            "improves": bool(hi < 0.0), "worsens": bool(lo > 0.0),
            "crosses_zero": bool(lo < 0.0 < hi)}


def print_table(title, table):
    print(f"  {title}")
    print(f"    {'bin':<14}{'n':>7}{'mean predicted':>17}{'observed':>11}"
          f"{'gap':>10}")
    print("    " + "-" * 59)
    for row in table["bins"]:
        if not row["n"]:
            print(f"    {row['bin']:<14}{0:>7}{'-':>17}{'-':>11}{'-':>10}")
            continue
        print(f"    {row['bin']:<14}{row['n']:>7}"
              f"{row['mean_predicted']:>17.4f}{row['observed']:>11.4f}"
              f"{row['gap']:>+10.4f}")
    share = table["reliability_share"]
    print(f"    ECE {table['ece']:.6f}   reliability "
          f"{table['reliability']:.6f}   resolution "
          f"{table['resolution']:.6f}"
          + (f"   share {share * 100:.1f}%" if share is not None else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=None)
    parser.add_argument("--elo-json", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    payload = json.loads(args.elo_json.read_text(encoding="utf-8"))
    elo = {"k": payload["k"], "home_advantage": payload["home_advantage"],
           "carryover": payload["carryover"]}

    games = F.load_games(args.table)
    frame = F.build_features(games, elo=elo, carry_rolling=True,
                             carry_rest=True)
    refuse_undeclared_seasons(frame)
    frame = (frame.sort_values(["GAME_DATE", "GAME_ID"])
             .reset_index(drop=True))

    frame["RAW"] = np.asarray(F.elo_win_probability(
        frame["HOME_ELO"], frame["AWAY_ELO"], elo["home_advantage"]),
        dtype=float)

    seasons = F.season_order(frame)
    train_seasons = [s for s in seasons if s not in F.TEST_SEASONS
                     and s not in F.VALIDATION_SEASONS]
    train = frame["SEASON"].isin(set(train_seasons))
    valid = frame["SEASON"].isin(set(F.VALIDATION_SEASONS))
    test = frame["SEASON"].isin(set(F.TEST_SEASONS))

    print(f"Elo k={elo['k']:g} home_advantage={elo['home_advantage']:g} "
          f"carryover={elo['carryover']:g}  (raw, uncalibrated)")
    print(f"training {int(train.sum())} over {len(train_seasons)} season(s) "
          f"| validation {int(valid.sum())} | test {int(test.sum())}")
    print("THE RULE, fixed before the numbers: a calibrator ships only if "
          "validation's")
    print("  ECE interval EXCLUDES zero on the improving side AND its log "
          "loss interval")
    print("  does NOT exclude zero on the worsening side.")
    print()

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    # ================================================================
    # 0. IS THERE ANY MISCALIBRATION TO FIX?
    # ================================================================
    # ASKED FIRST, BECAUSE A CALIBRATOR FOR A WELL-CALIBRATED MODEL CAN ONLY
    # DO HARM. Each row's observed ECE is compared with what a perfectly
    # calibrated predictor would show on that many rows.
    print("IS THERE ANY MISCALIBRATION TO FIX? "
          "(observed ECE against a perfectly-calibrated null)")
    print(f"    {'slice':<30}{'n':>7}{'observed':>11}"
          f"{'null median':>13}{'null p95':>11}{'p':>8}")
    print("    " + "-" * 80)
    report_null = {}
    slices = [("training pooled", train), ("validation", valid),
              ("test pooled", test)]
    slices += [(f"test {season}", frame["SEASON"] == season)
               for season in F.TEST_SEASONS]
    for label, mask in slices:
        actual = frame.loc[mask, "HOME_WIN"].to_numpy(dtype=float)
        raw = frame.loc[mask, "RAW"].to_numpy(dtype=float)
        observed = expected_calibration_error(actual, raw)
        null = null_ece(raw, rng)
        probability = float((null >= observed).mean())
        verdict = ("inside the null" if probability > 0.05
                   else "REAL miscalibration")
        print(f"    {label:<30}{int(mask.sum()):>7}{observed:>11.6f}"
              f"{float(np.median(null)):>13.6f}"
              f"{float(np.percentile(null, 95)):>11.6f}"
              f"{probability:>8.4f}  {verdict}")
        report_null[label] = {
            "n": int(mask.sum()), "observed_ece": observed,
            "null_median": float(np.median(null)),
            "null_p95": float(np.percentile(null, 95)),
            "p_null_at_least_observed": probability,
            "real_miscalibration": probability <= 0.05}
    print("    ECE is biased UPWARD by noise and the bias depends on n, so a")
    print("    training ECE and a test ECE are NOT comparable quantities.")
    print()

    report = {"elo": elo, "bins": CALIBRATION_BINS,
              "null_test": report_null,
              "bootstrap": {"resamples": BOOTSTRAP, "seed": BOOTSTRAP_SEED},
              "isotonic_clip": list(ISOTONIC_CLIP),
              "rule": "ships only if validation's ECE interval excludes "
                      "zero on the improving side and its log-loss interval "
                      "does not exclude zero on the worsening side",
              "out_of_fold": {}, "validation": {}, "test": {}}

    # ================================================================
    # 1. out-of-fold across the training seasons
    # ================================================================
    print("OUT-OF-FOLD ACROSS THE TRAINING SEASONS "
          "(leave one season out, 10 folds)")
    y_train = frame.loc[train, "HOME_WIN"].to_numpy(dtype=float)
    raw_train = frame.loc[train, "RAW"].to_numpy(dtype=float)
    before_oof = reliability(y_train, raw_train)
    print_table("BEFORE - raw Elo on the training seasons", before_oof)
    report["out_of_fold"]["before"] = before_oof

    for family in FAMILIES:
        out_of_fold = np.empty(int(train.sum()), dtype=float)
        season_of = frame.loc[train, "SEASON"].to_numpy()
        for held in train_seasons:
            others = season_of != held
            fitted = family().fit(raw_train[others], y_train[others])
            out_of_fold[~others] = fitted.apply(raw_train[~others])
        table = reliability(y_train, out_of_fold)
        print_table(f"AFTER - {family.name}, out of fold", table)
        print(f"      log loss {log_loss(y_train, out_of_fold):.6f} "
              f"against raw {log_loss(y_train, raw_train):.6f}")
        report["out_of_fold"][family.name] = dict(
            table, log_loss=log_loss(y_train, out_of_fold),
            raw_log_loss=log_loss(y_train, raw_train))
    print()

    # ================================================================
    # 2. the choice, on validation
    # ================================================================
    print("THE CHOICE, ON THE 2023-24 VALIDATION SEASON")
    y_valid = frame.loc[valid, "HOME_WIN"].to_numpy(dtype=float)
    raw_valid = frame.loc[valid, "RAW"].to_numpy(dtype=float)
    before_valid = reliability(y_valid, raw_valid)
    print_table("BEFORE - raw Elo on validation", before_valid)
    report["validation"]["before"] = dict(
        before_valid, log_loss=log_loss(y_valid, raw_valid),
        brier=brier(y_valid, raw_valid),
        accuracy=accuracy(y_valid, raw_valid))

    eligible = {}
    for family in FAMILIES:
        fitted = family().fit(raw_train, y_train)
        calibrated = fitted.apply(raw_valid)
        table = reliability(y_valid, calibrated)
        ece_interval = paired_interval(y_valid, raw_valid, calibrated,
                                       expected_calibration_error, rng)
        loss_interval = paired_interval(y_valid, raw_valid, calibrated,
                                        log_loss, rng)
        passes = ece_interval["improves"] and not loss_interval["worsens"]
        print_table(f"AFTER - {family.name}, fitted on the training seasons",
                    table)
        print(f"      ECE      {ece_interval['observed']:>+10.6f}  "
              f"[{ece_interval['ci_low']:+.6f}, "
              f"{ece_interval['ci_high']:+.6f}]  "
              + ("improves" if ece_interval["improves"]
                 else "worsens" if ece_interval["worsens"]
                 else "indistinguishable"))
        print(f"      log loss {loss_interval['observed']:>+10.6f}  "
              f"[{loss_interval['ci_low']:+.6f}, "
              f"{loss_interval['ci_high']:+.6f}]  "
              + ("improves" if loss_interval["improves"]
                 else "WORSENS" if loss_interval["worsens"]
                 else "indistinguishable"))
        print(f"      -> {'CLEARS' if passes else 'does NOT clear'} the rule")
        report["validation"][family.name] = dict(
            table, log_loss=log_loss(y_valid, calibrated),
            brier=brier(y_valid, calibrated),
            accuracy=accuracy(y_valid, calibrated),
            ece_interval=ece_interval, log_loss_interval=loss_interval,
            clears_rule=passes)
        if passes:
            eligible[family.name] = (family, table["ece"])
    print()

    # ================================================================
    # 3. the verdict
    # ================================================================
    if not eligible:
        chosen = None
        why = ("neither Platt nor isotonic cleared the rule on validation, "
               "so the winner SHIPS UNCALIBRATED. Recorded rather than "
               "retried with a third family: the rule was fixed before the "
               "numbers, and picking a family until one passes is how a "
               "threshold stops meaning anything.")
        print("VERDICT: SHIP UNCALIBRATED")
    else:
        chosen = min(eligible, key=lambda n: eligible[n][1])
        why = (f"{chosen} cleared the rule on validation with the lowest "
               f"ECE of the candidates that did")
        print(f"VERDICT: SHIP THE {chosen.upper()} CALIBRATOR")
    print(f"  {why}")
    report["chosen"] = chosen
    report["reason"] = why

    # ================================================================
    # 4. test - DESCRIPTION ONLY
    # ================================================================
    print()
    print("TEST - DESCRIPTION ONLY, never a re-decision")
    y_test = frame.loc[test, "HOME_WIN"].to_numpy(dtype=float)
    raw_test = frame.loc[test, "RAW"].to_numpy(dtype=float)
    before_test = reliability(y_test, raw_test)
    print_table("BEFORE - raw Elo on test (this is the FROZEN table)",
                before_test)
    report["test"]["before"] = dict(
        before_test, log_loss=log_loss(y_test, raw_test),
        brier=brier(y_test, raw_test), accuracy=accuracy(y_test, raw_test))

    # Both families are reported on test whatever the verdict was, because
    # "what would it have done" is the question a reader asks next, and
    # leaving it out invites someone to re-run the choice on test to find out.
    train_plus = train | valid
    y_plus = frame.loc[train_plus, "HOME_WIN"].to_numpy(dtype=float)
    raw_plus = frame.loc[train_plus, "RAW"].to_numpy(dtype=float)
    for family in FAMILIES:
        fitted = family().fit(raw_plus, y_plus)
        calibrated = fitted.apply(raw_test)
        table = reliability(y_test, calibrated)
        shipped = chosen == family.name
        print_table(f"AFTER - {family.name}, fitted on training+validation"
                    + ("   <- SHIPPED" if shipped else
                       "   (description; not shipped)"), table)
        print(f"      log loss {log_loss(y_test, calibrated):.6f} against "
              f"raw {log_loss(y_test, raw_test):.6f}   "
              f"ECE {table['ece']:.6f} against raw {before_test['ece']:.6f}")
        report["test"][family.name] = dict(
            table, log_loss=log_loss(y_test, calibrated),
            brier=brier(y_test, calibrated),
            accuracy=accuracy(y_test, calibrated), shipped=shipped)
        if shipped:
            args.out.mkdir(parents=True, exist_ok=True)
            joblib.dump(fitted, args.out / "winner_calibrator.joblib")
            print(f"      written: {args.out / 'winner_calibrator.joblib'}")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=float),
                             encoding="utf-8")
        print()
        print(f"written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
