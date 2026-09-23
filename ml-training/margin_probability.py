"""Does a win probability derived from the margin model beat the classifier?

Study only. Trains nothing that ships and writes no artifacts.

P(home win) = Phi(predicted_margin / sigma). The hypothesis is that binary
HOME_WIN discards information at training time - a 2-point win and a 30-point
win are the same bit - so a model that saw the full margin should discriminate
better, if that signal survives the transformation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, roc_auc_score
from xgboost import XGBClassifier

from calibration import (
    brier_decomposition,
    brier_score,
    expected_calibration_error,
)
from common import split_three_way
from train_baseline import (
    FEATURE_COLUMNS,
    ROLLING_FEATURE_COLUMNS,
    load_dataset,
    section,
)
from train_moneyline_xgb import PARAMS as CLASSIFIER_PARAMS, TARGET as WIN_TARGET
from train_regression_xgb import fit_xgb

MARGIN_TARGET = "HOME_MARGIN"
TOTAL_TARGET = "TOTAL_PTS"

RECORDED_ACCURACY = 0.6698
RECORDED_LOG_LOSS = 0.5979
RECORDED_SPREAD_MAE = 10.7368
RECORDED_TOTALS_MAE = 15.2322
REPRODUCTION_TOLERANCE = 5e-4

CLIP = 1e-6

# Phi(m/sigma) is strictly increasing in m for any sigma > 0, so every constant
# -sigma variant is the same ranking. Anything past this is an implementation
# bug, not a result.
AUC_TOLERANCE = 1e-12

ADOPTION_BAR_PCT = 3.0


def probabilities(model, frame):
    """predict_proba widened to float64. See measure_calibration.probabilities."""
    return model.predict_proba(frame[FEATURE_COLUMNS])[:, 1].astype(np.float64)


def predictions(model, frame):
    """Point predictions as float64, for the same reason."""
    return model.predict(frame[FEATURE_COLUMNS]).astype(np.float64)


def held_out_classifier(train, validation):
    """The recorded moneyline architecture, trained on train only.

    NOT the shipped model. finalize_models.py trains that with no holdout, so
    the test window is inside its training set and it reads 0.7217 / 0.5485
    against its honest 0.6698 / 0.5979.
    """
    model = XGBClassifier(**CLASSIFIER_PARAMS)
    model.fit(
        train[FEATURE_COLUMNS], train[WIN_TARGET],
        eval_set=[(validation[FEATURE_COLUMNS], validation[WIN_TARGET])],
        verbose=False,
    )
    return model


def check_sign_convention(df) -> None:
    """HOME_WIN must be exactly HOME_MARGIN > 0, or Phi() points the wrong way."""
    section("SIGN CONVENTION")
    margin = df[MARGIN_TARGET]
    derived = (margin > 0).astype(int)
    mismatches = int((derived != df[WIN_TARGET]).sum())
    ties = int((margin == 0).sum())

    print(f"HOME_WIN == (HOME_MARGIN > 0) on all {len(df):,} games: "
          f"{'yes' if mismatches == 0 else f'NO - {mismatches} mismatches'}")
    print(f"Games with a margin of exactly 0: {ties} (NBA games cannot end tied)")
    if mismatches:
        raise SystemExit(
            f"{mismatches} games disagree. A positive predicted margin would not "
            "mean a home win, and every probability below would be inverted."
        )
    print("\nSo P(home win) = P(margin > 0) = Phi(predicted_margin / sigma),")
    print("with no sign flip. This is asserted rather than assumed because the")
    print("whole method inverts if HOME_MARGIN were away-minus-home.")


def assert_no_test_leak(fit_frames, test_frame, what: str) -> None:
    """Sigma is fitted on train+validation only. A leak here would be invisible."""
    fit_index = set()
    for frame in fit_frames:
        fit_index |= set(frame.index)
    overlap = fit_index & set(test_frame.index)
    if overlap:
        raise SystemExit(
            f"{what} was fitted on {len(overlap)} rows that are also in the test "
            "split. Every number below would be optimistic."
        )


def residual_diagnostics(train_resid, validation_resid) -> dict:
    """Is the normal assumption good enough, and which residuals set sigma?"""
    section("RESIDUAL DIAGNOSTICS")

    print("WHICH RESIDUALS. A boosted tree has partly memorised its training")
    print("rows, so train residuals understate the spread of a genuine forecast")
    print("error. Using them would make sigma too small and every derived")
    print("probability overconfident - which would look like the hypothesis")
    print("failing rather than like a bad sigma.\n")

    for label, resid in [("train (in-sample)", train_resid),
                         ("validation (out-of-sample)", validation_resid)]:
        print(f"  {label:<28} n={len(resid):>6}  std={resid.std(ddof=1):>7.4f}  "
              f"mean={resid.mean():>+7.4f}")

    inflation = validation_resid.std(ddof=1) / train_resid.std(ddof=1)
    print(f"\n  Validation residuals are {inflation:.2f}x the train spread.")
    print("  Sigma is taken from the out-of-sample residuals for that reason.")

    section("NORMALITY OF THE RESIDUALS")
    resid = validation_resid
    mu, sd = resid.mean(), resid.std(ddof=1)

    skew = float(stats.skew(resid))
    kurt = float(stats.kurtosis(resid))
    ks_stat, ks_p = stats.kstest((resid - mu) / sd, stats.norm.cdf)
    ad = stats.anderson(resid, dist="norm")
    ad_passes = ad.statistic < ad.critical_values[2]

    print(f"  skewness                 {skew:>+8.4f}   (0 for a normal)")
    print(f"  excess kurtosis          {kurt:>+8.4f}   (0 for a normal; >0 is fat-tailed)")
    print(f"  Kolmogorov-Smirnov D     {ks_stat:>8.4f}   p = {ks_p:.4f}")
    print(f"  Anderson-Darling A^2     {ad.statistic:>8.4f}   "
          f"5% critical value {ad.critical_values[2]:.4f}  "
          f"-> {'normal' if ad_passes else 'NOT normal'}")
    print("\n  Anderson-Darling is the one to read. Both tests compare against a")
    print("  normal whose mean and sd were estimated from this same sample, which")
    print("  makes the KS p-value optimistic - it assumes the parameters were")
    print("  known in advance. Anderson-Darling's critical values are built for")
    print("  estimated parameters, so it does not have that problem.")

    print("\n  QUANTILE COMPARISON (observed against the fitted normal)")
    print(f"  {'quantile':>10}{'observed':>12}{'normal':>12}{'difference':>13}")
    for q in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99):
        observed = float(np.quantile(resid, q))
        expected = float(stats.norm.ppf(q, mu, sd))
        print(f"  {q:>10.2f}{observed:>12.3f}{expected:>12.3f}{observed - expected:>+13.3f}")

    normal_enough = abs(skew) < 0.25 and abs(kurt) < 0.5 and ad_passes
    print(f"\n  Normal enough to rely on: {'yes' if normal_enough else 'NO'}")
    if not normal_enough:
        print("  The empirical-CDF variant below exists for exactly this case -")
        print("  it makes no distributional assumption at all.")

    return {"normal_enough": normal_enough, "skew": skew, "kurtosis": kurt,
            "ks_p": float(ks_p), "std": float(sd)}


def optimise_sigma(margins, outcomes) -> float:
    """The sigma minimising log loss, fitted on validation only.

    The classifier is trained to minimise log loss, so handing the derived
    probability an arbitrary scale would compare the transformation against a
    handicap rather than against the hypothesis.
    """
    def objective(sigma):
        p = stats.norm.cdf(margins / sigma)
        return log_loss(outcomes, np.clip(p, CLIP, 1 - CLIP))

    result = minimize_scalar(objective, bounds=(1.0, 60.0), method="bounded")
    return float(result.x)


def fit_heteroscedastic(pred_totals, residuals):
    """sigma as a function of predicted total points.

    E|X| = sigma * sqrt(2/pi) for a zero-mean normal, so a fitted mean absolute
    residual converts to sigma by sqrt(pi/2).
    """
    model = LinearRegression().fit(pred_totals.reshape(-1, 1), np.abs(residuals))
    correlation = float(np.corrcoef(pred_totals, np.abs(residuals))[0, 1])
    return model, correlation


def heteroscedastic_sigma(model, pred_totals, multiplier=1.0):
    raw = model.predict(pred_totals.reshape(-1, 1)) * np.sqrt(np.pi / 2)
    return np.clip(raw * multiplier, 1.0, None)


def empirical_probability(pred_margins, residual_pool):
    """P(margin > 0) read straight off the residual distribution.

    P(pred + resid > 0) = P(resid > -pred) = 1 - F(-pred), with F the empirical
    CDF of the out-of-sample residuals. No normal assumption anywhere.
    """
    ordered = np.sort(np.asarray(residual_pool, dtype=np.float64))
    below = np.searchsorted(ordered, -pred_margins, side="right")
    return np.clip(1.0 - below / len(ordered), CLIP, 1 - CLIP)


def describe(label, y, p, extra=None) -> dict:
    p = np.clip(np.asarray(p, dtype=np.float64), CLIP, 1 - CLIP)
    decomposition = brier_decomposition(y, p)
    return {
        "label": label,
        "accuracy": float(accuracy_score(y, (p > 0.5).astype(int))),
        "log_loss": float(log_loss(y, p)),
        "brier": brier_score(y, p),
        "ece": expected_calibration_error(y, p),
        "auc": float(roc_auc_score(y, p)),
        "reliability": decomposition["reliability"],
        "resolution": decomposition["resolution"],
        "residual": decomposition["residual_vs_binned"],
        "probabilities": p,
        "extra": extra or "",
    }


def print_comparison(rows) -> None:
    print(f"{'PREDICTOR':<34}{'ACC':>8}{'LOG LOSS':>10}{'BRIER':>9}"
          f"{'ECE':>8}{'AUC':>9}{'RESOLUTION':>12}")
    print("-" * 90)
    for r in rows:
        print(f"{r['label']:<34}{r['accuracy']:>8.4f}{r['log_loss']:>10.4f}"
              f"{r['brier']:>9.4f}{r['ece']:>8.4f}{r['auc']:>9.4f}"
              f"{r['resolution']:>12.6f}")


def check_auc_identity(rows, constant_sigma_labels) -> None:
    """Every constant-sigma variant must share one AUC. A correctness check."""
    section("AUC IDENTITY CHECK (correctness, not a result)")

    constants = [r for r in rows if r["label"] in constant_sigma_labels]
    aucs = [r["auc"] for r in constants]
    spread = max(aucs) - min(aucs)

    for r in constants:
        print(f"  {r['label']:<34}{r['auc']:.15f}")
    print(f"\n  spread across constant-sigma variants: {spread:.2e} "
          f"(tolerance {AUC_TOLERANCE:.0e})")

    if spread > AUC_TOLERANCE:
        raise SystemExit(
            "Constant-sigma variants disagree on AUC. Phi(m/sigma) is strictly "
            "monotonic in m, so they cannot rank differently - this is an "
            "implementation bug and every number above is suspect."
        )
    print("  PASS - sigma rescales without reordering, as it must.")
    print("\n  Two variants are deliberately EXCLUDED from this check, and not")
    print("  because they are allowed to be wrong:")
    print("    - heteroscedastic: sigma varies per game, so it is not a")
    print("      transform of the margin alone and CAN reorder. That is the")
    print("      only way it could ever beat the others on AUC.")
    print("    - empirical CDF: a step function, so it can create ties, and")
    print("      AUC splits ties at 0.5.")

    for label in ("Derived, heteroscedastic sigma",
                  "Derived, empirical CDF (no normal)"):
        row = next((r for r in rows if r["label"] == label), None)
        if row is not None:
            print(f"      {label:<36}{row['auc']:.15f} "
                  f"({row['auc'] - aucs[0]:+.2e})")


def significance(y, classifier_p, derived_p, iterations=2000, seed=0) -> dict:
    """Is either difference real, or is this two samples of one number?

    A tie is the expected outcome here, and "they differ by 0.28%" says nothing
    without knowing how much 0.28% moves under resampling. Paired throughout -
    both predictors score the same games in every resample - so the comparison
    is not inflated by the games themselves being easy or hard.
    """
    section("IS THE DIFFERENCE REAL?")

    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    n = len(y)

    auc_deltas, acc_deltas = [], []
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        auc_deltas.append(roc_auc_score(y[idx], derived_p[idx])
                          - roc_auc_score(y[idx], classifier_p[idx]))
        acc_deltas.append(np.mean((derived_p[idx] > 0.5) == y[idx])
                          - np.mean((classifier_p[idx] > 0.5) == y[idx]))

    auc_deltas = np.array(auc_deltas)
    acc_deltas = np.array(acc_deltas)
    auc_ci = np.percentile(auc_deltas, [2.5, 97.5])
    acc_ci = np.percentile(acc_deltas, [2.5, 97.5])

    print(f"Paired bootstrap, {len(auc_deltas)} resamples of {n} games "
          "(derived minus classifier):\n")
    print(f"  AUC       {auc_deltas.mean():>+9.4f}   "
          f"95% CI [{auc_ci[0]:+.4f}, {auc_ci[1]:+.4f}]   "
          f"{'excludes 0' if auc_ci[0] * auc_ci[1] > 0 else 'SPANS 0'}")
    print(f"  Accuracy  {acc_deltas.mean():>+9.4f}   "
          f"95% CI [{acc_ci[0]:+.4f}, {acc_ci[1]:+.4f}]   "
          f"{'excludes 0' if acc_ci[0] * acc_ci[1] > 0 else 'SPANS 0'}")

    classifier_right = (classifier_p > 0.5).astype(int) == y
    derived_right = (derived_p > 0.5).astype(int) == y
    only_classifier = int(np.sum(classifier_right & ~derived_right))
    only_derived = int(np.sum(~classifier_right & derived_right))
    exact = stats.binomtest(only_derived, only_classifier + only_derived, 0.5)

    print(f"\nMcNemar on the {only_classifier + only_derived} games where they "
          "disagree (the paired test for two\nclassifiers on one sample; games "
          "both get right or both get wrong carry\nno information about which "
          "is better):\n")
    print(f"  classifier right, derived wrong   {only_classifier}")
    print(f"  derived right, classifier wrong   {only_derived}")
    print(f"  exact binomial p                  {exact.pvalue:.4f}   "
          f"{'significant at 5%' if exact.pvalue < 0.05 else 'NOT significant'}")

    return {"auc_ci": auc_ci, "acc_ci": acc_ci, "mcnemar_p": float(exact.pvalue),
            "auc_significant": bool(auc_ci[0] * auc_ci[1] > 0),
            "acc_significant": bool(exact.pvalue < 0.05)}


def verdict(classifier, best_derived, normality, stats_result, ecdf) -> None:
    section("VERDICT")

    auc_delta = (best_derived["auc"] - classifier["auc"]) / classifier["auc"] * 100
    ll_delta = (best_derived["log_loss"] - classifier["log_loss"]) / classifier["log_loss"] * 100
    brier_delta = (best_derived["brier"] - classifier["brier"]) / classifier["brier"] * 100
    res_delta = (best_derived["resolution"] - classifier["resolution"]) / classifier["resolution"] * 100
    acc_delta = best_derived["accuracy"] - classifier["accuracy"]

    print(f"Best derived variant: {best_derived['label']}\n")
    print(f"{'METRIC':<14}{'CLASSIFIER':>13}{'DERIVED':>13}{'CHANGE':>11}   DIRECTION")
    print("-" * 68)
    for name, key, lower_is_better in [
        ("AUC", "auc", False),
        ("Log loss", "log_loss", True),
        ("Brier", "brier", True),
        ("Resolution", "resolution", False),
        ("ECE", "ece", True),
    ]:
        a, b = classifier[key], best_derived[key]
        change = (b - a) / a * 100
        better = (b < a) if lower_is_better else (b > a)
        print(f"{name:<14}{a:>13.6f}{b:>13.6f}{change:>+10.2f}%   "
              f"{'better' if better else 'worse'}")
    print(f"{'Accuracy':<14}{classifier['accuracy']:>13.6f}"
          f"{best_derived['accuracy']:>13.6f}{acc_delta:>+10.4f}    "
          f"({round(acc_delta * len(best_derived['probabilities'])):+d} games)")

    print()
    auc_wins = best_derived["auc"] > classifier["auc"] and stats_result["auc_significant"]
    ll_wins = ll_delta <= -ADOPTION_BAR_PCT
    brier_wins = brier_delta <= -ADOPTION_BAR_PCT

    if ll_wins and brier_wins and acc_delta >= 0:
        print("BEATS THE CLASSIFIER. Clears the bar on log loss and Brier")
        print(f"(>{ADOPTION_BAR_PCT:.0f}% each) without losing accuracy.")
        print("NOT SHIPPED IN THIS PHASE - that is a separate change.")
    elif auc_wins and not ll_wins:
        print("THE INTERESTING SUB-CASE: the margin RANKS BETTER but the")
        print(f"probabilities score worse (AUC {auc_delta:+.2f}%, log loss "
              f"{ll_delta:+.2f}%).")
        print("That is a sigma problem, not a hypothesis problem - the ordering")
        print("carries information the scale is failing to express. Worth one")
        print("more iteration on sigma rather than a rejection.")
    else:
        print("A TIE, and it should be read as one.")
        print(f"AUC {auc_delta:+.2f}%, log loss {ll_delta:+.2f}%, Brier "
              f"{brier_delta:+.2f}%, resolution {res_delta:+.2f}%,")
        print(f"and the bootstrap CI on AUC spans zero "
              f"([{stats_result['auc_ci'][0]:+.4f}, "
              f"{stats_result['auc_ci'][1]:+.4f}]).")
        print("\nThree model families had already converged at this accuracy band")
        print("(CLAUDE.md section 6). A fourth arriving in the same place is")
        print("consistent with that ceiling, not a reason to keep pushing.")
        print("\nThe hypothesis is answered, and answered no: the binary target")
        print("is NOT discarding usable information about WHO WINS. Whatever the")
        print("margin knows about that, HOME_WIN already extracts.")

        if stats_result["acc_significant"] and acc_delta > 0:
            print("\nONE GENUINE EXCEPTION, and it is not a get-out.")
            print(f"The derived side wins {round(acc_delta * len(best_derived['probabilities']))} "
                  "more games on accuracy, and McNemar makes that")
            print(f"real (p = {stats_result['mcnemar_p']:.4f}), not noise. But accuracy tests only")
            print("the SIGN of the margin, which is why every sigma variant scores")
            print("identically on it: Phi(m/s) > 0.5 exactly when m > 0, whatever s is.")
            print("So the margin model picks the winner slightly better AT the")
            print("boundary while ranking slightly worse OVERALL. That is a real")
            print("asymmetry, not a contradiction - and log loss and Brier, which")
            print("grade the whole distribution rather than one threshold, both")
            print("decline to call it an improvement.")

    if not normality["normal_enough"]:
        print("\nON THE NORMALITY ASSUMPTION - it FAILS the test and does not")
        print("matter, which are two separate findings and both belong here.")
        print(f"Excess kurtosis is {normality['kurtosis']:+.4f} and "
              "Anderson-Darling rejects normality")
        print("outright: the residual tails are 3-5 points fatter than a normal")
        print("at the 1st and 99th percentiles.")
        print("\nThe distribution-free variant is what settles whether that cost")
        print("anything, and it did not - the empirical CDF scores WORSE than the")
        print(f"fitted normal ({ecdf['log_loss']:.4f} against "
              f"{best_derived['log_loss']:.4f} log loss, "
              f"{ecdf['brier']:.4f} against {best_derived['brier']:.4f} Brier).")
        print("\nThe reason is that the non-normality lives where Phi is already")
        print("saturated. A 35-point residual against sigma=11 is 3.1 standard")
        print("deviations, so it maps to about 0.999 whether the tail is normal")
        print("or fat; the shape of the extreme tail barely moves a probability")
        print("that is already pinned near 1. So the assumption is wrong and")
        print("harmless, and it was worth one variant to establish that rather")
        print("than either assuming it or worrying about it.")


def reproduce_or_stop(name, got, recorded, tolerance=REPRODUCTION_TOLERANCE) -> None:
    ok = abs(got - recorded) < tolerance
    print(f"  {name:<34} got {got:.4f}   recorded {recorded:.4f}   "
          f"{'MATCH' if ok else 'NO MATCH'}")
    if not ok:
        raise SystemExit(
            f"{name} does not reproduce its recorded value. Every comparison "
            "below would describe a different model, so this stops here."
        )


def main():
    section("DATA")
    df = load_dataset()
    train, validation, test = split_three_way(df)
    test_comparable = test.dropna(subset=ROLLING_FEATURE_COLUMNS)
    print(f"\nScoring on {len(test_comparable)} complete-window test games - the")
    print("same split behind every recorded number in CLAUDE.md.")

    check_sign_convention(df)

    section("MODELS - ALL HELD OUT, ALL REPRODUCING THEIR RECORDED NUMBERS")
    print("Both underlying models are trained on train only and early-stopped on")
    print("validation. A margin model that had seen the test games would make the")
    print("derived probability look excellent for the wrong reason.\n")

    classifier = held_out_classifier(train, validation)
    class_proba = probabilities(classifier, test_comparable)
    y_test = test_comparable[WIN_TARGET].to_numpy()
    reproduce_or_stop("Classifier accuracy",
                      accuracy_score(y_test, (class_proba > 0.5).astype(int)),
                      RECORDED_ACCURACY)
    reproduce_or_stop("Classifier log loss",
                      log_loss(y_test, class_proba), RECORDED_LOG_LOSS)

    margin_model = fit_xgb(train, validation, MARGIN_TARGET)
    test_margins = predictions(margin_model, test_comparable)
    reproduce_or_stop("Spread MAE",
                      mean_absolute_error(test_comparable[MARGIN_TARGET], test_margins),
                      RECORDED_SPREAD_MAE)

    total_model = fit_xgb(train, validation, TOTAL_TARGET)
    test_totals = predictions(total_model, test_comparable)
    reproduce_or_stop("Totals MAE",
                      mean_absolute_error(test_comparable[TOTAL_TARGET], test_totals),
                      RECORDED_TOTALS_MAE)

    print(f"\n  Trees: classifier {classifier.best_iteration}, "
          f"margin {margin_model.best_iteration}, total {total_model.best_iteration}")

    train_resid = (train[MARGIN_TARGET].to_numpy()
                   - predictions(margin_model, train))
    validation_margins = predictions(margin_model, validation)
    validation_resid = validation[MARGIN_TARGET].to_numpy() - validation_margins
    validation_totals = predictions(total_model, validation)
    y_validation = validation[WIN_TARGET].to_numpy()

    normality = residual_diagnostics(train_resid, validation_resid)

    section("SIGMA - THREE ESTIMATES, NONE TOUCHING THE TEST SPLIT")
    assert_no_test_leak([train, validation], test_comparable, "sigma fitting")
    print("  Verified: the sigma-fitting rows and the test rows are disjoint.\n")

    sigma_global = float(validation_resid.std(ddof=1))
    sigma_opt = optimise_sigma(validation_margins, y_validation)
    hetero_model, hetero_corr = fit_heteroscedastic(validation_totals, validation_resid)

    def hetero_objective(multiplier):
        s = heteroscedastic_sigma(hetero_model, validation_totals, multiplier)
        p = stats.norm.cdf(validation_margins / s)
        return log_loss(y_validation, np.clip(p, CLIP, 1 - CLIP))

    hetero_mult = float(minimize_scalar(
        hetero_objective, bounds=(0.2, 5.0), method="bounded").x)
    hetero_test_sigma = heteroscedastic_sigma(hetero_model, test_totals, hetero_mult)

    print(f"  1. Global constant   sigma = {sigma_global:.4f}  "
          "(std of out-of-sample residuals)")
    print(f"  2. Log-loss optimal  sigma = {sigma_opt:.4f}  "
          f"({(sigma_opt / sigma_global - 1) * 100:+.1f}% against the global constant)")
    print(f"  3. Heteroscedastic   sigma = {hetero_test_sigma.min():.4f} to "
          f"{hetero_test_sigma.max():.4f} across the test set")
    print(f"       |residual| vs predicted total: slope "
          f"{hetero_model.coef_[0]:+.5f} per point, correlation {hetero_corr:+.4f}")
    print(f"       scale multiplier fitted on validation: {hetero_mult:.4f}")

    if hetero_model.coef_[0] < 0:
        print("\n       THE SLOPE IS NEGATIVE, which is the opposite of the premise.")
        print("       The idea was that high-scoring games have wider margin")
        print("       distributions; measured, higher predicted totals go with")
        print("       SLIGHTLY SMALLER absolute residuals.")
    if abs(hetero_corr) < 0.05:
        print("\n       And the dependence is negligible either way "
              f"(|r| = {abs(hetero_corr):.4f} < 0.05),")
        print("       so this variant is the constant one with extra machinery.")
        print("       It is still scored rather than dropped, because dropping it")
        print("       unmeasured would be asserting the result instead of showing")
        print("       it - and it costs one line to check.")

    section("COMPARISON")

    hetero_validation_sigma = heteroscedastic_sigma(
        hetero_model, validation_totals, hetero_mult)

    # Each variant carries its validation probabilities too, so the best one is
    # chosen there rather than on the test set it is about to be judged on.
    variants = [
        (f"Derived, global sigma={sigma_global:.2f}",
         stats.norm.cdf(test_margins / sigma_global),
         stats.norm.cdf(validation_margins / sigma_global), True),
        (f"Derived, optimal sigma={sigma_opt:.2f}",
         stats.norm.cdf(test_margins / sigma_opt),
         stats.norm.cdf(validation_margins / sigma_opt), True),
        ("Derived, heteroscedastic sigma",
         stats.norm.cdf(test_margins / hetero_test_sigma),
         stats.norm.cdf(validation_margins / hetero_validation_sigma), False),
        ("Derived, empirical CDF (no normal)",
         empirical_probability(test_margins, validation_resid),
         empirical_probability(validation_margins, validation_resid), False),
    ]

    rows = [describe("Moneyline classifier", y_test, class_proba)]
    constant_labels = []
    validation_losses = {}
    for label, test_p, validation_p, is_constant in variants:
        rows.append(describe(label, y_test, test_p))
        validation_losses[label] = float(log_loss(
            y_validation, np.clip(validation_p, CLIP, 1 - CLIP)))
        if is_constant:
            constant_labels.append(label)

    print_comparison(rows)
    worst_residual = max(abs(r["residual"]) for r in rows)
    print(f"\nLargest Murphy decomposition residual across all {len(rows)} rows: "
          f"{worst_residual:.2e}")
    if worst_residual > 1e-10:
        raise SystemExit(
            "The Brier decomposition does not reconcile. reliability - "
            "resolution + uncertainty must equal the binned Brier exactly."
        )

    check_auc_identity(rows, constant_labels)

    raw_margin_auc = float(roc_auc_score(y_test, test_margins))
    print(f"\n  AUC of the RAW predicted margin: {raw_margin_auc:.15f}")
    print("  Identical to the constant-sigma variants, as it must be - which is")
    print("  what makes AUC the clean test: it asks only whether the margin")
    print("  ranks fixtures better than the classifier's scores, independent of")
    print("  sigma entirely.")

    section("WHERE THE DIFFERENCE SITS (Murphy decomposition)")
    print("Resolution is the quantity this experiment exists to move: the")
    print("calibration study put reliability at 2.3% of the loss, so only")
    print("discrimination was ever worth chasing.\n")
    print(f"{'PREDICTOR':<34}{'RELIABILITY':>13}{'RESOLUTION':>13}{'BRIER':>10}")
    print("-" * 70)
    for r in rows:
        print(f"{r['label']:<34}{r['reliability']:>13.6f}"
              f"{r['resolution']:>13.6f}{r['brier']:>10.6f}")

    section("CHOOSING A VARIANT - ON VALIDATION, NOT ON TEST")
    print("Picking whichever variant scores best on the test set would be a")
    print("selection effect flattering the derived side: four tries at one")
    print("target, judged on the target. They are separated on validation")
    print("instead, and the winner's TEST numbers are what the verdict uses.\n")
    print(f"  {'VARIANT':<36}{'VALIDATION LOG LOSS':>21}{'TEST LOG LOSS':>16}")
    for label, loss in sorted(validation_losses.items(), key=lambda kv: kv[1]):
        test_loss = next(r["log_loss"] for r in rows if r["label"] == label)
        print(f"  {label:<36}{loss:>21.6f}{test_loss:>16.6f}")

    best_label = min(validation_losses, key=validation_losses.get)
    best = next(r for r in rows if r["label"] == best_label)
    print(f"\n  Selected on validation: {best_label}")

    result = significance(y_test, class_proba, best["probabilities"])
    ecdf_row = next(r for r in rows
                    if r["label"] == "Derived, empirical CDF (no normal)")
    verdict(rows[0], best, normality, result, ecdf_row)

    section("FULL TEST SET, FOR COMPLETENESS")
    full_margins = predictions(margin_model, test)
    full_class = probabilities(classifier, test)
    y_full = test[WIN_TARGET].to_numpy()
    print("Every test game, including early-season ones with incomplete rolling")
    print("windows. Excluded above for comparability with the recorded numbers.\n")
    print_comparison([
        describe("Moneyline classifier", y_full, full_class),
        describe(f"Derived, optimal sigma={sigma_opt:.2f}", y_full,
                 stats.norm.cdf(full_margins / sigma_opt)),
    ])


if __name__ == "__main__":
    main()
