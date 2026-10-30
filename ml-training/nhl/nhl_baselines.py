r"""Section 5's baselines: what the NHL's models have to beat.

THREE MARKETS, chosen because they are the three the other leagues already
serve and because the corpus supports them without inventing anything:

  WINNER        home win / away win. There are no ties in this era, so it is
                a clean binary; a shootout win is a win, which is how the
                source records it and how a market settles it.
  MARGIN        home goals minus away goals.
  TOTAL GOALS   home goals plus away goals.

FOUR BASELINES, and the fourth is the one that decides whether a league is
worth modelling at all:

  NO SKILL      the best CONSTANT predictor, from the TRAINING rows: the
                base rate for a log loss, the median for an MAE. This is the
                retrain gate's definition, and it got there the hard way -
                the gate's first baseline was a rolling average that
                returned NaN on a window straddling a season opener, which
                made the guard silently inert on six markets.
  ALWAYS HOME   predict the home side every time. A separate line from NO
                SKILL because for the winner they are different objects: one
                is a probability of 0.5412, the other is a hard 1.
  ELO ALONE     the fitted Elo, with nothing else. The winner comes straight
                from the expectation; the margin from a one-variable fit of
                margin on ELO_DIFF. Elo says nothing about a total, and that
                is reported rather than faked.
  LINEAR        LogisticRegression for the winner, LinearRegression for the
                two continuous markets, on the full feature set.

THE ELO-ALONE LINE IS THE ONE THAT MATTERS. The NFL's margin and winner
models TIE Elo alone, and both tie verdicts travel on the wire - so for a
new league the question is settled before any gradient-boosted model is
fitted: if the linear baseline cannot beat Elo, the league's answer is Elo.

THE FIT IS ON TRAINING SEASONS ONLY and the report covers validation and
test separately. One fit, two reports: section 4's comparison needs the
validation numbers and the frozen record needs the test ones, and refitting
between them would make the two incomparable.

INCOMPLETE ROWS ARE DROPPED HERE, which is option (a)/(c)'s regime without
the decision. Section 4's comparison is what decides between them, and it
runs after this.

    python ml-training/nhl/nhl_baselines.py
    python ml-training/nhl/nhl_baselines.py --elo-json <tuning output>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import nhl_features as F                                        # noqa: E402
from nhl_features import refuse_undeclared_seasons    # noqa: E402

EPSILON = 1e-15

MARKETS = (
    ("winner", "HOME_WIN", "log loss"),
    ("margin", "HOME_MARGIN", "MAE"),
    ("total_goals", "TOTAL_GOALS", "MAE"),
)


def log_loss(actual, predicted):
    predicted = np.clip(np.asarray(predicted, dtype=float), EPSILON,
                        1.0 - EPSILON)
    actual = np.asarray(actual, dtype=float)
    return float(-np.mean(actual * np.log(predicted)
                          + (1.0 - actual) * np.log(1.0 - predicted)))


def mae(actual, predicted):
    return float(np.mean(np.abs(np.asarray(actual, dtype=float)
                                - np.asarray(predicted, dtype=float))))


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual, dtype=float)
                                  - np.asarray(predicted,
                                               dtype=float)) ** 2)))


def scaled(estimator):
    """A linear estimator behind a scaler fitted on the training rows only.

    EVERY OTHER LEAGUE DOES THIS AND THE FIRST VERSION OF THIS FILE DID NOT,
    which is worth recording because the omission is invisible in the output:
    `train_baseline.py` standardises, and the WNBA, G League and NFL all ship
    `make_pipeline(StandardScaler(), ...)`. Without it the L2 penalty is
    applied to coefficients on wildly different scales - an Elo near 1500
    against a win share in [0, 1] - so the penalty effectively silences the
    small-scale features and the comparison with Elo alone is not a fair one.
    Measured: unscaled, the logistic baseline's test log loss was 0.683717.
    """
    return make_pipeline(StandardScaler(), estimator)


def accuracy(actual, predicted):
    return float(np.mean((np.asarray(predicted) > 0.5)
                         == (np.asarray(actual) > 0.5)))


def brier(actual, predicted):
    return float(np.mean((np.asarray(predicted, dtype=float)
                          - np.asarray(actual, dtype=float)) ** 2))


def split(frame):
    """Training / validation / test, from the one definition in nhl_features.

    DEFINED ONCE. The NBA's `common.TEST_SEASONS` is the reason: a split
    written out twice drifts, and the copy nobody re-reads is the one that
    leaks.
    """
    test = frame["SEASON"].isin(set(F.TEST_SEASONS))
    validation = frame["SEASON"].isin(set(F.VALIDATION_SEASONS))
    training = ~test & ~validation

    if not (training.sum() and validation.sum() and test.sum()):
        raise ValueError(
            f"the split is degenerate: {int(training.sum())} training, "
            f"{int(validation.sum())} validation, {int(test.sum())} test "
            f"row(s); check the season labels' dashes")
    return training, validation, test


def elo_parameters(path):
    """The fitted triple, from the tuning run's own output.

    READ, NOT RETYPED. nhl_features.ELO_DEFAULTS are defaults for a smoke
    test; the fitted values belong to the run that fitted them, and copying
    them into a second file is how a recorded number stops matching the thing
    it describes.
    """
    if path is None:
        print("  no --elo-json given; using nhl_features.ELO_DEFAULTS, "
              "which are NOT the fitted values")
        return dict(F.ELO_DEFAULTS), None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ({"k": payload["k"], "home_advantage": payload["home_advantage"],
             "carryover": payload["carryover"]}, payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=None)
    parser.add_argument("--elo-json", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    params, tuning = elo_parameters(args.elo_json)
    print(f"Elo parameters: k={params['k']:g} "
          f"home_advantage={params['home_advantage']:g} "
          f"carryover={params['carryover']:g}"
          + ("  (fitted on the training seasons)" if tuning else ""))

    games = F.load_games(args.table)
    frame = F.build_features(games, elo=params)
    # BEFORE THE FILTER. See refuse_undeclared_seasons' docstring.
    refuse_undeclared_seasons(frame)
    complete = F.is_complete(frame)
    print(f"fixtures {len(frame)}, complete {int(complete.sum())} "
          f"({100.0 * complete.mean():.2f}%), incomplete "
          f"{int((~complete).sum())} - DROPPED for every figure below")
    frame = frame[complete].reset_index(drop=True)

    training, validation, test = split(frame)
    print(f"training {int(training.sum())} | validation "
          f"{int(validation.sum())} | test {int(test.sum())}")
    print(f"test seasons: {', '.join(F.TEST_SEASONS)}")
    print()

    features = F.feature_columns()
    x_train = frame.loc[training, features].to_numpy(dtype=float)
    x_valid = frame.loc[validation, features].to_numpy(dtype=float)
    x_test = frame.loc[test, features].to_numpy(dtype=float)

    report = {"elo": params, "rows": {
        "training": int(training.sum()),
        "validation": int(validation.sum()),
        "test": int(test.sum())}, "markets": {}}

    for name, target, metric in MARKETS:
        y_train = frame.loc[training, target].to_numpy(dtype=float)
        y_valid = frame.loc[validation, target].to_numpy(dtype=float)
        y_test = frame.loc[test, target].to_numpy(dtype=float)

        # metric name -> {label -> (validation, test)}
        rows = {}

        def record(label, metric_name, value_valid, value_test):
            rows.setdefault(metric_name, {})[label] = (value_valid,
                                                       value_test)

        if metric == "log loss":
            # ---- no skill: the best CONSTANT from the training rows ----
            base_rate = float(y_train.mean())
            for metric_name, fn in (("log loss", log_loss),
                                    ("Brier", brier)):
                record(f"no skill, base rate {base_rate:.4f}", metric_name,
                       fn(y_valid, np.full(len(y_valid), base_rate)),
                       fn(y_test, np.full(len(y_test), base_rate)))
            # ALWAYS HOME IS A DIFFERENT OBJECT from the base rate: a hard 1
            # rather than a probability. Its log loss is infinite by
            # construction, so it appears only under accuracy - which is the
            # only metric it has an answer for.
            record("always home", "accuracy",
                   accuracy(y_valid, np.ones(len(y_valid))),
                   accuracy(y_test, np.ones(len(y_test))))
            record(f"no skill, base rate {base_rate:.4f}", "accuracy",
                   accuracy(y_valid, np.full(len(y_valid), base_rate)),
                   accuracy(y_test, np.full(len(y_test), base_rate)))

            # ---- Elo alone --------------------------------------------
            elo_valid = F.elo_win_probability(
                frame.loc[validation, "HOME_ELO"],
                frame.loc[validation, "AWAY_ELO"],
                params["home_advantage"])
            elo_test = F.elo_win_probability(
                frame.loc[test, "HOME_ELO"], frame.loc[test, "AWAY_ELO"],
                params["home_advantage"])
            for metric_name, fn in (("log loss", log_loss),
                                    ("Brier", brier),
                                    ("accuracy", accuracy)):
                record("Elo alone", metric_name,
                       fn(y_valid, elo_valid), fn(y_test, elo_test))

            # ---- the linear baseline ----------------------------------
            model = scaled(LogisticRegression(max_iter=5000))
            model.fit(x_train, y_train)
            pv = model.predict_proba(x_valid)[:, 1]
            pt = model.predict_proba(x_test)[:, 1]
            for metric_name, fn in (("log loss", log_loss),
                                    ("Brier", brier),
                                    ("accuracy", accuracy)):
                record("LogisticRegression", metric_name,
                       fn(y_valid, pv), fn(y_test, pt))
        else:
            # TWO CONSTANTS, BECAUSE THERE ARE TWO METRICS WITH DIFFERENT
            # OPTIMA. The median minimises MAE and the mean minimises
            # squared error, so a single "naive" line would hand one of the
            # columns an unfairly weak opponent. It matters here
            # specifically: LinearRegression minimises squared error, so
            # judging it on MAE alone, against the MAE-optimal constant,
            # grades it on a metric it was not fitted for.
            median = float(np.median(y_train))
            mean = float(y_train.mean())
            for label, constant in ((f"no skill, median {median:g}", median),
                                    (f"no skill, mean {mean:.4f}", mean)):
                for metric_name, fn in (("MAE", mae), ("RMSE", rmse)):
                    record(label, metric_name,
                           fn(y_valid, np.full(len(y_valid), constant)),
                           fn(y_test, np.full(len(y_test), constant)))

            if name == "margin":
                # ELO-MOV: one variable, fitted on training rows only. The
                # NFL's margin model ties this, which is why it is a line
                # here rather than a footnote.
                mov = scaled(LinearRegression())
                mov.fit(frame.loc[training, ["ELO_DIFF"]]
                        .to_numpy(dtype=float), y_train)
                pv = mov.predict(frame.loc[validation, ["ELO_DIFF"]]
                                 .to_numpy(dtype=float))
                pt = mov.predict(frame.loc[test, ["ELO_DIFF"]]
                                 .to_numpy(dtype=float))
                for metric_name, fn in (("MAE", mae), ("RMSE", rmse)):
                    record("Elo alone (Elo-MOV)", metric_name,
                           fn(y_valid, pv), fn(y_test, pt))
            # ELO HAS NOTHING TO SAY ABOUT A TOTAL, and that is reported
            # rather than approximated with a rating sum: a sum of two
            # ratings measures combined strength, which predicts who wins,
            # not how many goals are scored. So the total has no Elo line at
            # all, which is a statement and not an omission.

            model = scaled(LinearRegression())
            model.fit(x_train, y_train)
            pv = model.predict(x_valid)
            pt = model.predict(x_test)
            for metric_name, fn in (("MAE", mae), ("RMSE", rmse)):
                record("LinearRegression", metric_name,
                       fn(y_valid, pv), fn(y_test, pt))

        # ---- print ----------------------------------------------------
        print(f"{name.upper()}  (target {target}; headline metric {metric})")
        for metric_name in rows:
            block = rows[metric_name]
            width = max(len(label) for label in block)
            print(f"  {metric_name}")
            print(f"    {'baseline':<{width}}{'validation':>14}{'test':>14}")
            print(f"    {'-' * width}{'-' * 14}{'-' * 14}")
            for label, (value_valid, value_test) in block.items():
                print(f"    {label:<{width}}{value_valid:>14.6f}"
                      f"{value_test:>14.6f}")
        if name == "total_goals":
            print("  Elo alone: no line - Elo does not predict a total")
        print()
        report["markets"][name] = {
            metric_name: {label: {"validation": v, "test": t}
                          for label, (v, t) in block.items()}
            for metric_name, block in rows.items()}

    # ---- the verdict that decides whether to model at all -------------
    print("DOES THE LINEAR BASELINE BEAT ELO ALONE?")
    print("  The NFL's margin and winner models TIE Elo alone and say so on")
    print("  the wire. For a new league this is the question that comes")
    print("  before any model, so it is asked here rather than later.")
    # THE TIE BAND IS RELATIVE, not an absolute 0.0005. A margin MAE near
    # 2.15 and a log loss near 0.68 do not share a scale, so one absolute
    # threshold would call a 0.02% move on the margin a tie and the same
    # relative move on the log loss a difference. 0.5% of the Elo figure,
    # which is the retrain gate's 2% tolerance tightened for a comparison
    # between two baselines rather than between two model generations.
    TIE_BAND = 0.005
    comparisons = (
        ("winner", "log loss", "Elo alone", "LogisticRegression"),
        ("margin", "MAE", "Elo alone (Elo-MOV)", "LinearRegression"),
        ("margin", "RMSE", "Elo alone (Elo-MOV)", "LinearRegression"),
    )
    for market, metric_name, elo_key, linear_key in comparisons:
        block = report["markets"][market][metric_name]
        for where in ("validation", "test"):
            elo_value = block[elo_key][where]
            linear_value = block[linear_key][where]
            delta = linear_value - elo_value
            relative = delta / elo_value if elo_value else 0.0
            verdict = ("linear is BETTER" if relative < -TIE_BAND else
                       "linear is WORSE" if relative > TIE_BAND else "a TIE")
            print(f"    {market + ' (' + metric_name + ')':<22}"
                  f"{where:<12}Elo {elo_value:.6f}  "
                  f"linear {linear_value:.6f}  {delta:+.6f} "
                  f"({relative * 100:+.2f}%)  {verdict}")

    print()
    print("AND DOES EITHER BEAT THE BEST CONSTANT?")
    print("  The question above is only interesting if there is skill to")
    print("  compare. A market where nothing beats a constant has no skill")
    print("  to allocate, and saying so is the finding.")
    for market, metric_name in (("winner", "log loss"), ("margin", "MAE"),
                                ("margin", "RMSE"), ("total_goals", "MAE"),
                                ("total_goals", "RMSE")):
        block = report["markets"][market][metric_name]
        constants = {label: v for label, v in block.items()
                     if label.startswith("no skill")}
        others = {label: v for label, v in block.items()
                  if not label.startswith("no skill")
                  and label != "always home"}
        if not constants or not others:
            continue
        floor_label = min(constants, key=lambda L: constants[L]["test"])
        floor = constants[floor_label]["test"]
        best_label = min(others, key=lambda L: others[L]["test"])
        best = others[best_label]["test"]
        relative = (best - floor) / floor if floor else 0.0
        verdict = ("BEATS the constant" if relative < -TIE_BAND else
                   "LOSES to the constant" if relative > TIE_BAND else
                   "TIES the constant")
        print(f"    {market + ' (' + metric_name + ')':<22}"
              f"best {best_label} {best:.6f} vs {floor_label} {floor:.6f}  "
              f"({relative * 100:+.2f}%)  {verdict}")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print()
        print(f"written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
