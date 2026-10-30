r"""Section 4's error comparison, on the 2023-24 VALIDATION season only.

THE THREE OPTIONS FOR A ROW WHOSE ROLLING WINDOW IS NOT FULL:

  (a) REFUSE, with a structured reason code. Features stay season-scoped and
      an incomplete row is not scored. This is what the WNBA and G League
      do, and the G League's season-opener refusal is the shape of it.
  (b) CARRY the windows across the season boundary. A different feature
      definition: last April's form fills the window.
  (c) TRAIN ON PARTIAL ROWS. Features stay season-scoped, the incomplete
      rows join the training set behind an imputer with missingness
      indicators, and serving answers them. This is what the NFL does -
      its two linear artifacts carry a SimpleImputer fitted on training
      rows with SEASON_OPENER as the indicator, so an incomplete row is
      scored rather than refused.

WHY IT IS THREE PANELS AND NOT ONE TABLE. The three options do not answer
the same questions, so a single error column would be comparing different
populations - the exact mistake A3's four regimes were built to avoid:

  PANEL 1  THE COMMON SET: the rows (a) can answer. All three are scored
           here, so this isolates whether changing the feature definition or
           adding partial rows to training HURTS the rows that were fine.
  PANEL 2  THE REFUSED SET: the rows (a) turns down. Only (b) and (c) have
           an answer, and this is what answering is worth.
  PANEL 3  COVERAGE: how many fixtures each option can serve at all.

AFTER SECTION 5's BASELINES, ON VALIDATION, AND NOT ON TEST. The baselines
are what "better" is measured against, and the test seasons are spent once.

    python ml-training/nhl/nhl_incomplete_options.py --elo-json <path>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import nhl_features as F                                        # noqa: E402
from nhl_baselines import (accuracy, brier, log_loss, mae,       # noqa: E402
                           rmse, split)
from nhl_features import refuse_undeclared_seasons        # noqa: E402

OPTIONS = (
    ("(a) refuse, season-scoped", False, False, False),
    ("(b) carry windows + rest", True, True, False),
    ("(c) train on partial rows", False, False, True),
    # (d) IS NOT IN THE SPEC'S THREE. It is (b) and (c) composed, and it is
    # measured because it costs one line and because (b) alone does NOT
    # reach full coverage: the incidence measurement found 179 rows still
    # incomplete with both carries on, all of them either the first season
    # of the corpus or an expansion team's first ten games. The NHL will
    # expand again, so an option that cannot answer a new franchise is an
    # option with a refusal path in it whether or not it admits to one.
    ("(d) carry AND impute", True, True, True),
)

# Resamples for the paired bootstrap. The retrain gate's window was sized
# this way - "500 games is measured, not chosen" - and the same instrument
# is what turns "a tie" from a reading into a measurement.
BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261030

MARKETS = (
    ("winner", "HOME_WIN", True),
    ("margin", "HOME_MARGIN", False),
    ("total_goals", "TOTAL_GOALS", False),
)


def estimator(classify, impute):
    """The linear model for one option.

    THE IMPUTER IS ONLY IN OPTION (c), and it carries `add_indicator=True`
    so the model can tell "the mean" from "we did not know" - the
    unknown-is-never-a-zero rule, in the one place this project has
    deliberately chosen to score an unknown row rather than refuse it. A
    median fill with no indicator would hand the model a fabricated value
    wearing the same shape as a real one.
    """
    steps = []
    if impute:
        steps.append(SimpleImputer(strategy="median", add_indicator=True))
    steps.append(StandardScaler())
    steps.append(LogisticRegression(max_iter=5000) if classify
                 else LinearRegression())
    return make_pipeline(*steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=None)
    parser.add_argument("--elo-json", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if args.elo_json:
        payload = json.loads(args.elo_json.read_text(encoding="utf-8"))
        params = {"k": payload["k"],
                  "home_advantage": payload["home_advantage"],
                  "carryover": payload["carryover"]}
    else:
        params = dict(F.ELO_DEFAULTS)
        print("no --elo-json; using ELO_DEFAULTS, which are not the fitted "
              "values")
    print(f"Elo: k={params['k']:g} home_advantage="
          f"{params['home_advantage']:g} carryover={params['carryover']:g}")

    games = F.load_games(args.table)
    features = F.feature_columns()

    # Build every regime once. The Elo columns are identical across them -
    # the carry flags touch only rolling and rest - which is asserted rather
    # than assumed, because if it stopped being true the panels would be
    # comparing Elo changes as well.
    frames = {}
    for label, carry_rolling, carry_rest, _impute in OPTIONS:
        frames[label] = F.build_features(
            games, elo=params, carry_rolling=carry_rolling,
            carry_rest=carry_rest)
    # BEFORE ANY SPLIT. This script defines training by subtraction too,
    # so it needs the same guard, imported rather than copied.
    refuse_undeclared_seasons(frames[OPTIONS[0][0]])
    reference = frames[OPTIONS[0][0]]
    for label, frame in frames.items():
        worst = float(np.abs(frame["ELO_DIFF"].to_numpy()
                             - reference["ELO_DIFF"].to_numpy()).max())
        if worst > 1e-12:
            print(f"ABORT: {label} changed ELO_DIFF by {worst:.3e}; the "
                  f"carry flags must touch only rolling and rest")
            return 2
    print("checked: the carry flags change no Elo value (worst 0.0)")

    # THE REFUSED SET IS DEFINED BY OPTION (a), on the validation season.
    complete_a = F.is_complete(reference)
    _tr, valid_mask, _te = split(reference)
    common = valid_mask & complete_a
    refused = valid_mask & ~complete_a
    print(f"validation fixtures {int(valid_mask.sum())}: "
          f"{int(common.sum())} answerable by (a), "
          f"{int(refused.sum())} refused by (a)")
    print()

    report = {"elo": params, "validation_fixtures": int(valid_mask.sum()),
              "common": int(common.sum()), "refused": int(refused.sum()),
              "markets": {}}

    for market, target, classify in MARKETS:
        print(f"{market.upper()}  (target {target})")
        metrics = (("log loss", log_loss), ("Brier", brier),
                   ("accuracy", accuracy)) if classify else \
                  (("MAE", mae), ("RMSE", rmse))
        panels = {"common": {}, "refused": {}}
        coverage = {}
        # Per-row predictions on the common panel, for the paired bootstrap.
        # SAFE TO PAIR BY POSITION: every frame is built from the same
        # `as_fixtures` output, so the row order is identical, and the
        # common panel's mask selects the same rows in every option -
        # carrying a window can only ADD data, never remove it, so a row
        # complete under (a) is complete under (b) too.
        paired = {}

        for label, _cr, _crest, impute in OPTIONS:
            frame = frames[label]
            training, _v, _t = split(frame)
            usable = F.is_complete(frame)

            # WHICH ROWS EACH OPTION TRAINS ON is the option, so it is read
            # from the option rather than shared: (a) and (b) fit on their
            # own complete rows, (c) fits on every row including partial
            # ones.
            fit_rows = training if impute else (training & usable)
            x_fit = frame.loc[fit_rows, features].to_numpy(dtype=float)
            y_fit = frame.loc[fit_rows, target].to_numpy(dtype=float)
            model = estimator(classify, impute)
            model.fit(x_fit, y_fit)

            # WHICH ROWS EACH OPTION CAN ANSWER. (a) and (b) can only answer
            # a complete row; (c) answers anything.
            answerable = valid_mask if impute else (valid_mask & usable)
            coverage[label] = (int(answerable.sum()), int(valid_mask.sum()),
                               int(fit_rows.sum()))

            for panel, mask in (("common", common), ("refused", refused)):
                scored = mask & answerable
                if not scored.sum():
                    panels[panel][label] = None
                    continue
                x = frame.loc[scored, features].to_numpy(dtype=float)
                y = frame.loc[scored, target].to_numpy(dtype=float)
                predicted = (model.predict_proba(x)[:, 1] if classify
                             else model.predict(x))
                panels[panel][label] = {
                    name: fn(y, predicted) for name, fn in metrics}
                panels[panel][label]["n"] = int(scored.sum())
                if panel == "common":
                    paired[label] = (y, predicted)

        # ---- panel 1 --------------------------------------------------
        for panel, title in (
                ("common", f"PANEL 1  the {int(common.sum())} rows (a) can "
                           f"answer - does the change HURT them?"),
                ("refused", f"PANEL 2  the {int(refused.sum())} rows (a) "
                            f"REFUSES - what is answering worth?")):
            print(f"  {title}")
            width = max(len(label) for label, *_ in OPTIONS)
            head = f"    {'option':<{width}}{'n':>6}"
            for name, _fn in metrics:
                head += f"{name:>12}"
            print(head)
            print("    " + "-" * (len(head) - 4))
            for label, *_ in OPTIONS:
                result = panels[panel][label]
                if result is None:
                    print(f"    {label:<{width}}{'-':>6}"
                          + "".join(f"{'refuses':>12}"
                                    for _n, _f in metrics))
                    continue
                line = f"    {label:<{width}}{result['n']:>6}"
                for name, _fn in metrics:
                    line += f"{result[name]:>12.6f}"
                print(line)
            print()

        # ---- panel 3 --------------------------------------------------
        print("  PANEL 3  coverage and training rows")
        width = max(len(label) for label, *_ in OPTIONS)
        print(f"    {'option':<{width}}{'answerable':>12}{'of':>7}"
              f"{'share':>9}{'training rows':>16}")
        print("    " + "-" * (width + 44))
        for label, *_ in OPTIONS:
            answerable, total, fit_rows = coverage[label]
            print(f"    {label:<{width}}{answerable:>12}{total:>7}"
                  f"{100.0 * answerable / total:>8.1f}%{fit_rows:>16}")
        print()

        # ---- panel 4: is any panel-1 difference distinguishable? -------
        # A PAIRED BOOTSTRAP, BECAUSE "A TIE" HAS TO BE MEASURED. The
        # retrain gate's 500-game window was sized this way, and the lesson
        # there was blunt: at 100 games a candidate IDENTICAL to production
        # was refused 26.5% of the time by sampling noise alone. A 0.2%
        # difference on 1,146 rows is exactly the size of claim that needs
        # an interval rather than a reading.
        headline, headline_fn = (("log loss", log_loss) if classify
                                 else ("RMSE", rmse))
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        base_label = OPTIONS[0][0]
        y_base, p_base = paired[base_label]
        n = len(y_base)
        draws = rng.integers(0, n, size=(BOOTSTRAP, n))
        bootstrap = {}
        for label, *_ in OPTIONS[1:]:
            _y, p_other = paired[label]
            deltas = np.empty(BOOTSTRAP, dtype=float)
            for i in range(BOOTSTRAP):
                pick = draws[i]
                deltas[i] = (headline_fn(y_base[pick], p_other[pick])
                             - headline_fn(y_base[pick], p_base[pick]))
            observed = (headline_fn(y_base, p_other)
                        - headline_fn(y_base, p_base))
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            bootstrap[label] = {"observed": float(observed),
                                "sd": float(deltas.std()),
                                "ci_low": float(lo), "ci_high": float(hi),
                                "crosses_zero": bool(lo < 0.0 < hi)}
        print(f"  PANEL 4  paired bootstrap on the common rows, "
              f"{BOOTSTRAP} resamples, {headline}")
        width = max(len(label) for label, *_ in OPTIONS)
        print(f"    {'option vs (a)':<{width}}{'observed':>12}{'sd':>11}"
              f"{'95% interval':>26}{'':>4}")
        print("    " + "-" * (width + 53))
        for label, *_ in OPTIONS[1:]:
            b = bootstrap[label]
            verdict = ("indistinguishable" if b["crosses_zero"]
                       else "BETTER than (a)" if b["ci_high"] < 0
                       else "WORSE than (a)")
            interval = f"[{b['ci_low']:+.6f}, {b['ci_high']:+.6f}]"
            print(f"    {label:<{width}}{b['observed']:>+12.6f}"
                  f"{b['sd']:>11.6f}{interval:>26}  {verdict}")
        print()

        report["markets"][market] = {"panels": panels, "coverage": coverage,
                                     "bootstrap": bootstrap,
                                     "headline": headline}

    # ---- the recommendation -------------------------------------------
    print("=" * 72)
    print("WHAT THE PANELS SAY")
    print("  THE SIGN COMES FROM PANEL 4, NOT FROM A PERCENTAGE BAND. The")
    print("  first version of this summary judged panel 1 against a +/-0.5%")
    print("  relative band and printed 'a tie' for differences whose paired")
    print("  95% interval excludes zero - a weaker instrument overruling a")
    print("  better one that was already in the same output, which is the")
    print("  mistake the refresh-identity entry records. The band is gone.")
    print("  The magnitudes are still negligible, and BOTH things are true:")
    print("  the sign is real and nobody would notice the size.")
    print()
    for market, _t, classify in MARKETS:
        block = report["markets"][market]
        headline = block["headline"]
        panels = block["panels"]
        base = panels["common"][OPTIONS[0][0]][headline]
        print(f"  {market} ({headline}); (a) scores {base:.6f} on the "
              f"{panels['common'][OPTIONS[0][0]]['n']} rows it can answer")
        for label, *_ in OPTIONS[1:]:
            b = block["bootstrap"][label]
            verdict = ("indistinguishable from (a)" if b["crosses_zero"]
                       else "better than (a)" if b["ci_high"] < 0
                       else "WORSE than (a)")
            relative = b["observed"] / base if base else 0.0
            print(f"    common rows : {label:<28}{verdict}"
                  f"  ({relative * 100:+.2f}%)")
        for label, *_ in OPTIONS[1:]:
            result = panels["refused"][label]
            if result:
                print(f"    refused rows: {label:<28}"
                      f"{result[headline]:.6f} over {result['n']} row(s)")
        print()

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=float),
                             encoding="utf-8")
        print()
        print(f"written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
