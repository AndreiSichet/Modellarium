r"""Select the NHL's models on the VALIDATION season. Test is not touched.

THE TWO MARKETS THAT GET A MODEL, and the one that does not:

  WINNER      home win. Selected here. **If the best candidate TIES Elo
              alone, Elo alone ships** - it is the simpler object, and a
              25-feature model that cannot be distinguished from three
              rating columns is complexity with no verdict behind it. The
              NFL already ships two tie verdicts on the wire, so this is the
              house convention rather than a new rule.
  PUCK LINE   "home wins by 2+", as a probability against its base rate.
              **It ships only if the paired bootstrap's 95% interval
              excludes zero.** Otherwise the NHL ships the winner alone.
  TOTALS      NOT SERVED. Section 5's baselines measured LinearRegression
              LOSING to a constant on MAE (+0.74% on test) and tying it on
              RMSE, so there is no skill to ship. Declared with its reason
              rather than shipped weak; the /health shape is Phase 3's.

THE REGIME IS (d): CARRY AND IMPUTE. Decided in section 4 - rolling windows
and rest carry across the season boundary, and whatever is still missing
(the first season of the corpus, and an expansion team's first ten games)
goes through a median imputer carrying missingness indicators, so a model
can tell *the median* from *we did not know*.

EVERY CANDIDATE SEES ONE MATRIX, FITTED ON THE FOLD'S OWN TRAINING ROWS.
The imputer and the scaler are fitted once per fold and applied to both
sides, so the tree models and the linear models read identical numbers.
Letting XGBoost take its own `missing=` default branch instead would make
the comparison meaningless, and it is also the degeneracy the NBA's
early-season note measures: a row of mostly-missing features puts two
different fixtures in the same leaf of all 108 trees.

SELECTION IS ON THE DECLARED VALIDATION SEASON, and the rolling-origin folds
are reported BESIDE it rather than instead of it. The reason is measured:
Elo's gain over no skill correlates with that season's competitive balance
at r = 0.9261, and 2023-24 is the 4th most spread-out of thirteen seasons -
so a single-season ranking is exactly the kind that can be an artefact. If
the folds disagree with validation this script SAYS SO, rather than letting
the chosen ranking stand unremarked.

    python ml-training/nhl/select_nhl_models.py --table <path> --elo-json <path>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import nhl_features as F                                        # noqa: E402
from nhl_features import refuse_undeclared_seasons              # noqa: E402
from nhl_baselines import accuracy, brier, log_loss             # noqa: E402

BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261030

# The folds are the last four training seasons, each scored on while
# everything before it trains - rolling origin, the only honest shape for a
# time series, and what the WNBA's manifest already records.
FOLD_COUNT = 4

# XGBoost's early-stopping slice comes off the END of the fold's own
# training rows, which are in date order, so it is the most recent training
# games and never the season being scored. The weekly gate's recorded
# early-stopping asymmetry is what this avoids: a tree count chosen against
# the thing you then report is not a held-out number.
EARLY_STOP_ROWS = 1200
EARLY_STOP_ROUNDS = 50

ELO_COLUMNS = ("HOME_ELO", "AWAY_ELO", "ELO_DIFF")


# ==========================================================================
# one matrix, shared
# ==========================================================================

def prepare(train_frame, score_frame, features):
    """Impute then scale, fitted on the training rows only.

    THE INDICATOR COLUMNS ARE NAMED after the features they describe, so a
    coefficient or an importance can be read back to "we did not know this
    one" rather than to an anonymous extra column.
    """
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(
        imputer.fit_transform(train_frame[list(features)]))
    x_score = scaler.transform(imputer.transform(score_frame[list(features)]))
    indicated = getattr(imputer, "indicator_", None)
    extra = ([f"MISSING_{features[i]}" for i in indicated.features_]
             if indicated is not None else [])
    return x_train, x_score, list(features) + extra, (imputer, scaler)


def columns_at(names, wanted):
    """Positions of `wanted` in `names`, raising rather than guessing."""
    missing = [w for w in wanted if w not in names]
    if missing:
        raise KeyError(f"columns absent from the prepared matrix: {missing}")
    return [names.index(w) for w in wanted]


# ==========================================================================
# candidates
# ==========================================================================

class EloAlone:
    """The Elo expectation, with nothing fitted. A candidate, not a floor.

    IT IS IN THE CANDIDATE LIST ON PURPOSE. Treating Elo as a bar a model
    must clear, rather than as something that can simply win, is how a
    league ends up shipping a model nobody compared properly - and the NFL's
    two tie verdicts are the record of taking it seriously.

    IT READS THE RAW FRAME, not the prepared matrix, because it is a formula
    over ratings rather than a fit over features - scaling its inputs would
    change nothing it does and would obscure that.
    """

    name = "elo_alone"
    simple = True
    trees = None

    def __init__(self, home_advantage):
        self.home_advantage = home_advantage

    def fit(self, x_train, y_train, names, train_frame):
        return self

    def predict(self, x_score, score_frame):
        return np.asarray(F.elo_win_probability(
            score_frame["HOME_ELO"], score_frame["AWAY_ELO"],
            self.home_advantage), dtype=float)


class BaseRate:
    """The training rows' base rate. The puck line's no-skill reference."""

    name = "base_rate"
    simple = True
    trees = None

    def __init__(self):
        self.rate = None

    def fit(self, x_train, y_train, names, train_frame):
        self.rate = float(np.mean(y_train))
        return self

    def predict(self, x_score, score_frame):
        return np.full(len(score_frame), self.rate)


class Linear:
    """LogisticRegression over a named subset of the prepared matrix."""

    trees = None

    def __init__(self, name, wanted, simple=False):
        self.name = name
        self.wanted = wanted
        self.simple = simple
        self.model = None
        self.positions = None

    def fit(self, x_train, y_train, names, train_frame):
        self.positions = columns_at(names, self.wanted)
        self.model = LogisticRegression(max_iter=5000)
        self.model.fit(x_train[:, self.positions], y_train)
        return self

    def predict(self, x_score, score_frame):
        return self.model.predict_proba(x_score[:, self.positions])[:, 1]


class Boosted:
    """XGBoost over the whole prepared matrix, early-stopped inside train."""

    simple = False

    def __init__(self, name, depth, rate):
        self.name = name
        self.depth = depth
        self.rate = rate
        self.model = None
        self.trees = None

    def fit(self, x_train, y_train, names, train_frame):
        self.model = XGBClassifier(
            max_depth=self.depth, learning_rate=self.rate,
            n_estimators=600, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, eval_metric="logloss",
            early_stopping_rounds=EARLY_STOP_ROUNDS, verbosity=0)
        cut = max(len(x_train) - EARLY_STOP_ROWS, 1)
        self.model.fit(x_train[:cut], y_train[:cut],
                       eval_set=[(x_train[cut:], y_train[cut:])],
                       verbose=False)
        best = getattr(self.model, "best_iteration", None)
        self.trees = (int(best) + 1) if best is not None else None
        return self

    def predict(self, x_score, score_frame):
        return self.model.predict_proba(x_score)[:, 1]


def candidates(reference_name, home_advantage):
    """The candidate list. Deliberately short, for a measured reason.

    SECTION 5 MEASURED THE WHOLE 25-FEATURE LOGISTIC TYING ELO ALONE on the
    winner, so the interesting question is not which of twelve architectures
    wins - it is whether anything beats three rating columns at all. A long
    search over a 1% effect is a search for noise, and this project has
    seven run-and-rejected accuracy experiments on record saying so.
    """
    out = [EloAlone(home_advantage) if reference_name == "elo_alone"
           else BaseRate()]
    out.append(Linear("logistic_elo", ELO_COLUMNS, simple=True))
    out.append(Linear("logistic_all", tuple(F.feature_columns())))
    out.append(Boosted("xgb_shallow", 3, 0.05))
    out.append(Boosted("xgb_deeper", 5, 0.05))
    return out


# ==========================================================================
# scoring
# ==========================================================================

def evaluate(frame, features, target_name, reference_name, home_advantage,
             train_mask, score_mask):
    """Fit every candidate on `train_mask`, score it on `score_mask`."""
    train_frame = frame[train_mask]
    score_frame = frame[score_mask]
    y_train = frame.loc[train_mask, target_name].to_numpy(dtype=float)
    y_score = frame.loc[score_mask, target_name].to_numpy(dtype=float)
    x_train, x_score, names, _fitted = prepare(train_frame, score_frame,
                                               features)
    results = {}
    for candidate in candidates(reference_name, home_advantage):
        candidate.fit(x_train, y_train, names, train_frame)
        predicted = np.asarray(candidate.predict(x_score, score_frame),
                               dtype=float)
        results[candidate.name] = {
            "log_loss": log_loss(y_score, predicted),
            "brier": brier(y_score, predicted),
            "accuracy": accuracy(y_score, predicted),
            "trees": candidate.trees,
            "simple": candidate.simple,
            "predicted": predicted,
        }
    return results, y_score


def paired_bootstrap(y, reference, other, rng):
    """95% interval for other's log loss minus reference's, paired by row."""
    n = len(y)
    draws = rng.integers(0, n, size=(BOOTSTRAP, n))
    deltas = np.empty(BOOTSTRAP, dtype=float)
    for i in range(BOOTSTRAP):
        pick = draws[i]
        deltas[i] = (log_loss(y[pick], other[pick])
                     - log_loss(y[pick], reference[pick]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"observed": float(log_loss(y, other) - log_loss(y, reference)),
            "sd": float(deltas.std()), "ci_low": float(lo),
            "ci_high": float(hi), "crosses_zero": bool(lo < 0.0 < hi)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=None)
    parser.add_argument("--elo-json", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if args.elo_json:
        payload = json.loads(args.elo_json.read_text(encoding="utf-8"))
        elo = {"k": payload["k"],
               "home_advantage": payload["home_advantage"],
               "carryover": payload["carryover"]}
    else:
        elo = dict(F.ELO_DEFAULTS)
        print("no --elo-json; using ELO_DEFAULTS, not the fitted values")

    games = F.load_games(args.table)
    frame = F.build_features(games, elo=elo, carry_rolling=True,
                             carry_rest=True)
    refuse_undeclared_seasons(frame)
    features = tuple(F.feature_columns())

    frame = (frame.sort_values(["GAME_DATE", "GAME_ID"])
             .reset_index(drop=True))
    # THE PUCK LINE'S TARGET, derived here and nowhere else. 2+ rather than
    # 1+ because the NHL margin is never 0 - "wins by 1+" is the winner
    # market under another name.
    frame["PUCK_LINE"] = (frame["HOME_MARGIN"] >= 2).astype(int)

    seasons = F.season_order(frame)
    train_seasons = [s for s in seasons if s not in F.TEST_SEASONS
                     and s not in F.VALIDATION_SEASONS]
    train_mask = frame["SEASON"].isin(set(train_seasons))
    valid_mask = frame["SEASON"].isin(set(F.VALIDATION_SEASONS))

    incomplete = int((~F.is_complete(frame)).sum())
    print(f"regime (d) carry + impute: {len(frame)} fixtures, {incomplete} "
          f"with a missing value "
          f"({100.0 * incomplete / len(frame):.2f}%) - IMPUTED, not dropped")
    print(f"Elo k={elo['k']:g} home_advantage={elo['home_advantage']:g} "
          f"carryover={elo['carryover']:g}")
    print(f"training {int(train_mask.sum())} rows over "
          f"{len(train_seasons)} season(s) | validation "
          f"{int(valid_mask.sum())} ({', '.join(F.VALIDATION_SEASONS)})")
    print(f"test {', '.join(F.TEST_SEASONS)} - NOT SCORED HERE")
    print()

    report = {"elo": elo, "regime": "carry_rolling+carry_rest+impute",
              "features": list(features), "incomplete_imputed": incomplete,
              "training_seasons": train_seasons,
              "validation_seasons": list(F.VALIDATION_SEASONS),
              "bootstrap": {"resamples": BOOTSTRAP, "seed": BOOTSTRAP_SEED},
              "markets": {}}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    fold_list = train_seasons[-FOLD_COUNT:]

    for market, target_name, reference_name, ship_rule in (
            ("winner", "HOME_WIN", "elo_alone",
             "a tie with Elo alone ships Elo alone"),
            ("puck_line", "PUCK_LINE", "base_rate",
             "a tie with the base rate ships nothing")):
        print(f"{market.upper()}  (target {target_name}; reference "
              f"{reference_name})")
        print(f"  ship rule: {ship_rule}")

        validation, y_valid = evaluate(
            frame, features, target_name, reference_name,
            elo["home_advantage"], train_mask, valid_mask)

        folds = {name: [] for name in validation}
        for season in fold_list:
            before = [s for s in train_seasons
                      if int(str(s)[:4]) < int(str(season)[:4])]
            if not before:
                continue
            fold_result, _y = evaluate(
                frame, features, target_name, reference_name,
                elo["home_advantage"],
                frame["SEASON"].isin(set(before)),
                frame["SEASON"] == season)
            for name, result in fold_result.items():
                folds[name].append(result["log_loss"])

        width = max(len(name) for name in validation)
        print(f"    {'candidate':<{width}}{'valid LL':>11}{'Brier':>10}"
              f"{'acc':>9}{'fold mean':>12}{'fold sd':>10}{'trees':>7}")
        print("    " + "-" * (width + 59))
        for name, result in validation.items():
            values = folds[name]
            mean = f"{np.mean(values):.6f}" if values else "-"
            sd = f"{np.std(values):.6f}" if values else "-"
            print(f"    {name:<{width}}{result['log_loss']:>11.6f}"
                  f"{result['brier']:>10.6f}{result['accuracy']:>9.4f}"
                  f"{mean:>12}{sd:>10}"
                  f"{str(result['trees'] or '-'):>7}")

        ranked = sorted(validation, key=lambda n: validation[n]["log_loss"])
        best = ranked[0]
        fold_ranked = sorted((n for n in validation if folds[n]),
                             key=lambda n: float(np.mean(folds[n])))

        # A RANK DISAGREEMENT IS NOT A DISAGREEMENT UNLESS IT CLEARS THE
        # FOLD SPREAD, and the first version of this flag ignored that. It
        # compared rank orderings alone, so with candidates this close it
        # fired on a 0.0018 gap between fold means whose standard error is
        # ~0.0055 - a false alarm by construction, and the same shape of
        # mistake as a percentage band overruling a bootstrap. The flag now
        # fires only when the gap exceeds the standard error of the fold
        # mean, and it prints both numbers so the reader can see why.
        fold_best = fold_ranked[0] if fold_ranked else None
        gap = standard_error = 0.0
        if fold_best and folds.get(best):
            gap = abs(float(np.mean(folds[fold_best]))
                      - float(np.mean(folds[best])))
            spread = float(np.std(folds[best]))
            standard_error = spread / max(np.sqrt(len(folds[best])), 1.0)
        same = fold_best == best
        material = bool(fold_best and not same and gap > standard_error)
        agree = bool(same or not material)
        print(f"    best on validation : {best}")
        print(f"    best on the folds  : {fold_best or '-'}")
        if same:
            print("      the folds AGREE with validation")
        elif material:
            print(f"      the folds DISAGREE materially: gap {gap:.6f} "
                  f"exceeds the standard error of the fold mean "
                  f"{standard_error:.6f} - this is the single-season risk "
                  f"this script exists to surface")
        else:
            print(f"      a RANK difference only: gap {gap:.6f} is inside "
                  f"the standard error of the fold mean "
                  f"{standard_error:.6f}, so the folds do not prefer "
                  f"either")

        reference = validation[reference_name]["predicted"]
        decision = {}
        for name in ranked:
            if name != reference_name:
                decision[name] = paired_bootstrap(
                    y_valid, reference, validation[name]["predicted"], rng)
        print(f"    paired bootstrap on validation vs {reference_name}, "
              f"{BOOTSTRAP} resamples:")
        for name in ranked:
            if name == reference_name:
                continue
            b = decision[name]
            verdict = ("indistinguishable" if b["crosses_zero"]
                       else "BETTER" if b["ci_high"] < 0 else "worse")
            print(f"      {name:<{width}}{b['observed']:>+12.6f}  "
                  f"[{b['ci_low']:+.6f}, {b['ci_high']:+.6f}]  {verdict}")

        if best == reference_name:
            chosen = reference_name
            why = f"{reference_name} was the best candidate outright"
        elif decision[best]["crosses_zero"]:
            chosen = reference_name
            why = (f"{best} is indistinguishable from {reference_name} on "
                   f"validation (95% interval "
                   f"[{decision[best]['ci_low']:+.6f}, "
                   f"{decision[best]['ci_high']:+.6f}] includes zero), so "
                   f"the SIMPLER object ships")
        elif decision[best]["ci_high"] < 0:
            chosen = best
            why = (f"{best} beats {reference_name} with the 95% interval "
                   f"excluding zero ([{decision[best]['ci_low']:+.6f}, "
                   f"{decision[best]['ci_high']:+.6f}])")
        else:
            chosen = reference_name
            why = f"{best} is WORSE than {reference_name}"

        ships = not (market == "puck_line" and chosen == "base_rate")
        print(f"    CHOSEN: {chosen}")
        print(f"    reason: {why}")
        if not ships:
            print("    -> THE PUCK LINE DOES NOT SHIP. The NHL serves the "
                  "winner only.")
        print()

        report["markets"][market] = {
            "target": target_name, "reference": reference_name,
            "ship_rule": ship_rule, "chosen": chosen, "reason": why,
            "ships": ships, "best_on_validation": best,
            "folds_agree": agree, "fold_seasons": fold_list,
            "fold_rank_best": fold_best,
            "fold_gap": gap, "fold_standard_error": standard_error,
            "fold_disagreement_material": material,
            "validation": {
                name: {k: v for k, v in result.items() if k != "predicted"}
                for name, result in validation.items()},
            "fold_log_loss": {n: v for n, v in folds.items() if v},
            "bootstrap_vs_reference": decision,
        }

    report["markets"]["totals"] = {
        "target": "TOTAL_GOALS", "ships": False, "chosen": None,
        "reason": "NOT SERVED. Section 5's baselines measured "
                  "LinearRegression losing to a constant on MAE (+0.74% on "
                  "test) and tying it on RMSE (+0.29%), so there is no "
                  "skill to ship. Declared rather than shipped weak.",
    }
    print("TOTALS: not served. " + report["markets"]["totals"]["reason"])

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=float),
                             encoding="utf-8")
        print()
        print(f"written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
