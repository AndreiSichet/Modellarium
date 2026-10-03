"""Walk-forward model selection for the WNBA. Touches no test season.

Four folds: each validation season is scored by a model trained on every season
before it. One WNBA season is 192-240 games, and choosing among candidates on a
single season would select whatever got lucky; four folds and ~900 validation
games means a winner has to hold up across several years.

Writes the selection to models_wnba/selection.json for the test script to read.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from wnba_common import (  # noqa: E402
    MODELS_DIR, TARGETS, WINDOW_CANDIDATES, feature_columns, folds,
    load_dataset, load_long, section, with_fold_elo,
)
from train_moneyline_xgb import PARAMS as CLASSIFIER_PARAMS  # noqa: E402
from train_regression_xgb import PARAMS as REGRESSION_PARAMS  # noqa: E402

FAMILIES = ["naive", "elo", "linear", "xgboost"]


def expected_score(rating, opponent):
    return 1 / (1 + 10 ** ((opponent - rating) / 400))


def naive_prediction(target, window, rows, train):
    """The NBA's naive baselines, scaled to this league."""
    if target == "moneyline":
        # Always-home, at the training home-win rate.
        rate = float(train["HOME_WIN"].mean())
        return np.full(len(rows), rate)
    if target == "spread":
        return (rows[f"HOME_{window}_PTS"] - rows[f"AWAY_{window}_PTS"]).to_numpy()
    return (rows[f"HOME_{window}_PTS"] + rows[f"AWAY_{window}_PTS"]).to_numpy()


def fit_and_predict(family, target, window, train, validate, features):
    """One (family, target, window) on one fold. Returns predictions."""
    label = TARGETS[target]["label"]
    classification = target == "moneyline"

    if family == "naive":
        return naive_prediction(target, window, validate, train)

    if family == "elo":
        if not classification:
            return None
        return expected_score(validate["HOME_TEAM_ELO"].to_numpy(),
                              validate["AWAY_TEAM_ELO"].to_numpy())

    if family == "linear":
        # Scaler inside the pipeline, so it is fit on TRAIN ROWS ONLY and the
        # validation rows are only ever transformed.
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000) if classification
            else LinearRegression())
        model.fit(train[features], train[label])
        return (model.predict_proba(validate[features])[:, 1] if classification
                else model.predict(validate[features]))

    # XGBoost. Early stopping needs an eval set, and using the season being
    # SCORED for it would fit on what is about to be measured. So the last
    # training season is carved off as an inner early-stopping set and the
    # model trains on the rest.
    seasons = sorted(train["SEASON"].unique())
    inner_stop = train[train["SEASON"] == seasons[-1]]
    inner_fit = train[train["SEASON"] < seasons[-1]]
    params = CLASSIFIER_PARAMS if classification else REGRESSION_PARAMS
    model = (XGBClassifier(**params) if classification
             else XGBRegressor(**params))
    model.fit(inner_fit[features], inner_fit[label],
              eval_set=[(inner_stop[features], inner_stop[label])],
              verbose=False)
    return (model.predict_proba(validate[features])[:, 1] if classification
            else model.predict(validate[features]))


def score(target, truth, predicted):
    if TARGETS[target]["metric"] == "log_loss":
        return float(log_loss(truth, np.clip(predicted, 1e-15, 1 - 1e-15)))
    return float(mean_absolute_error(truth, predicted))


def main() -> int:
    print(__doc__)
    dataset = load_dataset()
    long_frame = load_long()

    section("THE SPLIT")
    for validation_season, training in folds():
        print(f"  fold: train {training[0]}-{training[-1]}  -> validate "
              f"{validation_season}")
    print("\n  test seasons are not loaded by this script at all.")

    section("HOME-WIN RATE PER SEASON")
    rates = dataset.groupby("SEASON")["HOME_WIN"].agg(["mean", "size"])
    for season, row in rates.iterrows():
        print(f"  {season}  {row['mean'] * 100:5.1f}%  ({int(row['size'])} games)")
    print(f"\n  overall {dataset['HOME_WIN'].mean() * 100:.1f}%  "
          "(the NBA's is 56.4%)")

    results = []
    elo_per_fold = {}

    for validation_season, training in folds():
        section(f"FOLD: validate {validation_season}, train "
                f"{training[0]}-{training[-1]}")
        fold_data, elo_info = with_fold_elo(dataset, long_frame, training)
        elo_per_fold[validation_season] = elo_info
        print(f"  Elo refit on {training[0]}-{training[-1]}: "
              f"K={elo_info['k']}, carryover={elo_info['carryover']:.3f}, "
              f"train log loss {elo_info['train_log_loss']:.4f}")

        # The rows EVERY candidate can score, so the window comparison is a
        # comparison of windows rather than of row sets.
        all_features = sorted({c for w in WINDOW_CANDIDATES
                               for c in feature_columns(w)})
        season_rows = fold_data[fold_data["SEASON"] == validation_season]
        common_ids = set(season_rows.dropna(subset=all_features)["GAME_ID"])
        print(f"\n  rows all three candidates can score: {len(common_ids)} of "
              f"{len(season_rows)} ({len(common_ids) / len(season_rows) * 100:.0f}%)")

        for window in WINDOW_CANDIDATES:
            features = feature_columns(window)
            usable = fold_data.dropna(subset=features)
            train = usable[usable["SEASON"].isin(training)]
            validate = usable[usable["SEASON"] == validation_season]
            common_mask = validate["GAME_ID"].isin(common_ids)
            print(f"  {window}: {len(train):,} train rows, "
                  f"{len(validate):,} validation rows "
                  f"({len(validate) / len(season_rows) * 100:.0f}% retained), "
                  f"{int(common_mask.sum())} common")

            for target in TARGETS:
                truth = validate[TARGETS[target]["label"]]
                for family in FAMILIES:
                    predicted = fit_and_predict(family, target, window,
                                                train, validate, features)
                    if predicted is None:
                        continue
                    results.append({
                        "fold": validation_season, "window": window,
                        "target": target, "family": family,
                        "score": score(target, truth, predicted),
                        "score_common": score(
                            target, truth[common_mask.to_numpy()],
                            np.asarray(predicted)[common_mask.to_numpy()]),
                        "n": len(validate),
                        "n_common": int(common_mask.sum()),
                    })

    frame = pd.DataFrame(results)

    section("VALIDATION TABLE - EVERY CANDIDATE, EVERY TARGET")
    print("""Mean and standard deviation across the four folds; lower is better on all
three metrics. TWO columns of means, and the second is the one selection uses:

  OWN      scored on the rows that candidate can score
  COMMON   scored on the rows ALL THREE candidates can score

They differ because ROLL5 retains 82-85% of a season against CARRY's 97%, and
the rows it drops are early-season ones. Comparing OWN across windows compares
row sets as much as windows, so COMMON is the fair comparison and OWN is shown
beside it because retention is a real deployment consideration.
""")

    selection = {}
    for target in TARGETS:
        rows = frame[frame["target"] == target]
        metric = TARGETS[target]["metric"]
        print(f"  {target.upper()}  ({metric})")
        print(f"    {'FAMILY':<10}{'WINDOW':<10}{'OWN':>9}{'COMMON':>10}{'SD':>8}"
              + "".join(f"{s:>9}" for s in sorted(rows['fold'].unique())))
        print("    " + "-" * 73)

        table = []
        for family in FAMILIES:
            for window in WINDOW_CANDIDATES:
                cell = rows[(rows["family"] == family) & (rows["window"] == window)]
                if cell.empty:
                    continue
                own = cell.set_index("fold")["score"].sort_index()
                common = cell.set_index("fold")["score_common"].sort_index()
                table.append((common.mean(), common.std(), family, window,
                              common, own.mean()))
        table.sort()
        for mean, sd, family, window, per_fold, own_mean in table:
            star = " *" if (family, window) == (table[0][2], table[0][3]) else "  "
            print(f"   {star}{family:<10}{window:<10}{own_mean:>9.4f}"
                  f"{mean:>10.4f}{sd:>8.4f}"
                  + "".join(f"{v:>9.4f}" for v in per_fold))

        best_mean, best_sd, best_family, best_window, _, best_own = table[0]
        runner = table[1]
        selection[target] = {
            "family": best_family, "window": best_window,
            "validation_mean": best_mean, "validation_sd": best_sd,
            "runner_up": {"family": runner[2], "window": runner[3],
                          "validation_mean": runner[0], "validation_sd": runner[1]},
        }
        gap = (runner[0] - best_mean) / best_mean * 100
        print(f"\n    SELECTED {best_family} / {best_window}  "
              f"mean {best_mean:.4f} (sd {best_sd:.4f})")
        print(f"    runner-up {runner[2]} / {runner[3]} at {runner[0]:.4f} "
              f"- a {gap:.2f}% gap")
        if gap < 1.0:
            print(f"    NOTE: under 1%. The winner is barely separated from the")
            print(f"          runner-up, so treat this as a weak preference.")
        if best_sd > abs(best_mean) * 0.08:
            print(f"    NOTE: fold-to-fold spread is large relative to the mean.")
        print()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"selection": selection,
               "elo_per_fold": {str(k): v for k, v in elo_per_fold.items()},
               "candidates": WINDOW_CANDIDATES, "families": FAMILIES,
               "folds": {str(v): t for v, t in folds()}}
    (MODELS_DIR / "selection.json").write_text(json.dumps(payload, indent=2))
    frame.to_csv(MODELS_DIR / "validation_scores.csv", index=False)

    section("WRITTEN")
    print(f"  {MODELS_DIR / 'selection.json'}")
    print(f"  {MODELS_DIR / 'validation_scores.csv'}  ({len(frame)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
