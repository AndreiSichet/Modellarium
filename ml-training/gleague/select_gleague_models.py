"""G League model selection - stage A chooses the training span, stage B the
window and model family. Validation only; test is never read here.

Two stages in a fixed order, because three axes chosen simultaneously against
four folds is a large space and more axes means more ways to overfit the
selection. The order lives in gleague_common.STAGE_ORDER, declared before
anything is scored.

EVERY COMPARISON IS SCORED ON COMMON ROWS - the rows every candidate in that
comparison can score. The WNBA phase found this changed two of three
selections: a window that drops early-season games is being asked an easier
question, so scoring each candidate on its own rows compares row sets as much
as it compares windows.
"""

import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from gleague_common import (  # noqa: E402
    ADOPTION_NOTE, DISRUPTED_SEASONS, IDENTITY_PATH, MODELS_DIR,
    MODEL_FAMILIES, TEST_SEASONS,
    SPAN_CANDIDATES, STAGE_A_REFERENCE, STAGE_ORDER, TARGETS,
    VALIDATION_FOLDS, WINDOW_CANDIDATES, all_seasons, bootstrap_difference,
    errors, feature_columns, fit_elo_on, folds, load_dataset, load_long,
    score, section, span_seasons, usable, with_fold_elo)

SELECTION_PATH = None  # set in main, under MODELS_DIR

XGB_CONFIG = {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 2000,
              "early_stopping_rounds": 50, "random_state": 0,
              "verbosity": 0}


def build_model(kind: str, family: str):
    if family == "linear":
        estimator = (LogisticRegression(max_iter=2000)
                     if kind == "classification" else LinearRegression())
        return Pipeline([("scale", StandardScaler()),
                         ("model", estimator)])
    estimator = (XGBClassifier(eval_metric="logloss", **XGB_CONFIG)
                 if kind == "classification" else XGBRegressor(**XGB_CONFIG))
    return estimator


def fit_predict(family: str, kind: str, train: pd.DataFrame,
                evaluate: pd.DataFrame, columns: list, label: str,
                inner: pd.DataFrame = None):
    """Fit on `train`, predict `evaluate`.

    XGBOOST'S EARLY STOPPING NEEDS AN EVAL SET, AND THE OBVIOUS ONE IS A LEAK.
    Using the season about to be scored would fit on what is about to be
    measured, so each fold carves its LAST TRAINING SEASON off as an inner
    early-stopping set and trains on the rest. The scored season is never seen
    during fitting.
    """
    model = build_model(kind, family)
    x_train = train[columns].to_numpy(dtype=float)
    y_train = train[label].to_numpy(dtype=float)

    if family == "xgboost":
        x_inner = inner[columns].to_numpy(dtype=float)
        y_inner = inner[label].to_numpy(dtype=float)
        model.fit(x_train, y_train, eval_set=[(x_inner, y_inner)],
                  verbose=False)
    else:
        model.fit(x_train, y_train)

    x_eval = evaluate[columns].to_numpy(dtype=float)
    if kind == "classification":
        return model.predict_proba(x_eval)[:, 1]
    return model.predict(x_eval)


def split_inner(train: pd.DataFrame) -> tuple:
    """The last training season becomes the inner early-stopping set."""
    seasons = sorted(train["SEASON"].unique())
    if len(seasons) < 2:
        return train, train
    held = seasons[-1]
    return train[train["SEASON"] != held], train[train["SEASON"] == held]


def naive_prediction(target: str, window: str, train: pd.DataFrame,
                     evaluate: pd.DataFrame) -> np.ndarray:
    if target == "moneyline":
        rate = float(train["HOME_WIN"].mean())
        return np.full(len(evaluate), rate)
    if target == "spread":
        return (evaluate[f"HOME_{window}_PTS"]
                - evaluate[f"AWAY_{window}_PTS"]).to_numpy(dtype=float)
    return (evaluate[f"HOME_{window}_PTS"]
            + evaluate[f"AWAY_{window}_PTS"]).to_numpy(dtype=float)


def evaluate_fold(frame: pd.DataFrame, fold: dict, window: str, family: str,
                  target: str, common: pd.Index) -> dict:
    spec = TARGETS[target]
    columns = feature_columns(window)

    train = frame[frame["SEASON"].isin(fold["training"])]
    train = train[usable(train, window, target)]
    evaluate = frame.loc[common]

    outer, inner = split_inner(train)
    predicted = fit_predict(family, spec["kind"], outer, evaluate, columns,
                            spec["label"], inner)

    y = evaluate[spec["label"]].to_numpy(dtype=float)
    out = {
        "model": score(spec["kind"], y, predicted),
        "naive": score(spec["kind"], y,
                       naive_prediction(target, window, train, evaluate)),
        "rows": len(evaluate),
    }
    if spec["kind"] == "classification":
        out["elo"] = score(spec["kind"], y,
                           evaluate["ELO_EXPECTED"].to_numpy(dtype=float))
    return out


def common_rows(frame: pd.DataFrame, windows: list, target: str,
                seasons: list) -> pd.Index:
    """Rows EVERY window in the comparison can score, in those seasons."""
    mask = frame["SEASON"].isin(seasons)
    for window in windows:
        mask &= usable(frame, window, target)
    return frame.index[mask]


def fold_frames(dataset, long_frame, identity, span):
    """Each fold with its refitted Elo, computed once and reused."""
    prepared = []
    for fold in folds(dataset, span):
        k, carryover = fit_elo_on(long_frame, identity, fold["training"])
        frame = with_fold_elo(dataset, long_frame, identity, k, carryover)
        prepared.append({**fold, "k": k, "carryover": carryover,
                         "frame": frame})
    return prepared


def stage_a(dataset, long_frame, identity) -> str:
    section(f"STAGE {STAGE_ORDER[0]}")
    window = STAGE_A_REFERENCE["window"]
    family = STAGE_A_REFERENCE["family"]
    print(f"  reference configuration: {family} / {window}, fixed in "
          f"gleague_common before\n  anything was scored")
    print(f"  spans: " + ", ".join(
        f"{name} (from {start})" for name, start in SPAN_CANDIDATES.items()))

    table = {}
    for span in SPAN_CANDIDATES:
        prepared = fold_frames(dataset, long_frame, identity, span)
        print(f"\n  {span} (from {SPAN_CANDIDATES[span]}): "
              f"{len(prepared)} fold(s)")
        for fold in prepared:
            print(f"    {fold['validation']}  training "
                  f"{fold['training'][0]} .. {fold['training'][-1]} "
                  f"({len(fold['training'])} seasons)  "
                  f"Elo K={fold['k']}, carryover={fold['carryover']:.3f}")
        table[span] = prepared

    print(f"\n  {'target':<11}{'span':<9}" +
          "".join(f"{f:>10}" for f in VALIDATION_FOLDS) +
          f"{'mean':>9}{'sd':>8}")
    print("  " + "-" * (20 + 10 * len(VALIDATION_FOLDS) + 17))

    chosen = {}
    for target in TARGETS:
        means = {}
        for span, prepared in table.items():
            scores = []
            for fold in prepared:
                common = common_rows(fold["frame"], [window], target,
                                     [fold["validation"]])
                result = evaluate_fold(fold["frame"], fold, window, family,
                                       target, common)
                scores.append(result["model"])
            means[span] = (float(np.mean(scores)), float(np.std(scores)),
                           scores)
            print(f"  {target:<11}{span:<9}" +
                  "".join(f"{s:>10.4f}" for s in scores) +
                  f"{np.mean(scores):>9.4f}{np.std(scores):>8.4f}")
        best = min(means, key=lambda s: means[s][0])
        other = [s for s in means if s != best][0]
        gap = (means[other][0] - means[best][0]) / means[other][0] * 100
        spread = means[best][1] / means[best][0] * 100
        chosen[target] = {"best": best, "gap_pct": gap,
                          "fold_spread_pct": spread}
        print(f"  {'':<11}-> {best} better by {gap:.2f}%, "
              f"fold spread {spread:.2f}%"
              f"{'  (within the spread: a tie)' if gap < spread else ''}")

    # ONE SPAN FOR ALL THREE TARGETS. A per-target span would be a third
    # selection on four folds, and the artifacts would then disagree about
    # what history the league has - which phase 4 would have to serve.
    votes = [chosen[t]["best"] for t in TARGETS]
    winner = max(set(votes), key=votes.count)
    print(f"\n  per-target preference: " +
          ", ".join(f"{t}={chosen[t]['best']}" for t in TARGETS))
    print(f"  STAGE A CHOOSES: {winner} (from "
          f"{SPAN_CANDIDATES[winner]}), on {votes.count(winner)} of "
          f"{len(votes)} targets")
    if len(set(votes)) > 1:
        print("""  The targets disagree, so one span is taken on the majority. A per-target
  span would be a third selection on four folds, and the artifacts would then
  disagree about how much history the league has - which phase 4 must serve.""")
    return winner, chosen


def stage_b(dataset, long_frame, identity, span) -> dict:
    section(f"STAGE {STAGE_ORDER[1]}")
    print(f"  span: {span} (from {SPAN_CANDIDATES[span]}), chosen by stage A")
    print(f"  windows: {', '.join(WINDOW_CANDIDATES)}")
    print(f"  families: {', '.join(MODEL_FAMILIES)}")
    print(f"  {ADOPTION_NOTE}")

    prepared = fold_frames(dataset, long_frame, identity, span)
    selected = {}

    for target in TARGETS:
        spec = TARGETS[target]
        print(f"\n  {target.upper()}  ({spec['metric']}, lower is better)")
        print(f"    {'family':<9}{'window':<9}" +
              "".join(f"{f:>10}" for f in VALIDATION_FOLDS) +
              f"{'mean':>9}{'sd':>8}{'rows':>7}")
        print("    " + "-" * (18 + 10 * len(VALIDATION_FOLDS) + 24))

        results = {}
        naive_means, elo_means = [], []
        for family in MODEL_FAMILIES:
            for window in WINDOW_CANDIDATES:
                scores, rows = [], 0
                naive, elo = [], []
                for fold in prepared:
                    common = common_rows(fold["frame"], WINDOW_CANDIDATES,
                                         target, [fold["validation"]])
                    result = evaluate_fold(fold["frame"], fold, window,
                                           family, target, common)
                    scores.append(result["model"])
                    naive.append(result["naive"])
                    if "elo" in result:
                        elo.append(result["elo"])
                    rows += result["rows"]
                results[(family, window)] = (float(np.mean(scores)),
                                             float(np.std(scores)), scores)
                naive_means.append(float(np.mean(naive)))
                if elo:
                    elo_means.append(float(np.mean(elo)))
                print(f"    {family:<9}{window:<9}" +
                      "".join(f"{s:>10.4f}" for s in scores) +
                      f"{np.mean(scores):>9.4f}{np.std(scores):>8.4f}"
                      f"{rows:>7}")

        print(f"    {'naive':<9}{'':<9}" + " " * (10 * len(VALIDATION_FOLDS))
              + f"{np.mean(naive_means):>9.4f}")
        if elo_means:
            print(f"    {'Elo alone':<9}{'':<9}"
                  + " " * (10 * len(VALIDATION_FOLDS))
                  + f"{np.mean(elo_means):>9.4f}")

        best = min(results, key=lambda key: results[key][0])
        best_mean, best_sd, _ = results[best]
        runner = min((k for k in results if k != best),
                     key=lambda key: results[key][0])
        gap = (results[runner][0] - best_mean) / results[runner][0] * 100
        spread = best_sd / best_mean * 100

        carry = min((k for k in results if k[1].startswith("CARRY")),
                    key=lambda key: results[key][0])
        cup = min((k for k in results if k[1].startswith("CUP")),
                  key=lambda key: results[key][0])
        cup_gap = ((results[carry][0] - results[cup][0])
                   / results[carry][0] * 100)

        # THE PICK IS ARGMIN AND THE MARGIN IS A COIN FLIP, so the record says
        # which candidates it tied with rather than presenting it as a
        # finding. Phase 4 still needs one configuration to serve; what it
        # must not inherit is the impression that this one was measurably
        # better.
        tied = sorted(
            f"{f}/{w}" for (f, w) in results
            if (f, w) != best
            and abs(results[(f, w)][0] - best_mean) / best_mean * 100 < spread)

        selected[target] = {
            "tied_with": tied,
            "family": best[0], "window": best[1],
            "validation_mean": best_mean, "validation_sd": best_sd,
            "runner_up": {"family": runner[0], "window": runner[1],
                          "validation_mean": results[runner][0]},
            "gap_pct": gap, "fold_spread_pct": spread,
            "naive_mean": float(np.mean(naive_means)),
            "elo_mean": float(np.mean(elo_means)) if elo_means else None,
            "cup_vs_carry_pct": cup_gap,
        }
        print(f"    -> {best[0]} / {best[1]}, mean {best_mean:.4f}; "
              f"runner-up {runner[0]}/{runner[1]} by {gap:.2f}% "
              f"(fold spread {spread:.2f}%)"
              f"{'  TIE' if gap < spread else ''}")
        if tied:
            print(f"    -> tied with {len(tied)} other candidate(s) inside "
                  f"the fold spread: {', '.join(tied)}")
        print(f"    -> best CUP ({cup[1]}) against best CARRY ({carry[1]}): "
              f"{cup_gap:+.2f}% for CUP")

    return selected, prepared


def cup_question(selected: dict, prepared: list) -> None:
    section("CARRY AGAINST CUP - THE QUESTION PHASE 2 BUILT THE CANDIDATES FOR")
    cup_folds = [f["validation"] for f in prepared
                 if f["validation"] >= "2021-22"]
    print(f"  The Cup exists in {len(cup_folds)} of {len(prepared)} folds "
          f"({', '.join(cup_folds)}) and in both test\n  seasons, so this is "
          f"decided on more than one fold - better than phase 2's framing\n"
          f"  suggested, and still not the whole validation window.")
    print()
    for target, pick in selected.items():
        verdict = ("CUP" if pick["cup_vs_carry_pct"] > 0 else "CARRY")
        decisive = abs(pick["cup_vs_carry_pct"]) > pick["fold_spread_pct"]
        print(f"  {target:<11}{pick['cup_vs_carry_pct']:+.2f}% for CUP   "
              f"-> {verdict} ahead, "
              f"{'outside' if decisive else 'INSIDE'} the fold spread of "
              f"{pick['fold_spread_pct']:.2f}%"
              f"{'' if decisive else ' - a tie'}")
    chosen_cup = sum(1 for p in selected.values()
                     if p["window"].startswith("CUP"))
    print(f"\n  selected windows: {chosen_cup} of {len(selected)} are "
          f"Cup-inclusive")


def main() -> int:
    print(__doc__)
    dataset = load_dataset()
    long_frame = load_long()
    identity = pd.read_csv(IDENTITY_PATH, dtype={"TEAM_ID": "int64"})

    section("THE SPLIT, DECLARED BEFORE SCORING")
    seasons = all_seasons(dataset)
    print(f"  seasons in the dataset : {len(seasons)}  "
          f"({seasons[0]} .. {seasons[-1]})")
    print(f"  validation folds       : {', '.join(VALIDATION_FOLDS)}")
    print(f"  test (not read here)   : {', '.join(TEST_SEASONS)}")
    for season, why in DISRUPTED_SEASONS.items():
        in_val = season in VALIDATION_FOLDS
        print(f"  {season} ({why}): in training, "
              f"{'ALSO A VALIDATION TARGET - WRONG' if in_val else 'never a validation target'}")
        if in_val:
            raise SystemExit(f"{season} is a validation target")

    for span in SPAN_CANDIDATES:
        print(f"  span {span:<7} -> {len(span_seasons(dataset, span))} "
              f"seasons from {SPAN_CANDIDATES[span]}")

    print(f"\n  stages, in order:")
    for stage in STAGE_ORDER:
        print(f"    {stage}")

    span, stage_a_detail = stage_a(dataset, long_frame, identity)
    selected, prepared = stage_b(dataset, long_frame, identity, span)
    cup_question(selected, prepared)

    section("SELECTED")
    for target, pick in selected.items():
        print(f"  {target:<11}{pick['family']:<9}{pick['window']:<9}"
              f"validation {pick['validation_mean']:.4f}  "
              f"naive {pick['naive_mean']:.4f}"
              + (f"  Elo {pick['elo_mean']:.4f}"
                 if pick["elo_mean"] else ""))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "selection.json"
    path.write_text(json.dumps({
        "span": span,
        "span_start": SPAN_CANDIDATES[span],
        "stage_order": STAGE_ORDER,
        "stage_a_reference": STAGE_A_REFERENCE,
        "stage_a": stage_a_detail,
        "window_candidates": WINDOW_CANDIDATES,
        "model_families": MODEL_FAMILIES,
        "validation_folds": VALIDATION_FOLDS,
        "elo_per_fold": [{"validation": f["validation"], "k": f["k"],
                          "carryover": f["carryover"]} for f in prepared],
        "selected": selected,
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {path.name} - test has NOT been read")
    return 0


if __name__ == "__main__":
    sys.exit(main())
