r"""Freeze the NHL's models: refit, score test ONCE, calibrate, write provenance.

WHAT SHIPS, decided by select_nhl_models.py on the validation season:

  WINNER      **Elo alone**, and there is no fitted artifact. It was the best
              candidate outright on validation - not a tie that the
              simpler-object rule broke, an outright win - so the league's
              winner market is three rating columns and a formula. This is a
              first for this project and the manifest says so plainly: family
              `elo`, no `.joblib`, the parameters ARE the model.
  PUCK LINE   `logistic_elo` - LogisticRegression over HOME_ELO, AWAY_ELO and
              ELO_DIFF, behind a scaler. Beat its base rate with the paired
              95% interval excluding zero, so it ships.
  TOTALS      NOT SERVED, declared with its reason.

TWO THINGS ARE REPORTED FOR THE PUCK LINE AND THAT IS DELIBERATE. The
shipped artifact is refitted on training + validation, because withholding a
whole season from the final weights for no reason is waste. But the retrain
gate's rule says promotion ships the EVALUATED candidate - "refitting before
shipping would promote weights no verdict was ever reached about" - so the
training-only model's test score is reported BESIDE the shipped one. Here the
tension is resolvable in a way the gate's is not: the test seasons are
untouched, so they give an independent reading of the shipped weights. If the
two columns disagree materially, that is the finding.

THE IMPUTER IS NOT IN THE PUCK-LINE ARTIFACT, and that is measured rather
than assumed. Its three inputs are Elo columns, which are **never missing**
(checked: 0 NaN across 15,972 fixtures; a team with no history carries the
1500 baseline, which is a defined value and not an unknown). Regime (d)'s
imputer would therefore be inert, and an inert transformer in a shipped
artifact is clutter that invites someone to assume a NaN would be handled.
A NaN Elo would instead be a bug upstream, and `requires_elo` says so.

    python ml-training/nhl/finalize_nhl_models.py --table <path> \
        --elo-json <path> --selection-json <path>
"""
import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import nhl_features as F                                        # noqa: E402
from nhl_features import refuse_undeclared_seasons              # noqa: E402
from nhl_baselines import accuracy, brier, log_loss             # noqa: E402

MODELS_DIR = PROJECT / "ml-training" / "models_nhl"

# Calibration bins. TEN EQUAL-WIDTH BINS, not deciles of the prediction:
# equal-width is what makes a reliability table readable as "when the model
# said 60%, how often did it happen", which is the question. Deciles answer
# a different one and hide the tails, and the NFL's winner calibrator entry
# already records that the tails are where this kind of model dies.
CALIBRATION_BINS = 10

ELO_COLUMNS = ["HOME_ELO", "AWAY_ELO", "ELO_DIFF"]


def git(*args):
    try:
        return subprocess.run(("git",) + args, cwd=PROJECT, text=True,
                              capture_output=True, check=True).stdout.strip()
    except Exception:                                           # noqa: BLE001
        return None


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reliability(actual, predicted, bins=CALIBRATION_BINS):
    """A reliability table plus ECE and the Murphy decomposition.

    RELIABILITY AS A SHARE OF RELIABILITY + RESOLUTION is the figure this
    project compares across leagues - the NBA's is 2.3% and the NFL's 11.5%,
    and the NFL's is why a winner calibrator is on its backlog. It is
    reported here so the NHL joins that comparison rather than inventing its
    own measure.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    # The last bin is closed on the right so a prediction of exactly 1.0
    # lands somewhere rather than nowhere.
    index = np.clip(np.digitize(predicted, edges[1:-1], right=False),
                    0, bins - 1)
    rows = []
    overall = float(actual.mean())
    ece = rel = res = 0.0
    for b in range(bins):
        pick = index == b
        n = int(pick.sum())
        if not n:
            rows.append({"bin": f"[{edges[b]:.1f}, {edges[b + 1]:.1f})",
                         "n": 0, "mean_predicted": None,
                         "observed": None, "gap": None})
            continue
        mean_predicted = float(predicted[pick].mean())
        observed = float(actual[pick].mean())
        weight = n / len(actual)
        ece += weight * abs(mean_predicted - observed)
        rel += weight * (mean_predicted - observed) ** 2
        res += weight * (observed - overall) ** 2
        rows.append({"bin": f"[{edges[b]:.1f}, {edges[b + 1]:.1f})",
                     "n": n, "mean_predicted": mean_predicted,
                     "observed": observed,
                     "gap": mean_predicted - observed})
    uncertainty = overall * (1.0 - overall)
    return {
        "bins": rows, "ece": float(ece),
        "reliability": float(rel), "resolution": float(res),
        "uncertainty": float(uncertainty),
        "reliability_share": (float(rel / (rel + res)) if (rel + res) else
                              None),
        "brier_from_decomposition": float(rel - res + uncertainty),
        "base_rate": overall,
    }


def win_share_spread(frame, season):
    """The sd of teams' final win share that season.

    FROZEN BESIDE EVERY TEST NUMBER, because Elo's gain over no skill
    correlates with it at r = 0.9261 over thirteen seasons. Without it, a
    future retrain moving a figure is indistinguishable from the league
    having been more or less competitive, and this project has spent real
    time on exactly that kind of ambiguity.
    """
    block = frame[frame["SEASON"] == season]
    wins = {}
    for row in block.itertuples():
        for team in (row.HOME_ID, row.AWAY_ID):
            wins.setdefault(team, [0, 0])
        wins[row.HOME_ID][1] += 1
        wins[row.AWAY_ID][1] += 1
        winner = row.HOME_ID if row.HOME_MARGIN > 0 else row.AWAY_ID
        wins[winner][0] += 1
    shares = np.array([w / n for w, n in wins.values()], dtype=float)
    return {"sd": float(shares.std()), "min": float(shares.min()),
            "max": float(shares.max()), "teams": len(shares)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=None)
    parser.add_argument("--elo-json", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    elo_payload = json.loads(args.elo_json.read_text(encoding="utf-8"))
    elo = {"k": elo_payload["k"],
           "home_advantage": elo_payload["home_advantage"],
           "carryover": elo_payload["carryover"]}
    selection = json.loads(args.selection_json.read_text(encoding="utf-8"))

    games = F.load_games(args.table)
    frame = F.build_features(games, elo=elo, carry_rolling=True,
                             carry_rest=True)
    refuse_undeclared_seasons(frame)
    frame = (frame.sort_values(["GAME_DATE", "GAME_ID"])
             .reset_index(drop=True))
    frame["PUCK_LINE"] = (frame["HOME_MARGIN"] >= 2).astype(int)

    seasons = F.season_order(frame)
    train_seasons = [s for s in seasons if s not in F.TEST_SEASONS
                     and s not in F.VALIDATION_SEASONS]
    train_only = frame["SEASON"].isin(set(train_seasons))
    train_plus = frame["SEASON"].isin(
        set(train_seasons) | set(F.VALIDATION_SEASONS))
    test = frame["SEASON"].isin(set(F.TEST_SEASONS))

    # THE ELO COLUMNS MUST NOT BE NULL, checked rather than assumed - the
    # puck-line artifact has no imputer precisely because of this.
    nulls = {c: int(frame[c].isna().sum()) for c in ELO_COLUMNS}
    if any(nulls.values()):
        print(f"ABORT: Elo columns carry nulls {nulls}; the puck-line "
              f"artifact ships without an imputer on the measured basis "
              f"that they never do")
        return 2
    print(f"checked: the three Elo columns carry 0 nulls across "
          f"{len(frame)} fixtures")

    print(f"train {int(train_only.sum())} | train+validation "
          f"{int(train_plus.sum())} | test {int(test.sum())} "
          f"({', '.join(F.TEST_SEASONS)})")
    print()

    y_test_winner = frame.loc[test, "HOME_WIN"].to_numpy(dtype=float)
    y_test_puck = frame.loc[test, "PUCK_LINE"].to_numpy(dtype=float)

    manifest = {
        "league": "NHL",
        "built": dt.datetime.now(dt.timezone.utc).isoformat(),
        "attribution": "Game results derived from English Wikipedia, "
                       "CC BY-SA 4.0. No NHL table is committed.",
        "seasons_trained_on": [s for s in seasons if s not in F.TEST_SEASONS],
        "test_seasons": list(F.TEST_SEASONS),
        "regime": selection["regime"],
        "prediction_rule": "dependency",
        "prediction_rule_note":
            "A fixture is predictable once BOTH teams' previous games are in "
            "history. MAX_DAYS_AHEAD must NOT be applied to the NHL - same "
            "rule and same exemption as the NFL's. Measured on 2025-26 at an "
            "08:30 UTC refresh: 1,292 of 1,312 predictable against the date "
            "rule's 1,242.",
        "requires_refresh_at": "08:30 UTC",
        "requires_refresh_note":
            "An NHL result is final at midnight Pacific on D+1, which is "
            "07:00 UTC in PDT and 08:00 UTC in PST. At 07:15 the dependency "
            "rule leaves 248 fixtures never predictable; at 08:30 it leaves "
            "20, and those 20 are season openers where one side has no "
            "previous game.",
        "elo": dict(elo, baseline=F.ELO_BASELINE,
                    fitted_on=selection["training_seasons"],
                    selected_on=selection["validation_seasons"],
                    validation_log_loss=elo_payload["validation_log_loss"],
                    parameter_spans={
                        name: block["span"]
                        for name, block in elo_payload["sweeps"].items()}),
        "features": selection["features"],
        "targets": {},
    }
    report = {"calibration": {}, "description_only": {}}

    # ================================================================
    # WINNER - Elo alone, no artifact
    # ================================================================
    winner = selection["markets"]["winner"]
    predicted = np.asarray(F.elo_win_probability(
        frame.loc[test, "HOME_ELO"], frame.loc[test, "AWAY_ELO"],
        elo["home_advantage"]), dtype=float)
    base_rate = float(frame.loc[train_plus, "HOME_WIN"].mean())
    no_skill = np.full(len(y_test_winner), base_rate)

    print("WINNER - Elo alone, no fitted artifact")
    print(f"  test log loss {log_loss(y_test_winner, predicted):.6f}   "
          f"Brier {brier(y_test_winner, predicted):.6f}   "
          f"accuracy {accuracy(y_test_winner, predicted):.6f}")
    print(f"  no skill (train+validation base rate {base_rate:.4f}) "
          f"log loss {log_loss(y_test_winner, no_skill):.6f}")
    manifest["targets"]["winner"] = {
        "family": "elo",
        # STATED, NOT IMPLIED BY ABSENCE. verify_nhl_selection's ship-rule
        # check reads this key, and when only `totals` carried it the check
        # examined nothing and passed - the `all_match`-computed-and-never-
        # acted-on pattern. Every target declares it now.
        "ships": True,
        "artifact": None,
        "artifact_note":
            "THERE IS NO FILE. Elo alone was the best candidate outright on "
            "validation - not a tie broken by the simpler-object rule - so "
            "the parameters under `elo` ARE this model. A caller computes "
            "1 / (1 + 10 ** (-(home_elo + home_advantage - away_elo) / "
            "400)).",
        "label": "HOME_WIN",
        "metric": "log_loss",
        "requires_elo": True,
        "requires_complete_features": False,
        "selection": {
            "chosen": winner["chosen"], "reason": winner["reason"],
            "candidates": {
                name: block["log_loss"]
                for name, block in winner["validation"].items()},
            "folds_agree": winner["folds_agree"],
            "fold_gap": winner["fold_gap"],
            "fold_standard_error": winner["fold_standard_error"],
        },
        "test": {
            "n": int(test.sum()),
            "model": log_loss(y_test_winner, predicted),
            "brier": brier(y_test_winner, predicted),
            "accuracy": accuracy(y_test_winner, predicted),
            "no_skill": log_loss(y_test_winner, no_skill),
            "no_skill_base_rate": base_rate,
        },
    }
    report["calibration"]["winner"] = reliability(y_test_winner, predicted)

    # ================================================================
    # PUCK LINE - logistic over the Elo columns
    # ================================================================
    puck = selection["markets"]["puck_line"]
    if not puck["ships"]:
        print("PUCK LINE does not ship; nothing to finalize")
    else:
        def fit(mask):
            model = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=5000))
            model.fit(frame.loc[mask, ELO_COLUMNS].to_numpy(dtype=float),
                      frame.loc[mask, "PUCK_LINE"].to_numpy(dtype=float))
            return model

        shipped = fit(train_plus)
        evaluated = fit(train_only)
        x_test = frame.loc[test, ELO_COLUMNS].to_numpy(dtype=float)
        p_shipped = shipped.predict_proba(x_test)[:, 1]
        p_evaluated = evaluated.predict_proba(x_test)[:, 1]
        puck_base = float(frame.loc[train_plus, "PUCK_LINE"].mean())
        puck_no_skill = np.full(len(y_test_puck), puck_base)

        print()
        print("PUCK LINE - home wins by 2+, logistic over the Elo columns")
        print(f"  {'variant':<34}{'test LL':>11}{'Brier':>10}{'acc':>9}")
        print(f"  {'-' * 34}{'-' * 11}{'-' * 10}{'-' * 9}")
        for label, p in (("SHIPPED  refit on train+validation", p_shipped),
                         ("evaluated  trained on train only", p_evaluated),
                         (f"no skill  base rate {puck_base:.4f}",
                          puck_no_skill)):
            print(f"  {label:<34}{log_loss(y_test_puck, p):>11.6f}"
                  f"{brier(y_test_puck, p):>10.6f}"
                  f"{accuracy(y_test_puck, p):>9.6f}")
        drift = abs(log_loss(y_test_puck, p_shipped)
                    - log_loss(y_test_puck, p_evaluated))
        print(f"  the shipped and evaluated weights differ by "
              f"{drift:.6f} log loss on test"
              + ("  - material, read both columns" if drift > 0.002
                 else "  - immaterial"))

        out = args.out
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(shipped, out / "puck_line.joblib")
        manifest["targets"]["puck_line"] = {
            "family": "linear",
            "ships": True,
            "artifact": "puck_line.joblib",
            "pipeline": "StandardScaler -> LogisticRegression",
            "features": ELO_COLUMNS,
            "features_note":
                "THREE COLUMNS, NOT TWENTY-FIVE. `logistic_all` on the full "
                "set was indistinguishable from this on validation and on "
                "the folds, so the rolling form and rest features earn "
                "nothing here either. No imputer: these three are never "
                "null, checked at finalize time.",
            "label": "PUCK_LINE",
            "label_definition": "HOME_MARGIN >= 2",
            "label_note":
                "2+ rather than 1+ because an NHL margin is never 0 - there "
                "are no ties in this era - so 'wins by 1+' is the winner "
                "market under another name.",
            "metric": "log_loss",
            "requires_elo": True,
            "requires_complete_features": False,
            "selection": {
                "chosen": puck["chosen"], "reason": puck["reason"],
                "candidates": {
                    name: block["log_loss"]
                    for name, block in puck["validation"].items()},
                "bootstrap_vs_base_rate":
                    puck["bootstrap_vs_reference"].get(puck["chosen"]),
                "folds_agree": puck["folds_agree"],
                "fold_gap": puck["fold_gap"],
                "fold_standard_error": puck["fold_standard_error"],
            },
            "test": {
                "n": int(test.sum()),
                "model": log_loss(y_test_puck, p_shipped),
                "brier": brier(y_test_puck, p_shipped),
                "accuracy": accuracy(y_test_puck, p_shipped),
                "evaluated_weights_model": log_loss(y_test_puck,
                                                    p_evaluated),
                "evaluated_weights_note":
                    "The training-only model the validation verdict was "
                    "actually reached about. Reported beside the shipped "
                    "refit because the retrain gate's rule is that "
                    "promotion ships the EVALUATED candidate; here the "
                    "untouched test seasons give an independent reading of "
                    "both.",
                "no_skill": log_loss(y_test_puck, puck_no_skill),
                "no_skill_base_rate": puck_base,
            },
        }
        report["calibration"]["puck_line"] = reliability(y_test_puck,
                                                         p_shipped)

    # ================================================================
    # TOTALS - declared, not served
    # ================================================================
    manifest["targets"]["totals"] = {
        "family": None, "artifact": None, "ships": False,
        "label": "TOTAL_GOALS", "metric": "rmse",
        "reason": selection["markets"]["totals"]["reason"],
        "measured": {
            "linear_mae_test": 1.847400,
            "best_constant_mae_test": 1.833916,
            "linear_rmse_test": 2.297909,
            "best_constant_rmse_test": 2.291288,
            "note": "A constant BEATS the linear model on MAE and ties it "
                    "on RMSE, so there is no skill to ship. Section 5's "
                    "figures, carried here so the declaration is checkable "
                    "without re-running anything.",
        },
        "health_shape_for_phase_3": {
            "served": False,
            "reason_code": "no_skill_over_constant",
            "detail": "Totals are not served for the NHL: a constant "
                      "predictor beats the fitted model on test.",
            "note": "The shape only. Phase 3 wires it into /health; nothing "
                    "serves the NHL yet and models_loaded is unchanged.",
        },
    }

    # ================================================================
    # frozen numbers, with each test season's competitive balance
    # ================================================================
    print()
    print("EACH TEST SEASON'S COMPETITIVE BALANCE, frozen beside the numbers")
    print("  BOTH SERVED MARKETS, not just the winner - the first version of")
    print("  this block reported the winner only, which would have frozen a")
    print("  puck-line figure with no balance beside it and so defeated the")
    print("  point of recording the balance at all.")
    manifest["test_season_balance"] = {}
    for season in F.TEST_SEASONS:
        spread = win_share_spread(frame, season)
        manifest["test_season_balance"][season] = spread
        block = frame[frame["SEASON"] == season]
        y = block["HOME_WIN"].to_numpy(dtype=float)
        p = np.asarray(F.elo_win_probability(
            block["HOME_ELO"], block["AWAY_ELO"],
            elo["home_advantage"]), dtype=float)
        rate = float(y.mean())
        spread["winner_log_loss"] = log_loss(y, p)
        spread["winner_no_skill"] = log_loss(y, np.full(len(y), rate))
        spread["winner_gain"] = (spread["winner_no_skill"]
                                 - spread["winner_log_loss"])
        print(f"  {season}  win-share sd {spread['sd']:.4f} "
              f"(min {spread['min']:.3f} max {spread['max']:.3f})")
        print(f"      winner    log loss {spread['winner_log_loss']:.6f}  "
              f"no skill {spread['winner_no_skill']:.6f}  "
              f"gain {spread['winner_gain']:+.6f}")
        if "puck_line" in manifest["targets"]:
            yp = block["PUCK_LINE"].to_numpy(dtype=float)
            pp = shipped.predict_proba(
                block[ELO_COLUMNS].to_numpy(dtype=float))[:, 1]
            prate = float(yp.mean())
            spread["puck_line_log_loss"] = log_loss(yp, pp)
            spread["puck_line_no_skill"] = log_loss(
                yp, np.full(len(yp), prate))
            spread["puck_line_gain"] = (spread["puck_line_no_skill"]
                                        - spread["puck_line_log_loss"])
            spread["puck_line_base_rate"] = prate
            print(f"      puck line log loss "
                  f"{spread['puck_line_log_loss']:.6f}  "
                  f"no skill {spread['puck_line_no_skill']:.6f}  "
                  f"gain {spread['puck_line_gain']:+.6f}  "
                  f"(its own base rate {prate:.4f})")
    manifest["test_season_balance_note"] = (
        "Elo's gain over no skill correlates with a season's win-share "
        "spread at r = 0.9261 over thirteen seasons. These figures are "
        "frozen beside the test numbers so a future movement can be "
        "attributed to the league's competitive balance or to the model, "
        "rather than being ambiguous between them. 2025-26 is the "
        "second-flattest of the thirteen.")

    # ================================================================
    # Utah - DESCRIPTION ONLY
    # ================================================================
    print()
    print("UTAH IN THE TEST SEASONS - DESCRIPTION ONLY, NOT A JUSTIFICATION")
    new_id = F.RELOCATIONS[next(iter(F.RELOCATIONS))]
    utah = test & ((frame["HOME_ID"] == new_id)
                   | (frame["AWAY_ID"] == new_id))
    others = test & ~utah
    description = {
        "franchise_id": new_id,
        "decision": "carry",
        "decision_basis": "STRUCTURAL, made without an error comparison, "
                          "because every season containing the 2024 "
                          "relocation is a TEST season.",
        "note": "These figures describe what happened. They are NOT why the "
                "carry was chosen and must never be quoted as if they were "
                "- the choice was made before any test season was scored, "
                "and scoring it to justify the choice would be exactly the "
                "test-data reasoning the decision avoided.",
        "seasons": {},
    }
    for label, mask in (("utah_fixtures", utah), ("other_fixtures", others)):
        block = frame[mask]
        p = np.asarray(F.elo_win_probability(
            block["HOME_ELO"], block["AWAY_ELO"],
            elo["home_advantage"]), dtype=float)
        y = block["HOME_WIN"].to_numpy(dtype=float)
        rate = float(y.mean())
        description[label] = {
            "n": int(mask.sum()),
            "winner_log_loss": log_loss(y, p),
            "no_skill": log_loss(y, np.full(len(y), rate)),
            "accuracy": accuracy(y, p),
        }
        print(f"  {label:<16}n={int(mask.sum()):>5}  "
              f"log loss {log_loss(y, p):.6f}  "
              f"no skill {log_loss(y, np.full(len(y), rate)):.6f}  "
              f"accuracy {accuracy(y, p):.6f}")
    for season in F.TEST_SEASONS:
        mask = (frame["SEASON"] == season) & utah
        block = frame[mask]
        y = block["HOME_WIN"].to_numpy(dtype=float)
        p = np.asarray(F.elo_win_probability(
            block["HOME_ELO"], block["AWAY_ELO"],
            elo["home_advantage"]), dtype=float)
        rate = float(y.mean())
        description["seasons"][season] = {
            "n": int(mask.sum()),
            "winner_log_loss": log_loss(y, p),
            "no_skill": log_loss(y, np.full(len(y), rate)),
            "accuracy": accuracy(y, p),
            "elo_entering_season": float(
                block.iloc[0]["HOME_ELO"]
                if block.iloc[0]["HOME_ID"] == new_id
                else block.iloc[0]["AWAY_ELO"]),
        }
        print(f"    {season}  n={int(mask.sum()):>4}  "
              f"log loss {log_loss(y, p):.6f}  "
              f"no skill {log_loss(y, np.full(len(y), rate)):.6f}  "
              f"Elo entering the season "
              f"{description['seasons'][season]['elo_entering_season']:.1f}")
    report["description_only"]["utah"] = description
    manifest["utah_description_only"] = description

    # ================================================================
    # does TEST confirm the validation ship decision?
    # ================================================================
    # DESCRIPTION, NOT A RE-DECISION. The ship rule was a paired bootstrap on
    # VALIDATION and it has already been applied; re-running it on test and
    # acting on the answer would be selecting on test. It is reported because
    # the puck line's test gain is much smaller than its validation gain, and
    # a figure that shrinks between the two is exactly what a reader needs to
    # see rather than discover later.
    print()
    print("DOES TEST CONFIRM THE VALIDATION DECISION? - description only")
    rng = np.random.default_rng(20261030)
    confirmations = {}
    pairs = [("winner", y_test_winner, no_skill, predicted)]
    if "puck_line" in manifest["targets"]:
        pairs.append(("puck_line", y_test_puck, puck_no_skill, p_shipped))
    for market, y, reference, model in pairs:
        n = len(y)
        draws = rng.integers(0, n, size=(2000, n))
        deltas = np.empty(2000, dtype=float)
        for i in range(2000):
            pick = draws[i]
            deltas[i] = (log_loss(y[pick], model[pick])
                         - log_loss(y[pick], reference[pick]))
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        observed = log_loss(y, model) - log_loss(y, reference)
        crosses = bool(lo < 0.0 < hi)
        confirmations[market] = {
            "observed": float(observed), "ci_low": float(lo),
            "ci_high": float(hi), "crosses_zero": crosses,
            "validation_observed": (
                selection["markets"][market]["bootstrap_vs_reference"]
                .get(selection["markets"][market]["chosen"], {})
                .get("observed")),
        }
        verdict = ("does NOT confirm - the interval includes zero" if crosses
                   else "CONFIRMS" if hi < 0 else "CONTRADICTS - worse")
        print(f"  {market:<11}vs no skill on test {observed:>+10.6f}  "
              f"[{lo:+.6f}, {hi:+.6f}]  {verdict}")
    report["test_confirmation"] = confirmations

    # ================================================================
    # calibration tables
    # ================================================================
    print()
    for market, table in report["calibration"].items():
        print(f"CALIBRATION ON TEST - {market}")
        print(f"  {'bin':<14}{'n':>7}{'mean predicted':>17}"
              f"{'observed':>11}{'gap':>10}")
        print("  " + "-" * 57)
        for row in table["bins"]:
            if not row["n"]:
                print(f"  {row['bin']:<14}{0:>7}{'-':>17}{'-':>11}{'-':>10}")
                continue
            print(f"  {row['bin']:<14}{row['n']:>7}"
                  f"{row['mean_predicted']:>17.4f}{row['observed']:>11.4f}"
                  f"{row['gap']:>+10.4f}")
        share = table["reliability_share"]
        print(f"  ECE {table['ece']:.6f}   reliability "
              f"{table['reliability']:.6f}   resolution "
              f"{table['resolution']:.6f}")
        print(f"  reliability as a share of reliability+resolution: "
              f"{share * 100:.1f}%" if share is not None else
              "  reliability share: undefined")
        print(f"  (the NBA's is 2.3%, the NFL's 11.5% - the NFL's is why a "
              f"winner calibrator is on its backlog)")
        # READ THE DENOMINATOR BEFORE READING THE SHARE. The share is
        # reliability / (reliability + resolution), and the NHL's RESOLUTION
        # is tiny because the model barely discriminates - so a high share
        # here is partly the signal being small rather than the calibration
        # being uniquely bad. Both numbers are printed above for that reason.
        # The absolute ECE is the figure that does not depend on how much
        # skill there is to resolve.
        print(f"  resolution is {table['resolution']:.6f}; a small "
              f"denominator inflates the share, so read ECE "
              f"({table['ece']:.6f}) as the calibration figure that does "
              f"not depend on how much signal exists")
        print()

    # ================================================================
    # provenance
    # ================================================================
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    table_path = Path(args.table) if args.table else F.DEFAULT_TABLE
    provenance = {
        "note": "NO NHL TABLE IS COMMITTED - the corpus is derived from "
                "English Wikipedia and is CC BY-SA 4.0, which is a licence "
                "boundary and not a tidiness one. Its sha256 is recorded "
                "here BECAUSE it is not committed: that is the only way to "
                "tell later whether the table these numbers were produced "
                "from is the table a rebuild produces.",
        "written_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "league": "NHL",
        "code": {
            "commit": git("rev-parse", "HEAD"),
            "tracked_dirty": [line[3:] for line in
                              (git("status", "--porcelain",
                                   "--untracked-files=no") or "").splitlines()
                              if line],
        },
        "corpus": {
            "path": str(table_path.relative_to(PROJECT))
                    if str(table_path).startswith(str(PROJECT))
                    else str(table_path),
            "sha256": sha256(table_path),
            "committed": False,
            "licence": "CC BY-SA 4.0, English Wikipedia",
            "rows": int(len(games)),
            "games": int(games["GAME_ID"].nunique()),
            "seasons": seasons,
        },
        "split": {
            "training": train_seasons,
            "validation": list(F.VALIDATION_SEASONS),
            "test": list(F.TEST_SEASONS),
            "defined_in": "ml-training/nhl/nhl_features.py "
                          "(TEST_SEASONS, VALIDATION_SEASONS)",
        },
        "regime": selection["regime"],
        "elo": manifest["elo"],
        "features": selection["features"],
        "feature_count": len(selection["features"]),
        "excluded_features": {
            "goalie_of_record": "Decided BY the game - A3's trap. A "
                                "CONFIRMED STARTING goalie would be "
                                "pre-game and is the most predictive piece "
                                "of NHL information there is, but Phase 0b "
                                "found no source under this project's data "
                                "rules.",
            "standings": "Season-to-date aggregates whose value AT THE TIME "
                         "of an early game cannot be recovered from the "
                         "article as it stands now.",
            "derived_end_disputed": "Reconciliation notes about how a row "
                                    "was built, not facts about the teams.",
            "puck_drop_time": "Phase 1 measured that no article carries a "
                              "usable one.",
            "own_score_margin_result_game_end": "The outcome, and the "
                                                "targets.",
        },
        "selection": {
            "script": "ml-training/nhl/select_nhl_models.py",
            "bootstrap": selection["bootstrap"],
            "markets": {name: {k: v for k, v in block.items()
                               if k not in ("validation", "fold_log_loss")}
                        for name, block in selection["markets"].items()},
        },
        "calibration": report["calibration"],
        "test_season_balance": manifest["test_season_balance"],
        "utah_description_only": description,
        "serving": {
            "served": False,
            "note": "NOTHING SERVES THE NHL. It is outside "
                    "daily_refresh.LEAGUES, models_loaded is unchanged, and "
                    "the frontend knows nothing about it. Phase 3 is where "
                    "that changes.",
        },
    }

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                  default=float),
                                       encoding="utf-8")
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2,
                                                     default=float),
                                         encoding="utf-8")
    print(f"written: {out / 'manifest.json'}")
    print(f"written: {out / 'provenance.json'}")
    if "puck_line" in manifest["targets"]:
        print(f"written: {out / 'puck_line.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
