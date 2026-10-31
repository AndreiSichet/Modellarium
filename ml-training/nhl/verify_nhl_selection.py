r"""Check the NHL's frozen models against the corpus they were frozen from.

THE HOUSE CONVENTION. `verify_wnba_selection.py`, `verify_gleague_selection
.py` and `verify_nfl_selection.py` all exist for the same reason: a manifest
is a claim, and a claim nobody recomputes is a number that drifts from what
it describes. Every figure checked here is recomputed from the corpus and the
shipped artifact, never read from one place and compared with itself.

EXIT CODES. 0 every check passed, 1 a check failed, 2 crashed before a
verdict.

    python ml-training/nhl/verify_nhl_selection.py --table <path>
    python ml-training/nhl/verify_nhl_selection.py --table <path> --plant a
"""
import argparse
import hashlib
import json
import sys
import traceback
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
from nhl_baselines import accuracy, brier, log_loss             # noqa: E402

MODELS_DIR = PROJECT / "ml-training" / "models_nhl"
TOLERANCE = 1e-9
ELO_COLUMNS = ["HOME_ELO", "AWAY_ELO", "ELO_DIFF"]

PLANTS = {
    "a": "rewrite the manifest's recorded winner test log loss",
    "b": "let the ship rule prefer a candidate that only TIES the reference",
    "c": "move a test season into the training list",
    "d": "swap the puck-line artifact for one fitted on shuffled labels",
    "e": "delete the winner's test interval",
    "f": "claim the winner confirmed on test while its interval crosses zero",
    "g": "put a figure in the short caveat, so it can expire",
    "h": "claim a calibrator ships while no calibrator is on disk",
}


class PlantDidNothing(Exception):
    """A plant that changes no value is a harness bug, not a passing check."""


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(table, plant=None, models_dir=MODELS_DIR):
    results = []

    def check(name, ok, detail):
        results.append((name, bool(ok), detail))

    manifest = json.loads((models_dir / "manifest.json")
                          .read_text(encoding="utf-8"))
    provenance = json.loads((models_dir / "provenance.json")
                            .read_text(encoding="utf-8"))
    artifact = joblib.load(models_dir / "puck_line.joblib")

    # ------------------------------------------------------------ plants
    if plant == "a":
        before = manifest["targets"]["winner"]["test"]["model"]
        manifest["targets"]["winner"]["test"]["model"] = before + 0.01
        if manifest["targets"]["winner"]["test"]["model"] == before:
            raise PlantDidNothing("the winner test figure did not move")
    elif plant == "b":
        # THE SIMPLER-OBJECT RULE, BROKEN. A served market keeps `ships`
        # true while its interval against the reference is made to include
        # zero - which is the state the rule forbids, and the state a future
        # edit preferring the complicated model would produce.
        target = manifest["targets"].get("puck_line")
        if target is None:
            raise PlantDidNothing("there is no puck_line target to break")
        interval = target["selection"]["bootstrap_vs_base_rate"]
        if interval["crosses_zero"]:
            raise PlantDidNothing("the interval already includes zero, so "
                                  "the manifest was already in the state "
                                  "this plant creates")
        interval["crosses_zero"] = True
        interval["ci_high"] = abs(interval["ci_high"]) or 0.001
    elif plant == "c":
        moved = manifest["test_seasons"][0]
        if moved in manifest["seasons_trained_on"]:
            raise PlantDidNothing("that season was already in the list")
        manifest["seasons_trained_on"] = (manifest["seasons_trained_on"]
                                          + [moved])
    elif plant == "e":
        if manifest["targets"]["winner"]["test"].pop(
                "interval_vs_no_skill", None) is None:
            raise PlantDidNothing("there was no interval to delete")
    elif plant == "f":
        caveat = manifest["targets"]["winner"]["caveat"]
        interval = manifest["targets"]["winner"]["test"][
            "interval_vs_no_skill"]
        if not interval["crosses_zero"]:
            raise PlantDidNothing("the interval does not cross zero, so "
                                  "claiming confirmation is not a lie")
        if caveat["confirmed_on_test"]:
            raise PlantDidNothing("it already claims confirmation")
        caveat["confirmed_on_test"] = True
    elif plant == "g":
        caveat = manifest["targets"]["winner"]["caveat"]
        if any(ch.isdigit() for ch in caveat["short"]):
            raise PlantDidNothing("the short caveat already carries a digit")
        caveat["short"] = caveat["short"] + " (0.6816 on test)"
    elif plant == "h":
        record = manifest["targets"]["winner"].get("calibration")
        if record is None:
            raise PlantDidNothing("there is no calibration record")
        if record["ships_calibrated"]:
            raise PlantDidNothing("it already claims to ship calibrated")
        record["ships_calibrated"] = True
        record["calibrator"] = "platt"
    elo = manifest["elo"]
    games = F.load_games(table)
    frame = F.build_features(
        games, elo={"k": elo["k"], "home_advantage": elo["home_advantage"],
                    "carryover": elo["carryover"]},
        carry_rolling=True, carry_rest=True)
    frame = (frame.sort_values(["GAME_DATE", "GAME_ID"])
             .reset_index(drop=True))
    frame["PUCK_LINE"] = (frame["HOME_MARGIN"] >= 2).astype(int)
    test = frame["SEASON"].isin(set(manifest["test_seasons"]))
    train_plus = frame["SEASON"].isin(set(manifest["seasons_trained_on"]))

    if plant == "d":
        rng = np.random.default_rng(7)
        shuffled = make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=5000))
        y = frame.loc[train_plus, "PUCK_LINE"].to_numpy(dtype=float).copy()
        rng.shuffle(y)
        shuffled.fit(frame.loc[train_plus, ELO_COLUMNS]
                     .to_numpy(dtype=float), y)
        x = frame.loc[test, ELO_COLUMNS].to_numpy(dtype=float)
        if np.abs(shuffled.predict_proba(x)[:, 1]
                  - artifact.predict_proba(x)[:, 1]).max() < 1e-6:
            raise PlantDidNothing("the shuffled fit matches the real one")
        artifact = shuffled

    # --------------------------------------------------------- check 1
    # THE WINNER'S RECORDED TEST FIGURES RECOMPUTE from the corpus. Elo
    # alone has no artifact, so this is the only thing that can go wrong
    # with it - and it is exactly what a manifest nobody recomputes hides.
    y_winner = frame.loc[test, "HOME_WIN"].to_numpy(dtype=float)
    predicted = np.asarray(F.elo_win_probability(
        frame.loc[test, "HOME_ELO"], frame.loc[test, "AWAY_ELO"],
        elo["home_advantage"]), dtype=float)
    recorded = manifest["targets"]["winner"]["test"]
    gaps = {
        "model": abs(log_loss(y_winner, predicted) - recorded["model"]),
        "brier": abs(brier(y_winner, predicted) - recorded["brier"]),
        "accuracy": abs(accuracy(y_winner, predicted)
                        - recorded["accuracy"]),
    }
    check("1 the winner's recorded test figures recompute from the corpus",
          max(gaps.values()) < TOLERANCE and int(test.sum()) == recorded["n"],
          f"n {int(test.sum())} vs recorded {recorded['n']}; worst gap "
          f"{max(gaps.values()):.3e} ({max(gaps, key=gaps.get)})")

    # --------------------------------------------------------- check 2
    # THE PUCK-LINE ARTIFACT'S RECORDED FIGURES RECOMPUTE from the artifact
    # on disk, not from a fresh fit - so a manifest describing a model that
    # was never saved fails here.
    puck = manifest["targets"].get("puck_line")
    if puck is None:
        check("2 the puck line's recorded test figures recompute", False,
              "the manifest has no puck_line target but an artifact exists")
    else:
        y_puck = frame.loc[test, "PUCK_LINE"].to_numpy(dtype=float)
        x_test = frame.loc[test, ELO_COLUMNS].to_numpy(dtype=float)
        p = artifact.predict_proba(x_test)[:, 1]
        recorded = puck["test"]
        gaps = {
            "model": abs(log_loss(y_puck, p) - recorded["model"]),
            "brier": abs(brier(y_puck, p) - recorded["brier"]),
            "accuracy": abs(accuracy(y_puck, p) - recorded["accuracy"]),
        }
        check("2 the puck line's recorded test figures recompute from the "
              "ARTIFACT ON DISK",
              max(gaps.values()) < TOLERANCE,
              f"worst gap {max(gaps.values()):.3e} "
              f"({max(gaps, key=gaps.get)}); recorded log loss "
              f"{recorded['model']:.6f}, recomputed "
              f"{log_loss(y_puck, p):.6f}")

    # --------------------------------------------------------- check 3
    # THE SHIP RULE, APPLIED TO THE RECORDED NUMBERS. A chosen candidate
    # whose interval against the reference includes zero must BE the
    # reference - that is the simpler-object rule, and this is where a
    # future edit that quietly prefers the complicated thing goes red.
    violations = []
    examined = 0
    for market, block in manifest["targets"].items():
        # A MISSING `ships` IS AN ERROR, NOT A SILENT FALSE. The first
        # version used `block.get("ships")`, and only `totals` carried the
        # key - so for the two SERVED markets the condition was always falsy
        # and this check examined nothing while reporting "0 violation(s)".
        # It is the `all_match`-computed-and-never-acted-on pattern, and it
        # was found by the all-green-plant guard, which is the one
        # instrument that could find it.
        if "ships" not in block:
            violations.append(f"{market} does not declare `ships`")
            continue
        selection = block.get("selection")
        if not selection:
            continue
        interval = selection.get("bootstrap_vs_base_rate")
        if interval is None:
            continue
        examined += 1
        if block["ships"] and interval.get("crosses_zero"):
            violations.append(
                f"{market} ships a candidate whose interval includes zero")
        if (block["ships"] and not interval.get("crosses_zero")
                and interval["ci_high"] >= 0):
            violations.append(f"{market} ships a candidate that is worse")
    # AND THE CHECK MUST HAVE EXAMINED SOMETHING. A vacuous pass is exactly
    # what it did once.
    if not examined:
        violations.append("no served market carried an interval to judge, "
                          "so this check examined nothing")
    # The winner's rule is the other direction: it must either BE the
    # reference, or beat it with the interval excluding zero.
    winner_selection = manifest["targets"]["winner"]["selection"]
    chosen = winner_selection["chosen"]
    if chosen != "elo_alone":
        interval = (provenance["selection"]["markets"]["winner"]
                    ["bootstrap_vs_reference"].get(chosen, {}))
        if interval.get("crosses_zero", True) or interval.get("ci_high", 1) >= 0:
            violations.append(
                "the winner ships something other than elo_alone without "
                "an interval excluding zero")
    check("3 the ship rule holds for every served market",
          not violations,
          f"{examined} market(s) with an interval examined, "
          f"{len(violations)} violation(s)"
          + (f": {violations}" if violations else
             "; winner ships " + chosen))

    # --------------------------------------------------------- check 4
    # NO TEST SEASON IS IN THE TRAINING LIST. The plainest leak, and the
    # one the undeclared-season guard exists for - asserted here against
    # the FROZEN manifest rather than against the code that wrote it.
    overlap = sorted(set(manifest["seasons_trained_on"])
                     & set(manifest["test_seasons"]))
    check("4 no test season appears in seasons_trained_on",
          not overlap,
          f"trained on {len(manifest['seasons_trained_on'])} season(s), "
          f"test {manifest['test_seasons']}, overlap {overlap}")

    # --------------------------------------------------------- check 5
    # THE WINNER HAS NO ARTIFACT AND THE MANIFEST SAYS SO. An artifact
    # appearing later without the manifest changing would mean a caller
    # reading the formula while a file sat beside it unused.
    winner = manifest["targets"]["winner"]
    on_disk = sorted(p.name for p in models_dir.glob("*.joblib"))
    check("5 the winner declares family 'elo' with no artifact, and none is "
          "on disk",
          (winner["family"] == "elo" and winner["artifact"] is None
           and "winner.joblib" not in on_disk),
          f"family {winner['family']!r}, artifact {winner['artifact']!r}, "
          f"artifacts on disk {on_disk}")

    # --------------------------------------------------------- check 6
    # TOTALS IS DECLARED UNSERVED WITH A REASON AND THE MEASUREMENT BEHIND
    # IT. A declaration with no number is an opinion.
    totals = manifest["targets"]["totals"]
    measured = totals.get("measured", {})
    constant_wins = (
        measured.get("best_constant_mae_test", 1e9)
        < measured.get("linear_mae_test", -1e9))
    check("6 totals is declared unserved, with the measurement behind it",
          (totals["ships"] is False and totals.get("reason")
           and constant_wins and totals.get("health_shape_for_phase_3")),
          f"ships {totals['ships']}, constant MAE "
          f"{measured.get('best_constant_mae_test')} vs linear "
          f"{measured.get('linear_mae_test')}, health shape "
          f"{'present' if totals.get('health_shape_for_phase_3') else 'ABSENT'}")

    # --------------------------------------------------------- check 7
    # EVERY TEST SEASON CARRIES ITS COMPETITIVE BALANCE, and the balance
    # recomputes. A frozen number with no balance beside it is the
    # ambiguity this was all recorded to prevent.
    balance = manifest.get("test_season_balance", {})
    missing = [s for s in manifest["test_seasons"] if s not in balance]
    worst = 0.0
    for season in manifest["test_seasons"]:
        if season not in balance:
            continue
        block = frame[frame["SEASON"] == season]
        wins = {}
        for row in block.itertuples():
            for team in (row.HOME_ID, row.AWAY_ID):
                wins.setdefault(team, [0, 0])
            wins[row.HOME_ID][1] += 1
            wins[row.AWAY_ID][1] += 1
            wins[row.HOME_ID if row.HOME_MARGIN > 0 else row.AWAY_ID][0] += 1
        shares = np.array([w / n for w, n in wins.values()], dtype=float)
        worst = max(worst, abs(float(shares.std()) - balance[season]["sd"]))
    check("7 every test season carries a win-share spread, and it "
          "recomputes",
          not missing and worst < TOLERANCE,
          f"{len(balance)} season(s) recorded, missing {missing}, worst sd "
          f"gap {worst:.3e}")

    # --------------------------------------------------------- check 8
    # UTAH'S BLOCK IS MARKED DESCRIPTION-ONLY. The decision was taken
    # without an error comparison because every season containing the
    # relocation is test data; a reader finding these numbers must not be
    # able to mistake them for the reason.
    utah = manifest.get("utah_description_only", {})
    words = (utah.get("note", "") + utah.get("decision_basis", "")).lower()
    check("8 Utah's figures are marked description-only, with the basis",
          (utah.get("decision") == "carry" and "not why" in words
           and "structural" in words and utah.get("utah_fixtures")),
          f"decision {utah.get('decision')!r}, n "
          f"{utah.get('utah_fixtures', {}).get('n')}, note names it as "
          f"not-the-reason: {'not why' in words}")

    # --------------------------------------------------------- check 9
    # THE CORPUS HASH IN PROVENANCE MATCHES THE FILE. The corpus is NOT
    # committed - it is CC BY-SA - so this hash is the only way to tell
    # later whether these numbers came from the table a rebuild produces.
    recorded_hash = provenance["corpus"]["sha256"]
    actual_hash = sha256(table) if table else None
    check("9 provenance's corpus sha256 matches the table in hand",
          actual_hash is None or recorded_hash == actual_hash,
          f"recorded {recorded_hash[:16]}..., actual "
          f"{(actual_hash or 'n/a')[:16]}...; committed "
          f"{provenance['corpus']['committed']}")

    # -------------------------------------------------------- check 10
    # THE PUCK-LINE ARTIFACT READS ONLY THE ELO COLUMNS. Asserted by
    # feeding it a matrix whose Elo columns are unchanged - if it secretly
    # depended on anything else the manifest's feature list would be wrong,
    # and a serving caller would compute the wrong inputs.
    if puck is not None:
        check("10 the puck-line artifact's feature list is three Elo "
              "columns and the pipeline matches",
              (puck["features"] == ELO_COLUMNS
               and artifact.named_steps.get("logisticregression") is not None
               and artifact.named_steps.get("standardscaler") is not None
               and len(artifact.named_steps) == 2),
              f"features {puck['features']}, steps "
              f"{list(artifact.named_steps)}")

    # -------------------------------------------------------- check 11
    # NOTHING SERVES THE NHL, and the manifest says so. models_loaded is
    # {7, 6, 10, 3, 3, 3} and CI asserts it at exact equality, so a change
    # here without a change there would break the boot.
    check("11 provenance states that nothing serves the NHL",
          provenance["serving"]["served"] is False
          and "models_loaded" in provenance["serving"]["note"],
          f"served {provenance['serving']['served']}, note mentions "
          f"models_loaded: {'models_loaded' in provenance['serving']['note']}")

    # -------------------------------------------------------- check 12
    # THE WINNER'S TEST INTERVAL IS RECORDED AND RECOMPUTES. The whole point
    # of freezing it is that the next reader of 0.681557 finds the interval
    # without re-running anything - and a recorded interval nobody
    # recomputes is a number that drifts from what it describes.
    interval = manifest["targets"]["winner"]["test"].get(
        "interval_vs_no_skill")
    if interval is None:
        check("12 the winner's test interval against no skill is recorded "
              "and recomputes", False,
              "the manifest carries no interval_vs_no_skill for the winner, "
              "so the frozen log loss stands alone")
    else:
        rng = np.random.default_rng(20261030)
        base = manifest["targets"]["winner"]["test"]["no_skill_base_rate"]
        reference = np.full(len(y_winner), base)
        n = len(y_winner)
        draws = rng.integers(0, n, size=(2000, n))
        deltas = np.empty(2000, dtype=float)
        for i in range(2000):
            pick = draws[i]
            deltas[i] = (log_loss(y_winner[pick], predicted[pick])
                         - log_loss(y_winner[pick], reference[pick]))
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        observed = (log_loss(y_winner, predicted)
                    - log_loss(y_winner, reference))
        gaps = [abs(observed - interval["observed"]),
                abs(float(lo) - interval["ci_low"]),
                abs(float(hi) - interval["ci_high"])]
        agrees = (interval["crosses_zero"] == bool(lo < 0.0 < hi))
        check("12 the winner's test interval against no skill is recorded "
              "and recomputes",
              max(gaps) < 1e-9 and agrees,
              f"recorded [{interval['ci_low']:+.6f}, "
              f"{interval['ci_high']:+.6f}] crosses_zero="
              f"{interval['crosses_zero']}; recomputed [{lo:+.6f}, "
              f"{hi:+.6f}] crosses_zero={bool(lo < 0.0 < hi)}; worst gap "
              f"{max(gaps):.3e}")

    # -------------------------------------------------------- check 13
    # THE CAVEAT AGREES WITH THE INTERVAL, and the short form carries no
    # figure. A caveat that contradicts the number beside it is worse than
    # no caveat, and a caveat containing a figure expires at the next
    # retrain while still looking authoritative.
    caveat = manifest["targets"]["winner"].get("caveat", {})
    problems = []
    if not caveat:
        problems.append("there is no caveat block")
    else:
        if interval is not None:
            expected = not interval["crosses_zero"]
            if caveat.get("confirmed_on_test") != expected:
                problems.append(
                    f"confirmed_on_test is {caveat.get('confirmed_on_test')} "
                    f"but the interval says {expected}")
        for field in ("short", "reader", "for_maintainers"):
            if not caveat.get(field):
                problems.append(f"{field} is missing or empty")
        if any(ch.isdigit() for ch in caveat.get("short", "")):
            problems.append("the short caveat carries a figure, which will "
                            "expire at the next retrain")
        if len(caveat.get("short", "")) > 120:
            problems.append("the short caveat is too long to be short")
    check("13 the winner's caveat agrees with its interval and the short "
          "form carries no figure",
          not problems, f"{len(problems)} problem(s)"
          + (f": {problems}" if problems else
             f"; confirmed_on_test={caveat.get('confirmed_on_test')}, "
             f"short is {len(caveat.get('short', ''))} chars"))

    # -------------------------------------------------------- check 14
    # THE CALIBRATION VERDICT IS RECORDED, AND AN UNCALIBRATED SHIP IS A
    # RESULT RATHER THAN AN OMISSION. A missing record and a recorded
    # "we tried and it did not clear the rule" are different states, and
    # only the second one tells a later reader not to try again blindly.
    record = manifest["targets"]["winner"].get("calibration")
    problems = []
    if record is None:
        problems.append("there is no calibration record at all, which is "
                        "not the same as 'uncalibrated by decision'")
    else:
        if not record.get("rule"):
            problems.append("the rule it was judged against is not recorded")
        if len(record.get("families_tried") or []) < 2:
            problems.append("fewer than two families were tried")
        if not record.get("reason"):
            problems.append("no reason is recorded")
        if not record.get("null_test"):
            problems.append("the perfectly-calibrated null test is absent, "
                            "so the record cannot say whether there was "
                            "anything to fix")
        on_disk = (models_dir / "winner_calibrator.joblib").exists()
        if record.get("ships_calibrated") != on_disk:
            problems.append(
                f"ships_calibrated is {record.get('ships_calibrated')} but "
                f"an artifact on disk is {on_disk}")
    check("14 the calibration verdict is recorded, with its rule, its "
          "families and its null test",
          not problems, f"{len(problems)} problem(s)"
          + (f": {problems}" if problems else
             f"; calibrator {record.get('calibrator')!r}, families "
             f"{record.get('families_tried')}"))

    # -------------------------------------------------------- check 15
    # THE NULL TEST SAYS SOMETHING, AND IT IS THE THING THAT JUSTIFIES NOT
    # CALIBRATING. If raw Elo were miscalibrated on the data a calibrator
    # could be fitted from, "we tried and it did not help" would need a
    # different explanation - so the record has to carry which slices were
    # inside the null.
    if record and record.get("null_test"):
        null = record["null_test"]
        training = null.get("training pooled", {})
        validation = null.get("validation", {})
        check("15 the null test shows raw Elo inside the null where a "
              "calibrator would be fitted",
              (training.get("real_miscalibration") is False
               and validation.get("real_miscalibration") is False),
              f"training pooled real miscalibration "
              f"{training.get('real_miscalibration')} "
              f"(ECE {training.get('observed_ece'):.6f} vs null median "
              f"{training.get('null_median'):.6f}), validation "
              f"{validation.get('real_miscalibration')}")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=None)
    parser.add_argument("--models", type=Path, default=MODELS_DIR)
    parser.add_argument("--plant", choices=sorted(PLANTS), default=None)
    args = parser.parse_args()

    try:
        results = run(args.table, plant=args.plant, models_dir=args.models)
    except PlantDidNothing as error:
        print(f"HARNESS BUG: {error}")
        return 2
    except Exception:                                           # noqa: BLE001
        traceback.print_exc()
        return 2

    if args.plant:
        print(f"PLANT ({args.plant}) ACTIVE - {PLANTS[args.plant]}")
        print("checks are expected to FAIL\n")
    passed = 0
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"         {detail}")
        passed += int(ok)
    print()
    print(f"{passed} of {len(results)} check(s) passed")

    # A DECLARED PLANT THAT LEAVES EVERY CHECK GREEN IS A HARNESS BUG, AND
    # THIS GUARD EXISTS BECAUSE ONE DID. Plant (b) was listed in PLANTS and
    # never implemented: it ran, printed "checks are expected to FAIL", and
    # reported 11 of 11 passed and exit 0. Nothing caught it, because the
    # per-plant PlantDidNothing guards can only fire inside a plant that
    # actually runs code. The standing rule is that a plant which changes
    # nothing is a harness bug rather than a passing check, so an all-green
    # planted run is now exit 2 - crashed before a verdict - rather than the
    # most misleading green available.
    if args.plant and passed == len(results):
        print()
        print(f"HARNESS BUG: plant ({args.plant}) left every check green. "
              f"Either it is not implemented, or it no longer reaches the "
              f"check it was written for.")
        return 2
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
