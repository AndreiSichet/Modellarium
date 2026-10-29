r"""Controls and plants for the weekly retrain gate, against the real code.

WHAT THIS GUARDS. `continuous_retrain.py` decides whether retrained models may
be promoted. Two ways it can be broken are opposites, and both have happened
in this project's history:

  * refusing everything - the old rule compared an early-stopped candidate
    with production refitted at its full frozen tree count, which on the data
    of October 2026 refused moneyline twice at about +6.2%
  * promoting everything - the "make both sides identical" fix equalised the
    rows as well, which compares a model with itself: exactly 0.00% on every
    market, bit-identical predictions, every tolerance passed

So the controls here must PROMOTE and the plants must REFUSE, and neither is
allowed to be assumed.

HOW PRODUCTION IS OBTAINED. The shipped models were fitted through
2026-04-12, which is also where the committed dataset ends, so there is no
window after their cutoff to test on. This verifier therefore builds its own
"production" at a cutoff inside history, using the SAME function the gate
gives the candidate (`train_candidate`), writes it to a scratch directory,
and compares against that. Nothing in `ml-training/models/` is read or
written.

    python ml-training/verify_retrain_gate.py
    python ml-training/verify_retrain_gate.py --cutoff-games 1200
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import continuous_retrain as CR                                  # noqa: E402
from common import FEATURE_COLUMNS                               # noqa: E402
from model_evaluation import TASKS                                # noqa: E402
from train_baseline import load_dataset                          # noqa: E402

RESULTS = []


def record(name, passed, detail=""):
    RESULTS.append((name, passed, detail))
    print("  %-4s %-62s %s" % ("PASS" if passed else "FAIL", name, detail))


def verdict_of(results):
    blocking = [r["key"] for r in results if not r["passed"]]
    return ("PROMOTE" if not blocking else "REFUSED"), blocking


def summarise(results):
    return "  ".join(f"{r['key'][:9]}{r['change_pct']:+.2f}%" for r in results)


def build_production(dataset, cutoff, directory: Path) -> None:
    """A stand-in for the shipped models, at `cutoff`, by the gate's own rule."""
    rows = dataset[dataset["GAME_DATE"] <= cutoff].sort_values(
        ["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    directory.mkdir(parents=True, exist_ok=True)
    for key, _label, target, classification in TASKS:
        model, _trees = CR.train_candidate(target, classification, rows)
        model.save_model(str(directory / f"{key}.json"))
    CR.write_trained_through(directory, rows)


def gate_on(dataset, cutoff, production_dir, test_games=500, mutate=None,
            candidate_cutoff=None):
    """Run the real gate. `mutate` corrupts the CANDIDATE's rows only.

    EVERY CASE USES THE SAME TEST WINDOW - the `test_games` games immediately
    after `cutoff` - so the cases are comparable with each other. The first
    version let the window depend on whether `candidate_cutoff` was given,
    which made the control and the stale-data case score on a different 500
    games from the plants, and nothing said so.
    """
    visible = dataset[
        dataset["GAME_DATE"] <= _nth_game_date(dataset, cutoff, test_games)]
    rows, test = CR.cutoff_split(visible, cutoff, test_games)
    if candidate_cutoff is not None:
        rows = rows[rows["GAME_DATE"] <= candidate_cutoff]
    if mutate is not None:
        rows = mutate(rows.copy())
    # Printed AFTER the truncation, because cutoff_split reports the window
    # it carved and not what the candidate is finally given.
    print(f"  candidate finally {len(rows):>6,} games   "
          f"{rows['GAME_DATE'].min().date()} to "
          f"{rows['GAME_DATE'].max().date()}")
    return CR.run_gate(rows, test, production_dir, CR.DEFAULT_TOLERANCE_PCT)


def _nth_game_date(dataset, cutoff, n):
    """The date of the nth game after `cutoff` - the window's far edge."""
    after = dataset[dataset["GAME_DATE"] > cutoff].sort_values(
        ["GAME_DATE", "GAME_ID"])
    if len(after) < n:
        raise RuntimeError(f"only {len(after)} games after {cutoff}")
    return after.iloc[n - 1]["GAME_DATE"]


def strongest_features(dataset, cutoff, n=5):
    """The n highest-gain features of the spread model. Derived, not listed."""
    rows = dataset[dataset["GAME_DATE"] <= cutoff].sort_values(
        ["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    model, _ = CR.train_candidate("HOME_MARGIN", False, rows)
    gain = model.get_booster().get_score(importance_type="gain")
    return sorted(gain, key=gain.get, reverse=True)[:n]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff-games", type=int, default=1500,
                        help="place production's cutoff this many games "
                             "before the end of the data")
    args = parser.parse_args()

    dataset = load_dataset().sort_values(
        ["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    cutoff = dataset.iloc[-args.cutoff_games]["GAME_DATE"]
    print(f"dataset {len(dataset):,} games, "
          f"{dataset['GAME_DATE'].min().date()} to "
          f"{dataset['GAME_DATE'].max().date()}")
    print(f"production cutoff {cutoff.date()} "
          f"({args.cutoff_games} games from the end)")
    print()

    scratch = Path(tempfile.mkdtemp(prefix="gate-verify-"))
    try:
        production = scratch / "production"
        build_production(dataset, cutoff, production)
        print(f"stand-in production written to {production}")
        print()

        # ---- CONTROL: a candidate trained on production's own rows.
        print("CONTROL: the candidate sees exactly production's data")
        control = gate_on(dataset, cutoff, production, candidate_cutoff=cutoff)
        v, blocking = verdict_of(control)
        worst = max(abs(r["change_pct"]) for r in control)
        record("control: same data -> PROMOTE", v == "PROMOTE",
               f"{v}, worst |change| {worst:.6f}%")
        record("control: every change is ~0",
               worst < 0.01, f"worst |change| {worst:.9f}%")
        record("control: the naive guard passes on every market",
               all(r["naive"]["passed"] for r in control),
               f"{sum(1 for r in control if r['naive']['passed'])} of {len(control)}")
        print()

        # ---- PLANT: shuffled targets. Must refuse on EVERY market, which is
        # what the naive guard was added for - the tolerance alone let
        # reb_total through at +0.41%.
        print("PLANT: the candidate's targets are shuffled")
        rng = np.random.default_rng(11)

        def shuffle_targets(rows):
            order = rng.permutation(len(rows))
            for _k, _l, target, _c in TASKS:
                rows[target] = rows[target].to_numpy()[order]
            return rows

        shuffled = gate_on(dataset, cutoff, production, mutate=shuffle_targets)
        v, blocking = verdict_of(shuffled)
        record("PLANT: shuffled targets -> REFUSED", v == "REFUSED",
               f"{len(blocking)} of {len(shuffled)} markets blocked")
        record("PLANT: shuffled targets refused on EVERY market",
               len(blocking) == len(shuffled),
               f"blocked: {', '.join(blocking)}")
        print(f"       {summarise(shuffled)}")
        print()

        # ---- PLANT: every feature blanked, so the candidate can only predict
        # the mean. This is the plant the naive guard exists for, and unlike
        # shuffling it is DETERMINISTIC: a mean-predictor is exactly the
        # "no signal at all" model, and in a market where production barely
        # beats its naive baseline a mean-predictor sits inside the 2%
        # tolerance. Shuffling the targets does not reliably produce it -
        # early stopping on shuffled data is itself random, and on one run it
        # chose enough trees to make reb_total +1124%, which the tolerance
        # catches on its own. A check that depends on that would pass for the
        # wrong reason.
        print("PLANT: every feature blanked - the candidate can only predict "
              "the mean")

        def blank_all(rows):
            for column in FEATURE_COLUMNS:
                rows[column] = np.nan
            return rows

        blind = gate_on(dataset, cutoff, production, mutate=blank_all)
        v, blocking = verdict_of(blind)

        # ALL SEVEN SHIP TOGETHER, so what has to hold is that the SET is
        # refused. A market the guard cannot reach does not let a bad set
        # through; it only means that market was not the one that stopped it.
        record("PLANT: a mean-predictor -> the SET is REFUSED",
               v == "REFUSED", f"{v}, {len(blocking)} of {len(blind)} blocked")

        by_tolerance = [r["key"] for r in blind if not r["tolerance_passed"]]
        by_guard_only = [r["key"] for r in blind
                         if r["tolerance_passed"] and not r["naive"]["passed"]]
        inert = [r["key"] for r in blind if not r["naive"]["informative"]]
        missed = [r["key"] for r in blind if r["passed"]]
        print(f"       caught by the tolerance: {by_tolerance or 'none'}")
        print(f"       caught by the guard alone: {by_guard_only or 'none'}")
        print(f"       guard inert (production has no measurable edge): "
              f"{inert or 'none'}")
        print(f"       market-level misses: {missed or 'none'}")

        # Every market that let a mean-predictor through must be one where
        # production itself has no measurable edge. If a market with a real
        # edge missed it, the guard is broken rather than merely limited.
        record("every market-level miss is a market the guard cannot reach",
               set(missed) <= set(inert),
               f"missed {missed or 'none'}; inert {inert or 'none'}")

        # NON-VACUITY: the guard must be a live check somewhere. If it were
        # inert on all seven it would be decoration.
        record("the guard is informative on the markets with real skill",
               not ({"moneyline", "spread", "totals"} & set(inert)),
               f"inert among moneyline/spread/totals: "
               f"{sorted({'moneyline', 'spread', 'totals'} & set(inert)) or 'none'}")
        print(f"       {summarise(blind)}")
        print()

        # ---- PLANT: the five strongest features blanked in training.
        print("PLANT: the five highest-gain features blanked in the "
              "candidate's rows")
        top = strongest_features(dataset, cutoff, 5)
        print(f"       {top}")

        def blank_top(rows):
            for column in top:
                rows[column] = np.nan
            return rows

        blanked = gate_on(dataset, cutoff, production, mutate=blank_top)
        v, blocking = verdict_of(blanked)
        record("PLANT: strongest features blanked -> REFUSED",
               v == "REFUSED", f"blocked: {', '.join(blocking) or 'none'}")
        record("PLANT: the markets those features drive are among them",
               {"moneyline", "spread"} <= set(blocking),
               f"moneyline and spread blocked: "
               f"{'moneyline' in blocking}/{'spread' in blocking}")
        print(f"       {summarise(blanked)}")
        print()

        # ---- The documented blind spot, asserted as a blind spot.
        print("ACCEPTED LIMITATION: a candidate trained a full season earlier")
        earlier = cutoff - pd.Timedelta(days=365)
        stale = gate_on(dataset, cutoff, production, candidate_cutoff=earlier)
        v, blocking = verdict_of(stale)
        worst = max(r["change_pct"] for r in stale)
        record("stale-data candidate is NOT detected (documented, accepted)",
               v == "PROMOTE",
               f"{v}, worst change {worst:+.2f}% - the docstring says so")
        print(f"       {summarise(stale)}")
        print()

        # ---- PLANT: a promotion whose artifact no longer matches.
        print("PLANT: an artifact rewritten between evaluation and promotion")
        out = scratch / "candidate"
        out.mkdir(parents=True, exist_ok=True)
        evaluated = {}
        for r in control:
            path = out / f"{r['key']}.json"
            r["model"].save_model(str(path))
            evaluated[r["key"]] = CR.sha256_of(path)
        record("control: freshly written artifacts match their digests",
               CR.promotion_hash_mismatch(out, evaluated) == [],
               "no mismatch")

        victim = out / "spread.json"
        victim.write_text(victim.read_text(encoding="utf-8") + " ",
                          encoding="utf-8")
        caught = CR.promotion_hash_mismatch(out, evaluated)
        record("PLANT: a rewritten artifact is caught", caught == ["spread"],
               f"caught: {caught}")

        (out / "totals.json").unlink()
        caught = CR.promotion_hash_mismatch(out, evaluated)
        record("PLANT: a missing artifact is caught",
               caught == ["spread", "totals"], f"caught: {caught}")
        print()

        # ---- PLANT: a test window that reaches back past the cutoff.
        print("PLANT: a test window overlapping production's training data")
        try:
            CR.cutoff_split(dataset, dataset.iloc[-10]["GAME_DATE"], 500)
            record("PLANT: an overlapping window is refused", False,
                   "cutoff_split returned instead of raising")
        except RuntimeError as error:
            record("PLANT: an overlapping window is refused",
                   "already seen part of it" in str(error),
                   str(error)[:72])

        # ---- The threshold is derived in one place.
        record("games_required() = test window + new training games",
               CR.games_required(500) == 500 + CR.MIN_NEW_TRAINING_GAMES,
               f"{CR.games_required(500)} = 500 + {CR.MIN_NEW_TRAINING_GAMES}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    passed = sum(1 for _n, ok, _d in RESULTS if ok)
    print()
    print(f"  {passed} of {len(RESULTS)} checks passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:                                   # noqa: BLE001
        print(f"\nCRASHED before a verdict: {type(error).__name__}: {error}")
        raise SystemExit(2)
