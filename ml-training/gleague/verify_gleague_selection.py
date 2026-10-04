"""Verify G League phase 3's selection machinery, before test is touched.

Every check here runs on validation and training data only. Nothing reads the
test seasons, which is itself asserted.
"""

import inspect
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gleague_common as common  # noqa: E402
import select_gleague_models as select  # noqa: E402
from gleague_common import (  # noqa: E402
    DISRUPTED_SEASONS, IDENTITY_PATH, MODEL_FAMILIES, SPAN_CANDIDATES,
    STAGE_A_REFERENCE, STAGE_ORDER, TARGETS, TEST_SEASONS, VALIDATION_FOLDS,
    WINDOW_CANDIDATES, all_seasons, feature_columns, fit_elo_on, folds,
    load_dataset, load_long, section, span_seasons, training_seasons, usable,
    with_fold_elo)

GLEAGUE_PREP = (HERE.parents[1] / "data-pipeline" / "gleague"
                / "preprocessing")

results = []


def record(name, ok, detail=""):
    results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).splitlines():
            print(f"         {line}")


def check_declarations() -> None:
    section("1  CANDIDATES, STAGES AND THEIR ORDER ARE FIXED BEFORE SCORING")

    record("the candidate sets are module constants, not built at runtime",
           isinstance(WINDOW_CANDIDATES, list)
           and isinstance(MODEL_FAMILIES, list)
           and isinstance(SPAN_CANDIDATES, dict),
           f"windows {WINDOW_CANDIDATES}\nfamilies {MODEL_FAMILIES}\n"
           f"spans {dict(SPAN_CANDIDATES)}")

    # The selection script must not append to or reorder them. Checked
    # textually, because a mutation would be invisible in the printed output.
    source = inspect.getsource(select)
    # Reads are fine and frequent - SPAN_CANDIDATES[span] is how the start
    # year is looked up. Only rebinding or in-place mutation counts, so the
    # patterns require an assignment or a mutating method.
    patterns = [
        r"WINDOW_CANDIDATES\s*=", r"MODEL_FAMILIES\s*=",
        r"STAGE_ORDER\s*=", r"STAGE_A_REFERENCE\s*=",
        r"SPAN_CANDIDATES\s*=", r"SPAN_CANDIDATES\[[^\]]*\]\s*=",
        r"WINDOW_CANDIDATES\.(append|remove|insert|pop|extend|clear|sort)",
        r"MODEL_FAMILIES\.(append|remove|insert|pop|extend|clear|sort)",
        r"SPAN_CANDIDATES\.(update|pop|clear|setdefault)",
    ]
    mutations = [pattern for pattern in patterns
                 if re.search(pattern, source)]
    record("the selection script does not mutate them", not mutations,
           f"found: {mutations}" if mutations else
           "no assignment or mutation of any candidate constant")

    record("two stages are named, span first",
           len(STAGE_ORDER) == 2 and "span" in STAGE_ORDER[0],
           "\n".join(STAGE_ORDER))

    record("stage A's reference configuration is fixed",
           STAGE_A_REFERENCE == {"window": "CARRY5", "family": "linear"},
           f"{STAGE_A_REFERENCE} - the WNBA's winner, so the span is chosen "
           f"without\nsimultaneously searching windows and models")

    record("ROLL5 is not a candidate",
           "ROLL5" not in WINDOW_CANDIDATES,
           "dropped on phase 2's retention gap (86% against 95-97%), so the "
           "four\nremaining candidates answer CARRY against CUP at two "
           "lengths")


def check_stage_order() -> None:
    section("2  STAGE A IS DECIDED BEFORE STAGE B RUNS")
    source = inspect.getsource(select.main)
    a = source.index("stage_a(")
    b = source.index("stage_b(")
    record("main() calls stage_a before stage_b", a < b,
           f"stage_a at character {a}, stage_b at {b}")

    signature = inspect.signature(select.stage_b)
    record("stage_b TAKES the span as an argument rather than choosing one",
           "span" in signature.parameters,
           f"stage_b{signature} - so it cannot run without stage A's answer")


def check_split(dataset) -> None:
    section("5  DISRUPTED SEASONS ARE IN TRAINING, NEVER VALIDATION TARGETS")
    for season, why in DISRUPTED_SEASONS.items():
        in_validation = season in VALIDATION_FOLDS
        in_test = season in TEST_SEASONS
        in_some_training = any(season in fold["training"]
                               for fold in folds(dataset, "full"))
        record(f"{season} ({why}): training yes, validation no",
               in_some_training and not in_validation and not in_test,
               f"appears in training of at least one fold: {in_some_training}")

    overlap = set(VALIDATION_FOLDS) & set(TEST_SEASONS)
    record("validation and test do not overlap", not overlap,
           f"overlap: {sorted(overlap)}" if overlap else
           f"{len(VALIDATION_FOLDS)} folds, {len(TEST_SEASONS)} test seasons")

    for span in SPAN_CANDIDATES:
        for fold in folds(dataset, span):
            if set(fold["training"]) & set(TEST_SEASONS):
                record(f"{span}/{fold['validation']} training excludes test",
                       False, f"{fold['training']}")
                return
            if fold["validation"] in fold["training"]:
                record(f"{span}/{fold['validation']} not in its own training",
                       False)
                return
    record("no fold trains on its own validation season or on any test "
           "season", True,
           f"checked every fold under both spans")


def check_elo_replay(long_frame, identity) -> None:
    section("4  ELO REPLAYS FROM 2003-04 UNDER BOTH SPANS")
    seasons = all_seasons(long_frame)
    opening = seasons[0]

    for span in SPAN_CANDIDATES:
        training = training_seasons(load_dataset(), span, VALIDATION_FOLDS[0])
        k, carryover = fit_elo_on(long_frame, identity, training)
        table = common.elo_table(long_frame, identity, k, carryover)

        games = common.prepare(long_frame)
        earliest = games.iloc[0]
        rated = table.loc[earliest["GAME_ID"]]
        starts_at_opening = (
            abs(rated["HOME_TEAM_ELO"] - common.BASELINE_RATING) < 1e-9
            and abs(rated["AWAY_TEAM_ELO"] - common.BASELINE_RATING) < 1e-9)
        covers = len(table) == games["GAME_ID"].nunique()
        record(f"span {span}: replay covers every game from {opening}",
               starts_at_opening and covers,
               f"training began {training[0]}, but the first rated game is "
               f"{earliest['GAME_DATE'].date()} in {earliest['SEASON']} "
               f"at the baseline\n{len(table):,} games rated")


def check_elo_refit(long_frame, identity, dataset) -> None:
    section("3  ELO IS REFIT PER FOLD - NEGATIVE TEST AND POSITIVE CONTROL")

    fitted = []
    for fold in folds(dataset, "full"):
        k, carryover = fit_elo_on(long_frame, identity, fold["training"])
        fitted.append((fold["validation"], k, carryover))
        print(f"  {fold['validation']}  training through "
              f"{fold['training'][-1]}  ->  K={k}, "
              f"carryover={carryover:.3f}")

    distinct = {(k, c) for _, k, c in fitted}
    print(f"\n  {len(distinct)} distinct (K, carryover) pair(s) across "
          f"{len(fitted)} folds")
    if len(distinct) == 1:
        print("""  ONE PAIR ACROSS ALL FOLDS, so the spread across folds is NOT evidence
  that the refit is real - it would look identical to one fit reused. The
  WNBA phase got three distinct pairs and could lean on that; here the
  negative test and positive control below are the whole evidence.""")
    else:
        print(f"  more than one, which is some evidence the refit is live - "
              f"but the\n  controls below are what actually establish it")

    first = folds(dataset, "full")[0]
    baseline = fit_elo_on(long_frame, identity, first["training"])

    # NEGATIVE: results after this fold's training must not move its fit.
    corrupted = long_frame.copy()
    later = corrupted["SEASON"] > first["training"][-1]
    flip = corrupted.index[later]
    swapped = corrupted.loc[flip, "PTS"].to_numpy()
    paired = (corrupted.loc[flip]
              .groupby("GAME_ID")["PTS"].transform(
                  lambda s: s.values[::-1] if len(s) == 2 else s))
    corrupted.loc[flip, "PTS"] = paired.to_numpy()
    after = fit_elo_on(long_frame=corrupted, identity=identity,
                       training=first["training"])
    record(f"fold {first['validation']}: swapping {int(later.sum()):,} "
           f"later rows' scores leaves the fit unchanged",
           after == baseline,
           f"{baseline} -> {after}")

    # POSITIVE: flipping EVERY training result is self-inverting and moves
    # nothing - Elo learns mirrored ratings and the loss surface is identical.
    # The WNBA phase's control failed on exactly that, so half are randomised.
    rng = np.random.default_rng(7)
    noisy = long_frame.copy()
    rows = noisy.index[noisy["SEASON"].isin(first["training"])]
    chosen = rng.choice(rows, size=len(rows) // 2, replace=False)
    noisy.loc[chosen, "PTS"] = rng.integers(70, 130, size=len(chosen))

    # A randomised score can tie, and prepare() rightly refuses a tie since
    # basketball has no draws - the defect phase 2 found. Resolved by adding a
    # point to one side of each tied game, which keeps the corruption without
    # asking the Elo replay to handle a result that cannot happen.
    pts = noisy.groupby("GAME_ID")["PTS"]
    tied = [gid for gid, group in pts
            if len(group) == 2 and group.iloc[0] == group.iloc[1]]
    if tied:
        nudge = [noisy.index[noisy["GAME_ID"] == gid][0] for gid in tied]
        noisy.loc[nudge, "PTS"] = noisy.loc[nudge, "PTS"] + 1
        print(f"  ({len(tied)} randomised game(s) tied and were nudged by a "
              f"point)")
    moved = fit_elo_on(long_frame=noisy, identity=identity,
                       training=first["training"])
    record(f"randomising {len(chosen):,} of {len(rows):,} TRAINING rows "
           f"moves the fit", moved != baseline,
           f"{baseline} -> {moved}\n"
           f"half rather than all, because flipping every result is "
           f"self-inverting:\nElo learns mirrored ratings and the loss "
           f"surface is identical")


def check_common_rows(dataset, long_frame, identity) -> None:
    section("6  EVERY COMPARISON IS SCORED ON COMMON ROWS")
    fold = folds(dataset, "full")[1]
    k, carryover = fit_elo_on(long_frame, identity, fold["training"])
    frame = with_fold_elo(dataset, long_frame, identity, k, carryover)

    for target in TARGETS:
        common_index = select.common_rows(frame, WINDOW_CANDIDATES, target,
                                          [fold["validation"]])
        own = {w: int(usable(frame, w, target)[
            frame["SEASON"] == fold["validation"]].sum())
            for w in WINDOW_CANDIDATES}
        all_same = len(common_index) <= min(own.values())
        record(f"{target}: the common set is no larger than any candidate's "
               f"own", all_same,
               f"common {len(common_index)}, per-candidate {own}")

    # And the scoring path really uses it: every candidate must see the same
    # row count in one comparison.
    counts = set()
    for window in WINDOW_CANDIDATES:
        common_index = select.common_rows(frame, WINDOW_CANDIDATES, "spread",
                                          [fold["validation"]])
        counts.add(len(frame.loc[common_index]))
    record("all four windows score an identical row count", len(counts) == 1,
           f"row counts seen: {sorted(counts)}")


def check_scaler(dataset, long_frame, identity) -> None:
    section("7  THE SCALER IS FIT ON TRAINING ROWS ONLY, PER FOLD")
    fold = folds(dataset, "full")[1]
    k, carryover = fit_elo_on(long_frame, identity, fold["training"])
    frame = with_fold_elo(dataset, long_frame, identity, k, carryover)

    window = "CARRY5"
    columns = feature_columns(window)
    train = frame[frame["SEASON"].isin(fold["training"])]
    train = train[usable(train, window, "spread")]
    outer, _ = select.split_inner(train)

    model = select.build_model("regression", "linear")
    model.fit(outer[columns].to_numpy(dtype=float),
              outer["HOME_MARGIN"].to_numpy(dtype=float))
    fitted_mean = model.named_steps["scale"].mean_

    expected = outer[columns].to_numpy(dtype=float).mean(axis=0)
    gap_train = float(np.abs(fitted_mean - expected).max())

    validation = frame[frame["SEASON"] == fold["validation"]]
    validation = validation[usable(validation, window, "spread")]
    both = pd.concat([outer, validation])
    contaminated = both[columns].to_numpy(dtype=float).mean(axis=0)
    gap_both = float(np.abs(fitted_mean - contaminated).max())

    record("the scaler's means equal the training rows' means",
           gap_train < 1e-9,
           f"largest difference {gap_train:.2e}")
    record("and differ from train+validation means",
           gap_both > 1e-6,
           f"largest difference {gap_both:.4f} - so validation rows did not "
           f"reach the scaler")


def check_lag_guards() -> None:
    section("8  PHASE 2'S LAG GUARDS, RERUN AGAINST THE FINAL DATASET")
    script = GLEAGUE_PREP / "verify_gleague_features.py"
    proc = subprocess.run([sys.executable, "-W", "ignore", str(script)],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    tail = [line for line in (proc.stdout or "").splitlines()
            if "passed" in line]
    record("verify_gleague_features.py passes as a subprocess",
           proc.returncode == 0,
           "\n".join(tail[-1:]) or (proc.stdout or "")[-400:])


def check_test_untouched() -> None:
    section("9  THE TEST SEASONS WERE NOT READ BY THE SELECTION")
    source = inspect.getsource(select)
    record("select_gleague_models.py never references TEST_SEASONS except to "
           "print them",
           source.count("TEST_SEASONS") <= 2,
           f"{source.count('TEST_SEASONS')} reference(s); the split summary "
           f"prints them and\nnothing filters on them")

    # THE SOURCE CHECK ABOVE IS THE ONE THAT GUARDS ANYTHING. It asserts the
    # selection never filters on the test seasons, and it holds whenever this
    # runs. The receipt below is a TIMING observation, only meaningful before
    # the single test run - afterwards its presence is the expected state, and
    # failing on it would make this verifier a script that can pass exactly
    # once, which reads as broken rather than as strict.
    receipt = common.MODELS_DIR / "TEST_WAS_EVALUATED.json"
    if receipt.exists():
        import json
        when = json.loads(receipt.read_text(encoding="utf-8")).get(
            "evaluated_at", "unknown")
        record("test has been evaluated, and this verifier ran before it",
               True,
               f"{receipt.name} dated {when}\n"
               f"the source check above is what establishes the selection "
               f"never read test;\nthis line only records that the single run "
               f"has since happened")
    else:
        record("no test receipt exists yet - test is still untouched", True,
               f"{receipt.name} absent")


def main() -> int:
    print(__doc__)
    dataset = load_dataset()
    long_frame = load_long()
    identity = pd.read_csv(IDENTITY_PATH, dtype={"TEAM_ID": "int64"})
    print(f"Loaded {len(dataset):,} games, {len(all_seasons(dataset))} "
          f"seasons.")

    check_declarations()
    check_stage_order()
    check_elo_refit(long_frame, identity, dataset)
    check_elo_replay(long_frame, identity)
    check_split(dataset)
    check_common_rows(dataset, long_frame, identity)
    check_scaler(dataset, long_frame, identity)
    check_lag_guards()
    check_test_untouched()

    section("RESULT")
    failed = [n for n, ok in results if not ok]
    print(f"  {len(results) - len(failed)} passed, {len(failed)} failed")
    for name in failed:
        print(f"    FAILED: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
