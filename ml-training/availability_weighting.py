"""Does weighting absences by player QUALITY beat weighting them by MINUTES?

Study only. Trains nothing that ships, writes no artifacts, and mutates no
pipeline file - the variant columns are built in memory and swapped into a copy
of model_dataset.csv.

WEIGHTED_ABSENT_MIN weights an absence by the player's rolling minutes, which
is a proxy for a coach's trust rather than for contribution: a superstar and a
rotation player missing the same 32 minutes currently count identically. The
model has no representation of player quality anywhere else either - team
rolling features aggregate outcomes, and Elo is team-level by construction.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

from common import FEATURE_COLUMNS, split_three_way
from train_baseline import REGRESSION_TARGETS, ROLLING_FEATURE_COLUMNS, load_dataset, section
from train_moneyline_xgb import PARAMS as CLASSIFIER_PARAMS, TARGET as WIN_TARGET
from train_regression_xgb import PARAMS as REGRESSION_PARAMS

PROCESSED = Path(__file__).resolve().parents[1] / "data-pipeline" / "data" / "processed"
PLAYERS_PATH = PROCESSED / "player_boxscores_with_rolling.csv"
GAMES_FINAL_PATH = PROCESSED / "games_final.csv"

WEIGHT_COLUMN = "WEIGHTED_ABSENT_MIN"
SIDED_WEIGHT_COLUMNS = ["HOME_WEIGHTED_ABSENT_MIN", "AWAY_WEIGHTED_ABSENT_MIN"]
GROUP_KEYS = ["GAME_ID", "TEAM_ID"]
TEAM_MARGIN = "TEAM_MARGIN"

RECORDED = {
    "Spread": 10.7368,
    "Totals": 15.2322,
    "REB margin": 7.5095,
    "REB total": 7.3382,
    "AST margin": 5.3928,
    "AST total": 5.8356,
}
RECORDED_ACCURACY = 0.6698
RECORDED_LOG_LOSS = 0.5979
REPRODUCTION_TOLERANCE = 5e-4

# A 3-5% spread gain is what has justified adoption before. Above ~6% on this
# feature, suspect leakage before believing it.
ADOPTION_BAR_PCT = 3.0
SUSPICION_PCT = 6.0

# On/off needs games on BOTH sides of the split. Below this it is noise, and
# the value falls back to the per-minute weighting.
MIN_GAMES_PLAYED = 10
MIN_GAMES_ABSENT = 5

# ROLL10_PRA / ROLL10_MIN blows up as the denominator approaches zero.
MIN_MINUTES_FOR_RATE = 1.0


def load_players() -> pd.DataFrame:
    """The player-game table, with GAME_ID kept padded as the pipeline does."""
    players = pd.read_csv(
        PLAYERS_PATH,
        dtype={"GAME_ID": str, "TEAM_ID": str, "PLAYER_ID": str},
        low_memory=False,
    )
    for column in ("MIN_NUMERIC", "ROLL10_MIN", "ROLL10_PRA"):
        players[column] = pd.to_numeric(players[column], errors="coerce")
    players["GAME_DATE"] = pd.to_datetime(players["GAME_DATE"])
    players["IS_ABSENT"] = players["MIN_NUMERIC"].isna()
    return players


def attach_team_margin(players: pd.DataFrame) -> pd.DataFrame:
    """Each player row carries its team's point differential for that game."""
    games = pd.read_csv(
        GAMES_FINAL_PATH, usecols=["GAME_ID", "TEAM_ID", "PLUS_MINUS"],
        dtype={"TEAM_ID": str},
    ).rename(columns={"PLUS_MINUS": TEAM_MARGIN})
    games["GAME_ID"] = games["GAME_ID"].astype(str).str.zfill(10)

    # The player table has its own PLUS_MINUS - the PLAYER's on-court
    # differential. On/off needs the TEAM's margin for the whole game, so the
    # incoming column is renamed rather than left to collide.
    merged = players.merge(games, on=GROUP_KEYS, how="left", validate="many_to_one")
    if merged[TEAM_MARGIN].isna().any():
        raise SystemExit("some player rows have no team margin to build on/off from")
    return merged


def on_off_differential(players: pd.DataFrame) -> pd.Series:
    """Team margin with the player minus team margin without, PRIOR GAMES ONLY.

    The shift(1) is the leakage guard and is the whole correctness story here:
    an on/off computed over a player's full season would contain the outcome of
    the very game being predicted. Same discipline as every rolling feature in
    the pipeline.

    Scoped per (PLAYER_ID, TEAM_ID) because on/off is team-specific - a value
    earned in one jersey says nothing about the next.
    """
    df = players.sort_values(["PLAYER_ID", "TEAM_ID", "GAME_DATE", "GAME_ID"]).copy()
    keys = ["PLAYER_ID", "TEAM_ID"]

    played = ~df["IS_ABSENT"]
    df["_margin_on"] = df[TEAM_MARGIN].where(played, 0.0)
    df["_count_on"] = played.astype(float)
    df["_margin_off"] = df[TEAM_MARGIN].where(df["IS_ABSENT"], 0.0)
    df["_count_off"] = df["IS_ABSENT"].astype(float)

    grouped = df.groupby(keys, sort=False)
    prior = {}
    for column in ("_margin_on", "_count_on", "_margin_off", "_count_off"):
        prior[column] = grouped[column].cumsum() - df[column]

    usable = (prior["_count_on"] >= MIN_GAMES_PLAYED) & (prior["_count_off"] >= MIN_GAMES_ABSENT)
    with np.errstate(invalid="ignore", divide="ignore"):
        value = (prior["_margin_on"] / prior["_count_on"].replace(0, np.nan)
                 - prior["_margin_off"] / prior["_count_off"].replace(0, np.nan))

    return value.where(usable).reindex(players.index)


def build_weights(players: pd.DataFrame) -> dict:
    """One absence-weight column per variant, all on the same player rows."""
    rate = players["ROLL10_PRA"] / players["ROLL10_MIN"].where(
        players["ROLL10_MIN"] >= MIN_MINUTES_FOR_RATE)

    with_margin = attach_team_margin(players)
    on_off = on_off_differential(with_margin)
    # Spec: below the minimum sample, fall back to the B weighting rather than
    # emitting a noisy number.
    combined = on_off.where(on_off.notna(), rate)

    return {
        "baseline (ROLL10_MIN)": players["ROLL10_MIN"],
        "A: rolling PRA": players["ROLL10_PRA"],
        "B: rolling PRA per minute": rate,
        "C: on/off, B fallback": combined,
    }, on_off


def aggregate(players: pd.DataFrame, weight: pd.Series) -> pd.DataFrame:
    """Sum the weight over a team-game's absent players, unknown role as 0."""
    absent_weight = weight.where(players["IS_ABSENT"]).fillna(0.0)
    frame = players[GROUP_KEYS].assign(ABSENT_WEIGHT=absent_weight)
    return frame.groupby(GROUP_KEYS, as_index=False).agg(
        WEIGHTED=("ABSENT_WEIGHT", "sum"))


def side_map(players: pd.DataFrame) -> pd.DataFrame:
    """Which TEAM_ID was home in each game, so team-games become HOME_/AWAY_."""
    games = pd.read_csv(
        GAMES_FINAL_PATH, usecols=["GAME_ID", "TEAM_ID", "IS_HOME"],
        dtype={"TEAM_ID": str},
    )
    games["GAME_ID"] = games["GAME_ID"].astype(str).str.zfill(10)
    return games


def to_sided(availability: pd.DataFrame, sides: pd.DataFrame) -> pd.DataFrame:
    """One row per GAME_ID carrying the home and away weighted totals."""
    merged = availability.merge(sides, on=GROUP_KEYS, how="left", validate="one_to_one")
    if merged["IS_HOME"].isna().any():
        raise SystemExit("a team-game could not be assigned a side")

    home = merged[merged["IS_HOME"]].set_index("GAME_ID")["WEIGHTED"]
    away = merged[~merged["IS_HOME"]].set_index("GAME_ID")["WEIGHTED"]
    out = pd.DataFrame({
        "HOME_WEIGHTED_ABSENT_MIN": home,
        "AWAY_WEIGHTED_ABSENT_MIN": away,
    })
    out.index = out.index.astype(int)
    return out.sort_index()


def verify_reconstruction(rebuilt: pd.DataFrame, dataset: pd.DataFrame) -> None:
    """The baseline weighting rebuilt here must equal the shipped dataset's.

    This is the gate for the whole study. If the reconstruction does not
    reproduce the column the 38-feature models were actually trained on, then
    every variant below is being injected somewhere other than where the real
    feature lives, and no comparison means anything.
    """
    section("RECONSTRUCTION GATE")
    joined = dataset[["GAME_ID"]].merge(
        rebuilt, left_on="GAME_ID", right_index=True, how="left",
        suffixes=("", "_rebuilt"))
    if joined[SIDED_WEIGHT_COLUMNS].isna().any().any():
        raise SystemExit("the rebuilt availability does not cover every game")

    worst = 0.0
    for column in SIDED_WEIGHT_COLUMNS:
        diff = float(np.abs(joined[column].to_numpy()
                            - dataset[column].to_numpy()).max())
        worst = max(worst, diff)
        print(f"  {column:<28} largest absolute difference {diff:.3e}")

    print(f"\n  Rebuilt from {PLAYERS_PATH.name} and compared against the "
          "committed\n  model_dataset.csv the shipped models were trained on.")
    if worst > 1e-9:
        raise SystemExit(
            "The rebuilt baseline weighting does not match the dataset. Every "
            "variant below would be injected in the wrong place, so this stops."
        )
    print("  PASS - the variant columns slot in exactly where the real one does.")


def report_zero_fractions(players: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """How often each weighting silently scores a real absence as 0.

    WEIGHTED_ABSENT_MIN is known to be deflated early in each season: a player
    is NaN until 11 appearances and is then weighted 0. Any quality weighting
    inherits the same warm-up, so a variant that looks better partly because it
    has fewer zeros would not be better for the reason it appears to be.
    """
    section("ZERO-WEIGHTED FRACTION PER VARIANT (the shared warm-up)")
    absent = players["IS_ABSENT"]
    total = int(absent.sum())
    print(f"Absent player-rows across the corpus: {total:,}\n")
    print(f"  {'VARIANT':<28}{'WEIGHTED 0':>12}{'FRACTION':>11}{'NEGATIVE':>11}")
    print("-" * 64)

    rows = []
    for label, weight in weights.items():
        missing = int((absent & weight.isna()).sum())
        negative = int((absent & (weight < 0)).sum())
        rows.append({"variant": label, "zeros": missing,
                     "fraction": missing / total, "negative": negative})
        print(f"  {label:<28}{missing:>12,}{missing / total:>10.1%}{negative:>11,}")

    fractions = {r["fraction"] for r in rows}
    if max(fractions) - min(fractions) < 0.005:
        print("\n  All variants share the SAME zero fraction to within 0.5pp, which")
        print("  is expected: rolling PRA has the same 11-appearance warm-up as")
        print("  rolling minutes. So the comparison is clean and this is a")
        print("  recorded caveat rather than a confound - no variant gains an")
        print("  advantage from having fewer unknown players.")
    else:
        print("\n  The fractions DIFFER materially. A matched subset - games where")
        print("  every variant has a usable value - is needed as a secondary")
        print("  check before any difference is attributed to the weighting.")
    return pd.DataFrame(rows)


def report_on_off_coverage(players: pd.DataFrame, on_off: pd.Series) -> None:
    section("VARIANT C COVERAGE - HOW MUCH IS ACTUALLY ON/OFF?")
    absent = players["IS_ABSENT"]
    usable = on_off.notna()
    absent_usable = int((absent & usable).sum())
    total_absent = int(absent.sum())

    print(f"Thresholds: at least {MIN_GAMES_PLAYED} prior games played AND "
          f"{MIN_GAMES_ABSENT} prior games missed,\nboth for the same "
          "(player, team).\n")
    print(f"  player-rows with a usable on/off      {int(usable.sum()):>10,} "
          f"of {len(players):,} ({usable.mean():.1%})")
    print(f"  ABSENT rows with a usable on/off      {absent_usable:>10,} "
          f"of {total_absent:,} ({absent_usable / total_absent:.1%})")
    print(f"  ABSENT rows falling back to B         "
          f"{total_absent - absent_usable:>10,} "
          f"({1 - absent_usable / total_absent:.1%})")

    values = on_off[usable]
    if len(values):
        print(f"\n  on/off distribution: min {values.min():+.2f}  "
              f"median {values.median():+.2f}  max {values.max():+.2f}  "
              f"negative {(values < 0).mean():.1%}")
    if absent_usable / total_absent < 0.5:
        print("\n  C IS MOSTLY B. The on/off half is the minority of the column,")
        print("  so a C result close to B's is the expected outcome rather than")
        print("  a finding about on/off, and should be read that way.")
    print("\n  NOTE ON SCALE: where on/off applies the value is a point")
    print("  differential (roughly -20 to +20); where it falls back it is a PRA")
    print("  rate (roughly 0 to 1.2). One column holding two scales is odd, and")
    print("  it is what the spec asks for. Trees split on order rather than")
    print("  magnitude so it is workable, but it is a real reason to prefer B")
    print("  if the two tie.")


def check_no_leakage(players: pd.DataFrame) -> None:
    """The on/off value for a game must use only games strictly before it."""
    section("LEAKAGE GUARD ON VARIANT C - NEGATIVE-TESTED")

    with_margin = attach_team_margin(players)
    honest = on_off_differential(with_margin)

    # A player-team with a long history, so the guard has something to bite on.
    counts = with_margin.groupby(["PLAYER_ID", "TEAM_ID"]).size()
    player_id, team_id = counts.idxmax()
    subset = with_margin[(with_margin["PLAYER_ID"] == player_id)
                         & (with_margin["TEAM_ID"] == team_id)]
    subset = subset.sort_values(["GAME_DATE", "GAME_ID"])
    print(f"Probe: player {player_id} at team {team_id}, "
          f"{len(subset)} games on record.\n")

    target_row = subset.index[len(subset) // 2]
    target_date = with_margin.loc[target_row, "GAME_DATE"]

    # Rebuild that one value by hand from strictly-prior games only.
    prior = subset[subset["GAME_DATE"] < target_date]
    played = prior[~prior["IS_ABSENT"]]
    missed = prior[prior["IS_ABSENT"]]
    if len(played) >= MIN_GAMES_PLAYED and len(missed) >= MIN_GAMES_ABSENT:
        by_hand = played[TEAM_MARGIN].mean() - missed[TEAM_MARGIN].mean()
    else:
        by_hand = np.nan
    computed = honest.loc[target_row]
    agree = (np.isnan(by_hand) and np.isnan(computed)) or abs(by_hand - computed) < 1e-9
    print(f"  hand-computed from prior games only   {by_hand}")
    print(f"  what on_off_differential returned     {computed}")
    print(f"  agree: {'yes' if agree else 'NO'}")
    if not agree:
        raise SystemExit("the on/off value does not match a hand recomputation")

    # Negative test: corrupt the future. If the guard works, the value for this
    # game cannot move, because it never reads at-or-after its own date.
    poisoned = with_margin.copy()
    future = ((poisoned["PLAYER_ID"] == player_id)
              & (poisoned["TEAM_ID"] == team_id)
              & (poisoned["GAME_DATE"] >= target_date))
    poisoned.loc[future, TEAM_MARGIN] = 999.0
    after = on_off_differential(poisoned).loc[target_row]

    moved = not ((np.isnan(after) and np.isnan(computed))
                 or abs(after - computed) < 1e-9)
    print(f"\n  NEGATIVE TEST: {int(future.sum())} games at or after "
          f"{target_date.date()} set to a team margin of 999")
    print(f"  value for the target game afterwards  {after}")
    print(f"  moved: {'YES - LEAK' if moved else 'no'}")
    if moved:
        raise SystemExit(
            "Corrupting future games changed a past game's on/off value. The "
            "feature reads the outcome it is predicting, and any gain is fraud."
        )

    # Positive control: the guard must still be reading the PAST, or it would
    # pass the test above by computing nothing at all.
    poisoned_past = with_margin.copy()
    past = ((poisoned_past["PLAYER_ID"] == player_id)
            & (poisoned_past["TEAM_ID"] == team_id)
            & (poisoned_past["GAME_DATE"] < target_date))
    poisoned_past.loc[past, TEAM_MARGIN] = 999.0
    after_past = on_off_differential(poisoned_past).loc[target_row]
    responds = not ((np.isnan(after_past) and np.isnan(computed))
                    or abs(after_past - computed) < 1e-9)
    print(f"\n  POSITIVE CONTROL: the same corruption applied to PRIOR games")
    print(f"  value afterwards                      {after_past}")
    print(f"  moved: {'yes - the guard reads the past, as it must' if responds else 'NO'}")
    if not responds:
        raise SystemExit(
            "Corrupting prior games did NOT change the value, so the negative "
            "test above proved nothing - it would pass on a feature that reads "
            "no data at all."
        )


def train_and_score(dataset: pd.DataFrame) -> dict:
    """Every target, held out, on one feature set. Never loads a shipped model."""
    train, validation, test = split_three_way(dataset)
    comparable = test.dropna(subset=ROLLING_FEATURE_COLUMNS)

    results = {}

    classifier = XGBClassifier(**CLASSIFIER_PARAMS)
    classifier.fit(train[FEATURE_COLUMNS], train[WIN_TARGET],
                   eval_set=[(validation[FEATURE_COLUMNS], validation[WIN_TARGET])],
                   verbose=False)
    proba = classifier.predict_proba(comparable[FEATURE_COLUMNS])[:, 1].astype(np.float64)
    results["Moneyline"] = {
        "accuracy": float(accuracy_score(comparable[WIN_TARGET], (proba > 0.5).astype(int))),
        "log_loss": float(log_loss(comparable[WIN_TARGET], proba)),
        "trees": int(classifier.best_iteration),
    }

    for target, label, _stat, _combine in REGRESSION_TARGETS:
        model = XGBRegressor(**REGRESSION_PARAMS)
        model.fit(train[FEATURE_COLUMNS], train[target],
                  eval_set=[(validation[FEATURE_COLUMNS], validation[target])],
                  verbose=False)
        predicted = model.predict(comparable[FEATURE_COLUMNS])
        results[label] = {
            "mae": float(mean_absolute_error(comparable[target], predicted)),
            "trees": int(model.best_iteration),
        }
        if label == "Spread":
            results[label]["errors"] = np.abs(
                comparable[target].to_numpy() - predicted.astype(np.float64))
            results[label]["game_ids"] = comparable["GAME_ID"].to_numpy()
    return results


def verify_baseline(results: dict) -> None:
    section("BASELINE GATE - THE 38-FEATURE NUMBERS MUST REPRODUCE")
    print("Held out on train only, early-stopped on validation. Nothing here")
    print("loads ml-training/models/: finalize_models.py trains those with no")
    print("holdout, so they have seen the test window and read far too well.\n")

    failures = []
    got_acc = results["Moneyline"]["accuracy"]
    got_ll = results["Moneyline"]["log_loss"]
    for name, got, want in [("Moneyline accuracy", got_acc, RECORDED_ACCURACY),
                            ("Moneyline log loss", got_ll, RECORDED_LOG_LOSS)]:
        ok = abs(got - want) < REPRODUCTION_TOLERANCE
        failures += [] if ok else [name]
        print(f"  {name:<22} got {got:.4f}   recorded {want:.4f}   "
              f"{'MATCH' if ok else 'NO MATCH'}")

    for label, want in RECORDED.items():
        got = results[label]["mae"]
        ok = abs(got - want) < REPRODUCTION_TOLERANCE
        failures += [] if ok else [label]
        print(f"  {label + ' MAE':<22} got {got:.4f}   recorded {want:.4f}   "
              f"{'MATCH' if ok else 'NO MATCH'}")

    if failures:
        raise SystemExit(
            f"{failures} did not reproduce. Every comparison below would be "
            "against a different baseline, so this stops rather than reporting."
        )
    print("\n  PASS - all seven reproduce.")


def print_results(all_results: dict, labels: list) -> None:
    section("SPREAD - THE HEADLINE")
    baseline = all_results[labels[0]]["Spread"]["mae"]
    print(f"{'VARIANT':<28}{'MAE':>10}{'CHANGE':>10}{'TREES':>8}   VERDICT")
    print("-" * 72)
    for label in labels:
        mae = all_results[label]["Spread"]["mae"]
        change = (mae - baseline) / baseline * 100
        trees = all_results[label]["Spread"]["trees"]
        if label == labels[0]:
            verdict = "baseline"
        elif change <= -SUSPICION_PCT:
            verdict = "SUSPICIOUS - check leakage"
        elif change <= -ADOPTION_BAR_PCT:
            verdict = "clears the bar"
        else:
            verdict = "no meaningful gain"
        print(f"{label:<28}{mae:>10.4f}{change:>+9.2f}%{trees:>8}   {verdict}")

    section("ALL SEVEN TARGETS")
    targets = ["Moneyline"] + [label for _t, label, _s, _c in REGRESSION_TARGETS]
    header = f"{'TARGET':<12}" + "".join(f"{label.split(':')[0]:>18}" for label in labels)
    print(header)
    print("-" * len(header))
    for target in targets:
        row = f"{target:<12}"
        for label in labels:
            r = all_results[label][target]
            value = r["log_loss"] if target == "Moneyline" else r["mae"]
            row += f"{value:>18.4f}"
        print(row)
    print("(moneyline is log loss, the rest are MAE; lower is better throughout)")

    print(f"\n{'TARGET':<12}" + "".join(f"{label.split(':')[0]:>18}" for label in labels)
          + "   <- TREE COUNTS")
    for target in targets:
        row = f"{target:<12}"
        for label in labels:
            row += f"{all_results[label][target]['trees']:>18}"
        print(row)
    print("Collapsing tree counts alongside a better MAE indicates dilution")
    print("rather than signal - that is how the advanced-stats experiment was")
    print("diagnosed (CLAUDE.md section 16).")

    print("\nDILUTION READ (tree count down >25% AND the metric no better):")
    flagged = False
    for target in targets:
        key = "log_loss" if target == "Moneyline" else "mae"
        base_trees = all_results[labels[0]][target]["trees"]
        base_metric = all_results[labels[0]][target][key]
        for label in labels[1:]:
            trees = all_results[label][target]["trees"]
            metric = all_results[label][target][key]
            if trees < base_trees * 0.75 and metric >= base_metric:
                flagged = True
                print(f"  {label:<28}{target:<12} trees {base_trees} -> {trees}"
                      f"   {key} {base_metric:.4f} -> {metric:.4f}")
    if flagged:
        print("  Early stopping converging sooner while the score fails to")
        print("  improve is added variance, not added information - the same")
        print("  signature the 58-feature advanced-stats run produced.")
    else:
        print("  Nothing flagged.")


def clean_game_ids(players: pd.DataFrame, weights: dict) -> set:
    """Games where EVERY absent player has a known weight under EVERY variant.

    Section 5's matched subset. C covers more absences than A and B (its
    on/off half needs no rolling window), so a raw comparison gives C an
    advantage in coverage that has nothing to do with the weighting being
    better. This strips that out: only team-games with no unknown-role
    absentee under any variant survive, and both sides of a game must qualify.
    """
    absent = players["IS_ABSENT"]
    unknown = pd.Series(False, index=players.index)
    for weight in weights.values():
        unknown |= absent & weight.isna()

    dirty = players.loc[unknown, "GAME_ID"].unique()
    every = players["GAME_ID"].unique()
    return {int(g) for g in set(every) - set(dirty)}


def matched_subset_check(all_results: dict, labels: list, clean: set) -> None:
    section("MATCHED SUBSET - THE SECONDARY CHECK SECTION 5 ASKS FOR")
    print("The zero fractions differed (C covers 25.7% of absences as unknown")
    print("against 40.5% for the others), so a raw comparison could reward C for")
    print("coverage rather than for weighting. This restricts scoring to games")
    print("where no variant has an unknown-role absentee on either side.\n")

    ids = all_results[labels[0]]["Spread"]["game_ids"]
    mask = np.array([int(g) in clean for g in ids])
    print(f"  test games scored in full          {len(ids):,}")
    print(f"  test games in the matched subset   {int(mask.sum()):,} "
          f"({mask.mean():.1%})\n")

    if mask.sum() < 200:
        print("  Too few games survive for this to say anything. Reported as a")
        print("  limitation rather than as a result.")
        return

    base = float(all_results[labels[0]]["Spread"]["errors"][mask].mean())
    print(f"  {'VARIANT':<28}{'MAE (matched)':>15}{'CHANGE':>10}")
    print("-" * 55)
    for label in labels:
        mae = float(all_results[label]["Spread"]["errors"][mask].mean())
        change = (mae - base) / base * 100
        print(f"  {label:<28}{mae:>15.4f}{change:>+9.2f}%")
    print("\n  The ordering is what matters here, not the level: MAE on a")
    print("  subset is not comparable to MAE on the whole test set, because")
    print("  the subset is a different population of games.")


def significance(all_results: dict, labels: list, best_label: str,
                 iterations=2000, seed=0) -> dict:
    """Is the best variant's spread difference real, or resampling noise?

    -0.26% on one metric is unreadable without knowing how far that metric
    moves under resampling. Paired: both variants score the same games in
    every resample.
    """
    section("IS THE BEST VARIANT'S DIFFERENCE REAL?")

    base = all_results[labels[0]]["Spread"]["errors"]
    best = all_results[best_label]["Spread"]["errors"]
    rng = np.random.default_rng(seed)
    n = len(base)

    deltas = []
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        deltas.append(best[idx].mean() - base[idx].mean())
    deltas = np.array(deltas)
    ci = np.percentile(deltas, [2.5, 97.5])
    base_mae = base.mean()

    print(f"Paired bootstrap, {iterations} resamples of {n:,} games "
          f"({best_label} minus baseline):\n")
    print(f"  spread MAE difference   {deltas.mean():>+9.4f}   "
          f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"  as a percentage         {deltas.mean() / base_mae * 100:>+9.2f}%   "
          f"95% CI [{ci[0] / base_mae * 100:+.2f}%, {ci[1] / base_mae * 100:+.2f}%]")
    spans_zero = ci[0] * ci[1] <= 0
    print(f"\n  {'SPANS ZERO - not distinguishable from noise' if spans_zero else 'excludes zero'}")
    return {"spans_zero": bool(spans_zero), "ci": ci}


def verdict(all_results: dict, labels: list, stats_result=None) -> None:
    section("VERDICT")
    baseline_label = labels[0]
    baseline_spread = all_results[baseline_label]["Spread"]["mae"]

    best_label, best_change = None, 0.0
    for label in labels[1:]:
        change = ((all_results[label]["Spread"]["mae"] - baseline_spread)
                  / baseline_spread * 100)
        if best_label is None or change < best_change:
            best_label, best_change = label, change

    print(f"Best variant on spread: {best_label} at {best_change:+.2f}%\n")

    clears = best_change <= -ADOPTION_BAR_PCT
    if clears:
        regressions = []
        for target in ["Moneyline"] + [l for _t, l, _s, _c in REGRESSION_TARGETS]:
            key = "log_loss" if target == "Moneyline" else "mae"
            base = all_results[baseline_label][target][key]
            got = all_results[best_label][target][key]
            if (got - base) / base * 100 > 1.0:
                regressions.append(target)
        if regressions:
            print(f"Clears the spread bar but REGRESSES {regressions} by more")
            print("than 1%. The bar is an improvement without regressing the")
            print("other six, so this is not an adoption.")
        else:
            print("CLEARS THE BAR on spread without regressing the other six.")
            print("NOT SHIPPED IN THIS PHASE - that is a separate change with")
            print("its own verification, and it would need the live serving path")
            print("to be able to compute the same weighting.")
    else:
        print("NO VARIANT CLEARS THE BAR, and the result should be read as a")
        print("rejection with numbers rather than as an inconclusive study.")
        if stats_result is not None and stats_result["spans_zero"]:
            print("\nAnd the best variant's difference is not distinguishable")
            print("from noise: a paired bootstrap puts its 95% CI across zero.")
            print("So the best number here is not a small real gain that missed")
            print("the bar - it is nothing.")
        print(f"\nThe best of three quality weightings moves spread by "
              f"{best_change:+.2f}%,")
        print(f"against a bar of {ADOPTION_BAR_PCT:.0f}% and against the "
              "5.83% that availability")
        print("itself delivered when it was first added.")
        print("\nWhat that says: the MINUTES weighting was already capturing")
        print("most of what a team loses when a player sits. Minutes are a")
        print("coach's revealed judgement of who matters, aggregated over a")
        print("season - a cruder signal than production, but evidently not a")
        print("materially worse one for this purpose.")
        print("\nThat makes this the third rejected feature experiment, and the")
        print("first of the three whose premise was NOT that the model already")
        print("had the information. The model genuinely had no player-quality")
        print("representation; adding one did not help, which is a stronger")
        print("result than the advanced-stats null.")

    simpler = [l for l in labels[1:]
               if abs(((all_results[l]["Spread"]["mae"] - baseline_spread)
                       / baseline_spread * 100) - best_change) < 0.5]
    if len(simpler) > 1:
        print(f"\nWithin 0.5pp of each other on spread: {simpler}.")
        print("Prefer the simpler of these - B over C, A over B. C carries an")
        print("ongoing correctness burden around its leakage guard that buys")
        print("nothing measurable.")


def main():
    section("DATA")
    dataset = load_dataset()
    players = load_players()
    print(f"Loaded {len(players):,} player-rows from {PLAYERS_PATH.name}")

    weights, on_off = build_weights(players)
    labels = list(weights)
    sides = side_map(players)

    sided = {label: to_sided(aggregate(players, weight), sides)
             for label, weight in weights.items()}

    verify_reconstruction(sided[labels[0]], dataset)
    report_zero_fractions(players, weights)
    report_on_off_coverage(players, on_off)
    check_no_leakage(players)

    section("TRAINING - 4 VARIANTS x 7 TARGETS, ALL HELD OUT")
    print("Same 38 features, same frozen split, same frozen architecture. Only")
    print("WEIGHTED_ABSENT_MIN changes; ABSENT_COUNT is untouched.\n")

    all_results = {}
    for label in labels:
        variant = dataset.copy()
        replacement = variant[["GAME_ID"]].merge(
            sided[label], left_on="GAME_ID", right_index=True, how="left")
        for column in SIDED_WEIGHT_COLUMNS:
            variant[column] = replacement[column].to_numpy()
        if variant[SIDED_WEIGHT_COLUMNS].isna().any().any():
            raise SystemExit(f"{label} left NaN in the weight columns")

        print(f"  training {label} ...", flush=True)
        all_results[label] = train_and_score(variant)
        if label == labels[0]:
            verify_baseline(all_results[label])

    print_results(all_results, labels)
    matched_subset_check(all_results, labels, clean_game_ids(players, weights))

    best_label = min(labels[1:], key=lambda l: all_results[l]["Spread"]["mae"])
    stats_result = significance(all_results, labels, best_label)
    verdict(all_results, labels, stats_result)


if __name__ == "__main__":
    main()
