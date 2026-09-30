"""Does it matter HOW absences are aggregated, rather than how they are weighted?

Study only. Trains nothing that ships, writes no artifacts, and mutates no
pipeline file - the variant columns are built in memory and swapped into a copy
of model_dataset.csv.

The previous study replaced the WEIGHT on each absence (rolling PRA, PRA per
minute, on/off) and all three failed. That left one explanation standing:
WEIGHTED_ABSENT_MIN SUMS over absentees, so a team missing one 35-minute
starter and a team missing three 12-minute bench players land on nearly the same
number. The sum conflates depth with concentration.

So this holds the weight fixed at ROLL10_MIN and varies only the reduction.
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

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data-pipeline" / "data" / "processed"
PLAYERS_PATH = PROCESSED / "player_boxscores_with_rolling.csv"
GAMES_FINAL_PATH = PROCESSED / "games_final.csv"

# The reduction is imported, never reimplemented: the pipeline sums and this
# varies that, and a second copy would let the two drift apart.
sys.path.insert(0, str(PROJECT_ROOT / "data-pipeline" / "preprocessing"))
from build_team_availability import absence_weights, reduce_absence_weights  # noqa: E402

GROUP_KEYS = ["GAME_ID", "TEAM_ID"]
ROLLING_COLUMN = "ROLL10_MIN"

SUM_COLUMNS = ["HOME_WEIGHTED_ABSENT_MIN", "AWAY_WEIGHTED_ABSENT_MIN"]
MAX_COLUMNS = ["HOME_MAX_ABSENT_MIN", "AWAY_MAX_ABSENT_MIN"]

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

ADOPTION_BAR_PCT = 3.0
EXPECTED_TEAM_GAMES = 26_398

BASELINE = "baseline (sum)"
V1 = "V1: max only"
V2 = "V2: sum + max"
V3 = "V3: top-2 sum"


# --------------------------------------------------------------------------
# building the aggregations
# --------------------------------------------------------------------------

def load_players() -> pd.DataFrame:
    players = pd.read_csv(
        PLAYERS_PATH,
        dtype={"GAME_ID": str, "TEAM_ID": str, "PLAYER_ID": str},
        low_memory=False,
    )
    for column in ("MIN_NUMERIC", ROLLING_COLUMN):
        players[column] = pd.to_numeric(players[column], errors="coerce")
    players["IS_ABSENT"] = players["MIN_NUMERIC"].isna()
    print(f"Loaded {len(players):,} player-rows from {PLAYERS_PATH.name}")
    return players


def top_n_sum(players: pd.DataFrame, weights: pd.Series, n=2) -> pd.DataFrame:
    """Sum of the n largest absence weights per team-game.

    Not expressible as a single .agg(), so it is written here rather than
    forced into reduce_absence_weights(). Vectorised by rank rather than
    groupby.apply: 26,398 groups over 339,841 rows.
    """
    frame = players[GROUP_KEYS].assign(ABSENT_WEIGHT=weights)
    frame = frame.sort_values(GROUP_KEYS + ["ABSENT_WEIGHT"],
                              ascending=[True, True, False])
    frame["rank"] = frame.groupby(GROUP_KEYS).cumcount()
    return (frame[frame["rank"] < n]
            .groupby(GROUP_KEYS, as_index=False)
            .agg(WEIGHTED=("ABSENT_WEIGHT", "sum")))


def build_aggregations(players: pd.DataFrame) -> dict:
    """One reduction per variant, all over the SAME per-absence weights."""
    weights = absence_weights(players)
    return {
        "sum": reduce_absence_weights(players, weights, how="sum"),
        "max": reduce_absence_weights(players, weights, how="max"),
        "top2": top_n_sum(players, weights, n=2),
    }


def side_map() -> pd.DataFrame:
    games = pd.read_csv(
        GAMES_FINAL_PATH, usecols=["GAME_ID", "TEAM_ID", "IS_HOME"],
        dtype={"TEAM_ID": str},
    )
    games["GAME_ID"] = games["GAME_ID"].astype(str).str.zfill(10)
    return games


def to_sided(availability: pd.DataFrame, sides: pd.DataFrame,
             home_name: str, away_name: str) -> pd.DataFrame:
    merged = availability.merge(sides, on=GROUP_KEYS, how="left",
                                validate="one_to_one")
    if merged["IS_HOME"].isna().any():
        raise SystemExit("a team-game could not be assigned a side")
    home = merged[merged["IS_HOME"]].set_index("GAME_ID")["WEIGHTED"]
    away = merged[~merged["IS_HOME"]].set_index("GAME_ID")["WEIGHTED"]
    out = pd.DataFrame({home_name: home, away_name: away})
    out.index = out.index.astype(int)
    return out.sort_index()


# --------------------------------------------------------------------------
# gates and diagnostics
# --------------------------------------------------------------------------

def verify_universe(players: pd.DataFrame) -> None:
    """Every fixture must contribute exactly two team-games.

    A rolling mean over a frame with holes silently shortens its own window,
    which has bitten twice. Nothing here rolls, but a gap would also mean a
    fixture with one side's absences missing, which would corrupt every variant
    equally and invisibly.
    """
    section("UNIVERSE CHECK")
    team_games = players[GROUP_KEYS].drop_duplicates()
    per_game = team_games.groupby("GAME_ID").size()
    print(f"  team-games          : {len(team_games):,} "
          f"(expected {EXPECTED_TEAM_GAMES:,})")
    print(f"  games               : {per_game.size:,}")
    bad = per_game[per_game != 2]
    print(f"  games without exactly 2 sides: {len(bad)}")
    if len(bad) or len(team_games) != EXPECTED_TEAM_GAMES:
        raise SystemExit("the team-game universe is incomplete")
    print("  PASS")


def verify_reconstruction(sided_sum: pd.DataFrame, dataset: pd.DataFrame) -> None:
    """The rebuilt SUM must equal the column the shipped models trained on.

    The gate for the whole study. If the reconstruction does not reproduce the
    real feature, the variants are being injected somewhere other than where it
    lives and no comparison means anything.
    """
    section("RECONSTRUCTION GATE")
    joined = dataset[["GAME_ID"]].merge(
        sided_sum, left_on="GAME_ID", right_index=True, how="left")
    if joined[SUM_COLUMNS].isna().any().any():
        raise SystemExit("the rebuilt availability does not cover every game")

    worst = 0.0
    for column in SUM_COLUMNS:
        diff = float(np.abs(joined[column].to_numpy()
                            - dataset[column].to_numpy()).max())
        worst = max(worst, diff)
        print(f"  {column:<28} largest absolute difference {diff:.3e}")
    if worst > 1e-9:
        raise SystemExit("the rebuilt sum does not match the dataset - stopping")
    print("\n  PASS - the variant columns slot in exactly where the real one does.")


def report_correlations(sided: dict) -> None:
    """How different is each aggregation from the sum it would replace?

    RUN BEFORE ANY MODEL IS SCORED, because it decides how a null should be
    read. Weak correlation plus a null is strong evidence the model was handed
    a genuinely different view and could not use it. Correlation above ~0.8
    plus a null means the premise was wrong and the experiment never tested
    what it claimed.
    """
    section("CORRELATION DIAGNOSTIC - REPORTED BEFORE ANY SCORE")
    frame = pd.concat([sided["sum"], sided["max"], sided["top2"]], axis=1)
    frame.columns = SUM_COLUMNS + MAX_COLUMNS + ["HOME_TOP2", "AWAY_TOP2"]

    print("Each aggregation against the SUM it would replace, per team-game:\n")
    pairs = [
        ("max  vs sum, home", "HOME_MAX_ABSENT_MIN", "HOME_WEIGHTED_ABSENT_MIN"),
        ("max  vs sum, away", "AWAY_MAX_ABSENT_MIN", "AWAY_WEIGHTED_ABSENT_MIN"),
        ("top2 vs sum, home", "HOME_TOP2", "HOME_WEIGHTED_ABSENT_MIN"),
        ("top2 vs sum, away", "AWAY_TOP2", "AWAY_WEIGHTED_ABSENT_MIN"),
    ]
    values = {}
    for label, a, b in pairs:
        r = float(frame[a].corr(frame[b]))
        values[label] = r
        read = ("LARGELY THE SAME QUANTITY" if abs(r) > 0.8
                else "substantially different")
        print(f"  {label:<20} r = {r:+.4f}   {read}")

    worst = max(abs(v) for v in values.values())
    print()
    if worst > 0.8:
        print("  READ: a null here would mean the PREMISE was wrong - max is not")
        print("  meaningfully different from sum on real absence patterns, and")
        print("  the experiment did not test what it claimed.")
    else:
        print("  READ: a null here would be STRONG evidence - the model is being")
        print("  handed a genuinely different view of the same absences.")


def report_absence_distribution(players: pd.DataFrame, sided: dict) -> None:
    """On how many team-games are max and sum identical by construction?

    With zero or one absence the two reductions cannot differ, so the
    experiment only has purchase on the remainder. That fraction bounds how
    much any of this could have mattered.
    """
    section("ABSENCE DISTRIBUTION")
    counts = (players.groupby(GROUP_KEYS)["IS_ABSENT"].sum().astype(int))
    dist = counts.value_counts().sort_index()
    total = len(counts)
    print(f"{'ABSENCES':>9}{'TEAM-GAMES':>13}{'SHARE':>9}")
    print("-" * 31)
    for n, c in dist.items():
        if n <= 6:
            print(f"{n:>9}{c:>13,}{c / total * 100:>8.1f}%")
    over = dist[dist.index > 6].sum()
    if over:
        print(f"{'7+':>9}{over:>13,}{over / total * 100:>8.1f}%")

    trivial = int(dist.get(0, 0) + dist.get(1, 0))
    print(f"\n  max == sum BY CONSTRUCTION on {trivial:,} of {total:,} "
          f"team-games ({trivial / total * 100:.1f}%)")
    print(f"  so the experiment has purchase on {total - trivial:,} "
          f"({(total - trivial) / total * 100:.1f}%)")

    # Measured rather than assumed - zero-weighted absences make them agree too.
    agree = np.isclose(sided["sum"].to_numpy(), sided["max"].to_numpy()).sum()
    cells = sided["sum"].size
    print(f"\n  actually identical in value: {agree:,} of {cells:,} "
          f"sided values ({agree / cells * 100:.1f}%)")


def report_zero_fractions(players: pd.DataFrame, sided: dict) -> None:
    """How often each reduction silently reports 0 for a real absence.

    40.5% of absences have no ROLL10_MIN and weigh 0. A MAX is MORE exposed to
    that than a sum: a zero-weighted star vanishes entirely, where in a sum the
    other absentees still populate the total.
    """
    section("ZERO-WEIGHTED EXPOSURE")
    absent = players["IS_ABSENT"]
    unknown = absent & players[ROLLING_COLUMN].isna()
    print(f"  absent player-rows             : {int(absent.sum()):,}")
    print(f"  of those with no ROLL10_MIN    : {int(unknown.sum()):,} "
          f"({unknown.sum() / absent.sum() * 100:.1f}%) -> weighted 0")

    print("\n  team-games reporting 0 despite having at least one absence:")
    counts = players.groupby(GROUP_KEYS)["IS_ABSENT"].sum()
    has_absence = counts[counts > 0].index
    for name in ("sum", "max", "top2"):
        # Counted on the team-game frame, not the sided one: a sided row is a
        # fixture and would conflate the two teams' exposure.
        agg = reduce_index(name, players, sided)
        zero = int(((agg.loc[has_absence] == 0)).sum())
        print(f"    {name:<5} {zero:>7,} of {len(has_absence):,} "
              f"({zero / len(has_absence) * 100:.1f}%)")


def reduce_index(name: str, players: pd.DataFrame, sided: dict) -> pd.Series:
    """Team-game indexed values for one reduction, for the zero-exposure count."""
    weights = absence_weights(players)
    if name == "top2":
        frame = top_n_sum(players, weights, n=2)
    else:
        frame = reduce_absence_weights(players, weights, how=name)
    return frame.set_index(GROUP_KEYS)["WEIGHTED"]


def check_group_isolation(players: pd.DataFrame) -> None:
    """A team-game's value must depend only on its OWN player rows.

    THIS IS NOT A LAG GUARD, AND THAT IS DELIBERATE. The travel and shot-quality
    studies guarded a rolling window against reaching forward in time. This
    reduction has no time dimension at all: it reads ROLL10_MIN and IS_ABSENT on
    the rows of one team-game and reduces them. A temporal guard here would pass
    on a feature that reads nothing, which is exactly the vacuous-guard shape
    caught twice in this project.

    The real property to test is isolation ACROSS GROUPS, and it is tested the
    same way: negative test plus positive control.
    """
    section("GROUP-ISOLATION GUARD - NEGATIVE-TESTED AND POSITIVE-CONTROLLED")
    print("The reduction has no temporal component, so a lag guard would be")
    print("vacuous. ROLL10_MIN's own backward-looking guard lives in")
    print("build_player_rolling_minutes.py and is unchanged by this study.\n")

    weights = absence_weights(players)
    honest = reduce_absence_weights(players, weights, how="max").set_index(GROUP_KEYS)

    # A team-game with several absences AND a non-zero honest value. A probe
    # whose true value is 0 makes the negative test weak: "still 0" is also what
    # a function returning a constant would do. The positive control would catch
    # that, but a probe that cannot be trivially satisfied is better.
    counts = players.groupby(GROUP_KEYS)["IS_ABSENT"].sum()
    candidates = counts[counts >= 3].index
    non_zero = honest.loc[candidates]
    probe = non_zero[non_zero["WEIGHTED"] > 0].index[0]
    print(f"  probe team-game: {probe}, {int(counts.loc[probe])} absences, "
          f"max = {honest.loc[probe, 'WEIGHTED']:.4f}\n")

    in_probe = (players["GAME_ID"] == probe[0]) & (players["TEAM_ID"] == probe[1])

    # Negative test: wreck every OTHER team-game.
    corrupted = players.copy()
    corrupted.loc[~in_probe, ROLLING_COLUMN] = 999.0
    other = reduce_absence_weights(
        corrupted, absence_weights(corrupted), how="max").set_index(GROUP_KEYS)
    moved = not np.isclose(other.loc[probe, "WEIGHTED"],
                           honest.loc[probe, "WEIGHTED"])
    print(f"  every OTHER team-game set to 999 -> "
          f"{other.loc[probe, 'WEIGHTED']:.4f}  "
          f"{'MOVED - LEAK' if moved else 'unchanged, correct'}")
    if moved:
        raise SystemExit("a team-game's value depends on other team-games")

    # Positive control: wreck THIS team-game. Without this, the negative test
    # would pass identically on a function that reads no data at all.
    corrupted = players.copy()
    corrupted.loc[in_probe & corrupted["IS_ABSENT"], ROLLING_COLUMN] = 999.0
    own = reduce_absence_weights(
        corrupted, absence_weights(corrupted), how="max").set_index(GROUP_KEYS)
    changed = not np.isclose(own.loc[probe, "WEIGHTED"],
                             honest.loc[probe, "WEIGHTED"])
    print(f"  THIS team-game's absences to 999 -> "
          f"{own.loc[probe, 'WEIGHTED']:.4f}  "
          f"{'moved, correct' if changed else 'UNCHANGED - reads nothing'}")
    if not changed:
        raise SystemExit("the positive control did not move - the guard is vacuous")
    print("\n  PASS - isolated across groups, responsive within one.")


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def train_and_score(dataset: pd.DataFrame, features: list) -> dict:
    """Every target, held out, on one feature set. Never loads a shipped model."""
    train, validation, test = split_three_way(dataset)
    comparable = test.dropna(subset=ROLLING_FEATURE_COLUMNS)

    results = {}
    classifier = XGBClassifier(**CLASSIFIER_PARAMS)
    classifier.fit(train[features], train[WIN_TARGET],
                   eval_set=[(validation[features], validation[WIN_TARGET])],
                   verbose=False)
    proba = classifier.predict_proba(comparable[features])[:, 1].astype(np.float64)
    results["Moneyline"] = {
        "accuracy": float(accuracy_score(comparable[WIN_TARGET], (proba > 0.5).astype(int))),
        "log_loss": float(log_loss(comparable[WIN_TARGET], proba)),
        "trees": int(classifier.best_iteration),
    }

    for target, label, _stat, _combine in REGRESSION_TARGETS:
        model = XGBRegressor(**REGRESSION_PARAMS)
        model.fit(train[features], train[target],
                  eval_set=[(validation[features], validation[target])],
                  verbose=False)
        predicted = model.predict(comparable[features])
        results[label] = {
            "mae": float(mean_absolute_error(comparable[target], predicted)),
            "trees": int(model.best_iteration),
        }
        if label == "Spread":
            results[label]["errors"] = np.abs(
                comparable[target].to_numpy() - predicted.astype(np.float64))
    results["rows"] = len(comparable)
    return results


def verify_baseline(results: dict) -> None:
    section("BASELINE GATE - THE 38-FEATURE NUMBERS MUST REPRODUCE")
    print("Held out on train only, early-stopped on validation. Nothing here")
    print("loads ml-training/models/: those trained with no holdout and have")
    print("seen the test window.\n")

    failures = []
    for name, got, want in [
            ("Moneyline accuracy", results["Moneyline"]["accuracy"], RECORDED_ACCURACY),
            ("Moneyline log loss", results["Moneyline"]["log_loss"], RECORDED_LOG_LOSS)]:
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
        raise SystemExit(f"{failures} did not reproduce - stopping rather than reporting.")
    print("\n  PASS - all seven reproduce.")


def print_results(all_results: dict, labels: list) -> None:
    section("SPREAD - THE HEADLINE")
    baseline = all_results[labels[0]]["Spread"]["mae"]
    print(f"{'VARIANT':<22}{'FEATURES':>10}{'MAE':>10}{'CHANGE':>10}{'TREES':>8}")
    print("-" * 60)
    for label in labels:
        r = all_results[label]["Spread"]
        change = (r["mae"] - baseline) / baseline * 100
        shown = "-" if label == labels[0] else f"{change:+.2f}%"
        print(f"{label:<22}{all_results[label]['n_features']:>10}"
              f"{r['mae']:>10.4f}{shown:>10}{r['trees']:>8}")

    section("ALL SEVEN TARGETS")
    header = f"{'TARGET':<14}" + "".join(f"{l:>20}" for l in labels)
    print(header)
    print("-" * len(header))
    rows = [("Moneyline (log loss)", "log_loss")] + \
           [(label, "mae") for _t, label, _s, _c in REGRESSION_TARGETS]
    for label, key in rows:
        name = label.split(" (")[0]
        line = f"{name:<14}"
        for variant in labels:
            line += f"{all_results[variant][name][key]:>20.4f}"
        print(line)

    print(f"\n{'TREE COUNTS':<14}" + "".join(f"{l:>20}" for l in labels))
    for label, _key in rows:
        name = label.split(" (")[0]
        line = f"{name:<14}"
        for variant in labels:
            line += f"{all_results[variant][name]['trees']:>20}"
        print(line)


def significance(all_results: dict, labels: list, best_label: str,
                 iterations=2000, seed=0) -> dict:
    section("IS THE BEST VARIANT'S DIFFERENCE REAL?")
    base = all_results[labels[0]]["Spread"]["errors"]
    best = all_results[best_label]["Spread"]["errors"]
    rng = np.random.default_rng(seed)
    n = len(base)

    base_mae = base.mean()

    def interval(s):
        r = np.random.default_rng(s)
        d = np.array([(best[i].mean() - base[i].mean())
                      for i in (r.integers(0, n, n) for _ in range(iterations))])
        return d.mean(), np.percentile(d, [2.5, 97.5])

    mean, ci = interval(seed)
    print(f"Paired bootstrap, {iterations} resamples of {n:,} games "
          f"({best_label} minus baseline):\n")
    print(f"  spread MAE difference   {mean:>+9.4f}   "
          f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"  as a percentage         {mean / base_mae * 100:>+9.2f}%   "
          f"95% CI [{ci[0] / base_mae * 100:+.2f}%, {ci[1] / base_mae * 100:+.2f}%]")
    spans = ci[0] * ci[1] <= 0
    print(f"\n  {'SPANS ZERO - not distinguishable from noise' if spans else 'EXCLUDES ZERO'}")

    # SEED STABILITY. An interval whose bound sits a hair from zero is a
    # different claim from one that clears it comfortably, and reporting
    # "excludes zero" off a single seed would overclaim. Ten seeds, and the
    # verdict is only "reliable" if they agree.
    print(f"\n  Bound nearest zero is {min(abs(ci[0]), abs(ci[1])):.4f}. "
          "Re-running across seeds:\n")
    excludes = 0
    for s in range(10):
        _m, c = interval(s)
        clear = c[0] * c[1] > 0
        excludes += clear
        print(f"    seed {s}: [{c[0]:+.4f}, {c[1]:+.4f}]  "
              f"{'excludes zero' if clear else 'SPANS ZERO'}")
    print(f"\n  {excludes} of 10 seeds exclude zero.")
    stable = excludes == 10
    if not stable:
        print("  -> NOT STABLE. Treat this as a tie, not a reliable difference.")
    return {"spans_zero": bool(spans), "ci": ci, "stable": stable,
            "seeds_excluding": excludes}


def verdict(all_results: dict, labels: list, stats) -> None:
    section("VERDICT")
    baseline = all_results[labels[0]]["Spread"]["mae"]
    best_label, best_change = None, 0.0
    for label in labels[1:]:
        change = (all_results[label]["Spread"]["mae"] - baseline) / baseline * 100
        if best_label is None or change < best_change:
            best_label, best_change = label, change

    print(f"  best variant           : {best_label} at {best_change:+.2f}% spread MAE")
    print(f"  adoption bar           : {ADOPTION_BAR_PCT:.1f}% improvement")
    if stats and (stats["spans_zero"] or not stats["stable"]):
        why = ("spans zero" if stats["spans_zero"]
               else f"excludes zero on only {stats['seeds_excluding']}/10 seeds")
        print(f"  bootstrap interval     : {why}")
        print("\n  TIE. Not a small gain that missed the bar - nothing.")
    elif best_change < -ADOPTION_BAR_PCT:
        print("\n  CLEARS THE BAR - investigate before believing it.")
    else:
        print("\n  REJECTED. The difference is reliable across seeds but far")
        print(f"  below the bar: {abs(best_change):.2f}% against {ADOPTION_BAR_PCT:.1f}%.")
        print("  Reliably measurable and practically irrelevant are different")
        print("  things, and this is the second.")
    print("\n  ml-training/models/ is untouched. Nothing here ships.")


# --------------------------------------------------------------------------

def main():
    players = load_players()
    verify_universe(players)

    aggregations = build_aggregations(players)
    sides = side_map()
    sided = {
        "sum": to_sided(aggregations["sum"], sides, *SUM_COLUMNS),
        "max": to_sided(aggregations["max"], sides, *MAX_COLUMNS),
        "top2": to_sided(aggregations["top2"], sides, "HOME_TOP2", "AWAY_TOP2"),
    }

    dataset = load_dataset()
    verify_reconstruction(sided["sum"], dataset)

    # Diagnostics BEFORE any score, because they decide how a null reads.
    report_correlations(sided)
    report_absence_distribution(players, sided)
    report_zero_fractions(players, sided)
    check_group_isolation(players)

    def swapped(**columns) -> pd.DataFrame:
        frame = dataset.copy()
        for name, series in columns.items():
            frame[name] = frame["GAME_ID"].map(series)
        if frame[list(columns)].isna().any().any():
            raise SystemExit("a variant column did not cover every game")
        return frame

    max_home = sided["max"]["HOME_MAX_ABSENT_MIN"]
    max_away = sided["max"]["AWAY_MAX_ABSENT_MIN"]
    top2_home = sided["top2"]["HOME_TOP2"]
    top2_away = sided["top2"]["AWAY_TOP2"]

    arms = {
        BASELINE: (dataset, list(FEATURE_COLUMNS)),
        # V2 FIRST: the hypothesis argues that concentration ADDS to depth, not
        # that it replaces it. The narrow variant is the weaker expression here,
        # which inverts this project's usual narrow-first ordering.
        V2: (swapped(HOME_MAX_ABSENT_MIN=max_home, AWAY_MAX_ABSENT_MIN=max_away),
             list(FEATURE_COLUMNS) + MAX_COLUMNS),
        V1: (swapped(HOME_WEIGHTED_ABSENT_MIN=max_home,
                     AWAY_WEIGHTED_ABSENT_MIN=max_away), list(FEATURE_COLUMNS)),
        V3: (swapped(HOME_WEIGHTED_ABSENT_MIN=top2_home,
                     AWAY_WEIGHTED_ABSENT_MIN=top2_away), list(FEATURE_COLUMNS)),
    }

    all_results, labels = {}, [BASELINE, V2, V1, V3]
    for label in labels:
        frame, features = arms[label]
        section(f"TRAINING: {label}  ({len(features)} features)")
        all_results[label] = train_and_score(frame, features)
        all_results[label]["n_features"] = len(features)
        print(f"  spread MAE {all_results[label]['Spread']['mae']:.4f} on "
              f"{all_results[label]['rows']:,} test games")
        if label == BASELINE:
            verify_baseline(all_results[label])

    print_results(all_results, labels)

    baseline_mae = all_results[BASELINE]["Spread"]["mae"]
    best = min(labels[1:],
               key=lambda l: all_results[l]["Spread"]["mae"])
    stats = significance(all_results, labels, best)
    verdict(all_results, labels, stats)


if __name__ == "__main__":
    main()
