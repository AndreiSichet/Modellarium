"""Turn per-shot location data into team-game shot-quality features.

The hypothesis: recent scoring inflated by unsustainable shooting should be
discounted. Box scores record that a team made 42 of 90 field goals; they do
not record whether those were open looks at the rim or contested mid-range
jumpers, so ROLL10_PTS cannot tell a team scoring well from a team shooting
above its sustainable rate.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_rolling_features import derive_season, trailing_mean  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SHOTS_DIR = DATA_DIR / "raw" / "shot_charts"
FAILURES_LOG_PATH = DATA_DIR / "raw" / "shot_charts_failures.log"
GAMES_FINAL_PATH = DATA_DIR / "processed" / "games_final.csv"
OUTPUT_PATH = DATA_DIR / "processed" / "shot_quality.csv"

MERGE_KEYS = ["GAME_ID", "TEAM_ID"]
EXPECTED_TEAMS_PER_GAME = 2

TRAIN_SEASONS = [2021, 2022, 2023]
TEST_SEASONS = [2024, 2025]
SUBSET_SEASONS = TRAIN_SEASONS + TEST_SEASONS

# The zone baseline draws on 2021-2023 (everything that is not test). The MODEL
# splits 2023 off for early stopping, so a game's role differs between the two
# and is labelled by the model's split, which is the one that affects fitting.
MODEL_ROLE = {2021: "train", 2022: "train", 2023: "validation",
              2024: "test", 2025: "test"}

# End-of-quarter heaves. They convert at a rate reflecting desperation rather
# than shot selection, so leaving them in drags the league baseline every other
# zone is measured against. A decision, not an oversight.
EXCLUDED_ZONE = "Backcourt"

ZONE = "SHOT_ZONE_BASIC"
ROLLING_WINDOW = 10

# eFG% counts a made three as 1.5 makes: (FGM + 0.5*FG3M) / FGA.
THREE_POINT_VALUE = 1.5
TWO_POINT_VALUE = 1.0


def load_shots() -> pd.DataFrame:
    """Every per-game shot file on disk, concatenated."""
    files = sorted(SHOTS_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(
            f"no shot files in {SHOTS_DIR}. Run ingestion/fetch_shot_charts.py first."
        )

    frames = [pd.read_csv(f, dtype={"GAME_ID": str}) for f in files]
    shots = pd.concat(frames, ignore_index=True)
    shots["GAME_ID_INT"] = shots["GAME_ID"].astype(int)
    print(f"Loaded {len(shots):,} shots across {len(files):,} games.")
    return shots


def attach_season(shots: pd.DataFrame) -> pd.DataFrame:
    """Season and date from the history table, not re-derived from the shots."""
    games = pd.read_csv(GAMES_FINAL_PATH,
                        usecols=["GAME_ID", "GAME_DATE", "SEASON"])
    games = games.drop_duplicates("GAME_ID")
    merged = shots.merge(games, left_on="GAME_ID_INT", right_on="GAME_ID",
                         how="left", suffixes=("", "_g"))
    if merged["SEASON"].isna().any():
        raise SystemExit("some shots belong to games absent from games_final.csv")
    merged["GAME_DATE"] = pd.to_datetime(merged["GAME_DATE"])
    return merged


def drop_backcourt(shots: pd.DataFrame) -> pd.DataFrame:
    before = len(shots)
    kept = shots[shots[ZONE] != EXCLUDED_ZONE].copy()
    removed = before - len(kept)
    print(f"Dropped {removed:,} {EXCLUDED_ZONE} shots "
          f"({removed / before * 100:.2f}% of all attempts).")
    if EXCLUDED_ZONE in kept[ZONE].unique():
        raise SystemExit(f"{EXCLUDED_ZONE} survived the exclusion")
    return kept


def zone_baseline(shots: pd.DataFrame) -> pd.DataFrame:
    """League conversion rate per zone, from TRAINING SEASONS ONLY.

    The test seasons must never contribute to the baseline they are scored
    against - otherwise every test game is measured partly against itself.
    """
    train = shots[shots["SEASON"].isin(TRAIN_SEASONS)]
    if train.empty:
        raise SystemExit(f"no training-season shots; seasons present: "
                         f"{sorted(shots['SEASON'].unique())}")

    baseline = (train.groupby(ZONE)
                .agg(ATTEMPTS=("SHOT_MADE_FLAG", "size"),
                     MADE=("SHOT_MADE_FLAG", "sum"))
                .reset_index())
    baseline["LEAGUE_FG_PCT"] = baseline["MADE"] / baseline["ATTEMPTS"]
    return baseline.sort_values(ZONE).reset_index(drop=True)


def assert_baseline_is_train_only(shots: pd.DataFrame, baseline: pd.DataFrame) -> None:
    """Negative test: corrupting the test seasons must not move the baseline.

    A guard that has never been shown to fail has not been shown to work, and
    "trained on train only" is exactly the claim that is invisible when wrong.
    """
    poisoned = shots.copy()
    is_test = poisoned["SEASON"].isin(TEST_SEASONS)

    if not is_test.any():
        print("\n  Negative test NOT RUN - no test-season shots on disk, so "
              "corrupting them\n  changes nothing and a pass would be vacuous. "
              "This is unexercised, not\n  verified. Rerun once the full "
              "ingestion covers 2024-25.")
        return

    poisoned.loc[is_test, "SHOT_MADE_FLAG"] = 1
    poisoned.loc[is_test, ZONE] = "Restricted Area"

    after = zone_baseline(poisoned)
    same = baseline[[ZONE, "LEAGUE_FG_PCT"]].equals(after[[ZONE, "LEAGUE_FG_PCT"]])
    print(f"\n  Negative test - every test-season shot forced to a made "
          f"Restricted Area attempt ({int(is_test.sum()):,} rows):")
    print(f"    baseline unchanged: {'yes' if same else 'NO - TEST DATA LEAKS IN'}")
    if not same:
        raise SystemExit("the zone baseline reads test-season shots")

    # Positive control: corrupting the TRAIN seasons must move it, or the test
    # above would pass on a baseline computed from nothing at all.
    control = shots.copy()
    is_train = control["SEASON"].isin(TRAIN_SEASONS)
    control.loc[is_train, "SHOT_MADE_FLAG"] = 1
    moved = not baseline[["LEAGUE_FG_PCT"]].equals(
        zone_baseline(control)[["LEAGUE_FG_PCT"]])
    print(f"    positive control - corrupting TRAIN seasons moves it: "
          f"{'yes' if moved else 'NO'}")
    if not moved:
        raise SystemExit(
            "corrupting the training seasons did not move the baseline, so the "
            "negative test proved nothing"
        )


def team_game_quality(shots: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Expected eFG%, actual eFG% and their difference, per team per game."""
    rates = baseline.set_index(ZONE)["LEAGUE_FG_PCT"]
    unknown = set(shots[ZONE].unique()) - set(rates.index)
    if unknown:
        raise SystemExit(f"zones present in the data but not the baseline: {unknown}")

    shots = shots.copy()
    shots["SHOT_VALUE"] = shots["SHOT_TYPE"].map(
        {"3PT Field Goal": THREE_POINT_VALUE, "2PT Field Goal": TWO_POINT_VALUE})
    if shots["SHOT_VALUE"].isna().any():
        raise SystemExit(f"unexpected SHOT_TYPE values: "
                         f"{sorted(shots['SHOT_TYPE'].unique())}")

    shots["EXPECTED_POINTS"] = shots[ZONE].map(rates) * shots["SHOT_VALUE"]
    shots["ACTUAL_POINTS"] = shots["SHOT_MADE_FLAG"] * shots["SHOT_VALUE"]

    grouped = (shots.groupby(["GAME_ID_INT", "TEAM_ID", "SEASON", "GAME_DATE"])
               .agg(SHOTS=("SHOT_MADE_FLAG", "size"),
                    EXPECTED=("EXPECTED_POINTS", "sum"),
                    ACTUAL=("ACTUAL_POINTS", "sum"))
               .reset_index()
               .rename(columns={"GAME_ID_INT": "GAME_ID"}))

    grouped["SHOT_QUALITY"] = grouped["EXPECTED"] / grouped["SHOTS"]
    grouped["ACTUAL_EFG"] = grouped["ACTUAL"] / grouped["SHOTS"]
    # Positive = converted above what this shot selection is worth.
    grouped["SHOOTING_LUCK"] = grouped["ACTUAL_EFG"] - grouped["SHOT_QUALITY"]

    return grouped.drop(columns=["EXPECTED", "ACTUAL"])


def read_known_fetch_failures() -> set:
    """Games the fetcher recorded as unfetchable, from its own durable log.

    Derived rather than hardcoded, the same reasoning as attach_quarter_half:
    a literal list has no expiry condition and goes stale silently. The log
    stores zero-padded ids while GAME_ID here is an int, so the int() is doing
    real work - compared as-is the two sets could never intersect.
    """
    if not FAILURES_LOG_PATH.exists():
        raise SystemExit(f"{FAILURES_LOG_PATH.name} is missing, so absent games "
                         "cannot be distinguished from an incomplete ingestion")

    failures = set()
    for line in FAILURES_LOG_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        failures.add(int(line.split("\t")[0].strip()))
    return failures


def load_universe() -> pd.DataFrame:
    """Every team-game in the subset seasons. The frame everything aligns to."""
    games = pd.read_csv(GAMES_FINAL_PATH,
                        usecols=MERGE_KEYS + ["GAME_DATE", "SEASON"])
    universe = games[games["SEASON"].isin(SUBSET_SEASONS)].copy()
    universe["GAME_DATE"] = pd.to_datetime(universe["GAME_DATE"])

    duplicates = int(universe.duplicated(subset=MERGE_KEYS).sum())
    if duplicates:
        raise SystemExit(f"games_final.csv has {duplicates:,} duplicate "
                         f"(GAME_ID, TEAM_ID) rows in the subset seasons")
    return universe.sort_values(MERGE_KEYS).reset_index(drop=True)


def reindex_to_universe(universe: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    """Put the missing games back as NaN rows BEFORE rolling.

    Rolling over a frame with holes silently shortens the window: a team's
    "last 10 games" would quietly reach 11 calendar games back, producing a
    confident number computed from a window other than the one it claims. The
    same trap build_quarter_half_rolling.py exists to avoid.
    """
    merged = universe.merge(quality.drop(columns=["SEASON", "GAME_DATE"]),
                            on=MERGE_KEYS, how="left", validate="one_to_one",
                            indicator="_merge_flag")

    if len(merged) != len(universe):
        raise SystemExit(f"reindex produced {len(merged):,} rows, expected "
                         f"{len(universe):,}")

    unmatched = merged.loc[merged["_merge_flag"] != "both", "GAME_ID"]
    absent_games = sorted(unmatched.unique())

    print(f"\nReindexed to the full universe: {len(universe):,} team-games, "
          f"{len(unmatched)} without shot data\nfrom {len(absent_games)} game(s).")

    per_game = unmatched.value_counts()
    lopsided = per_game[per_game != EXPECTED_TEAMS_PER_GAME]
    if len(lopsided):
        raise SystemExit(
            f"{len(lopsided)} game(s) lack shot data for only ONE team. The gap "
            f"is per-game by construction, so this is a different fault:\n"
            f"{lopsided.to_string()}"
        )

    # One-directional, the same asymmetry attach_quarter_half uses: every absent
    # game must be logged, but a logged game that later succeeded is not a
    # problem. The reverse would turn a stale log entry into a false alarm.
    logged = read_known_fetch_failures()
    unexplained = [g for g in absent_games if g not in logged]
    if unexplained:
        raise SystemExit(
            f"games absent from the corpus with no entry in "
            f"{FAILURES_LOG_PATH.name}: {unexplained}. An unexplained gap is an "
            "incomplete ingestion, not a known exclusion."
        )
    print(f"  All {len(absent_games)} are recorded in "
          f"{FAILURES_LOG_PATH.name} (which lists {len(logged)} distinct game(s)).")
    for game in absent_games:
        season = int(universe.loc[universe["GAME_ID"] == game, "SEASON"].iloc[0])
        role = MODEL_ROLE.get(season, "?")
        print(f"    {str(game).zfill(10)}  season {season}  ({role})")

    return merged.drop(columns=["_merge_flag"])


def add_rolling(quality: pd.DataFrame) -> pd.DataFrame:
    """Trailing averages using the pipeline's one lag convention."""
    quality = quality.sort_values(["TEAM_ID", "SEASON", "GAME_DATE", "GAME_ID"])
    quality = quality.reset_index(drop=True)

    grouped = quality.groupby(["TEAM_ID", "SEASON"])
    for column in ("SHOT_QUALITY", "SHOOTING_LUCK"):
        quality[f"ROLL{ROLLING_WINDOW}_{column}"] = trailing_mean(
            grouped, column, ROLLING_WINDOW)

    return quality


def assert_no_lookahead(quality: pd.DataFrame, recompute_with=None) -> None:
    """A row's rolling value must use only that team's strictly-earlier games.

    Negative-tested AND positive-controlled: corrupting the future must not
    move a value, and corrupting the past must, or the check would pass on a
    feature that reads nothing at all.

    recompute_with is the implementation under test, defaulting to add_rolling.
    It is a parameter so a harness can hand in a deliberately leaky version and
    confirm this guard rejects it - otherwise the guard could only ever check
    the one implementation it hardcoded, which is not a test of the guard.
    """
    rebuild = recompute_with or add_rolling
    print("\n  Lag guard:")
    quality = rebuild(quality)
    rolled = quality[quality[f"ROLL{ROLLING_WINDOW}_SHOOTING_LUCK"].notna()]
    if rolled.empty:
        print(f"    NOT RUN - no team has {ROLLING_WINDOW + 1} games on disk "
              "yet, so there is no\n    rolling value to corrupt. Unexercised, "
              "not verified.")
        return

    target = rolled.index[len(rolled) // 2]
    team = quality.loc[target, "TEAM_ID"]
    season = quality.loc[target, "SEASON"]
    date = quality.loc[target, "GAME_DATE"]
    original = quality.loc[target, f"ROLL{ROLLING_WINDOW}_SHOOTING_LUCK"]

    def recompute(frame):
        return rebuild(frame).loc[target, f"ROLL{ROLLING_WINDOW}_SHOOTING_LUCK"]

    same_team = (quality["TEAM_ID"] == team) & (quality["SEASON"] == season)

    future = quality.copy()
    hit = same_team & (future["GAME_DATE"] >= date)
    future.loc[hit, "SHOOTING_LUCK"] = 999.0
    after_future = recompute(future)
    leaked = abs(after_future - original) > 1e-12
    print(f"    {int(hit.sum())} games at or after {date.date()} set to 999 "
          f"-> value {'MOVED - LEAK' if leaked else 'unchanged'}")
    if leaked:
        raise SystemExit("a rolling value reads its own game or later")

    past = quality.copy()
    hit_past = same_team & (past["GAME_DATE"] < date)
    past.loc[hit_past, "SHOOTING_LUCK"] = 999.0
    moved = abs(recompute(past) - original) > 1e-12
    print(f"    {int(hit_past.sum())} PRIOR games set to 999 "
          f"-> value {'moved, as it must' if moved else 'UNCHANGED'}")
    if not moved:
        raise SystemExit(
            "corrupting prior games did not move the value, so the leak test "
            "above proved nothing"
        )


def report_missing_impact(quality: pd.DataFrame) -> None:
    """How far four missing games reach once a 10-game window runs over them.

    One absent team-game does not cost one rolling value - it costs up to
    ROLLING_WINDOW of them, because every window that would have contained it
    is now incomplete. Counting the games understates the downstream reach.
    """
    rolled_column = f"ROLL{ROLLING_WINDOW}_SHOOTING_LUCK"
    print(f"\nHOW THE {int(quality['SHOT_QUALITY'].isna().sum())} MISSING "
          "TEAM-GAMES PROPAGATE")

    missing_raw = int(quality["SHOT_QUALITY"].isna().sum())
    missing_rolled = int(quality[rolled_column].isna().sum())

    # What the warm-up alone would cost: the first ROLLING_WINDOW games of each
    # team-season are NaN regardless of any gap.
    team_seasons = quality.groupby(["TEAM_ID", "SEASON"]).size()
    warmup = int((team_seasons.clip(upper=ROLLING_WINDOW)).sum())

    print(f"  team-games with no shot data            {missing_raw:>7,}")
    print(f"  rolling values NaN in total             {missing_rolled:>7,}")
    print(f"  of which the unavoidable warm-up        {warmup:>7,}  "
          f"(first {ROLLING_WINDOW} per team-season)")
    print(f"  attributable to the four missing games  "
          f"{missing_rolled - warmup:>7,}")
    print(f"  as a share of all {len(quality):,} team-games       "
          f"{(missing_rolled - warmup) / len(quality):>7.2%}")

    print("\n  XGBoost takes NaN natively and learns a default split direction,")
    print("  so these rows are scored rather than dropped - the same treatment")
    print("  every other early-season rolling feature already gets.")


def report_coverage(quality: pd.DataFrame) -> None:
    """How many team-games survive, by season."""
    print("\n  Coverage by season (team-games):")
    counts = quality.groupby("SEASON")["GAME_ID"].nunique()
    for season, games in counts.items():
        role = MODEL_ROLE.get(season, "?")
        print(f"    {season}  {games:>5,} games  ({role})")
    print(f"    total {quality['GAME_ID'].nunique():,} games, "
          f"{len(quality):,} team-games")


def main():
    shots = attach_season(load_shots())
    shots = drop_backcourt(shots)

    baseline = zone_baseline(shots)
    print(f"\nZONE BASELINE - training seasons {TRAIN_SEASONS} only")
    print(f"  {'ZONE':<24}{'ATTEMPTS':>10}{'MADE':>9}{'FG%':>9}")
    for _, row in baseline.iterrows():
        print(f"  {row[ZONE]:<24}{row['ATTEMPTS']:>10,}{int(row['MADE']):>9,}"
              f"{row['LEAGUE_FG_PCT']:>9.4f}")
    print(f"  ({EXCLUDED_ZONE} excluded by design - see EXCLUDED_ZONE)")

    assert_baseline_is_train_only(shots, baseline)

    raw = team_game_quality(shots, baseline)
    quality = add_rolling(reindex_to_universe(load_universe(), raw))
    assert_no_lookahead(quality)
    report_missing_impact(quality)
    report_coverage(quality)

    quality.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\nWrote {OUTPUT_PATH}")
    print(quality[["SHOT_QUALITY", "ACTUAL_EFG", "SHOOTING_LUCK"]]
          .describe().to_string())


if __name__ == "__main__":
    main()
