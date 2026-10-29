"""Trailing averages of each team's advanced ratings, one row per team-game."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_rolling_features import (  # noqa: E402
    WINDOWS, derive_season_for_frame,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
INPUT_DIR = DATA_DIR / "raw" / "team_advanced_stats"
PROCESSED_DIR = DATA_DIR / "processed"
GAMES_FINAL_PATH = PROCESSED_DIR / "games_final.csv"
OUTPUT_PATH = PROCESSED_DIR / "team_advanced_rolling.csv"

METRICS = {
    "offensiveRating": "OFF_RATING",
    "defensiveRating": "DEF_RATING",
    "netRating": "NET_RATING",
    "pace": "PACE",
    "trueShootingPercentage": "TS_PCT",
}

GROUP_KEYS = ["TEAM_ID", "SEASON"]
SORT_KEYS = ["TEAM_ID", "SEASON", "GAME_DATE", "GAME_ID"]

# EXPECTED_ROWS = 26_398 WAS HERE, compared print-only as MATCH/MISMATCH.
# It is the same literal `build_quarter_half_rolling` removed from itself,
# for the same reason: a row count is not a durable fact. It still read
# MATCH on 2026-10-29 only because nothing refreshes this builder's input -
# no job runs `fetch_team_advanced_stats.py`, so the corpus is frozen at
# 13,199 games while `games_final.csv` has moved on to 13,267. The first
# time anyone fetched advanced stats for a new season it would have read
# MISMATCH and failed nothing, because it was a print.
#
# The expectation is now DERIVED FROM games_final.csv, the way
# `build_quarter_half_rolling.load_universe` does it: the universe of
# team-games is read from that table, the corpus is checked against it, and
# the row count the output must have follows from the two. See
# `expected_output_rows`.
EXPECTED_TEAMS_PER_GAME = 2
GAMES_FINAL_PATH = PROCESSED_DIR / "games_final.csv"
PROGRESS_EVERY = 2000


def expected_output_rows(teams: pd.DataFrame) -> tuple:
    """How many rows the output must have, from games_final.csv.

    THE UNIVERSE IS games_final.csv, exactly as the quarter/half builder
    treats it. Two things are derived from it and neither expires:

      * every game in the corpus must exist in games_final.csv. A game that
        does not is an ORPHAN - an id this pipeline cannot place - and is
        fatal, because every downstream merge keys on that pair.
      * the output must carry two rows for each game that is in BOTH, so the
        count is `2 x |corpus games AND universe games|`.

    WHAT IS DELIBERATELY NOT FATAL: games in games_final.csv that the corpus
    has not fetched. On 2026-10-29 that is 68 games (136 team-rows), because
    nothing schedules `fetch_team_advanced_stats.py` - the advanced columns
    feed no model (tier G3) so the gap has never mattered. It is REPORTED as
    a derived number rather than hidden behind a literal that happened to
    agree with a frozen corpus.
    """
    universe = pd.read_csv(GAMES_FINAL_PATH, usecols=["GAME_ID", "TEAM_ID"],
                           dtype={"GAME_ID": str}, low_memory=False)
    universe["GAME_ID"] = universe["GAME_ID"].str.zfill(10)
    universe_games = set(universe["GAME_ID"])

    corpus_games = set(teams["GAME_ID"].astype(str).str.zfill(10))
    orphans = sorted(corpus_games - universe_games)
    if orphans:
        raise RuntimeError(
            f"{len(orphans):,} game(s) in the advanced corpus are not in "
            f"{GAMES_FINAL_PATH.name}, e.g. {orphans[:3]}. Every downstream "
            f"merge keys on (GAME_ID, TEAM_ID), so an id this pipeline "
            f"cannot place is fatal."
        )

    covered = corpus_games & universe_games
    not_fetched = universe_games - corpus_games
    return len(covered) * EXPECTED_TEAMS_PER_GAME, len(covered), len(not_fetched)

def load_all_advanced() -> pd.DataFrame:
    """Read every per-game file into one frame."""
    paths = sorted(INPUT_DIR.glob("*.csv"))
    if not paths:
        raise SystemExit(f"{INPUT_DIR} is empty - run the ingestion first.")

    print(f"Reading {len(paths):,} advanced box score files...")

    keep = ["gameId", "teamId"] + list(METRICS)
    frames = []
    for i, path in enumerate(paths, start=1):
        frames.append(pd.read_csv(path, usecols=keep, dtype={"gameId": str}))
        if i % PROGRESS_EVERY == 0:
            print(f"  {i:,} / {len(paths):,}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"gameId": "GAME_ID", "teamId": "TEAM_ID"})
    print(f"  combined: {len(combined):,} team-game rows\n")
    return combined

def attach_game_date(teams: pd.DataFrame) -> pd.DataFrame:
    """Join GAME_DATE from the already-trusted game table."""
    games = pd.read_csv(GAMES_FINAL_PATH, usecols=["GAME_ID", "GAME_DATE"])
    games["GAME_ID"] = games["GAME_ID"].astype(str).str.zfill(10)

    before = len(games)
    games = games.drop_duplicates(subset="GAME_ID")
    print(f"games_final.csv: {before:,} rows -> {len(games):,} unique GAME_IDs")

    merged = teams.merge(games, on="GAME_ID", how="left", validate="m:1")

    if len(merged) != len(teams):
        raise RuntimeError(f"join changed rows: {len(teams):,} -> {len(merged):,}")

    unmatched = int(merged["GAME_DATE"].isna().sum())
    if unmatched:
        raise RuntimeError(f"{unmatched:,} rows have no matching GAME_DATE")

    merged["GAME_DATE"] = pd.to_datetime(merged["GAME_DATE"])
    return merged

def add_rolling(teams: pd.DataFrame) -> pd.DataFrame:
    """ROLL5/ROLL10 trailing means per team, reset each season."""
    teams = teams.sort_values(SORT_KEYS).reset_index(drop=True)
    grouped = teams.groupby(GROUP_KEYS)

    for window in WINDOWS:
        for source_column, feature_name in METRICS.items():
            teams[f"ROLL{window}_{feature_name}"] = grouped[source_column].transform(
                lambda s, w=window: s.shift(1).rolling(w).mean()
            )

    return teams

def expected_nan_counts(teams: pd.DataFrame) -> dict:
    """How many NaN each window must produce, derived rather than eyeballed."""
    sizes = teams.groupby(GROUP_KEYS).size()
    return {window: int(sizes.clip(upper=window).sum()) for window in WINDOWS}

def check_output_rows(output: pd.DataFrame, teams: pd.DataFrame) -> tuple:
    """No duplicates, and the row count games_final.csv implies.

    EXTRACTED SO IT CAN BE PLANTED. Both of these lived inline in `main()`,
    where the only way to exercise them was to run the whole builder on a
    corrupted corpus. See `verify_warmup_rules.py`.
    """
    duplicates = int(output.duplicated(subset=["GAME_ID", "TEAM_ID"]).sum())
    if duplicates:
        raise RuntimeError(
            f"{duplicates:,} duplicate (GAME_ID, TEAM_ID) rows - every "
            f"downstream merge assumes this pair is unique."
        )

    expected_rows, covered, not_fetched = expected_output_rows(teams)
    if len(output) != expected_rows:
        raise RuntimeError(
            f"{len(output):,} output rows where {GAMES_FINAL_PATH.name} and "
            f"the corpus between them require {expected_rows:,} "
            f"({covered:,} games x {EXPECTED_TEAMS_PER_GAME}). A row is "
            f"missing or duplicated - this count is DERIVED from "
            f"{GAMES_FINAL_PATH.name}, so a short in-progress season cannot "
            f"explain it."
        )
    return covered, not_fetched


def check_nan_counts(output: pd.DataFrame, teams: pd.DataFrame) -> dict:
    """Every rolling column's NaN count against the warm-up expectation.

    RAISES, RATHER THAN PRINTING. `all_match` used to be computed in `main()`
    and never acted on, so a column whose NaN count disagreed with the
    derived expectation left the step green and the MISMATCH line scrolled
    past - the misleading-green pattern this project keeps tallying. The
    expectation is exact (`sizes.clip(upper=window)`), so a disagreement
    means the rolling is wrong, not that a season is short.
    """
    expected = expected_nan_counts(teams)
    print("\n  NaN counts (early-season warm-up, by design):")
    wrong = []
    for window in WINDOWS:
        for name in METRICS.values():
            column = f"ROLL{window}_{name}"
            actual = int(output[column].isna().sum())
            ok = actual == expected[window]
            if not ok:
                wrong.append(f"{column} {actual:,} != {expected[window]:,}")
            print(f"    {column:<22} {actual:>6,}  expected "
                  f"{expected[window]:>6,}  {'OK' if ok else 'MISMATCH'}")
    print(f"\n  all NaN counts as predicted: {not wrong}")
    if wrong:
        raise RuntimeError(
            f"{len(wrong)} rolling column(s) disagree with the warm-up "
            f"expectation: {wrong[:3]}. The expectation is sum over "
            f"team-seasons of min(window, games), so a short in-progress "
            f"season cannot explain it."
        )
    return expected


def main():
    teams = load_all_advanced()
    teams = attach_game_date(teams)
    teams["SEASON"] = derive_season_for_frame(teams)
    teams = add_rolling(teams)

    rolling_columns = [
        f"ROLL{window}_{name}" for window in WINDOWS for name in METRICS.values()
    ]
    output = teams[["GAME_ID", "TEAM_ID"] + rolling_columns]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    covered, not_fetched = check_output_rows(output, teams)

    print(f"\nWrote {OUTPUT_PATH}")
    print(f"  rows    : {len(output):,}  (derived from "
          f"{GAMES_FINAL_PATH.name}: {covered:,} covered games x "
          f"{EXPECTED_TEAMS_PER_GAME})")
    if not_fetched:
        print(f"  COVERAGE: {not_fetched:,} game(s) in "
              f"{GAMES_FINAL_PATH.name} have no advanced stats fetched yet "
              f"({not_fetched * EXPECTED_TEAMS_PER_GAME:,} team-rows). "
              f"Not fatal - nothing schedules")
        print(f"            fetch_team_advanced_stats.py and these columns "
              f"feed no model - but this table describes a SMALLER universe")
        print(f"            than games_final.csv until someone fetches them.")
    print(f"  columns : {len(output.columns)}  "
          f"({len(rolling_columns)} rolling + GAME_ID + TEAM_ID)")
    print(f"  teams   : {output['TEAM_ID'].nunique()}")
    print(f"  team-seasons: {teams.groupby(GROUP_KEYS).ngroups:,}")

    check_nan_counts(output, teams)

    print("\n  Preview (first rows past the ROLL10 warm-up):")
    preview = output[output["ROLL10_OFF_RATING"].notna()].head(5)
    print(preview.to_string(index=False))

if __name__ == "__main__":
    main()
