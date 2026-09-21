"""Trailing Q1 and first-half form, per team-game."""

from pathlib import Path

import pandas as pd

from build_rolling_features import WINDOWS, derive_season

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RAW_PATH = PROCESSED_DIR / "quarter_half_raw.csv"
GAMES_FINAL_PATH = PROCESSED_DIR / "games_final.csv"
OUTPUT_PATH = PROCESSED_DIR / "quarter_half_rolling.csv"

TEAM_KEY = "TEAM_ID"
MERGE_KEYS = ["GAME_ID", TEAM_KEY]

OWN_COLUMNS = ["Q1_PTS", "HALF1_PTS"]

ROLLING_METRICS = [
    "Q1_MARGIN",
    "Q1_PTS",
    "Q1_PTS_ALLOWED",
    "HALF1_MARGIN",
    "HALF1_PTS",
    "HALF1_PTS_ALLOWED",
]

EXPECTED_UNIVERSE_ROWS = 26_398
EXPECTED_TEAMS_PER_GAME = 2

def load_universe() -> pd.DataFrame:
    """Every team-game, with its date. The frame everything is aligned to."""
    games = pd.read_csv(
        GAMES_FINAL_PATH, usecols=MERGE_KEYS + ["GAME_DATE"]
    )
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])

    duplicates = int(games.duplicated(subset=MERGE_KEYS).sum())
    if duplicates:
        raise RuntimeError(
            f"games_final.csv has {duplicates:,} duplicate (GAME_ID, TEAM_ID) "
            f"pairs; every join below assumes it is unique."
        )

    print(f"Universe: {len(games):,} team-games from {GAMES_FINAL_PATH.name}.")
    return games

def load_raw() -> pd.DataFrame:
    """Quarter/half scores, GAME_ID reconciled to the team pipeline's int."""
    raw = pd.read_csv(RAW_PATH, dtype={"GAME_ID": str})
    raw["GAME_ID"] = raw["GAME_ID"].astype(int)

    sizes = raw.groupby("GAME_ID").size()
    wrong = sizes[sizes != EXPECTED_TEAMS_PER_GAME]
    if len(wrong):
        raise RuntimeError(
            f"{len(wrong):,} games do not have exactly "
            f"{EXPECTED_TEAMS_PER_GAME} team rows, so the pairing below cannot "
            f"identify an opponent. First few:\n{wrong.head().to_string()}"
        )

    print(f"Raw quarter/half: {len(raw):,} team-games, "
          f"{raw['GAME_ID'].nunique():,} games, all with two teams.")
    return raw

def attach_opponent(raw: pd.DataFrame) -> pd.DataFrame:
    """Read the other team's scores across onto each row."""
    other = raw.rename(columns={
        TEAM_KEY: "OPP_TEAM_ID",
        **{col: f"OPP_{col}" for col in OWN_COLUMNS},
    })

    paired = raw.merge(other, on="GAME_ID", how="inner")
    paired = paired[paired[TEAM_KEY] != paired["OPP_TEAM_ID"]]

    if len(paired) != len(raw):
        raise RuntimeError(
            f"pairing produced {len(paired):,} rows from {len(raw):,}; expected "
            f"one opponent per team-game exactly."
        )

    print(f"Opponent scores paired onto {len(paired):,} rows.")
    return paired.drop(columns=["OPP_TEAM_ID"]).reset_index(drop=True)

def derive_metrics(paired: pd.DataFrame) -> pd.DataFrame:
    """Margin and points-allowed, from the two sides now on one row."""
    out = paired[MERGE_KEYS].copy()
    for period in ("Q1", "HALF1"):
        own, opponent = f"{period}_PTS", f"OPP_{period}_PTS"
        out[f"{period}_PTS"] = paired[own]
        out[f"{period}_PTS_ALLOWED"] = paired[opponent]
        out[f"{period}_MARGIN"] = paired[own] - paired[opponent]
    return out

def reindex_to_universe(universe: pd.DataFrame,
                        metrics: pd.DataFrame) -> pd.DataFrame:
    """Put the missing games back as NaN rows. See the module docstring."""
    merged = universe.merge(
        metrics, on=MERGE_KEYS, how="left", validate="one_to_one",
        indicator="_raw_merge",
    )

    if len(merged) != EXPECTED_UNIVERSE_ROWS:
        raise RuntimeError(
            f"reindex produced {len(merged):,} rows, expected "
            f"{EXPECTED_UNIVERSE_ROWS:,}."
        )

    unmatched = merged.loc[merged["_raw_merge"] != "both", "GAME_ID"]
    print(f"Reindexed to the full universe: {len(unmatched)} rows without raw "
          f"data, from games {sorted(unmatched.unique())}.")

    per_game = unmatched.value_counts()
    lopsided = per_game[per_game != EXPECTED_TEAMS_PER_GAME]
    if len(lopsided):
        raise RuntimeError(
            f"{len(lopsided)} game(s) are missing quarter data for only ONE "
            f"team. A margin cannot be formed from one side, and the gap was "
            f"expected to be per-game, not per-team:\n{lopsided.to_string()}"
        )

    return merged.drop(columns=["_raw_merge"])

def add_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """shift(1) then roll, per team-season. Same as build_rolling_features."""
    df["SEASON"] = derive_season(df["GAME_DATE"])
    df = df.sort_values([TEAM_KEY, "SEASON", "GAME_DATE"]).reset_index(drop=True)

    grouped = df.groupby([TEAM_KEY, "SEASON"])
    for window in WINDOWS:
        for metric in ROLLING_METRICS:
            df[f"ROLL{window}_{metric}"] = grouped[metric].transform(
                lambda s, w=window: s.shift(1).rolling(w).mean()
            )
    return df

def check_warmup(df: pd.DataFrame) -> None:
    """The NaN count each window must produce if shift-then-roll is right."""
    team_seasons = df.groupby([TEAM_KEY, "SEASON"]).ngroups
    print(f"\nRolling warm-up check ({team_seasons:,} team-seasons):")

    for window in WINDOWS:
        floor = team_seasons * window
        column = f"ROLL{window}_Q1_MARGIN"
        actual = int(df[column].isna().sum())
        print(f"  ROLL{window:<3} expected floor {floor:>6,}   actual {actual:>6,}   "
              f"extra {actual - floor:>4,} (from the 3 missing games)")
        if actual < floor:
            raise RuntimeError(
                f"{column} has fewer NaN than the warm-up alone requires - "
                f"shift(1) may not be applied."
            )

def main():
    universe = load_universe()
    raw = load_raw()

    paired = attach_opponent(raw)
    metrics = derive_metrics(paired)
    df = reindex_to_universe(universe, metrics)
    df = add_rolling(df)

    check_warmup(df)

    rolling_columns = [f"ROLL{w}_{m}" for w in WINDOWS for m in ROLLING_METRICS]
    out = df[MERGE_KEYS + rolling_columns].copy()
    out["GAME_ID"] = out["GAME_ID"].astype(str).str.zfill(10)
    out = out.sort_values(MERGE_KEYS).reset_index(drop=True)

    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(out):,} rows x {len(out.columns)} columns "
          f"({len(rolling_columns)} rolling) to {OUTPUT_PATH}")

    print("\nDistributions (NaN excluded):")
    print(df[rolling_columns].describe().T[["count", "mean", "std", "min", "max"]]
          .to_string())

if __name__ == "__main__":
    main()
