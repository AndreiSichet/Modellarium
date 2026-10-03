"""Reshape the WNBA feature table into one row per game.

Mirrors build_final_dataset.py: HOME_/AWAY_ prefixed features, game-level
columns unprefixed, labels derived from PTS/REB/AST and never from a stored
margin column.
"""

import sys
from pathlib import Path

import pandas as pd

WNBA = Path(__file__).resolve().parents[1]
PIPELINE = WNBA.parent
PROCESSED = PIPELINE / "data" / "wnba" / "processed"
INPUT_PATH = PROCESSED / "wnba_games_final_features.csv"
OUTPUT_PATH = PROCESSED / "wnba_model_dataset.csv"

GAME_LEVEL_COLUMNS = ["GAME_ID", "GAME_DATE", "SEASON"]
# Never prefixed onto a side, because they describe the pairing rather than a team.
NOT_PER_SIDE = ["OPPONENT", "OPPONENT_TEAM_ID", "WL", "WIN"]


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def split_and_prefix(frame, prefix, is_home):
    side = frame[frame["IS_HOME"] == is_home].copy()
    keep = [c for c in side.columns
            if c not in GAME_LEVEL_COLUMNS + NOT_PER_SIDE + ["IS_HOME"]]
    side = side[GAME_LEVEL_COLUMNS + keep]
    return side.rename(columns={c: f"{prefix}_{c}" for c in keep})


def main() -> int:
    print(__doc__)
    frame = pd.read_csv(INPUT_PATH, dtype={"GAME_ID": str})
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])
    # WL is needed for the labels before it is dropped per side.
    wl = frame[frame["IS_HOME"]][["GAME_ID", "WL", "PTS", "REB", "AST"]].rename(
        columns={"WL": "HOME_WL"})

    home = split_and_prefix(frame, "HOME", True)
    away = split_and_prefix(frame, "AWAY", False)
    merged = home.merge(away, on=GAME_LEVEL_COLUMNS, how="inner",
                        validate="one_to_one")
    merged = merged.merge(wl[["GAME_ID", "HOME_WL"]], on="GAME_ID",
                          validate="one_to_one")

    # LABELS, all from PTS/REB/AST or WL - never from a stored margin. This is
    # the NBA pipeline's arrangement, and gaps entry 28 records why it matters:
    # LeagueGameFinder's PLUS_MINUS is wrong on ~1% of games in both leagues,
    # so a label taken from it would be wrong on 1% of rows.
    merged["HOME_WIN"] = (merged["HOME_WL"] == "W").astype(int)
    merged["HOME_MARGIN"] = merged["HOME_PTS"] - merged["AWAY_PTS"]
    merged["TOTAL_PTS"] = merged["HOME_PTS"] + merged["AWAY_PTS"]
    merged["REB_MARGIN"] = merged["HOME_REB"] - merged["AWAY_REB"]
    merged["TOTAL_REB"] = merged["HOME_REB"] + merged["AWAY_REB"]
    merged["AST_MARGIN"] = merged["HOME_AST"] - merged["AWAY_AST"]
    merged["TOTAL_AST"] = merged["HOME_AST"] + merged["AWAY_AST"]
    merged = merged.drop(columns=["HOME_WL"])

    # The label must agree with the recorded result, or one of them is wrong.
    disagree = int((merged["HOME_WIN"] != (merged["HOME_MARGIN"] > 0)).sum())
    if disagree:
        raise SystemExit(f"{disagree} games where HOME_WIN disagrees with the margin")

    merged = merged.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    section("OUTPUT")
    print(f"  wrote {OUTPUT_PATH.name}")
    print(f"  rows {len(merged):,}  columns {len(merged.columns)}")
    print(f"  seasons {merged['SEASON'].min()} .. {merged['SEASON'].max()}")
    print(f"  home win rate {merged['HOME_WIN'].mean() * 100:.1f}%  "
          "(a real-world sanity check, not a coincidence)")
    print(f"  HOME_WIN agrees with the margin on all {len(merged):,} games")

    section("FEATURE GROUPS AVAILABLE TO PHASE 3")
    groups = {}
    for col in merged.columns:
        for prefix in ("HOME_ROLL", "AWAY_ROLL", "HOME_CARRY", "AWAY_CARRY"):
            if col.startswith(prefix):
                window = col[len(prefix):].split("_")[0]
                groups.setdefault(f"{prefix}{window}", []).append(col)
    for name in sorted(groups):
        print(f"  {name:<18} {len(groups[name])} columns")
    others = [c for c in merged.columns
              if not any(c.startswith(p) for p in
                         ("HOME_ROLL", "AWAY_ROLL", "HOME_CARRY", "AWAY_CARRY"))]
    print(f"\n  non-rolling ({len(others)}): {others}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
