"""Add trailing rolling-window features to the WNBA games table.

Builds SIX candidate feature sets rather than choosing one: windows of 3, 5 and
10, each in two scopes - within-season (NaN before the window fills) and
carried across the season boundary. Phase 3 picks against a validation split;
picking here would be intuition dressed as a pipeline step.

Why this is a decision at all: the NBA drops 13.1% of games to ROLL10, which is
roughly 10/82 and tolerable. A WNBA season is 22 to 44 games, so ten games is a
quarter of a season and nearly half of 2020 - and because a game needs BOTH
teams complete, the loss compounds.
"""

import sys
from pathlib import Path

import pandas as pd

WNBA = Path(__file__).resolve().parents[1]
PIPELINE = WNBA.parent
sys.path.insert(0, str(PIPELINE / "preprocessing"))
from build_rolling_features import trailing_mean  # noqa: E402  pure utility

PROCESSED = PIPELINE / "data" / "wnba" / "processed"
INPUT_PATH = PROCESSED / "wnba_games_final.csv"
OUTPUT_PATH = PROCESSED / "wnba_games_with_rolling.csv"

TEAM_KEY = "TEAM_ID"

WINDOWS = [3, 5, 10]

# Within-season uses the NBA's ROLL prefix so the concepts line up; the carried
# variant gets its own so the two can never be confused in a feature list.
SCOPES = {
    "ROLL": [TEAM_KEY, "SEASON"],
    "CARRY": [TEAM_KEY],
}

# The NBA's metric set plus PTS_ALLOWED, which the NBA only carries implicitly
# through PLUS_MINUS.
METRICS = {
    "WIN": "WIN_PCT",
    "PTS": "PTS",
    "PTS_ALLOWED": "PTS_ALLOWED",
    "PLUS_MINUS": "PLUS_MINUS",
    "FG_PCT": "FG_PCT",
    "REB": "REB",
    "AST": "AST",
    "TOV": "TOV",
}


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def assert_universe_complete(frame: pd.DataFrame) -> None:
    """Every game must carry exactly two rows before anything is rolled.

    A missing team-game makes a "last N games" window silently reach further
    back than it claims - this has bitten twice in the NBA pipeline. The check
    is that the universe is already whole, not that a row is invented to make
    it look whole.
    """
    per_game = frame.groupby("GAME_ID").size()
    bad = per_game[per_game != 2]
    print(f"  games                     : {frame['GAME_ID'].nunique():,}")
    print(f"  rows                      : {len(frame):,}")
    print(f"  games with != 2 rows      : {len(bad)}")
    if len(bad):
        raise SystemExit(f"universe incomplete: {len(bad)} game(s) unpaired")

    # The 2018 forfeit is a genuine absence. No row is invented for it; the two
    # affected teams simply have one fewer prior game from that point on.
    absent = 1021800162
    present = absent in set(frame["GAME_ID"].astype(int))
    print(f"  2018 forfeit {absent} present: {present}  "
          f"(must be False - a real gap, not an invented row)")
    if present:
        raise SystemExit("a row was invented for the 2018 forfeit")


def assert_corrected_margin(frame: pd.DataFrame) -> None:
    """PLUS_MINUS must be the margin derived from PTS, not the raw column.

    Phase 1 established the source column is wrong on ~1% of games, and in a
    rolling feature one bad row reaches up to ten later values. This asserts
    the correction survived into the file being rolled.
    """
    opponent = frame.groupby("GAME_ID")["PTS"].transform(
        lambda s: s.values[::-1] if len(s) == 2 else None)
    derived = frame["PTS"] - opponent
    off = int((derived != frame["PLUS_MINUS"]).sum())
    print(f"  PLUS_MINUS == PTS - opponent PTS on all rows: {off == 0} "
          f"({off} mismatching)")
    if off:
        raise SystemExit(
            f"{off} rows where PLUS_MINUS is not the derived margin - the "
            "phase-1 correction did not survive")


def add_rolling(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values([TEAM_KEY, "GAME_DATE"]).reset_index(drop=True)
    for prefix, keys in SCOPES.items():
        grouped = frame.groupby(keys)
        for window in WINDOWS:
            for source, label in METRICS.items():
                frame[f"{prefix}{window}_{label}"] = trailing_mean(
                    grouped, source, window)
    return frame


def retention_report(frame: pd.DataFrame) -> None:
    """How many GAMES survive each candidate, per season.

    Measured at game level, not row level: a game is usable only if BOTH teams
    have a complete window, which is where the loss compounds.
    """
    section("RETENTION - the trade-off phase 3 decides against")
    print("A game counts as retained only when BOTH teams' windows are complete.\n")

    seasons = sorted(frame["SEASON"].unique())
    candidates = [(p, w) for p in SCOPES for w in WINDOWS]

    print(f"{'SEASON':<8}{'GAMES':>7}" +
          "".join(f"{p + str(w):>10}" for p, w in candidates))
    print("-" * (15 + 10 * len(candidates)))

    totals = {c: 0 for c in candidates}
    total_games = 0
    for season in seasons:
        rows = frame[frame["SEASON"] == season]
        games = rows["GAME_ID"].nunique()
        total_games += games
        line = f"{season:<8}{games:>7}"
        for prefix, window in candidates:
            cols = [f"{prefix}{window}_{m}" for m in METRICS.values()]
            complete = rows.dropna(subset=cols)
            both = complete.groupby("GAME_ID").size()
            kept = int((both == 2).sum())
            totals[(prefix, window)] += kept
            line += f"{kept:>10}"
        print(line)

    print("-" * (15 + 10 * len(candidates)))
    line = f"{'TOTAL':<8}{total_games:>7}"
    for c in candidates:
        line += f"{totals[c]:>10}"
    print(line)
    line = f"{'KEPT %':<8}{'':>7}"
    for c in candidates:
        line += f"{totals[c] / total_games * 100:>9.1f}%"
    print(line)

    print("\n  For reference the NBA keeps 86.9% of its games at ROLL10")
    print("  (11,465 of 13,199), which is the 13.1% drop recorded in CLAUDE.md.")


def boundary_check(frame: pd.DataFrame) -> None:
    """A season's first game must have NaN within-season history.

    Not zero, and not last season's value - that is what CARRY is for, and
    conflating them would hide which feature set is being used.
    """
    section("SEASON-BOUNDARY BEHAVIOUR")
    firsts = frame.sort_values("GAME_DATE").groupby([TEAM_KEY, "SEASON"]).head(1)
    col_roll, col_carry = f"ROLL{WINDOWS[0]}_PTS", f"CARRY{WINDOWS[0]}_PTS"
    print(f"  team-season openers                   : {len(firsts):,}")
    print(f"  with ROLL{WINDOWS[0]}_PTS NaN (required)       : "
          f"{int(firsts[col_roll].isna().sum()):,}")
    print(f"  with ROLL{WINDOWS[0]}_PTS == 0 (must be 0)     : "
          f"{int((firsts[col_roll] == 0).sum()):,}")
    if int(firsts[col_roll].isna().sum()) != len(firsts):
        raise SystemExit("a season opener has within-season rolling history")

    # The carried variant SHOULD have history for every opener except the very
    # first season of each franchise.
    has_carry = int(firsts[col_carry].notna().sum())
    print(f"  with CARRY{WINDOWS[0]}_PTS present            : {has_carry:,} "
          f"(expected: all but each franchise's first {WINDOWS[0]} games)")


def main() -> int:
    print(__doc__)
    frame = pd.read_csv(INPUT_PATH, dtype={"GAME_ID": str})
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])

    section("INPUT GATES")
    assert_universe_complete(frame)
    assert_corrected_margin(frame)

    frame["WIN"] = (frame["WL"] == "W").astype(int)
    frame["PTS_ALLOWED"] = frame["PTS"] - frame["PLUS_MINUS"]
    frame = add_rolling(frame)

    retention_report(frame)
    boundary_check(frame)

    frame.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    added = len(SCOPES) * len(WINDOWS) * len(METRICS)
    section("OUTPUT")
    print(f"  wrote {OUTPUT_PATH.name}")
    print(f"  rows {len(frame):,}, columns {len(frame.columns)} "
          f"({added} rolling features added)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
