"""Read-only quality gate for the quarter scores in"""

import glob
import sys
from pathlib import Path

import pandas as pd

INGESTION_DIR = Path(__file__).resolve().parents[1] / "ingestion"
sys.path.insert(0, str(INGESTION_DIR))
from fetch_quarter_scores import (  # noqa: E402
    EXPECTED_ROWS_PER_GAME,
    OUTPUT_DIR,
    TARGET_COLUMNS,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GAMES_FINAL_PATH = DATA_DIR / "processed" / "games_final.csv"
RAW_SEASONS_GLOB = str(DATA_DIR / "raw" / "games_*.csv")

QUARTER_COLUMNS = ["PTS_Q1", "PTS_Q2", "PTS_Q3", "PTS_Q4"]

OVERTIME_MIN_THRESHOLD = 255

MAX_PLAUSIBLE_OVERTIME_POINTS = 60

EXPECTED_OT_SHARE_RANGE = (0.02, 0.12)

EXAMPLES_TO_PRINT = 5
PROGRESS_EVERY = 2500

ISSUE_LABELS = {
    "unreadable": "files that could not be read",
    "column_mismatch": "columns differ from the canonical set",
    "wrong_row_count": f"not exactly {EXPECTED_ROWS_PER_GAME} team rows",
    "game_id_mismatch": "GAME_ID inside file != filename",
    "duplicate_team": "the two rows share a TEAM_ID",
    "unparseable": "quarter or final score not numeric",
    "quarters_exceed_final": "quarters sum to MORE than games_final PTS",
    "implausible_overtime": f"shortfall > {MAX_PLAUSIBLE_OVERTIME_POINTS} points",
    "team_not_in_games_final": "team missing from games_final (cannot classify)",
}

DIAGNOSTIC_LABELS = {
    "final_disagrees": "LineScore FINAL_PTS != games_final PTS",
}

def load_expected_game_ids() -> set:
    games = pd.read_csv(GAMES_FINAL_PATH, usecols=["GAME_ID"])
    return {str(gid).zfill(10) for gid in games["GAME_ID"].unique()}

def load_final_points() -> dict:
    """(game_id, team_id) -> final PTS, from the already-trusted table."""
    games = pd.read_csv(GAMES_FINAL_PATH, usecols=["GAME_ID", "TEAM_ID", "PTS"])
    return {
        (str(row.GAME_ID).zfill(10), int(row.TEAM_ID)): float(row.PTS)
        for row in games.itertuples()
    }

def load_overtime_by_minutes() -> set:
    """Game ids that went to overtime, judged only by elapsed minutes."""
    overtime = set()
    for path in glob.glob(RAW_SEASONS_GLOB):
        season = pd.read_csv(path, usecols=["GAME_ID", "MIN"])
        hits = season.loc[season["MIN"] >= OVERTIME_MIN_THRESHOLD, "GAME_ID"]
        overtime |= {str(gid).zfill(10) for gid in hits}
    return overtime

def check_structure(game_id, frame, issues) -> bool:
    if list(frame.columns) != TARGET_COLUMNS:
        issues["column_mismatch"].append(game_id)
        return False
    if len(frame) != EXPECTED_ROWS_PER_GAME:
        issues["wrong_row_count"].append(f"{game_id} ({len(frame)} rows)")
        return False
    if not (frame["GAME_ID"].astype(str).str.zfill(10) == game_id).all():
        issues["game_id_mismatch"].append(game_id)
    if frame["TEAM_ID"].nunique() != EXPECTED_ROWS_PER_GAME:
        issues["duplicate_team"].append(game_id)
    return True

def check_quarters(game_id, frame, final_points, issues, diagnostics, gaps) -> bool:
    """Returns True if this game looks like overtime by the score shortfall."""
    is_overtime = False

    for row in frame.itertuples():
        team_id = int(row.TEAM_ID)
        quarters = [getattr(row, c) for c in QUARTER_COLUMNS]

        if any(pd.isna(q) for q in quarters) or pd.isna(row.FINAL_PTS):
            issues["unparseable"].append(f"{game_id} team {team_id}")
            continue

        actual = final_points.get((game_id, team_id))
        if actual is None:
            issues["team_not_in_games_final"].append(f"{game_id} team {team_id}")
            continue

        summed = float(sum(quarters))
        gap = actual - summed

        if gap < 0:
            issues["quarters_exceed_final"].append(
                f"{game_id} team {team_id}: quarters {summed:.0f} > "
                f"games_final {actual:.0f}"
            )
        elif gap > 0:
            is_overtime = True
            gaps.append(gap)
            if gap > MAX_PLAUSIBLE_OVERTIME_POINTS:
                issues["implausible_overtime"].append(
                    f"{game_id} team {team_id}: {gap:.0f} points beyond regulation"
                )

        reported = float(row.FINAL_PTS)
        if abs(reported - actual) > 1e-9:
            diagnostics["final_disagrees"].append(
                f"{game_id} team {team_id}: quarters {summed:.0f}, "
                f"LineScore {reported:.0f} vs games_final {actual:.0f}"
            )

    return is_overtime

def report(title, labels, found_map, flag="FAIL"):
    """Print one block. `flag` is the marker for a non-empty list."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    for key, label in labels.items():
        found = found_map[key]
        print(f"  [{'OK  ' if not found else flag}] {label:<48} {len(found):,}")
        for example in found[:EXAMPLES_TO_PRINT]:
            print(f"           {example}")
        if len(found) > EXAMPLES_TO_PRINT:
            print(f"           ... and {len(found) - EXAMPLES_TO_PRINT:,} more")

def main():
    if not OUTPUT_DIR.exists():
        raise SystemExit(f"{OUTPUT_DIR} does not exist - run the ingestion first.")

    expected_ids = load_expected_game_ids()
    final_points = load_final_points()

    paths = sorted(OUTPUT_DIR.glob("*.csv"))
    found_ids = {p.stem for p in paths}

    print(f"Validating {len(paths):,} files in {OUTPUT_DIR}")
    print(f"games_final.csv lists {len(expected_ids):,} unique games.\n")

    print("=" * 72)
    print("COVERAGE")
    print("=" * 72)
    missing, extra = expected_ids - found_ids, found_ids - expected_ids
    print(f"  files found : {len(found_ids):,}")
    print(f"  missing     : {len(missing):,}")
    print(f"  extra       : {len(extra):,}")
    for label, group in (("missing", missing), ("extra", extra)):
        if group:
            print(f"    {label}: {sorted(group)[:EXAMPLES_TO_PRINT]}")

    issues = {key: [] for key in ISSUE_LABELS}
    diagnostics = {key: [] for key in DIAGNOSTIC_LABELS}
    gaps = []
    overtime_by_score = set()

    for i, path in enumerate(paths, start=1):
        game_id = path.stem
        try:
            frame = pd.read_csv(path, dtype={"GAME_ID": str})
        except Exception as exc:
            issues["unreadable"].append(f"{game_id} ({type(exc).__name__})")
            continue

        if not check_structure(game_id, frame, issues):
            continue

        if check_quarters(game_id, frame, final_points, issues, diagnostics, gaps):
            overtime_by_score.add(game_id)

        if i % PROGRESS_EVERY == 0:
            print(f"  ...{i:,} / {len(paths):,} files checked")

    report("CHECKS (any failure here blocks the run)", ISSUE_LABELS, issues)
    report("DIAGNOSTICS (reported, deliberately not fatal - see docstring)",
           DIAGNOSTIC_LABELS, diagnostics, flag="NOTE")

    print("\n" + "=" * 72)
    print("REGULATION vs OVERTIME (the split the sum check needs)")
    print("=" * 72)
    checked = len(paths) - len(issues["unreadable"]) - len(issues["column_mismatch"])
    regulation = checked - len(overtime_by_score)
    share = len(overtime_by_score) / checked if checked else 0

    print(f"  games checked       : {checked:,}")
    print(f"  regulation (exact)  : {regulation:,}  ({regulation / checked:.1%})")
    print(f"  overtime (shortfall): {len(overtime_by_score):,}  ({share:.1%})")
    print(f"  measured against    : games_final.csv PTS")

    low, high = EXPECTED_OT_SHARE_RANGE
    print(f"  expected OT share   : {low:.0%}-{high:.0%}  -> "
          f"{'in range' if low <= share <= high else 'OUTSIDE RANGE - investigate'}")

    if gaps:
        gap_series = pd.Series(gaps)
        print(f"\n  overtime shortfall per team-game (points beyond regulation):")
        print(f"    min {gap_series.min():.0f}   median {gap_series.median():.0f}   "
              f"max {gap_series.max():.0f}")
        print(f"    all strictly positive: {bool((gap_series > 0).all())}")

    print("\n" + "=" * 72)
    print("TWO INDEPENDENT OVERTIME SIGNALS - do they agree?")
    print("=" * 72)
    overtime_by_minutes = load_overtime_by_minutes() & found_ids
    only_score = overtime_by_score - overtime_by_minutes
    only_minutes = overtime_by_minutes - overtime_by_score
    both = overtime_by_score & overtime_by_minutes

    print(f"  by score shortfall  : {len(overtime_by_score):,}")
    print(f"  by MIN >= {OVERTIME_MIN_THRESHOLD}      : {len(overtime_by_minutes):,}")
    print(f"  agreed by both      : {len(both):,}")
    print(f"  score only          : {len(only_score):,}")
    print(f"  minutes only        : {len(only_minutes):,}")

    if not only_score and not only_minutes:
        print("\n  Perfect agreement between two independently-derived signals.")
    else:
        print("\n  DISAGREEMENT - worth investigating before trusting either:")
        for game_id in sorted(only_score)[:EXAMPLES_TO_PRINT]:
            print(f"    shortfall but not minutes: {game_id}")
        for game_id in sorted(only_minutes)[:EXAMPLES_TO_PRINT]:
            print(f"    minutes but not shortfall: {game_id}")

    total_failures = sum(len(v) for v in issues.values()) + len(missing) + len(extra)
    total_diagnostics = sum(len(v) for v in diagnostics.values())

    print("\n" + "=" * 72)
    print("PASS" if total_failures == 0 else f"FAIL - {total_failures:,} issue(s)")
    if total_diagnostics:
        print(f"({total_diagnostics:,} diagnostic(s) reported above, not counted)")
    print("=" * 72)
    return 0 if total_failures == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
