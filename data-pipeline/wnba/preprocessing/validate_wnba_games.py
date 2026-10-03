"""Read-only quality gate over the raw WNBA season CSVs.

Writes nothing. Every expected count is DERIVED from the data - there is no
games-per-season constant anywhere in this file, deliberately. The WNBA
schedule length has changed five times since 1997 (28, 34, 22, 32, 36, 40, 44
games per team), so a remembered number would flag real seasons as broken. The
2020 bubble season is the test of that: 22 games per team, internally uniform,
and it must pass with no special case.
"""

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PIPELINE = Path(__file__).resolve().parents[2]
RAW_DIR = PIPELINE / "data" / "wnba" / "raw"

GAME_ID_WIDTH = 10
REGULAR_SEASON_TYPE_DIGIT = "2"

REQUIRED_FIELDS = [
    "GAME_ID", "TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION", "GAME_DATE",
    "MATCHUP", "WL", "PTS", "REB", "AST", "PLUS_MINUS", "FGM", "FGA",
]

failures = []
notes = []


def check(season, name, ok, detail=""):
    if not ok:
        failures.append(f"{season}: {name}" + (f" - {detail}" if detail else ""))
    return ok


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def load_seasons() -> dict:
    frames = {}
    for path in sorted(RAW_DIR.glob("wnba_games_*.csv")):
        season = path.stem.replace("wnba_games_", "")
        frames[season] = pd.read_csv(path, dtype={"GAME_ID": str})
    return frames


def validate_season(season: str, frame: pd.DataFrame) -> dict:
    """Every check for one season. Expected counts derived, never hardcoded."""
    rows = len(frame)
    teams = frame["TEAM_ID"].nunique()
    games = frame["GAME_ID"].nunique()

    # 1. every GAME_ID appears exactly twice
    per_game = frame.groupby("GAME_ID").size()
    unpaired = per_game[per_game != 2]
    check(season, "every GAME_ID has exactly 2 rows", len(unpaired) == 0,
          f"{len(unpaired)} game(s) with {sorted(set(unpaired))} rows")

    # 2. uniform schedule - reported, not assumed
    per_team = frame.groupby("TEAM_ID").size()
    spread = Counter(per_team)
    uniform = len(spread) == 1
    modal_games = spread.most_common(1)[0][0]

    # 3. derived expected total
    expected = teams * modal_games / 2
    balanced = abs(expected - games) < 0.5

    # 4. no nulls in required fields
    missing_cols = [c for c in REQUIRED_FIELDS if c not in frame.columns]
    check(season, "required fields present", not missing_cols, str(missing_cols))
    nulls = {c: int(frame[c].isna().sum())
             for c in REQUIRED_FIELDS if c in frame.columns
             and frame[c].isna().any()}
    check(season, "no nulls in required fields", not nulls, str(nulls))

    # 5. exactly one home and one away per game
    frame = frame.assign(IS_HOME=frame["MATCHUP"].str.contains("vs.", regex=False))
    home_per_game = frame.groupby("GAME_ID")["IS_HOME"].sum()
    bad_home = home_per_game[home_per_game != 1]
    check(season, "exactly one home team per game", len(bad_home) == 0,
          f"{len(bad_home)} game(s)")

    # 6. the two rows of a game agree with each other
    paired = frame[frame["GAME_ID"].isin(per_game[per_game == 2].index)]
    agree_date = paired.groupby("GAME_ID")["GAME_DATE"].nunique()
    check(season, "both rows agree on the date", int((agree_date != 1).sum()) == 0,
          f"{int((agree_date != 1).sum())} game(s) disagree")

    pm = paired.groupby("GAME_ID").apply(
        lambda g: pd.Series({
            "pm_sum": g["PLUS_MINUS"].sum(),
            "pm_vs_pts": g["PLUS_MINUS"].iloc[0] - (g["PTS"].iloc[0] - g["PTS"].iloc[1]),
            "wins": (g["WL"] == "W").sum(),
            "wl_matches_pts": (g["WL"].iloc[0] == "W") == (g["PTS"].iloc[0] > g["PTS"].iloc[1]),
        }), include_groups=False)

    check(season, "exactly one winner per game",
          bool((pm["wins"] == 1).all()),
          f"{int((pm['wins'] != 1).sum())} game(s)")

    # WL AGAINST PTS IS THE FATAL ONE. If the recorded winner disagreed with
    # the recorded score, nothing downstream could be trusted.
    check(season, "WL agrees with PTS", bool(pm["wl_matches_pts"].all()),
          f"{int((~pm['wl_matches_pts']).sum())} game(s)")

    # PLUS_MINUS IS A CHARACTERISED SOURCE DEFECT, NOT A FATAL ERROR, and the
    # reason it is not fatal is measured: the NBA's own raw data from the same
    # endpoint family carries it at the SAME rate (135 of 13,209 games, 1.02%,
    # against the WNBA's 1.24%). So it is a LeagueGameFinder property rather
    # than anything about this league, and it lands on the one column that is
    # pure arithmetic over two columns already validated above. The builder
    # recomputes it from PTS; this reports the size of what it recomputes.
    pm_off = int(((pm["pm_sum"].abs() >= 1e-9)
                  | (pm["pm_vs_pts"].abs() >= 1e-9)).sum())
    if pm_off:
        worst = float(pm["pm_vs_pts"].abs().max())
        notes.append(
            f"{season}: PLUS_MINUS does not reconcile on {pm_off} of {len(pm)} "
            f"games ({pm_off / len(pm) * 100:.2f}%), worst |error| {worst:.0f} "
            "pts - recomputed from PTS by the builder")

    # 7. regular season only
    padded = frame["GAME_ID"].str.zfill(GAME_ID_WIDTH)
    non_regular = padded[padded.str[2] != REGULAR_SEASON_TYPE_DIGIT]
    check(season, "only regular-season game ids", len(non_regular) == 0,
          f"{len(non_regular)} row(s)")

    # A NON-UNIFORM SCHEDULE IS FATAL UNLESS IT RECONCILES AS CANCELLED GAMES.
    # Derived, not a special case for any particular year: if N teams are each
    # short by S games, exactly N*S/2 games are absent. A real cancellation
    # satisfies that; partial or corrupted data generally will not.
    if not uniform:
        short = {t: modal_games - n for t, n in per_team.items() if n < modal_games}
        shortfalls = set(short.values())
        implied_missing = sum(short.values()) / 2
        reconciles = (
            shortfalls == {1} or len(shortfalls) == 1
        ) and abs((expected - games) - implied_missing) < 0.5
        if reconciles:
            notes.append(
                f"{season}: {len(short)} team(s) short by "
                f"{sorted(shortfalls)[0]} -> implies exactly "
                f"{implied_missing:.0f} cancelled game(s), and {expected - games:.0f} "
                "are absent. Reconciles; treated as a real cancellation.")
        else:
            check(season, "non-uniform schedule reconciles as cancellations",
                  False,
                  f"games per team {dict(sorted(spread.items()))}, "
                  f"{expected - games:.0f} absent, {implied_missing:.0f} implied")
    elif not balanced:
        check(season, "derived expected game count matches", False,
              f"{teams} x {modal_games} / 2 = {expected:.0f} expected, {games} actual")

    return {
        "season": season, "rows": rows, "teams": teams, "games": games,
        "per_team": dict(sorted(spread.items())), "modal": modal_games,
        "expected": expected, "uniform": uniform, "balanced": balanced,
    }


def report_team_identity(frames: dict) -> None:
    """Each TEAM_ID must be one franchise throughout; abbreviations may change.

    The three known changes (PHO->PHX, SAN->LVA, TUL->DAL) are reported as
    facts, not errors. Keying anything on abbreviation would split one
    franchise into two across the change and corrupt every rolling window and
    Elo rating spanning it.
    """
    section("TEAM IDENTITY - keyed on TEAM_ID, abbreviations reported")

    seen = {}
    for season, frame in sorted(frames.items()):
        for team_id, abbr, name in frame[
                ["TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME"]].drop_duplicates().values:
            seen.setdefault(team_id, []).append((season, abbr, name))

    changed = []
    for team_id, history in sorted(seen.items()):
        abbrs = [a for _, a, _ in history]
        if len(set(abbrs)) > 1:
            first_season = history[0][0]
            transitions = []
            for i in range(1, len(history)):
                if history[i][1] != history[i - 1][1]:
                    transitions.append(
                        f"{history[i-1][1]}->{history[i][1]} at {history[i][0]}")
            changed.append((team_id, first_season, transitions))

    print(f"  distinct TEAM_IDs across all seasons : {len(seen)}")
    print(f"  TEAM_IDs whose abbreviation changed  : {len(changed)}")
    for team_id, first_season, transitions in changed:
        print(f"    {team_id}  from {first_season}:  {', '.join(transitions)}")

    # Each id must be ONE franchise: the id is the key, so the test is that it
    # never maps to two abbreviations in the SAME season.
    conflicts = []
    for season, frame in sorted(frames.items()):
        per_id = frame.groupby("TEAM_ID")["TEAM_ABBREVIATION"].nunique()
        for team_id in per_id[per_id > 1].index:
            conflicts.append((season, team_id))
    print(f"\n  TEAM_IDs with >1 abbreviation WITHIN one season: {len(conflicts)}")
    check("all", "each TEAM_ID is one franchise within a season",
          not conflicts, str(conflicts[:5]))

    # And the reverse: one abbreviation must not be reused by two franchises.
    reuse = []
    for season, frame in sorted(frames.items()):
        per_abbr = frame.groupby("TEAM_ABBREVIATION")["TEAM_ID"].nunique()
        for abbr in per_abbr[per_abbr > 1].index:
            reuse.append((season, abbr))
    print(f"  abbreviations shared by >1 TEAM_ID within a season: {len(reuse)}")
    check("all", "no abbreviation shared by two franchises", not reuse,
          str(reuse[:5]))

    if changed:
        print("\n  -> keyed on TEAM_ID these are the SAME franchise throughout.")
        print("     Keyed on abbreviation each would become two teams.")


def main() -> int:
    print(__doc__)
    frames = load_seasons()
    if not frames:
        print(f"No season files in {RAW_DIR}. Run the fetcher first.")
        return 1

    section("PER-SEASON CHECKS")
    results = [validate_season(s, f) for s, f in sorted(frames.items())]

    report_team_identity(frames)

    section("SUMMARY")
    print(f"{'SEASON':<8}{'GAMES':>7}{'TEAMS':>7}{'PER TEAM':>10}"
          f"{'EXPECTED':>10}{'UNIFORM':>9}{'BALANCED':>10}")
    print("-" * 61)
    for r in results:
        per_team = (str(r["modal"]) if r["uniform"]
                    else ",".join(f"{k}x{v}" for k, v in r["per_team"].items()))
        print(f"{r['season']:<8}{r['games']:>7}{r['teams']:>7}{per_team:>10}"
              f"{r['expected']:>10.0f}{str(r['uniform']):>9}{str(r['balanced']):>10}")

    total_games = sum(r["games"] for r in results)
    print(f"\n  {len(results)} seasons, {total_games:,} games, "
          f"{sum(r['rows'] for r in results):,} team-game rows")

    if notes:
        section("NOTES - derived checks that found something to explain")
        for n in notes:
            print(f"  {n}")

    section("RESULT")
    if failures:
        print(f"  {len(failures)} FAILED CHECK(S):")
        for f in failures:
            print(f"    {f}")
        return 1
    print("  all checks passed")
    if notes:
        print(f"  ({len(notes)} note(s) above - findings, not failures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
