"""THROWAWAY PROBE: can the WNBA be built from nba_api?

Answers the five feasibility questions and nothing else. Writes no data, trains
nothing, and imports nothing from the NBA pipeline except RateLimiter - the NBA
pipeline must not be touched before opening night.

Hypothesis: nba_api serves WNBA through the same endpoints, selected by
league_id '10' where the NBA is '00'. If so this is the lowest-risk new league
by a wide margin.
"""

import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE / "ingestion"))
from fetch_player_boxscores import RateLimiter  # noqa: E402

from nba_api.stats.endpoints import leaguegamefinder  # noqa: E402

WNBA_LEAGUE_ID = "10"
REQUESTS_PER_SECOND = 1.0
TIMEOUT = 60

limiter = RateLimiter(REQUESTS_PER_SECOND)


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def fetch_season(season, season_type="Regular Season"):
    """One LeagueGameFinder pull. Returns (frame, seconds) or (None, error)."""
    limiter.acquire()
    started = time.monotonic()
    try:
        finder = leaguegamefinder.LeagueGameFinder(
            league_id_nullable=WNBA_LEAGUE_ID,
            season_nullable=season,
            season_type_nullable=season_type,
            timeout=TIMEOUT,
        )
        frame = finder.get_data_frames()[0]
        return frame, time.monotonic() - started
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def probe_season_format():
    """Which season string does the WNBA accept?

    The NBA uses '2025-26' because its season spans two calendar years. A WNBA
    season sits inside one. Assuming the NBA format would be a quiet bug, so
    this asks rather than assumes.
    """
    section("1. SEASON LABEL FORMAT - asked, not assumed")
    results = {}
    for candidate in ("2024", "2024-25"):
        frame, meta = fetch_season(candidate)
        if frame is None:
            print(f"  {candidate:<9} ERROR  {meta}")
            results[candidate] = 0
            continue
        print(f"  {candidate:<9} {len(frame):>5} rows   ({meta:.2f}s)")
        results[candidate] = len(frame)

    working = [k for k, v in results.items() if v > 0]
    print(f"\n  accepted: {working or 'NONE'}")
    if len(working) == 1:
        print(f"  -> WNBA seasons are labelled '{working[0]}', a single calendar year.")
        print("     fetch_games.seasons_through()'s October boundary does NOT apply.")
    return working[0] if working else None


def completeness(frame, season):
    """Structural check: is this season internally whole?

    Derived from the data rather than from a remembered schedule length. Three
    independent properties, any of which a partial pull would break:
      - every GAME_ID appears exactly twice (one row per team)
      - every team plays the same number of games
      - total games == teams * games_per_team / 2
    """
    teams = frame["TEAM_ID"].nunique()
    per_game = frame.groupby("GAME_ID").size()
    per_team = frame.groupby("TEAM_ID").size()
    counts = Counter(per_team)
    modal_games, modal_teams = counts.most_common(1)[0]

    games = frame["GAME_ID"].nunique()
    expected = teams * modal_games / 2

    print(f"  teams                     : {teams}")
    print(f"  games (distinct GAME_ID)  : {games:,}")
    print(f"  rows                      : {len(frame):,}")
    print(f"  GAME_IDs with != 2 rows   : {int((per_game != 2).sum())}")
    print(f"  games per team            : {dict(sorted(counts.items()))}")
    print(f"  expected = {teams} x {modal_games} / 2 = {expected:,.0f}"
          f"   actual {games:,}   "
          f"{'MATCH' if abs(expected - games) < 0.5 else 'MISMATCH'}")

    uniform = len(counts) == 1
    if not uniform:
        print(f"  NOTE: {modal_teams} of {teams} teams played {modal_games}; "
              "an uneven schedule is normal in some WNBA seasons.")
    return {
        "season": season, "teams": teams, "games": games,
        "paired": int((per_game != 2).sum()) == 0,
        "balanced": abs(expected - games) < 0.5,
        "uniform": uniform,
        "games_per_team": modal_games,
    }


FIELDS_NEEDED = {
    "date": "GAME_DATE",
    "home/away": "MATCHUP",
    "final score": "PTS",
    "points": "PTS",
    "rebounds": "REB",
    "assists": "AST",
}


def report_fields(frame):
    section("3. FIELDS - the minimum for the team-level markets")
    for label, column in FIELDS_NEEDED.items():
        present = column in frame.columns
        nulls = int(frame[column].isna().sum()) if present else "-"
        print(f"  {label:<14} {column:<12} "
              f"{'PRESENT' if present else 'MISSING':<8} nulls {nulls}")

    print("\n  home/away derivation, as the NBA pipeline does it:")
    if "MATCHUP" in frame.columns:
        home = frame["MATCHUP"].str.contains("vs.", regex=False)
        print(f"    'vs.' rows {int(home.sum()):,}   "
              f"'@' rows {int((~home).sum()):,}   "
              f"{'balanced' if home.sum() == (~home).sum() else 'UNBALANCED'}")
    print(f"\n  all columns ({len(frame.columns)}): {list(frame.columns)}")


def probe_box_score(game_id):
    section("4. TEAM BOX SCORE for one game")
    from nba_api.stats.endpoints import boxscoretraditionalv3
    limiter.acquire()
    try:
        box = boxscoretraditionalv3.BoxScoreTraditionalV3(
            game_id=str(game_id).zfill(10), timeout=TIMEOUT)
        frames = box.get_data_frames()
        print(f"  game {str(game_id).zfill(10)}: {len(frames)} frame(s)")
        for i, f in enumerate(frames):
            print(f"    frame {i}: {len(f)} rows, {len(f.columns)} cols")
        team_frame = next((f for f in frames if len(f) == 2), None)
        if team_frame is not None:
            print(f"  team totals frame found: {list(team_frame.columns)[:14]}")
        return True
    except Exception as exc:
        print(f"  ERROR {type(exc).__name__}: {exc}")
        return False


def probe_schedule():
    section("5. UPCOMING FIXTURES - can the browse view be populated?")
    from nba_api.stats.endpoints import scheduleleaguev2
    for season in ("2026", "2027"):
        limiter.acquire()
        try:
            sched = scheduleleaguev2.ScheduleLeagueV2(
                league_id=WNBA_LEAGUE_ID, season=season, timeout=TIMEOUT)
            frame = sched.get_data_frames()[0]
            print(f"  season {season}: {len(frame):,} rows")
            if len(frame):
                cols = [c for c in frame.columns
                        if c.lower() in ("gamedate", "gamestatus", "gameid")]
                print(f"    {cols} -> "
                      f"{frame[cols].head(2).to_dict('records') if cols else 'n/a'}")
        except Exception as exc:
            print(f"  season {season}: ERROR {type(exc).__name__}: {exc}")


def main():
    print(__doc__)
    print("SOURCE: stats.nba.com via nba_api - the same source and the same")
    print("library the NBA pipeline already uses, at the same 1 req/s ceiling.")
    print("No new terms question: this is an additional league_id on an")
    print("endpoint this project already calls.\n")

    season_format = probe_season_format()
    if season_format is None:
        print("\nVERDICT: NOT OBTAINABLE - no season label produced data.")
        return 1

    # Three seasons spread across the range, plus the bubble year by name.
    samples = ["2015", "2020", "2026"]
    section("2. HISTORICAL COVERAGE AND COMPLETENESS")
    results, frames = [], {}
    for season in samples:
        print(f"\n--- season {season} ---")
        frame, meta = fetch_season(season)
        if frame is None or not len(frame):
            print(f"  no data ({meta})")
            results.append({"season": season, "games": 0})
            continue
        frames[season] = frame
        results.append(completeness(frame, season))

    if frames:
        newest = frames[max(frames)]
        report_fields(newest)
        probe_box_score(newest.iloc[0]["GAME_ID"])

    probe_schedule()

    section("HOW FAR BACK DOES IT GO?")
    for season in ("2000", "2005", "2010"):
        frame, meta = fetch_season(season)
        n = 0 if frame is None else frame["GAME_ID"].nunique()
        print(f"  {season}: {n:,} games" if frame is not None
              else f"  {season}: ERROR {meta}")

    section("SUMMARY")
    for r in results:
        if r.get("games"):
            print(f"  {r['season']}: {r['games']:>3} games, {r['teams']} teams, "
                  f"{r['games_per_team']}/team, "
                  f"paired={r['paired']} balanced={r['balanced']} "
                  f"uniform={r['uniform']}")
        else:
            print(f"  {r['season']}: NO DATA")
    return 0


if __name__ == "__main__":
    sys.exit(main())
