"""THROWAWAY PROBE: can the NBA G League be built?

Answers five questions and nothing else: access, historical coverage,
completeness, fields, upcoming fixtures. Writes nothing, trains nothing,
imports nothing from the NBA or WNBA pipelines except RateLimiter.

THE TERMS QUESTION IS ALREADY SETTLED, AND THAT IS THE HYPOTHESIS UNDER TEST.
If the G League is served by stats.nba.com under a league id, it is the same
source this project already calls for the NBA (league 00) and the WNBA
(league 10) through the same library. No new party holds the data, so no new
terms apply. That claim is only worth anything if league id 20 genuinely
answers, which is the first thing checked.

Run:  python data-pipeline/probes/probe_gleague.py
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE / "ingestion"))
from fetch_player_boxscores import RateLimiter  # noqa: E402

from nba_api.stats.endpoints import (  # noqa: E402
    boxscoretraditionalv3,
    leaguegamefinder,
    scheduleleaguev2,
)

G_LEAGUE_ID = "20"
TIMEOUT = 60
GAME_ID_WIDTH = 10

# Spread across the range rather than consecutive: one near the project's
# start, both disrupted seasons, and the most recent completed one.
SAMPLE_SEASONS = ["2015-16", "2019-20", "2020-21", "2025-26"]

# How far back to look, probed coarsely before anything else.
COVERAGE_PROBES = ["2001-02", "2005-06", "2010-11", "2015-16"]

TYPE_DIGIT_MEANING = {
    "1": "preseason",
    "2": "regular season",
    "3": "all-star",
    "4": "playoffs",
    "5": "play-in",
    "6": "cup / tournament final",
}

limiter = RateLimiter(requests_per_second=1.0)


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def fetch(season):
    """One season of team-level rows, or None if the season returns nothing."""
    limiter.acquire()
    try:
        frame = leaguegamefinder.LeagueGameFinder(
            league_id_nullable=G_LEAGUE_ID,
            season_nullable=season,
            season_type_nullable="Regular Season",
            timeout=TIMEOUT,
        ).get_data_frames()[0]
    except Exception as error:
        print(f"  {season}: {type(error).__name__}: {str(error)[:90]}")
        return None
    return frame


def fetch_all_types(season):
    """Every competition type in one season, to see what the digit encodes."""
    limiter.acquire()
    try:
        return leaguegamefinder.LeagueGameFinder(
            league_id_nullable=G_LEAGUE_ID,
            season_nullable=season,
            timeout=TIMEOUT,
        ).get_data_frames()[0]
    except Exception as error:
        print(f"  {season} (all types): {type(error).__name__}: {str(error)[:80]}")
        return None


def digits_of(frame):
    padded = frame["GAME_ID"].astype(str).str.zfill(GAME_ID_WIDTH)
    return padded.str[2]


def access_and_coverage():
    section("1 + 2. ACCESS AND HISTORICAL COVERAGE")
    print("""Probing how far back the G League is served. Coarse first - the point is
whether it matches the NBA's eleven seasons or stops somewhere else, not to
enumerate every season.
""")
    found = {}
    for season in COVERAGE_PROBES:
        frame = fetch(season)
        if frame is None or frame.empty:
            print(f"  {season:<9} 0 rows")
            continue
        found[season] = frame
        print(f"  {season:<9} {len(frame):>6} rows  "
              f"{frame['GAME_ID'].nunique():>5} games  "
              f"{frame['TEAM_ID'].nunique():>3} teams  "
              f"{frame['GAME_DATE'].min()} .. {frame['GAME_DATE'].max()}")
    return found


def completeness(seasons):
    section("3. COMPLETENESS, AGAINST COUNTS DERIVED FROM THE DATA")
    print("""games_per_team is DERIVED, never remembered: the G League schedule length
has changed repeatedly and two sampled seasons were disrupted. If N teams are
each short by S games, exactly N*S/2 games must be absent for the shortfall to
reconcile - the same rule that passed the WNBA's 2020 bubble and caught its
2018 forfeit without a special case.
""")
    for season, frame in seasons.items():
        games = frame["GAME_ID"].nunique()
        rows = len(frame)
        teams = frame["TEAM_ID"].nunique()

        per_team = frame.groupby("TEAM_ID")["GAME_ID"].nunique()
        modal = per_team.mode().iloc[0] if not per_team.empty else 0
        expected = teams * modal / 2

        pairs = frame.groupby("GAME_ID").size()
        unpaired = int((pairs != 2).sum())

        short = per_team[per_team < modal]
        shortfall = int((modal - short).sum())
        reconciles = (shortfall / 2) == (expected - games) if expected else False

        print(f"  {season}")
        print(f"    rows {rows:>6}  games {games:>5}  teams {teams:>3}  "
              f"rows==2*games {rows == 2 * games}")
        print(f"    games per team: modal {modal}, "
              f"min {per_team.min()}, max {per_team.max()}")
        print(f"    expected at modal: {expected:.0f}  "
              f"actual {games}  difference {expected - games:.0f}")
        print(f"    games with != 2 rows: {unpaired}")
        if expected != games:
            print(f"    {len(short)} team(s) short, total shortfall {shortfall} "
                  f"team-games -> reconciles: {reconciles}")
        print()


def competition_types(season):
    section("4a. COMPETITION TYPES WITHIN ONE SEASON")
    print(f"""The G League has run a Showcase Cup / Tip-Off Tournament alongside its
regular season. The NBA and WNBA models are regular-season only, so a separate
competition mixed in would change what every feature means. Fetching {season}
with NO season-type filter to see what the id's third digit encodes.
""")
    frame = fetch_all_types(season)
    if frame is None or frame.empty:
        print("  nothing returned, so this cannot be answered from here")
        return

    frame = frame.assign(digit=digits_of(frame))
    counts = frame.groupby("digit")["GAME_ID"].nunique().sort_index()

    print(f"  {'digit':<7}{'meaning':<28}{'games':>7}")
    print("  " + "-" * 42)
    for digit, n in counts.items():
        print(f"  {digit:<7}{TYPE_DIGIT_MEANING.get(digit, 'UNKNOWN'):<28}{n:>7}")

    regular = fetch(season)
    if regular is not None and not regular.empty:
        filtered = regular["GAME_ID"].nunique()
        print(f"\n  season_type_nullable='Regular Season' returns {filtered} games")
        digit_2 = int(counts.get("2", 0))
        print(f"  digit == '2' returns {digit_2} games")
        print(f"  the two agree: {filtered == digit_2}")
        if filtered != digit_2:
            print("  -> they DISAGREE, so one of the two filters is admitting"
                  " something the other excludes")


def franchise_identity(seasons):
    section("4b. FRANCHISE IDENTITY ACROSS SEASONS")
    print("""Affiliates rename, relocate and change parent club often. The question is
whether a persisting franchise keeps a stable TEAM_ID through a rename, as the
WNBA's three did - because keying on abbreviation would split one franchise
into two and corrupt every rolling window and Elo rating crossing the change.
""")
    by_id = defaultdict(set)
    seasons_of = defaultdict(set)
    for season, frame in seasons.items():
        for _, row in frame[["TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME"]].iterrows():
            by_id[row["TEAM_ID"]].add((row["TEAM_ABBREVIATION"], row["TEAM_NAME"]))
            seasons_of[row["TEAM_ID"]].add(season)

    print(f"  distinct TEAM_IDs across the sample: {len(by_id)}")

    renamed = {tid: v for tid, v in by_id.items() if len(v) > 1}
    print(f"  ids carrying more than one (abbr, name): {len(renamed)}")
    for tid, variants in sorted(renamed.items())[:12]:
        shown = ", ".join(f"{a}/{n}" for a, n in sorted(variants))
        print(f"    {tid}  {shown[:88]}")

    abbr_to_ids = defaultdict(set)
    for tid, variants in by_id.items():
        for abbr, _ in variants:
            abbr_to_ids[abbr].add(tid)
    shared = {a: ids for a, ids in abbr_to_ids.items() if len(ids) > 1}
    print(f"\n  abbreviations used by more than one TEAM_ID: {len(shared)}")
    for abbr, ids in sorted(shared.items())[:10]:
        print(f"    {abbr}: {sorted(ids)}")

    persisting = [t for t, s in seasons_of.items() if len(s) == len(seasons)]
    print(f"\n  ids present in ALL {len(seasons)} sampled seasons: {len(persisting)}")


def fields_and_churn(season, frame):
    section("5. FIELDS, PER GAME")
    wanted = {
        "date": "GAME_DATE",
        "teams": "TEAM_ID",
        "home/away": "MATCHUP",
        "final score": "PTS",
        "win/loss": "WL",
        "rebounds": "REB",
        "assists": "AST",
        "plus/minus": "PLUS_MINUS",
    }
    print(f"  checked on {season}, {len(frame)} rows\n")
    print(f"  {'field':<14}{'column':<18}{'present':<10}{'non-null':>10}")
    print("  " + "-" * 54)
    for label, column in wanted.items():
        present = column in frame.columns
        filled = f"{frame[column].notna().mean() * 100:.1f}%" if present else "-"
        print(f"  {label:<14}{column:<18}{str(present):<10}{filled:>10}")

    if "MATCHUP" in frame.columns:
        home = frame["MATCHUP"].str.contains("vs.").sum()
        print(f"\n  MATCHUP marks home with 'vs.': {home} of {len(frame)} rows "
              f"({home / len(frame) * 100:.1f}%, expect ~50%)")

    section("5b. ROSTER CHURN, AS A NOTE")
    print("""Call-ups and assignments move team composition mid-season more than in any
other league here. Measured cheaply on one team's season: of the players who
appeared for it at all, how many appeared in only a handful of games.
""")
    team_id = int(frame["TEAM_ID"].iloc[0])
    team_games = sorted(frame[frame["TEAM_ID"] == team_id]["GAME_ID"].unique())
    sample = team_games[:12]
    print(f"  team {team_id}, first {len(sample)} games of {season}")

    appearances = Counter()
    for game_id in sample:
        limiter.acquire()
        try:
            box = boxscoretraditionalv3.BoxScoreTraditionalV3(
                game_id=str(game_id).zfill(GAME_ID_WIDTH), timeout=TIMEOUT
            ).get_data_frames()[0]
        except Exception as error:
            print(f"    {game_id}: {type(error).__name__}: {str(error)[:70]}")
            continue
        side = box[box["teamId"] == team_id]
        for pid in side["personId"].unique():
            appearances[pid] += 1

    if appearances:
        n = len(sample)
        total = len(appearances)
        once_or_twice = sum(1 for c in appearances.values() if c <= 2)
        most = sum(1 for c in appearances.values() if c >= n - 2)
        print(f"\n    distinct players used over {n} games : {total}")
        print(f"    appeared in <= 2 of them            : {once_or_twice} "
              f"({once_or_twice / total * 100:.0f}%)")
        print(f"    appeared in >= {n - 2} of them             : {most} "
              f"({most / total * 100:.0f}%)")
        print("\n    A high transient share is the churn this league is known"
              " for, and it\n    sets expectations rather than blocking"
              " anything: team-level features\n    are unaffected, player"
              " props would be much harder here than in the NBA.")


def upcoming():
    section("6. UPCOMING FIXTURES")
    print("""History without a forward schedule can be modelled but not served. The NBA
and WNBA both use ScheduleLeagueV2; the question is whether it answers for
league 20 and in what season format.
""")
    for season in ("2026-27", "2026"):
        limiter.acquire()
        try:
            frames = scheduleleaguev2.ScheduleLeagueV2(
                league_id=G_LEAGUE_ID, season=season, timeout=TIMEOUT
            ).get_data_frames()
            frame = frames[0] if frames else pd.DataFrame()
        except Exception as error:
            print(f"  season {season!r}: {type(error).__name__}: "
                  f"{str(error)[:80]}")
            continue

        if frame.empty:
            print(f"  season {season!r}: 0 rows")
            continue

        frame = frame.assign(
            digit=frame["gameId"].astype(str).str.zfill(GAME_ID_WIDTH).str[2])
        unplayed = frame[frame["gameStatus"] == 1]
        print(f"  season {season!r}: {len(frame)} rows, "
              f"{len(unplayed)} unplayed")
        print(f"    dates {frame['gameDate'].min()} .. {frame['gameDate'].max()}")
        print(f"    type digits (all)      : "
              f"{frame['digit'].value_counts().to_dict()}")
        if len(unplayed):
            print(f"    type digits (unplayed) : "
                  f"{unplayed['digit'].value_counts().to_dict()}")


def main() -> int:
    print(__doc__)

    seasons = access_and_coverage()
    if not seasons:
        section("VERDICT")
        print("  NOT OBTAINABLE from this source: league id 20 returns nothing,")
        print("  so the hypothesis that it rides on stats.nba.com is false and")
        print("  a separate source with its own terms would be needed.")
        return 1

    sampled = {}
    for season in SAMPLE_SEASONS:
        if season in seasons:
            sampled[season] = seasons[season]
        else:
            frame = fetch(season)
            if frame is not None and not frame.empty:
                sampled[season] = frame
            else:
                print(f"\n  note: {season} returned nothing")

    completeness(sampled)
    competition_types(SAMPLE_SEASONS[-1])
    franchise_identity(sampled)

    newest = SAMPLE_SEASONS[-1]
    if newest in sampled:
        fields_and_churn(newest, sampled[newest])

    upcoming()

    section("WHAT THIS PROBE DOES NOT ESTABLISH")
    print("""  A handful of requests at 1 req/s is the friendliest possible load profile.
  This shows the data is served and complete for the sampled seasons; it does
  not show a full historical pull would go unthrottled. Same honest limit the
  shot-chart probe recorded.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
