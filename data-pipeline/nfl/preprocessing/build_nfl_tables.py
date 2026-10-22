"""Build the four NFL processed tables. Validate first; this writes.

Outputs, all gitignored (see data-pipeline/nfl/DATA_LICENSE.md):
    nfl_games_final.csv         two rows per played game, one per team
    nfl_fixtures.csv            unplayed regular-season games
    nfl_franchise_identity.csv  franchise id, season, link target, name, abbr
    nfl_quarter_scores.csv      per game and side, Q1-Q4 and overtime

Run from the project root:
    python data-pipeline/nfl/preprocessing/build_nfl_tables.py
"""
import argparse
import collections
import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NFL = HERE.parent
PIPELINE = NFL.parent          # data-pipeline/, so the NFL's data sits
                               # under data/nfl/ like the other three
                               # leagues' - which is what lets one
                               # DATA_DIR point at either the repo or a
                               # served snapshot with no per-league
                               # special case (served_data.py)
sys.path.insert(0, str(HERE))

import nfl_franchises as F              # noqa: E402
import parse_nfl_wikitext as P          # noqa: E402
import validate_nfl_games as V          # noqa: E402

PROCESSED = PIPELINE / "data" / "nfl" / "processed"


def abbrev(fid, season):
    for target, (f_id, abbr, _name) in F.franchise_table(season).items():
        if f_id == fid:
            return abbr
    raise KeyError(f"no abbreviation for {fid} in {season}")


def target_of(fid, season):
    for target, (f_id, _abbr, _name) in F.franchise_table(season).items():
        if f_id == fid:
            return target
    raise KeyError(fid)


def game_id(season, week, away_fid, home_fid):
    """season_week_away_home, with the abbreviations of that season.

    The franchise_id columns are the real keys; this is a readable synthetic
    id, and the recap slug is only ever a cross-check.
    """
    week_token = re.sub(r"\W+", "", str(week)) or "0"
    return (f"{season}_{week_token}_{abbrev(away_fid, season)}_"
            f"{abbrev(home_fid, season)}")


# --------------------------------------------------------- neutral sites

def home_venues(rows):
    """{(season, fid): modal venue} from each team's own home rows."""
    seen = collections.defaultdict(collections.Counter)
    for row in rows:
        if row["away"] or not row["venue"]:
            continue
        seen[(row["season"], row["team_fid"])][row["venue"]] += 1
    return {key: counts.most_common(1)[0][0] for key, counts in seen.items()}


def neutral_signals(row, is_home, modal, owners):
    """Which independent signals call this a neutral or relocated venue.

    Three signals, and the third needs corroboration. A home game at a venue
    that is not the team's usual one is either a relocation or a stadium
    RENAME, and the two look identical: Seattle's 2020 home games read
    `CenturyLink Field` four times and `Lumen Field` four times for one
    unchanged stadium. Measured discriminator - a relocation carries an
    explanatory footnote on the venue cell and a rename does not:

        San Francisco 2020, State Farm Stadium x3   note present
        Seattle 2020, Lumen Field x4                no note

    so a differing venue without a note is reported as a probable rename
    rather than flagged as neutral.
    """
    signals = []
    if row["marker_neutral"]:
        signals.append("vs-marker")
    if row["venue"] and P.INTERNATIONAL_CITY.search(row["venue"]):
        signals.append("international-venue")
    usual = modal.get((row["season"], row["team_fid"]))
    if is_home and row["venue"] and usual and row["venue"] != usual:
        owner = owners.get((row["season"], row["venue"]))
        if owner is not None and owner != row["team_fid"]:
            # A stadium that is ANOTHER franchise's home venue that season is
            # borrowed, so this is a relocation. This is the primary test
            # because it needs no footnote: the 2014 Buffalo game moved to
            # Ford Field (Detroit's) carries no note at all, while Seattle's
            # renamed Lumen Field is nobody else's home venue.
            signals.append("borrowed-stadium")
        elif row["venue_has_note"]:
            signals.append("relocated-with-footnote")
        else:
            signals.append("venue-differs-no-note")
    return signals


# ------------------------------------------------------------------ build

def collect(seasons):
    all_rows, all_articles = [], {}
    for season in seasons:
        rows, articles = P.parse_season(season)
        all_rows += rows
        all_articles[season] = articles
    return all_rows, all_articles


def write_csv(path, fieldnames, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    with open(partial, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    partial.replace(path)
    return len(records)


# --------------------------------------------------------------------------
# QUARANTINE, FOR THE OPEN SEASON ONLY
#
# Two copies of a game that contradict each other currently fail the NFL
# build. For a COMPLETED season that is right: the articles have settled and a
# disagreement is a real defect. For the OPEN season it is the wrong
# instrument - the source is Wikipedia, anyone can edit it, and one half-typed
# score should not cost the whole league a day of fresh data.
#
# So in the open season a disagreeing game is set aside with both copies and a
# reason, the rest of the day's data lands, and the game is retried on the
# next build. This is the rule the NHL's phase 1 already has, and the CSV
# columns are deliberately the same - `nhl_quarantine.csv` is
# season,date,copy_a,copy_b,reason,left_quarantine - so the two read alike.
#
# WHAT IS NOT SHARED: the NHL keys a game on ({both teams}, date +/- 1 day)
# because its two copies can disagree about the date by a day; the NFL keys on
# V.game_key, (date, team pair), because a flexed fixture is the only date
# ambiguity and it is unplayed. Forcing one module over both would mean
# forcing one key over both, so the shape is reused and the code is not.
#
# NO REVERSE GUARD THAT FAILS, AND THAT IS A DELIBERATE DIFFERENCE from the
# NFL's authored KNOWN_UNUSABLE_GAMES list, where a listed game that stops
# failing IS fatal (check_known_list enforces it). This file is GENERATED and
# rebuilt every run, so a game leaving it is ordinary movement rather than a
# stale hand-written entry. The movement is reported instead.
QUARANTINE_COLUMNS = ["season", "date", "copy_a", "copy_b", "reason",
                      "left_quarantine"]


def copies_agree(home, away) -> tuple:
    """(True, "") if the two copies of a played game tell the same story."""
    if (home["points_for"], home["points_against"]) != \
            (away["points_against"], away["points_for"]):
        return False, (f"scores disagree: {home['points_for']}-"
                       f"{home['points_against']} against "
                       f"{away['points_against']}-{away['points_for']}")
    if home["date"] != away["date"]:
        return False, (f"dates disagree: {home['date'].isoformat()} against "
                       f"{away['date'].isoformat()}")
    return True, ""


def describe_copy(row, is_home) -> str:
    """One copy of a game, as the quarantine file records it.

    `team_target` is the team-season article this copy was parsed from, which
    is what makes a quarantine row actionable: it names the page to look at.
    Same intent as the NHL's "<article> says home X, 2-3, REG".
    """
    return (f"{row['team_target']} says {'home' if is_home else 'away'} "
            f"{row['team_fid']}, {row['points_for']}-"
            f"{row['points_against']}, {row['wl']}, "
            f"{row['date'].isoformat()}")


def previous_quarantine(path) -> dict:
    """{game_id: row} from the last build, so movement can be reported."""
    if not path.is_file():
        return {}
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        return {row["date"] + "|" + row["season"]: row
                for row in csv.DictReader(handle)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="*", type=int)
    args = parser.parse_args()
    seasons = args.seasons or F.seasons_through()

    rows, articles = collect(seasons)
    # THE CURRENT SEASON ONLY, not F.open_seasons().
    #
    # `open_seasons()` is the FETCHER's question - which seasons to re-read
    # every run - and it answers "the current one and the one before", because
    # late corrections happen. Quarantine asks a different question: which
    # season is still being played, and therefore still being edited by
    # hand hours after a game. The 2025 season ended in February 2026; a
    # disagreement in it is a settled defect and must still fail, which is
    # what "completed seasons stay strict" means. Using open_seasons() here
    # would have quietly relaxed strictness for a finished season - caught by
    # the verifier asserting that 2025 is NOT quarantine-eligible.
    open_now = {F.current_season()}
    quarantine_path = PROCESSED / "nfl_quarantine.csv"
    was_quarantined = previous_quarantine(quarantine_path)
    modal = home_venues(rows)
    # {(season, venue): the franchise whose home venue it is}
    owners = {(season, venue): fid
              for (season, fid), venue in modal.items()}
    games = V.group_games(rows)

    team_records, fixture_records, quarter_records = [], [], []
    quarantine_records, left_quarantine = [], []
    signal_counts = collections.Counter()
    neutral_games = []
    renames = []
    bad_boxes = []
    unresolved = []

    for key, pair in games.items():
        season = pair[0]["season"]
        home, away, how = P.resolve_pair(pair)
        if home is None:
            unresolved.append((key, how))
            continue
        status = home["status"]
        gid = game_id(season, home["week"], away["team_fid"],
                      home["team_fid"])

        signals = sorted(set(neutral_signals(home, True, modal, owners)
                             + neutral_signals(away, False, modal, owners)))
        for signal in signals:
            signal_counts[signal] += 1
        # A differing venue with no footnote is a probable stadium rename, not
        # a neutral site, so it is reported and NOT flagged.
        neutral = bool(set(signals) - {"venue-differs-no-note"})
        if neutral:
            neutral_games.append((gid, home["venue"], signals))
        if set(signals) == {"venue-differs-no-note"}:
            renames.append((gid, home["venue"],
                            modal.get((season, home["team_fid"]))))

        if status == "played":
            agreed, why = copies_agree(home, away)
            marker = home["date"].isoformat() + "|" + str(season)
            if agreed and marker in was_quarantined:
                # LEFT QUARANTINE. Reported, not fatal: this file is rebuilt
                # every run, so a game ceasing to disagree is the system
                # working rather than a stale entry someone forgot.
                left_quarantine.append((gid, home["date"].isoformat()))
            if not agreed and season in open_now:
                quarantine_records.append(dict(
                    season=season, date=home["date"].isoformat(),
                    copy_a=describe_copy(home, True),
                    copy_b=describe_copy(away, False),
                    reason=why, left_quarantine=0))
                continue
            for row, other in ((home, away), (away, home)):
                margin = row["points_for"] - row["points_against"]
                team_records.append(dict(
                    game_id=gid, season=season, week=row["week"],
                    date=row["date"].isoformat(),
                    franchise_id=row["team_fid"],
                    team=abbrev(row["team_fid"], season),
                    opponent_franchise_id=other["team_fid"],
                    opponent=abbrev(other["team_fid"], season),
                    is_home=int(row is home),
                    neutral_site=int(neutral),
                    points_for=row["points_for"],
                    points_against=row["points_against"],
                    result=row["wl"], margin=margin,
                    is_tie=int(row["wl"] == "T"),
                    venue=row["venue"]))
        elif status == "fixture":
            fixture_records.append(dict(
                game_id=gid, season=season, week=home["week"],
                date=home["date"].isoformat(),
                kickoff_local=home["time_text"],
                kickoff_zone_as_written=home["zone_text"],
                away_franchise_id=away["team_fid"],
                away=abbrev(away["team_fid"], season),
                home_franchise_id=home["team_fid"],
                home=abbrev(home["team_fid"], season),
                neutral_site=int(neutral),
                flex_two_dates=int(home["flex"] or away["flex"]),
                venue=home["venue"]))

    # quarter scores, from the boxes the validator already reconciles
    for season, by_target in articles.items():
        boxes = V.index_boxes(by_target)
        season_rows = [r for r in rows if r["season"] == season]
        for key, pair in V.group_games(season_rows).items():
            home, away, how = P.resolve_pair(pair)
            if home is None or home["status"] != "played":
                continue
            box = V.match_box((home, away), boxes, season)
            if box is None:
                continue
            gid = game_id(season, home["week"], away["team_fid"],
                          home["team_fid"])
            for side, row in (("H", home), ("R", away)):
                quarters, overtime, complete = P.box_periods(box, side)
                if not complete:
                    continue
                total = (sum(x for x in quarters if x is not None)
                         + (overtime or 0))
                if total != row["points_for"]:
                    # A breakdown that contradicts the final score is not
                    # shipped. The validator lists the one known case in
                    # KNOWN_BAD_BOXES; writing it anyway would hand a future
                    # quarter/half model a wrong row.
                    bad_boxes.append((gid, row["team_fid"], total,
                                      row["points_for"]))
                    continue
                quarter_records.append(dict(
                    game_id=gid, season=season,
                    date=row["date"].isoformat(),
                    franchise_id=row["team_fid"],
                    team=abbrev(row["team_fid"], season),
                    side="home" if side == "H" else "road",
                    q1=quarters[0], q2=quarters[1], q3=quarters[2],
                    q4=quarters[3],
                    ot="" if overtime is None else overtime,
                    quarters_total=sum(q for q in quarters if q is not None)
                    + (overtime or 0),
                    final=row["points_for"]))

    identity_records = []
    for season in seasons:
        for target, (fid, abbr, name) in sorted(
                F.franchise_table(season).items(), key=lambda kv: kv[1][0]):
            identity_records.append(dict(
                franchise_id=fid, season=season, link_target=target,
                display_name=name, abbreviation=abbr))

    print("NFL tables")
    print(f"  seasons {seasons[0]}..{seasons[-1]}")
    if unresolved:
        print(f"  REFUSING to write: {len(unresolved)} game(s) whose sides "
              f"cannot be resolved: {unresolved[:3]}")
        return 1
    n1 = write_csv(PROCESSED / "nfl_games_final.csv",
                   list(team_records[0]), team_records)
    n2 = write_csv(PROCESSED / "nfl_fixtures.csv",
                   list(fixture_records[0]) if fixture_records
                   else ["game_id"], fixture_records)
    n3 = write_csv(PROCESSED / "nfl_franchise_identity.csv",
                   list(identity_records[0]), identity_records)
    n4 = write_csv(PROCESSED / "nfl_quarter_scores.csv",
                   list(quarter_records[0]), quarter_records)
    print(f"  nfl_games_final.csv        {n1:>6} rows "
          f"({n1 // 2} games, two rows each)")
    print(f"  nfl_fixtures.csv           {n2:>6} rows")
    print(f"  nfl_franchise_identity.csv {n3:>6} rows")
    # ALWAYS WRITTEN, even empty, so "no quarantine file" and "nothing
    # quarantined" are different states rather than the same silence.
    n5 = write_csv(PROCESSED / "nfl_quarantine.csv", QUARANTINE_COLUMNS,
                   quarantine_records)
    print(f"  nfl_quarter_scores.csv     {n4:>6} rows")
    print(f"  nfl_quarantine.csv         {n5:>6} rows")
    print()
    print()
    print(f"  QUARANTINED (open season only) : {len(quarantine_records)}")
    for record in quarantine_records:
        print(f"      {record['season']} {record['date']}  {record['reason']}")
        print(f"        A: {record['copy_a']}")
        print(f"        B: {record['copy_b']}")
    if left_quarantine:
        print(f"  LEFT QUARANTINE since the last build : "
              f"{len(left_quarantine)}")
        for gid, date in left_quarantine:
            print(f"      {gid}  {date}  the two copies now agree")
    if quarantine_records:
        # A QUARANTINED GAME BLOCKS ITS TEAMS' NEXT FIXTURES, and that is
        # correct rather than a side effect: the NFL's prediction rule needs
        # both teams' previous games in history, and a quarantined game is
        # not in history. Reported so the count in /health has an explanation.
        blocked_fids = set()
        for record in quarantine_records:
            for copy in (record["copy_a"], record["copy_b"]):
                for token in copy.split():
                    if token.rstrip(",").isdigit() and len(token.rstrip(",")) == 10:
                        blocked_fids.add(token.rstrip(","))
        upcoming = [f for f in fixture_records
                    if str(f["home_franchise_id"]) in blocked_fids
                    or str(f["away_franchise_id"]) in blocked_fids]
        print(f"  fixtures blocked by quarantine : {len(upcoming)}")
        for f in sorted(upcoming, key=lambda r: (r["date"], r["game_id"]))[:8]:
            print(f"      {f['game_id']:<22} {f['date']}  "
                  f"{f['away']} @ {f['home']}")
    print()
    print(f"  neutral-site games flagged : {len(neutral_games)}")
    for signal, count in signal_counts.most_common():
        print(f"      by {signal:<26} {count}")
    print(f"  venue differs, no footnote, no other signal : {len(renames)}")
    print(f"      (a stadium RENAME, or an unlisted international city - "
          f"inspected, not flagged neutral)")
    for gid, venue, usual in renames:
        print(f"      {gid:<22} {venue!r} vs usual {usual!r}")
    print(f"  quarter rows dropped (box contradicts the final score): "
          f"{len(bad_boxes)}")
    for gid, fid, total, final in bad_boxes:
        print(f"      {gid} fid {fid}: quarters {total} vs final {final}")
    print(f"  game ids unique            : "
          f"{len({r['game_id'] for r in team_records})} for "
          f"{len(team_records) // 2} games")
    return 0


if __name__ == "__main__":
    sys.exit(main())
