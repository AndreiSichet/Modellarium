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
sys.path.insert(0, str(HERE))

import nfl_franchises as F              # noqa: E402
import parse_nfl_wikitext as P          # noqa: E402
import validate_nfl_games as V          # noqa: E402

PROCESSED = NFL / "data" / "processed"


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="*", type=int)
    args = parser.parse_args()
    seasons = args.seasons or F.seasons_through()

    rows, articles = collect(seasons)
    modal = home_venues(rows)
    # {(season, venue): the franchise whose home venue it is}
    owners = {(season, venue): fid
              for (season, fid), venue in modal.items()}
    games = V.group_games(rows)

    team_records, fixture_records, quarter_records = [], [], []
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
    print(f"  nfl_quarter_scores.csv     {n4:>6} rows")
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
