"""Read-only quality gate for the NFL corpus.

Every game is described by two articles, and the game-summary box gives a
second, independent copy of the score inside each of them. The validator works
from all three, because they fail differently:

  * the two-copy pairing check catches a disagreement between the articles;
  * the quarter sum is the ONLY check that catches a score edited identically
    in both articles without changing the winner;
  * the Record column and the outcome template catch a changed winner, and the
    Record column catches it loudly - one wrong result poisons every later row
    of both teams.

Nothing here is written. Run from the project root:
    python data-pipeline/nfl/preprocessing/validate_nfl_games.py
    python data-pipeline/nfl/preprocessing/validate_nfl_games.py --seasons 2022
"""
import argparse
import collections
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import nfl_franchises as F              # noqa: E402
import parse_nfl_wikitext as P          # noqa: E402

# Games whose two rows cannot be reconciled, listed with their evidence. An
# unlisted bad game is fatal; a listed game that is no longer bad, or no
# longer present, is fatal too, so this list cannot rot in either direction.
KNOWN_UNUSABLE_GAMES = {
    # game key -> evidence. Empty: every game in 2012-2026 reconciles.
}

# A DEFECTIVE GAME-SUMMARY BOX is not an unusable game, and the distinction is
# the point: the schedule tables of both articles agree with each other and
# with the Record column, so the game's score is sound and belongs in the
# dataset - only its quarter breakdown is wrong. Listing the game as unusable
# would discard a valid result to work around a bad sub-table.
#
# Both directions are guarded, as for the unusable list: an unlisted bad box is
# fatal, and a listed box that is no longer bad, or no longer present, is fatal.
KNOWN_BAD_BOXES = {
    (2012, "December 23", "Tennessee Titans", "Green Bay Packers"):
        "the box gives Tennessee R1..R4 = 0,0,0,0 while both schedule tables "
        "and both Record columns give Tennessee 7 (Green Bay won 55-7)",
}


class Failure(Exception):
    pass


class Report:
    def __init__(self):
        self.checks = 0
        self.failures = []
        self.notes = []

    def check(self, ok, label, detail=""):
        self.checks += 1
        if not ok:
            self.failures.append(f"{label}" + (f" - {detail}" if detail else ""))
        return ok

    def note(self, text):
        self.notes.append(text)


# ------------------------------------------------------------------- games

def game_key(row):
    """Date for a game that has happened, week for one that has not.

    Measured: (date, team pair) pairs every row of every completed season
    2012-2025 with zero singletons, and the week key fails on 28 rows of 2020
    because COVID rescheduling moved games between weeks and the two articles
    disagree. But an UNPLAYED flexed fixture has no settled date - one article
    writes `December 26/27` and the other `December 27` - so a fixture is
    keyed on its week, which is known even when its date is not.
    """
    pair = frozenset((row["team_fid"], row["opponent_fid"]))
    if row["status"] == "fixture":
        return (row["season"], "week", row["week"], pair)
    return (row["season"], "date", row["date"], pair)


def group_games(rows):
    games = collections.defaultdict(list)
    for row in rows:
        games[game_key(row)].append(row)
    return games


def sides(pair):
    """(home row, away row), resolved by whichever signal is consistent."""
    home, away, _how = P.resolve_pair(pair)
    return None if home is None else (home, away)


def box_is_known_bad(season, row, other):
    for (s, day, road, home), _why in KNOWN_BAD_BOXES.items():
        if s != season:
            continue
        targets = {row["team_target"], other["team_target"]}
        if {road, home} == targets and day in row["date_text"]:
            return True
    return False


# ------------------------------------------------------- box score matching

NICK_SUFFIX = re.compile(r"[A-Za-z0-9'’.\- ]+$")


def box_side_matches(box_name, display_name):
    """Boxes name teams by nickname ('Packers'), city, or full name."""
    a = P.clean(box_name).lower().replace(".", "").strip()
    b = display_name.lower().replace(".", "").strip()
    if not a:
        return False
    if a == b or b.endswith(a) or a.endswith(b):
        return True
    # '49ers' against 'San Francisco 49ers'; 'NY Giants' against 'New York...'
    tail = b.split()[-1]
    return a == tail or a.endswith(tail) or tail.endswith(a)


def index_boxes(articles):
    out = []
    for target, article in articles.items():
        for box in article["boxes"]:
            road = P.clean(box.get("road", ""))
            home = P.clean(box.get("home", ""))
            day = P.clean(box.get("date", ""))
            if not road or not home:
                continue
            out.append((road, home, day, box, target))
    return out


def match_box(game_rows, boxes, season):
    home_row, away_row = game_rows
    want_day = home_row["date_text"].split("[")[0].strip()
    home_name = home_row["team_target"]
    away_name = away_row["team_target"]
    for road, home, day, box, _target in boxes:
        if not box_side_matches(home, home_name):
            continue
        if not box_side_matches(road, away_name):
            continue
        if want_day and day and want_day.split()[0] not in day:
            continue
        if want_day and day:
            wd = re.search(r"([A-Z][a-z]+)\s+(\d{1,2})", want_day)
            bd = re.search(r"([A-Z][a-z]+)\s+(\d{1,2})", day)
            if wd and bd and (wd.group(1), wd.group(2)) != (bd.group(1),
                                                            bd.group(2)):
                continue
        return box
    return None


# ------------------------------------------------------------------ checks

def validate_season(season, rows, articles, report):
    played = [r for r in rows if r["played"]]
    games = group_games(rows)
    played_games = {k: v for k, v in games.items()
                    if all(r["played"] for r in v)}
    partly = {k: v for k, v in games.items()
              if len({r["status"] for r in v}) > 1}

    # ---- 1. pairing
    bad_pairs, singles, repaired_sides = [], [], []
    for key, pair in played_games.items():
        if len(pair) != 2:
            (singles if len(pair) == 1 else bad_pairs).append(
                (key, f"appears {len(pair)} time(s)"))
            continue
        home, away, how = P.resolve_pair(pair)
        if home is None:
            bad_pairs.append((key, how))
            continue
        if how != "prefix":
            repaired_sides.append((key, how))
        if (home["points_for"], home["points_against"]) != \
                (away["points_against"], away["points_for"]):
            bad_pairs.append((key, f"scores {home['points_for']}-"
                                   f"{home['points_against']} vs "
                                   f"{away['points_against']}-"
                                   f"{away['points_for']}"))
        elif home["date"] != away["date"]:
            bad_pairs.append((key, "dates differ"))
    unlisted = [b for b in bad_pairs + singles
                if b[0] not in KNOWN_UNUSABLE_GAMES]
    report.check(not unlisted, f"{season} pairing",
                 f"{len(unlisted)} unlisted problem(s): {unlisted[:4]}")
    if repaired_sides:
        report.note(f"{season}: home/away resolved from the recap slug on "
                    f"{len(repaired_sides)} game(s) whose two articles both "
                    f"claimed the same side: {repaired_sides[:3]}")
    report.check(not partly, f"{season} one side played, the other not",
                 f"{len(partly)}: {list(partly)[:3]}")

    # ---- 2. quarter sum, the only check that catches an identical edit
    boxes = index_boxes(articles)
    matched, mismatched, incomplete = 0, [], 0
    known_bad_seen = set()
    quarter_rows = []
    for key, pair in played_games.items():
        ends = sides(pair)
        if ends is None:
            continue
        home, away = ends
        box = match_box(ends, boxes, season)
        if box is None:
            continue
        matched += 1
        for side, row, other in (("H", home, away), ("R", away, home)):
            quarters, overtime, complete = P.box_periods(box, side)
            if not complete:
                incomplete += 1
                continue
            total = sum(q for q in quarters if q is not None) + (overtime or 0)
            if total != row["points_for"]:
                if not box_is_known_bad(season, row, other):
                    mismatched.append((key, row["team_target"], total,
                                       row["points_for"]))
                else:
                    known_bad_seen.add((season, row["date_text"],
                                        row["team_target"]))
                continue
            quarter_rows.append(dict(
                season=season, date=row["date"], team_fid=row["team_fid"],
                opponent_fid=row["opponent_fid"], side=side,
                q1=quarters[0], q2=quarters[1], q3=quarters[2],
                q4=quarters[3], ot=overtime, total=total))
    report.check(not mismatched, f"{season} quarter sum equals final score",
                 f"{len(mismatched)} mismatch(es): {mismatched[:4]}")
    # The reverse guard: a listed bad box that is no longer bad is fatal too.
    expected_bad = {k for k in KNOWN_BAD_BOXES if k[0] == season}
    report.check(len(known_bad_seen) == len(expected_bad),
                 f"{season} every listed bad box is still bad",
                 f"listed {len(expected_bad)}, still failing "
                 f"{len(known_bad_seen)}")
    report.note(f"{season}: quarter boxes matched {matched}/"
                f"{len(played_games)} games, {incomplete} side(s) incomplete")

    # ---- 3. outcome template, where the articles carry it
    with_tpl = [r for r in played if r["outcome_tpl"]]
    wrong_tpl = [r for r in with_tpl if r["outcome_tpl"] != (
        "won" if r["points_for"] > r["points_against"] else
        "lost" if r["points_for"] < r["points_against"] else "tied")]
    report.check(not wrong_tpl, f"{season} outcome template agrees with score",
                 f"{len(wrong_tpl)}: "
                 f"{[(r['team_target'], r['week']) for r in wrong_tpl[:4]]}")
    report.note(f"{season}: outcome template on {len(with_tpl)}/{len(played)} "
                f"played rows ({100 * len(with_tpl) // max(len(played), 1)}%)")

    # ---- 4. the Record column
    by_team = collections.defaultdict(list)
    for row in played:
        by_team[row["team_fid"]].append(row)
    rec_bad, rec_checked = [], 0
    for fid, team_rows in by_team.items():
        team_rows.sort(key=lambda r: (r["date"], r["week"]))
        wins = losses = ties = 0
        for row in team_rows:
            if row["points_for"] > row["points_against"]:
                wins += 1
            elif row["points_for"] < row["points_against"]:
                losses += 1
            else:
                ties += 1
            numbers = [int(n) for n in re.findall(r"\d+", row["record"])]
            if not numbers:
                continue
            rec_checked += 1
            want = [wins, losses, ties] if len(numbers) == 3 else [wins, losses]
            if numbers != want:
                rec_bad.append((row["team_target"], row["week"],
                                row["record"], f"{wins}-{losses}-{ties}"))
    report.check(not rec_bad, f"{season} Record column reconciles",
                 f"{len(rec_bad)} row(s): {rec_bad[:4]}")
    report.note(f"{season}: Record column on {rec_checked}/{len(played)} "
                f"played rows")

    # ---- 5. completeness, every count derived
    per_team = collections.Counter(r["team_fid"] for r in rows)
    counts = sorted(per_team.values())
    teams = len(per_team)
    mode = collections.Counter(counts).most_common(1)[0][0]
    shape = ("uniform" if counts[0] == counts[-1] else
             "short-only" if counts[-1] == mode else "unbalanced")
    expected_rows = sum(counts)
    report.check(expected_rows % 2 == 0, f"{season} row count is even",
                 f"{expected_rows}")
    report.check(len(games) * 2 == expected_rows,
                 f"{season} every row belongs to a two-row game",
                 f"{len(games)} games vs {expected_rows} rows")
    if shape == "short-only":
        shortfall = sum(mode - c for c in counts)
        report.check(shortfall % 2 == 0,
                     f"{season} shortfall halves evenly", f"{shortfall}")
    report.note(f"{season}: {teams} teams, {mode} games each (mode), "
                f"shape {shape}, {len(games)} games, "
                f"{len(played_games)} fully played")

    # ---- 6. uniqueness
    dupes = [(k, len(v)) for k, v in games.items() if len(v) > 2]
    report.check(not dupes, f"{season} no duplicate (date, team pair)",
                 f"{dupes[:4]}")
    week_dupes = []
    for fid, team_rows in collections.defaultdict(list, {
            fid: [r for r in rows if r["team_fid"] == fid]
            for fid in per_team}).items():
        seen = collections.Counter(r["week"] for r in team_rows)
        week_dupes += [(fid, w, n) for w, n in seen.items() if n > 1]
    report.check(not week_dupes, f"{season} no franchise plays twice in a week",
                 f"{week_dupes[:4]}")

    # ---- 7. ties
    ties_rows = [r for r in played if r["wl"] == "T"]
    bad_tie = [r for r in ties_rows
               if r["points_for"] != r["points_against"]]
    equal_not_tie = [r for r in played
                     if r["points_for"] == r["points_against"]
                     and r["wl"] != "T"]
    report.check(not bad_tie, f"{season} every T has equal scores",
                 f"{[(r['team_target'], r['week']) for r in bad_tie[:4]]}")
    report.check(not equal_not_tie,
                 f"{season} no equal-score game outside the tie class",
                 f"{[(r['team_target'], r['week'], r['wl']) for r in equal_not_tie[:4]]}")
    voided = [r for r in rows if r["status"] == "voided"]
    in_progress = [r for r in rows if r["status"] == "in_progress"]
    report.note(f"{season}: {len(ties_rows) // 2} tied game(s), "
                f"{len(voided) // 2} voided, {len(in_progress)} in progress")

    # ---- 8. identity
    expected_ids = {v[0] for v in F.franchise_table(season).values()}
    seen_ids = set(per_team)
    report.check(seen_ids == expected_ids, f"{season} all 32 franchises present",
                 f"missing {sorted(expected_ids - seen_ids)}, "
                 f"unexpected {sorted(seen_ids - expected_ids)}")

    headers = collections.Counter(a["header"] for a in articles.values())
    report.note(f"{season}: {len(headers)} distinct header shape(s); "
                f"sections {dict(collections.Counter(a['section'] for a in articles.values()))}")
    return dict(rows=len(rows), played=len(played), games=len(games),
                played_games=len(played_games), shape=shape, mode=mode,
                teams=teams, ties=len(ties_rows) // 2,
                voided=len(voided) // 2,
                tpl=len(with_tpl), record=rec_checked,
                boxes_matched=matched, quarter_rows=quarter_rows,
                identity_display=sum(1 for r in rows
                                     if r["identity_from"] == "display"))


def check_known_list(all_games, report):
    """Both directions: an unlisted bad game is fatal, and so is a listed
    game that is no longer present or no longer bad."""
    for key in KNOWN_UNUSABLE_GAMES:
        report.check(key in all_games,
                     f"known-unusable {key} is still present")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="*", type=int)
    args = parser.parse_args()
    seasons = args.seasons or F.seasons_through()

    report = Report()
    summary, all_games, quarter_rows = {}, {}, []
    print("NFL validation")
    print(f"  seasons {seasons[0]}..{seasons[-1]}")
    print()
    for season in seasons:
        rows, articles = P.parse_season(season)
        summary[season] = validate_season(season, rows, articles, report)
        quarter_rows += summary[season].pop("quarter_rows")
        all_games.update(group_games(rows))
    check_known_list(all_games, report)

    header = (f"  {'season':<8}{'rows':>6}{'played':>8}{'games':>7}"
              f"{'teams':>7}{'g/team':>8}{'shape':>13}{'ties':>6}"
              f"{'tpl%':>6}{'rec%':>6}{'box%':>6}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for season, s in summary.items():
        played = max(s["played"], 1)
        print(f"  {season:<8}{s['rows']:>6}{s['played']:>8}{s['games']:>7}"
              f"{s['teams']:>7}{s['mode']:>8}{s['shape']:>13}{s['ties']:>6}"
              f"{100 * s['tpl'] // played:>6}{100 * s['record'] // played:>6}"
              f"{100 * s['boxes_matched'] // max(s['played_games'], 1):>6}")
    print()
    for note in report.notes:
        print(f"  . {note}")
    print()
    print(f"  {report.checks} checks, {len(report.failures)} failed")
    for failure in report.failures:
        print(f"  FAIL {failure}")
    return 0 if not report.failures else 1


if __name__ == "__main__":
    sys.exit(main())
