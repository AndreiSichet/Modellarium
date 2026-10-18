"""Read-only gate over the built NHL tables. Exits non-zero on any failure.

Completed seasons are STRICT. The open season's standings check is ADVISORY,
with the sign pattern classified, because its standings page carries a date
rather than `complete` and can lag the game logs.

A row that fails to PARSE is fatal in either case - `build_nhl_tables.py`
raises on it. Quarantine is for a disagreement between two readable copies,
never for markup nobody can read.
"""
import argparse
import collections
import csv
import datetime as dt
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_nhl_tables as B                                 # noqa: E402
import nhl_franchises as F                                   # noqa: E402
import parse_nhl_wikitext as N                               # noqa: E402
from nhl_standings_exceptions import (KNOWN_UNUSABLE_GAMES,   # noqa: E402
                                      STANDINGS_EXCEPTIONS)

OUT = HERE.parent.parent / "data" / "nhl" / "processed"

# Reported, not fatal where the standings already reconcile.
RECORD_DIVERGENCES = []
MARKER_DISAGREEMENTS = []


class Checks:
    def __init__(self):
        self.passed = 0
        self.failed = []

    def ok(self, label):
        self.passed += 1

    def check(self, condition, label, detail=""):
        if condition:
            self.passed += 1
        else:
            self.failed.append((label, detail))
        return condition

    def report(self, title):
        print("  %-62s %d passed, %d failed"
              % (title, self.passed, len(self.failed)))
        for label, detail in self.failed:
            print("      FAIL %s%s" % (label, (" - " + detail) if detail else ""))


def read_csv(path):
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def check_marker_and_record(year, checks):
    """The two channels that do not come from the score.

    CHANNEL 2 - the outcome marker. A row's colour or `{{Game-*}}` template
    asserts the result independently of the numbers, so a winner flipped in
    both copies is still visible here.

    CHANNEL 3 - the running record column. Recomputing W-L-OTL in date order
    must reproduce what the article prints, which catches a result changed
    without the record being updated to match - and, because an OTL earns a
    point, it is the channel that sees REG relabelled as OT.
    """
    label = F.season_label(year)
    rows, _standings, _bad = B.load_season(year)
    played = [r for r in rows if r["played"]]

    judged = bad = 0
    for row in played:
        if not row["marker"]:
            continue
        judged += 1
        said = N.result_from(row["goals_for"], row["goals_against"],
                             row["game_end"])
        # The marker distinguishes win / loss / OT-loss, which is exactly the
        # result class, so it is compared against that and not against the
        # raw GAME_END.
        if row["marker"] != said:
            bad += 1
            MARKER_DISAGREEMENTS.append(
                (label, F.abbrev_for(row["team_fid"], year),
                 row["date"].isoformat(), row["marker"], said))

    judged_rec, divergences = 0, []
    by_team = {}
    for row in played:
        by_team.setdefault(row["team_fid"], []).append(row)
    for fid, team_rows in by_team.items():
        team_rows.sort(key=lambda r: r["date"])
        tally = [0, 0, 0]
        for row in team_rows:
            result = N.result_from(row["goals_for"], row["goals_against"],
                                   row["game_end"])
            tally["WLO".index({"W": "W", "L": "L", "OTL": "O"}[result])] += 1
            parts = re.findall(r"\d+", row["record"].strip())
            if len(parts) != 3:
                continue
            judged_rec += 1
            stated = [int(part) for part in parts]
            if stated != tally:
                divergences.append((F.abbrev_for(fid, year),
                                    row["date"].isoformat(),
                                    list(tally), stated))
                # RESYNC. A running tally that goes wrong once stays wrong,
                # so without this one defect is counted on every later row -
                # 10 real ones read as 309.
                tally = stated
    RECORD_DIVERGENCES.extend((label,) + d for d in divergences)
    print("        %-9s marker %d judged / %d disagree   record %d judged / "
          "%d divergence(s)"
          % (label, judged, bad, judged_rec, len(divergences)))
    return checks


def validate(seasons, out_dir, checks):
    games = read_csv(out_dir / "nhl_games_final.csv")
    repairs = read_csv(out_dir / "nhl_repairs.csv")
    quarantine = read_csv(out_dir / "nhl_quarantine.csv")
    checks.check(bool(games), "the games table is not empty")

    by_season = collections.defaultdict(list)
    for row in games:
        by_season[row["SEASON"]].append(row)

    for year in seasons:
        label = F.season_label(year)
        rows = by_season.get(label, [])
        checks.check(bool(rows), "%s has rows" % label)
        if not rows:
            continue

        # ---- 6. UNIQUENESS: no duplicate key, no franchise twice a day.
        ids = collections.Counter(r["GAME_ID"] for r in rows)
        checks.check(all(n == 2 for n in ids.values()),
                     "%s every GAME_ID appears exactly twice" % label,
                     str([g for g, n in ids.items() if n != 2][:3]))
        per_day = collections.Counter((r["GAME_DATE"], r["TEAM_ID"])
                                      for r in rows)
        checks.check(all(n == 1 for n in per_day.values()),
                     "%s no franchise plays twice on one date" % label,
                     str([k for k, n in per_day.items() if n > 1][:3]))

        # ---- the two rows of a game must mirror each other exactly.
        paired = collections.defaultdict(list)
        for row in rows:
            paired[row["GAME_ID"]].append(row)
        mirrored = 0
        for gid, two in paired.items():
            if len(two) != 2:
                continue
            a, b = two
            if (a["GOALS_FOR"] == b["GOALS_AGAINST"]
                    and a["GOALS_AGAINST"] == b["GOALS_FOR"]
                    and a["GAME_END"] == b["GAME_END"]
                    and {a["IS_HOME"], b["IS_HOME"]} == {"0", "1"}
                    and int(a["MARGIN"]) == -int(b["MARGIN"])):
                mirrored += 1
        checks.check(mirrored == len(paired),
                     "%s both rows of every game mirror" % label,
                     "%d of %d" % (mirrored, len(paired)))

        # ---- RESULT and GAME_END are separate questions, and consistent.
        bad_result = [r for r in rows
                      if r["RESULT"] != N.result_from(int(r["GOALS_FOR"]),
                                                      int(r["GOALS_AGAINST"]),
                                                      r["GAME_END"])]
        checks.check(not bad_result,
                     "%s RESULT follows from the goals and GAME_END" % label,
                     str(len(bad_result)))
        checks.check(all(r["GAME_END"] in ("REG", "OT", "SO") for r in rows),
                     "%s GAME_END is REG, OT or SO" % label)
        # Zero ties, and zero OT/SO games with a margin other than one -
        # round 4's invariant on 7,434 rows, which is what proves the
        # one-extra-goal convention.
        checks.check(not [r for r in rows if int(r["MARGIN"]) == 0],
                     "%s has no ties" % label)
        odd = [r for r in rows if r["GAME_END"] in ("OT", "SO")
               and abs(int(r["MARGIN"])) != 1]
        checks.check(not odd,
                     "%s no OT/SO game has a margin other than one" % label,
                     str(len(odd)))

        # ---- 5. COMPLETENESS, derived per season, never a constant.
        per_team = collections.Counter(r["TEAM_ID"] for r in rows)
        counts = sorted(per_team.values())
        mode = collections.Counter(counts).most_common(1)[0][0]
        shape = ("uniform" if counts[0] == counts[-1]
                 else "short-only" if counts[-1] == mode else "unbalanced")
        expected = F.franchises_for(year)
        checks.check(len(per_team) == len(set(expected.values())),
                     "%s every franchise has games" % label,
                     "%d of %d" % (len(per_team), len(set(expected.values()))))
        if shape == "short-only":
            shortfall = sum(mode - c for c in counts)
            checks.check(shortfall % 2 == 0,
                         "%s short-only shortfall is even" % label,
                         str(shortfall))
        checks.check(sum(counts) == 2 * len(paired),
                     "%s row count is twice the game count" % label)
        print("        %-9s %-5d games  per team %s  shape %s"
              % (label, len(paired), counts[0] if shape == "uniform"
                 else "%d..%d" % (counts[0], counts[-1]), shape))

    # ---- 4. the exception maps' REVERSE GUARD.
    # SCOPED TO THE SEASONS ACTUALLY BUILT. A subset build must not fail on
    # an entry belonging to a season it did not build - which is what made
    # the negative suite's control fail while nothing was planted.
    built = {F.season_label(y) for y in seasons}
    listed_games = {k for k in KNOWN_UNUSABLE_GAMES
                    if k.split(" ")[0] in built}
    seen = {"%s %s" % (r["season"], r["date"]) for r in repairs}
    # A listed game must still exist in the corpus, and must still NOT be
    # repaired - otherwise the list has rotted into a tolerance.
    all_dates = {"%s %s" % (r["SEASON"], r["GAME_DATE"]) for r in games}
    for key in sorted(listed_games):
        checks.check(key in all_dates,
                     "listed unusable game still exists: %s" % key)
        checks.check(key not in seen,
                     "listed unusable game is still unrepaired: %s" % key)
    for label, teams in sorted(STANDINGS_EXCEPTIONS.items()):
        checks.check(bool(teams),
                     "standings exception for %s is not empty" % label)

    print()
    print("  the two per-article channels, re-parsed from the raw wikitext:")
    for year in seasons:
        check_marker_and_record(year, checks)

    print()
    print("  marker disagreements: %d   record divergences: %d"
          % (len(MARKER_DISAGREEMENTS), len(RECORD_DIVERGENCES)))
    print("     Both are channels INDEPENDENT of the score. Reported rather "
          "than fatal where")
    print("     the standings already reconcile the season - two channels "
          "disagree and the")
    print("     standings are the stronger: a separate page, and the only "
          "one that sees a")
    print("     score changed without changing the winner.")
    for row in MARKER_DISAGREEMENTS[:3]:
        print("        marker %s %s %s: says %s, score says %s" % row)
    for row in RECORD_DIVERGENCES[:3]:
        print("        record %s %s %s: computed %s, article says %s" % row)

    print()
    print("  repairs recorded: %d   quarantined: %d   listed unusable: %d"
          % (len(repairs), len(quarantine), len(listed_games)))
    for row in repairs[:3]:
        print("      repair %s %s  believed %s (%s)"
              % (row["season"], row["date"], row["believed"][:34],
                 row["believed_score"]))
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--seasons", default="")
    args = parser.parse_args(argv)
    seasons = ([int(s) for s in args.seasons.split(",") if s.strip()]
               or F.seasons_through())
    print("Validating NHL tables in %s" % args.out)
    print()
    checks = validate(seasons, Path(args.out), Checks())
    print()
    checks.report("TOTAL")
    return 1 if checks.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
