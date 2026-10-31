"""Build the NHL tables from the fetched wikitext.

Five outputs, all gitignored (CC BY-SA 4.0 - see DATA_LICENSE.md):

    nhl_games_final.csv         two rows per game, one per team
    nhl_fixtures.csv            unplayed games, the union of both articles
    nhl_franchise_identity.csv  the authored table, as built
    nhl_repairs.csv             games the standings resolved, with evidence
    nhl_quarantine.csv          open-season disagreements, set aside

THE KEY IS `({both teams}, date +/- 1 day)`. Round 4 measured it as the one
key that pairs every row; `(date, home, away)` leaves 4 singletons in 2021-22
alone, because two games' two copies disagree about the date by a day. The two
copies never disagree about which team was at HOME - measured again here as 0
conflicts in 13 of 14 seasons, after a `\\b` in the away marker was fixed.

COMPLETED SEASONS ARE STRICT; THE OPEN SEASON QUARANTINES. A season is
completed when its own standings page says `complete` rather than carrying a
date - the page's own word, not a calendar rule, because that is the thing
that can lag.

THE STANDINGS ARE A REPAIR RULE, NOT ONLY A DETECTOR, which is what makes a
non-zero baseline usable. Where the two copies disagree and exactly one
assignment reconciles both teams' goals for and against with the standings,
that assignment is applied and the game is recorded as DERIVED - a value no
single article states. Where none or several do, the game is named with its
evidence. A blanket tolerance is the wrong instrument: one wide enough for the
real failures is wide enough to admit a planted error.
"""
import argparse
import csv
import datetime as dt
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import nhl_franchises as F                                   # noqa: E402
import nhl_standings as S                                    # noqa: E402
import parse_nhl_wikitext as N                               # noqa: E402
from nhl_standings_exceptions import (KNOWN_UNUSABLE_GAMES,   # noqa: E402
                                      STANDINGS_EXCEPTIONS)

RAW = HERE.parent.parent / "data" / "nhl" / "raw"
OUT = HERE.parent.parent / "data" / "nhl" / "processed"

# A team's own suspects are searched exhaustively. Capped so a pathological
# season cannot turn into 2**n; measured, no team has had more than 4.
MAX_SUSPECTS = 12

# OT-versus-SO label disagreements: recorded, never absorbed.
END_DISPUTES = []
DISCOVERED = []


def season_dir(year):
    return "%d-%02d" % (year, (year + 1) % 100)


def advisory_years(today=None):
    """Which seasons may have an ADVISORY standings check: the current one.

    KEYED ON THE CURRENT SEASON, NOT ON `open_seasons()`, which is the
    quarantine's recorded lesson: `open_seasons()` returns the current season
    AND THE ONE BEFORE, so a disagreement in a finished 2025-26 would have
    been quietly set aside instead of failing. One season is eligible, and
    the verifier asserts the previous one is not.
    """
    return {F.seasons_through(today)[-1]}


def load_season(year, strict=None, today=None):
    """(rows, standings, unreadable) for one season.

    `strict=None` DECIDES FOR ITSELF, and it needs two things to agree before
    it relaxes anything:

      * the season is the CURRENT one (`advisory_years`), and
      * its standings page does not say `complete`.

    BOTH, not either. The page's stamp is a content signal and the clock is a
    clock signal, and requiring both means a finished season whose template
    somebody edited to carry a date cannot become lenient, while the current
    season whose page already says `complete` stays strict. Either alone would
    be weaker than the pair.
    """
    directory = RAW / season_dir(year)
    manifest = json.loads(
        (directory / "manifest.json").read_text(encoding="utf-8"))
    rows, unreadable = [], 0
    for title, fid in sorted(F.titles_for(year).items()):
        entry = manifest["pages"].get(title)
        if entry is None:
            raise RuntimeError(
                "%r is absent from the manifest. Re-run the fetcher; a "
                "missing article is a coverage problem and not skipped."
                % title)
        article = directory / entry["file"]
        if not article.is_file():
            raise RuntimeError(
                "%r is in the manifest and its file %s is absent. A silently "
                "missing article is a COVERAGE problem - the completeness "
                "check cannot see a team that contributed no rows at all - "
                "so this is fatal rather than skipped."
                % (title, article.name))
        text = article.read_text(encoding="utf-8")
        got, bad = N.parse_article(text, year, fid, title)
        rows += got
        unreadable += bad
    texts = {t: (directory / v["file"]).read_text(encoding="utf-8")
             for t, v in manifest["pages"].items() if t.startswith("Template:")}
    # THE STAMP IS READ WITHOUT PARSING THE FIELDS, which is the whole reason
    # the decision can be made here: `S.stamps_say_complete` is a regex over
    # `update=` and does not care whether the numbers agree. Without that the
    # builder would have to parse the fields to learn whether it was allowed
    # to parse them leniently.
    if strict is None:
        says_complete = S.stamps_say_complete(texts)
        strict = says_complete or year not in advisory_years(today)
    standings = S.read_season(texts, strict=strict)
    standings["strict"] = strict
    standings["completeness_problems"] = S.require_complete(
        F.season_label(year), standings, F.abbrev_index(year).keys(),
        strict=strict)
    return rows, standings, unreadable


def is_complete(standings):
    """The page's own word. An open season's page carries a date instead."""
    stamps = {v.strip().lower() for v in standings["update"].values()}
    return bool(stamps) and all(s == "complete" for s in stamps)


def sides(row):
    """(home_fid, away_fid, home_goals, away_goals) as THIS copy states it."""
    if row["away"]:
        return (row["opponent_fid"], row["team_fid"],
                row["goals_against"], row["goals_for"])
    return (row["team_fid"], row["opponent_fid"],
            row["goals_for"], row["goals_against"])


def game_id(season_label, date, home, away):
    return "%s_%s_%d_%d" % (season_label, date.isoformat(), away, home)


def pair_rows(rows):
    """(pairs, singletons) on ({both teams}, date +/- 1 day)."""
    buckets = {}
    for row in rows:
        buckets.setdefault(
            frozenset((row["team_fid"], row["opponent_fid"])), []).append(row)
    pairs, singles = [], []
    for _key, group in buckets.items():
        group.sort(key=lambda r: (r["date"], r["team_fid"]))
        used = [False] * len(group)
        for i, row in enumerate(group):
            if used[i]:
                continue
            twin = next((j for j in range(i + 1, len(group))
                         if not used[j]
                         and group[j]["team_fid"] != row["team_fid"]
                         and abs((group[j]["date"] - row["date"]).days) <= 1),
                        None)
            if twin is None:
                singles.append(row)
                used[i] = True
            else:
                used[i] = used[twin] = True
                pairs.append((row, group[twin]))
    return pairs, singles


def agree(a, b):
    """Do the two copies of a game state the same thing?

    `game_end` is part of the answer, because it decides W/L/OTL: an OTL is a
    loss that earns a point. A pair agreeing on the score and disagreeing on
    OT-versus-SO left Detroit and Ottawa 2013-14 off by exactly one loss and
    one overtime loss, with the goals right.
    """
    ha, aa, hga, aga = sides(a)
    hb, ab, hgb, agb = sides(b)
    return ((ha, aa, hga, aga, _class(hga, aga, a["game_end"]))
            == (hb, ab, hgb, agb, _class(hgb, agb, b["game_end"])))


def _class(home_goals, away_goals, end):
    """The pair of result classes the standings can see: W/L/OTL per side."""
    return (N.result_from(home_goals, away_goals, end),
            N.result_from(away_goals, home_goals, end))


def end_disputed(a, b):
    """Do the copies agree on the class and differ on OT versus SO?"""
    return a["game_end"] != b["game_end"] and agree(a, b)


def settle_end(a, b):
    """The label to record, and whether the copies disputed it.

    The HOME team's copy wins - a stated tie-break. The standings cannot
    adjudicate this, so a rule is needed and it is written down rather than
    left to whichever article happened to parse first.
    """
    if a["game_end"] == b["game_end"]:
        return a["game_end"], False
    home_copy = a if not a["away"] else b
    return home_copy["game_end"], True


def team_totals(assignment):
    """{fid: {gp,w,l,otl,gf,ga}} from a chosen set of games."""
    out = {}
    for home, away, hg, ag, end in assignment:
        for fid, gf, ga in ((home, hg, ag), (away, ag, hg)):
            row = out.setdefault(fid, dict(gp=0, w=0, l=0, otl=0, gf=0, ga=0))
            row["gp"] += 1
            row["gf"] += gf
            row["ga"] += ga
            if gf > ga:
                row["w"] += 1
            elif end in ("OT", "SO"):
                row["otl"] += 1
            else:
                row["l"] += 1
    return out


def stated(standings, abbrevs):
    """{fid: {w,l,otl,gf,ga}} from the standings, keyed by franchise."""
    out = {}
    for abbrev, fid in abbrevs.items():
        row = standings["teams"].get(abbrev)
        if row is None:
            continue
        out[fid] = dict(w=row.get("win"), l=row.get("loss"),
                        otl=row.get("OTloss"), gf=row.get("gf"),
                        ga=row.get("ga"))
    return out


def diffs(computed, declared):
    """{fid: {field: computed - stated}} for the fields the standings carry."""
    out = {}
    for fid, want in declared.items():
        mine = computed.get(fid, dict(w=0, l=0, otl=0, gf=0, ga=0))
        delta = {k: mine[k] - want[k] for k in ("w", "l", "otl", "gf", "ga")
                 if want.get(k) is not None}
        delta = {k: v for k, v in delta.items() if v}
        if delta:
            out[fid] = delta
    return out


def _choices(pair):
    """The two readings of a disagreeing game: believe copy A, or copy B."""
    out = []
    for row, which in ((pair[0], 0), (pair[1], 1)):
        home, away, hg, ag = sides(row)
        out.append(((home, away, hg, ag, row["game_end"]), which))
    return out


def _component_of(suspects, singles):
    """Group suspects into connected components by shared team.

    A game's two readings can only move the two teams that played it, so the
    search factorises. Measured: the largest component in the corpus is 2
    games, against a season's 14 suspects - the difference between 4 and
    16,384 assignments, and between a localised answer and a season-wide one.
    """
    items = [("pair", i, frozenset(sides(a)[:2]))
             for i, (a, _b) in enumerate(suspects)]
    items += [("single", i, frozenset(sides(r)[:2]))
              for i, r in enumerate(singles)]
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i][2] & items[j][2]:
                parent[find(i)] = find(j)
    groups = {}
    for i, item in enumerate(items):
        groups.setdefault(find(i), []).append(item)
    return list(groups.values())


def repair_season(pairs, singles, standings, abbrevs, season_label):
    """(settled, repairs, residual) - the standings as a repair rule."""
    settled, suspects = [], []
    for a, b in pairs:
        if agree(a, b):
            home, away, hg, ag = sides(a)
            end, disputed = settle_end(a, b)
            settled.append((home, away, hg, ag, end, a, b, False, disputed))
            if disputed:
                END_DISPUTES.append(dict(
                    season=season_label, date=a["date"].isoformat(),
                    home=home, away=away,
                    copy_a="%s says %s" % (a["article"], a["game_end"]),
                    copy_b="%s says %s" % (b["article"], b["game_end"]),
                    recorded=end,
                    reason="the standings cannot see OT versus SO; the home "
                           "team's copy is the stated tie-break"))
        else:
            suspects.append((a, b))

    declared = stated(standings, abbrevs)
    base = [g[:5] for g in settled]
    repairs, residual = [], []
    resolved = {}

    for component in _component_of(suspects, singles):
        if len(component) > MAX_SUSPECTS:
            raise RuntimeError(
                "%s has a component of %d suspect game(s), above the cap of "
                "%d. That is a different problem from the one this search "
                "solves; look at the parse before widening the cap."
                % (season_label, len(component), MAX_SUSPECTS))
        teams = set()
        for _kind, _index, pair_teams in component:
            teams |= set(pair_teams)
        # Everything not in this component, as the component's baseline.
        others = list(base)
        for kind, index, _t in [(k, i, t) for k, i, t in
                                [("pair", i, None) for i in range(len(suspects))]
                                + [("single", i, None)
                                   for i in range(len(singles))]]:
            if (kind, index) in {(k, i) for k, i, _ in component}:
                continue
            if (kind, index) in resolved:
                others.append(resolved[(kind, index)])
        options = []
        for kind, index, _t in component:
            if kind == "pair":
                options.append([(v, w, kind, index)
                                for v, w in _choices(suspects[index])])
            else:
                row = singles[index]
                home, away, hg, ag = sides(row)
                options.append([((home, away, hg, ag, row["game_end"]),
                                 True, kind, index),
                                (None, False, kind, index)])
        wanted = {f: d for f, d in declared.items() if f in teams}
        solutions = []
        for combo in itertools.product(*options):
            chosen = list(others)
            for value, _which, _k, _i in combo:
                if value is not None:
                    chosen.append(value)
            totals = team_totals(chosen)
            if not {f: d for f, d in diffs(totals, wanted).items()}:
                solutions.append(combo)

        if len(solutions) == 1:
            for value, which, kind, index in solutions[0]:
                if kind == "pair":
                    a, b = suspects[index]
                    believed, other = (a, b)[which], (a, b)[1 - which]
                    home, away, hg, ag = value[:4]
                    settled.append((home, away, hg, ag, value[4], a, b, True,
                                False))
                    resolved[(kind, index)] = value
                    repairs.append(dict(
                        season=season_label,
                        date=believed["date"].isoformat(),
                        home=home, away=away, believed=believed["article"],
                        rejected=other["article"],
                        believed_score="%d-%d (%s)" % (hg, ag, value[4]),
                        rejected_score="%d-%d (%s)" % (
                            sides(other)[2], sides(other)[3],
                            other["game_end"]),
                        evidence="the only assignment that reconciles both "
                                 "teams with the standings",
                        kind="copies disagree"))
                else:
                    row = singles[index]
                    if value is None:
                        residual.append(dict(
                            season=season_label,
                            date=row["date"].isoformat(),
                            reason="a singleton the standings do not want",
                            article=row["article"]))
                        continue
                    home, away, hg, ag = value[:4]
                    settled.append((home, away, hg, ag, value[4],
                                    row, row, True, False))
                    resolved[(kind, index)] = value
                    repairs.append(dict(
                        season=season_label, date=row["date"].isoformat(),
                        home=home, away=away, believed=row["article"],
                        rejected="(absent from the opponent's log)",
                        believed_score="%d-%d (%s)" % (hg, ag, value[4]),
                        rejected_score="",
                        evidence="the opponent carries no copy, and adding "
                                 "this one reconciles the standings exactly",
                        kind="row absent from one log"))
            continue

        # Not uniquely repairable. Carried through on copy A and NAMED, so
        # nothing is silently dropped and nothing silently invented.
        why = ("ambiguous - %d assignments reconcile" % len(solutions)
               if solutions else "indeterminate - no assignment reconciles")
        for kind, index, _t in component:
            if kind == "pair":
                a, b = suspects[index]
                home, away, hg, ag = sides(a)
                settled.append((home, away, hg, ag, a["game_end"],
                                a, b, False, False))
                residual.append(dict(
                    season=season_label, date=a["date"].isoformat(),
                    reason=why,
                    article="%s / %s" % (a["article"], b["article"])))
            else:
                row = singles[index]
                home, away, hg, ag = sides(row)
                settled.append((home, away, hg, ag, row["game_end"],
                                row, row, False, False))
                residual.append(dict(
                    season=season_label, date=row["date"].isoformat(),
                    reason=why + " (singleton)", article=row["article"]))
    return settled, repairs, residual


def emit_games(settled, season_label, rows_out):
    """Two rows per game, one per team."""
    for home, away, hg, ag, end, a, _b, derived, disputed in settled:
        date = a["date"]
        gid = game_id(season_label, date, home, away)
        for fid, opp, gf, ga, is_home in ((home, away, hg, ag, 1),
                                          (away, home, ag, hg, 0)):
            rows_out.append(dict(
                GAME_ID=gid, SEASON=season_label, GAME_DATE=date.isoformat(),
                TEAM_ID=fid, OPPONENT_ID=opp, IS_HOME=is_home,
                GOALS_FOR=gf, GOALS_AGAINST=ga,
                RESULT=N.result_from(gf, ga, end), GAME_END=end,
                MARGIN=gf - ga, DERIVED=int(derived),
                END_DISPUTED=int(disputed)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", default="")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--discover", action="store_true",
                        help="report unreconciled teams instead "
                             "of failing, to populate the "
                             "exception list from a real build")
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seasons = ([int(s) for s in args.seasons.split(",") if s.strip()]
               or F.seasons_through())
    games, fixtures, repairs, quarantine, residual = [], [], [], [], []
    report = []
    standings_report = []

    for year in seasons:
        label = F.season_label(year)
        rows, standings, unreadable = load_season(year)
        if unreadable:
            raise N.ParseError(
                "%s: %d row(s) could not be read. A row that fails parsing "
                "is fatal in either a completed or an open season - "
                "quarantine is for a disagreement between two READABLE "
                "copies." % (label, unreadable))
        played = [r for r in rows if r["played"]]
        unplayed = [r for r in rows if not r["played"]]
        complete = is_complete(standings)
        pairs, singles = pair_rows(played)
        abbrevs = F.abbrev_index(year)

        known = {g for g, _why in KNOWN_UNUSABLE_GAMES.items()}
        if complete:
            settled, fixed, left = repair_season(
                pairs, singles, standings, abbrevs, label)
            repairs += fixed
            residual += left
            unresolved = [d for d in left
                          if "%s %s" % (label, d["date"]) not in known]
            allowed = STANDINGS_EXCEPTIONS.get(label, set())
            flagged = diffs(team_totals([s[:5] for s in settled]),
                            stated(standings, abbrevs))
            unexpected = {F.abbrev_for(f, year): d for f, d in flagged.items()
                          if F.abbrev_for(f, year) not in allowed}
            if unexpected and args.discover:
                DISCOVERED.append((label, unexpected))
            elif unexpected:
                raise RuntimeError(
                    "%s does not reconcile with its own standings for %s, "
                    "and these are not named in STANDINGS_EXCEPTIONS. The "
                    "standings are the only channel that sees a score "
                    "changed without changing the winner, so this is fatal."
                    % (label, unexpected))
            if unresolved and not args.discover:
                # AN UNNAMED RESIDUAL IS FATAL. Without this the pipeline
                # ABSORBS a corruption instead of reporting it: a duplicated
                # game becomes a singleton, the standings decline to want it,
                # and the build succeeds with the duplicate quietly dropped.
                # That is gaps entry 40's lesson - a step that cleans can
                # swallow the very defect a plant is testing for - and the
                # negative suite's duplicate-game plant is what found it
                # here, by passing before this check existed and failing
                # after an unrelated fix removed the accident that caught it.
                raise RuntimeError(
                    "%s leaves %d game(s) the standings cannot resolve and "
                    "KNOWN_UNUSABLE_GAMES does not name: %s. Name them with "
                    "their evidence, or fix the parse - a residual that is "
                    "only reported is a corruption absorbed."
                    % (label, len(unresolved),
                       [(d["date"], d["reason"]) for d in unresolved][:4]))
            note = "%d repaired, %d residual" % (len(fixed), len(unresolved))
            # A COMPLETED SEASON GETS A ROW TOO, and it always reads
            # `reconciled`, because anything else raised above. Recorded so
            # the report covers every season rather than only the troubled
            # one - a table that lists only failures cannot be used to show
            # that the rest are fine.
            standings_report.append(dict(
                SEASON=label, STATE="complete",
                STRICT=int(standings["strict"]),
                CLASS=S.CLASS_RECONCILED, CLASSES=S.CLASS_RECONCILED,
                DETAIL="strict check passed", CONFLICTS=0,
                TEAMS_FLAGGED=0,
                UPDATE="|".join(sorted(set(standings["update"].values())))))
        else:
            # OPEN SEASON: quarantine, not failure.
            settled = []
            for a, b in pairs:
                if agree(a, b):
                    home, away, hg, ag = sides(a)
                    settled.append((home, away, hg, ag, a["game_end"],
                                    a, b, False, False))
                    continue
                quarantine.append(dict(
                    season=label, date=a["date"].isoformat(),
                    copy_a="%s says %s" % (a["article"], _say(a)),
                    copy_b="%s says %s" % (b["article"], _say(b)),
                    reason="the two copies disagree", left_quarantine=0))
            for row in singles:
                quarantine.append(dict(
                    season=label, date=row["date"].isoformat(),
                    copy_a="%s says %s" % (row["article"], _say(row)),
                    copy_b="(no second copy yet)",
                    reason="only one article carries this game",
                    left_quarantine=0))
            flagged = diffs(team_totals([s[:5] for s in settled]),
                            stated(standings, abbrevs))
            # THE RAW DELTAS, NOT THEIR SIGNS. `S.shape_of` needs the
            # magnitudes to tell a cancelling result-label difference from a
            # changed score, and the first version of this call threw them
            # away - which would have made every mislabelled result and
            # every altered score read as the same "mixed sign".
            by_abbrev = {F.abbrev_for(fid, year): d
                         for fid, d in flagged.items()}
            primary, classes, detail = S.classify(
                standings.get("conflicts") or [],
                standings.get("completeness_problems") or [], by_abbrev)
            located = _localise(by_abbrev, settled, year)
            if located:
                detail = "%s; %s" % (detail, located)
            standings_report.append(dict(
                SEASON=label, STATE="OPEN", STRICT=int(standings["strict"]),
                CLASS=primary, CLASSES="|".join(classes), DETAIL=detail,
                CONFLICTS=len(standings.get("conflicts") or []),
                TEAMS_FLAGGED=len(flagged),
                UPDATE="|".join(sorted(set(standings["update"].values())))))
            note = "%s - %s" % (primary, detail)

        emit_games(settled, label, games)
        for row in unplayed:
            home, away, _hg, _ag = sides(row)
            fixtures.append(dict(
                SEASON=label, GAME_DATE=row["date"].isoformat(),
                GAME_ID=game_id(label, row["date"], home, away),
                HOME_ID=home, AWAY_ID=away, SOURCE=row["article"]))
        report.append((label, len(played) // 2, len(pairs), len(singles),
                       "complete" if complete else "OPEN", note))

    # Fixtures are the UNION of both articles: one key, either source.
    seen, union = set(), []
    for row in sorted(fixtures, key=lambda r: (r["GAME_DATE"], r["GAME_ID"])):
        if row["GAME_ID"] in seen:
            continue
        seen.add(row["GAME_ID"])
        union.append(row)

    identity = [dict(FRANCHISE_ID=fid, ABBREV=abbrev, FIRST_SEASON=first,
                     LAST_SEASON=last, LINK_TARGET=target)
                for fid, abbrev, first, last, target in F._spans()]

    _write(out_dir / "nhl_games_final.csv", games)
    _write(out_dir / "nhl_fixtures.csv", union)
    _write(out_dir / "nhl_franchise_identity.csv", identity)
    _write(out_dir / "nhl_repairs.csv", repairs)
    _write(out_dir / "nhl_quarantine.csv", quarantine)
    _write(out_dir / "nhl_end_disputes.csv", END_DISPUTES)
    # THE STANDINGS REPORT IS A TABLE, not a line of stdout, because section
    # 3 has to carry it into the snapshot and `/health` has to show it. One
    # row per season, with a stable CLASS a comparison can key on and a
    # DETAIL for reading - the gate's false REGRESSED is the recorded reason
    # those are two columns and not one.
    _write(out_dir / "nhl_standings_report.csv", standings_report)

    print("NHL tables -> %s" % out_dir)
    print()
    print("%-9s %-7s %-7s %-7s %-9s %s"
          % ("season", "games", "pairs", "single", "state", "notes"))
    for line in report:
        print("%-9s %-7d %-7d %-7d %-9s %s" % line)
    print()
    print("  games_final  %6d row(s)  (%d derived)"
          % (len(games), sum(r["DERIVED"] for r in games)))
    print("  fixtures     %6d" % len(union))
    print("  repairs      %6d" % len(repairs))
    print("  quarantine   %6d" % len(quarantine))
    print("  residual     %6d" % len(residual))
    print()
    print("STANDINGS CHECK PER SEASON")
    print("  %-9s %-9s %-7s %-30s %s"
          % ("season", "state", "strict", "class", "detail"))
    for row in standings_report:
        print("  %-9s %-9s %-7s %-30s %s"
              % (row["SEASON"], row["STATE"],
                 "yes" if row["STRICT"] else "ADVISORY",
                 row["CLASS"], row["DETAIL"][:64]))
    print()
    print("  end disputes %6d  (OT vs SO; the standings cannot see these)"
          % len(END_DISPUTES))
    if DISCOVERED:
        print()
        print("  UNRECONCILED (discovery mode; name these with evidence):")
        for label, teams in DISCOVERED:
            for abbrev, delta in sorted(teams.items()):
                print("     %-9s %-5s %s" % (label, abbrev, delta))
    if N.HEADING_DISAGREEMENTS:
        print()
        print("  caption/heading disagreements, recovered and reported:")
        for article, why in N.HEADING_DISAGREEMENTS:
            print("     %-44s %s" % (article[:44], why))
    return 0


def _say(row):
    home, away, hg, ag = sides(row)
    return "home %d, %d-%d, %s" % (home, hg, ag, row["game_end"])


def _localise(by_abbrev, settled, year):
    """Name the game behind a RESULT-LABEL difference, where one game can be.

    THE SPEC ASKS FOR THIS AND IT IS CHEAP. A result-label difference means
    the same games are counted with one of them typed differently - so the
    candidates are that team's games whose end marker is OT or SO, and when
    there is exactly one the report can name it instead of naming the team.
    Measured on 2026-27: OTT's `l -1 / otl +1` with gf and ga agreeing at
    44/44 localises to LAK 6-5 OTT on 2026-10-29, end OT - the article calls
    it an overtime loss and the standings page counts it as a regulation one.
    """
    notes = []
    for abbrev, delta in sorted(by_abbrev.items()):
        if S.shape_of(delta) != S.SHAPE_RESULT_LABEL:
            continue
        fid = F.abbrev_index(year).get(abbrev)
        candidates = []
        for home, away, hg, ag, end, *_rest in settled:
            if fid not in (home, away) or end not in ("OT", "SO"):
                continue
            loser = away if hg > ag else home
            # The disagreement is about a LOSS type when otl moved, so the
            # candidate is a game this team LOST in OT or SO.
            if delta.get("otl", 0) != 0 and loser != fid:
                continue
            candidates.append("%s %d-%d %s (%s)"
                              % (F.abbrev_for(home, year), hg, ag,
                                 F.abbrev_for(away, year), end))
        if len(candidates) == 1:
            notes.append("%s localises to %s" % (abbrev, candidates[0]))
        elif candidates:
            notes.append("%s has %d candidate game(s)"
                         % (abbrev, len(candidates)))
    return "; ".join(notes)


def _write(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8", newline="\n")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
