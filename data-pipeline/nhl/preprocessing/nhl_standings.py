"""Read a season's standings template(s) into {abbreviation: record}.

This is the channel that makes a wrong score detectable at all. A score
altered WITHOUT changing the winner moves `gf` and `ga` and nothing else, and
the NHL publishes no period scores, so there is no box-score analogue to check
against. Round 4 established it and measured its baseline.

Three shapes, all read by the same function because the fields are named:

  * one WHOLE-LEAGUE template from 2020-21;
  * one per DIVISION before that;
  * one per CONFERENCE in 2019-20, the season COVID stopped.

TWO THINGS THAT WOULD SILENTLY DOUBLE-COUNT OR MISLEAD, both measured:

  * The whole-league template is a `#switch` over sub-tables, so one page
    declares a team several times - 52 entries for 32 teams in 2021-22,
    because the wild-card sub-tables repeat teams the division ones already
    list. Repeats are DEDUPED BY REQUIRING AGREEMENT rather than by picking a
    sub-table: a disagreement between two copies of one team's row is fatal,
    which makes the dedupe self-checking instead of a guess about ordering.
  * The tiebreak column CHANGES MEANING - `RW` (regulation wins) from
    2020-21, `ROW` (regulation and overtime wins) before. Comparing a ROW
    column against regulation wins made all 30 teams of 2016-17 look wrong by
    3 to 9. Worse than round 4 recorded: it moves WITHIN one page, which the
    agreement rule above caught on 2021-22's `tiebr_WPG` (31 and 32). So the
    column is NOT READ at all - only its header token is noted.

PHASE 3: STRICT FOR A COMPLETED SEASON, CLASSIFIED FOR THE OPEN ONE
-------------------------------------------------------------------
Every raise above stays exactly as it was for a COMPLETED season. For the
open season it becomes a CLASS, because the failure is GLOBAL rather than
per-game: the NFL-style quarantine sets one disagreeing GAME aside, but the
standings are a season-level channel, so a single editor's half-finished edit
made every 2026-27 game unbuildable. Measured 2026-10-30:
`Template:2026-27 NHL standings` carried LAK in two sub-tables, Pacific saying
`4-6-2, gf 34, ga 42` and Western Wild Card saying `3-6-2, gf 28, ga 37` -
one 6-5 win written into one table and not the other. Transient, and it will
recur; a daily refresh cannot wait for an editor.

A CONFLICTED FIELD IS OMITTED, NOT RESOLVED. When the check is advisory the
field is dropped from the record rather than taken from the first sub-table
seen. Picking a copy would invent a standings value and feed it to the
reconciliation, which is the only channel that can see a changed score -
so a wrong value there is worse than no value. `build_nhl_tables.diffs`
already skips a field the standings do not carry, so an omitted field
degrades the check instead of corrupting it. The unknown-is-never-a-zero rule,
applied to a season table.
"""
import re

# `tiebr` is DELIBERATELY NOT READ. Round 4 warned that comparing it is the
# error, because its meaning moves between seasons; measured here, it moves
# WITHIN ONE PAGE - 2021-22 states tiebr_WPG as both 31 and 32, because the
# division and wild-card sub-tables of the same `#switch` use the two
# different definitions. Not reading it removes the trap instead of tolerating
# it, and nothing downstream wants the column.
FIELD = re.compile(r"\|\s*(win|loss|OTloss|gf|ga)_([A-Za-z]{2,4})\s*"
                   r"=\s*(-?\d+)")
TEAM = re.compile(r"\|\s*team\d*\s*=\s*([A-Za-z]{2,4})\s*[|}\n]")
# Which quantity the tiebreak column holds. Both tokens can appear on one
# page, so the record keeps the set rather than one value.
TIEBREAK = re.compile(r"\{\{\s*abbr\s*\|\s*(RW|ROW)\s*\|", re.I)
UPDATE = re.compile(r"\|\s*update\s*=\s*([^|\n}]+)")

# ---------------------------------------------------------------- classes
# ONE NAME PER FAILURE SHAPE, defined here so `/health`, the refresh summary
# and the verifier all key on the same string. A code is what a comparison
# keys on and prose is for reading - the gate's false REGRESSED is the
# recorded reason that distinction is not optional.
CLASS_RECONCILED = "reconciled"
CLASS_SELF_INCONSISTENT = "standings_self_inconsistent"
CLASS_INCOMPLETE = "standings_incomplete"
CLASS_LAG = "standings_lag"
CLASS_PROBABLE_ERROR = "standings_probable_error"

# Most serious first. A season can be in more than one of these at once, so
# the report carries every class it is in AND a single primary for a caller
# that needs one word.
CLASS_ORDER = (CLASS_SELF_INCONSISTENT, CLASS_PROBABLE_ERROR,
               CLASS_INCOMPLETE, CLASS_LAG, CLASS_RECONCILED)


class StandingsDisagree(RuntimeError):
    """One page states two different values for the same team's field."""


def parse_template(text, strict=True):
    """({abbrev: {win, loss, OTloss, gf, ga}}, conflicts) from one page.

    `strict` raises on a self-inconsistent repeat, exactly as before.
    Otherwise the conflicted FIELD is omitted and reported - see the module
    docstring on why omitting beats picking.
    """
    seen, conflicts = {}, []
    for field, team, value in FIELD.findall(text):
        abbrev = team.upper()
        row = seen.setdefault(abbrev, {})
        number = int(value)
        if field in row:
            existing = row[field]
            # ALREADY CONFLICTED IS CHECKED FIRST. With the order the other
            # way round, a third sub-table listing the same field compared
            # `None != number`, reported the conflict a second time and
            # inflated the count - a wrong number in the very report that
            # exists to be read.
            if existing is None:
                continue
            if existing == number:
                continue
            if strict:
                raise StandingsDisagree(
                    "the standings page states %s_%s as both %d and %d. A "
                    "repeated sub-table must agree; this is fatal rather "
                    "than resolved by picking one."
                    % (field, abbrev, existing, number))
            conflicts.append(dict(scope="within one template", field=field,
                                  team=abbrev,
                                  values=sorted({existing, number})))
            row[field] = None
            continue
        row[field] = number
    # An omitted field is absent, not None, so `row.get(field)` downstream
    # reads the same as a field the page never carried.
    for row in seen.values():
        for field in [f for f, v in row.items() if v is None]:
            del row[field]
    return seen, conflicts


def tiebreak_tokens(text):
    return {t.upper() for t in TIEBREAK.findall(text)}


def update_stamp(text):
    """What the page says about its own currency: 'complete', or a date.

    An open season's page carries a date rather than `complete`, so it can
    lag the game logs - which is why reconciliation is advisory there and
    fatal for a completed season.

    READ WITHOUT PARSING THE FIELDS, which is what makes the strict/advisory
    decision possible at all: the builder has to know whether a season is
    complete BEFORE it parses the numbers, and this regex does not care
    whether the numbers agree.
    """
    found = UPDATE.search(text)
    return found.group(1).strip() if found else ""


def stamps_say_complete(texts):
    """The pages' own word, from `texts` alone. No field parsing."""
    stamps = {update_stamp(text).strip().lower() for text in texts.values()}
    return bool(stamps) and all(s == "complete" for s in stamps)


def read_season(texts, strict=True):
    """Merge the season's template(s). `texts` is {title: wikitext}.

    Merging across pages uses the same agreement rule as merging within one,
    so a team listed by two templates must be listed identically.
    """
    merged, tokens, stamps, conflicts = {}, set(), {}, []
    for title, text in sorted(texts.items()):
        tokens |= tiebreak_tokens(text)
        stamps[title] = update_stamp(text)
        rows, within = parse_template(text, strict=strict)
        for entry in within:
            conflicts.append(dict(entry, title=title))
        for team, row in rows.items():
            target = merged.setdefault(team, {})
            for field, value in row.items():
                if field in target:
                    existing = target[field]
                    # Already-conflicted first, for the same reason as in
                    # parse_template.
                    if existing is None:
                        continue
                    if existing == value:
                        continue
                    if strict:
                        raise StandingsDisagree(
                            "%s_%s is %d in one of this season's standings "
                            "templates and %d in another (%r)."
                            % (field, team, existing, value, title))
                    conflicts.append(dict(
                        scope="between templates", field=field, team=team,
                        values=sorted({existing, value}), title=title))
                    target[field] = None
                    continue
                target[field] = value
    for row in merged.values():
        for field in [f for f, v in row.items() if v is None]:
            del row[field]
    return dict(teams=merged, tiebreak=tokens, update=stamps,
                conflicts=conflicts)


def require_complete(season_label, standings, expected_abbrevs, strict=True):
    """Every team that played must have a standings row, and vice versa.

    Without this the reconciliation would quietly cover half a league - which
    is the shape of failure the 2019-20 conference templates could have
    produced had the division pattern been assumed.

    `strict` raises as before. Otherwise the problems are returned for
    classification, because in the open season a missing or thin row is an
    edit in flight rather than a corpus fault - AND because a conflicted
    field is now omitted, which can MAKE a row thin. Raising there would
    turn every self-inconsistency into the incompleteness failure instead.
    """
    problems = []
    have = set(standings["teams"])
    want = set(expected_abbrevs)
    missing, extra = sorted(want - have), sorted(have - want)
    if missing or extra:
        message = (
            "%s standings cover %d team(s), the season has %d. missing=%s "
            "unexpected=%s. A partial table would make the goals channel "
            "silently cover only part of the league."
            % (season_label, len(have), len(want), missing, extra))
        if strict:
            raise RuntimeError(message)
        problems.append(dict(kind="coverage", detail=message,
                             missing=missing, unexpected=extra))
    thin = sorted(t for t, row in standings["teams"].items()
                  if "gf" not in row or "ga" not in row)
    if thin:
        message = (
            "%s standings carry no gf/ga for %s, which is the only channel "
            "that sees a score changed without changing the winner."
            % (season_label, thin))
        if strict:
            raise RuntimeError(message)
        problems.append(dict(kind="thin", detail=message, teams=thin))
    return problems


RESULT_FIELDS = ("w", "l", "otl")
GOAL_FIELDS = ("gf", "ga")

SHAPE_GAMES_MISSING = "games_missing"
SHAPE_RESULT_LABEL = "result_label"
SHAPE_SCORE_ONLY = "score_only"
SHAPE_MIXED = "mixed"


def shape_of(delta):
    """What ONE team's {field: computed - stated} looks like.

    THE SIGN DIRECTION, MEASURED 2026-10-31, AND THE ORIGINAL COMMENT HAD IT
    BACKWARDS. `diffs` is computed MINUS stated, and in an open season
    `computed` is built only from games BOTH team articles carry and agree
    on. So a NEGATIVE difference means the TEAM ARTICLES LAG THE STANDINGS
    PAGE - the page already counts a game the articles have not both
    recorded. `build_nhl_tables._advisory` said the opposite ("all-negative
    -> the standings page lags the game logs") and has been removed rather
    than corrected in place, because two copies of one rule is how a gate
    ends up grading the wrong thing. Verified on 2026-27: 8 of 9 flagged
    teams were purely negative, and every one of them was either in a
    quarantined game, in a one-article game, or - CGY and SEA - short by a
    game the standings already counted.

    FOUR SHAPES, and the third is the one this whole channel exists for:

      games_missing  every difference negative and the result counts down.
                     The articles lag. Expected, and the steady state of an
                     open season.
      result_label   the w/l/otl differences CANCEL and the goals agree
                     exactly. The same games are counted, one of them with a
                     different result type - an OT/SO loss recorded as a
                     regulation loss, or the reverse. Localisable to a game.
      score_only     the goals differ and the result counts AGREE. **A score
                     changed without changing the winner**, which is the only
                     thing the standings channel can see and nothing else
                     can. The most serious reading available.
      mixed          none of the above cleanly.
    """
    results = {f: v for f, v in delta.items() if f in RESULT_FIELDS}
    goals = {f: v for f, v in delta.items() if f in GOAL_FIELDS}
    if goals and not results:
        return SHAPE_SCORE_ONLY
    if results and not goals and sum(results.values()) == 0:
        return SHAPE_RESULT_LABEL
    if delta and all(v < 0 for v in delta.values()):
        return SHAPE_GAMES_MISSING
    return SHAPE_MIXED


def classify(conflicts, completeness_problems, flagged):
    """(primary class, every class, detail) for an advisory season.

    `flagged` is {team: {field: computed - stated}} from the reconciliation.
    Each team's difference is given a SHAPE (see `shape_of`) rather than only
    a sign, because "mixed sign" alone cannot tell a mislabelled result from
    a changed score, and those are different problems with different
    urgencies. A self-inconsistent page is its own class, because the page
    disagrees with ITSELF rather than with the game logs - and the first copy
    is not silently picked.
    """
    classes, details = [], []
    if conflicts:
        classes.append(CLASS_SELF_INCONSISTENT)
        shown = ", ".join(
            "%s_%s %s (%s)" % (c["field"], c["team"],
                               " vs ".join(str(v) for v in c["values"]),
                               c["scope"])
            for c in conflicts[:4])
        details.append("%d conflicted field(s): %s%s"
                       % (len(conflicts), shown,
                          "" if len(conflicts) <= 4 else ", ..."))
    if completeness_problems:
        classes.append(CLASS_INCOMPLETE)
        details.append("; ".join(p["detail"] for p in
                                 completeness_problems[:2]))
    if flagged:
        shapes = {team: shape_of(delta) for team, delta in flagged.items()}
        by_shape = {}
        for team, shape in sorted(shapes.items()):
            by_shape.setdefault(shape, []).append(team)
        suspect = (by_shape.get(SHAPE_SCORE_ONLY, [])
                   + by_shape.get(SHAPE_RESULT_LABEL, [])
                   + by_shape.get(SHAPE_MIXED, []))
        if suspect:
            classes.append(CLASS_PROBABLE_ERROR)
            for shape in (SHAPE_SCORE_ONLY, SHAPE_RESULT_LABEL, SHAPE_MIXED):
                teams = by_shape.get(shape)
                if teams:
                    details.append("%s on %s" % (shape, ", ".join(teams)))
            missing = by_shape.get(SHAPE_GAMES_MISSING)
            if missing:
                details.append("%d team(s) short of the standings page "
                               "(articles lagging)" % len(missing))
        elif len(flagged) > 1:
            classes.append(CLASS_LAG)
            details.append("%d team(s), all short of the standings page; the "
                           "TEAM ARTICLES lag the page" % len(flagged))
        else:
            classes.append(CLASS_PROBABLE_ERROR)
            details.append("one team short of the standings page: %s"
                           % ", ".join(sorted(flagged)))
    if not classes:
        return CLASS_RECONCILED, [CLASS_RECONCILED], "standings reconcile"
    primary = next(c for c in CLASS_ORDER if c in classes)
    return primary, classes, "; ".join(details)
