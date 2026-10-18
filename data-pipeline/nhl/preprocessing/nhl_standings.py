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


class StandingsDisagree(RuntimeError):
    """One page states two different values for the same team's field."""


def parse_template(text):
    """{abbrev: {win, loss, OTloss, gf, ga}} from one template page."""
    seen = {}
    for field, team, value in FIELD.findall(text):
        row = seen.setdefault(team.upper(), {})
        number = int(value)
        if field in row and row[field] != number:
            raise StandingsDisagree(
                "the standings page states %s_%s as both %d and %d. A "
                "repeated sub-table must agree; this is fatal rather than "
                "resolved by picking one."
                % (field, team, row[field], number))
        row[field] = number
    return seen


def tiebreak_tokens(text):
    return {t.upper() for t in TIEBREAK.findall(text)}


def update_stamp(text):
    """What the page says about its own currency: 'complete', or a date.

    An open season's page carries a date rather than `complete`, so it can
    lag the game logs - which is why reconciliation is advisory there and
    fatal for a completed season.
    """
    found = UPDATE.search(text)
    return found.group(1).strip() if found else ""


def read_season(texts):
    """Merge the season's template(s). `texts` is {title: wikitext}.

    Merging across pages uses the same agreement rule as merging within one,
    so a team listed by two templates must be listed identically.
    """
    merged, tokens, stamps = {}, set(), {}
    for title, text in sorted(texts.items()):
        tokens |= tiebreak_tokens(text)
        stamps[title] = update_stamp(text)
        for team, row in parse_template(text).items():
            target = merged.setdefault(team, {})
            for field, value in row.items():
                if field in target and target[field] != value:
                    raise StandingsDisagree(
                        "%s_%s is %d in one of this season's standings "
                        "templates and %d in another (%r)."
                        % (field, team, target[field], value, title))
                target[field] = value
    return dict(teams=merged, tiebreak=tokens, update=stamps)


def require_complete(season_label, standings, expected_abbrevs):
    """Every team that played must have a standings row, and vice versa.

    Without this the reconciliation would quietly cover half a league - which
    is the shape of failure the 2019-20 conference templates could have
    produced had the division pattern been assumed.
    """
    have = set(standings["teams"])
    want = set(expected_abbrevs)
    missing, extra = sorted(want - have), sorted(have - want)
    if missing or extra:
        raise RuntimeError(
            "%s standings cover %d team(s), the season has %d. missing=%s "
            "unexpected=%s. A partial table would make the goals channel "
            "silently cover only part of the league."
            % (season_label, len(have), len(want), missing, extra))
    thin = sorted(t for t, row in standings["teams"].items()
                  if "gf" not in row or "ga" not in row)
    if thin:
        raise RuntimeError(
            "%s standings carry no gf/ga for %s, which is the only channel "
            "that sees a score changed without changing the winner."
            % (season_label, thin))
    return True
