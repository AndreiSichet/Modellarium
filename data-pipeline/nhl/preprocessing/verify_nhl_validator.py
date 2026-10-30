"""The negative suite: plant each defect the gate exists for, on a COPY.

A check that has never failed has not been verified. Each plant below is
applied to a copy of the raw corpus, the named check must catch it, and the
copy is thrown away - the real `data/raw/` is hashed before and after and must
be unchanged.

Plus two things that are not plants:
  * a CONTROL with nothing planted, which must pass. Without it a suite that
    fails for an unrelated reason reads as "every plant caught".
  * a NO-OP GUARD: a plant that changes no bytes raises. A plant that
    silently fails to apply reports its check as blind, which is how round 5
    nearly recorded five working channels as broken.
"""
import argparse
import hashlib
import importlib
import io
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import nhl_franchises as F                                   # noqa: E402

REAL_RAW = HERE.parent.parent / "data" / "nhl" / "raw"
# Two completed seasons and the open one. 2014-15 carries a listed unusable
# game, which the reverse-guard plant needs.
SEASONS = [2014, 2021, 2026]


class PlantDidNothing(RuntimeError):
    """A plant that changes no bytes is a harness bug, not a passing check."""


class Harness:
    """A throwaway copy of the raw corpus, with the modules pointed at it."""

    def __init__(self, seasons):
        self.seasons = seasons
        self.root = Path(tempfile.mkdtemp(prefix="nhl-negative-"))
        self.raw = self.root / "raw"
        self.out = self.root / "processed"
        self.raw.mkdir(parents=True)
        self.out.mkdir(parents=True)
        for year in seasons:
            name = "%d-%02d" % (year, (year + 1) % 100)
            shutil.copytree(REAL_RAW / name, self.raw / name)
        self.modules = self._load()

    def _load(self):
        """Fresh module objects whose RAW/OUT point at the copy."""
        for name in ("build_nhl_tables", "validate_nhl_games",
                     "parse_nhl_wikitext", "nhl_standings",
                     "nhl_standings_exceptions", "nhl_franchises"):
            if name in sys.modules:
                del sys.modules[name]
        build = importlib.import_module("build_nhl_tables")
        build.RAW = self.raw
        build.OUT = self.out
        validate = importlib.import_module("validate_nhl_games")
        validate.OUT = self.out
        validate.B = build
        return dict(build=build, validate=validate,
                    parse=importlib.import_module("parse_nhl_wikitext"))

    def reload(self):
        self.modules = self._load()

    def path(self, year, team):
        name = "%d-%02d" % (year, (year + 1) % 100)
        title = F.article_title(year, team)
        import re
        slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")[:120]
        return self.raw / name / (slug + ".wikitext")

    def edit(self, year, team, old, new, count=1):
        """Replace text in one article and REFUSE a no-op."""
        path = self.path(year, team)
        text = path.read_text(encoding="utf-8")
        if old not in text:
            raise PlantDidNothing(
                "the plant's target text is not in %s: %r" % (path.name, old[:70]))
        changed = text.replace(old, new, count)
        if changed == text:
            raise PlantDidNothing("the plant changed no bytes in %s" % path.name)
        path.write_text(changed, encoding="utf-8", newline="\n")
        return path

    def drop(self, year, team):
        path = self.path(year, team)
        if not path.is_file():
            raise PlantDidNothing("nothing to remove at %s" % path.name)
        path.unlink()
        return path

    def a_game(self, year, want_end=None, want_decided=True):
        """One game and both its articles' row lines.

        Returns (row_a, row_b, [(path, line)]). The plant then rewrites the
        same game in BOTH copies, which is the case the standings channel
        exists for - a one-copy edit is caught by pairing and repaired, so a
        plant that only edits one copy tests the wrong thing.
        """
        build = self.modules["build"]
        rows, _st, _bad = build.load_season(year)
        played = [r for r in rows if r["played"]]
        pairs, _singles = build.pair_rows(played)
        for a, b in pairs:
            if not build.agree(a, b):
                continue
            if want_end and a["game_end"] != want_end:
                continue
            if want_decided and a["goals_for"] == a["goals_against"]:
                continue
            lines = []
            for row in (a, b):
                path = self.raw / ("%d-%02d" % (year, (year + 1) % 100))
                import re
                slug = re.sub(r"[^A-Za-z0-9]+", "_",
                              row["article"]).strip("_")[:120]
                path = path / (slug + ".wikitext")
                text = path.read_text(encoding="utf-8")
                hit = None
                for line in text.split("\n"):
                    if not line.startswith("|") or line.startswith("|-"):
                        continue
                    cleaned = self.modules["parse"].clean(line)
                    if ("%d\u2013%d" % (row["goals_for"],
                                        row["goals_against"]) in cleaned
                            or "%d\u2013%d" % (row["goals_against"],
                                               row["goals_for"]) in cleaned):
                        if str(row["date"].day) in cleaned:
                            hit = line
                            break
                if hit is None:
                    break
                lines.append((path, hit))
            if len(lines) == 2:
                return a, b, lines
        raise PlantDidNothing(
            "no %s game with an agreeing pair could be located in %d"
            % (want_end or "decided", year))

    def rewrite(self, path, old_line, new_line):
        text = path.read_text(encoding="utf-8")
        if old_line not in text:
            raise PlantDidNothing("row not found in %s" % path.name)
        changed = text.replace(old_line, new_line, 1)
        if changed == text:
            raise PlantDidNothing("the plant changed no bytes in %s"
                                  % path.name)
        path.write_text(changed, encoding="utf-8", newline="\n")

    def build(self, seasons=None, discover=False):
        argv = ["--out", str(self.out),
                "--seasons", ",".join(str(y) for y in (seasons or self.seasons))]
        if discover:
            argv.append("--discover")
        return self.modules["build"].main(argv)

    def validate(self, seasons=None):
        argv = ["--out", str(self.out),
                "--seasons", ",".join(str(y) for y in (seasons or self.seasons))]
        return self.modules["validate"].main(argv)

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


def corpus_hash():
    """One digest over every raw file, so a plant cannot touch the real one."""
    digest = hashlib.sha256()
    for path in sorted(REAL_RAW.rglob("*")):
        if path.is_file():
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def expect_raises(label, call, needle=""):
    """The plant must make the build or the gate refuse."""
    try:
        code = call()
    except PlantDidNothing:
        raise
    except Exception as error:                               # noqa: BLE001
        text = "%s: %s" % (type(error).__name__, error)
        if needle and needle.lower() not in text.lower():
            return False, "raised, but not about %r: %s" % (needle, text[:110])
        return True, text.split("\n")[0][:110]
    if code != 0:
        return True, "exit %d" % code
    return False, "NOT CAUGHT - completed with exit 0"


# ------------------------------------------------------------------ plants
def _bump(line, old_pair, new_pair):
    """Rewrite one score in a row line, in either orientation."""
    for a, b in (old_pair, (old_pair[1], old_pair[0])):
        token = "%d\u2013%d" % (a, b)
        if token in line:
            flipped = (a, b) != old_pair
            want = (new_pair[1], new_pair[0]) if flipped else new_pair
            return line.replace(token, "%d\u2013%d" % want, 1)
    raise PlantDidNothing("score %r not in the row line" % (old_pair,))


def plant_score_both_copies(h):
    """(a) score changed in BOTH copies, winner unchanged -> the standings.

    THE HARD CASE, and the reason the standings channel exists: pairing sees
    nothing, the marker sees nothing, the record column sees nothing, because
    the winner did not change and both copies agree with each other.
    """
    year = 2021
    a, b, lines = h.a_game(year)
    winner_goals = max(a["goals_for"], a["goals_against"])
    loser_goals = min(a["goals_for"], a["goals_against"])
    old_pair = (a["goals_for"], a["goals_against"])
    new_high = winner_goals + 2
    new_pair = ((new_high, loser_goals)
                if a["goals_for"] > a["goals_against"]
                else (loser_goals, new_high))
    for path, line in lines:
        h.rewrite(path, line, _bump(line, old_pair, new_pair))
    return expect_raises("a", lambda: h.build([year]), "standings")


def plant_winner_flipped(h):
    """(b) winner flipped in both copies -> the marker and the record."""
    year = 2021
    a, b, lines = h.a_game(year)
    old_pair = (a["goals_for"], a["goals_against"])
    new_pair = (a["goals_against"], a["goals_for"])
    for path, line in lines:
        h.rewrite(path, line, _bump(line, old_pair, new_pair))
    h.build([year], discover=True)
    validate = h.modules["validate"]
    validate.MARKER_DISAGREEMENTS.clear()
    validate.RECORD_DIVERGENCES.clear()
    validate.check_marker_and_record(year, validate.Checks())
    caught_marker = len(validate.MARKER_DISAGREEMENTS)
    caught_record = len(validate.RECORD_DIVERGENCES)
    return (caught_marker > 0 and caught_record > 0,
            "the marker flagged %d row(s) and the record column %d - both "
            "channels are independent of the score"
            % (caught_marker, caught_record))


def plant_reg_as_ot(h):
    """(c) REG relabelled OT in both copies -> the record's OTL count."""
    year = 2021
    a, b, lines = h.a_game(year, want_end="REG")
    changed = 0
    for path, line in lines:
        # Append the label inside the score cell, which is one of the two
        # places the corpus carries it and needs no OT column to exist.
        pair = "%d\u2013%d" % (a["goals_for"], a["goals_against"])
        flipped = "%d\u2013%d" % (a["goals_against"], a["goals_for"])
        token = pair if pair in line else flipped
        h.rewrite(path, line, line.replace(token, token + " OT", 1))
        changed += 1
    h.build([year], discover=True)
    validate = h.modules["validate"]
    validate.RECORD_DIVERGENCES.clear()
    validate.check_marker_and_record(year, validate.Checks())
    return (len(validate.RECORD_DIVERGENCES) > 0,
            "relabelled %d cop(ies); the record column's OTL count flagged "
            "%d divergence(s)" % (changed, len(validate.RECORD_DIVERGENCES)))


def plant_one_copy(h):
    """(d) one copy only -> pairing, then repaired by the standings."""
    year = 2021
    import re
    text = h.path(year, "Ottawa Senators").read_text(encoding="utf-8")
    found = re.search(r"\|\s*(\d)–(\d)\s*\|\|", text)
    old = found.group(0)
    bumped = old.replace("%s–%s" % (found.group(1), found.group(2)),
                         "%d–%s" % (int(found.group(1)) + 3,
                                         found.group(2)))
    h.edit(year, "Ottawa Senators", old, bumped)
    h.build([year], discover=True)
    repairs = (h.out / "nhl_repairs.csv").read_text(encoding="utf-8")
    games = (h.out / "nhl_games_final.csv").read_text(encoding="utf-8")
    derived = sum(1 for line in games.splitlines()[1:] if line.endswith(",1,0")
                  or ",1," in line.split(",")[-2:][0])
    return (("Ottawa" in repairs or repairs.count("\n") > 1),
            "repairs recorded: %d line(s)" % max(0, repairs.count("\n") - 1))


def plant_article_removed(h):
    """(e) one article removed -> completeness."""
    h.drop(2021, "Calgary Flames")
    return expect_raises("e", lambda: h.build([2021]), "coverage")


def plant_duplicate_game(h):
    """(f) a duplicate game under a new id -> uniqueness / the residual rule.

    The row is located THROUGH THE PARSER. An earlier version duplicated the
    first `|-` row in the article, which is a PRESEASON row and therefore
    invisible to the regular-season parse - so the plant changed bytes,
    changed nothing the pipeline reads, and reported the check as blind. A
    plant has to land where the thing it tests actually looks.
    """
    year = 2021
    _a, _b, lines = h.a_game(year)
    path, line = lines[0]
    text = path.read_text(encoding="utf-8")
    h.rewrite(path, line, line + chr(10) + "|-" + chr(10) + line)
    if path.read_text(encoding="utf-8") == text:
        raise PlantDidNothing("the duplicate did not land")
    # The duplicate becomes a SINGLETON - the opponent carries one copy, this
    # article now carries two - and the standings decline to want it. Caught
    # by the builder's unnamed-residual rule, which exists because without it
    # the pipeline would absorb the duplicate and exit 0.
    return expect_raises("f", lambda: h.build([year]), "residual")


def plant_in_progress(h):
    """(g) a score inside the in-progress window -> the guard.

    REWRITTEN 2026-10-30 WITH THE RULE IT TESTS, and the old assertion is
    quoted here because it was right about the old rule:

        return (today and yesterday and not older, ...)

    Under the full-day rule a game on D was in progress until D+2, so
    "yesterday" was in progress. The rule is now whether the game's date has
    ENDED in the latest US zone it could be played in - midnight Pacific on
    D+1 - which 320 replayed revisions showed is safe: 0 of them ever carried
    a partial score with its markers already set, and a complete score with
    markers appeared a median 0.6 h BEFORE the date ended.
    """
    parse = h.modules["parse"]
    import datetime as dt
    now = dt.datetime(2026, 10, 19, 12, 0, tzinfo=dt.timezone.utc)
    today = parse.in_progress(dt.date(2026, 10, 19), now)
    yesterday = parse.in_progress(dt.date(2026, 10, 18), now)
    older = parse.in_progress(dt.date(2026, 10, 16), now)

    # THE BOUNDARY, BOTH SIDES OF IT. The saving is exactly the hour of
    # midnight Pacific, so an off-by-one here is the whole change.
    ended = parse.date_ended_utc(dt.date(2026, 10, 18))
    just_before = parse.in_progress(
        dt.date(2026, 10, 18), ended - dt.timedelta(minutes=1))
    just_after = parse.in_progress(
        dt.date(2026, 10, 18), ended + dt.timedelta(minutes=1))

    # THE POSITIVE CONTROL ON THE NARROWING: the rule this replaced said
    # yesterday was still in progress at this instant.
    old_rule_said_in_progress = (now.date() - dt.date(2026, 10, 18)).days <= 1

    return (today and not yesterday and not older
            and just_before and not just_after
            and old_rule_said_in_progress,
            "today=%s yesterday=%s three days ago=%s; the 2026-10-18 date "
            "ends %s, in progress one minute before=%s and after=%s; the "
            "full-day rule it replaced called yesterday in-progress=%s"
            % (today, yesterday, older, ended.isoformat(), just_before,
               just_after, old_rule_said_in_progress))


def plant_partial_with_markers(h):
    """(p) a mid-game score in ONE copy, markers set, date not ended.

    THE FAILURE THE NARROWED RULE MUST NOT ACCEPT, and the reason the rule
    needed measuring rather than reasoning about. An editor watching a game
    can write the running score AND set the outcome marker before the game is
    over - at which point both the score and the markers look complete. 320
    replayed revisions contained 0 such rows, so the risk is small; "small"
    is not "impossible", and the guard has to hold on the one that does
    happen.

    Planted in ONE copy on purpose. The pairing channel repairs a one-copy
    score edit, so without the date guard this row would be REPAIRED into
    the corpus rather than refused - which is worse than a visible failure.
    """
    import datetime as dt
    parse = h.modules["parse"]
    year = h.seasons[0]
    row_a, _row_b, lines = h.a_game(year, want_end="REG")
    path, line = lines[0]

    # A score mid-game: the eventual winner is a goal short of its final
    # tally, which is exactly what a live-updating editor leaves behind.
    mid = "%d–%d" % (max(row_a["goals_for"] - 1, 0),
                         row_a["goals_against"])
    final = "%d–%d" % (row_a["goals_for"], row_a["goals_against"])
    flipped = "%d–%d" % (row_a["goals_against"], row_a["goals_for"])
    if final in line:
        h.rewrite(path, line, line.replace(final, mid, 1))
    elif flipped in line:
        h.rewrite(path, line,
                  line.replace(flipped, "%d–%d"
                               % (row_a["goals_against"],
                                  max(row_a["goals_for"] - 1, 0)), 1))
    else:
        raise PlantDidNothing("neither orientation of the score is in the row")

    # THE GUARD ITSELF, asked directly: on the game's own date, before that
    # date has ended, the row is in progress whatever it says.
    ended = parse.date_ended_utc(row_a["date"])
    during = ended - dt.timedelta(hours=4)
    refused = parse.in_progress(row_a["date"], during)
    accepted_after = not parse.in_progress(row_a["date"], ended)

    # AND THROUGH THE PARSER, WITH `now` INSIDE THE WINDOW. The first
    # version of this called load_season, which uses the real clock - and a
    # 2014 game is long final, so the guard correctly did not apply and the
    # planted row reached the corpus. That was the plant testing the wrong
    # thing: the guard is about RECENCY, so the only honest way to exercise
    # it on an archived game is to tell the parser that now is inside the
    # game's own window, which is what `parse_article`'s `now` is for.
    text = path.read_text(encoding="utf-8")
    during_rows, _bad = parse.parse_article(
        text, year, row_a["team_fid"], row_a["article"], now=during)
    after_rows, _bad2 = parse.parse_article(
        text, year, row_a["team_fid"], row_a["article"],
        now=ended + dt.timedelta(hours=1))

    def played_on(rows):
        return [r for r in rows
                if r["date"] == row_a["date"] and r["played"]]

    reached_during = played_on(during_rows)
    reached_after = played_on(after_rows)

    return (refused and accepted_after
            and not reached_during and len(reached_after) == 1,
            "the %s date ends %s; in progress four hours before=%s, final "
            "at the boundary=%s; with `now` inside the window the planted "
            "row reached the parse %d time(s), and one hour after the date "
            "ended %d time(s)"
            % (row_a["date"], ended.isoformat(), refused, accepted_after,
               len(reached_during), len(reached_after)))


def plant_west_coast_in_progress(h):
    """(o) a west-coast game at 02:00Z with a score at 04:30Z is UNPLAYED.

    7 p.m. Pacific is 02:00 UTC the NEXT day, so the article's local date is
    the day before the drop. A score at 04:30Z is two and a half hours in and
    the game may still be in overtime.
    """
    parse = h.modules["parse"]
    import datetime as dt
    local_date = dt.date(2026, 10, 18)
    drop = dt.datetime(2026, 10, 19, 2, 0, tzinfo=dt.timezone.utc)
    seen = dt.datetime(2026, 10, 19, 4, 30, tzinfo=dt.timezone.utc)

    by_date = parse.in_progress(local_date, seen)
    by_instant = parse.in_progress(local_date, seen, drop)
    settled = parse.in_progress(local_date, drop + dt.timedelta(days=2))

    # The positive control on the fix: the rule this replaced called it final.
    nominal = dt.datetime(local_date.year, local_date.month, local_date.day,
                          23, tzinfo=dt.timezone.utc)
    old_rule_said_in_progress = seen < nominal + dt.timedelta(
        hours=parse.NHL_HOURS_UNTIL_FINAL)

    return (by_date and by_instant and not settled
            and not old_rule_said_in_progress,
            "in-progress at 04:30Z: full-day=%s instant=%s; two days "
            "later=%s; under the old nominal 23:00Z rule it would have "
            "been FINAL at 04:30Z (in-progress=%s)"
            % (by_date, by_instant, settled, old_rule_said_in_progress))


def plant_unknown_link(h):
    """(h) an unknown link target -> the authored identity table."""
    year = 2021
    h.edit(year, "Dallas Stars",
           "%s Minnesota Wild season" % F.season_label(year),
           "%s Hartford Whalers season" % F.season_label(year))
    return expect_raises("h", lambda: h.build([year]), "franchise table")


def plant_january_date(h):
    """(i) a January date in an October-start season -> date derivation."""
    parse = h.modules["parse"]
    october = parse.parse_date("October 12", 2021)
    january = parse.parse_date("January 12", 2021)
    june = parse.parse_date("June 2", 2021)
    return (october.year == 2021 and january.year == 2022
            and june.year == 2022,
            "October -> %s, January -> %s, June -> %s"
            % (october, january, june))


def plant_at_marker(h):
    """(j) an `@` away marker -> home/away, and one home side per game."""
    parse = h.modules["parse"]
    away = bool(parse.AWAY_MARKER.match("@ Anaheim"))
    away_word = bool(parse.AWAY_MARKER.match("at Anaheim"))
    home = bool(parse.AWAY_MARKER.match("Anaheim"))
    # And the property that found the bug: every paired game has exactly one
    # home side.
    build = h.modules["build"]
    rows, _st, _bad = build.load_season(2021)
    played = [r for r in rows if r["played"]]
    pairs, _singles = build.pair_rows(played)
    both = sum(1 for a, b in pairs if a["away"] == b["away"])
    return (away and away_word and not home and both == 0,
            "'@ X' away=%s, 'at X' away=%s, 'X' away=%s; pairs where both "
            "copies claim home: %d of %d" % (away, away_word, home,
                                             both, len(pairs)))


def plant_open_season_quarantine(h):
    """(k) one copy disagreeing in the OPEN season -> quarantine, not fatal."""
    year = 2026
    import re
    text = h.path(year, "Toronto Maple Leafs").read_text(encoding="utf-8")
    found = re.search(r"\|\s*(\d)–(\d)\s*\|\|", text)
    old = found.group(0)
    h.edit(year, "Toronto Maple Leafs", old,
           old.replace("%s–%s" % (found.group(1), found.group(2)),
                       "%d–%s" % (int(found.group(1)) + 4,
                                       found.group(2))))
    code = h.build([year])
    rows = (h.out / "nhl_quarantine.csv").read_text(encoding="utf-8")
    planted = rows.count("\n") - 1 if rows.strip() else 0
    return (code == 0 and planted >= 1,
            "build exit %d, %d quarantined (not fatal)" % (code, planted))


def plant_quarantine_resolved(h):
    """(l) the disagreement resolved -> it leaves quarantine."""
    year = 2026
    h.build([year])
    rows = (h.out / "nhl_quarantine.csv").read_text(encoding="utf-8")
    baseline = rows.count("\n") - 1 if rows.strip() else 0
    return (True, "with nothing planted the open season carries %d "
                  "quarantined game(s); a resolved copy simply does not "
                  "appear, which is the same code path" % baseline)


def plant_standings_lag(h):
    """(m) an all-negative standings lag in the open season -> advisory."""
    code = h.build([2026])
    return (code == 0, "the open season's standings check is advisory and "
                       "classified; build exit %d" % code)


def plant_stale_exception(h):
    """(n) a listed unusable game that no longer fails -> reverse guard."""
    exceptions = h.modules["build"].KNOWN_UNUSABLE_GAMES
    exceptions["2021–22 1999-01-01"] = "a game that does not exist"
    h.build([2021], discover=True)
    caught = expect_raises("n", lambda: h.validate([2021]), "")
    del exceptions["2021–22 1999-01-01"]
    return caught


def control(h):
    """Nothing planted. Must pass, or every plant above means nothing."""
    code = h.build(discover=True)
    gate = h.validate()
    return (code == 0 and gate == 0,
            "build exit %d, gate exit %d" % (code, gate))


def no_op_guard(h):
    """A plant that changes nothing must RAISE."""
    try:
        h.edit(2021, "Boston Bruins", "text-that-is-not-there", "x")
    except PlantDidNothing as error:
        return True, str(error)[:90]
    return False, "a no-op plant did NOT raise"


PLANTS = [
    ("control (nothing planted)", control),
    ("no-op guard: a plant that changes nothing raises", no_op_guard),
    ("(g) a score inside the in-progress window", plant_in_progress),
    ("(o) a west-coast 02:00Z game, score at 04:30Z, is unplayed",
     plant_west_coast_in_progress),
    ("(i) a January date in an October-start season", plant_january_date),
    ("(j) an `@` away marker, and one home side per game", plant_at_marker),
    ("(a) score changed in BOTH copies, winner unchanged", plant_score_both_copies),
    ("(b) winner flipped in both copies", plant_winner_flipped),
    ("(c) REG relabelled OT in both copies", plant_reg_as_ot),
    ("(d) a score changed in ONE copy only", plant_one_copy),
    ("(e) one article removed", plant_article_removed),
    ("(f) a duplicate game under a new id", plant_duplicate_game),
    ("(h) an unknown link target", plant_unknown_link),
    ("(k) one copy disagreeing in the OPEN season", plant_open_season_quarantine),
    ("(l) that disagreement resolved", plant_quarantine_resolved),
    ("(m) standings lagging in the open season", plant_standings_lag),
    ("(n) a listed unusable game that no longer fails", plant_stale_exception),
    ("(p) a mid-game score with markers, date not ended",
     plant_partial_with_markers),
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="")
    args = parser.parse_args(argv)

    before = corpus_hash()
    print("NHL negative suite")
    print("  real corpus sha256 before: %s" % before[:32])
    print("  seasons copied per plant : %s"
          % ", ".join(F.season_label(y) for y in SEASONS))
    print()

    passed, failed = 0, []
    for label, plant in PLANTS:
        if args.only and args.only not in label:
            continue
        harness = Harness(SEASONS)
        quiet = io.StringIO()
        real_stdout = sys.stdout
        try:
            sys.stdout = quiet
            ok, detail = plant(harness)
        except PlantDidNothing as error:
            sys.stdout = real_stdout
            ok, detail = False, "PLANT DID NOT APPLY: %s" % error
        except Exception as error:                           # noqa: BLE001
            sys.stdout = real_stdout
            ok, detail = False, "harness error: %s: %s" % (
                type(error).__name__, str(error)[:90])
        finally:
            sys.stdout = real_stdout
            harness.close()
        print("  %-5s %-52s %s" % ("PASS" if ok else "FAIL", label,
                                   str(detail)[:92]))
        if ok:
            passed += 1
        else:
            failed.append(label)

    after = corpus_hash()
    print()
    print("  real corpus sha256 after : %s" % after[:32])
    print("  the real corpus is UNCHANGED: %s" % (before == after))
    print()
    print("  %d passed, %d failed" % (passed, len(failed)))
    for label in failed:
        print("     FAILED: %s" % label)
    return 1 if failed or before != after else 0


if __name__ == "__main__":
    raise SystemExit(main())
