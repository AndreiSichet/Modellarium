"""Parse one NHL team-season article's regular-season game log.

Everything here was built against measurements in
`data-pipeline/probes/NOTES_round4_nhl.md`, re-measured on the fetched corpus.
The rules that matter:

COLUMNS ARE LOCATED BY NAME, never by position. The names are in the markup -
either as positional parameters of `{{Game log section start}}` or as an
ordinary `! # !! Date !! Visitor !! Score !!` header row - so a parser never
classifies a schema. 33 distinct header tuples carrying a Score column
collapse to two families. A table whose required columns cannot be named is
FATAL, not guessed.

WHOSE SCORE COMES FIRST IS READ FROM THE COLUMN NAMES. A `Visitor`/`Home`
pair means the visitor's goals are first; an `Opponent` column means this
team's are. Nothing is inferred from the numbers.

TWO CONTAINERS, and neither is "the" markup: per-month `{{Game log section
start}}` templates, and per-month raw `{| class="toccolours"` wikitables
nested inside an outer table captioned `<season> game log`. Measured on the
fetched corpus: 1,023 template sections and 3,285 toccolours tables.

THE HEADING HIERARCHY DISAMBIGUATES, because four articles carry three
sections all titled `Game log` - under Pre-season, Regular season and
Playoffs. A flat scan reads all three or none.
"""
import datetime as dt
from zoneinfo import ZoneInfo
import re

import nhl_franchises as F

# A hockey game runs about two and a half hours; three covers overtime and a
# shootout. This bound is only reachable when a puck-drop INSTANT is known,
# and this source never supplies one - see `kickoff_utc` below for the
# measurement. It is kept because the day a source does supply times, this is
# the branch that should run.
NHL_HOURS_UNTIL_FINAL = 3.0

# The latest US zone an NHL game is played in. See
# date_ended_utc: the season straddles the DST boundary, so
# this has to be a zone rather than an offset.
PACIFIC = ZoneInfo("America/Los_Angeles")

DASH = "[–—−-]"
# THE AWAY MARKER, and the `\b` that mis-oriented up to 451 rows a season.
# `\b` after `@` can never match - `@` and the space after it are both
# non-word characters - so `^\s*(@|at)\b` recognised `at Anaheim` and read
# every `@ Anaheim` as a home game. Each alternative now carries its own
# boundary, and `vs.` is matched so a home marker is recognised rather than
# merely failing to look like an away one.
AWAY_MARKER = re.compile(r"^\s*(@|at\b)", re.I)
HOME_MARKER = re.compile(r"^\s*vs\.?\b", re.I)
SCORE = re.compile(r"(\d{1,2})\s*%s\s*(\d{1,2})" % DASH)
# The label can sit in its own column or inside the score cell ("2-3 SO").
END_IN_CELL = re.compile(r"\b(OT|SO)\b", re.I)
LINK = re.compile(r"\[\[\s*([^\]|#]+?)\s*(?:\||\]\])")
SEASON_LINK = re.compile(r"^\d{4}%s\d{2}\s+(.+?)\s+season$" % DASH)
MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], 1)}
MONTH_TOKENS = {}
for _name, _n in MONTHS.items():
    MONTH_TOKENS[_name.lower()] = _n
    MONTH_TOKENS[_name[:3].lower()] = _n
MONTH_TOKENS["sept"] = 9

# Column name -> role. Measured from the corpus rather than guessed; a name
# that is not here is reported with the article and season that carried it.
ROLES = {
    "#": "NUM", "no.": "NUM", "no": "NUM", "game": "NUM", "gm": "NUM",
    "date": "DATE",
    "visitor": "VISITOR", "visitors": "VISITOR", "visiting team": "VISITOR",
    "home": "HOME", "home team": "HOME",
    "opponent": "OPPONENT", "opponents": "OPPONENT",
    "score": "SCORE", "result": "SCORE",
    "ot": "END", "ot/so": "END", "so": "END",
    "decision": "DECISION", "goalie": "DECISION",
    "record": "RECORD",
    "attendance": "ATTENDANCE", "att.": "ATTENDANCE",
    "location": "LOCATION", "arena": "LOCATION", "venue": "LOCATION",
    "location/attendance": "LOCATION",
    "pts": "POINTS", "points": "POINTS", "pts.": "POINTS",
    "recap": "RECAP", "summary": "RECAP", "notes": "RECAP",
    "time": "TIME", "time (et)": "TIME", "time (pt)": "TIME",
    # A `Series` column marks a PLAYOFF table, which this parser rejects
    # rather than reads.
    "series": "SERIES",
}

# Row outcome, from the template form. FIVE spellings exist, not the three
# round 4 recorded: Game-win and Game-loss are synonyms of Game-won and
# Game-lost, on 1,058 rows between them.
OUTCOME_TEMPLATE = {
    "game-won": "W", "game-win": "W",
    "game-lost": "L", "game-loss": "L",
    "game-otl": "OTL", "game-otloss": "OTL", "game-tied": "T",
}
LEGEND = re.compile(r"\{\{\s*legend2?\s*\|\s*(#?\w+)\s*\|\s*([^|}]+)")


# Tables whose caption and heading disagree about which part of the
# season they belong to. Recovered, counted and reported - never
# silently resolved either way.
HEADING_DISAGREEMENTS = []


class ParseError(RuntimeError):
    """The markup could not be read. Always fatal - never quarantined.

    Quarantine is for a disagreement between two READABLE copies. Markup
    nobody can read is a parser problem and must stop the build.
    """


def clean(cell):
    """Cell text with markup removed, link DISPLAY text preserved."""
    text = cell
    text = re.sub(r"<ref[^>]*/>", " ", text)
    text = re.sub(r"<ref.*?</ref>", " ", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)
    # A DISPLAY TEMPLATE STANDS FOR ITS FIRST PARAMETER. Deleting it instead
    # lost the OT/SO label on every Pittsburgh 2017-18 row, where the label
    # lives inside `{{small|OT}}` in an external link's display text and
    # there is no OT column to fall back on.
    for _ in range(4):
        reduced = re.sub(r"\{\{[^{}|]*\|([^{}|]*)(?:\|[^{}]*)?\}\}",
                         r"\1", text)
        if reduced == text:
            break
        text = reduced
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    text = text.replace("&nbsp;", " ").replace("'''", "").replace("''", "")
    text = re.sub(r"^\s*[a-z\-]+\s*=\s*\"[^\"]*\"\s*\|", "", text)
    return re.sub(r"\s+", " ", text).strip()


def split_cells(line):
    """Split a wikitable row on `||`, DEPTH-AWARE.

    `||` appears inside templates and links, so a plain split loses cells and
    silently shifts every column after it.
    """
    out, depth, buf, i = [], 0, [], 0
    while i < len(line):
        two = line[i:i + 2]
        if two in ("{{", "[["):
            depth += 1
            buf.append(two)
            i += 2
            continue
        if two in ("}}", "]]"):
            depth = max(0, depth - 1)
            buf.append(two)
            i += 2
            continue
        if two == "||" and depth == 0:
            out.append("".join(buf))
            buf = []
            i += 2
            continue
        buf.append(line[i])
        i += 1
    out.append("".join(buf))
    return out


def match_braces(text, start):
    """The index just past the `}}` closing the `{{` at `start`.

    Brace-matched rather than non-greedy: round 4 found a caption containing
    a stray `{{DEFAULTSORT:}}`, where `\\{\\{...(.*?)\\}\\}` ends inside it and
    SILENTLY loses every column name for that month.
    """
    depth, i = 0, start
    while i < len(text):
        if text[i:i + 2] == "{{":
            depth += 1
            i += 2
            continue
        if text[i:i + 2] == "}}":
            depth -= 1
            i += 2
            if depth == 0:
                return i
            continue
        i += 1
    raise ParseError("an unclosed {{ at offset %d" % start)


def sections(text):
    """[(level, title, body_start, body_end)] in document order."""
    marks = list(re.finditer(r"^(=+)\s*(.+?)\s*=+\s*$", text, re.M))
    out = []
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((len(mark.group(1)), mark.group(2), mark.end(), end))
    return out


def ancestry(secs, position):
    """The heading stack enclosing a position, outermost first."""
    stack = []
    for level, title, start, _end in secs:
        if start > position:
            break
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
    return [title for _level, title in stack]


SCHEDULE = re.compile(r"schedule|game log|results", re.I)
REGULAR = re.compile(r"^regular[\s\-]?season", re.I)
NOT_REGULAR = re.compile(r"pre[\s\-]?season|playoff|post[\s\-]?season|"
                         r"exhibition|stanley cup", re.I)


def in_regular_season(path):
    """Does this heading stack put us in the regular-season game log?

    Both halves are needed. An article can carry `Regular season` twice - once
    for standings and once for the log - and four articles carry three
    sections all called `Game log`.
    """
    if any(NOT_REGULAR.search(title) for title in path):
        return False
    return any(REGULAR.search(title) for title in path)


def legend_palette(text):
    """{colour: outcome} as THIS ARTICLE declares it.

    The palette varies per article - #cfc/#fcc/#fff in some, #CCFFCC or #bfb
    in others - so a fixed hex table is the wrong instrument. The articles
    state their own.
    """
    palette = {}
    for colour, label in LEGEND.findall(text):
        low = label.lower()
        if "win" in low:
            outcome = "W"
        elif "ot" in low or "shootout" in low or "so loss" in low:
            outcome = "OTL"
        elif "loss" in low or "lost" in low:
            outcome = "L"
        else:
            continue
        palette.setdefault(colour.lower().lstrip("#"), outcome)
    return palette


def column_roles(names, article, season, where):
    """[(role, raw name)] for one table's columns, by NAME.

    A month name in a column heading is the month-in-the-heading form, where
    the cell holds only a day number. 830 rows in the corpus use it.
    """
    roles = []
    for raw in names:
        name = clean(raw).strip().rstrip(":")
        key = name.lower()
        if key in ROLES:
            roles.append((ROLES[key], name))
            continue
        token = re.sub(r"[^a-z]", "", key)
        if token in MONTH_TOKENS:
            roles.append(("DAY:%d" % MONTH_TOKENS[token], name))
            continue
        if not name:
            # AN UNNAMED COLUMN IS NOT A COLUMN. These come from a `||` in
            # the markup - `|title=October: ...||# |Date |Visitor` - which
            # yields an empty positional parameter and shifted every column
            # by one in five articles a season, making their whole game log
            # unreadable. Dropping it is also self-checking: if a table ever
            # did carry a genuinely blank-headed DATA column, the cell count
            # would stop matching and those rows would be reported rather
            # than mis-read, because this parser addresses columns by name.
            continue
        roles.append(("UNKNOWN", name))
    return roles


def _require(roles, article, season, where):
    kinds = [r.split(":")[0] for r, _n in roles]
    if "SERIES" in kinds:
        return None                       # a playoff table; not ours
    has_date = "DATE" in kinds or any(k == "DAY" for k in kinds)
    sided = ("VISITOR" in kinds and "HOME" in kinds) or "OPPONENT" in kinds
    if not (has_date and "SCORE" in kinds and sided):
        unknown = [n for (r, n) in roles if r == "UNKNOWN"]
        raise ParseError(
            "%s (%s) %s: the required columns cannot be named from %r. "
            "Unrecognised: %r. A table whose columns cannot be named is "
            "fatal rather than guessed."
            % (article, season, where, [n for _r, n in roles], unknown))
    return roles


def parse_score(cell):
    """(first, second, end) from a score cell. `end` may be None."""
    text = clean(cell)
    found = SCORE.search(text)
    if not found:
        return None
    label = END_IN_CELL.search(text[found.end():])
    return (int(found.group(1)), int(found.group(2)),
            label.group(1).upper() if label else None)


def parse_end(cell):
    """REG/OT/SO from a dedicated OT column.

    A DASH MEANS REGULATION. 2012-13 writes an em dash there and reading it
    as truthy made 74 regulation games look like one-goal overtimes.
    """
    text = clean(cell)
    if not text or re.fullmatch(r"%s+" % DASH, text):
        return "REG"
    found = END_IN_CELL.search(text)
    return found.group(1).upper() if found else "REG"


def parse_day(cell, month, season_start):
    text = clean(cell)
    found = re.search(r"(\d{1,2})", text)
    if not found:
        return None
    return _date(season_start, month, int(found.group(1)))


def parse_date(cell, season_start):
    """A date from an explicit cell such as `September 25` or `Oct 13`."""
    text = clean(cell).replace(",", " ")
    found = re.search(r"([A-Za-z]{3,9})\.?\s+(\d{1,2})", text)
    if not found:
        found = re.search(r"(\d{1,2})\s+([A-Za-z]{3,9})", text)
        if not found:
            return None
        day, token = int(found.group(1)), found.group(2)
    else:
        token, day = found.group(1), int(found.group(2))
    month = MONTH_TOKENS.get(re.sub(r"[^a-z]", "", token.lower()))
    if month is None:
        return None
    return _date(season_start, month, day)


def _date(season_start, month, day):
    """Which calendar year a month belongs to, DERIVED from the season.

    September-December belong to the first calendar year, January-August to
    the second. This is the CFL's silent `season_start_month` bug in another
    league: a default bound at import dated a spring league's rows a year
    late with no error at all.
    """
    year = season_start if month >= 9 else season_start + 1
    try:
        return dt.date(year, month, day)
    except ValueError:
        raise ParseError("%d-%02d-%02d is not a date" % (year, month, day))


def opponent_identity(cell, season_start):
    """(franchise_id, link_target, display, note) for a team cell."""
    raw = cell
    link = LINK.search(raw)
    target = None
    if link:
        text = link.group(1).strip()
        seasonal = SEASON_LINK.match(text)
        if seasonal:
            target = seasonal.group(1)
        else:
            # A season-prefixed target that omits the word `season`:
            # `[[2021-22 Los Angeles Kings|Los Angeles]]`. Stripping the
            # prefix is safe because the remainder is then matched against
            # the authored table like any other name.
            target = re.sub(r"^\d{4}%s\d{2}\s+" % DASH, "", text)
    display = clean(raw)
    probe = target or display
    # An `@`/`at`/`vs.` prefix belongs to the marker, not the name. The CFL's
    # silent bug was reading `away` from a prefix the regex did not cover.
    probe = re.sub(r"^\s*(@|at|vs\.?)\s+", "", probe, flags=re.I).strip()
    fid, note = F.resolve_link_target(probe, season_start)
    return fid, target, display, note


def row_marker(line, palette):
    """The outcome a row's own marker asserts, independent of the score."""
    found = re.search(r"\{\{\s*(Game-[a-z]+)\s*\}\}", line, re.I)
    if found:
        key = found.group(1).lower()
        if key not in OUTCOME_TEMPLATE:
            raise ParseError("unknown outcome template %r" % found.group(1))
        return OUTCOME_TEMPLATE[key], "template"
    colour = re.search(r"background:\s*(#?\w+)", line, re.I)
    if colour:
        key = colour.group(1).lower().lstrip("#")
        if key in palette:
            return palette[key], "colour"
    return None, ""


CAPTION = re.compile(r"^!\s*colspan[^|\n]*\|\s*(.+)$", re.M)
BOLD_HEADER = re.compile(r"^\|\s*'''")


def _regions(text, secs):
    """[(start, end)] of the article's regular-season regions.

    A region is a section whose heading stack contains no pre-season,
    playoff or exhibition heading, and which is either named for the regular
    season or sits under a schedule/results heading. The second half is why
    2014-15 Vancouver works: it has no `Regular season` sub-heading.
    """
    out = []
    for level, _title, body_start, body_end in secs:
        path = ancestry(secs, body_start)
        if any(NOT_REGULAR.search(t) for t in path):
            continue
        if not (any(REGULAR.search(t) for t in path)
                or any(SCHEDULE.search(t) for t in path)):
            continue
        out.append((body_start, body_end))
    return out


def _in_region(regions, position):
    for start, end in regions:
        if start <= position < end:
            return end
    return None


def _caption_before(text, position, floor):
    """The nearest table caption above a header, which says what table it is.

    Measured across the corpus: `<season> game log` and `<season> season` for
    the regular season, `<year> pre-season game log` for the preseason. The
    caption is a single colspan cell, which is also why it is never mistaken
    for a header row.
    """
    best = ""
    for found in CAPTION.finditer(text, floor, position):
        best = clean(found.group(1))[:80]
    return best


NOT_OURS = re.compile(r"pre[\s\-]?season|playoff|post[\s\-]?season|"
                      r"exhibition|stanley cup|round", re.I)

# A regular-season caption, measured: a season label, or a month name.
CAPTION_REGULAR = re.compile(
    r"^(\d{4}%s\d{2}\b|(%s)\b)" % (DASH, "|".join(MONTHS)), re.I)


def _caption_kind(caption):
    """'regular' | 'not-ours' | 'unknown' for one table caption."""
    if not caption:
        return "unknown"
    if NOT_OURS.search(caption):
        return "not-ours"
    if CAPTION_REGULAR.match(caption.strip()):
        return "regular"
    return "unknown"


def _header_candidates(text):
    """Every header row, in either spelling, with its cell names."""
    for found in re.finditer(r"^(!(?!\s*colspan)|\|\s*''')(.*)$", text, re.M):
        line = found.group(0)
        if line.startswith("!"):
            names = split_bangs(line[1:])
        else:
            if not BOLD_HEADER.match(line):
                continue
            # A bold header row separates its cells with `||`, not `!!` - it
            # is a data row being used as a header. The `!!` splitter left
            # 13 of 2013-14 Chicago's tables invisible.
            names = split_cells(line.lstrip("|"))
        if len(names) >= 4:
            yield found, names


def _tables(text, secs, article, season, allow_outside=False):
    """[(where, roles, body)] for every regular-season game table."""
    out, regions = [], _regions(text, secs)

    # Form B: one {{Game log section start}} per month. BRACE-MATCHED,
    # because a caption containing a stray {{DEFAULTSORT:}} ends a non-greedy
    # match inside itself and silently loses every column name.
    for found in re.finditer(r"\{\{\s*Game log section start\s*", text, re.I):
        region_end = _in_region(regions, found.start())
        if region_end is None and not allow_outside:
            continue
        end = match_braces(text, found.start())
        inner = text[found.start() + 2:end - 2]
        params = split_pipes(inner)[1:]
        title = next((p.split("=", 1)[1] for p in params
                      if p.strip().lower().startswith("title=")), "")
        if _caption_kind(clean(title)) != "regular":
            continue
        if region_end is None:
            region_end = _section_end(secs, found.start())
        names = [p for p in params if "=" not in p.split("|")[0][:24]]
        body_end = text.find("{{Game log section end", end)
        nxt = text.lower().find("{{game log section start", end)
        for candidate in (nxt, region_end):
            if candidate != -1 and (body_end == -1 or candidate < body_end):
                body_end = candidate
        if body_end == -1:
            body_end = region_end
        roles = _require(column_roles(names, article, season, "template"),
                         article, season, "template section")
        if roles:
            out.append(("template", roles, text[end:body_end]))

    # Form A: a header ROW inside a nested wikitable, in either spelling.
    for found, names in _header_candidates(text):
        region_end = _in_region(regions, found.start())
        if region_end is None and not allow_outside:
            continue
        caption = _caption_before(text, found.start(),
                                  max(0, found.start() - 4000))
        if _caption_kind(caption) != "regular":
            continue
        roles = column_roles(names, article, season, "header row")
        if "SCORE" not in [r.split(":")[0] for r, _n in roles]:
            continue
        roles = _require(roles, article, season, "header row")
        if not roles:
            continue
        if region_end is None:
            region_end = _section_end(secs, found.start())
        rest = text[found.end():region_end]
        stop = re.search(r"^(!(?!\s*colspan)|\|\s*''')", rest, re.M)
        out.append(("header row", roles,
                    rest[:stop.start()] if stop else rest))
    return out


def _section_end(secs, position):
    """The end of the innermost section containing a position."""
    best = None
    for _level, _title, start, end in secs:
        if start <= position < end and (best is None or end < best):
            best = end
    return best if best is not None else position + 20000


def split_pipes(inner):
    """Depth-aware split of a template's parameters on `|`."""
    out, depth, buf, i = [], 0, [], 0
    while i < len(inner):
        two = inner[i:i + 2]
        if two in ("{{", "[["):
            depth += 1
            buf.append(two)
            i += 2
            continue
        if two in ("}}", "]]"):
            depth = max(0, depth - 1)
            buf.append(two)
            i += 2
            continue
        if inner[i] == "|" and depth == 0:
            out.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(inner[i])
        i += 1
    out.append("".join(buf))
    return out


def split_bangs(line):
    """Depth-aware split of a header row on `!!`."""
    parts, depth, buf, i = [], 0, [], 0
    while i < len(line):
        two = line[i:i + 2]
        if two in ("{{", "[["):
            depth += 1
            buf.append(two)
            i += 2
            continue
        if two in ("}}", "]]"):
            depth = max(0, depth - 1)
            buf.append(two)
            i += 2
            continue
        if two == "!!" and depth == 0:
            parts.append("".join(buf))
            buf = []
            i += 2
            continue
        buf.append(line[i])
        i += 1
    parts.append("".join(buf))
    return parts


def parse_article(text, season_start, team_fid, article, now=None):
    """[row] for one team-season's regular-season games and fixtures."""
    now = now or dt.datetime.now(dt.timezone.utc)
    season = F.season_label(season_start)
    secs = sections(text)
    palette = legend_palette(text)
    rows, unreadable = [], 0

    tables = _tables(text, secs, article, season)
    if not tables:
        # NO REGULAR-SEASON REGION AT ALL. 2014-15 Vancouver has no `Regular
        # season` heading and its months sit inside `Pre-season`, so the
        # caption is the only thing that identifies them. Recovered rather
        # than returning an empty season, and RECORDED, because a team with
        # no games is exactly the shape of silent emptiness that would
        # otherwise reach the completeness check as a coverage problem.
        tables = _tables(text, secs, article, season, allow_outside=True)
        if tables:
            HEADING_DISAGREEMENTS.append(
                (article, "no regular-season heading; %d table(s) "
                          "identified by caption alone" % len(tables)))

    for where, roles, body in tables:
        kinds = [r.split(":")[0] for r, _n in roles]
        month = next((int(r.split(":")[1]) for r, _n in roles
                      if r.startswith("DAY:")), None)
        # ROWS ARE PHYSICAL LINES, and the marker comes from the most recent
        # `|-` above. Splitting the body on "\n|-" instead looked equivalent
        # and is not: a cell whose template or archive link contains a
        # newline followed by `|-` fragments the row, and the fragment starts
        # mid-link - which produced six rows across the corpus whose team
        # cell read `Dallas Stars season|Dallas`. Found by chasing an
        # unresolved team name back to its row, not by review.
        marker_line = ""
        for line in body.split("\n"):
            if line.startswith("|-"):
                marker_line = line
                continue
            if not line.startswith("|") or line.startswith(("|}", "|+")):
                continue
            first = marker_line
            cells = split_cells(line.lstrip("|"))
            # A row that SPANS the table is not a game row. These are the
            # month subtotals and break annotations, and they are the whole
            # of what an earlier version counted as "rows with an unreadable
            # date" - 400 a season in the recent corpus, which reconciled
            # suspiciously exactly to round 4's totals and turned out to be
            # a different thing entirely. A `colspan` on a LATER cell is
            # ordinary (a postponed game writes its reason that way), so the
            # test is one cell, not any colspan.
            if len(cells) == 1 and "colspan" in cells[0].lower():
                continue
            if len(cells) < len(roles) - 2:
                continue
            get = {}
            for (role, _name), cell in zip(roles, cells):
                get.setdefault(role.split(":")[0], cell)
            if "SCORE" not in get:
                continue

            date = (parse_day(get.get("DAY", ""), month, season_start)
                    if month is not None
                    else parse_date(get.get("DATE", ""), season_start))
            if date is None:
                # A row with no readable date in a table whose columns WERE
                # named is a blank fixture row in a schedule-only table, or
                # markup nobody can read. Counted and reported, never guessed.
                unreadable += 1
                continue

            score = parse_score(get["SCORE"])
            end = None
            if "END" in get:
                end = parse_end(get["END"])
            if score and score[2]:
                end = score[2]

            if "OPPONENT" in get:
                opponent_cell = get["OPPONENT"]
                bare = clean(opponent_cell)
                away = bool(AWAY_MARKER.match(bare))
                ours_first = True
            else:
                visitor, home = get["VISITOR"], get["HOME"]
                v_fid = _side_fid(visitor, season_start)
                away = (v_fid == team_fid)
                opponent_cell = home if away else visitor
                ours_first = False

            fid, target, display, note = opponent_identity(
                opponent_cell, season_start)
            marker, marker_from = row_marker(first, palette)

            if score is None:
                goals_for = goals_against = None
                played = False
            else:
                if ours_first:
                    goals_for, goals_against = score[0], score[1]
                else:
                    visitor_goals, home_goals = score[0], score[1]
                    goals_for = visitor_goals if away else home_goals
                    goals_against = home_goals if away else visitor_goals
                played = True

            if played and in_progress(date, now, kickoff_utc(get)):
                # A score on a game whose nominal puck drop is inside the
                # in-progress window is not a final score.
                goals_for = goals_against = None
                played = False

            rows.append(dict(
                season=season, season_start=season_start, date=date,
                team_fid=team_fid, opponent_fid=fid, away=away,
                goals_for=goals_for, goals_against=goals_against,
                game_end=(end or ("REG" if played else None)),
                marker=marker, marker_from=marker_from,
                record=clean(get.get("RECORD", "")),
                decision=_decision(get.get("DECISION", "")),
                opponent_link=target, opponent_display=display,
                identity_note=note, played=played,
                source_table=where, article=article))
    return rows, unreadable


def _side_fid(cell, season_start):
    try:
        fid, _t, _d, _n = opponent_identity(cell, season_start)
        return fid
    except F.UnknownFranchise:
        return None


def _decision(cell):
    """The goalie of record's link target where present, else the text."""
    link = LINK.search(cell)
    if link:
        return link.group(1).strip()
    return clean(cell)


def kickoff_utc(row):
    """The game's own puck drop as a UTC instant, or None.

    ALWAYS None for this source, and that is measured rather than assumed:
    412 of 34,965 rows carry a time column at all, every one of them in
    2013-14 to 2016-17 and every one a bare `7:00 pm` with no zone - and one
    of the 412 reads `Staples Center`. A clock time with no zone cannot be
    made into an instant, and inferring the venue's zone would mean authoring
    an arena table this phase does not have.

    The open season, which is the only place an in-progress guard can matter,
    has no time column in any of its articles.
    """
    return None


def date_ended_utc(date):
    """The instant date D has ended in the latest US zone a game is played.

    PACIFIC, FROM THE ZONE, NOT A CONSTANT. The latest zone an NHL game is
    played in is Pacific, so a US calendar date D is over everywhere relevant
    at midnight Pacific on D+1 - which is D+1 **07:00 UTC in PDT** and D+1
    **08:00 UTC in PST**. Every NHL season straddles that boundary (US DST
    ends on 1 November 2026), and the one-hour difference is exactly the
    window the refresh time turns on, so a hardcoded offset would be right
    for most of a season and wrong for the rest.
    """
    midnight_pacific = dt.datetime.combine(
        date + dt.timedelta(days=1), dt.time(0, 0), tzinfo=PACIFIC)
    return midnight_pacific.astimezone(dt.timezone.utc)


def in_progress(date, now, kickoff=None):
    """Might a score on this row be a score written DURING the game?

    Two branches, the same shape as the NFL's:

      * with an instant, the game is in progress for NHL_HOURS_UNTIL_FINAL
        after it;
      * without one, it is judged on its DATE - and the test is whether that
        date has ENDED in the latest US zone the game could have been played
        in, which is `date_ended_utc`.

    NARROWED 2026-10-30 FROM A FULL EXTRA DAY, AND THE SAVING IS MEASURED.
    This used to be `(now.date() - date).days <= 1`, which accepted a game on
    D only from **D+2** - a whole day later than the information was
    available. 320 Wikipedia revisions across 8 articles were replayed to
    check the narrower rule is safe:

      * a complete score with its markers appeared a median **0.6 h BEFORE**
        the date ended, and 55 of 98 first-completions were already in place
        before it;
      * **0 of 320 revisions ever showed a PARTIAL score with its markers
        already set**, which is the one failure this rule must not accept.

    So the rule is both safe on the evidence and conservative: it still waits
    for more than half of the games whose scores were already correct.

    WHY NOT A NOMINAL CLOCK TIME, still: assuming 23:00 UTC puts a 7 p.m.
    Pacific game's drop three hours before it actually starts and accepts a
    score written in overtime as final. The date's end in the latest zone is
    the weakest claim that is always true.
    """
    if kickoff is not None:
        return (now - kickoff) < dt.timedelta(hours=NHL_HOURS_UNTIL_FINAL)
    return now < date_ended_utc(date)


def _too_recent(date, now):
    return in_progress(date, now, kickoff_utc(None))


def result_from(goals_for, goals_against, game_end):
    """W / L / OTL, which is NOT the same question as REG / OT / SO."""
    if goals_for is None or goals_against is None:
        return None
    if goals_for > goals_against:
        return "W"
    return "OTL" if game_end in ("OT", "SO") else "L"
