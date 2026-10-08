"""Parse NFL team-season wikitext into rows, plus the game-summary boxes.

Read from the stored raw files, never the network. Three things come from the
wikitext that the rendered HTML throws away, and the parse depends on all
three:

  * the opponent cell links to the opponent's own season article, so identity
    comes from a LINK TARGET rather than a rendered name;
  * each row carries {{Game-won}} / {{Game-lost}}, an outcome flag independent
    of the score string;
  * the recap cell carries an nfl.com slug, <away>-at-<home>-<season>-<type>-
    <week>, whose type token cross-checks the section heading.

Separately, {{Americanfootballbox}} templates carry the quarter-by-quarter
score - a second, independent copy of the final score on the same page, and
the only thing that catches a score edited identically in both articles
without changing the winner.
"""
import collections
import datetime
import itertools
import re
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
NFL = HERE.parent
sys.path.insert(0, str(HERE))

import nfl_franchises as F          # noqa: E402

RAW = NFL / "data" / "raw"

MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], start=1)}

# Section headings that hold the regular-season table. Measured across
# 2006-2026: exactly these two, and nothing else.
SECTION_REGULAR = {"Regular season", "Regular Season", "Schedule"}
SECTION_PRESEASON = {"Preseason", "Pre-season", "Preseason schedule"}

COL_ALIASES = {
    "Game site": "Venue", "Site": "Venue", "Location": "Venue",
    "Game Site": "Venue", "Stadium": "Venue",
    "NFL.com recap": "Recap", "NFL.com Recap": "Recap", "NFL Recap": "Recap",
    "Game recap": "Recap", "Sources": "Recap", "Source": "Recap",
    "NFL.com GameBook": "GameBook", "TV": "Network",
    "Results": "Result", "Playoff round": "Round", "Kickoff": "Time",
    "Score": "Result",
}

# Zone labels as the articles write them. IANA names, not fixed offsets: a
# fixed offset is wrong for Arizona for part of every season, which section 33
# already records. Used ONLY for the "kicked off less than five hours ago"
# guard; the fixtures table stores the zone as written.
ZONES = {
    "Eastern Time Zone": "America/New_York",
    "Central Time Zone": "America/Chicago",
    "Mountain Time Zone": "America/Denver",
    "Pacific Time Zone": "America/Los_Angeles",
    "Eastern Time": "America/New_York", "Central Time": "America/Chicago",
    "Mountain Time": "America/Denver", "Pacific Time": "America/Los_Angeles",
    "ET": "America/New_York", "CT": "America/Chicago",
    "MT": "America/Denver", "PT": "America/Los_Angeles",
    "EST": "America/New_York", "EDT": "America/New_York",
    "CST": "America/Chicago", "CDT": "America/Chicago",
    "MST": "America/Denver", "MDT": "America/Denver",
    "PST": "America/Los_Angeles", "PDT": "America/Los_Angeles",
}
# Arizona does not observe daylight saving, and its article still labels the
# column "Mountain Time Zone".
ZONE_OVERRIDE = {"Arizona Cardinals": "America/Phoenix"}

# A venue written with a parenthetical city outside the United States. Used as
# one of three neutral-site signals, reported separately so the method behind
# each flag is visible.
INTERNATIONAL_CITY = re.compile(
    r"\((London|Munich|Frankfurt|Mexico City|Toronto|Sao Paulo|São Paulo|"
    r"Rio de Janeiro|Madrid|Dublin|Melbourne|Berlin|Paris|Wembley)\)", re.I)

RESULT = re.compile(r"^([WLT])\s*(\d+)[–−-](\d+)")
LINK = re.compile(r"\[\[\s*(\d{4})\s+(.+?)\s+season\s*(?:\||\]\])")
SLUG = re.compile(r"nfl\.com/games/([a-z0-9\-]+)")
BOX = re.compile(r"\{\{\s*Americanfootballbox", re.I)
ROW_SEPARATOR = r"\n\|-"


class ParseError(RuntimeError):
    """A row this parser will not guess about. Never skipped."""


# ------------------------------------------------------------------ markup

def strip_templates(text):
    out = text
    while re.search(r"\{\{[^{}]*\}\}", out):
        out = re.sub(r"\{\{[^{}]*\}\}", "", out)
    return out


TRANSPARENT = ("nowrap", "dow tooltip", "tooltip", "abbr", "small",
               "nobr", "sortname")
# Footnote templates are DROPPED whole. Unwrapping them keeps their text,
# which put an explanatory paragraph about a COVID stadium ban inside a
# venue cell.
FOOTNOTE_TEMPLATES = ("efn-ua", "efn", "sfn", "refn")
TRANSPARENT_OPEN = re.compile(
    r"\{\{\s*(?:" + "|".join(re.escape(t) for t in TRANSPARENT) + r")\s*\|",
    re.I)


FOOTNOTE_OPEN = re.compile(
    r"\{\{\s*(?:" + "|".join(re.escape(t) for t in FOOTNOTE_TEMPLATES) +
    r")\s*[|}]", re.I)
NOTE_MARKER = re.compile(r"<ref\b|\{\{\s*(?:efn|sfn|refn)", re.I)


def drop_footnotes(text):
    """Remove footnote templates entirely, braces matched."""
    out, guard = text, 0
    while guard < 12:
        guard += 1
        m = FOOTNOTE_OPEN.search(out)
        if m is None:
            return out
        depth, i = 1, m.end() - 1
        while i < len(out) and depth:
            if out[i:i + 2] == "{{":
                depth += 1
                i += 2
                continue
            if out[i:i + 2] == "}}":
                depth -= 1
                if depth == 0:
                    break
                i += 2
                continue
            i += 1
        out = out[:m.start()] + out[i + 2:]
    return out


def unwrap_transparent(text):
    """Drop display-only template wrappers, keeping ALL of their content.

    A naive `{{nowrap|([^}|]*)...}}` capture stops at the first pipe, which
    truncates a piped link inside the wrapper: `{{nowrap|[[NFL on Thanksgiving
    Day|November 27]]}}` became `[[NFL on Thanksgiving Day` and the date then
    failed to parse.
    """
    out, guard = text, 0
    while guard < 8:
        guard += 1
        m = TRANSPARENT_OPEN.search(out)
        if m is None:
            return out
        depth, i = 1, m.end()
        while i < len(out) and depth:
            if out[i:i + 2] == "{{":
                depth += 1
                i += 2
                continue
            if out[i:i + 2] == "}}":
                depth -= 1
                if depth == 0:
                    break
                i += 2
                continue
            i += 1
        out = out[:m.start()] + out[m.end():i] + out[i + 2:]
    return out


def clean(cell):
    """Wiki markup down to display text.

    TemplateStyles emit <style> blocks and tooltip templates wrap dates;
    phase 0 found CSS inside a date cell silently dropping 11 of 256 games,
    so both are removed before anything is parsed.
    """
    c = re.sub(r"<!--.*?-->", "", cell, flags=re.S)
    c = re.sub(r"<ref[^>]*>.*?</ref>", "", c, flags=re.S)
    c = re.sub(r"<ref[^>]*/>", "", c)
    c = re.sub(r"<style[^>]*>.*?</style>", "", c, flags=re.S | re.I)
    c = re.sub(r"\.mw-parser-output[^{]*\{[^}]*\}", "", c)
    c = drop_footnotes(c)
    c = unwrap_transparent(c)
    c = strip_templates(c)
    c = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", c)
    c = re.sub(r"\[\[([^\]]*)\]\]", r"\1", c)
    c = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", c)
    c = re.sub(r"\[https?://\S+\]", "", c)
    c = c.replace("'''", "").replace("''", "")
    c = c.replace("&nbsp;", " ").replace("&ndash;", "–")
    c = c.replace("&mdash;", "—").replace("&amp;", "&")
    c = re.sub(r"<[^>]+>", "", c)
    return re.sub(r"\s+", " ", c).strip()


def cell_body(line):
    """A wikitable cell may carry attributes before a pipe."""
    # A stray '!' may sit between the attributes and the pipe:
    # `rowspan="2" ! | NFL.com<br>recap`.
    m = re.match(r'\s*((?:[a-z\-]+="[^"]*"\s*)+)!?\s*\|(?!\|)(.*)$', line,
                 re.S)
    if m:
        return m.group(2)
    # Malformed but rendered: `! + style="..."|Week`. The 2015 Oakland
    # Raiders article is written this way and yields nothing to a parser that
    # requires clean attribute syntax - visible only as 16 unpaired games on
    # its opponents' pages.
    m = re.match(r'\s*\S{0,3}\s*(?:[a-z\-]+="[^"]*"\s*)+!?\s*\|(?!\|)(.*)$',
                 line, re.S)
    body = m.group(1) if m else line
    # A cell body never legitimately begins with a bare pipe; some rows are
    # written `| |[[NFL on Thanksgiving Day|November 22]]`, which otherwise
    # leaves '|November 22' and fails the date parse.
    return re.sub(r"^\s*\|+\s*", "", body)


def header_name(line):
    """One canonical spelling per column.

    `<br>` becomes a space BEFORE cleaning, or `NFL.com<br>recap` collapses to
    `NFL.comrecap` and misses its alias. Any `Time (...)` heading reduces to
    `Time`, since the zone it names is read separately.
    """
    text = re.sub(r"<br\s*/?>", " ", cell_body(strip_templates(line)),
                  flags=re.I)
    name = re.sub(r"\s+", " ", clean(text)).strip()
    if name.startswith("Time") or name.startswith("Kickoff"):
        return "Time"
    return COL_ALIASES.get(name, name)


# ------------------------------------------------------------------ tables

# Six articles (4 in 2024, 2 in 2025) open the schedule with a TEMPLATE
# instead of `{|`, so the article carries no header row at all - the columns
# are defined inside the template. Measured: one spelling, and every row has
# exactly these seven cells, which is the canonical order.
TEMPLATE_TABLE = re.compile(r"\{\{\s*NFL [Ss]chedule [Ss]tart[^}]*\}\}")
TEMPLATE_HEADER = ["Week", "Date", "Opponent", "Result", "Record", "Venue",
                   "Recap"]


def tables_with_sections(text):
    """Every schedule table, attributed to the heading it sits under.

    Yields (section, raw, implied header or None).
    """
    section, out = "", []
    for chunk in re.split(r"\n(?==+[^=\n]+=+[ \t]*\n)", text):
        h = re.match(r"=+\s*([^=\n]+?)\s*=+", chunk)
        if h:
            section = h.group(1).strip()
        for raw in re.findall(r"\{\|.*?\n\|\}", chunk, flags=re.S):
            out.append((section, raw, None))
        for m in TEMPLATE_TABLE.finditer(chunk):
            end = chunk.find("\n|}", m.end())
            body = chunk[m.end():end if end != -1 else len(chunk)]
            out.append((section, body, list(TEMPLATE_HEADER)))
    return out


def split_table(raw, implied=None):
    """(header names, data row blocks). The header row is detected, not
    assumed to precede the first separator - both layouts are legal wikitext
    and this corpus uses both."""
    parts = re.split(r"\n\|-", raw)
    if implied:
        # No header row exists in the article: the template defines the
        # columns. The cell count is GUARDED rather than trusted, because a
        # row of a different width would silently shift every column.
        blocks = [b for b in parts if b.strip()]
        widths = {len(row_cells(b)) for b in blocks
                  if not re.search(r"colspan", b, re.I)}
        if widths - {len(implied)}:
            raise ParseError(
                f"template-opened schedule has rows of width "
                f"{sorted(widths)}, expected {len(implied)} - refusing to "
                f"assign columns by position")
        return list(implied), blocks
    for index, part in enumerate(parts):
        cells = []
        for line in re.findall(r"^!(.*)$", part, flags=re.M):
            cells += [c for c in line.split("!!")]
        names = [n for n in (header_name(c) for c in cells) if n]
        if "Date" not in names:
            continue
        rest = parts[index + 1:]
        # A two-level header: a cell with colspan=N covers N columns and a
        # continuation row names them. `colspan="2" | Results` over
        # `! Score !! Record` is seven columns, not six.
        spans = [(header_name(c), header_colspan(c)) for c in cells]
        spans = [(n, s) for n, s in spans if n]
        if any(s > 1 for _n, s in spans) and rest and is_header_block(rest[0]):
            sub = []
            for line in re.findall(r"^!(.*)$", rest[0], flags=re.M):
                sub += [header_name(c) for c in line.split("!!")]
            sub = [n for n in sub if n]
            flat, cursor = [], 0
            for name, span in spans:
                if span == 1:
                    flat.append(name)
                else:
                    flat += sub[cursor:cursor + span] or [name] * span
                    cursor += span
            return flat, rest[1:]
        return names, rest
    return [], []


def header_colspan(cell):
    m = re.search(r'colspan\s*=\s*"?(\d+)', cell, re.I)
    return int(m.group(1)) if m else 1


def is_header_block(block):
    """A row consisting only of header cells - a continuation header."""
    lines = [l for l in block.splitlines() if l.strip()]
    return bool(lines) and all(l.lstrip().startswith("!") for l in lines)


# The column order every variant of this table follows. Used only to repair a
# header that is missing a cell, never to assign columns by position outright.
CANONICAL_ORDER = ["Week", "Date", "Time", "Opponent", "Result", "Record",
                   "Venue", "Network", "Attendance", "GameBook", "Recap"]


def modal_width(blocks):
    widths = collections.Counter(
        len(row_cells(b)) for b in blocks
        if b.strip() and not re.search(r"colspan", b, re.I))
    return widths.most_common(1)[0][0] if widths else 0


def reconcile_header(names, blocks):
    """Header width must match the rows, or the columns shift silently.

    The 2024 Washington Commanders header omits `Opponent` while its rows
    carry seven cells, so `Result` would be read out of the opponent cell.
    Repair is attempted only when the header is a subsequence of the canonical
    order and filling the gaps makes the widths agree; otherwise it is fatal.
    """
    width = modal_width(blocks)
    # Only a header SHORTER than its rows is dangerous: the columns after the
    # gap all shift. A header longer than the rows is safe - unplayed rows
    # legitimately omit trailing cells such as Attendance, and a missing index
    # simply reads as empty.
    if width == 0 or len(names) >= width:
        return names, blocks
    candidates = []
    if all(n in CANONICAL_ORDER for n in names):
        positions = [CANONICAL_ORDER.index(n) for n in names]
        if positions == sorted(positions):
            missing = [n for n in CANONICAL_ORDER if n not in names]
            for count in range(1, len(missing) + 1):
                for combo in itertools.combinations(missing, count):
                    merged = [n for n in CANONICAL_ORDER
                              if n in names or n in combo]
                    if len(merged) == width:
                        candidates.append(merged)
    for candidate in candidates:
        if header_fits(candidate, blocks):
            return candidate, blocks
    raise ParseError(
        f"header has {len(names)} column(s) {names} but rows carry {width} "
        f"cell(s); {len(candidates)} repair(s) tried and none produced a "
        f"Date column that parses and a Result column that looks like one - "
        f"refusing to assign columns by position")


def header_fits(names, blocks):
    """A repaired header is a hypothesis, so it is tested against the rows.

    Without this, a missing column can be filled in the wrong place and the
    Result cell gets read as the opponent - which is exactly what happened to
    the 2014 New England article before two-level headers were handled.
    """
    index = {name: i for i, name in enumerate(names)}
    if "Date" not in index or "Opponent" not in index:
        return False
    dates = results = opponents = total = 0
    for block in blocks:
        cells = row_cells(block)
        if len(cells) != len(names) or re.search(r"colspan", block, re.I):
            continue
        total += 1
        if parse_date(cells[index["Date"]], 2000)[0] is not None:
            dates += 1
        if LINK.search(cells[index["Opponent"]]):
            opponents += 1
        if "Result" in index and RESULT.match(clean(cells[index["Result"]])):
            results += 1
    if total == 0:
        return False
    return (dates >= 0.9 * total and opponents >= 0.8 * total
            and ("Result" not in index or results >= 0.5 * total))


def row_cells(block):
    cells = []
    for line in re.findall(r"^[!|]\s?(.*)$", block, flags=re.M):
        cells += [cell_body(c) for c in re.split(r"\|\||!!", line)]
    return cells


# ------------------------------------------------------------------ fields

def parse_date(cell, season, season_start_month=F.SEASON_START_MONTH):
    """A month-and-day with no year. January and February are season + 1."""
    text = clean(cell)
    flex = "/" in text
    m = re.match(r"(?:[A-Za-z]+,\s*)?([A-Z][a-z]+)\s+(\d{1,2})(?!\d)", text)
    if not m or m.group(1) not in MONTHS:
        return None, flex, text
    month = MONTHS[m.group(1)]
    year = season + 1 if month < season_start_month - 4 else season
    try:
        return datetime.date(year, month, int(m.group(2))), flex, text
    except ValueError:
        return None, flex, text


KICK = re.compile(r"(\d{1,2}):(\d{2})\s*([ap])\.?\s*m", re.I)


def kickoff_utc(day, time_text, zone_name, link_target):
    if day is None or not time_text:
        return None
    m = KICK.search(time_text.replace(" ", " "))
    if not m:
        return None
    hour, minute = int(m.group(1)) % 12, int(m.group(2))
    if m.group(3).lower() == "p":
        hour += 12
    iana = ZONE_OVERRIDE.get(link_target) or ZONES.get(zone_name or "")
    if iana is None:
        return None
    try:
        local = datetime.datetime(day.year, day.month, day.day, hour, minute,
                                 tzinfo=ZoneInfo(iana))
    except Exception:                                      # noqa: BLE001
        return None
    return local.astimezone(datetime.timezone.utc)


def zone_label(header_line_text):
    for name in ZONES:
        if name in header_line_text:
            return name
    return ""


# ------------------------------------------------------- the summary boxes

def _template_args(text, start):
    depth, i, args, cur = 0, start, {}, ""

    def add(chunk):
        if "=" in chunk:
            key, value = chunk.split("=", 1)
            args[key.strip().lstrip("{").strip()] = value.strip()

    while i < len(text):
        pair = text[i:i + 2]
        if pair in ("{{", "[["):
            depth += 1
            cur += pair
            i += 2
            continue
        if pair in ("}}", "]]"):
            depth -= 1
            if depth == 0:
                add(cur)
                return args
            cur += pair
            i += 2
            continue
        if text[i] == "|" and depth == 1:
            add(cur)
            cur = ""
            i += 1
            continue
        cur += text[i]
        i += 1
    add(cur)
    return args


def game_boxes(text):
    return [_template_args(text, m.start()) for m in BOX.finditer(text)]


def box_periods(box, side):
    """(quarters, overtime, complete?) for 'R' (road) or 'H' (home)."""
    quarters, complete = [], True
    for index in range(1, 5):
        raw = re.sub(r"\D", "", clean(box.get(f"{side}{index}", "")))
        if raw == "":
            complete = False
            quarters.append(None)
        else:
            quarters.append(int(raw))
    overtime = None
    for key in (f"{side}OT", f"{side}5"):
        raw = re.sub(r"\D", "", clean(box.get(key, "")))
        if raw != "":
            overtime = int(raw)
            break
    return quarters, overtime, complete


# ------------------------------------------------------------------ article

def looks_like_preseason(names, blocks, season):
    """A preseason table, by its content rather than by its heading.

    Some articles put both schedules under one `Schedule` heading, so the
    heading alone cannot tell them apart - and with the regular-season header
    broken, the parser otherwise falls back to the preseason table and returns
    four rows that look perfectly legitimate. Two content signals:

      * the recap slug's own type token says `pre`;
      * the games are played in August, which no regular-season game is.
    """
    index = {name: i for i, name in enumerate(names)}
    pre = reg = august = dated = 0
    for block in blocks:
        cells = row_cells(block)
        m = SLUG.search(block)
        if m:
            parts = m.group(1).split("-")
            if len(parts) >= 2 and parts[-2] == "pre":
                pre += 1
            elif len(parts) >= 2 and parts[-2] == "reg":
                reg += 1
        position = index.get("Date")
        if position is not None and position < len(cells):
            day, _flex, _text = parse_date(cells[position], season)
            if day is not None:
                dated += 1
                if day.month == 8:
                    august += 1
    if pre and pre > reg:
        return True
    return dated > 0 and august > dated / 2


def parse_article(season, link_target, text, now=None):
    """Regular-season rows plus the article's game-summary boxes."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    fid = F.franchise_id(link_target, season)
    chosen, chosen_section, saw_preseason, saw_round = None, None, False, False

    for section, raw, implied in tables_with_sections(text):
        names, blocks = split_table(raw, implied)
        if not names:
            continue
        if "Round" in names:
            saw_round = True
            continue                       # postseason, by its own header
        if "Week" not in names:
            continue
        if section in SECTION_PRESEASON:
            saw_preseason = True
            continue                       # preseason, by its own heading
        if section not in SECTION_REGULAR:
            continue
        if looks_like_preseason(names, blocks, season):
            saw_preseason = True
            continue
        scored = sum(1 for b in blocks if RESULT.match(
            clean(dict(zip(names, row_cells(b))).get("Result", ""))))
        if chosen is None or scored > chosen[0]:
            chosen = (scored, names, blocks, raw)
            chosen_section = section

    if chosen is None:
        raise ParseError(
            f"{season} {link_target}: no regular-season schedule table found "
            f"under {sorted(SECTION_REGULAR)}. A silently missing article is "
            f"the failure the completeness check exists for - fix the parse.")

    _scored, names, blocks, chosen_raw = chosen
    # Only the table actually being used is reconciled: the postseason table
    # has its own shape and is discarded above, so widening the check to every
    # table would fail on a table nothing reads.
    names, blocks = reconcile_header(names, blocks)
    # The zone comes from THIS table's Time header cell. Scanning the whole
    # article finds "Eastern Time Zone" in prose on nearly every page and
    # reports every team as Eastern.
    zone = ""
    if "Time" in names:
        parts = re.split(ROW_SEPARATOR, chosen_raw)
        header_block = "\n".join(parts[:2])
        for line in re.findall(r"^!(.*)$", header_block, flags=re.M):
            for piece in line.split("!!"):
                if "Time" in piece or "Kickoff" in piece:
                    zone = zone_label(piece)
                    if zone:
                        break
            if zone:
                break
    index = {name: i for i, name in enumerate(names)}
    rows = []

    for block in blocks:
        cells = row_cells(block)
        if re.search(r"colspan=.*\bBye\b", block, re.I):
            continue
        if not cells:
            continue

        def get(name):
            i = index.get(name)
            return cells[i] if i is not None and i < len(cells) else ""

        week_text = clean(cells[0])
        if week_text in ("", "Week") or week_text.lower().startswith("note"):
            continue
        opponent_raw = get("Opponent")
        if not opponent_raw and not get("Result"):
            continue

        result_text = clean(get("Result"))
        match = RESULT.match(result_text)
        day, flex, date_text = parse_date(get("Date"), season)
        if day is None:
            raise ParseError(
                f"{season} {link_target} week {week_text!r}: date "
                f"{date_text!r} did not parse. A row whose date does not "
                f"parse is fatal, not skipped.")

        link = LINK.search(opponent_raw)
        opponent_display = re.sub(r"^(at |vs\.?\s*)", "", clean(opponent_raw))
        if link is not None:
            opponent_target, identity_from = link.group(2), "link"
        else:
            # Measured: a small number of rows bold the opponent without
            # linking it - 3 of 542 in 2022. The display name is resolved
            # through the SAME authored franchise table, so identity is still
            # authored rather than guessed, and an unresolvable name is fatal.
            opponent_target, identity_from = opponent_display, "display"
        opponent_fid = F.franchise_id(opponent_target, season)

        bare = clean(opponent_raw)
        away = bare.startswith("at ")
        marker_neutral = bool(re.match(r"vs\.?\s", bare))
        time_text = clean(get("Time"))
        kickoff = kickoff_utc(day, time_text, zone, link_target)

        # Four states, not two. A result cell that holds text but not a score
        # is a game that will never be played - the 2022 Bills-Bengals game
        # reads "No contest" - and must go to neither the games table nor the
        # fixtures table.
        if match is not None:
            status = "played"
        elif result_text:
            status = "voided"
        else:
            status = "fixture"
        too_recent = False
        if status == "played" and kickoff is not None and \
                (now - kickoff) < datetime.timedelta(hours=5):
            # A score written during the game must never enter history.
            status, too_recent = "in_progress", True
        played = status == "played"

        rows.append(dict(
            season=season, week=week_text, date=day, date_text=date_text,
            flex=flex, team_target=link_target, team_fid=fid,
            opponent_target=opponent_target, opponent_fid=opponent_fid,
            opponent_display=opponent_display, away=away,
            identity_from=identity_from,
            marker_neutral=marker_neutral,
            wl=match.group(1) if match else "",
            points_for=int(match.group(2)) if match else None,
            points_against=int(match.group(3)) if match else None,
            record=clean(get("Record")), venue=clean(get("Venue")),
            venue_has_note=bool(NOTE_MARKER.search(get("Venue"))),
            network=clean(get("Network")), time_text=time_text,
            zone_text=zone, kickoff_utc=kickoff, played=played,
            status=status, too_recent=too_recent, result_text=result_text,
            outcome_tpl=("won" if re.search(r"\{\{\s*Game-won", block) else
                         "lost" if re.search(r"\{\{\s*Game-lost", block) else
                         "tied" if re.search(r"\{\{\s*Game-(?:tied|drawn)",
                                             block) else ""),
            slug=(SLUG.search(block).group(1) if SLUG.search(block) else ""),
            section=chosen_section, header=tuple(names),
        ))

    return dict(rows=rows, boxes=game_boxes(text), section=chosen_section,
                header=tuple(names), saw_preseason=saw_preseason,
                saw_postseason=saw_round, zone=zone)


SLUG_SIDES = re.compile(
    r"^([a-z0-9\-]+)-at-([a-z0-9\-]+)-(\d{4})-(reg|pre|post)-(\w+)$")


def nickname(link_target):
    """The token the nfl.com slug uses for a franchise."""
    special = {"Washington Football Team": "football-team",
               "San Francisco 49ers": "49ers"}
    if link_target in special:
        return special[link_target]
    return link_target.split()[-1].lower()


def slug_sides(row):
    """(away target, home target) as the recap slug states them, or None."""
    m = SLUG_SIDES.match(row.get("slug") or "")
    if not m:
        return None
    away_token, home_token = m.group(1), m.group(2)
    pair = {row["team_target"], row["opponent_target"]}
    away = [t for t in pair if nickname(t) == away_token]
    home = [t for t in pair if nickname(t) == home_token]
    if len(away) != 1 or len(home) != 1 or away[0] == home[0]:
        return None
    return away[0], home[0]


def resolve_pair(pair):
    """(home row, away row, how) for a game's two rows, or (None, None, why).

    The `at` prefix and the recap slug are independent statements about which
    side was home, and EITHER can be the wrong one:

      * 2022 week 12, Chargers at Cardinals - the two prefixes agreed and the
        Chargers' own slug contradicted its own prefix;
      * 2025 week 7, Colts at Chargers - both slugs agreed and the Colts'
        article was missing its `at`.

    So neither signal is preferred a priori. Whichever is internally
    consistent across both rows decides, and a repair is reported rather than
    applied silently.
    """
    if len(pair) != 2:
        return None, None, f"appears {len(pair)} time(s)"
    first, second = pair
    home = [r for r in pair if not r["away"]]
    away = [r for r in pair if r["away"]]
    if len(home) == 1 and len(away) == 1:
        return home[0], away[0], "prefix"
    verdicts = {slug_sides(r) for r in pair}
    verdicts.discard(None)
    if len(verdicts) == 1:
        away_target, home_target = verdicts.pop()
        by_target = {r["team_target"]: r for r in pair}
        if home_target in by_target and away_target in by_target:
            return by_target[home_target], by_target[away_target], "slug"
    return None, None, ("both rows claim the same side and the slugs do not "
                        "agree either")


def load_raw(season, link_target):
    safe = link_target.replace(" ", "_").replace(".", "")
    path = RAW / str(season) / f"{safe}.wiki"
    if not path.exists():
        raise ParseError(f"{path} is absent - run the fetcher first")
    return path.read_text(encoding="utf-8")


def parse_season(season, now=None, targets=None):
    """Every article of one season. Returns (rows, {target: article})."""
    out, articles = [], {}
    for link_target in (targets or F.teams_for(season)):
        article = parse_article(season, link_target,
                               load_raw(season, link_target), now=now)
        articles[link_target] = article
        out.extend(article["rows"])
    return out, articles
