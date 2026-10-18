"""The authored NHL franchise identity table.

The source carries no stable team id - only a display name and a link target -
so as for the NFL this table is authored by the project. Every entry is
checkable against the article titles themselves, which is what makes it
reference data rather than invention: `fetch_nhl_seasons.py` asks the API
whether each season's titles exist, and `build_nhl_tables.py` refuses a link
target that is not in here.

`franchise_id` values are project-assigned and chosen to be disjoint from the
four leagues already served:

    NBA       1610612737 .. 1610612766   (nba_api)
    WNBA      1611661313 .. 1611661332   (nba_api)
    G League  1612709889 .. 1612709934   (nba_api)
    NFL       1613000001 .. 1613000032   (project-assigned)
    NHL       1614000001 .. 1614000033   <- this table

A season is named by its STARTING year: 2021 means 2021-22. The article title
uses an EN DASH, which `season_label` is the only place that knows.
"""

# A scope decision, not a fact that drifts: 2012-13 is excluded because 11 of
# its 30 articles have no game log at all, so its failures are a coverage
# problem rather than a data one (round 4, Q8). This is the ONLY literal in
# this pipeline that an operator would ever want to change.
FIRST_SEASON = 2013

EN_DASH = "–"


def season_label(start_year):
    """2021 -> '2021-22' with an EN DASH, which is what the titles use.

    A hyphen is NOT equivalent. `2021-22 Montreal Canadiens season` exists as
    a REDIRECT, and this pipeline fetches without resolving redirects, so a
    hyphen would return a page whose entire content is `#REDIRECT [[...]]` -
    which parses to zero games and raises nothing. Measured, not assumed.
    """
    return "%d%s%02d" % (start_year, EN_DASH, (start_year + 1) % 100)


def article_title(start_year, link_target):
    """The team-season article title for one franchise in one season."""
    return "%s %s season" % (season_label(start_year), link_target)


def season_article_title(start_year):
    """The league-season article, which names the division standings
    templates for every season before the whole-league one existed."""
    return "%s NHL season" % season_label(start_year)


# (franchise_id, current abbreviation, [(first_season, last_season|None,
#  link target)]). last_season None means "still current"; first_season None
#  means "since before this window opened".
FRANCHISES = [
    (1614000001, "ANA", [(None, None, "Anaheim Ducks")]),
    (1614000002, "BOS", [(None, None, "Boston Bruins")]),
    (1614000003, "BUF", [(None, None, "Buffalo Sabres")]),
    (1614000004, "CGY", [(None, None, "Calgary Flames")]),
    (1614000005, "CAR", [(None, None, "Carolina Hurricanes")]),
    (1614000006, "CHI", [(None, None, "Chicago Blackhawks")]),
    (1614000007, "COL", [(None, None, "Colorado Avalanche")]),
    (1614000008, "CBJ", [(None, None, "Columbus Blue Jackets")]),
    (1614000009, "DAL", [(None, None, "Dallas Stars")]),
    (1614000010, "DET", [(None, None, "Detroit Red Wings")]),
    (1614000011, "EDM", [(None, None, "Edmonton Oilers")]),
    (1614000012, "FLA", [(None, None, "Florida Panthers")]),
    (1614000013, "LAK", [(None, None, "Los Angeles Kings")]),
    (1614000014, "MIN", [(None, None, "Minnesota Wild")]),
    (1614000015, "MTL", [(None, None, "Montreal Canadiens")]),
    (1614000016, "NSH", [(None, None, "Nashville Predators")]),
    (1614000017, "NJD", [(None, None, "New Jersey Devils")]),
    (1614000018, "NYI", [(None, None, "New York Islanders")]),
    (1614000019, "NYR", [(None, None, "New York Rangers")]),
    (1614000020, "OTT", [(None, None, "Ottawa Senators")]),
    (1614000021, "PHI", [(None, None, "Philadelphia Flyers")]),
    (1614000022, "PIT", [(None, None, "Pittsburgh Penguins")]),
    (1614000023, "SJS", [(None, None, "San Jose Sharks")]),
    (1614000024, "STL", [(None, None, "St. Louis Blues")]),
    (1614000025, "TBL", [(None, None, "Tampa Bay Lightning")]),
    (1614000026, "TOR", [(None, None, "Toronto Maple Leafs")]),
    (1614000027, "VAN", [(None, None, "Vancouver Canucks")]),
    (1614000028, "WSH", [(None, None, "Washington Capitals")]),
    (1614000029, "WPG", [(None, None, "Winnipeg Jets")]),
    # --- expansion inside the window ---
    (1614000030, "VGK", [(2017, None, "Vegas Golden Knights")]),
    (1614000031, "SEA", [(2021, None, "Seattle Kraken")]),
    # --- THE RENAME CHAIN, AND IT IS TWO FRANCHISES, NOT ONE ---
    # Phoenix -> Arizona is a rename of one club. Arizona -> Utah is NOT: the
    # league records the Coyotes' franchise as deactivated and Utah as a new
    # club, so they get separate ids here. Whether Elo should carry across the
    # relocation is a PHASE 2 decision and is recorded there, not settled by
    # this table.
    (1614000032, "ARI", [(None, 2013, "Phoenix Coyotes"),
                         (2014, 2023, "Arizona Coyotes")]),
    (1614000033, "UTA", [(2024, 2024, "Utah Hockey Club"),
                         (2025, None, "Utah Mammoth")]),
]

# Abbreviation as the standings templates wrote it at the time, where it
# differs from the current one.
PERIOD_ABBREV = {
    (1614000032, "Phoenix Coyotes"): "PHX",
}

# LINK TARGETS THAT ARE NOT NHL FRANCHISES, named with their evidence.
#
# Round 4 found exactly two, on three rows in the whole corpus. They are
# listed rather than tolerated, and `resolve_link_target` raises on anything
# not in this map or in FRANCHISES - so a NEW bad target is fatal, which is
# the point. The reverse guard lives in validate_nhl_games.py: a listed entry
# that no longer appears is also fatal, so this map cannot quietly rot.
KNOWN_BAD_LINK_TARGETS = {
    # A basketball team. The hockey club in Dallas is the Stars, and the
    # cell's own display text says so.
    "Dallas Mavericks": (1614000009, "a basketball team; the Dallas hockey "
                                     "club is the Stars"),
    # A misspelling of Avalanche, one character short.
    "Colorado Avalance": (1614000007, "misspelling of Colorado Avalanche"),
    # Found on the fetched corpus, which round 4's sample did not reach.
    "Colombus": (1614000008, "misspelling of Columbus, in 2026-27 San Jose"),
}


# SHORT FORMS, as the unlinked team cells write them. Round 4 measured 7,907
# rows with both sides unlinked; scanning the fetched corpus for every cell
# the link-target rule could not resolve returned exactly 34 distinct strings,
# all of them these. The counts beside each are from that scan, which is what
# makes this measured reference data rather than a guess at a naming
# convention - and note the two New York clubs are written `NY Rangers` and
# `NY Islanders`, never `New York`, so a mechanical "drop the nickname" rule
# would have made them ambiguous.
ALIASES = {
    "Anaheim": 1614000001,          # 925
    "Boston": 1614000002,           # 991
    "Buffalo": 1614000003,          # 1304
    "Calgary": 1614000004,          # 1342
    "Carolina": 1614000005,         # 813
    "Chicago": 1614000006,          # 554
    "Colorado": 1614000007,         # 1012
    "Columbus": 1614000008,         # 1295
    "Dallas": 1614000009,           # 1338
    "Detroit": 1614000010,          # 1311
    "Edmonton": 1614000011,         # 1346
    "Florida": 1614000012,          # 1276
    "Los Angeles": 1614000013,      # 840
    "Minnesota": 1614000014,        # 1292
    "Montreal": 1614000015,         # 1285
    "Nashville": 1614000016,        # 1139
    "New Jersey": 1614000017,       # 1305
    "NY Islanders": 1614000018,     # 1065
    "NY Rangers": 1614000019,       # 204
    "Ottawa": 1614000020,           # 1273
    "Philadelphia": 1614000021,     # 830
    "Pittsburgh": 1614000022,       # 722
    "San Jose": 1614000023,         # 1179
    "St. Louis": 1614000024,        # 1317
    "Tampa Bay": 1614000025,        # 200
    "Toronto": 1614000026,          # 774
    "Vancouver": 1614000027,        # 1230
    "Washington": 1614000028,       # 786
    "Winnipeg": 1614000029,         # 1146
    "Vegas": 1614000030,            # 865
    "Seattle": 1614000031,          # 538
    "Phoenix": 1614000032,          # 142, and the full name for 2013-14
    "Arizona": 1614000032,          # 927
    "Utah": 1614000033,             # 267
}

# MALFORMED LINKS IN THE SOURCE, named with their evidence. Two rows, both in
# 2018-19 Minnesota, where the cell reads `[|Minnesota` - a single bracket
# then a pipe. The display name is unambiguous and the row is otherwise
# complete, so it is resolved rather than discarded; an UNLISTED malformed
# cell is still fatal, which is what keeps this a list of two rather than a
# tolerance.
MALFORMED_CELLS = {
    "[|Minnesota": (1614000014, "a single-bracket link in 2018-19 "
                                "Minnesota; the display name is Minnesota"),
    # 2025-26 St. Louis, game 53: the cell reads `Dallas Stars season|Dallas`
    # with the `[[2025-26 ` opening lost entirely. Both halves name Dallas.
    "Dallas Stars season|Dallas": (
        1614000009, "a link that lost its brackets and season prefix in "
                    "2025-26 St. Louis game 53; both halves say Dallas"),
}


def _spans():
    for fid, abbrev, spans in FRANCHISES:
        for first, last, target in spans:
            yield fid, abbrev, first, last, target


def franchises_for(start_year):
    """{link target: franchise_id} for the teams that played that season."""
    out = {}
    for fid, _abbrev, first, last, target in _spans():
        if (first is None or start_year >= first) and \
           (last is None or start_year <= last):
            out[target] = fid
    return out


def titles_for(start_year):
    """{article title: franchise_id} for one season."""
    return {article_title(start_year, target): fid
            for target, fid in franchises_for(start_year).items()}


def abbrev_for(fid, start_year):
    """The abbreviation that season's standings template would have used."""
    for f, abbrev, first, last, target in _spans():
        if f != fid:
            continue
        if (first is None or start_year >= first) and \
           (last is None or start_year <= last):
            return PERIOD_ABBREV.get((fid, target), abbrev)
    raise UnknownFranchise(
        "franchise %s did not play in %s" % (fid, season_label(start_year)))


def abbrev_index(start_year):
    """{abbreviation: franchise_id} for one season's standings templates."""
    return {abbrev_for(fid, start_year): fid
            for fid in set(franchises_for(start_year).values())}


class UnknownFranchise(RuntimeError):
    """A link target or abbreviation that this table does not describe."""


def resolve_link_target(target, start_year=None):
    """(franchise_id, note) for one opponent-cell link target.

    Fatal on anything unknown. A link target is VALIDATED against this table,
    never trusted: the corpus contains two that are not NHL franchises.
    """
    clean = target.strip().replace("_", " ")
    for fid, _abbrev, first, last, name in _spans():
        if name != clean:
            continue
        if start_year is not None:
            if first is not None and start_year < first:
                continue
            if last is not None and start_year > last:
                continue
        return fid, ""
    if clean in ALIASES:
        fid = ALIASES[clean]
        if start_year is not None and fid not in \
                set(franchises_for(start_year).values()):
            raise UnknownFranchise(
                "%r resolves to a franchise that did not play in %s"
                % (clean, season_label(start_year)))
        return fid, ""
    if clean in KNOWN_BAD_LINK_TARGETS:
        fid, why = KNOWN_BAD_LINK_TARGETS[clean]
        return fid, why
    if clean in MALFORMED_CELLS:
        fid, why = MALFORMED_CELLS[clean]
        return fid, why
    raise UnknownFranchise(
        "%r is not in the authored franchise table%s. Add it with its "
        "evidence, or correct the parse - an unrecognised link target is "
        "fatal rather than guessed."
        % (clean, "" if start_year is None
           else " for %s" % season_label(start_year)))


def seasons_through(today=None):
    """[FIRST_SEASON .. the season currently underway].

    Derived, never a literal list: the hardcoded-season-list failure in §35
    was a silent no-op for an entire offseason. An NHL season starts in
    October, so a date in September still belongs to the previous one.
    """
    import datetime as dt
    today = today or dt.date.today()
    newest = today.year if today.month >= 10 else today.year - 1
    return list(range(FIRST_SEASON, newest + 1))
