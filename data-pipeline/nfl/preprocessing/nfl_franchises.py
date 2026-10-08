"""The authored NFL franchise identity table.

The source carries no stable team id - only a display name and a link target -
so unlike the other three leagues, where TEAM_ID comes from nba_api, this
table is authored by the project. Every entry is checkable against the article
titles themselves, which is what makes it reference data rather than
invention.

`franchise_id` values are project-assigned and chosen to be disjoint from the
three leagues already served, whose ids come from nba_api:

    NBA       1610612737 .. 1610612766
    WNBA      1611661313 .. 1611661332
    G League  1612709889 .. 1612709934
    NFL       1613000001 .. 1613000032   <- this table

Identity is keyed on the opponent cell's LINK TARGET (the season article it
points at), never the rendered name, because the link target is the stable
spelling and is present on 96-100% of rows.
"""

# (franchise_id, abbreviation, [(first_season, last_season|None, link target)])
# last_season None means "still current". Only the four franchises that
# changed name or city inside 2012-2026 have more than one span.
FRANCHISES = [
    (1613000001, "ARI", [(None, None, "Arizona Cardinals")]),
    (1613000002, "ATL", [(None, None, "Atlanta Falcons")]),
    (1613000003, "BAL", [(None, None, "Baltimore Ravens")]),
    (1613000004, "BUF", [(None, None, "Buffalo Bills")]),
    (1613000005, "CAR", [(None, None, "Carolina Panthers")]),
    (1613000006, "CHI", [(None, None, "Chicago Bears")]),
    (1613000007, "CIN", [(None, None, "Cincinnati Bengals")]),
    (1613000008, "CLE", [(None, None, "Cleveland Browns")]),
    (1613000009, "DAL", [(None, None, "Dallas Cowboys")]),
    (1613000010, "DEN", [(None, None, "Denver Broncos")]),
    (1613000011, "DET", [(None, None, "Detroit Lions")]),
    (1613000012, "GB", [(None, None, "Green Bay Packers")]),
    (1613000013, "HOU", [(None, None, "Houston Texans")]),
    (1613000014, "IND", [(None, None, "Indianapolis Colts")]),
    (1613000015, "JAX", [(None, None, "Jacksonville Jaguars")]),
    (1613000016, "KC", [(None, None, "Kansas City Chiefs")]),
    (1613000017, "MIA", [(None, None, "Miami Dolphins")]),
    (1613000018, "MIN", [(None, None, "Minnesota Vikings")]),
    (1613000019, "NE", [(None, None, "New England Patriots")]),
    (1613000020, "NO", [(None, None, "New Orleans Saints")]),
    (1613000021, "NYG", [(None, None, "New York Giants")]),
    (1613000022, "NYJ", [(None, None, "New York Jets")]),
    (1613000023, "PHI", [(None, None, "Philadelphia Eagles")]),
    (1613000024, "PIT", [(None, None, "Pittsburgh Steelers")]),
    (1613000025, "SF", [(None, None, "San Francisco 49ers")]),
    (1613000026, "SEA", [(None, None, "Seattle Seahawks")]),
    (1613000027, "TB", [(None, None, "Tampa Bay Buccaneers")]),
    (1613000028, "TEN", [(None, None, "Tennessee Titans")]),
    # --- the four that moved or were renamed inside the window ---
    (1613000029, "LAR", [(None, 2015, "St. Louis Rams"),
                         (2016, None, "Los Angeles Rams")]),
    (1613000030, "LAC", [(None, 2016, "San Diego Chargers"),
                         (2017, None, "Los Angeles Chargers")]),
    (1613000031, "LV", [(None, 2019, "Oakland Raiders"),
                        (2020, None, "Las Vegas Raiders")]),
    (1613000032, "WAS", [(None, 2019, "Washington Redskins"),
                         (2020, 2021, "Washington Football Team"),
                         (2022, None, "Washington Commanders")]),
]

# Abbreviation as written at the time, where it differs from the current one.
PERIOD_ABBREV = {
    (1613000029, "St. Louis Rams"): "STL",
    (1613000030, "San Diego Chargers"): "SD",
    (1613000031, "Oakland Raiders"): "OAK",
}

SEASON_START_MONTH = 9     # an NFL season is labelled by the year it starts in
FIRST_SEASON = 2012        # a scope decision; see nfl/README.md


class UnknownFranchise(KeyError):
    """A link target that is not in this table. Fatal by design."""


def _spans():
    for fid, abbr, spans in FRANCHISES:
        for first, last, target in spans:
            yield fid, abbr, first, last, target


def franchise_table(season):
    """{link target: (franchise_id, abbreviation, display name)}."""
    out = {}
    for fid, abbr, first, last, target in _spans():
        if first is not None and season < first:
            continue
        if last is not None and season > last:
            continue
        out[target] = (fid, PERIOD_ABBREV.get((fid, target), abbr), target)
    return out


def franchise_id(link_target, season):
    table = franchise_table(season)
    if link_target not in table:
        raise UnknownFranchise(
            f"{link_target!r} is not an NFL franchise in {season}. "
            f"Add it to nfl_franchises.FRANCHISES with its seasons, or fix "
            f"the parse - an unrecognised link target is never skipped.")
    return table[link_target][0]


def teams_for(season):
    """The 32 link targets for one season, which are the article names."""
    targets = sorted(franchise_table(season))
    if len(targets) != 32:
        raise RuntimeError(
            f"{season}: franchise table yields {len(targets)} teams, not 32")
    return targets


def article_title(link_target, season):
    return f"{season} {link_target} season"


def current_season(today=None):
    """Today -> the season label now underway."""
    import datetime
    today = today or datetime.date.today()
    return today.year if today.month >= SEASON_START_MONTH else today.year - 1


def open_seasons(today=None):
    """Seasons that may still change: the current one and the one before it.

    The same rule the other three leagues use. A completed season is frozen
    by its stored revision ids and is never refetched unless asked.
    """
    now = current_season(today)
    return {now, now - 1}


def seasons_through(today=None):
    return list(range(FIRST_SEASON, current_season(today) + 1))


def changes_in_window(first=FIRST_SEASON, last=None):
    """Every rename or relocation whose boundary falls inside the window."""
    last = last or current_season()
    out = []
    for fid, abbr, spans in FRANCHISES:
        if len(spans) < 2:
            continue
        for prev, nxt in zip(spans, spans[1:]):
            if first < nxt[0] <= last:
                out.append((fid, abbr, prev[2], nxt[2], nxt[0]))
    return out
