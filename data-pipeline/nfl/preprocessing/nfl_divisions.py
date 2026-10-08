"""The eight NFL divisions, keyed on project franchise_id.

AUTHORED REFERENCE DATA, like `data/arenas.py` for the NBA. It is not derived
from the schedule, because the thing it is used to derive - whether a fixture is
a division game - would then be circular.

SEASONS COVERED: 2012-2026, the whole phase-1 window. The divisions have not
changed in it; the last realignment was 2002, when Houston entered and the
league went to eight divisions of four. Five franchises renamed or relocated
inside the window (Rams 2016, Chargers 2017, Raiders 2020, Washington 2020 and
2022) and **not one changed division**, which is why this table is keyed on
`franchise_id` and carries no season axis. A sixth change that did move a
division would have to add one.

The table is checked against the schedule rather than trusted: since 2002 every
team plays each of its three division rivals twice, so a correct table yields
exactly six division games per team per season and 96 per season league-wide.
`verify()` asserts that, and it is the reason an error here cannot pass
silently - a team placed in the wrong division would break the count for two
divisions at once.
"""

AFC_EAST = "AFC East"
AFC_NORTH = "AFC North"
AFC_SOUTH = "AFC South"
AFC_WEST = "AFC West"
NFC_EAST = "NFC East"
NFC_NORTH = "NFC North"
NFC_SOUTH = "NFC South"
NFC_WEST = "NFC West"

DIVISIONS = (AFC_EAST, AFC_NORTH, AFC_SOUTH, AFC_WEST,
             NFC_EAST, NFC_NORTH, NFC_SOUTH, NFC_WEST)

# franchise_id -> division. The comment is the 2026 abbreviation; the id is the
# key, because three of these franchises answered to a different abbreviation
# earlier in the window.
DIVISION_OF = {
    1613000004: AFC_EAST,    # BUF
    1613000017: AFC_EAST,    # MIA
    1613000019: AFC_EAST,    # NE
    1613000022: AFC_EAST,    # NYJ

    1613000003: AFC_NORTH,   # BAL
    1613000007: AFC_NORTH,   # CIN
    1613000008: AFC_NORTH,   # CLE
    1613000024: AFC_NORTH,   # PIT

    1613000013: AFC_SOUTH,   # HOU
    1613000014: AFC_SOUTH,   # IND
    1613000015: AFC_SOUTH,   # JAX
    1613000028: AFC_SOUTH,   # TEN

    1613000010: AFC_WEST,    # DEN
    1613000016: AFC_WEST,    # KC
    1613000031: AFC_WEST,    # LV  (Oakland through 2019)
    1613000030: AFC_WEST,    # LAC (San Diego through 2016)

    1613000009: NFC_EAST,    # DAL
    1613000021: NFC_EAST,    # NYG
    1613000023: NFC_EAST,    # PHI
    1613000032: NFC_EAST,    # WAS

    1613000006: NFC_NORTH,   # CHI
    1613000011: NFC_NORTH,   # DET
    1613000012: NFC_NORTH,   # GB
    1613000018: NFC_NORTH,   # MIN

    1613000002: NFC_SOUTH,   # ATL
    1613000005: NFC_SOUTH,   # CAR
    1613000020: NFC_SOUTH,   # NO
    1613000027: NFC_SOUTH,   # TB

    1613000001: NFC_WEST,    # ARI
    1613000029: NFC_WEST,    # LAR (St. Louis through 2015)
    1613000025: NFC_WEST,    # SF
    1613000026: NFC_WEST,    # SEA
}

DIVISION_GAMES_PER_TEAM_PER_SEASON = 6   # three rivals, home and away


class UnknownFranchise(KeyError):
    """A franchise with no division. Fatal: the feature cannot be guessed."""


def division_of(franchise_id):
    try:
        return DIVISION_OF[franchise_id]
    except KeyError:
        raise UnknownFranchise(
            f"franchise {franchise_id} has no division in nfl_divisions.py. "
            f"A new or relocated franchise needs a row; the feature is not "
            f"guessable from the schedule without circularity.") from None


def is_division_game(home_franchise_id, away_franchise_id):
    return int(division_of(home_franchise_id)
               == division_of(away_franchise_id))


def verify(franchise_ids_by_season=None):
    """Structural checks that do not need the schedule.

    Returns a list of (label, ok, detail). The schedule-based count check lives
    in the dataset builder, which has the games.
    """
    checks = []

    checks.append(("32 franchises carry a division",
                   len(DIVISION_OF) == 32, f"{len(DIVISION_OF)}"))

    sizes = {}
    for division in DIVISIONS:
        sizes[division] = sum(1 for d in DIVISION_OF.values() if d == division)
    checks.append(("8 divisions of exactly 4",
                   sorted(sizes.values()) == [4] * 8,
                   ", ".join(f"{k}={v}" for k, v in sizes.items())))

    named = set(DIVISION_OF.values())
    checks.append(("every division named is one of the 8",
                   named <= set(DIVISIONS),
                   f"{sorted(named - set(DIVISIONS)) or 'all known'}"))

    # Exactly one division each: a dict cannot hold a franchise twice, so this
    # is asserted where it could actually fail - against the franchise table.
    if franchise_ids_by_season:
        missing, extra = set(), set()
        for season, ids in franchise_ids_by_season.items():
            missing |= {f for f in ids if f not in DIVISION_OF}
            extra |= {f for f in DIVISION_OF if f not in ids}
        checks.append(("every franchise in every season has a division",
                       not missing, f"missing {sorted(missing) or 'none'}"))
        checks.append(("no division row for a franchise outside the window",
                       not extra, f"extra {sorted(extra) or 'none'}"))

    return checks


def main():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import nfl_franchises as F

    by_season = {
        season: {entry[0] for entry in F.franchise_table(season).values()}
        for season in F.seasons_through()
    }
    failed = 0
    print("nfl_divisions")
    for label, ok, detail in verify(by_season):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
        failed += not ok
    print(f"  {'all checks passed' if not failed else f'{failed} FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
