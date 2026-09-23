"""Arena coordinates and IANA time zones for the 30 NBA teams.

Reference data, not derived output - committed rather than gitignored.

Two accuracy choices worth stating:

ARENA COORDINATES, NOT CITY CENTROIDS. The difference is small against typical
NBA trip distances, but it is free to be correct.

IANA ZONE NAMES, NOT UTC OFFSETS. America/Phoenix does not observe daylight
saving, so a fixed offset is wrong for the Suns for part of every season and
wrong for every opponent travelling to or from Phoenix. Offsets are computed
per game date from the zone.

KNOWN SIMPLIFICATION: one current arena per team. Four teams changed venue
inside the 2015-2025 window - Warriors (Oakland to San Francisco, 2019),
Pistons (Auburn Hills to downtown Detroit, 2017), Kings (Natomas to downtown
Sacramento, 2016) and Clippers (Crypto.com to Intuit Dome, 2024). Every one of
those moves is under ~40 km against a feature whose range is 0-4,400 km, and
none crosses a time zone. Modelling arena history would cost more than the
error it removes, so it is recorded here rather than built.
"""

ARENAS = {
    1610612737: ("Atlanta Hawks", "State Farm Arena", 33.7573, -84.3963, "America/New_York"),
    1610612738: ("Boston Celtics", "TD Garden", 42.3662, -71.0621, "America/New_York"),
    1610612739: ("Cleveland Cavaliers", "Rocket Arena", 41.4965, -81.6882, "America/New_York"),
    1610612740: ("New Orleans Pelicans", "Smoothie King Center", 29.9490, -90.0821, "America/Chicago"),
    1610612741: ("Chicago Bulls", "United Center", 41.8807, -87.6742, "America/Chicago"),
    1610612742: ("Dallas Mavericks", "American Airlines Center", 32.7905, -96.8103, "America/Chicago"),
    1610612743: ("Denver Nuggets", "Ball Arena", 39.7487, -105.0077, "America/Denver"),
    1610612744: ("Golden State Warriors", "Chase Center", 37.7680, -122.3878, "America/Los_Angeles"),
    1610612745: ("Houston Rockets", "Toyota Center", 29.7508, -95.3621, "America/Chicago"),
    1610612746: ("LA Clippers", "Intuit Dome", 33.9453, -118.3417, "America/Los_Angeles"),
    1610612747: ("Los Angeles Lakers", "Crypto.com Arena", 34.0430, -118.2673, "America/Los_Angeles"),
    1610612748: ("Miami Heat", "Kaseya Center", 25.7814, -80.1870, "America/New_York"),
    1610612749: ("Milwaukee Bucks", "Fiserv Forum", 43.0451, -87.9172, "America/Chicago"),
    1610612750: ("Minnesota Timberwolves", "Target Center", 44.9795, -93.2760, "America/Chicago"),
    1610612751: ("Brooklyn Nets", "Barclays Center", 40.6826, -73.9754, "America/New_York"),
    1610612752: ("New York Knicks", "Madison Square Garden", 40.7505, -73.9934, "America/New_York"),
    1610612753: ("Orlando Magic", "Kia Center", 28.5392, -81.3839, "America/New_York"),
    # Indiana observes DST now, so this behaves as Eastern - but the zone name
    # is kept exact rather than folded into America/New_York.
    1610612754: ("Indiana Pacers", "Gainbridge Fieldhouse", 39.7640, -86.1555, "America/Indiana/Indianapolis"),
    1610612755: ("Philadelphia 76ers", "Wells Fargo Center", 39.9012, -75.1720, "America/New_York"),
    1610612756: ("Phoenix Suns", "Footprint Center", 33.4457, -112.0712, "America/Phoenix"),
    1610612757: ("Portland Trail Blazers", "Moda Center", 45.5316, -122.6668, "America/Los_Angeles"),
    1610612758: ("Sacramento Kings", "Golden 1 Center", 38.5802, -121.4997, "America/Los_Angeles"),
    1610612759: ("San Antonio Spurs", "Frost Bank Center", 29.4270, -98.4375, "America/Chicago"),
    1610612760: ("Oklahoma City Thunder", "Paycom Center", 35.4634, -97.5151, "America/Chicago"),
    1610612761: ("Toronto Raptors", "Scotiabank Arena", 43.6435, -79.3791, "America/Toronto"),
    1610612762: ("Utah Jazz", "Delta Center", 40.7683, -111.9011, "America/Denver"),
    1610612763: ("Memphis Grizzlies", "FedExForum", 35.1382, -90.0506, "America/Chicago"),
    1610612764: ("Washington Wizards", "Capital One Arena", 38.8981, -77.0209, "America/New_York"),
    1610612765: ("Detroit Pistons", "Little Caesars Arena", 42.3410, -83.0552, "America/Detroit"),
    1610612766: ("Charlotte Hornets", "Spectrum Center", 35.2251, -80.8392, "America/New_York"),
}

# The one zone in the league that does not observe daylight saving.
NO_DST_ZONE = "America/Phoenix"

LATITUDE_RANGE = (25.0, 50.0)
LONGITUDE_RANGE = (-125.0, -70.0)


def team_ids() -> set:
    return set(ARENAS)


def coordinates(team_id: int) -> tuple:
    return ARENAS[team_id][2], ARENAS[team_id][3]


def timezone_name(team_id: int) -> str:
    return ARENAS[team_id][4]


def validate() -> None:
    """Coordinates plausible, zones loadable, no duplicate venues."""
    from zoneinfo import ZoneInfo

    if len(ARENAS) != 30:
        raise SystemExit(f"expected 30 teams, got {len(ARENAS)}")

    seen = {}
    for team_id, (name, arena, lat, lon, zone) in ARENAS.items():
        if not LATITUDE_RANGE[0] <= lat <= LATITUDE_RANGE[1]:
            raise SystemExit(f"{name}: latitude {lat} outside {LATITUDE_RANGE}")
        if not LONGITUDE_RANGE[0] <= lon <= LONGITUDE_RANGE[1]:
            raise SystemExit(f"{name}: longitude {lon} outside {LONGITUDE_RANGE}")
        ZoneInfo(zone)

        key = (round(lat, 3), round(lon, 3))
        if key in seen:
            raise SystemExit(f"{name} and {seen[key]} share a venue location")
        seen[key] = name
