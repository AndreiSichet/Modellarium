"""A deterministic synthetic NFL table set, for CI only.

WHY THIS EXISTS. The real NFL tables are derived from English Wikipedia and are
CC BY-SA 4.0; this repository is not licensed to redistribute a derived
database, so no NFL table is committed (`data-pipeline/nfl/DATA_LICENSE.md`).
But CI boots the inference container against the COMMITTED data copy, and from
phase 4 that container refuses to boot without the NFL tables. Those two facts
are only compatible if CI gets a table set that carries no Wikipedia-derived
content at all.

SO EVERY NUMBER HERE IS INVENTED, AND NOTHING IS COPIED. The scores come from a
seeded PRNG; the dates come from arithmetic on a September Sunday; the venues
are the string "Synthetic Stadium". The only thing shared with the real tables
is the SCHEMA and the 32 franchise ids - and those are project-authored
reference data in `nfl_franchises.py`, already committed, not content from the
source.

IT IS MARKED, AND PRODUCTION REFUSES IT. `nfl_synthetic.json` sits beside the
tables and `live_nfl_features.refuse_synthetic` raises on it unless
MODELLARIUM_ALLOW_SYNTHETIC_NFL=1. A marker file rather than a column, because
a column would have to be read, trusted, and kept out of every feature list;
a file is checked once at boot and can never reach a model.

WHAT THIS SET IS AND IS NOT FOR. It exists so the container BOOTS and so the
`models_loaded` assertion can run. No recorded number, no wire-shape fixture
and no prediction value may be taken from it - the scores are noise, so a
prediction computed from them means nothing beyond "the code path ran".

    python data-pipeline/nfl/preprocessing/make_synthetic_nfl_tables.py [root]
"""

import csv
import datetime
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NFL = HERE.parent
PIPELINE = NFL.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nfl_franchises as F  # noqa: E402

DEFAULT_ROOT = PIPELINE / "data"

# Enough seasons for the carried windows to fill and for an Elo replay to have
# something to work on, and no more: this set exists to boot a container.
SEASONS = (2023, 2024, 2025)
FIXTURE_SEASON = 2026
WEEKS_PER_SEASON = 8
FIXTURE_WEEKS = (1, 2)
SEED = 4269

GAMES_COLUMNS = ["game_id", "season", "week", "date", "franchise_id", "team",
                 "opponent_franchise_id", "opponent", "is_home",
                 "neutral_site", "points_for", "points_against", "result",
                 "margin", "is_tie", "venue"]
FIXTURES_COLUMNS = ["game_id", "season", "week", "date", "kickoff_local",
                    "kickoff_zone_as_written", "away_franchise_id", "away",
                    "home_franchise_id", "home", "neutral_site",
                    "flex_two_dates", "venue"]
IDENTITY_COLUMNS = ["franchise_id", "season", "link_target", "display_name",
                    "abbreviation"]

VENUE = "Synthetic Stadium"
ZONE = "Eastern Time Zone"


def first_sunday(season: int) -> datetime.date:
    """The second Sunday of September, so week 1 lands in a plausible place."""
    day = datetime.date(season, 9, 8)
    while day.weekday() != 6:
        day += datetime.timedelta(days=1)
    return day


def franchises():
    """(franchise_id, abbreviation, link target) for the newest season.

    From the AUTHORED table, which is committed project reference data rather
    than source content - the ids have to be the real ones or the backend's
    seeded teams would not match what this set serves.
    """
    table = F.franchise_table(max(SEASONS))
    out = [(fid, abbr, target) for target, (fid, abbr, _name)
           in sorted(table.items(), key=lambda kv: kv[1][0])]
    if len(out) != 32:
        raise RuntimeError(f"expected 32 franchises, got {len(out)}")
    return out


def build():
    """Rows for the three tables. Deterministic for a fixed SEED."""
    rng = random.Random(SEED)
    teams = franchises()
    games, fixtures, identity = [], [], []

    for season in SEASONS:
        for fid, abbr, target in teams:
            identity.append({
                "franchise_id": fid, "season": season, "link_target": target,
                "display_name": target, "abbreviation": abbr,
            })
        opening = first_sunday(season)
        for week in range(1, WEEKS_PER_SEASON + 1):
            day = opening + datetime.timedelta(days=7 * (week - 1))
            # A fixed rotation rather than a random pairing, so every team
            # plays exactly once a week and the schedule-shape checks hold.
            rotated = teams[:1] + teams[1:][-(week - 1):] + teams[1:][:-(week - 1)] \
                if week > 1 else list(teams)
            for index in range(0, len(rotated), 2):
                home_id, home_abbr, _h = rotated[index]
                away_id, away_abbr, _a = rotated[index + 1]
                home_points = rng.randint(3, 38)
                away_points = rng.randint(3, 38)
                if home_points == away_points:          # no ties in this set
                    home_points += 3
                game_id = f"{season}_{week}_{away_abbr}_{home_abbr}"
                for own_id, own_abbr, other_id, other_abbr, is_home in (
                        (home_id, home_abbr, away_id, away_abbr, 1),
                        (away_id, away_abbr, home_id, home_abbr, 0)):
                    pf = home_points if is_home else away_points
                    pa = away_points if is_home else home_points
                    games.append({
                        "game_id": game_id, "season": season, "week": week,
                        "date": day.isoformat(), "franchise_id": own_id,
                        "team": own_abbr,
                        "opponent_franchise_id": other_id,
                        "opponent": other_abbr, "is_home": is_home,
                        "neutral_site": 0, "points_for": pf,
                        "points_against": pa,
                        "result": "W" if pf > pa else "L",
                        "margin": pf - pa, "is_tie": 0, "venue": VENUE,
                    })

    for fid, abbr, target in franchises():
        identity.append({
            "franchise_id": fid, "season": FIXTURE_SEASON,
            "link_target": target, "display_name": target,
            "abbreviation": abbr,
        })
    opening = first_sunday(FIXTURE_SEASON)
    for week in FIXTURE_WEEKS:
        day = opening + datetime.timedelta(days=7 * (week - 1))
        rotated = teams[:1] + teams[1:][-(week - 1):] + teams[1:][:-(week - 1)] \
            if week > 1 else list(teams)
        for index in range(0, len(rotated), 2):
            home_id, home_abbr, _h = rotated[index]
            away_id, away_abbr, _a = rotated[index + 1]
            fixtures.append({
                "game_id": f"{FIXTURE_SEASON}_{week}_{away_abbr}_{home_abbr}",
                "season": FIXTURE_SEASON, "week": week,
                "date": day.isoformat(), "kickoff_local": "1:00p.m.",
                "kickoff_zone_as_written": ZONE,
                "away_franchise_id": away_id, "away": away_abbr,
                "home_franchise_id": home_id, "home": home_abbr,
                "neutral_site": 0, "flex_two_dates": 0, "venue": VENUE,
            })
    return games, fixtures, identity


def _write_csv(path: Path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class WouldOverwriteRealTables(RuntimeError):
    """The target already holds a real NFL table set."""


def write(root: Path = DEFAULT_ROOT, force: bool = False) -> dict:
    """Write the three tables plus the marker. Returns the marker's contents.

    REFUSES TO OVERWRITE A REAL SET, and that guard is not theoretical. CI
    calls this with no argument, so the target is `data-pipeline/data` - which
    in CI holds no NFL tables at all, because they are never committed, and on
    a development machine or on the self-hosted runner holds 46 MB of fetched
    history and the tables the daily refresh stages from. Running this there
    by accident would replace served history with seeded noise, and the daily
    refresh would then stage it.

    So: a target carrying nfl_games_final.csv WITHOUT the synthetic marker is
    refused. Re-running over an existing synthetic set is allowed, because
    that is what CI does on every push.
    """
    processed = Path(root) / "nfl" / "processed"
    existing = processed / "nfl_games_final.csv"
    if existing.is_file() and not (processed / "nfl_synthetic.json").is_file():
        if not force:
            raise WouldOverwriteRealTables(
                f"{existing} exists and carries no synthetic marker, so it is "
                f"a REAL NFL table set. Refusing to replace it with invented "
                f"scores. In CI this path is empty because no NFL table is "
                f"ever committed; if you meant to do this, pass --force, and "
                f"re-run the pipeline afterwards to restore the real tables."
            )
        print(f"  --force: overwriting what looks like a REAL table set at "
              f"{processed}")

    games, fixtures, identity = build()
    _write_csv(processed / "nfl_games_final.csv", GAMES_COLUMNS, games)
    _write_csv(processed / "nfl_fixtures.csv", FIXTURES_COLUMNS, fixtures)
    _write_csv(processed / "nfl_franchise_identity.csv", IDENTITY_COLUMNS,
               identity)

    marker = {
        "synthetic": True,
        "generated_by": "make_synthetic_nfl_tables.py",
        "seed": SEED,
        "seasons": list(SEASONS),
        "fixture_season": FIXTURE_SEASON,
        "games": len(games) // 2,
        "fixtures": len(fixtures),
        "why": "the real NFL tables are CC BY-SA 4.0 and are never committed, "
               "so CI boots the inference container against invented scores",
        "do_not": "no recorded number, fixture or prediction may come from "
                  "this set - the scores are noise",
    }
    (processed / "nfl_synthetic.json").write_text(
        json.dumps(marker, indent=2), encoding="utf-8")
    return marker


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=DEFAULT_ROOT, type=Path)
    parser.add_argument("--force", action="store_true",
                       help="overwrite even a real-looking NFL table set")
    args = parser.parse_args()
    root = args.root
    try:
        marker = write(root, force=args.force)
    except WouldOverwriteRealTables as error:
        print(f"REFUSING: {error}")
        return 1
    processed = Path(root) / "nfl" / "processed"
    print(f"wrote a SYNTHETIC NFL table set to {processed}")
    print(f"  {marker['games']} games across seasons {marker['seasons']}")
    print(f"  {marker['fixtures']} fixtures in {marker['fixture_season']}")
    print(f"  marker nfl_synthetic.json - production refuses this set unless "
          f"MODELLARIUM_ALLOW_SYNTHETIC_NFL=1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
