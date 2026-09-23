"""Travel, time-zone, road-trip and schedule-density features per team-game.

What REST_DAYS already carries: days since the previous game, so a back-to-back
IS REST_DAYS == 1. The model already has the largest fatigue effect, and any
feature amounting to "flag back-to-backs" is a re-encoding.

What it cannot carry: REST_DAYS == 1 is identical whether it is the second game
of a road trip or the fourth game in six nights. Density and recency are
different quantities and only recency is encoded.
"""

import sys
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(DATA_DIR))
from arenas import ARENAS, coordinates, timezone_name, validate  # noqa: E402

GAMES_FINAL_PATH = DATA_DIR / "processed" / "games_final.csv"
OUTPUT_PATH = DATA_DIR / "processed" / "travel_features.csv"

MERGE_KEYS = ["GAME_ID", "TEAM_ID"]
EXPECTED_UNIVERSE_ROWS = 26_398
EXPECTED_TEAMS_PER_GAME = 2

DENSITY_WINDOW_DAYS = 7
EARTH_RADIUS_KM = 6371.0088

# Miami to Portland is about 4,350 km, the longest regular trip in the league.
# Anything past this is a season boundary leaking in, not a real journey.
MAX_PLAUSIBLE_TRAVEL_KM = 5000.0

FEATURES = ["TRAVEL_KM", "TZ_SHIFT", "ROAD_TRIP_GAME", "GAMES_LAST_7"]


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    d_lat, d_lon = lat2 - lat1, lon2 - lon1
    a = sin(d_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(d_lon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(a))


def utc_offset_hours(team_id: int, game_date) -> float:
    """Offset at that venue on that date, so Arizona's missing DST is handled."""
    moment = datetime(game_date.year, game_date.month, game_date.day, 19)
    zone = ZoneInfo(timezone_name(team_id))
    return moment.replace(tzinfo=zone).utcoffset().total_seconds() / 3600.0


def load_universe() -> pd.DataFrame:
    """Every team-game with its venue. The frame the sequences are built on."""
    games = pd.read_csv(GAMES_FINAL_PATH,
                        usecols=MERGE_KEYS + ["GAME_DATE", "SEASON", "IS_HOME"])
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])

    if len(games) != EXPECTED_UNIVERSE_ROWS:
        raise SystemExit(f"expected {EXPECTED_UNIVERSE_ROWS:,} team-games, "
                         f"got {len(games):,}")

    duplicates = int(games.duplicated(subset=MERGE_KEYS).sum())
    if duplicates:
        raise SystemExit(f"{duplicates:,} duplicate (GAME_ID, TEAM_ID) rows")

    # Per-game gaps must affect both teams, or a fixture is half-present and a
    # "previous game" lookup would silently reach past it for one side only.
    per_game = games["GAME_ID"].value_counts()
    lopsided = per_game[per_game != EXPECTED_TEAMS_PER_GAME]
    if len(lopsided):
        raise SystemExit(f"{len(lopsided)} game(s) do not have exactly "
                         f"{EXPECTED_TEAMS_PER_GAME} team rows:\n{lopsided.head()}")

    # The venue is the home team's arena.
    hosts = games.loc[games["IS_HOME"], ["GAME_ID", "TEAM_ID"]]
    hosts = hosts.rename(columns={"TEAM_ID": "VENUE_TEAM_ID"})
    if len(hosts) != games["GAME_ID"].nunique():
        raise SystemExit("some games have no home team, so they have no venue")

    universe = games.merge(hosts, on="GAME_ID", how="left", validate="many_to_one")
    unknown = set(universe["VENUE_TEAM_ID"]) - set(ARENAS)
    if unknown:
        raise SystemExit(f"venues with no arena entry: {unknown}")

    print(f"Universe: {len(universe):,} team-games, "
          f"{universe['GAME_ID'].nunique():,} games, "
          f"{universe['TEAM_ID'].nunique()} teams, "
          f"seasons {universe['SEASON'].min()}-{universe['SEASON'].max()}.")
    return universe


def build_features(universe: pd.DataFrame) -> pd.DataFrame:
    """The four features, per team-season, strictly backward-looking.

    Each team-season is independent. The game before a season's first is months
    earlier at a different venue, so carrying it across would produce a
    3,000 km offseason "trip" that looks like signal.
    """
    df = universe.sort_values(["TEAM_ID", "SEASON", "GAME_DATE", "GAME_ID"])
    df = df.reset_index(drop=True)

    grouped = df.groupby(["TEAM_ID", "SEASON"], sort=False)

    # NaN at each season's first game, not zero: an unknown is not a zero, and
    # that conflation has already produced a real bug in absence weighting.
    previous_venue = grouped["VENUE_TEAM_ID"].shift(1)

    coords = {tid: coordinates(tid) for tid in ARENAS}
    travel, shift = [], []
    for prev, here, date in zip(previous_venue, df["VENUE_TEAM_ID"], df["GAME_DATE"]):
        if pd.isna(prev):
            travel.append(np.nan)
            shift.append(np.nan)
            continue
        prev = int(prev)
        travel.append(haversine_km(*coords[prev], *coords[int(here)]))
        # Positive = eastward. A model handed abs() cannot learn that eastward
        # travel is generally harder than westward.
        shift.append(utc_offset_hours(int(here), date) - utc_offset_hours(prev, date))

    df["TRAVEL_KM"] = travel
    df["TZ_SHIFT"] = shift
    df["ROAD_TRIP_GAME"] = road_trip_length(df)
    df["GAMES_LAST_7"] = games_in_window(df)

    return df


def road_trip_length(df: pd.DataFrame) -> np.ndarray:
    """Consecutive away games including this one; 0 at home. Resets each season."""
    lengths = np.zeros(len(df), dtype=float)
    run = 0
    previous_key = None
    for i, (key, is_home) in enumerate(zip(
            zip(df["TEAM_ID"], df["SEASON"]), df["IS_HOME"])):
        if key != previous_key:
            run = 0
            previous_key = key
        run = 0 if is_home else run + 1
        lengths[i] = run
    return lengths


def games_in_window(df: pd.DataFrame) -> np.ndarray:
    """Games by this team in the preceding 7 days, EXCLUDING this one.

    The density term REST_DAYS cannot express. Counted within the season only.
    """
    counts = np.zeros(len(df), dtype=float)
    for _, block in df.groupby(["TEAM_ID", "SEASON"], sort=False):
        dates = block["GAME_DATE"].to_numpy()
        positions = block.index.to_numpy()
        for offset, date in enumerate(dates):
            window_start = date - np.timedelta64(DENSITY_WINDOW_DAYS, "D")
            earlier = dates[:offset]
            counts[positions[offset]] = int(
                ((earlier > window_start) & (earlier < date)).sum())
    return counts


def assert_backward_looking(universe: pd.DataFrame, built: pd.DataFrame) -> None:
    """No feature may read the target team's later games.

    Negative-tested AND positive-controlled on all four. The control has caught
    a vacuous guard twice in this project - without it, a feature that reads
    nothing at all passes the leak test perfectly.
    """
    print("\nBACKWARD-LOOKING GUARD")

    ordered = built.sort_values(["TEAM_ID", "SEASON", "GAME_DATE", "GAME_ID"])
    candidates = ordered[ordered["GAMES_LAST_7"] >= 2].index.to_list()

    def probe(target, frame_filter):
        """Recompute one team-season with some rows removed."""
        row = built.loc[target]
        slice_mask = ((universe["TEAM_ID"] == row["TEAM_ID"])
                      & (universe["SEASON"] == row["SEASON"]))
        block = universe[slice_mask & frame_filter(row)]
        out = build_features(block)
        hit = out["GAME_ID"] == row["GAME_ID"]
        return row, out.loc[hit, FEATURES].iloc[0]

    # Negative test, on a spread of rows rather than one: delete every LATER
    # game for that team-season. A strictly backward-looking feature cannot
    # notice, and a wider sample makes a lucky probe less likely.
    sampled = candidates[::max(1, len(candidates) // 40)][:40]
    leaked = []
    for target in sampled:
        row, after = probe(target,
                           lambda r: universe["GAME_DATE"] <= r["GAME_DATE"])
        leaked += [f for f in FEATURES if not _same_value(row[f], after[f])]
    print(f"  negative test on {len(sampled)} rows, every later game of each "
          f"team-season removed\n    -> "
          f"{'no feature moved' if not leaked else f'MOVED: {sorted(set(leaked))}'}")
    if leaked:
        raise SystemExit(f"{sorted(set(leaked))} read games after the one "
                         "being predicted")

    # Positive control, PER FEATURE. Demanding one row where all four respond
    # fails on correct behaviour: ROAD_TRIP_GAME is 0 at a home game whatever
    # preceded it, and TZ_SHIFT does not move when the two prior venues share a
    # zone. Each feature is therefore controlled where its control can mean
    # something, and every one must be shown to read the past somewhere.
    print("  positive control - remove the immediately preceding game:")
    responded = {}
    for target in candidates:
        if len(responded) == len(FEATURES):
            break
        row = built.loc[target]
        same = ((universe["TEAM_ID"] == row["TEAM_ID"])
                & (universe["SEASON"] == row["SEASON"]))
        earlier = universe[same & (universe["GAME_DATE"] < row["GAME_DATE"])]
        if earlier.empty:
            continue
        previous_game = earlier.sort_values("GAME_DATE").iloc[-1]["GAME_ID"]
        _row, after = probe(target,
                            lambda r, g=previous_game: universe["GAME_ID"] != g)
        for feature in FEATURES:
            if feature not in responded and not _same_value(row[feature],
                                                            after[feature]):
                responded[feature] = (int(row["TEAM_ID"]), int(row["SEASON"]),
                                      row["GAME_DATE"].date())

    for feature in FEATURES:
        where = responded.get(feature)
        print(f"    {feature:<16}"
              + (f"moves at team {where[0]}, {where[2]}" if where
                 else "NEVER MOVED"))
    inert = [f for f in FEATURES if f not in responded]
    if inert:
        raise SystemExit(
            f"{inert} never moved when the previous game was removed, anywhere "
            "in the corpus, so the negative test proved nothing for them - a "
            "feature reading no data at all would pass it identically."
        )
    print("  all four are backward-looking AND demonstrably read the past.")


def _same_value(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    return bool(abs(float(a) - float(b)) < 1e-9)


def assert_season_boundaries(built: pd.DataFrame) -> None:
    print("\nSEASON BOUNDARIES")
    firsts = built.sort_values(["TEAM_ID", "SEASON", "GAME_DATE", "GAME_ID"])
    firsts = firsts.groupby(["TEAM_ID", "SEASON"]).head(1)

    bad = firsts[firsts["TRAVEL_KM"].notna() | firsts["TZ_SHIFT"].notna()]
    print(f"  {len(firsts):,} team-season openers, "
          f"{len(bad)} with a travel value (must be 0)")
    if len(bad):
        raise SystemExit("a season opener has travel computed across the offseason")

    longest = built["TRAVEL_KM"].max()
    print(f"  longest single trip: {longest:,.0f} km "
          f"(ceiling {MAX_PLAUSIBLE_TRAVEL_KM:,.0f} km)")
    if longest > MAX_PLAUSIBLE_TRAVEL_KM:
        raise SystemExit(f"a {longest:,.0f} km trip is not a real NBA journey")

    row = built.loc[built["TRAVEL_KM"].idxmax()]
    print(f"    {ARENAS[int(row['VENUE_TEAM_ID'])][1]} on {row['GAME_DATE'].date()}")
    print(f"  openers also carry GAMES_LAST_7 = "
          f"{sorted(firsts['GAMES_LAST_7'].unique())} (must be [0.0])")
    if list(firsts["GAMES_LAST_7"].unique()) != [0.0]:
        raise SystemExit("a season opener counts games from the previous season")


def report_correlations(built: pd.DataFrame) -> None:
    """Is the density term actually independent of REST_DAYS?

    This decides how to read a null. If GAMES_LAST_7 correlates with REST_DAYS
    above ~0.8 the hypothesis was wrong at the premise and a null says nothing
    new; if it correlates weakly, a null is informative - the model was handed a
    genuinely new quantity and could not use it.
    """
    print("\nCORRELATION WITH REST_DAYS - read this before any score")

    rest = pd.read_csv(GAMES_FINAL_PATH, usecols=MERGE_KEYS + ["REST_DAYS"])
    merged = built.merge(rest, on=MERGE_KEYS, how="left", validate="one_to_one")

    print(f"  {'FEATURE':<18}{'CORR vs REST_DAYS':>19}{'READ':>28}")
    print("  " + "-" * 63)
    for feature in FEATURES:
        r = merged[feature].corr(merged["REST_DAYS"])
        if abs(r) >= 0.8:
            read = "re-encoding of rest"
        elif abs(r) >= 0.5:
            read = "substantially overlapping"
        else:
            read = "largely independent"
        print(f"  {feature:<18}{r:>+19.4f}{read:>28}")

    print("\n  Between the new features themselves:")
    print(merged[FEATURES].corr().round(3).to_string(
        float_format=lambda v: f"{v:+.3f}"))


def main():
    validate()
    universe = load_universe()
    built = build_features(universe)

    assert_backward_looking(universe, built)
    assert_season_boundaries(built)
    report_correlations(built)

    output = built[MERGE_KEYS + FEATURES].sort_values(MERGE_KEYS)
    output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\nWrote {OUTPUT_PATH}  ({len(output):,} team-games)")
    print(built[FEATURES].describe().to_string())


if __name__ == "__main__":
    main()
