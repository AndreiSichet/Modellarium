"""
Reconcile the live injury report against the player history, so an
upcoming game can be scored the same way a played one is.

Serving-time glue, not a batch step - it sits beside live_features.py for
the same reason that file does, rather than in data-pipeline/.

WHAT THIS BRIDGES. The historical pipeline knows players by PLAYER_ID and
knows how important each one is (ROLL10_MIN, their trailing minutes over
their last ten appearances). The injury report knows players by name only,
and says whether they are expected to play. Joining the two is what makes
ABSENT_COUNT / WEIGHTED_ABSENT_MIN computable before tip-off.

THE JOIN IS A STRAIGHT EQUALITY MATCH, verified rather than assumed. Box
scores store "Otto Porter Jr." - First Last Suffix, no commas anywhere -
and fetch_current_injury_report.normalize_player_name() already turns the
report's "Porter Jr., Michael" into exactly that shape. So beyond case and
whitespace folding there is no transformation to do.

MATCHING IS TWO-PASS, and the second pass exists for a real reason.
First pass is name AND team together, never name alone: Craig Porter Jr.,
Michael Porter Jr. and Otto Porter Jr. all exist here, so a surname or even
a full name can collide. Team names on the report match the pipeline's
TEAM_NAME exactly, including the awkward "LA Clippers" (not "Los Angeles
Clippers"), so the team side needs no special casing.

But a player's team in the history is only as current as his last
appearance, and a traded player has not appeared for his new team yet.
Anthony Davis shows up on the 2026-03-14 report for Washington while his
last game in the data is 2026-01-08 for Dallas. First pass misses him, and
silently weighting a 30-minute starter as 0 is exactly the kind of
understatement this feature exists to avoid. So a second pass matches on
name alone, and accepts it **only when the name is unique across the whole
league** - an ambiguous name stays unmatched rather than guessing. Those
rows are flagged MATCH_TYPE "name-only (team changed)". Their ROLL10_MIN
comes from their old team, which is the right approximation: it measures
how big a role the player normally plays, not which jersey he wore.

"MOST RECENT ROW" IS ENOUGH, and that is not an accident. Absence rows in
player_boxscores_with_rolling.csv already carry a forward-filled
ROLL10_MIN, decided when that column was built, so a player's latest row
holds their current known team and current known role whether or not they
played in it. groupby(PLAYER_ID).last() is therefore the whole of "current
player state" - no new logic, just reusing something already correct.

UNMATCHED NAMES ARE LOGGED, NEVER SILENTLY DROPPED. A rookie or a recent
call-up can genuinely be absent from eleven seasons of history; that is
expected, not a bug. Such a player contributes 0 to a weighted sum, the
same as any player whose role is not yet known - but the row is kept with
IS_MATCHED False and the name is reported, so the choice stays visible
rather than becoming an invisible gap. A *long* unmatched list means the
matching logic is wrong, not that the league got younger.
"""

import glob
import os
import sys
import unicodedata
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data-pipeline" / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data-pipeline" / "data" / "raw"

PLAYER_HISTORY_PATH = PROCESSED_DIR / "player_boxscores_with_rolling.csv"
GAMES_GLOB = str(RAW_DIR / "games_*.csv")

ROLLING_COLUMN = "ROLL10_MIN"

# WHERE THE SIDECAR LIVES, and why this is an environment variable rather
# than a constant. The two contexts genuinely need different values and it
# must not be discovered at runtime:
#
#   in compose      http://injury-service:8001   a service name, resolvable
#                                                only inside that network
#   on the host     http://localhost:8001        what a developer running
#                                                the harness or main() sees
#
# Defaulting to localhost means a host run is unchanged and compose sets the
# override, which is exactly how DB_HOST and INFERENCE_URL already work in
# the backend.
INJURY_SERVICE_URL = os.environ.get(
    "INJURY_SERVICE_URL", "http://localhost:8001") + "/injury-report"

# Only what the join needs; the full file is 339,841 rows x 34 columns.
HISTORY_COLUMNS = ["GAME_ID", "GAME_DATE", "TEAM_ID", "PLAYER_ID",
                   "PLAYER_NAME", ROLLING_COLUMN]


def name_key(name: str) -> str:
    """Fold a player name to a comparison key.

    NFKD then drop combining marks, so accented spellings compare equal to
    the report's ASCII ones. This is not cosmetic: the box scores store
    "Nikola Vucevic" with diacritics while the injury report writes it
    plain, and NFKC alone does NOT remove them - it preserves the combining
    characters, so the two never match. Five real players (Vucevic, Jovic,
    Jakucionis, Salaun, Demin) failed to resolve until this was fixed, and
    they looked exactly like "not in eleven seasons of history".
    """
    if not isinstance(name, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", name)
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(without_accents.split()).casefold()


def load_team_lookup() -> dict:
    """TEAM_NAME -> TEAM_ID, from the same raw CSVs TeamSeeder came from."""
    files = glob.glob(GAMES_GLOB)
    if not files:
        raise FileNotFoundError(f"no season files matching {GAMES_GLOB}")

    frames = [pd.read_csv(f, usecols=["TEAM_ID", "TEAM_NAME"]) for f in files]
    teams = pd.concat(frames).drop_duplicates(subset=["TEAM_ID"])

    return {name_key(row.TEAM_NAME): int(row.TEAM_ID) for row in teams.itertuples()}


def load_current_player_state() -> pd.DataFrame:
    """Each player's most recent known team and trailing minutes.

    One row per PLAYER_ID. See the module docstring for why the latest row
    is sufficient even when that row is a game the player missed.
    """
    history = pd.read_csv(
        PLAYER_HISTORY_PATH,
        usecols=HISTORY_COLUMNS,
        dtype={"GAME_ID": str, "TEAM_ID": "Int64", "PLAYER_ID": "Int64"},
        low_memory=False,
    )
    history["GAME_DATE"] = pd.to_datetime(history["GAME_DATE"])
    history[ROLLING_COLUMN] = pd.to_numeric(history[ROLLING_COLUMN], errors="coerce")

    history = history.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"])
    current = history.groupby("PLAYER_ID", as_index=False).last()

    current["NAME_KEY"] = current["PLAYER_NAME"].map(name_key)
    return current[["PLAYER_ID", "PLAYER_NAME", "NAME_KEY", "TEAM_ID",
                    "GAME_DATE", ROLLING_COLUMN]]


def reconcile(report_players: pd.DataFrame,
              current: pd.DataFrame = None,
              team_lookup: dict = None) -> pd.DataFrame:
    """Attach PLAYER_ID and ROLL10_MIN to each injury-report row.

    Expects the `players` frame from fetch_current_injury_report, i.e. real
    rows only - pending teams must never reach here, since they describe
    the absence of information rather than an absent player.
    """
    if current is None:
        current = load_current_player_state()
    if team_lookup is None:
        team_lookup = load_team_lookup()

    rows = report_players.copy()
    rows["TEAM_ID"] = rows["Team"].map(lambda t: team_lookup.get(name_key(t)))
    rows["NAME_KEY"] = rows["PLAYER_NAME_NORMALIZED"].map(name_key)

    # Name AND team together - see the module docstring on shared surnames.
    by_team_name = current.set_index(["TEAM_ID", "NAME_KEY"])
    index = pd.MultiIndex.from_arrays([rows["TEAM_ID"].astype("Int64"), rows["NAME_KEY"]])
    rows["PLAYER_ID"] = by_team_name["PLAYER_ID"].reindex(index).to_numpy()
    rows[ROLLING_COLUMN] = by_team_name[ROLLING_COLUMN].reindex(index).to_numpy()
    rows["MATCH_TYPE"] = pd.Series(
        ["team+name" if pd.notna(v) else None for v in rows["PLAYER_ID"]],
        index=rows.index,
    )

    # Second pass: traded players, whose history still shows the old team.
    # Only names unique league-wide are accepted, so a collision is never
    # resolved by guessing.
    unique_names = current[~current["NAME_KEY"].duplicated(keep=False)]
    by_name = unique_names.set_index("NAME_KEY")

    needs_fallback = rows["PLAYER_ID"].isna()
    if needs_fallback.any():
        keys = rows.loc[needs_fallback, "NAME_KEY"]
        rows.loc[needs_fallback, "PLAYER_ID"] = by_name["PLAYER_ID"].reindex(keys).to_numpy()
        rows.loc[needs_fallback, ROLLING_COLUMN] = (
            by_name[ROLLING_COLUMN].reindex(keys).to_numpy()
        )
        recovered = needs_fallback & rows["PLAYER_ID"].notna()
        rows.loc[recovered, "MATCH_TYPE"] = "name-only (team changed)"

    rows["MATCH_TYPE"] = rows["MATCH_TYPE"].fillna("unmatched")
    rows["IS_MATCHED"] = rows["PLAYER_ID"].notna()
    return rows.drop(columns=["NAME_KEY"])


ABSENT_COUNT_COLUMN = "ABSENT_COUNT"
WEIGHTED_ABSENT_MIN_COLUMN = "WEIGHTED_ABSENT_MIN"

# THIS CACHE NOW GUARDS THE RECONCILIATION, NOT THE FETCH, and the comment
# that used to sit here described the wrong cost entirely.
#
# It was written when this function did both halves: it explained the ~40
# seconds of HTTP probing a missing report used to cost, because
# find_latest_report() walks 48 quarter-hour candidates before giving up.
# That probing now happens in the injury-service sidecar and is cached
# there. Left unchanged, this comment would explain a cost that no longer
# exists in this process - worse than no comment, because the next reader
# would trust it.
#
# What it actually guards, measured rather than assumed:
#
#   load_current_player_state()   2.11s, then 2.09s on a repeat call
#
# An 80.6 MB CSV read and a 1,578-player groupby, with no incidental
# memoisation saving it. Without this cache every prediction request would
# pay that.
#
# So there are two caches in this path, in two processes, over two different
# operations - one per expensive thing, not a duplicated one. The shape that
# WOULD be duplication, two caches over the same fetch, cannot occur here:
# after the split this module performs no fetch at all.
#
# 15 minutes is still short enough that a late scratch is picked up within
# the hour it matters.
CACHE_TTL_SECONDS = 15 * 60

# FAILURES ARE CACHED TOO, at a shorter TTL. The reason changed with the
# split but did not go away: a sidecar that is down, or a league that has
# published nothing, should not be retried on every single prediction - and
# equally should not pin availability to NaN for a quarter of an hour once
# it recovers.
FAILURE_CACHE_TTL_SECONDS = 5 * 60

_cache: dict = {}


def get_team_live_availability(team_id: int,
                               reconciled: pd.DataFrame,
                               pending_team_ids=frozenset()) -> dict:
    """ABSENT_COUNT and WEIGHTED_ABSENT_MIN for one team, right now.

    Same two numbers build_team_availability.py computes retrospectively,
    and the same conventions:

      * An absent player counts toward ABSENT_COUNT whether or not he
        resolved to a PLAYER_ID. The report says he is out; that is a fact
        about the team regardless of whether this code could name him.
      * An unmatched player contributes 0 to the weighted sum, exactly as a
        player with no computable ROLL10_MIN already does in the historical
        pipeline. One unknown rookie must not null out an otherwise-real
        signal for the rest of the team.

    A team whose report is NOT YET SUBMITTED returns NaN for both, not
    zero. Nothing has been filed, so nothing is known - and "no absences
    reported" and "no report" are different claims. Zero here would be the
    same unknown-as-zero bug this project has already fixed three times.
    """
    if team_id in pending_team_ids:
        return {ABSENT_COUNT_COLUMN: float("nan"),
                WEIGHTED_ABSENT_MIN_COLUMN: float("nan")}

    team_rows = reconciled[
        (reconciled["TEAM_ID"] == team_id) & reconciled["IS_ABSENT"]
    ]

    return {
        ABSENT_COUNT_COLUMN: float(len(team_rows)),
        WEIGHTED_ABSENT_MIN_COLUMN: float(
            team_rows[ROLLING_COLUMN].fillna(0.0).sum()
        ),
    }


class NoReportAvailable(Exception):
    """No injury report exists for the requested moment.

    Defined here now rather than imported. The fetch itself moved to the
    injury-service sidecar, which is the only part of this path that needs a
    Java runtime, and this module no longer has that file on its path - it
    talks HTTP instead. The exception stays because its job never depended
    on where the PDF was parsed: raised rather than returning an empty frame
    so that "nothing published" cannot be read as "nobody is injured".
    """


def fetch_report(as_of=None) -> dict:
    """The parsed report from the sidecar.

    The one network call in this module. It returns the PARSED REPORT, not
    computed features - the reconciliation below runs here, against
    player_boxscores_with_rolling.csv, which this image already carries.
    Sending it over the wire instead would mean a second copy of that 80.6 MB
    file and a second place for the reconciliation to drift out of step with
    what the models were trained on.
    """
    import requests

    params = {"as_of": as_of} if as_of is not None else None
    try:
        response = requests.get(INJURY_SERVICE_URL, params=params, timeout=90)
        response.raise_for_status()
    except Exception as error:
        # Named rather than re-raised bare: the sidecar being down and the
        # league not having published are different situations, and a
        # developer reading a log should not have to tell them apart from a
        # stack trace.
        raise NoReportAvailable(
            f"injury-service unreachable at {INJURY_SERVICE_URL} "
            f"({type(error).__name__}: {error}). "
            f"Start it with: docker compose up -d injury-service"
        ) from error

    payload = response.json()
    if not payload.get("report_available"):
        raise NoReportAvailable(
            payload.get("reason") or "no injury report published")
    return payload


def get_live_availability(as_of=None, use_cache: bool = True):
    """Fetch, parse and reconcile the current report.

    Returns (reconciled_rows, pending_team_ids). Raises NoReportAvailable
    when nothing is published, which the caller must handle - see
    live_features.get_live_features().
    """
    import time

    key = ("live" if as_of is None else str(as_of))

    if use_cache and key in _cache:
        cached_at, payload = _cache[key]
        ttl = FAILURE_CACHE_TTL_SECONDS if isinstance(payload, Exception) else CACHE_TTL_SECONDS
        if time.monotonic() - cached_at < ttl:
            if isinstance(payload, Exception):
                raise payload
            return payload

    try:
        payload = fetch_report(as_of=as_of)
        reconciled = reconcile(pd.DataFrame(payload["players"]))

        team_lookup = load_team_lookup()
        pending = pd.DataFrame(payload["pending"])
        pending_teams = pending["Team"] if "Team" in pending.columns else []
        pending_team_ids = frozenset(
            tid for tid in (team_lookup.get(name_key(t)) for t in pending_teams)
            if tid is not None
        )
        payload = (reconciled, pending_team_ids)
    except Exception as error:
        # Remember the failure, so the next request answers immediately
        # instead of repeating the whole 48-candidate search.
        if use_cache:
            _cache[key] = (time.monotonic(), error)
        raise

    if use_cache:
        _cache[key] = (time.monotonic(), payload)
    return payload


def report_unmatched(rows: pd.DataFrame):
    """Print every name that did not resolve. Never silent."""
    unmatched = rows[~rows["IS_MATCHED"]]

    print(f"\n  matched   : {int(rows['IS_MATCHED'].sum()):>3} of {len(rows)}")
    for kind, count in rows["MATCH_TYPE"].value_counts().items():
        print(f"    {kind:<26}{count:>4}")

    if unmatched.empty:
        return

    print("\n  UNMATCHED - each contributes 0 to any weighted sum:")
    for row in unmatched.itertuples():
        team = row.Team if pd.notna(row.TEAM_ID) else f"{row.Team} (TEAM UNRESOLVED)"
        print(f"    {row.PLAYER_NAME_NORMALIZED:<28} {team:<24} "
              f"{getattr(row, '_asdict', lambda: {})().get('Current Status', '')}")


def main():
    """Ad-hoc check of the report and its reconciliation.

    GOES THROUGH THE SIDECAR, exactly as a prediction does. It used to import
    the fetch directly, which was convenient and wrong for the same reason a
    test that builds its own JSON reader is wrong: a tool that exercises a
    different path than production exercises the wrong thing. This needs
    injury-service running, and says so plainly when it is not.
    """
    as_of = None
    if len(sys.argv) > 1:
        as_of = pd.Timestamp(sys.argv[1]).isoformat()
        print(f"Using as_of = {as_of} ET\n")

    print(f"Asking {INJURY_SERVICE_URL}\n")
    try:
        payload = fetch_report(as_of=as_of)
    except NoReportAvailable as error:
        print("NO REPORT CURRENTLY AVAILABLE")
        print(f"  {error}")
        return 0

    players = pd.DataFrame(payload["players"])
    pending = pd.DataFrame(payload["pending"])
    print(f"Report: {payload['timestamp']}")
    print(f"  player rows {len(players)}, pending rows {len(pending)}")

    rows = reconcile(players)
    report_unmatched(rows)

    absent = rows[rows["IS_ABSENT"]]
    print(f"\n  absent players: {len(absent)}")
    print(f"  their combined trailing minutes: "
          f"{absent[ROLLING_COLUMN].fillna(0).sum():.1f}")

    print("\n  sample of matched rows:")
    cols = ["Team", "PLAYER_NAME_NORMALIZED", "Current Status", "IS_ABSENT",
            "PLAYER_ID", ROLLING_COLUMN]
    print(rows[rows["IS_MATCHED"]][cols].head(12).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
