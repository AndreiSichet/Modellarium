"""Reconcile the live injury report against the player history, so an"""

import glob
import os
import sys
import unicodedata
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data-pipeline" / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data-pipeline" / "data" / "raw"

# RESOLVED THROUGH served_data, NOT FROM THE IMAGE. A4 moved all six served
# tables onto a mounted volume and left the image carrying none of them - and
# this module was the one site that kept pointing inside the image. The effect
# was silent and total: load_current_player_state() raised FileNotFoundError,
# _availability_for_both_teams caught it like any other failure, and the four
# features resolved to NaN. So availability has been UNSERVABLE since A4 for a
# reason that has nothing to do with the offseason, and nothing said so.
#
# The team lookup moved to games_final.csv for the same reason. It used to read
# the eleven raw season files, which are gitignored and absent from the image;
# games_final.csv carries TEAM_ID and TEAM_NAME, is already mounted, and yields
# the identical mapping (verified, 30 of 30).


def _served_processed() -> Path:
    """The processed directory actually being served."""
    from served_data import data_root

    return data_root() / "processed"


def player_history_path() -> Path:
    return _served_processed() / "player_boxscores_with_rolling.csv"


def team_lookup_path() -> Path:
    return _served_processed() / "games_final.csv"

ROLLING_COLUMN = "ROLL10_MIN"

INJURY_SERVICE_BASE = os.environ.get("INJURY_SERVICE_URL", "http://localhost:8001")
INJURY_SERVICE_URL = INJURY_SERVICE_BASE + "/injury-report"
INJURY_SERVICE_STATE_URL = INJURY_SERVICE_BASE + "/state"

# FIVE STATES, BECAUSE THERE WERE EFFECTIVELY TWO. Before 2026-10-07 everything
# that was not a usable report arrived as one NoReportAvailable and resolved to
# NaN, so "the NBA published nothing today", "the sidecar is down" and "the
# NBA's server is unreachable" were a single indistinguishable outcome - and
# /health said `ok` through all of them while the NBA was served on 34 of 38
# features. The first three strings are the sidecar's own wire values; the last
# two are ours.
STATE_USED = "used"
STATE_NONE_PUBLISHED = "none_published"
STATE_SOURCE_FAILED = "source_failed"
STATE_UNREACHABLE = "unreachable"
STATE_NOT_APPLICABLE = "not_applicable"

# Not one of the five: it describes the OBSERVER, not the source. Nothing has
# looked yet, which is different from having looked and found nothing.
STATE_UNKNOWN = "unknown"

STATE_PROBE_TIMEOUT_SECONDS = 1.5

# A SHORT CACHE ON THE PROBE, MEASURED RATHER THAN GUESSED AT. With neither of
# these, /health took 4.0s whenever the sidecar was down: requests retries a
# failed connection, and DNS failure for an absent container costs more than the
# nominal timeout. 4s is not a health endpoint. max_retries=0 removes the
# doubling and this cache means at most one slow call per window, so a sustained
# outage costs one wait rather than one per request.
STATE_PROBE_CACHE_SECONDS = 10.0

# A HARD DEADLINE, because the timeout above does not bound what /health pays.
# Measured 2026-10-07: with the sidecar's container stopped, the first probe took
# 3.97s even at max_retries=0, because the wait is Docker's DNS resolution for an
# absent name and `requests`' timeout only covers connect and read AFTER
# resolution. So the probe runs on a thread and /health waits this long at most;
# the thread keeps going and fills the cache, so the real answer arrives for the
# next caller rather than being lost.
PROBE_DEADLINE_SECONDS = 0.6

_probe_cache: dict = {}
_probe_inflight: dict = {}

_last_observation: dict = {}

def record_observation(state: str, detail: str = None,
                       report_timestamp=None, for_date=None) -> None:
    """Remember what the last availability resolution actually did."""
    import datetime as _dt

    _last_observation.clear()
    _last_observation.update({
        "state": state,
        "detail": detail,
        "report_timestamp": report_timestamp,
        "for_date": str(for_date) if for_date is not None else None,
        "observed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    })

def last_observation() -> dict:
    """What this process last saw, or {} if it has never resolved availability."""
    return dict(_last_observation)

def probe_service_state(timeout: float = STATE_PROBE_TIMEOUT_SECONDS) -> dict:
    """The sidecar's own cached state. Cheap by construction - it never fetches.

    A FAILED PROBE IS THE ANSWER, NOT AN ERROR: if the sidecar cannot be
    reached then availability is unreachable, and /health must say so rather
    than raise. Short timeout because /health must not become slow.
    """
    import time

    cached = _probe_cache.get("state")
    if cached and time.monotonic() - cached[0] < STATE_PROBE_CACHE_SECONDS:
        return dict(cached[1])

    worker = _probe_inflight.get("thread")
    if worker is None or not worker.is_alive():
        import threading

        worker = threading.Thread(target=_probe_now, args=(timeout,), daemon=True)
        _probe_inflight["thread"] = worker
        worker.start()

    worker.join(PROBE_DEADLINE_SECONDS)
    if worker.is_alive():
        # Deliberately NOT cached: the thread is still working, and its real
        # answer should win rather than being hidden behind this placeholder.
        return {"state": STATE_UNREACHABLE,
                "detail": f"sidecar did not answer within "
                          f"{PROBE_DEADLINE_SECONDS}s",
                "observed_at": None, "report_timestamp": None}

    fresh = _probe_cache.get("state")
    if fresh:
        return dict(fresh[1])
    return {"state": STATE_UNKNOWN, "detail": "probe produced no result",
            "observed_at": None, "report_timestamp": None}

def _probe_now(timeout: float) -> dict:
    """The blocking half of probe_service_state, run on a thread."""
    import time

    import requests
    from requests.adapters import HTTPAdapter

    def remember(result: dict) -> dict:
        _probe_cache["state"] = (time.monotonic(), dict(result))
        return result

    try:
        session = requests.Session()
        # max_retries=0: a retry on an unreachable host doubles the wait and
        # tells us nothing new. Unreachable IS the answer.
        session.mount("http://", HTTPAdapter(max_retries=0))
        session.mount("https://", HTTPAdapter(max_retries=0))
        response = session.get(INJURY_SERVICE_STATE_URL, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception as error:  # noqa: BLE001
        return remember({"state": STATE_UNREACHABLE,
                         "detail": f"{type(error).__name__}: {error}",
                         "observed_at": None, "report_timestamp": None})

    mapped = {
        "available": STATE_USED,
        "none_published": STATE_NONE_PUBLISHED,
        "source_failed": STATE_SOURCE_FAILED,
    }.get(payload.get("state"))

    return remember({
        "state": mapped or STATE_UNKNOWN,
        "detail": payload.get("reason"),
        "observed_at": payload.get("observed_at"),
        "report_timestamp": payload.get("report_timestamp"),
        "age_seconds": payload.get("age_seconds"),
    })

def availability_state(for_date=None,
                       timeout: float = STATE_PROBE_TIMEOUT_SECONDS) -> dict:
    """The state to report, preferring what was actually applied.

    The sidecar's cache is the live signal; this process's own last resolution
    is ground truth for `used`, because the sidecar having a report does not by
    itself mean a prediction applied one. The probe wins when it disagrees on
    being reachable, since that is the more recent fact.
    """
    probed = probe_service_state(timeout=timeout)
    mine = last_observation()

    if probed["state"] in (STATE_UNKNOWN,) and mine:
        chosen = {**mine, "source": "inference (sidecar cache empty)"}
    else:
        chosen = {**probed, "source": "injury-service /state"}

    chosen.setdefault("for_date", None)
    if for_date is not None:
        chosen["for_date"] = str(for_date)
    return chosen

HISTORY_COLUMNS = ["GAME_ID", "GAME_DATE", "TEAM_ID", "PLAYER_ID",
                   "PLAYER_NAME", ROLLING_COLUMN]

def name_key(name: str) -> str:
    """Fold a player name to a comparison key."""
    if not isinstance(name, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", name)
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(without_accents.split()).casefold()

def load_team_lookup() -> dict:
    """TEAM_NAME -> TEAM_ID, from the same raw CSVs TeamSeeder came from."""
    path = team_lookup_path()
    if not path.is_file():
        raise FileNotFoundError(f"no served history table at {path}")

    teams = pd.read_csv(path, usecols=["TEAM_ID", "TEAM_NAME"]).drop_duplicates(
        subset=["TEAM_ID"])

    return {name_key(row.TEAM_NAME): int(row.TEAM_ID) for row in teams.itertuples()}

def load_current_player_state() -> pd.DataFrame:
    """Each player's most recent known team and trailing minutes."""
    history = pd.read_csv(
        player_history_path(),
        usecols=HISTORY_COLUMNS,
        dtype={"GAME_ID": str, "TEAM_ID": "Int64", "PLAYER_ID": "Int64"},
        low_memory=False,
    )
    history["GAME_DATE"] = pd.to_datetime(history["GAME_DATE"])
    history[ROLLING_COLUMN] = pd.to_numeric(history[ROLLING_COLUMN], errors="coerce")

    history = history.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"])
    # KNOWN RISK, measured and deliberately kept: .last() takes each
    # column's last non-null value, so it can assemble a row that never
    # existed. Only ROLL10_MIN is exposed. See PROJECT_INSIGHTS.md ch. 13.
    current = history.groupby("PLAYER_ID", as_index=False).last()

    current["NAME_KEY"] = current["PLAYER_NAME"].map(name_key)
    return current[["PLAYER_ID", "PLAYER_NAME", "NAME_KEY", "TEAM_ID",
                    "GAME_DATE", ROLLING_COLUMN]]

def reconcile(report_players: pd.DataFrame,
              current: pd.DataFrame = None,
              team_lookup: dict = None) -> pd.DataFrame:
    """Attach PLAYER_ID and ROLL10_MIN to each injury-report row."""
    if current is None:
        current = load_current_player_state()
    if team_lookup is None:
        team_lookup = load_team_lookup()

    rows = report_players.copy()
    rows["TEAM_ID"] = rows["Team"].map(lambda t: team_lookup.get(name_key(t)))
    rows["NAME_KEY"] = rows["PLAYER_NAME_NORMALIZED"].map(name_key)

    by_team_name = current.set_index(["TEAM_ID", "NAME_KEY"])
    index = pd.MultiIndex.from_arrays([rows["TEAM_ID"].astype("Int64"), rows["NAME_KEY"]])
    rows["PLAYER_ID"] = by_team_name["PLAYER_ID"].reindex(index).to_numpy()
    rows[ROLLING_COLUMN] = by_team_name[ROLLING_COLUMN].reindex(index).to_numpy()
    rows["MATCH_TYPE"] = pd.Series(
        ["team+name" if pd.notna(v) else None for v in rows["PLAYER_ID"]],
        index=rows.index,
    )

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

CACHE_TTL_SECONDS = 15 * 60

FAILURE_CACHE_TTL_SECONDS = 5 * 60

_cache: dict = {}

def get_team_live_availability(team_id: int,
                               reconciled: pd.DataFrame,
                               pending_team_ids=frozenset()) -> dict:
    """ABSENT_COUNT and WEIGHTED_ABSENT_MIN for one team, right now."""
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
    """No injury report exists for the requested moment."""

def fetch_report(as_of=None) -> dict:
    """The parsed report from the sidecar."""
    import requests

    params = {"as_of": as_of} if as_of is not None else None
    try:
        response = requests.get(INJURY_SERVICE_URL, params=params, timeout=90)
        response.raise_for_status()
    except Exception as error:
        record_observation(STATE_UNREACHABLE,
                           detail=f"{type(error).__name__}: {error}",
                           for_date=as_of)
        raise NoReportAvailable(
            f"injury-service unreachable at {INJURY_SERVICE_URL} "
            f"({type(error).__name__}: {error}). "
            f"Start it with: docker compose up -d injury-service"
        ) from error

    payload = response.json()
    if not payload.get("report_available"):
        # report_state tells none_published from source_failed; a sidecar too
        # old to send it degrades to none_published, which is what the single
        # outcome used to mean.
        record_observation(payload.get("report_state") or STATE_NONE_PUBLISHED,
                           detail=payload.get("reason"), for_date=as_of)
        raise NoReportAvailable(
            payload.get("reason") or "no injury report published")

    record_observation(STATE_USED, detail=None,
                       report_timestamp=payload.get("timestamp"),
                       for_date=as_of)
    return payload

def get_live_availability(as_of=None, use_cache: bool = True):
    """Fetch, parse and reconcile the current report."""
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
    """Ad-hoc check of the report and its reconciliation."""
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
