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

PLAYER_HISTORY_PATH = PROCESSED_DIR / "player_boxscores_with_rolling.csv"
GAMES_GLOB = str(RAW_DIR / "games_*.csv")

ROLLING_COLUMN = "ROLL10_MIN"

INJURY_SERVICE_URL = os.environ.get(
    "INJURY_SERVICE_URL", "http://localhost:8001") + "/injury-report"

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
    files = glob.glob(GAMES_GLOB)
    if not files:
        raise FileNotFoundError(f"no season files matching {GAMES_GLOB}")

    frames = [pd.read_csv(f, usecols=["TEAM_ID", "TEAM_NAME"]) for f in files]
    teams = pd.concat(frames).drop_duplicates(subset=["TEAM_ID"])

    return {name_key(row.TEAM_NAME): int(row.TEAM_ID) for row in teams.itertuples()}

def load_current_player_state() -> pd.DataFrame:
    """Each player's most recent known team and trailing minutes."""
    history = pd.read_csv(
        PLAYER_HISTORY_PATH,
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
