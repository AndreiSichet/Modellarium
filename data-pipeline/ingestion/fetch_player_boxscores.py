"""Pull per-game player box scores from nba_api's BoxScoreTraditionalV3, one"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import boxscoretraditionalv3

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GAMES_FINAL_PATH = DATA_DIR / "processed" / "games_final.csv"
OUTPUT_DIR = DATA_DIR / "raw" / "player_boxscores"
FAILURES_LOG_PATH = DATA_DIR / "raw" / "player_boxscores_failures.log"

REQUESTS_PER_SECOND = 1.0
MAX_WORKERS = 6

MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 30

PROGRESS_EVERY = 100

V3_TO_TARGET = {
    "gameId": "GAME_ID",
    "teamId": "TEAM_ID",
    "teamTricode": "TEAM_ABBREVIATION",
    "teamCity": "TEAM_CITY",
    "personId": "PLAYER_ID",
    "position": "POSITION",
    "comment": "COMMENT",
    "minutes": "MIN",
    "fieldGoalsMade": "FGM",
    "fieldGoalsAttempted": "FGA",
    "fieldGoalsPercentage": "FG_PCT",
    "threePointersMade": "FG3M",
    "threePointersAttempted": "FG3A",
    "threePointersPercentage": "FG3_PCT",
    "freeThrowsMade": "FTM",
    "freeThrowsAttempted": "FTA",
    "freeThrowsPercentage": "FT_PCT",
    "reboundsOffensive": "OREB",
    "reboundsDefensive": "DREB",
    "reboundsTotal": "REB",
    "assists": "AST",
    "steals": "STL",
    "blocks": "BLK",
    "turnovers": "TO",
    "foulsPersonal": "PF",
    "points": "PTS",
    "plusMinusPoints": "PLUS_MINUS",
}

TARGET_COLUMNS = [
    "GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_CITY",
    "PLAYER_ID", "PLAYER_NAME",
    "NICKNAME", "START_POSITION", "POSITION", "COMMENT", "MIN",
    "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TO", "PF", "PTS", "PLUS_MINUS",
]

V3_MISSING_COLUMNS = ["NICKNAME", "START_POSITION"]

STAT_COLUMNS = TARGET_COLUMNS[TARGET_COLUMNS.index("FGM"):]

def has_played(minutes) -> "pd.Series | bool":
    """Did this player actually appear?"""
    if isinstance(minutes, pd.Series):
        return minutes.notna() & (minutes.astype(str).str.strip() != "")
    return minutes is not None and not pd.isna(minutes) and str(minutes).strip() != ""

def normalize_player_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn a V3 PlayerStats frame into the schema kept on disk."""
    required = set(V3_TO_TARGET) | {"firstName", "familyName"}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"V3 response missing expected columns: {sorted(missing)}")

    normalized = raw.rename(columns=V3_TO_TARGET)
    normalized["PLAYER_NAME"] = (
        raw["firstName"].fillna("").astype(str).str.strip()
        + " "
        + raw["familyName"].fillna("").astype(str).str.strip()
    ).str.strip()

    for column in V3_MISSING_COLUMNS:
        normalized[column] = ""

    normalized = normalized[TARGET_COLUMNS].copy()
    absent = ~has_played(normalized["MIN"])
    if absent.any():
        normalized.loc[absent, STAT_COLUMNS] = ""

    return normalized

class RateLimiter:
    """Allows one request per interval, shared across every worker."""

    def __init__(self, requests_per_second: float):
        self._min_interval = 1.0 / requests_per_second
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait < 0:
                wait = 0.0
            self._next_allowed = max(now, self._next_allowed) + self._min_interval

        if wait:
            time.sleep(wait)

rate_limiter = RateLimiter(REQUESTS_PER_SECOND)
stop_requested = threading.Event()
print_lock = threading.Lock()
failures_lock = threading.Lock()

def load_game_ids() -> list:
    """Every game id in the history table, zero-padded to 10 characters."""
    games = pd.read_csv(GAMES_FINAL_PATH, usecols=["GAME_ID"])
    # Unpadded ids return an EMPTY frame rather than an error.
    return sorted(str(game_id).zfill(10) for game_id in games["GAME_ID"].unique())

def record_failure(game_id: str, reason: str):
    with failures_lock:
        with FAILURES_LOG_PATH.open("a", encoding="utf-8") as log:
            log.write(f"{game_id}\t{reason}\n")

def fetch_one(game_id: str) -> str:
    """Fetch and save one game. Returns "skipped", "fetched" or "failed"."""
    out_path = OUTPUT_DIR / f"{game_id}.csv"
    if out_path.exists():
        return "skipped"

    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if stop_requested.is_set():
            return "failed"

        rate_limiter.acquire()

        try:
            box_score = boxscoretraditionalv3.BoxScoreTraditionalV3(
                game_id=game_id, timeout=REQUEST_TIMEOUT_SECONDS
            )
            players = box_score.player_stats.get_data_frame()
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if players.empty:
                last_error = "empty PlayerStats frame"
            else:
                try:
                    normalized = normalize_player_stats(players)
                except KeyError as exc:
                    last_error = str(exc)
                else:
                    temp_path = out_path.with_suffix(".csv.partial")
                    normalized.to_csv(temp_path, index=False, encoding="utf-8")
                    temp_path.replace(out_path)
                    return "fetched"

        if attempt < MAX_ATTEMPTS and not stop_requested.is_set():
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    record_failure(game_id, last_error or "unknown error")
    return "failed"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    game_ids = load_game_ids()
    already_present = sum(1 for gid in game_ids if (OUTPUT_DIR / f"{gid}.csv").exists())
    remaining = len(game_ids) - already_present

    print(f"{len(game_ids)} games in {GAMES_FINAL_PATH.name}.")
    print(f"{already_present} already downloaded, {remaining} to fetch via V3.")
    print(
        f"Rate limit {REQUESTS_PER_SECOND}/s across {MAX_WORKERS} workers "
        f"-> roughly {remaining / REQUESTS_PER_SECOND / 3600:.1f}h if nothing fails.\n"
    )

    counts = {"skipped": 0, "fetched": 0, "failed": 0}
    processed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_one, gid): gid for gid in game_ids}

        try:
            for future in as_completed(futures):
                counts[future.result()] += 1
                processed += 1

                if processed % PROGRESS_EVERY == 0:
                    with print_lock:
                        print(
                            f"{processed:>6,} / {len(game_ids):,}  "
                            f"(fetched {counts['fetched']:,}, "
                            f"skipped {counts['skipped']:,}, "
                            f"failed {counts['failed']:,})"
                        )
        except KeyboardInterrupt:
            print("\nInterrupted - finishing in-flight requests, then stopping.")
            print("Rerun this script to continue from where it left off.")
            stop_requested.set()
            for future in futures:
                future.cancel()
            raise SystemExit(1)

    print(
        f"\nDone. {processed:,} games processed: "
        f"{counts['fetched']:,} fetched, "
        f"{counts['skipped']:,} already present, "
        f"{counts['failed']:,} failed."
    )

    if counts["failed"]:
        print(f"Failed game ids logged to {FAILURES_LOG_PATH}.")
        print("Rerunning this script retries only those, since the rest are on disk.")

if __name__ == "__main__":
    main()
