"""Pull per-game team-level advanced stats from BoxScoreAdvancedV3, one file"""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import boxscoreadvancedv3

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_player_boxscores import RateLimiter  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GAMES_FINAL_PATH = DATA_DIR / "processed" / "games_final.csv"
OUTPUT_DIR = DATA_DIR / "raw" / "team_advanced_stats"
FAILURES_LOG_PATH = DATA_DIR / "raw" / "team_advanced_stats_failures.log"

REQUESTS_PER_SECOND = 1.0
MAX_WORKERS = 6

MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 30

PROGRESS_EVERY = 100

EXPECTED_ROWS_PER_GAME = 2

REQUIRED_COLUMNS = (
    "gameId",
    "teamId",
    "offensiveRating",
    "defensiveRating",
    "netRating",
    "pace",
    "trueShootingPercentage",
    "possessions",
)

rate_limiter = RateLimiter(REQUESTS_PER_SECOND)
stop_requested = threading.Event()
print_lock = threading.Lock()
failures_lock = threading.Lock()

def load_game_ids() -> list:
    """Every game id in the history table, zero-padded to 10 characters."""
    games = pd.read_csv(GAMES_FINAL_PATH, usecols=["GAME_ID"])
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
            box_score = boxscoreadvancedv3.BoxScoreAdvancedV3(
                game_id=game_id, timeout=REQUEST_TIMEOUT_SECONDS
            )
            teams = box_score.team_stats.get_data_frame()
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            problem = validate_frame(teams)
            if problem:
                last_error = problem
            else:
                temp_path = out_path.with_suffix(".csv.partial")
                teams.to_csv(temp_path, index=False, encoding="utf-8")
                temp_path.replace(out_path)
                return "fetched"

        if attempt < MAX_ATTEMPTS and not stop_requested.is_set():
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    record_failure(game_id, last_error or "unknown error")
    return "failed"

def validate_frame(teams: pd.DataFrame) -> str:
    """Reasons this frame must not be written, or "" if it is fine."""
    if teams.empty:
        return "empty TeamStats frame"
    if len(teams) != EXPECTED_ROWS_PER_GAME:
        return f"expected {EXPECTED_ROWS_PER_GAME} team rows, got {len(teams)}"

    missing = [c for c in REQUIRED_COLUMNS if c not in teams.columns]
    if missing:
        return f"missing expected columns: {missing}"

    return ""

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    game_ids = load_game_ids()
    already_present = sum(1 for gid in game_ids if (OUTPUT_DIR / f"{gid}.csv").exists())
    remaining = len(game_ids) - already_present

    print(f"{len(game_ids):,} games in {GAMES_FINAL_PATH.name}.")
    print(f"{already_present:,} already downloaded, {remaining:,} to fetch via V3.")
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
