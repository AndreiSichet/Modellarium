"""Pull per-shot location data from ShotChartDetail, one file per game.

Scoped to the five seasons the subset experiment needs (2021-2025), not all
eleven: the full pull is justified by that experiment's result or not at all.
"""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import shotchartdetail

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_player_boxscores import RateLimiter  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GAMES_FINAL_PATH = DATA_DIR / "processed" / "games_final.csv"
OUTPUT_DIR = DATA_DIR / "raw" / "shot_charts"
FAILURES_LOG_PATH = DATA_DIR / "raw" / "shot_charts_failures.log"

SUBSET_SEASONS = [2021, 2022, 2023, 2024, 2025]

REQUESTS_PER_SECOND = 1.0
MAX_WORKERS = 6

MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 30

PROGRESS_EVERY = 100

# The default is 'PTS', which returns MADE shots only - a ~45% shortfall that
# reads as partial data rather than as a wrong parameter. Found by the probe.
CONTEXT_MEASURE = "FGA"

TARGET_COLUMNS = [
    "GAME_ID", "TEAM_ID", "PLAYER_ID", "PERIOD",
    "SHOT_TYPE", "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE",
    "SHOT_DISTANCE", "LOC_X", "LOC_Y", "SHOT_MADE_FLAG",
]

rate_limiter = RateLimiter(REQUESTS_PER_SECOND)
stop_requested = threading.Event()
print_lock = threading.Lock()
failures_lock = threading.Lock()


def load_subset_games() -> dict:
    """{padded game id: total FGA} for the subset seasons.

    The FGA side is what makes an empty response distinguishable from a real
    game with no shots: rows-returned alone proves nothing, and rows-but-fewer
    is worse than missing because it yields a feature that looks fine and is
    wrong.
    """
    games = pd.read_csv(GAMES_FINAL_PATH, usecols=["GAME_ID", "SEASON", "FGA"])
    subset = games[games["SEASON"].isin(SUBSET_SEASONS)]
    totals = subset.groupby("GAME_ID")["FGA"].sum()
    return {str(int(gid)).zfill(10): int(fga) for gid, fga in totals.items()}


def record_failure(game_id: str, reason: str):
    with failures_lock:
        with FAILURES_LOG_PATH.open("a", encoding="utf-8") as log:
            log.write(f"{game_id}\t{reason}\n")


def validate_frame(shots: pd.DataFrame, game_id: str, expected_fga: int) -> str:
    """Reasons this frame must not be written, or "" if it is fine."""
    if shots.empty:
        return "empty shot frame"

    missing = [c for c in TARGET_COLUMNS if c not in shots.columns]
    if missing:
        return f"schema changed - missing {missing}"

    # Strict by design. The probe got exact agreement on 5 of 5 games, so a
    # mismatch is a real signal rather than an expected tolerance. Relaxing
    # this needs a measured reason, not a convenient one.
    if len(shots) != expected_fga:
        return (f"FGA mismatch - {len(shots)} shot rows against "
                f"{expected_fga} known FGA (difference {len(shots) - expected_fga:+d})")

    if shots["SHOT_MADE_FLAG"].isna().any():
        return "SHOT_MADE_FLAG has nulls"
    if shots["SHOT_ZONE_BASIC"].isna().any():
        return "SHOT_ZONE_BASIC has nulls"

    return ""


def fetch_one(game_id: str, expected_fga: int) -> str:
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
            endpoint = shotchartdetail.ShotChartDetail(
                team_id=0,
                player_id=0,
                game_id_nullable=game_id,
                context_measure_simple=CONTEXT_MEASURE,
                season_type_all_star="Regular Season",
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            shots = endpoint.get_data_frames()[0]
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            problem = validate_frame(shots, game_id, expected_fga)
            if problem:
                last_error = problem
            else:
                temp_path = out_path.with_suffix(".csv.partial")
                shots[TARGET_COLUMNS].to_csv(temp_path, index=False, encoding="utf-8")
                temp_path.replace(out_path)
                return "fetched"

        if attempt < MAX_ATTEMPTS and not stop_requested.is_set():
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    record_failure(game_id, last_error or "unknown error")
    return "failed"


def main(limit: int = None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    expected = load_subset_games()
    game_ids = sorted(expected)
    if limit:
        game_ids = game_ids[:limit]
        print(f"VERIFICATION RUN - first {limit} games only.\n")

    already = sum(1 for gid in game_ids if (OUTPUT_DIR / f"{gid}.csv").exists())
    remaining = len(game_ids) - already

    print(f"{len(game_ids):,} games across seasons {SUBSET_SEASONS}.")
    print(f"{already:,} already downloaded, {remaining:,} to fetch.")
    print(f"Rate limit {REQUESTS_PER_SECOND}/s across {MAX_WORKERS} workers "
          f"-> roughly {remaining / REQUESTS_PER_SECOND / 3600:.1f}h if nothing fails.")
    print(f"Every game is cross-checked against its known FGA; a mismatch is a "
          f"failure, not a warning.\n")

    counts = {"skipped": 0, "fetched": 0, "failed": 0}
    processed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_one, gid, expected[gid]): gid
                   for gid in game_ids}

        try:
            for future in as_completed(futures):
                counts[future.result()] += 1
                processed += 1

                if processed % PROGRESS_EVERY == 0:
                    with print_lock:
                        print(f"{processed:>6,} / {len(game_ids):,}  "
                              f"(fetched {counts['fetched']:,}, "
                              f"skipped {counts['skipped']:,}, "
                              f"failed {counts['failed']:,})", flush=True)
        except KeyboardInterrupt:
            print("\nInterrupted - finishing in-flight requests, then stopping.")
            print("Rerun this script to continue from where it left off.")
            stop_requested.set()
            for future in futures:
                future.cancel()
            raise SystemExit(1)

    print(f"\nDone. {processed:,} games processed: "
          f"{counts['fetched']:,} fetched, "
          f"{counts['skipped']:,} already present, "
          f"{counts['failed']:,} failed.")

    if counts["failed"]:
        print(f"Failed game ids logged to {FAILURES_LOG_PATH}.")
        print("Rerunning retries only those, since the rest are on disk.")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit)
