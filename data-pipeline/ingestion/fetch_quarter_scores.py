"""Pull per-game quarter scoring from BoxScoreSummaryV3's LineScore, one file"""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import boxscoresummaryv3

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_player_boxscores import RateLimiter  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GAMES_FINAL_PATH = DATA_DIR / "processed" / "games_final.csv"
OUTPUT_DIR = DATA_DIR / "raw" / "quarter_scores"
FAILURES_LOG_PATH = DATA_DIR / "raw" / "quarter_scores_failures.log"

REQUESTS_PER_SECOND = 1.0
MAX_WORKERS = 6

MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 30

PROGRESS_EVERY = 100

EXPECTED_ROWS_PER_GAME = 2

# ---------------------------------------------------------------------------
# PERMANENTLY UNFETCHABLE GAMES, AUTHORED WITH EVIDENCE.
#
# This list decides THIS SCRIPT'S EXIT CODE and nothing else. It is the second
# of two lists that both concern missing quarter scores, and they answer
# opposite questions - see `build_final_dataset.read_known_fetch_failures()`
# and NOTES_HISTORY.md section 25:
#
#   * read_known_fetch_failures() derives its set from this fetcher's
#     append-only failures LOG, and must TOLERATE a new failure, because
#     `attach_quarter_half()` would otherwise crash a whole retrain the first
#     time any game failed for any reason. Section 25 replaced a literal list
#     with the log for exactly that reason, and rejected a reverse guard
#     there: with an append-only log a stale entry cannot be retracted, so
#     the reverse direction would be a false alarm.
#
#   * THIS list must NOT tolerate a new failure - zero tolerance is the point,
#     so that a partial fetch carries the NBA's tables instead of being staged.
#     It is authored rather than log-derived precisely because a log would
#     auto-absolve any new failure, which is "adding a game to make a run
#     green" with the hand removed. Being authored, it CAN be edited, which is
#     what makes a reverse guard sound here where section 25 ruled it out.
#
# Evidence per entry, gathered 2026-10-22 on the runner's checkout: the three
# games are consecutive GAME_IDs on one date, so this is a contiguous gap in
# the source's feed rather than three independent failures; every OTHER feed
# serves them (`raw/player_boxscores/` and `raw/team_advanced_stats/` both
# hold all three); and the gap already closes arithmetically in the served
# data - games_final 26,426 rows minus 6 equals quarter_half_raw's 26,420.
KNOWN_UNFETCHABLE_QUARTER_SCORES = {
    "0022500259": {
        "date": "2025-11-19",
        "teams": "MIN 120 W vs WAS 109 L",
        "errors": ("AttributeError: 'NoneType' object has no attribute 'get'",),
        "failed_attempts": 36,
    },
    "0022500260": {
        "date": "2025-11-19",
        "teams": "DEN 125 W vs NOP 118 L",
        "errors": ("AttributeError: 'NoneType' object has no attribute 'get'",
                   "JSONDecodeError: Expecting value: line 2 column 1 (char 1)"),
        "failed_attempts": 36,
    },
    "0022500261": {
        "date": "2025-11-19",
        "teams": "OKC 113 W vs SAC 99 L",
        "errors": ("AttributeError: 'NoneType' object has no attribute 'get'",),
        "failed_attempts": 36,
    },
}

EXPECTED_SOURCE_COLUMNS = [
    "gameId", "teamId", "teamCity", "teamName", "teamTricode", "teamSlug",
    "teamWins", "teamLosses",
    "period1Score", "period2Score", "period3Score", "period4Score", "score",
]

COLUMN_RENAMES = {
    "gameId": "GAME_ID",
    "teamId": "TEAM_ID",
    "teamTricode": "TEAM_ABBREVIATION",
    "period1Score": "PTS_Q1",
    "period2Score": "PTS_Q2",
    "period3Score": "PTS_Q3",
    "period4Score": "PTS_Q4",
    "score": "FINAL_PTS",
}

TARGET_COLUMNS = [
    "GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION",
    "PTS_Q1", "PTS_Q2", "PTS_Q3", "PTS_Q4",
    "FINAL_PTS",
]

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

def validate_frame(line_score: pd.DataFrame) -> str:
    """Reasons this frame must not be written, or "" if it is fine."""
    if line_score.empty:
        return "empty LineScore frame"
    if len(line_score) != EXPECTED_ROWS_PER_GAME:
        return f"expected {EXPECTED_ROWS_PER_GAME} team rows, got {len(line_score)}"

    if list(line_score.columns) != EXPECTED_SOURCE_COLUMNS:
        unexpected = [c for c in line_score.columns if c not in EXPECTED_SOURCE_COLUMNS]
        missing = [c for c in EXPECTED_SOURCE_COLUMNS if c not in line_score.columns]
        return (f"LineScore schema changed - missing {missing}, "
                f"unexpected {unexpected}")

    return ""

def normalize(line_score: pd.DataFrame) -> pd.DataFrame:
    """Rename into the project's conventions and keep only what is used."""
    return line_score.rename(columns=COLUMN_RENAMES)[TARGET_COLUMNS]

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
            summary = boxscoresummaryv3.BoxScoreSummaryV3(
                game_id=game_id, timeout=REQUEST_TIMEOUT_SECONDS
            )
            line_score = summary.line_score.get_data_frame()
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            problem = validate_frame(line_score)
            if problem:
                last_error = problem
            else:
                temp_path = out_path.with_suffix(".csv.partial")
                normalize(line_score).to_csv(temp_path, index=False, encoding="utf-8")
                temp_path.replace(out_path)
                return "fetched"

        if attempt < MAX_ATTEMPTS and not stop_requested.is_set():
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    record_failure(game_id, last_error or "unknown error")
    return "failed"

def logged_failure_ids() -> set:
    """Game ids this fetcher has ever recorded a failure for.

    Parsed the same shape as `build_final_dataset.read_known_fetch_failures()`
    but kept local and as STRINGS: the log holds the zero-padded form
    (`"0022500259"`) while that function returns `int`s to match the dataset's
    int64 GAME_ID. Section 25 records that mismatch as a type bug wearing a
    data bug's clothes, so this one does not convert.
    """
    if not FAILURES_LOG_PATH.exists():
        return set()
    seen = set()
    for line in FAILURES_LOG_PATH.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        game_id, _, _reason = text.partition("\t")
        if game_id:
            seen.add(game_id.zfill(10))
    return seen


def check_known_list(game_ids: list) -> list:
    """Problems with KNOWN_UNFETCHABLE_QUARTER_SCORES itself. Empty if sound."""
    problems = []
    corpus = set(game_ids)
    logged = logged_failure_ids()

    for game_id, evidence in sorted(KNOWN_UNFETCHABLE_QUARTER_SCORES.items()):
        # EVIDENCE IS REQUIRED, not decorative. An entry with no recorded
        # failure has nothing behind it, and the point of the list is that a
        # game is on it because the source demonstrably cannot serve it.
        if game_id not in logged:
            problems.append(
                f"{game_id} is listed as permanently unfetchable but appears "
                f"nowhere in {FAILURES_LOG_PATH.name}. A list entry with no "
                f"logged failure has no evidence.")
        for field in ("date", "teams", "errors", "failed_attempts"):
            if not evidence.get(field):
                problems.append(
                    f"{game_id} is listed without {field}. Every entry carries "
                    f"the date, the error and how many attempts failed.")
        # A LISTED GAME THAT NO LONGER EXISTS is also fatal: the list would be
        # silently excusing nothing, and the next real failure would be the
        # first thing anyone noticed.
        if game_id not in corpus:
            problems.append(
                f"{game_id} is listed as permanently unfetchable but is not in "
                f"{GAMES_FINAL_PATH.name} at all. Remove the entry.")
    return problems


def post_run_verdict(failed_ids, present=None) -> tuple:
    """(exit_code, lines) from one run's per-game outcomes.

    Pure so both guards can be planted without fetching 13,000 games.
    `present` answers "does this game now have a file", defaulting to disk.
    """
    if present is None:
        def present(game_id):
            return (OUTPUT_DIR / f"{game_id}.csv").exists()

    listed = set(KNOWN_UNFETCHABLE_QUARTER_SCORES)
    failed = {str(game_id).zfill(10) for game_id in failed_ids}
    lines = []

    # THE REVERSE GUARD, AND IT STAYS LIVE BECAUSE LISTED GAMES ARE RETRIED
    # EVERY RUN. They have no file on disk, so they are attempted like any
    # other game - three requests each - and the day the source starts serving
    # one, this fails rather than quietly keeping a stale excuse.
    recovered = sorted(game_id for game_id in listed
                       if game_id not in failed and present(game_id))
    if recovered:
        lines.append(f"\nFAIL: {len(recovered)} game(s) listed as permanently "
                     f"unfetchable NOW FETCH: {', '.join(recovered)}")
        lines.append("  The source has changed. Remove them from "
                     "KNOWN_UNFETCHABLE_QUARTER_SCORES rather than leaving a "
                     "stale excuse in place.")
        return 1, lines

    unexpected = sorted(failed - listed)
    if unexpected:
        lines.append(f"\nFAIL: {len(unexpected)} game(s) failed and are NOT on "
                     f"the permanently-unfetchable list:")
        for game_id in unexpected[:20]:
            lines.append(f"  {game_id}")
        if len(unexpected) > 20:
            lines.append(f"  ... and {len(unexpected) - 20} more")
        lines.append("  Any failure for a game fetched in this run fails the "
                     "step. With per-league isolation that means the NBA keeps "
                     "yesterday's tables and the game is retried tomorrow - it "
                     "does NOT mean a partial corpus is staged.")
        lines.append("  Do NOT add a game here to make a run green. An entry "
                     "needs the date, the error and the attempt count, and the "
                     "source has to be demonstrably unable to serve it.")
        return 1, lines

    expected = sorted(failed & listed)
    if expected:
        lines.append(f"\n{len(expected)} known-unfetchable game(s) failed as "
                     f"expected: {', '.join(expected)}")
    return 0, lines


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    game_ids = load_game_ids()

    problems = check_known_list(game_ids)
    if problems:
        print(f"FAIL: {len(problems)} problem(s) with "
              f"KNOWN_UNFETCHABLE_QUARTER_SCORES:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"{len(KNOWN_UNFETCHABLE_QUARTER_SCORES)} game(s) listed as "
          f"permanently unfetchable, each with logged evidence.")
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

    failed_ids = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_one, gid): gid for gid in game_ids}

        try:
            for future in as_completed(futures):
                outcome = future.result()
                counts[outcome] += 1
                if outcome == "failed":
                    failed_ids.append(futures[future])
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

    code, lines = post_run_verdict(failed_ids)
    for line in lines:
        print(line)
    return code

if __name__ == "__main__":
    sys.exit(main())
