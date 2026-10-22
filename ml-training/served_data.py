"""Where the served game history lives.

ONE PLACE RESOLVES IT, for all three leagues and all five live-feature
modules. Before this, each module computed its own path from `__file__`, which
was fine while the data was baked into the image at a fixed relative location
and is not fine once it arrives on a mounted volume.

THERE IS NO FALLBACK, AND THAT IS THE POINT. `require_data_root()` raises if
`DATA_DIR` is unset or does not look like a data root. A fallback to the copy
inside the image would mean the service could be serving either of two
snapshots with nothing saying which - the same shape as the two Postgres
instances on port 5432, where a host-run backend silently reached the wrong
one and cost real debugging time. One hazard of that kind is enough.

`data_root()` is the lenient one, and exists only so the dozens of training,
verification and study scripts in this repo keep working unchanged: with
DATA_DIR unset it returns the repo's own `data-pipeline/data`. The SERVICE
calls `require_data_root()` at boot, so the strict answer is the one that
reaches serving.

THE SNAPSHOT LAYOUT MIRRORS THE REPO'S, deliberately:

    <root>/processed/games_final.csv
    <root>/processed/player_boxscores_with_rolling.csv
    <root>/processed/quarter_half_raw.csv
    <root>/wnba/processed/wnba_games_final.csv
    <root>/gleague/processed/gleague_games_final.csv
    <root>/gleague/processed/gleague_showcase_games.csv
    <root>/nfl/processed/nfl_games_final.csv
    <root>/nfl/processed/nfl_fixtures.csv
    <root>/nfl/processed/nfl_franchise_identity.csv

so one environment variable can point at either the repo or a snapshot with no
per-file configuration, and CI can supply the committed copy without any other
change.

PHASE 4 MOVED THE NFL TABLES TO MAKE THAT TRUE FOR FOUR LEAGUES. Phase 1 wrote
them to `data-pipeline/nfl/data/processed/`, outside this root, so the mirror
promise above held for three leagues and not the fourth. They now live at
`data-pipeline/data/nfl/processed/` like everyone else's, which is why no NFL
special case appears anywhere below.

THE NFL'S COMMITTED COPY IS SYNTHETIC, AND ONLY IN CI. The real tables are
CC BY-SA 4.0 and are never committed, so `make_synthetic_nfl_tables.py`
generates a schema-identical set of invented scores for CI to boot against,
marked with `nfl/processed/nfl_synthetic.json`. Production refuses that marker
unless MODELLARIUM_ALLOW_SYNTHETIC_NFL=1 - see
`ml-training/nfl/live_nfl_features.refuse_synthetic`.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPO_DATA_ROOT = PROJECT / "data-pipeline" / "data"

DATA_DIR_ENV = "DATA_DIR"
SNAPSHOT_FILE = "SNAPSHOT.json"

# What a data root must contain to be one. Checked at boot rather than at
# first use: a missing file should refuse the service, not surface as a
# 500 on whichever league happens to be asked for first.
REQUIRED_FILES = [
    Path("processed") / "games_final.csv",
    Path("processed") / "player_boxscores_with_rolling.csv",
    Path("processed") / "quarter_half_raw.csv",
    Path("wnba") / "processed" / "wnba_games_final.csv",
    Path("gleague") / "processed" / "gleague_games_final.csv",
    Path("gleague") / "processed" / "gleague_showcase_games.csv",
    # The NFL, from phase 4. THREE tables rather than one, and each is read:
    # the games table is the history the features and the Elo replay are built
    # from, the fixtures table is both /schedule/nfl and the dependency rule
    # that decides what is predictable, and the identity table is how a
    # franchise id becomes a name. Nothing else the NFL pipeline writes is
    # copied - nfl_model_dataset.csv in particular is deliberately absent,
    # because its stored Elo was fitted under phase 2's parameters and serving
    # replays under the production ones.
    Path("nfl") / "processed" / "nfl_games_final.csv",
    Path("nfl") / "processed" / "nfl_fixtures.csv",
    Path("nfl") / "processed" / "nfl_franchise_identity.csv",
]


# TABLES A SNAPSHOT MAY CARRY AND NEED NOT, kept separate from REQUIRED_FILES
# rather than mixed in with a flag, because the two have opposite failure
# modes: an absent required table refuses the boot, an absent optional one is
# a fact to report.
#
# WHY THIS EXISTS AT ALL: the NFL's quarantine table is written by a refresh
# that postdates every snapshot currently retained. Adding it to
# REQUIRED_FILES would make the seven snapshots already on the volume
# unservable - including the one being served right now - so a rollback would
# have nowhere to roll back to. That is the same ordering trap the NFL's own
# LEAGUES entry has a comment about, arriving from the other direction.
#
# AN ABSENT OPTIONAL TABLE IS NOT AN EMPTY ONE, and `table_hashes` says
# "absent" rather than hashing nothing. A snapshot built before the quarantine
# existed has an UNKNOWN quarantine state, not a clean one - reporting zero
# quarantined games for it would be inventing a fact about data nobody
# checked.
OPTIONAL_FILES = [
    Path("nfl") / "processed" / "nfl_quarantine.csv",
]


POINTER_FILE = "current"
SNAPSHOTS_DIR = "snapshots"

# WHEN THE REFRESH LAST REACHED A VERDICT, kept on the volume beside the
# pointer rather than inside a snapshot.
#
# WHY IT CANNOT LIVE IN A SNAPSHOT: the run that most needs recording is the
# one that writes no snapshot at all. `nothing-new` is green and discards its
# staging directory, so a snapshot-local record would be written on exactly
# the days something moved and absent on the quiet ones - which inverts the
# question. "Has the job run?" has to be answerable when the job changed
# nothing, because a job that silently stopped running looks identical to a
# league on a bye week.
LAST_REFRESH_FILE = "last_refresh.json"

# TWO THRESHOLDS, NOT ONE, AND BOTH ARE MEASURED RATHER THAN CHOSEN.
#
# The cron fires at 07:00 daily, but it only QUEUES - the runner is on a
# machine that sleeps, so the job runs shortly after the host next wakes.
# Measured 17-21 October 2026, the 07:00 slot actually executed at 13:38,
# 12:28, 16:22 and 19:10 (CLAUDE.md s6). So consecutive runs legitimately sit
# up to about 36 hours apart, and a single 26-hour alarm would have cried wolf
# on an ordinary week - the same failure mode as a gate going red on the clock
# rather than the code, which this project has recorded four times.
#
# So the two facts are separated instead of merged:
#
#   LATE      past one cron interval plus a couple of hours. The 07:00 slot
#             was missed; on this runner that usually means a sleeping laptop
#             and is NOT actionable on its own.
#   OVERDUE   past the worst legitimate gap with margin. A whole cron slot
#             produced nothing, which no amount of host sleep explains - the
#             job is wedged, the runner is offline, or the schedule is
#             disabled.
#
# Merging them would force a choice between crying wolf and hiding a real
# miss. `hours_since` is reported alongside, because the number is the fact
# and the state is the reading of it.
REFRESH_LATE_HOURS = 26
REFRESH_OVERDUE_HOURS = 40

# Which league each served table belongs to, so /health can report them grouped
# the way every other per-league field already is.
TABLE_LEAGUE = {
    "processed/games_final.csv": "nba",
    "processed/player_boxscores_with_rolling.csv": "nba",
    "processed/quarter_half_raw.csv": "nba",
    "wnba/processed/wnba_games_final.csv": "wnba",
    "gleague/processed/gleague_games_final.csv": "gleague",
    "gleague/processed/gleague_showcase_games.csv": "gleague",
    "nfl/processed/nfl_games_final.csv": "nfl",
    "nfl/processed/nfl_fixtures.csv": "nfl",
    "nfl/processed/nfl_franchise_identity.csv": "nfl",
    "nfl/processed/nfl_quarantine.csv": "nfl",
}

HASH_CHUNK_BYTES = 1 << 20


def table_hashes(root: Path = None) -> dict:
    """sha256 per served table, per league, from the BYTES ON DISK.

    DELIBERATELY NOT READ FROM SNAPSHOT.json OR ANY MANIFEST THE REFRESH WROTE.
    A snapshot mutated in place would then report the hash of what it used to
    be, which is the exact failure this exists to show - the same reasoning that
    made the directory NAME the snapshot's identity rather than the stamp inside
    it. Hashing what was actually read is the only version of this check that
    cannot be fooled by the thing it is checking.

    A name alone was never enough: the snapshot id is a timestamp, and a
    timestamp says the identity changed, not whether the contents are right.
    """
    import hashlib

    root = root or data_root()
    hashes = {}
    for relative in REQUIRED_FILES + OPTIONAL_FILES:
        key = relative.as_posix()
        league = TABLE_LEAGUE.get(key, "other")
        if relative in OPTIONAL_FILES and not (root / relative).is_file():
            # NAMED, not omitted and not hashed as empty. See OPTIONAL_FILES:
            # a snapshot from before this table existed has an unknown state
            # here, and leaving the key out would read as "nothing to say".
            hashes.setdefault(league, {})[relative.name] = "absent"
            continue
        digest = hashlib.sha256()
        try:
            with open(root / relative, "rb") as handle:
                for block in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
                    digest.update(block)
            value = digest.hexdigest()
        except OSError as error:
            value = f"unreadable: {type(error).__name__}"
        hashes.setdefault(league, {})[relative.name] = value
    return hashes


def resolve_root(configured: Path) -> Path:
    """Follow the `current` pointer if there is one, else take the directory.

    TWO SHAPES, ONE RULE, AND THE RULE IS WRITTEN HERE RATHER THAN GUESSED AT
    EACH CALL SITE:

      a volume   <root>/current names a directory under <root>/snapshots/
      a plain root   no `current` file, so <root> IS the data root

    The second is what the repo and CI use. The first is what the refresh job
    maintains, and the pointer is why: a swap rewrites ONE SMALL TEXT FILE
    rather than moving 80 MB of tables or relying on a Windows directory
    junction surviving a Docker Desktop bind mount. Rollback is the same
    write in reverse, which is what makes check 6 of this item cheap enough
    to actually test.
    """
    pointer = configured / POINTER_FILE
    if not pointer.is_file():
        return configured

    name = pointer.read_text(encoding="utf-8").strip()
    if not name:
        raise RuntimeError(
            f"{pointer} is empty, so no snapshot is named. A swap writes this "
            f"file last; an empty one means a swap was interrupted.")

    snapshot = configured / SNAPSHOTS_DIR / name
    if not snapshot.is_dir():
        raise RuntimeError(
            f"{pointer} names {name!r}, but {snapshot} is not a directory. "
            f"Refusing rather than falling back to another snapshot - which "
            f"would serve data nobody chose.")
    return snapshot


def data_root() -> Path:
    """The data root, falling back to the repo's own copy.

    For local scripts only. Serving uses require_data_root().
    """
    configured = os.environ.get(DATA_DIR_ENV)
    return resolve_root(Path(configured)) if configured else REPO_DATA_ROOT


def configured_root() -> Path:
    """What DATA_DIR names, BEFORE the pointer is followed.

    `data_root()` and `resolve_root()` both answer "where are the tables",
    which for a volume is a snapshot directory several levels down. The
    refresh's own record sits beside the pointer, one level up, so it needs
    the configured path rather than the resolved one.
    """
    configured = os.environ.get(DATA_DIR_ENV)
    return Path(configured) if configured else REPO_DATA_ROOT


def write_last_refresh(volume: Path, outcome: str, snapshot: str = None,
                       per_league: dict = None) -> Path:
    """Record that the refresh reached a verdict, whatever the verdict was.

    WRITTEN FOR EVERY TERMINAL OUTCOME INCLUDING THE GREEN QUIET ONE, because
    the question this answers is "did the job run", not "did the data move".
    A `nothing-new` run that recorded nothing would be indistinguishable from
    a job that stopped being scheduled - and `nothing-new` is the normal
    outcome between seasons, so that gap would be open for months at a time.

    `failed` is recorded too, and separately from the timestamp: a job that
    runs and fails every morning has a current `finished_at` and an outcome
    nobody should read as healthy, which is why /health reports both rather
    than collapsing them into one "fresh" boolean.
    """
    volume.mkdir(parents=True, exist_ok=True)
    body = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "outcome": outcome,
        "snapshot": snapshot,
        "leagues": {league: entry.get("outcome")
                    for league, entry in (per_league or {}).items()},
    }
    path = volume / LAST_REFRESH_FILE
    path.write_text(json.dumps(body, indent=2) + chr(10), encoding="utf-8")
    return path


def read_last_refresh(volume: Path = None, now=None) -> dict:
    """What /health reports about the refresh job itself.

    `state` is one of:

      ok           a verdict within REFRESH_LATE_HOURS
      late         past that, which on this runner usually means host sleep
      overdue      past REFRESH_OVERDUE_HOURS, which host sleep cannot explain
      failed       the last verdict reached was a failure, however recent
      not_a_volume this DATA_DIR is a plain data root - CI and the repo copy,
                   where no refresh job exists to have run
      unknown      a volume with no record, or one that cannot be read

    `not_a_volume` AND `unknown` ARE DIFFERENT ANSWERS and are kept apart. The
    first says no job was ever meant to write here; the second says one was
    and did not. Collapsing them would make CI's permanent silence look like
    the production job having stopped - a zero standing in for an unknown,
    which is the conflation this project has four recorded bugs from.
    """
    volume = volume or configured_root()
    if not (volume / POINTER_FILE).is_file():
        return {"state": "not_a_volume", "last_verdict_at": None,
                "hours_since": None, "outcome": None, "leagues": {},
                "note": "DATA_DIR is a plain data root, not a volume, so no "
                        "refresh job maintains a record here"}

    path = volume / LAST_REFRESH_FILE
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
        finished = datetime.fromisoformat(body["finished_at"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"state": "unknown", "last_verdict_at": None,
                "hours_since": None, "outcome": None, "leagues": {},
                "note": f"{LAST_REFRESH_FILE} is absent or unreadable "
                        f"({type(error).__name__}), so when the refresh last "
                        f"ran is not known"}

    now = now or datetime.now(timezone.utc)
    if finished.tzinfo is None:
        finished = finished.replace(tzinfo=timezone.utc)
    hours = (now - finished).total_seconds() / 3600.0

    if body.get("outcome") in ("failed", "partial", "drift-report-failed"):
        state = "failed"
        note = (f"the last verdict was {body.get('outcome')!r}; the data being "
                f"served may be older than this timestamp")
    elif hours >= REFRESH_OVERDUE_HOURS:
        state = "overdue"
        note = (f"no verdict for {hours:.1f}h, past "
                f"{REFRESH_OVERDUE_HOURS}h - host sleep does not explain a "
                f"gap this long")
    elif hours >= REFRESH_LATE_HOURS:
        state = "late"
        note = (f"no verdict for {hours:.1f}h, past {REFRESH_LATE_HOURS}h. "
                f"The cron QUEUES at 07:00 and the self-hosted runner starts "
                f"when the host wakes, so this is usually a sleeping machine "
                f"rather than a fault")
    else:
        state = "ok"
        note = f"last verdict {hours:.1f}h ago"

    return {
        "state": state,
        "last_verdict_at": body["finished_at"],
        "hours_since": round(hours, 2),
        "outcome": body.get("outcome"),
        "snapshot": body.get("snapshot"),
        "leagues": body.get("leagues") or {},
        "note": note,
    }


def missing_files(root: Path) -> list:
    return [str(relative) for relative in REQUIRED_FILES
            if not (root / relative).is_file()]


def require_data_root() -> Path:
    """The data root, or a refusal naming what is wrong.

    Called once at service boot. Three distinct failures rather than one, so
    the message says which: unset, not a directory, or incomplete.
    """
    configured = os.environ.get(DATA_DIR_ENV)
    if not configured:
        raise RuntimeError(
            f"{DATA_DIR_ENV} is not set. The served game history lives on a "
            f"mounted volume, and this service deliberately has NO fallback "
            f"to a copy inside the image - a fallback would let it serve "
            f"either of two snapshots with nothing reporting which. Set "
            f"{DATA_DIR_ENV} to a data root containing "
            f"{REQUIRED_FILES[0].as_posix()} and the other five tables."
        )

    if not Path(configured).is_dir():
        raise RuntimeError(
            f"{DATA_DIR_ENV}={configured} is not a directory. On a container "
            f"this usually means the bind mount is absent or the host path "
            f"does not exist."
        )

    root = resolve_root(Path(configured))

    missing = missing_files(root)
    if missing:
        raise RuntimeError(
            f"{DATA_DIR_ENV}={configured} is missing {len(missing)} required "
            f"table(s):\n  " + "\n  ".join(missing)
            + "\nRefusing to boot rather than serving a partial snapshot."
        )
    return root


def snapshot_metadata(root: Path = None) -> dict:
    """What is being served, for /health to report.

    A SERVICE MUST BE ABLE TO SAY WHICH SNAPSHOT IT HAS. The whole reason
    there is no fallback is that two indistinguishable sources are a hazard;
    reporting the identifier is the other half of that - without it, "the
    refresh ran" and "the service picked it up" are two different facts with
    one observation between them.

    Absent or unreadable metadata is reported as such rather than raising: a
    data root seeded by hand legitimately has none, and the files themselves
    are what the service needs.
    """
    root = root or data_root()

    # THE DIRECTORY NAME IS THE IDENTITY, NOT THE STAMP INSIDE IT. Found by
    # testing: a snapshot copied by hand carries the SOURCE's SNAPSHOT.json,
    # so /health reported the id of a snapshot it was not serving - which is
    # precisely the "nothing says which" hazard the no-fallback rule exists
    # to remove, reintroduced one level in. `current` names a DIRECTORY, so
    # that name is what the service is actually reading.
    #
    # The stamp is still read, for `created` and `source`, and a disagreement
    # between the two is REPORTED rather than silently preferred either way.
    identity = root.name
    path = root / SNAPSHOT_FILE
    if not path.is_file():
        return {"snapshot": identity, "created": None,
                "source": "no SNAPSHOT.json in the data root"}
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"snapshot": identity, "created": None,
                "source": f"unreadable SNAPSHOT.json: {error}"}

    stamped = meta.get("snapshot")
    out = {
        "snapshot": identity,
        "created": meta.get("created"),
        "source": meta.get("source", "unknown"),
    }
    if stamped and stamped != identity:
        out["stamp_disagrees"] = stamped
        out["source"] = (f"{out['source']} (SNAPSHOT.json claims "
                         f"{stamped!r}; the directory name wins)")
    return out


def write_snapshot_metadata(root: Path, snapshot_id: str,
                            source: str, per_league: dict = None,
                            leagues: dict = None) -> Path:
    """Stamp a staged snapshot. Written by the refresh job, read by /health.

    `leagues` carries each league's refresh OUTCOME - advanced, unchanged,
    carried or skipped - plus, for a carried league, the snapshot its tables
    came from and why it was carried. ADDITIVE AND OPTIONAL, which is what
    makes the two shipping orders both work: an older reader ignores the key,
    and a newer reader given an older snapshot finds it absent and reports
    the outcome as unknown rather than inventing one.
    """
    path = root / SNAPSHOT_FILE
    path.write_text(json.dumps({
        "snapshot": snapshot_id,
        "created": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "data_as_of": per_league or {},
        "leagues": leagues or {},
    }, indent=2), encoding="utf-8")
    return path
