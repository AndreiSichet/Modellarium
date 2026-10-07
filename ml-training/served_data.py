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

so one environment variable can point at either the repo or a snapshot with no
per-file configuration, and CI can supply the committed copy without any other
change.
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
]


POINTER_FILE = "current"
SNAPSHOTS_DIR = "snapshots"

# Which league each served table belongs to, so /health can report them grouped
# the way every other per-league field already is.
TABLE_LEAGUE = {
    "processed/games_final.csv": "nba",
    "processed/player_boxscores_with_rolling.csv": "nba",
    "processed/quarter_half_raw.csv": "nba",
    "wnba/processed/wnba_games_final.csv": "wnba",
    "gleague/processed/gleague_games_final.csv": "gleague",
    "gleague/processed/gleague_showcase_games.csv": "gleague",
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
    for relative in REQUIRED_FILES:
        key = relative.as_posix()
        league = TABLE_LEAGUE.get(key, "other")
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
                            source: str, per_league: dict = None) -> Path:
    """Stamp a staged snapshot. Written by the refresh job, read by /health."""
    path = root / SNAPSHOT_FILE
    path.write_text(json.dumps({
        "snapshot": snapshot_id,
        "created": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "data_as_of": per_league or {},
    }, indent=2), encoding="utf-8")
    return path
