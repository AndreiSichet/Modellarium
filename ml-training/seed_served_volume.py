"""Seed the served-data volume from this checkout's committed tables.

Run once. Creates D:\\modellarium-data with one snapshot and a `current`
pointer at it, then reports what a service booting against it would see.

THE HOST PATH IS OUTSIDE EVERY CHECKOUT, and that is the decision rather than
a convenience. Two writers have to agree on it: the refresh job, which runs
from the self-hosted runner's working copy on D:, and the compose stack, which
runs from the Desktop checkout on C:. A path inside either one would tie the
served data to that checkout - and section 4 already records the runner's
`_work` path moving twice and orphaning a seeded corpus both times.
"""

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from served_data import (  # noqa: E402
    POINTER_FILE, REPO_DATA_ROOT, REQUIRED_FILES, SNAPSHOTS_DIR,
    missing_files, resolve_root, write_snapshot_metadata)

DEFAULT_VOLUME = Path("D:/modellarium-data")


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def snapshot_id(when=None) -> str:
    when = when or datetime.now(timezone.utc)
    return when.strftime("%Y-%m-%dT%H%M")


def copy_tables(source: Path, target: Path) -> list:
    copied = []
    for relative in REQUIRED_FILES:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
        copied.append((relative, destination.stat().st_size))
    return copied


def per_league_cutoffs(root: Path) -> dict:
    """The newest date in each league's history, for the snapshot stamp."""
    import pandas as pd

    def newest(relative, column="GAME_DATE"):
        frame = pd.read_csv(root / relative, usecols=[column])
        return str(pd.to_datetime(frame[column]).max().date())

    return {
        "nba": newest(Path("processed") / "games_final.csv"),
        "wnba": newest(Path("wnba") / "processed" / "wnba_games_final.csv"),
        "gleague": newest(
            Path("gleague") / "processed" / "gleague_games_final.csv"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", default=str(DEFAULT_VOLUME))
    parser.add_argument("--force", action="store_true",
                        help="add a snapshot even if the volume already has "
                             "one")
    args = parser.parse_args()

    volume = Path(args.volume)
    print(__doc__)

    section("SOURCE")
    missing = missing_files(REPO_DATA_ROOT)
    if missing:
        raise SystemExit(f"this checkout is missing {missing}")
    print(f"  {REPO_DATA_ROOT}")
    print(f"  all {len(REQUIRED_FILES)} required tables present")

    section("TARGET")
    snapshots = volume / SNAPSHOTS_DIR
    existing = sorted(p.name for p in snapshots.glob("*")) \
        if snapshots.is_dir() else []
    print(f"  {volume}")
    print(f"  existing snapshot(s): {existing or 'none'}")
    if existing and not args.force:
        raise SystemExit(
            "the volume already holds a snapshot. Seeding again would add a "
            "second one from the same committed data; pass --force if that "
            "is intended.")

    identifier = snapshot_id()
    staging = snapshots / f"{identifier}.staging"
    final = snapshots / identifier
    if staging.exists():
        shutil.rmtree(staging)

    section("COPYING")
    copied = copy_tables(REPO_DATA_ROOT, staging)
    total = sum(size for _, size in copied)
    for relative, size in copied:
        print(f"  {str(relative):<56}{size:>12,} bytes")
    print(f"  {'total':<56}{total:>12,} bytes")

    cutoffs = per_league_cutoffs(staging)
    write_snapshot_metadata(staging, identifier,
                            source="seeded from the committed tables",
                            per_league=cutoffs)
    print(f"\n  data_as_of per league: {cutoffs}")

    # Staged first, then renamed, so an interrupted copy cannot leave a
    # half-written directory under a name the pointer could be made to
    # reference. Same discipline as the pipeline's .partial writes.
    staging.rename(final)
    print(f"  staged, then renamed to {final.name}")

    section("THE POINTER")
    pointer = volume / POINTER_FILE
    pointer.write_text(identifier, encoding="utf-8")
    print(f"  {pointer} -> {identifier}")

    resolved = resolve_root(volume)
    print(f"  resolves to {resolved}")
    still_missing = missing_files(resolved)
    print(f"  missing after resolution: {still_missing or 'none'}")
    if still_missing:
        raise SystemExit("the seeded snapshot is incomplete")

    section("WHAT A SERVICE WOULD SEE")
    print(f"  DATA_DIR={volume}")
    print(f"  -> snapshot {identifier}")
    for league, date in cutoffs.items():
        print(f"     {league:<9}data_as_of {date}")
    print("\n  Next: point the compose stack at it and run the gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
