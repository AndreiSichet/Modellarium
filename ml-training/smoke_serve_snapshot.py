"""Compute live features for one fixture per league from a given data root.

WHAT A VALIDATOR CANNOT TELL YOU. Each league's validator checks its table's
own shape - columns, nulls, every game having two sides. None of them can say
whether a live feature row COMPUTES from it: that needs the rolling windows,
the rest-day lookup and the Elo replay to run against the actual file, which
is a different question and the one that decides whether a snapshot is
servable.

So this is the last gate before a swap. It is deliberately NOT the full
inference service: no models are loaded and nothing is scored. A snapshot
cannot change which model serves - models stay baked into the image - so the
only new risk a snapshot carries is that features cannot be built from it.

Usage:  python smoke_serve_snapshot.py <data-root>
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    root = Path(sys.argv[1])
    # Set before importing anything that resolves a path, because the
    # league modules read the environment when their loaders run.
    os.environ["DATA_DIR"] = str(root)

    for directory in (HERE, HERE / "wnba", HERE / "gleague",
                      PROJECT / "data-pipeline" / "preprocessing",
                      PROJECT / "data-pipeline" / "ingestion",
                      PROJECT / "data-pipeline" / "wnba" / "preprocessing",
                      PROJECT / "data-pipeline" / "wnba" / "ingestion",
                      PROJECT / "data-pipeline" / "gleague" / "preprocessing",
                      PROJECT / "data-pipeline" / "gleague" / "ingestion"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))

    import pandas as pd
    from served_data import require_data_root, snapshot_metadata

    resolved = require_data_root()
    meta = snapshot_metadata(resolved)
    print(f"  root     {resolved}")
    print(f"  snapshot {meta.get('snapshot') or 'unstamped'}")

    failures = []

    # --- NBA: the three markets' shared feature row
    try:
        from live_features import get_live_features, load_games_final

        games = load_games_final()
        newest = pd.Timestamp(games["GAME_DATE"].max())
        # The NBA table carries OPPONENT as an ABBREVIATION, not an id - the
        # OPPONENT_TEAM_ID column exists only in the WNBA's and G League's
        # tables, which were built later. So the pair comes from the two rows
        # that share a GAME_ID.
        last = games[games["GAME_DATE"] == newest]
        pair = last[last["GAME_ID"] == last.iloc[0]["GAME_ID"]]
        home = int(pair[pair["IS_HOME"]].iloc[0]["TEAM_ID"])
        away = int(pair[~pair["IS_HOME"]].iloc[0]["TEAM_ID"])
        features = get_live_features(
            home, away, newest + pd.Timedelta(days=1), games)
        print(f"  nba      {features.shape[1]} features computed for "
              f"{(newest + pd.Timedelta(days=1)).date()}")
        # THE PASS CONDITION IS UNCHANGED. This records the availability state
        # rather than gating on it: the four availability features are NaN out
        # of season by design, so failing here would make the smoke test red
        # every day from April to October for correct behaviour. What was wrong
        # before was that it was SILENT - an unreachable sidecar and a quiet
        # offseason produced identical output.
        try:
            import injury_availability as availability

            observed = availability.availability_state()
            print(f"           availability {observed.get('state')}"
                  f"  ({observed.get('source')})")
            if observed.get("state") in ("unreachable", "source_failed"):
                print("           NOT the offseason case - this snapshot would be "
                      "served on 34 of 38 features")
        except Exception as error:  # noqa: BLE001
            print(f"           availability state unknown: "
                  f"{type(error).__name__}: {error}")
        if features.isna().all(axis=None):
            failures.append("nba: every feature is NaN")
    except Exception as error:  # noqa: BLE001
        failures.append(f"nba: {type(error).__name__}: {error}")

    # --- WNBA
    try:
        import live_wnba_features as wnba

        games = wnba.load_games()
        newest = pd.Timestamp(games["GAME_DATE"].max())
        row = games[games["GAME_DATE"] == newest].iloc[0]
        result = wnba.get_live_features(
            int(row["TEAM_ID"]), int(row["OPPONENT_TEAM_ID"]),
            newest + pd.Timedelta(days=1))
        print(f"  wnba     {len(result['rows'])} target row(s) computed for "
              f"{(newest + pd.Timedelta(days=1)).date()}")
    except Exception as error:  # noqa: BLE001
        # A fixture whose rolling window is incomplete is REFUSED by design,
        # and that refusal is correct behaviour rather than an unservable
        # snapshot. Only an unexpected failure counts.
        if type(error).__name__ == "NotScoreable":
            print(f"  wnba     refused as unscoreable, which is by design")
        else:
            failures.append(f"wnba: {type(error).__name__}: {error}")

    # --- G League
    try:
        import live_gleague_features as gleague

        state = gleague.load_state()
        newest = state["data_as_of"]
        regular = state["regular"]
        row = regular[regular["GAME_DATE"] == newest].iloc[0]
        result = gleague.get_live_features(
            int(row["TEAM_ID"]), int(row["OPPONENT_TEAM_ID"]),
            newest + pd.Timedelta(days=1), state)
        print(f"  gleague  {len(result['rows'])} target row(s) computed for "
              f"{(newest + pd.Timedelta(days=1)).date()}")
    except Exception as error:  # noqa: BLE001
        if type(error).__name__ == "NotScoreable":
            print(f"  gleague  refused as unscoreable, which is by design")
        else:
            failures.append(f"gleague: {type(error).__name__}: {error}")

    if failures:
        print("\n  SNAPSHOT IS NOT SERVABLE:")
        for failure in failures:
            print(f"    {failure}")
        return 1

    print("  all three leagues compute live features from this snapshot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
