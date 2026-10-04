"""Build the G League long-format tables, regular season and Showcase Cup.

Column naming matches the NBA's games_final.csv and the WNBA's
wnba_games_final.csv so phase 2 reads the same names against any league.
Every season is carried; phase 3 chooses the training span.

Cleaning is delegated to clean_gleague_rows.clean_season, the same function
the validator runs, so the output cannot be cleaned by a rule the validation
never saw.
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parents[1]
sys.path.insert(0, str(HERE))
from clean_gleague_rows import VALID_WL, clean_season  # noqa: E402

RAW_DIR = PIPELINE / "data" / "gleague" / "raw"
OUT_DIR = PIPELINE / "data" / "gleague" / "processed"

OUTPUTS = {
    "regular": OUT_DIR / "gleague_games_final.csv",
    "showcase": OUT_DIR / "gleague_showcase_games.csv",
}
IDENTITY_OUT = OUT_DIR / "gleague_franchise_identity.csv"

DROP_COLUMNS = ["SEASON_ID", "MIN", "TEAM_ABBREVIATION", "MATCHUP"]


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def load_and_clean(kind: str) -> pd.DataFrame:
    frames = []
    for path in sorted(RAW_DIR.glob(f"gleague_{kind}_*.csv")):
        season = path.stem.split("_")[-1]
        raw = pd.read_csv(path, dtype={"GAME_ID": str})
        raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"])
        report = clean_season(raw)
        frame = report["clean"].copy()
        frame["SEASON"] = season
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def derive_home_and_opponent(frame: pd.DataFrame) -> pd.DataFrame:
    """IS_HOME from MATCHUP, OPPONENT by pairing on GAME_ID.

    OPPONENT is derived by pairing rather than by parsing MATCHUP text, for
    the reason the NBA pipeline records: a corrupted MATCHUP is identical on
    both rows, so parsing it would silently make a team its own opponent.
    Games whose home side is ambiguous are already removed upstream.
    """
    frame = frame.copy()
    frame["IS_HOME"] = frame["MATCHUP"].str.contains("vs.", regex=False)

    paired = frame.groupby("GAME_ID")["TEAM_ID"].transform(
        lambda s: s.values[::-1] if len(s) == 2 else pd.NA)
    frame["OPPONENT_TEAM_ID"] = paired

    abbr = frame.set_index(["GAME_ID", "TEAM_ID"])["TEAM_ABBREVIATION"]
    frame["OPPONENT"] = [
        abbr.get((g, o), pd.NA)
        for g, o in zip(frame["GAME_ID"], frame["OPPONENT_TEAM_ID"])
    ]
    return frame


def recompute_margin(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace PLUS_MINUS with the margin derived from PTS.

    The source column disagrees with PTS on 6.79% of G League games where it
    is populated, against 1.02% for the NBA and 1.24% for the WNBA - same
    endpoint, same defect, an order of magnitude more of it. It is also
    entirely absent before 2005-06. Deriving it from PTS is arithmetic over a
    column the validator has already checked, not invented data.
    """
    opponent_pts = frame.groupby("GAME_ID")["PTS"].transform(
        lambda s: s.values[::-1] if len(s) == 2 else pd.NA)
    derived = frame["PTS"] - opponent_pts

    had = frame["PLUS_MINUS"].notna()
    changed = int((derived[had] != frame.loc[had, "PLUS_MINUS"]).sum())
    print(f"  PLUS_MINUS: recomputed from PTS. It was populated on "
          f"{int(had.sum()):,} of {len(frame):,} rows and disagreed on "
          f"{changed:,} of those.")

    # The sign must agree with WL wherever WL exists. This is what makes the
    # recomputation a correction rather than a substitution.
    known = frame["WL"].isin(VALID_WL)
    expected_win = derived[known] > 0
    actually_win = frame.loc[known, "WL"] == "W"
    disagree = int((expected_win != actually_win).sum())
    if disagree:
        raise SystemExit(
            f"the derived margin's sign disagrees with WL on {disagree} "
            f"row(s). The validator should have removed these; refusing to "
            f"build a table whose margins contradict its own results.")
    print(f"  Derived sign agrees with WL on all {int(known.sum()):,} rows "
          f"where WL is present ({len(frame) - int(known.sum())} rows have no "
          f"WL; their margin still comes from PTS).")

    frame = frame.copy()
    frame["PLUS_MINUS"] = derived
    return frame


def franchise_identity(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (TEAM_ID, SEASON) with that season's labels, plus a flag
    marking the seasons where the labels changed."""
    per = (frame.groupby(["TEAM_ID", "SEASON"])
           .agg(abbr=("TEAM_ABBREVIATION", "first"),
                name=("TEAM_NAME", "first"),
                games=("GAME_ID", "nunique"))
           .reset_index()
           .sort_values(["TEAM_ID", "SEASON"]))

    per["previous_abbr"] = per.groupby("TEAM_ID")["abbr"].shift()
    per["previous_name"] = per.groupby("TEAM_ID")["name"].shift()
    per["identity_changed"] = (
        per["previous_abbr"].notna()
        & ((per["previous_abbr"] != per["abbr"])
           | (per["previous_name"] != per["name"])))
    per["first_season"] = per["previous_abbr"].isna()
    return per


def main() -> int:
    print(__doc__)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    built = {}
    for kind, path in OUTPUTS.items():
        section(f"{kind.upper()}")
        frame = load_and_clean(kind)
        if frame.empty:
            print(f"  nothing to build for {kind}")
            continue

        print(f"  {len(frame):,} clean rows across "
              f"{frame['SEASON'].nunique()} season(s)")

        frame["WIN"] = (frame["WL"] == "W").astype("Int64")
        frame.loc[~frame["WL"].isin(VALID_WL), "WIN"] = pd.NA

        frame = derive_home_and_opponent(frame)
        frame = recompute_margin(frame)

        unpaired = frame["OPPONENT_TEAM_ID"].isna().sum()
        if unpaired:
            raise SystemExit(f"{unpaired} row(s) have no opponent")

        per_game_home = frame.groupby("GAME_ID")["IS_HOME"].sum()
        if not (per_game_home == 1).all():
            raise SystemExit("a game does not have exactly one home side")

        out = frame.drop(columns=[c for c in DROP_COLUMNS
                                  if c in frame.columns])
        out = out.sort_values(["GAME_DATE", "GAME_ID", "IS_HOME"])
        out.to_csv(path, index=False, encoding="utf-8")
        built[kind] = frame  # pre-drop: the identity table needs the labels
        print(f"  wrote {path.name}: {len(out):,} rows x {out.shape[1]} cols, "
              f"{out['GAME_ID'].nunique():,} games")

    if "regular" in built:
        section("FRANCHISE IDENTITY TABLE")
        identity = franchise_identity(built["regular"])
        identity.to_csv(IDENTITY_OUT, index=False, encoding="utf-8")
        print(f"  wrote {IDENTITY_OUT.name}: {len(identity):,} rows")
        print(f"  {identity['TEAM_ID'].nunique()} franchises, "
              f"{int(identity['identity_changed'].sum())} identity changes, "
              f"{int(identity['first_season'].sum())} first appearances")

    if {"regular", "showcase"} <= built.keys():
        section("THE TWO COMPETITIONS ARE DISJOINT")
        r = set(built["regular"]["GAME_ID"])
        s = set(built["showcase"]["GAME_ID"])
        print(f"  regular {len(r):,} games, showcase {len(s):,} games, "
              f"shared {len(r & s)}")
        if r & s:
            raise SystemExit("a game appears in both competitions")
        print("  The Showcase Cup is absent from the regular-season table, "
              "which is what\n  keeps the models regular-season only - and "
              "present on disk, which is what\n  lets phase 2 compute rest "
              "days across the competition boundary.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
