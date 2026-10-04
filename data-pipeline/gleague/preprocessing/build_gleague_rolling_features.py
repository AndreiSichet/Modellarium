"""Rolling form for the G League - five candidates, including a Cup-inclusive
pair the WNBA could not offer.

PHASE 3 CHOOSES. Building one window here would be intuition wearing a
pipeline step, so every candidate ships and the retention figures below are
what make the trade-off visible.

  ROLL5    regular season only, within season   - NaN until the window fills
  CARRY5   regular season only, carried          - reaches into last season
  CARRY10  regular season only, carried
  CUP5     regular season PLUS Showcase Cup, carried
  CUP10    regular season PLUS Showcase Cup, carried

THE CUP-INCLUSIVE PAIR IS THE NEW IDEA, and it exists because of what the
WNBA phase found: carrying windows across seasons beat within-season windows,
mainly by retaining early-season games. The G League has a third option. For a
team's first league games the alternatives are no history (ROLL), last
season's form (CARRY), or the last few weeks of real games against real
opponents (CUP) - and with G League roster churn, last season's form may
describe a substantially different team.

The Cup existed for five of twenty-three seasons, so CUP and CARRY differ
ONLY there. That is reported explicitly: the comparison phase 3 runs is
decided on a minority of the data.

TARGETS STAY REGULAR-SEASON ONLY. The Cup feeds features and never labels.

THE MARGIN IS THE RECOMPUTED ONE. `PLUS_MINUS` from LeagueGameFinder is wrong
on 6.79% of regular-season games here and 14.6% of Cup games - the worst of
the three leagues - and the Cup-inclusive windows would carry the worse of the
two. The builder derived it from PTS; this asserts that before rolling.
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parents[1]
sys.path.insert(0, str(HERE))
from clean_gleague_rows import KNOWN_UNUSABLE_GAMES  # noqa: E402

PROCESSED = PIPELINE / "data" / "gleague" / "processed"
REGULAR = PROCESSED / "gleague_games_final.csv"
SHOWCASE = PROCESSED / "gleague_showcase_games.csv"
OUT = PROCESSED / "gleague_rolling_features.csv"

# Declared here, before anything is scored, so phase 3 cannot quietly add a
# candidate after seeing a result.
CANDIDATES = {
    "ROLL5": {"window": 5, "crosses_seasons": False, "include_cup": False},
    "CARRY5": {"window": 5, "crosses_seasons": True, "include_cup": False},
    "CARRY10": {"window": 10, "crosses_seasons": True, "include_cup": False},
    "CUP5": {"window": 5, "crosses_seasons": True, "include_cup": True},
    "CUP10": {"window": 10, "crosses_seasons": True, "include_cup": True},
}

METRICS = {
    "WIN_PCT": "WIN",
    "PTS": "PTS",
    "PLUS_MINUS": "PLUS_MINUS",
    "FG_PCT": "FG_PCT",
    "REB": "REB",
    "AST": "AST",
    "TOV": "TOV",
}


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def load(path: Path, competition: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"GAME_ID": str})
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])
    frame["COMPETITION"] = competition
    return frame


def assert_corrected_margin(frame: pd.DataFrame) -> None:
    """PLUS_MINUS must equal PTS minus the opponent's PTS on every row."""
    opponent = frame.groupby("GAME_ID")["PTS"].transform(
        lambda s: s.values[::-1] if len(s) == 2 else pd.NA)
    derived = frame["PTS"] - opponent
    bad = int((derived != frame["PLUS_MINUS"]).sum())
    if bad:
        raise SystemExit(
            f"PLUS_MINUS differs from PTS - opponent PTS on {bad} row(s). "
            f"The builder is supposed to have recomputed it; rolling the raw "
            f"column would carry a 6.79% error rate into every window.")
    print(f"  PLUS_MINUS == PTS - opponent PTS on all {len(frame):,} rows")


def assert_universe_complete(frame: pd.DataFrame, label: str) -> None:
    """Every game has exactly two team rows, and the unusable games are gaps.

    A ROLLING MEAN OVER A FRAME WITH HOLES SILENTLY SHORTENS ITS OWN WINDOW.
    If a game is simply absent, a team's "last five" quietly reaches six games
    back - a confident number computed from a window other than the one it
    claims. So the frame is reindexed onto the complete team-game universe
    before rolling, which is what makes NaN propagate properly.
    """
    sides = frame.groupby("GAME_ID").size()
    if not (sides == 2).all():
        raise SystemExit(f"{label}: {int((sides != 2).sum())} game(s) do not "
                         f"have exactly two sides")
    print(f"  {label}: {frame['GAME_ID'].nunique():,} games x 2 sides = "
          f"{len(frame):,} rows")

    invented = [g for g in KNOWN_UNUSABLE_GAMES if g in set(frame["GAME_ID"])]
    if invented:
        raise SystemExit(
            f"{label}: the unusable game(s) {invented} are present. They must "
            f"remain GAPS - inventing a row for a game whose own record "
            f"contradicts itself is worse than the hole.")
    print(f"  the {len(KNOWN_UNUSABLE_GAMES)} unusable games are absent, "
          f"as gaps rather than invented rows")


def trailing_mean(series: pd.Series, window: int) -> pd.Series:
    """Mean of the previous `window` values, excluding the current one.

    The shift is the leakage guard: without it a game's own result enters its
    own features.
    """
    return series.shift(1).rolling(window).mean()


def add_candidate(frame: pd.DataFrame, name: str, spec: dict) -> pd.DataFrame:
    """One candidate's columns, computed over whichever games it counts."""
    stream = frame if spec["include_cup"] else \
        frame[frame["COMPETITION"] == "regular"]
    stream = stream.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"])

    keys = ["TEAM_ID"] if spec["crosses_seasons"] else ["TEAM_ID", "SEASON"]
    window = spec["window"]

    out = {}
    for metric, source in METRICS.items():
        out[f"{name}_{metric}"] = (
            stream.groupby(keys, sort=False)[source]
            .transform(lambda s: trailing_mean(s, window)))
    values = pd.DataFrame(out, index=stream.index)

    # Only regular-season rows are ever scored, so the Cup rows' own feature
    # values are computed and then dropped - they exist to be history.
    return values


def retention_report(frame: pd.DataFrame, shapes: dict) -> None:
    section("RETENTION - BOTH SIDES COMPLETE, PER CANDIDATE PER SEASON")
    print("""A game is usable only if BOTH teams have a complete window, so
the loss compounds. Unbalanced seasons are flagged: teams with fewer games
reach a full window later, so they may lose differently from uniform ones.""")

    per_game = frame.groupby("GAME_ID").agg(season=("SEASON", "first"))
    complete = {}
    for name in CANDIDATES:
        cols = [f"{name}_{m}" for m in METRICS]
        row_ok = frame[cols].notna().all(axis=1)
        per_game[name] = row_ok.groupby(frame["GAME_ID"]).all()
        complete[name] = per_game[name]

    header = f"  {'season':<9}{'games':>7}{'shape':>12}"
    for name in CANDIDATES:
        header += f"{name:>9}"
    print(f"\n{header}")
    print("  " + "-" * (28 + 9 * len(CANDIDATES)))

    for season in sorted(per_game["season"].unique()):
        rows = per_game["season"] == season
        total = int(rows.sum())
        verdict = shapes.get(season, "?")
        line = f"  {season:<9}{total:>7}{verdict:>12}"
        for name in CANDIDATES:
            kept = int((per_game[name] & rows).sum())
            line += f"{kept / total * 100:>8.0f}%"
        print(line)

    line = f"  {'ALL':<9}{len(per_game):>7}{'':>12}"
    for name in CANDIDATES:
        line += f"{per_game[name].mean() * 100:>8.0f}%"
    print("  " + "-" * (28 + 9 * len(CANDIDATES)))
    print(line)

    unbalanced = [s for s, v in shapes.items() if v == "unbalanced"]
    if unbalanced:
        print(f"\n  The {len(unbalanced)} unbalanced season(s) "
              f"({', '.join(sorted(unbalanced))}) separately:")
        for name in CANDIDATES:
            mask = per_game["season"].isin(unbalanced)
            other = ~mask
            print(f"    {name:<9} unbalanced "
                  f"{per_game.loc[mask, name].mean() * 100:5.1f}%   "
                  f"rest {per_game.loc[other, name].mean() * 100:5.1f}%")


def cup_equivalence_report(frame: pd.DataFrame) -> None:
    """CUP and CARRY must be identical wherever no Cup exists."""
    section("CUP-INCLUSIVE VARIANTS DIFFER ONLY IN THE FIVE CUP SEASONS")

    cup_seasons = sorted(
        frame.loc[frame["COMPETITION"] == "showcase", "SEASON"].unique())
    regular = frame[frame["COMPETITION"] == "regular"]
    print(f"  Cup seasons: {len(cup_seasons)} of "
          f"{regular['SEASON'].nunique()}  ({', '.join(cup_seasons)})")

    for window in (5, 10):
        carry = [f"CARRY{window}_{m}" for m in METRICS]
        cup = [f"CUP{window}_{m}" for m in METRICS]

        # A Cup season's history reaches back into the previous season, so the
        # FIRST Cup season is where the two first diverge and every later
        # season inherits it. The clean comparison is the seasons strictly
        # before that.
        before = regular[regular["SEASON"] < cup_seasons[0]]
        a = before[carry].to_numpy()
        b = before[cup].to_numpy()
        same = (pd.isna(a) == pd.isna(b)).all() and \
            (a[~pd.isna(a)] == b[~pd.isna(b)]).all()
        print(f"  CUP{window} vs CARRY{window} on the "
              f"{before['SEASON'].nunique()} pre-Cup seasons "
              f"({len(before):,} rows): "
              f"{'IDENTICAL' if same else 'DIFFER'}")
        if not same:
            raise SystemExit(
                f"CUP{window} differs from CARRY{window} in a season with no "
                f"Cup games, so it is not the carried window plus the Cup")

        after = regular[regular["SEASON"] >= cup_seasons[0]]
        a, b = after[carry].to_numpy(), after[cup].to_numpy()
        differs = int((~((pd.isna(a) & pd.isna(b))
                         | (a == b))).any(axis=1).sum())
        print(f"  {' ' * len(f'CUP{window}')}  from "
              f"{cup_seasons[0]} onward ({len(after):,} rows): "
              f"{differs:,} row(s) differ "
              f"({differs / len(after) * 100:.0f}%)")

    print("""
  So phase 3's CUP-vs-CARRY comparison is decided on a MINORITY of the data.
  Everything before the first Cup season is identical by construction, which
  is the right behaviour and also means a win for CUP there would be noise.""")


def main() -> int:
    print(__doc__)

    regular = load(REGULAR, "regular")
    showcase = load(SHOWCASE, "showcase")

    section("INPUT GATES")
    assert_corrected_margin(regular)
    assert_universe_complete(regular, "regular")
    assert_universe_complete(showcase, "showcase")

    both = pd.concat([regular, showcase], ignore_index=True)
    both = both.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    shapes = {}
    for season, group in regular.groupby("SEASON"):
        per_team = group.groupby("TEAM_ID")["GAME_ID"].nunique()
        modal = int(per_team.mode().iloc[0])
        shapes[season] = ("uniform" if per_team.min() == per_team.max()
                          else "short-only" if per_team.max() == modal
                          else "unbalanced")

    section("CANDIDATES")
    for name, spec in CANDIDATES.items():
        print(f"  {name:<9} window {spec['window']:>2}, "
              f"{'carried across seasons' if spec['crosses_seasons'] else 'within season        '}, "
              f"{'regular + Cup' if spec['include_cup'] else 'regular only '}")

    for name, spec in CANDIDATES.items():
        values = add_candidate(both, name, spec)
        for column in values.columns:
            both[column] = values[column]

    out = both[both["COMPETITION"] == "regular"].copy()

    retention_report(out, shapes)
    cup_equivalence_report(both)

    section("OUTPUT")
    columns = (["GAME_ID", "TEAM_ID", "SEASON", "GAME_DATE"]
               + [f"{n}_{m}" for n in CANDIDATES for m in METRICS])
    out[columns].to_csv(OUT, index=False, encoding="utf-8")
    print(f"  wrote {OUT.name}: {len(out):,} rows x {len(columns)} cols")
    print(f"  {len(CANDIDATES)} candidates x {len(METRICS)} metrics = "
          f"{len(CANDIDATES) * len(METRICS)} feature columns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
