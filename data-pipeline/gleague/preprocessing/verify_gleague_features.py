"""Verify G League phase 2 - the lag guards, the Cup's reach, and Elo identity.

Every check runs against real data. The frames are modified in memory only;
nothing on disk is written, and the hashes of the processed tables are compared
before and after.

THE LAG GUARD IS THE ONE THAT MATTERS, and it is negative-tested AND
positive-controlled on all five candidates. A negative test alone passes
identically on a feature that reads no data whatever - which is how three
guards in this project turned out to be vacuous, each caught by its positive
control rather than by review.
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parents[1]
sys.path.insert(0, str(HERE))
from build_gleague_rolling_features import (  # noqa: E402
    CANDIDATES, METRICS, REGULAR, SHOWCASE, add_candidate, load)
from build_gleague_elo import (  # noqa: E402
    BASELINE_RATING, first_seasons, new_franchises, prepare, run_elo)
from clean_gleague_rows import KNOWN_UNUSABLE_GAMES  # noqa: E402

PROCESSED = PIPELINE / "data" / "gleague" / "processed"
IDENTITY = PROCESSED / "gleague_franchise_identity.csv"
DATASET = PROCESSED / "gleague_model_dataset.csv"

results = []


def record(name, ok, detail=""):
    results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).splitlines():
            print(f"         {line}")


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def hash_tree() -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(PROCESSED.glob("*.csv"))}


def compute(both: pd.DataFrame, name: str, spec: dict) -> pd.Series:
    """One candidate's PTS window, aligned to `both`'s own index."""
    values = add_candidate(both, name, spec)
    return values[f"{name}_PTS"].reindex(both.index)


def pick_probe(both: pd.DataFrame, series: pd.Series) -> int:
    """A regular-season row with a real window value and games on both sides.

    A probe whose honest value is 0, or that sits at the edge of its team's
    history, makes "unchanged" a weak claim - a function returning a constant
    would pass too.
    """
    candidates = both.index[
        series.notna() & (series > 0)
        & (both["COMPETITION"] == "regular")]
    for index in candidates:
        team = both.at[index, "TEAM_ID"]
        date = both.at[index, "GAME_DATE"]
        same = both[both["TEAM_ID"] == team]
        if (same["GAME_DATE"] < date).sum() >= 11 and \
                (same["GAME_DATE"] > date).sum() >= 11:
            return index
    raise SystemExit("no suitable probe row")


def lag_guards(both: pd.DataFrame) -> None:
    section("LAG GUARD - NEGATIVE TEST AND POSITIVE CONTROL, ALL CANDIDATES")

    for name, spec in CANDIDATES.items():
        series = compute(both, name, spec)
        index = pick_probe(both, series)
        team = both.at[index, "TEAM_ID"]
        date = both.at[index, "GAME_DATE"]
        honest = series.at[index]

        later = both.index[(both["TEAM_ID"] == team)
                           & (both["GAME_DATE"] > date)]
        earlier = both.index[(both["TEAM_ID"] == team)
                             & (both["GAME_DATE"] < date)]

        after = both.copy()
        after.loc[later, "PTS"] = 999
        unchanged = compute(after, name, spec).at[index]

        before = both.copy()
        before.loc[earlier, "PTS"] = 999
        moved = compute(before, name, spec).at[index]

        ok = (abs(unchanged - honest) < 1e-12
              and abs(moved - honest) > 1e-9)
        record(f"{name}: reads only earlier games",
               ok,
               f"probe team {team} on {date.date()}, honest {honest:.4f}\n"
               f"corrupting {len(later)} LATER games  -> {unchanged:.4f}\n"
               f"corrupting {len(earlier)} EARLIER games -> {moved:.4f}")


def cup_reaches_cup_candidates(both: pd.DataFrame) -> None:
    """A Cup game must move CUP5/CUP10 and must not move CARRY5/CARRY10.

    This is what proves the Cup-inclusive candidates are genuinely
    Cup-inclusive, rather than carried windows under a different name.
    """
    section("THE CUP ACTUALLY FEEDS THE CUP CANDIDATES")

    cup_rows = both.index[both["COMPETITION"] == "showcase"]
    cup_seasons = sorted(both.loc[cup_rows, "SEASON"].unique())
    first_cup = cup_seasons[0]

    corrupted = both.copy()
    corrupted.loc[cup_rows, "PTS"] = 999

    for window in (5, 10):
        for prefix, should_move in ((f"CUP{window}", True),
                                    (f"CARRY{window}", False)):
            spec = CANDIDATES[prefix]
            honest = compute(both, prefix, spec)
            after = compute(corrupted, prefix, spec)

            scope = both.index[(both["COMPETITION"] == "regular")
                               & (both["SEASON"] >= first_cup)]
            differs = int((honest.loc[scope].fillna(-1)
                           != after.loc[scope].fillna(-1)).sum())
            ok = (differs > 0) if should_move else (differs == 0)
            record(f"{prefix}: corrupting every Cup game "
                   f"{'moves' if should_move else 'does NOT move'} it",
                   ok,
                   f"{differs:,} of {len(scope):,} regular-season rows from "
                   f"{first_cup} onward changed")


def carry_crosses_the_boundary(both: pd.DataFrame) -> None:
    """The carried candidates must reach into the previous season; ROLL5 must
    not.

    THE GENERIC LAG PROBE CANNOT ESTABLISH THIS, and that is why this check
    exists. It picked a row in 2003-04 for all five candidates - the first
    season, where a carried window has nothing to carry and behaves exactly
    like a within-season one. So the five passes above prove no lookahead and
    say nothing about whether CARRY and CUP reach where their names claim.
    """
    section("THE CARRIED CANDIDATES REACH INTO THE PREVIOUS SEASON")

    regular = both[both["COMPETITION"] == "regular"]
    seasons = sorted(regular["SEASON"].unique())

    # A team's opening game in a season that is not its first, so a carried
    # window must look back across the boundary to be populated at all.
    probe = None
    for season in seasons[1:]:
        rows = regular[regular["SEASON"] == season]
        for team in rows["TEAM_ID"].unique():
            prior = regular[(regular["TEAM_ID"] == team)
                            & (regular["SEASON"] < season)]
            if len(prior) < 11:
                continue
            opener = rows[rows["TEAM_ID"] == team].sort_values("GAME_DATE")
            probe = (opener.index[0], team, season,
                     prior["SEASON"].max())
            break
        if probe:
            break
    index, team, season, previous = probe
    print(f"  probe: team {team}'s first game of {season}, "
          f"previous season {previous}")

    corrupted = both.copy()
    target = both.index[(both["TEAM_ID"] == team)
                        & (both["SEASON"] == previous)]
    corrupted.loc[target, "PTS"] = 999
    print(f"  corrupting that team's {len(target)} games in {previous} "
          f"only\n")

    for name, spec in CANDIDATES.items():
        honest = compute(both, name, spec).at[index]
        after = compute(corrupted, name, spec).at[index]
        moved = not (pd.isna(honest) and pd.isna(after)) and honest != after
        should = spec["crosses_seasons"]
        record(f"{name}: {'reaches' if should else 'does NOT reach'} "
               f"across the boundary", moved == should,
               f"honest {honest!r} -> {after!r}")


def cup_equivalence(both: pd.DataFrame) -> None:
    section("CUP AND CARRY ARE IDENTICAL WHERE NO CUP EXISTS")

    cup_seasons = sorted(
        both.loc[both["COMPETITION"] == "showcase", "SEASON"].unique())
    regular = both[both["COMPETITION"] == "regular"]
    before = regular[regular["SEASON"] < cup_seasons[0]]

    for window in (5, 10):
        carry = compute(both, f"CARRY{window}", CANDIDATES[f"CARRY{window}"])
        cup = compute(both, f"CUP{window}", CANDIDATES[f"CUP{window}"])
        a = carry.loc[before.index]
        b = cup.loc[before.index]
        same = bool(((a.isna() & b.isna()) | (a == b)).all())
        record(f"CUP{window} == CARRY{window} on the "
               f"{before['SEASON'].nunique()} pre-Cup seasons", same,
               f"{len(before):,} rows compared")


def unusable_games_are_gaps(both: pd.DataFrame) -> None:
    section("THE UNUSABLE GAMES REMAIN GAPS")
    present = [g for g in KNOWN_UNUSABLE_GAMES
               if g in set(both["GAME_ID"].astype(str))]
    record(f"none of the {len(KNOWN_UNUSABLE_GAMES)} unusable games was "
           f"invented as a row", not present,
           f"present: {present}" if present else
           f"{sorted(KNOWN_UNUSABLE_GAMES)}")


def margin_is_recomputed(both: pd.DataFrame) -> None:
    section("THE RECOMPUTED MARGIN IS WHAT WAS ROLLED")
    opponent = both.groupby("GAME_ID")["PTS"].transform(
        lambda s: s.values[::-1] if len(s) == 2 else pd.NA)
    derived = both["PTS"] - opponent
    bad = int((derived != both["PLUS_MINUS"]).sum())
    record("PLUS_MINUS == PTS - opponent PTS on every row, both "
           "competitions", not bad,
           f"{len(both):,} rows checked, {bad} differ")


def elo_identity(games: pd.DataFrame, identity: pd.DataFrame) -> None:
    """One known relocation continues; one known new franchise starts fresh."""
    section("ELO: RELOCATION CONTINUES, NEW FRANCHISE STARTS FRESH")

    seasons = sorted(games["SEASON"].unique())
    newcomers = new_franchises(identity, seasons)
    played = run_elo(games, 20, 0.2, 0.0, newcomers)

    def entering(team, season):
        rows = played[(played["SEASON"] == season)
                      & ((played["HOME_TEAM_ID"] == team)
                         | (played["AWAY_TEAM_ID"] == team))]
        if not len(rows):
            return None
        first = rows.iloc[0]
        return float(first["HOME_TEAM_ELO"]
                     if first["HOME_TEAM_ID"] == team
                     else first["AWAY_TEAM_ELO"])

    changes = identity[identity["identity_changed"]]
    # A relocation that is NOT the franchise's first season, so there is a
    # prior rating for it to continue from.
    row = changes.iloc[0]
    value = entering(row["TEAM_ID"], row["SEASON"])
    record(f"relocation {row['previous_abbr']} -> {row['abbr']} "
           f"({row['TEAM_ID']}) at {row['SEASON']} continues its rating",
           value is not None and abs(value - BASELINE_RATING) > 1e-9,
           f"entering rating {value:.2f}, not the {BASELINE_RATING:.0f} "
           f"baseline")

    team, season = sorted(newcomers.items(), key=lambda kv: kv[1])[0]
    value = entering(team, season)
    record(f"new franchise {team} starts from the baseline at {season}",
           value is not None and abs(value - BASELINE_RATING) < 1e-9,
           f"entering rating {value:.2f}")

    firsts = first_seasons(identity)
    restarts = 0
    checked = 0
    for team, first in firsts.items():
        for season in seasons:
            if season <= first:
                continue
            value = entering(team, season)
            if value is None:
                continue
            checked += 1
            if abs(value - BASELINE_RATING) < 1e-9:
                restarts += 1
    record("no continuing franchise ever restarts at the baseline",
           restarts == 0,
           f"{checked} season boundaries checked, {restarts} restarts")


def dataset_shape() -> None:
    section("THE DATASET ITSELF")
    data = pd.read_csv(DATASET, dtype={"GAME_ID": str},
                       parse_dates=["GAME_DATE"])

    expected = {f"{side}_{name}_{metric}"
                for side in ("HOME", "AWAY")
                for name in CANDIDATES for metric in METRICS}
    record(f"all {len(expected)} candidate feature columns present",
           expected <= set(data.columns),
           f"missing: {sorted(expected - set(data.columns))[:4]}")

    record("one row per game", data["GAME_ID"].is_unique,
           f"{len(data):,} rows, {data['GAME_ID'].nunique():,} ids")

    record("no tied game reached the dataset",
           not bool((data["HOME_MARGIN"] == 0).any()))

    digits = set(data["GAME_ID"].str.zfill(10).str[2])
    record("every row is a regular-season game", digits == {"2"},
           f"digits found: {sorted(digits)}")


def main() -> int:
    print(__doc__)
    before = hash_tree()

    regular = load(REGULAR, "regular")
    showcase = load(SHOWCASE, "showcase")
    both = pd.concat([regular, showcase], ignore_index=True)
    both = both.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    print(f"Loaded {len(regular):,} regular and {len(showcase):,} Cup "
          f"team-games.")

    identity = pd.read_csv(IDENTITY, dtype={"TEAM_ID": "int64"})
    games = prepare(regular)

    margin_is_recomputed(both)
    unusable_games_are_gaps(both)
    lag_guards(both)
    cup_reaches_cup_candidates(both)
    carry_crosses_the_boundary(both)
    cup_equivalence(both)
    elo_identity(games, identity)
    dataset_shape()

    section("NOTHING ON DISK WAS WRITTEN")
    after = hash_tree()
    changed = [n for n in before if after.get(n) != before[n]]
    record(f"all {len(before)} processed table hashes unchanged",
           not changed and set(after) == set(before),
           f"changed: {changed}" if changed else "")

    section("RESULT")
    failed = [n for n, ok in results if not ok]
    print(f"  {len(results) - len(failed)} passed, {len(failed)} failed")
    for name in failed:
        print(f"    FAILED: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
