"""Read-only quality gate over the fetched G League seasons.

Validating 23 seasons is a much harder test of the method than the WNBA's 12,
and it found four defect classes the WNBA's data did not have plus a flaw in
the completeness rule itself. See clean_gleague_rows.py for the classes and
for why partial stat lines are summed rather than de-duplicated - the data
settled that, not reasoning.

Reports per season, and exits non-zero on anything it cannot classify. Writes
nothing.
"""

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parents[1]
sys.path.insert(0, str(HERE))
from clean_gleague_rows import (KNOWN_UNUSABLE_GAMES, VALID_WL,  # noqa: E402
                               clean_season, schedule_shape)

RAW_DIR = PIPELINE / "data" / "gleague" / "raw"

GAME_ID_WIDTH = 10
REGULAR_SEASON_TYPE_DIGIT = "2"
SHOWCASE_CUP_TYPE_DIGIT = "5"

# PLUS_MINUS and WL are deliberately absent. PLUS_MINUS is 100% null before
# 2005-06 and ~98% through 2006-07 - a field-availability boundary in the
# source, not corruption - and the builder recomputes it from PTS anyway, so
# its nullity cannot reach the output. WL is absent on 35 games and the margin
# is still exactly derivable from PTS there; requiring it would discard real
# games over a cross-check rather than over data.
REQUIRED_FIELDS = [
    "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_ID",
    "GAME_DATE", "MATCHUP", "PTS", "REB", "AST",
]

results = []


def check(season, name, ok, detail=""):
    results.append((season, name, ok))
    if not ok:
        print(f"    [FAIL] {name}")
        if detail:
            print(f"           {detail}")
    return ok


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def load(kind: str) -> dict:
    frames = {}
    for path in sorted(RAW_DIR.glob(f"gleague_{kind}_*.csv")):
        season = path.stem.split("_")[-1]
        frame = pd.read_csv(path, dtype={"GAME_ID": str})
        frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])
        frames[season] = frame
    return frames


def type_digit(frame: pd.DataFrame) -> pd.Series:
    return frame["GAME_ID"].astype(str).str.zfill(GAME_ID_WIDTH).str[2]


def validate_season(season: str, raw: pd.DataFrame, kind: str) -> dict:
    report = clean_season(raw)
    frame = report["clean"]

    dropped = []
    if report["unidentifiable_rows"]:
        dropped.append(f"{report['unidentifiable_rows']} unidentifiable row(s)")
    if report["partial_line_rows"]:
        dropped.append(f"{report['partial_line_rows']} partial line(s) summed")
    if report["unknown_wl_rows"]:
        dropped.append(f"{report['unknown_wl_rows']} unknown WL -> absent")
    if report["irreconcilable_games"]:
        dropped.append(f"{len(report['irreconcilable_games'])} "
                       f"irreconcilable game(s) "
                       f"{report['irreconcilable_games']}")
    if report["ambiguous_home_away_games"]:
        dropped.append(f"{len(report['ambiguous_home_away_games'])} "
                       f"ambiguous home/away game(s) "
                       f"{report['ambiguous_home_away_games']}")

    print(f"  {season}  ({kind}, {len(raw):,} raw -> {len(frame):,} clean)")
    if dropped:
        print(f"    classified: {'; '.join(dropped)}")

    # A game dropped for contradicting itself is only acceptable if it is one
    # of the three this corpus is known to contain. This check is what makes
    # the cleaning a classification of known defects rather than a filter that
    # silently swallows whatever arrives - a negative test found it swallowing
    # a deliberately corrupted game and leaving the run green.
    check(season, "every self-contradicting game is a named known one",
          not report["unlisted_drops"],
          f"unlisted: {report['unlisted_drops']}")

    missing_cols = [c for c in REQUIRED_FIELDS if c not in frame.columns]
    check(season, "all required fields present", not missing_cols,
          f"missing: {missing_cols}")

    nulls = {c: int(frame[c].isna().sum()) for c in REQUIRED_FIELDS
             if c in frame.columns}
    check(season, "no nulls in required fields after cleaning",
          all(v == 0 for v in nulls.values()),
          f"{ {c: v for c, v in nulls.items() if v} }")

    pairs = frame.groupby("GAME_ID").size()
    check(season, "every GAME_ID appears exactly twice after cleaning",
          bool((pairs == 2).all()),
          f"{int((pairs != 2).sum())} game(s): "
          f"{dict(pairs[pairs != 2].value_counts())}")

    digits = set(type_digit(frame).unique())
    wanted = (REGULAR_SEASON_TYPE_DIGIT if kind == "regular"
              else SHOWCASE_CUP_TYPE_DIGIT)
    check(season, f"every row carries type digit '{wanted}'",
          digits == {wanted}, f"found {sorted(digits)}")

    bad_wl = frame[frame["WL"].notna() & ~frame["WL"].isin(VALID_WL)]
    check(season, "WL is W, L or absent - never another value",
          not len(bad_wl), f"{sorted(bad_wl['WL'].unique())}")

    paired = frame[frame["GAME_ID"].isin(pairs[pairs == 2].index)]
    agree = paired.groupby("GAME_ID").agg(
        dates=("GAME_DATE", "nunique"),
        homes=("MATCHUP", lambda s: int(s.str.contains("vs.").sum())))
    check(season, "the two rows of a game agree on date",
          bool((agree["dates"] == 1).all()),
          f"{int((agree['dates'] != 1).sum())} game(s) disagree")
    check(season, "each game has exactly one home and one away",
          bool((agree["homes"] == 1).all()),
          f"{int((agree['homes'] != 1).sum())} game(s) wrong")

    def verdict(group):
        a, b = group.iloc[0], group.iloc[1]
        if not ({a["WL"], b["WL"]} <= VALID_WL):
            return "wl absent"
        winner = a if a["PTS"] > b["PTS"] else b
        return "ok" if a["PTS"] != b["PTS"] and winner["WL"] == "W" else "bad"

    v = paired.groupby("GAME_ID").apply(verdict, include_groups=False)
    counts = v.value_counts().to_dict()
    check(season, "WL agrees with PTS wherever WL is present",
          counts.get("bad", 0) == 0, f"{counts}")

    pm_null = (float(frame["PLUS_MINUS"].isna().mean())
               if "PLUS_MINUS" in frame else 1.0)

    # An explicit loop rather than groupby().apply(): a season whose
    # PLUS_MINUS is entirely null makes apply() return a DataFrame instead of
    # a Series, and the dtype surprise is not worth the brevity.
    offsets = []
    for _, group in paired.groupby("GAME_ID"):
        a, b = group.iloc[0], group.iloc[1]
        if pd.isna(a["PLUS_MINUS"]):
            continue
        offsets.append(abs(float(a["PLUS_MINUS"] - (a["PTS"] - b["PTS"]))))
    scored = offsets
    pm_wrong = sum(1 for o in offsets if o > 0)

    shape = schedule_shape(frame)
    print(f"    teams {shape['teams']:>3}  games {shape['games']:>5}  "
          f"per team {shape['min']}-{shape['max']} (mode {shape['modal']})  "
          f"[{shape['verdict']}]")

    extra = []
    if counts.get("wl absent"):
        extra.append(f"WL absent on {counts['wl absent']} game(s), margin "
                     f"still derivable from PTS")
    if pm_null:
        extra.append(f"PLUS_MINUS null on {pm_null * 100:.0f}% of rows")
    if pm_wrong:
        extra.append(f"PLUS_MINUS wrong on {pm_wrong} of {len(scored)}")
    if extra:
        print(f"    {'; '.join(extra)}")

    check(season, "row-count identity holds (every game has two sides)",
          shape["identity_holds"],
          "per-team game counts do not sum to twice the game count")

    if shape["verdict"] == "unbalanced":
        print(f"    COMPLETENESS RULE INAPPLICABLE: {shape['detail']}")
    else:
        check(season, f"completeness ({shape['verdict']})", shape["ok"],
              shape["detail"])

    return {"season": season, "pm_null": pm_null, "pm_wrong": pm_wrong,
            "pm_scored": len(scored),
            "wl_absent": counts.get("wl absent", 0),
            "raw": len(raw), "clean": len(frame), **shape}


def report_team_identity(frames: dict) -> None:
    section("FRANCHISE IDENTITY, KEYED ON TEAM_ID")

    within, clashes = [], []
    history = defaultdict(list)
    first_seen = {}

    for season in sorted(frames):
        frame = clean_season(frames[season])["clean"]
        per_id = frame.groupby("TEAM_ID").agg(
            abbrs=("TEAM_ABBREVIATION", "nunique"),
            names=("TEAM_NAME", "nunique"))
        bad = per_id[(per_id["abbrs"] > 1) | (per_id["names"] > 1)]
        if len(bad):
            within.append((season, bad.index.tolist()))

        pairs = frame.groupby("TEAM_ID").agg(
            abbr=("TEAM_ABBREVIATION", "first"),
            name=("TEAM_NAME", "first"))
        for tid, row in pairs.iterrows():
            history[int(tid)].append((season, row["abbr"], row["name"]))
            first_seen.setdefault(int(tid), season)

        by_abbr = pairs.reset_index().groupby("abbr")["TEAM_ID"].nunique()
        if len(by_abbr[by_abbr > 1]):
            clashes.append((season, by_abbr[by_abbr > 1].to_dict()))

    check("all", "each TEAM_ID is internally consistent within a season",
          not within, f"{within[:3]}")
    check("all", "no abbreviation shared by two TEAM_IDs in one season",
          not clashes, f"{clashes[:3]}")

    print(f"\n  distinct TEAM_IDs across all {len(frames)} seasons: "
          f"{len(history)}")

    print("\n  IDENTITY CHANGES, per id, with the season of the change")
    print("  Phase 2's Elo depends on these being ONE franchise, not two.\n")
    changed = 0
    for tid in sorted(history):
        rows = history[tid]
        moves = [f"{rows[i][0]}: {rows[i-1][1]}/{rows[i-1][2]} -> "
                 f"{rows[i][1]}/{rows[i][2]}"
                 for i in range(1, len(rows))
                 if (rows[i-1][1], rows[i-1][2]) != (rows[i][1], rows[i][2])]
        if moves:
            changed += 1
            print(f"    {tid}  ({rows[0][0]} .. {rows[-1][0]})")
            for m in moves:
                print(f"      {m}")
    print(f"\n  {changed} of {len(history)} ids changed identity at least once")

    print("\n  FIRST APPEARANCES, per season")
    print("  The expansion record phase 2 needs to fit a starting rating.\n")
    by_season = defaultdict(list)
    for tid, season in first_seen.items():
        by_season[season].append(tid)
    for season in sorted(by_season):
        ids = sorted(by_season[season])
        print(f"    {season}: {len(ids):>2} new"
              f"{('  ' + str(ids)) if len(ids) <= 5 else ''}")


def assert_id_ranges_disjoint(frames: dict) -> None:
    section("TEAM_ID RANGES ACROSS THE THREE LEAGUES")
    print("""Phases 4 and 5 share one team table and one id space. Team.id is a primary
key, so a collision would be one league's row overwriting another's.
""")
    gl, abbrs = set(), set()
    for frame in frames.values():
        clean = clean_season(frame)["clean"]
        gl |= set(clean["TEAM_ID"].astype(int))
        abbrs |= set(clean["TEAM_ABBREVIATION"].unique())
    print(f"  G League: {len(gl)} ids, {min(gl)}-{max(gl)}, "
          f"{len(abbrs)} abbreviations")

    for name, rel in (
            ("NBA", Path("data") / "processed" / "games_final.csv"),
            ("WNBA", Path("data") / "wnba" / "processed"
             / "wnba_games_final.csv")):
        path = PIPELINE / rel
        if not path.exists():
            print(f"  {name}: {path.name} absent, cannot compare")
            continue
        other = pd.read_csv(path)
        ids = set(other["TEAM_ID"])
        overlap = gl & ids
        print(f"  {name:<8}: {len(ids)} ids, {min(ids)}-{max(ids)}  "
              f"overlap {len(overlap)}")
        check("all", f"G League ids disjoint from the {name}'s", not overlap,
              f"shared: {sorted(overlap)[:8]}")
        if "TEAM_ABBREVIATION" in other.columns:
            shared = sorted(abbrs & set(other["TEAM_ABBREVIATION"]))
            print(f"            abbreviations also used by the {name}: "
                  f"{len(shared)}  {shared[:12]}")

    print("""
  Abbreviation collisions are EXPECTED and are a phase-5 note, not a defect
  here: eight WNBA abbreviations already collide with NBA ones, which is why
  the frontend resolves teams by id and never by short code.""")


def assert_exclusions_are_live(frames: dict) -> None:
    """Every named unusable game must still be present and still unusable.

    The other half of the strictness. The per-season check stops a NEW
    self-contradicting game from being absorbed; this one stops a DEAD entry
    from sitting in the list forever, excusing a game that no longer needs
    excusing. Without it the set would be exactly the expiring expectation
    this project has had to correct twice - the quarter/half id list and the
    stale reference number.
    """
    section("THE NAMED EXCLUSIONS ARE ALL STILL LIVE")

    present = {}
    for season, raw in frames.items():
        for game_id in raw["GAME_ID"].astype(str):
            present[game_id] = season

    still_dropped = set()
    for season, raw in frames.items():
        report = clean_season(raw)
        for game_id in (list(report["irreconcilable_games"])
                        + list(report["ambiguous_home_away_games"])):
            still_dropped.add(str(game_id))

    for game_id, reason in KNOWN_UNUSABLE_GAMES.items():
        where = present.get(game_id)
        check("exclusions", f"{game_id} is still in the corpus",
              where is not None,
              "listed as unusable but no longer present in any season file")
        check("exclusions", f"{game_id} is still unusable",
              game_id in still_dropped,
              "listed as unusable but the cleaner no longer drops it, so "
              "this entry is stale and excuses nothing")
        if where:
            print(f"  {game_id}  {where}  {reason}")


def main() -> int:
    print(__doc__)

    regular = load("regular")
    showcase = load("showcase")
    if not regular:
        print(f"No regular-season files in {RAW_DIR}. Run the fetcher first.")
        return 1

    section(f"REGULAR SEASON - {len(regular)} SEASONS")
    summaries = {s: validate_season(s, regular[s], "regular")
                 for s in sorted(regular)}

    section(f"SHOWCASE CUP - {len(showcase)} SEASONS")
    print("Stored apart from the regular season; phase 2 needs these for rest "
          "days.\n")
    cup = {s: validate_season(s, showcase[s], "showcase")
           for s in sorted(showcase)}

    report_team_identity(regular)
    assert_id_ranges_disjoint(regular)
    assert_exclusions_are_live(regular)

    section("PER-SEASON SUMMARY")
    print(f"  {'season':<9}{'raw':>7}{'clean':>7}{'games':>7}{'shape':>12}"
          f"{'PM null':>9}{'PM wrong':>9}{'WL absent':>10}")
    print("  " + "-" * 70)
    for s in sorted(summaries):
        v = summaries[s]
        print(f"  {s:<9}{v['raw']:>7}{v['clean']:>7}{v['games']:>7}"
              f"{v['verdict']:>12}{v['pm_null'] * 100:>8.0f}%"
              f"{v['pm_wrong']:>9}{v['wl_absent']:>10}")

    wrong = sum(v["pm_wrong"] for v in summaries.values())
    scored = sum(v["pm_scored"] for v in summaries.values())
    section("PLUS_MINUS, MEASURED FOR THIS LEAGUE")
    print(f"  disagrees with PTS on {wrong:,} of {scored:,} games where it is "
          f"populated ({wrong / scored * 100:.2f}%)")
    print("""
  AN ORDER OF MAGNITUDE WORSE THAN THE OTHER TWO LEAGUES, which is itself the
  finding: 1.02% of NBA games and 1.24% of WNBA ones, same endpoint, same
  defect. It does not reach the output - the builder recomputes the margin
  from PTS and refuses if the derived sign disagrees with WL - but it is the
  clearest evidence yet that this column should never be trusted from
  LeagueGameFinder in any league.""")

    section("RESULT")
    failed = [(s, n) for s, n, ok in results if not ok]
    print(f"  {sum(1 for _, _, ok in results if ok)} checks passed, "
          f"{len(failed)} failed")
    unbalanced = [s for s, v in summaries.items()
                  if v["verdict"] == "unbalanced"]
    if unbalanced:
        print(f"\n  completeness rule inapplicable to {len(unbalanced)} "
              f"season(s): {unbalanced}")
        print("  Derived from the distribution, not named in a list. Each is "
              "reported\n  above with the evidence, and the row-count "
              "identity is asserted on all.")
    if failed:
        print("\n  FAILED:")
        for s, n in failed:
            print(f"    {s}: {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
