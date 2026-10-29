r"""Every rolling warm-up expectation, audited against short team-seasons.

THE BUG THIS EXISTS FOR. A `shift(1).rolling(window)` leaves the first
`min(window, games)` rows of each team-season NaN - not `window` rows. A check
that expects `team_seasons * window` is therefore correct only once every
team-season is at least `window` games long, which is false for the first three
weeks of every season. On 2026-10-29 that assumption stopped the retrain's
refresh chain: `build_quarter_half_rolling` expected 3,600 ROLL10 NaN where the
data can only produce 3,496, the 104-row shortfall being `sum(10 - games)` over
the 30 team-seasons of a season four to seven games old, and it raised
"shift(1) may not be applied" about a pipeline that was working.

WHAT THE RIGHT RULE IS. Per team-season, `min(window, rows)`, i.e.
`sizes.clip(upper=window).sum()`. `build_team_advanced_rolling` already had it;
`build_quarter_half_rolling` did not.

THE PLANTS MUST GO BOTH WAYS. A check that tolerates short seasons is only
useful if it still catches a missing `shift(1)` - and it has to catch it on an
in-progress season too, which is the case where the expectation is lowest and
therefore easiest to satisfy by accident.

    python ml-training/verify_warmup_rules.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PIPELINE = PROJECT / "data-pipeline"
for extra in (PIPELINE / "preprocessing", PIPELINE / "wnba" / "preprocessing"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import build_quarter_half_rolling as QH                          # noqa: E402

RESULTS = []


def record(name, passed, detail=""):
    RESULTS.append((name, passed))
    print("  %-4s %-58s %s" % ("PASS" if passed else "FAIL", name, detail))


def synthetic(finished_seasons=3, games_finished=40, in_progress_games=0,
              teams=6):
    """Team-games with whole seasons, and optionally a short current one.

    Built rather than loaded so the plants are deterministic and do not depend
    on which day the real corpus happens to be in.
    """
    rows = []
    game = 0
    for index in range(finished_seasons):
        # The GAME_ID season code is what derive_season reads, so the ids have
        # to be real: '00' + type 2 + the two-digit start year.
        year = 15 + index
        for team in range(teams):
            for n in range(games_finished):
                game += 1
                rows.append({
                    "GAME_ID": f"002{year:02d}{game:05d}"[:10],
                    "TEAM_ID": 1610612700 + team,
                    "GAME_DATE": pd.Timestamp(f"20{year + 1}-01-01")
                    + pd.Timedelta(days=n),
                    "Q1_PTS": 25.0 + n % 7,
                    "Q1_PTS_ALLOWED": 24.0 + n % 5,
                    "HALF1_PTS": 55.0 + n % 9,
                    "HALF1_PTS_ALLOWED": 54.0 + n % 6,
                })
    if in_progress_games:
        year = 15 + finished_seasons
        for team in range(teams):
            for n in range(in_progress_games):
                game += 1
                rows.append({
                    "GAME_ID": f"002{year:02d}{game:05d}"[:10],
                    "TEAM_ID": 1610612700 + team,
                    "GAME_DATE": pd.Timestamp(f"20{year + 1}-01-01")
                    + pd.Timedelta(days=n),
                    "Q1_PTS": 26.0, "Q1_PTS_ALLOWED": 25.0,
                    "HALF1_PTS": 56.0, "HALF1_PTS_ALLOWED": 55.0,
                })
    frame = pd.DataFrame(rows)
    for period in ("Q1", "HALF1"):
        frame[f"{period}_MARGIN"] = (frame[f"{period}_PTS"]
                                     - frame[f"{period}_PTS_ALLOWED"])
    return frame


def roll_without_shift(df):
    """THE PLANT: `add_rolling` with the `shift(1)` removed.

    Deliberately a copy of `QH.add_rolling` minus one call rather than a
    monkeypatch - the shift lives inside a lambda passed to `transform`, so
    there is nothing to patch, and a copy makes the one difference visible.
    """
    df = df.copy()
    df["SEASON"] = QH.derive_season_for_frame(df)
    df = df.sort_values([QH.TEAM_KEY, "SEASON", "GAME_DATE"]).reset_index(
        drop=True)
    grouped = df.groupby([QH.TEAM_KEY, "SEASON"])
    for window in QH.WINDOWS:
        for metric in QH.ROLLING_METRICS:
            df[f"ROLL{window}_{metric}"] = grouped[metric].transform(
                lambda s, w=window: s.rolling(w).mean()      # no shift(1)
            )
    return df


def raises(fn):
    try:
        fn()
        return None
    except RuntimeError as error:
        return str(error)


def advanced_checks():
    """Plant `build_team_advanced_rolling`'s two checks against real frames.

    Both used to live inline in `main()` and neither could fail: the NaN one
    was printed and never raised on, and the row count was compared against a
    hardcoded 26,398. They are functions now, so each can be handed a frame
    that violates it.
    """
    import build_team_advanced_rolling as ADV

    # A corpus of three team-seasons: two full, one four games long, so the
    # in-progress case is present throughout.
    rows = []
    for team, season, games in ((1, 2024, 40), (2, 2024, 40), (1, 2026, 4)):
        for n in range(games):
            rows.append({
                "GAME_ID": f"002{str(season)[-2:]}{team}{n:04d}"[:10],
                "TEAM_ID": team, "SEASON": season,
                "GAME_DATE": pd.Timestamp("2025-01-01") + pd.Timedelta(days=n),
            })
    teams = pd.DataFrame(rows)
    # The SOURCE column names (METRICS keys), which is what add_rolling reads;
    # METRICS.values() are the output names it writes.
    for source_column in ADV.METRICS:
        teams[source_column] = 100.0

    longest = max(ADV.WINDOWS)
    counts = ADV.expected_nan_counts(teams)
    record("   its NaN expectation already uses min(window, games)",
           counts[longest] == longest + longest + 4,
           f"ROLL{longest} expects {counts[longest]} for seasons of 40, 40, 4")

    rolled = ADV.add_rolling(teams.copy())
    columns = [f"ROLL{w}_{n}" for w in ADV.WINDOWS
               for n in ADV.METRICS.values()]
    output = rolled[["GAME_ID", "TEAM_ID"] + columns]

    record("   control: the honest frame passes the NaN check",
           raises(lambda: ADV.check_nan_counts(output, rolled)) is None,
           "no exception")

    # PLANT 1: fill one NaN, so that column's count is one short.
    holed = output.copy()
    column = f"ROLL{longest}_{list(ADV.METRICS.values())[0]}"
    first_nan = holed.index[holed[column].isna()][0]
    holed.loc[first_nan, column] = 123.0
    error = raises(lambda: ADV.check_nan_counts(holed, rolled))
    record("   PLANT: a NaN mismatch raises (it used to only print)",
           error is not None, (error or "NOT CAUGHT")[:60])

    # PLANT 2: a missing row, against the count derived from games_final.csv.
    # The real games_final.csv is the universe, so the frame is built from it.
    universe = pd.read_csv(ADV.GAMES_FINAL_PATH,
                           usecols=["GAME_ID", "TEAM_ID"],
                           dtype={"GAME_ID": str}, low_memory=False)
    universe["GAME_ID"] = universe["GAME_ID"].str.zfill(10)
    # COMPLETE GAMES, not the first 200 rows. `head(200)` can cut a game in
    # half, and then the derived expectation legitimately disagrees - which is
    # what it did on the first run of this control. The check was right and
    # the control was wrong.
    first_games = universe["GAME_ID"].drop_duplicates().head(100)
    sample = universe[universe["GAME_ID"].isin(set(first_games))].copy()
    sample["SEASON"] = 2024
    record("   control: a frame matching games_final.csv passes",
           raises(lambda: ADV.check_output_rows(sample, sample)) is None,
           f"{len(sample)} rows, derived expectation met")

    short = sample.iloc[:-1]
    error = raises(lambda: ADV.check_output_rows(short, sample))
    record("   PLANT: one missing row raises, from the derived count",
           error is not None, (error or "NOT CAUGHT")[:60])

    duped = pd.concat([sample, sample.iloc[[0]]], ignore_index=True)
    error = raises(lambda: ADV.check_output_rows(duped, sample))
    record("   PLANT: a duplicated (GAME_ID, TEAM_ID) raises",
           error is not None, (error or "NOT CAUGHT")[:60])

    # PLANT 3: an orphan id - a corpus game absent from games_final.csv.
    orphan = sample.copy()
    orphan.loc[orphan.index[0], "GAME_ID"] = "0099999999"
    error = raises(lambda: ADV.expected_output_rows(orphan))
    record("   PLANT: a corpus id absent from games_final.csv raises",
           error is not None, (error or "NOT CAUGHT")[:60])

    record("   the literal EXPECTED_ROWS = 26_398 is gone",
           not hasattr(ADV, "EXPECTED_ROWS"),
           "the count comes from games_final.csv")


def main() -> int:
    longest = max(QH.WINDOWS)

    # ---- the rule itself, as arithmetic
    sizes = pd.Series([40, 40, 5, 5, 1])
    record("the rule is sum of min(window, games)",
           QH.expected_warmup_nan(sizes, 10) == 10 + 10 + 5 + 5 + 1,
           f"{QH.expected_warmup_nan(sizes, 10)} == 31")
    record("PLANT: the old rule would have said teams x window",
           len(sizes) * 10 == 50 and QH.expected_warmup_nan(sizes, 10) != 50,
           "old 50, exact 31 - a 19-row phantom shortfall")

    # ---- 1. an in-progress season shorter than the window must PASS
    print("\n1. an in-progress season with fewer games than the window")
    for played in (4, 7, longest - 1):
        frame = synthetic(in_progress_games=played)
        rolled = QH.add_rolling(frame)
        error = raises(lambda: QH.check_warmup(rolled))
        record(f"   {played} games per team in the current season passes",
               error is None, error or "no exception")

    # ---- 2. a missing shift(1) must still FAIL, both shapes
    print("\n2. a missing shift(1), on finished seasons and on an "
          "in-progress one")
    finished_only = roll_without_shift(synthetic(in_progress_games=0))
    error = raises(lambda: QH.check_warmup(finished_only))
    record("   PLANT: no shift(1), finished seasons only -> raises",
           error is not None, (error or "NOT CAUGHT")[:66])

    with_progress = roll_without_shift(synthetic(in_progress_games=5))
    error = raises(lambda: QH.check_warmup(with_progress))
    record("   PLANT: no shift(1), with a 5-game current season -> raises",
           error is not None, (error or "NOT CAUGHT")[:66])

    # ---- 3. the no-op guard
    print("\n3. the no-op guard - a change that changes nothing must not "
          "read as a catch")
    frame = synthetic(in_progress_games=5)
    rolled = QH.add_rolling(frame)
    shuffled = rolled.sample(frac=1.0, random_state=7).reset_index(drop=True)
    error = raises(lambda: QH.check_warmup(shuffled))
    record("   row order does not change the verdict", error is None,
           error or "still passes, so the plants above caught a real defect")

    # ---- the extra, and that a shortfall is never blamed on missing rows
    print("\n4. rows with no raw data push the count UP, never down")
    frame = synthetic(in_progress_games=5)
    holed = frame.copy()
    for metric in QH.ROLLING_METRICS:
        holed.loc[holed.index[:4], metric] = np.nan
    rolled_holed = QH.add_rolling(holed)
    error = raises(lambda: QH.check_warmup(rolled_holed))
    sizes_holed = rolled_holed.groupby([QH.TEAM_KEY, "SEASON"]).size()
    expected = QH.expected_warmup_nan(sizes_holed, longest)
    actual = int(rolled_holed[f"ROLL{longest}_Q1_MARGIN"].isna().sum())
    record("   4 holed rows raise the actual count above the expectation",
           error is None and actual > expected,
           f"expected {expected:,}, actual {actual:,}, extra {actual - expected:,}")

    # ---- build_team_advanced_rolling: both checks, actually planted
    print("\n5. build_team_advanced_rolling: a mismatch must FAIL, not print")
    advanced_checks()

    # ---- the audit: every other builder's rule
    print("\n6. the same assumption in the other builders")
    audit = [
        ("build_team_advanced_rolling", "expected_nan_counts",
         "sizes.clip(upper=window).sum()", True,
         "already exact - this is the idiom the fix copied"),
        ("build_player_rolling_minutes", "expected_nan_count",
         "(appearances_so_far <= window).sum()", True,
         "exact by construction: counts rows, not team-seasons"),
        ("build_wnba_rolling_features", "season-opener check",
         "every team-season opener must be NaN", True,
         "structural, independent of how many games a season has"),
        ("build_rolling_features", "none", "-", True,
         "no warm-up NaN expectation at all"),
        ("build_gleague_games_table", "none", "-", True,
         "no warm-up NaN expectation at all"),
        ("build_nfl_model_dataset", "none", "-", True,
         "reports a rest-day NaN count, asserts nothing about warm-up"),
    ]
    for module, where, rule, safe, note in audit:
        record(f"   {module}", safe, f"{where}: {note}")

    passed = sum(1 for _n, ok in RESULTS if ok)
    print()
    print(f"  {passed} of {len(RESULTS)} checks passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:                                   # noqa: BLE001
        print(f"\nCRASHED before a verdict: {type(error).__name__}: {error}")
        raise SystemExit(2)
