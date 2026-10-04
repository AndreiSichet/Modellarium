"""Prove the live WNBA feature module reproduces the pipeline's own columns.

The live module computes rolling windows, rest days and Elo from the raw long
table. The pipeline computed the same quantities into
wnba_games_final_features.csv by a different route - vectorised groupby
instead of per-team tails, and a single full Elo pass instead of a replay plus
a carryover step. Agreement across sampled historical games is what makes the
serving path trustworthy, and it is the same argument NBA live_features.py
makes by reproducing 200 games to 1e-13.

Elo is compared at FIXED parameters, taken from the manifest and applied to
both sides. Otherwise this would be measuring the difference between two
parameter fits rather than the difference between two implementations - and
the pipeline's stored column was fitted on 2015-2023 while the manifest's were
fitted on everything, so those genuinely differ.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from live_wnba_features import (  # noqa: E402
    GAMES_PATH, METRICS, MODELS_DIR, NotScoreable, PIPELINE_FEATURES_PATH,
    TEAM_KEY, elo_for, get_live_features, load_games, load_manifest,
    replay_elo, required_windows, rest_for, rolling_for, season_of,
    split_window, team_history,
)
from build_wnba_elo import prepare, run_elo  # noqa: E402


def pipeline_elo(frame, k, carryover):
    """run_elo's own per-row pre-game rating, at the given parameters.

    Built here from the pipeline's functions rather than by adding a helper to
    build_wnba_elo: this is a harness convenience, and a pipeline script
    should not grow an entry point that only a test calls.
    """
    team_elo, _, _, _ = run_elo(prepare(frame), k, carryover)
    return pd.Series(team_elo)

SAMPLE_SIZE = 150
SEED = 42
TOLERANCE = 1e-9

results = []


def record(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"         {detail}")
    results.append(ok)


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    print(__doc__)

    manifest = load_manifest()
    frame = load_games()
    windows = required_windows(manifest)
    k = manifest["elo"]["k"]
    carryover = manifest["elo"]["carryover"]

    print(f"  served table : {len(frame):,} rows, "
          f"{frame['GAME_ID'].nunique():,} games")
    print(f"  windows      : {windows}  (from the manifest, not hardcoded)")
    print(f"  elo          : K={k}, carryover={carryover}")

    if not PIPELINE_FEATURES_PATH.exists():
        print(f"\n  {PIPELINE_FEATURES_PATH.name} is absent. It is not in the "
              "inference image\n  by design, so this harness only runs on a "
              "host with the pipeline output.")
        return 1

    pipeline = pd.read_csv(PIPELINE_FEATURES_PATH, dtype={"GAME_ID": str})
    pipeline["GAME_DATE"] = pd.to_datetime(pipeline["GAME_DATE"])

    section("1. ROLLING WINDOWS AND REST, AGAINST THE PIPELINE'S COLUMNS")
    rng = np.random.default_rng(SEED)
    # Sampled from rows the pipeline itself could complete, since a NaN on
    # both sides proves nothing about agreement.
    checkable = pipeline.dropna(
        subset=[f"{w}_{m}" for w in windows for m in METRICS.values()]
        + ["REST_DAYS"])
    picks = checkable.iloc[rng.choice(len(checkable),
                                      size=min(SAMPLE_SIZE, len(checkable)),
                                      replace=False)]

    worst = {"rolling": 0.0, "rest": 0.0}
    flag_mismatches = 0
    for _, row in picks.iterrows():
        history = team_history(frame, row[TEAM_KEY], row["GAME_DATE"])
        season = int(row["SEASON"])

        for window in windows:
            mine = rolling_for(history, window, season)
            for label in METRICS.values():
                column = f"{window}_{label}"
                gap = abs(mine[column] - row[column])
                worst["rolling"] = max(worst["rolling"], gap)

        mine = rest_for(history, row["GAME_DATE"], season)
        worst["rest"] = max(worst["rest"],
                            abs(mine["REST_DAYS"] - row["REST_DAYS"]))
        for flag in ("IS_LONG_BREAK", "IS_BACK_TO_BACK"):
            if bool(mine[flag]) != bool(row[flag]):
                flag_mismatches += 1

    record(f"rolling windows match on {len(picks)} sampled games",
           worst["rolling"] < TOLERANCE,
           f"largest difference {worst['rolling']:.3e} across "
           f"{len(windows) * len(METRICS)} columns per game")
    record("REST_DAYS matches", worst["rest"] < TOLERANCE,
           f"largest difference {worst['rest']:.3e}")
    record("IS_LONG_BREAK and IS_BACK_TO_BACK match",
           flag_mismatches == 0, f"{flag_mismatches} mismatching flag(s)")

    section("2. THE SEASON OPENER IS NaN, NOT THE CAP")
    openers = pipeline.sort_values("GAME_DATE").groupby(
        [TEAM_KEY, "SEASON"]).head(1)
    mine_nan = 0
    for _, row in openers.iterrows():
        history = team_history(frame, row[TEAM_KEY], row["GAME_DATE"])
        if np.isnan(rest_for(history, row["GAME_DATE"],
                             int(row["SEASON"]))["REST_DAYS"]):
            mine_nan += 1
    record(f"all {len(openers)} team-season openers are NaN on REST_DAYS",
           mine_nan == len(openers),
           f"{mine_nan} of {len(openers)}; filling these with the 7-day cap "
           "would make a seven-month offseason look like a week of rest")

    section("3. ELO, AT FIXED PARAMETERS, INCLUDING ACROSS A SEASON BOUNDARY")
    print("""The live module replays every game and then applies the
between-season carryover itself, because run_elo keeps that step inside a
closure. This compares it against run_elo's own output at the same K and
carryover, which is the only check that formula has.
""")
    reference = pipeline_elo(frame, k, carryover)

    # Replay only games before a cut, then ask the live module for the rating
    # it would serve - and compare against the full pass's stored pre-game
    # value for that same game.
    boundary_checked = 0
    worst_elo = 0.0
    ordered = frame.sort_values(["GAME_DATE", "GAME_ID"])
    cut_games = ordered["GAME_ID"].drop_duplicates().tolist()
    rng = np.random.default_rng(SEED + 1)

    # Random games hit the carryover branch only rarely - the first pass over
    # 40 of them exercised it twice. Since that branch is the ONE restated
    # formula in the serving path, every season's opening fixtures are added
    # deliberately rather than left to the sample.
    first_of_season = (ordered.groupby("SEASON")["GAME_ID"]
                       .apply(lambda s: s.drop_duplicates().head(3)).tolist())
    probes = list(dict.fromkeys(
        first_of_season
        + list(rng.choice(cut_games[200:], size=40, replace=False))))

    for game_id in probes:
        rows = ordered[ordered["GAME_ID"] == game_id]
        when = rows["GAME_DATE"].iloc[0]
        season = int(rows["SEASON"].iloc[0])
        history = ordered[ordered["GAME_DATE"] < when]
        if history.empty:
            continue
        state = replay_elo(history, k, carryover)
        for _, row in rows.iterrows():
            mine = elo_for(state, row[TEAM_KEY], season)
            theirs = reference.loc[row.name]
            worst_elo = max(worst_elo, abs(mine - theirs))
            if state["last_season"].get(row[TEAM_KEY]) != season:
                boundary_checked += 1

    record("replayed Elo matches run_elo's own pre-game rating",
           worst_elo < 1e-6, f"largest difference {worst_elo:.3e} over "
           f"{len(probes)} sampled games, both teams")
    record("and the carryover branch was actually exercised",
           boundary_checked > 0,
           f"{boundary_checked} team-game(s) crossed a season boundary - if "
           "this were 0 the restated formula would be untested")

    # POSITIVE CONTROL. The agreement above is EXACTLY zero, which is the
    # right answer for deterministic arithmetic and also what a check
    # comparing a value against itself would report. Serving the same probes
    # under a wrong carryover must move it; if it does not, the comparison is
    # not reaching the carryover at all.
    wrong = 1.0 - carryover if carryover != 0.5 else 0.0
    moved = 0.0
    for game_id in first_of_season:
        rows = ordered[ordered["GAME_ID"] == game_id]
        when, season = rows["GAME_DATE"].iloc[0], int(rows["SEASON"].iloc[0])
        history = ordered[ordered["GAME_DATE"] < when]
        if history.empty:
            continue
        bad = replay_elo(history, k, wrong)
        for _, row in rows.iterrows():
            if bad["last_season"].get(row[TEAM_KEY]) != season:
                moved = max(moved, abs(elo_for(bad, row[TEAM_KEY], season)
                                       - reference.loc[row.name]))
    record(f"a wrong carryover ({wrong}) DOES move the comparison",
           moved > 1.0,
           f"largest difference {moved:.3f} rating points - so the zero above "
           "is agreement, not a comparison against itself")

    section("4. THE MANIFEST DRIVES THE FEATURE ORDER")
    for target, entry in manifest["targets"].items():
        rebuilt = []
        prefix, _ = split_window(entry["window"])
        for side in ("HOME", "AWAY"):
            rebuilt += [f"{side}_{entry['window']}_{m}"
                        for m in METRICS.values()]
        record(f"{target}: the manifest's first 16 columns are this window's "
               "rolling set",
               entry["features"][:16] == rebuilt,
               f"window {entry['window']} (scope {prefix})")

    section("5. A FIXTURE THAT CANNOT BE SCORED IS REFUSED, NOT FILLED")
    # A franchise's very first game has no history under any window.
    first_season = int(frame["SEASON"].min())
    debut = frame[frame["SEASON"] == first_season].sort_values("GAME_DATE")
    pair = debut.iloc[:2]
    refused = False
    try:
        get_live_features(int(pair.iloc[0][TEAM_KEY]),
                          int(pair.iloc[1][TEAM_KEY]),
                          pair.iloc[0]["GAME_DATE"])
    except NotScoreable as exc:
        refused = True
        detail = str(exc)[:120]
    record("the first game in the corpus raises NotScoreable", refused,
           detail if refused else "it returned a row instead of refusing")

    section("6. A COMPLETE FIXTURE PRODUCES EVERY DECLARED FEATURE")
    latest = frame["GAME_DATE"].max()
    recent = frame[frame["GAME_DATE"] == latest]
    try:
        out = get_live_features(int(recent.iloc[0][TEAM_KEY]),
                                int(recent.iloc[1][TEAM_KEY]),
                                latest + pd.Timedelta(days=1))
        ok = all(len(out["rows"][t]) == len(manifest["targets"][t]["features"])
                 and not any(np.isnan(v) for v in out["rows"][t])
                 for t in manifest["targets"])
        detail = ", ".join(f"{t}={len(v)} values" for t, v in out["rows"].items())
    except NotScoreable as exc:
        ok, detail = False, str(exc)[:120]
    record("the day after the newest game scores on all three targets", ok,
           detail)

    section("RESULT")
    print(f"  {sum(results)} of {len(results)} checks passed")
    print(f"\n  served table ends {latest.date()}, so the only scoreable date "
          f"is {(latest + pd.Timedelta(days=1)).date()}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
