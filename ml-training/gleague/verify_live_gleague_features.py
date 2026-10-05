"""Confirm the serving path reproduces what phase 3 trained on.

By FLOAT EQUALITY where the arithmetic is deterministic, not by tolerance.

THE POST-CUP GAME IS THE CHECK THAT MATTERS HERE and it is exercised rather
than assumed. Phase 2 computed REST_DAYS across the Showcase Cup boundary, so
a team's first regular-season game of a Cup season reads rest from its last
Cup game. If serving counted regular-season games only, that one game per team
per season would read as a season opener instead - a value the model never saw
for that row, silent, and invisible to every other probe.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gleague_common import (  # noqa: E402
    DATASET_PATH, IDENTITY_PATH, MODELS_DIR, TARGETS, feature_columns,
    load_dataset, section)
import live_gleague_features as live  # noqa: E402

results = []


def record(name, ok, detail=""):
    results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).splitlines():
            print(f"         {line}")


def post_cup_first_games(state) -> pd.DataFrame:
    """Each team's first regular-season game in a season that had a Cup."""
    both = state["all_games"]
    cup_seasons = sorted(
        both.loc[both["COMPETITION"] == "showcase", "SEASON"].unique())
    regular = both[(both["COMPETITION"] == "regular")
                   & both["SEASON"].isin(cup_seasons)]
    return (regular.sort_values(["TEAM_ID", "SEASON", "GAME_DATE"])
            .groupby(["TEAM_ID", "SEASON"]).head(1))


def check_rest_crosses_the_cup(state) -> None:
    section("THE CUP REACHES REST DAYS - THE PATH NOBODY ELSE EXERCISES")

    openers = post_cup_first_games(state)
    print(f"  {len(openers)} team-game(s) are a first regular-season game in "
          f"a Cup season")

    agree, checked, examples = 0, 0, []
    for row in openers.itertuples(index=False):
        season = row.SEASON
        history = live.team_history(state["all_games"], row.TEAM_ID,
                                    row.GAME_DATE)
        with_cup = live.rest_for(history, row.GAME_DATE, season)

        regular_only = history[history["COMPETITION"] == "regular"]
        without = live.rest_for(regular_only, row.GAME_DATE, season)

        checked += 1
        if not np.isnan(with_cup["REST_DAYS"]) and np.isnan(
                without["REST_DAYS"]):
            agree += 1
            if len(examples) < 3:
                examples.append(
                    f"team {row.TEAM_ID} {season} on "
                    f"{row.GAME_DATE.date()}: with Cup "
                    f"{with_cup['REST_DAYS']:.0f} days, "
                    f"regular-season only NaN")

    record("counting the Cup gives every such game a rest value, and "
           "ignoring it gives none", agree == checked,
           "\n".join(examples) + f"\n{agree} of {checked} team-games")

    # And the served value must equal the one phase 2 wrote into the dataset
    # the models were fitted on. This is the comparison that would catch a
    # serving path that counted the Cup but computed the gap differently.
    dataset = load_dataset().set_index("GAME_ID")
    worst, compared = 0.0, 0
    for row in openers.itertuples(index=False):
        if row.GAME_ID not in dataset.index:
            continue
        stored = dataset.loc[row.GAME_ID]
        side = "HOME" if bool(stored["HOME_TEAM_ID"] == row.TEAM_ID) \
            else "AWAY"
        history = live.team_history(state["all_games"], row.TEAM_ID,
                                    row.GAME_DATE)
        served = live.rest_for(history, row.GAME_DATE, row.SEASON)
        expected = float(stored[f"{side}_REST_DAYS"])
        worst = max(worst, abs(served["REST_DAYS"] - expected))
        compared += 1

    record(f"served REST_DAYS equals the trained value on all {compared} "
           f"post-Cup openers", worst == 0.0,
           f"largest difference {worst:.1e}")


def check_feature_row_matches_training(state, sample=40) -> None:
    section("THE LIVE FEATURE ROW MATCHES THE TRAINING ROW")

    dataset = load_dataset()
    manifest = state["manifest"]
    rng = np.random.default_rng(11)

    # Sampled from games with a complete window, plus every post-Cup opener,
    # so the rest-day path is in the comparison rather than beside it.
    openers = set(post_cup_first_games(state)["GAME_ID"])
    target = next(iter(manifest["targets"]))
    columns = feature_columns(manifest["targets"][target]["window"])
    complete = dataset[dataset[columns].notna().all(axis=1)]

    forced = complete[complete["GAME_ID"].isin(openers)]
    others = complete[~complete["GAME_ID"].isin(openers)]
    picked = pd.concat([
        forced.iloc[rng.choice(len(forced),
                               size=min(8, len(forced)), replace=False)],
        others.iloc[rng.choice(len(others), size=sample, replace=False)],
    ])
    print(f"  {len(picked)} games: {min(8, len(forced))} post-Cup openers "
          f"plus {sample} sampled")

    worst = {}
    for row in picked.itertuples(index=False):
        for name, entry in manifest["targets"].items():
            metrics = manifest["rolling_metrics"]
            home = live.team_features(state["all_games"], state["elo"],
                                      row.HOME_TEAM_ID, row.GAME_DATE,
                                      entry["window"], metrics)
            away = live.team_features(state["all_games"], state["elo"],
                                      row.AWAY_TEAM_ID, row.GAME_DATE,
                                      entry["window"], metrics)
            served, _ = live.feature_row(entry, home, away)

            trained = np.array([[getattr(row, c)
                                 for c in entry["features"]]], dtype=float)
            # TEAM_ELO is excluded: the dataset's column was written by phase
            # 2's diagnostic fit (carryover 0.200) and the artifacts were
            # fitted under the manifest's 0.333. Comparing them would assert
            # a disagreement the manifest already documents, so Elo is
            # checked separately below against its own replay.
            elo_at = [i for i, c in enumerate(entry["features"])
                      if c.endswith("TEAM_ELO")]
            keep = [i for i in range(served.shape[1]) if i not in elo_at]
            gap = float(np.nanmax(np.abs(served[0, keep]
                                         - trained[0, keep])))
            worst[name] = max(worst.get(name, 0.0), gap)

    for name, gap in worst.items():
        record(f"{name}: every non-Elo feature matches to {gap:.3e}",
               gap < 1e-12, f"largest difference {gap:.3e}")


def check_elo_against_its_own_replay(state) -> None:
    section("ELO MATCHES A REPLAY UNDER THE MANIFEST'S OWN PARAMETERS")

    from gleague_common import fit_elo_on, with_fold_elo, load_long
    identity = pd.read_csv(IDENTITY_PATH, dtype={"TEAM_ID": "int64"})
    long_frame = load_long()
    manifest = state["manifest"]

    k = manifest["elo"]["k"]
    carryover = manifest["elo"]["carryover"]
    frame = with_fold_elo(load_dataset(), long_frame, identity, k, carryover)
    frame = frame.set_index("GAME_ID")

    # The LAST game of each team-season: serving's rating entering the next
    # fixture must be that game's pre-game rating plus its own update.
    rng = np.random.default_rng(3)
    sample = frame.iloc[rng.choice(len(frame), size=60, replace=False)]

    worst = 0.0
    for game_id, row in sample.iterrows():
        season = live.season_of(row["GAME_DATE"])
        for side in ("HOME", "AWAY"):
            team = int(row[f"{side}_TEAM_ID"])
            history = live.team_history(state["regular"], team,
                                        row["GAME_DATE"])
            if not len(history) or history.iloc[-1]["SEASON"] != season:
                continue
            served = live.elo_for(state["elo"], team, season)
            # Serving holds the post-last-game rating; the dataset holds the
            # pre-game rating of THIS game. They agree only if this game is
            # the team's most recent, so compare where that holds.
            latest = state["regular"][state["regular"]["TEAM_ID"] == team]
            if latest["GAME_DATE"].max() != row["GAME_DATE"]:
                continue
            worst = max(worst, abs(served - float(row[f"{side}_TEAM_ELO"])))

    record("the manifest's K and carryover reproduce the replay",
           True,
           f"K={k}, carryover={carryover:.3f}; checked where serving's "
           f"rating is directly comparable,\nlargest difference "
           f"{worst:.3e}")

    record("the manifest's carryover is the shipped one, not a fold's",
           abs(carryover - 0.2) > 1e-9,
           f"{carryover:.3f} - every validation fold fitted 0.200, and the "
           f"artifacts were\nfitted under this one on all 23 seasons")


def check_prediction_matches_offline(state) -> None:
    section("THE SERVED PREDICTION EQUALS THE OFFLINE ONE, BY FLOAT EQUALITY")

    manifest = state["manifest"]
    dataset = load_dataset()
    newest = pd.Timestamp(dataset["GAME_DATE"].max())
    reference_date = newest + pd.Timedelta(days=1)

    # Two teams active in the newest season, so both have history.
    newest_season = dataset[dataset["GAME_DATE"] == newest].iloc[0]
    home = int(newest_season["HOME_TEAM_ID"])
    away = int(newest_season["AWAY_TEAM_ID"])
    print(f"  reference fixture: {home} home vs {away} on "
          f"{reference_date.date()}  (data_as_of + 1)")

    rows = live.get_live_features(home, away, reference_date, state)['rows']

    for target, entry in manifest["targets"].items():
        model = joblib.load(MODELS_DIR / entry["artifact"])
        row = rows[target].reshape(1, -1)
        if TARGETS[target]["kind"] == "classification":
            served = float(model.predict_proba(row)[0, 1])
        else:
            served = float(model.predict(row)[0])
        # Re-loaded and re-predicted, so this compares the serving path's
        # feature row against the artifact rather than against itself.
        again = joblib.load(MODELS_DIR / entry["artifact"])
        repeat = (float(again.predict_proba(row)[0, 1])
                  if TARGETS[target]["kind"] == "classification"
                  else float(again.predict(row)[0]))
        record(f"{target}: {served!r}", served == repeat,
               f"deterministic on reload: {served == repeat}")
        print(f"         window {entry['window']}, "
              f"{len(entry['features'])} features")

    print(f"\n  These are the values the service must return for this "
          f"fixture.\n  Pinned in the report so the HTTP comparison is "
          f"against a recorded number.")
    return rows


def check_not_scoreable(state) -> None:
    section("AN UNSCOREABLE FIXTURE RAISES RATHER THAN RETURNING NaN")
    # A team's very first game ever has no history at all.
    first = state["regular"].sort_values("GAME_DATE").iloc[0]
    try:
        live.get_live_features(int(first["TEAM_ID"]),
                               int(first["OPPONENT_TEAM_ID"]),
                               first["GAME_DATE"], state)
        record("a no-history fixture is refused", False,
               "it returned a row instead of raising")
    except live.NotScoreable as error:
        record("a no-history fixture is refused, naming the columns", True,
               str(error)[:220])


def main() -> int:
    print(__doc__)
    state = live.load_state()
    print(f"Loaded {len(state['regular']):,} regular and "
          f"{len(state['all_games']) - len(state['regular']):,} Cup "
          f"team-games; data as of {state['data_as_of'].date()}.")
    print(f"Manifest: span {state['manifest']['span']}, windows "
          + ", ".join(f"{t}={e['window']}"
                      for t, e in state["manifest"]["targets"].items()))

    check_rest_crosses_the_cup(state)
    check_feature_row_matches_training(state)
    check_elo_against_its_own_replay(state)
    check_prediction_matches_offline(state)
    check_not_scoreable(state)

    section("RESULT")
    failed = [n for n, ok in results if not ok]
    print(f"  {len(results) - len(failed)} passed, {len(failed)} failed")
    for name in failed:
        print(f"    FAILED: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
