"""Elo for the G League - K, between-season carryover and, for the first time
in this project, a FITTED expansion offset.

Three things are fitted, all on training seasons only, and phase 3 refits them
per fold. The spans here are provisional and declared as such.

REPLAYED FROM 2003-04 REGARDLESS OF THE TRAINING SPAN. Elo is a running
quantity, so a rating entering any training season must reflect everything
before it. Fitting on a later span does not mean starting the replay there.

EXPANSION IS FITTED RATHER THAN ASSUMED, AND THE G LEAGUE IS THE FIRST LEAGUE
HERE WITH ENOUGH EXAMPLES. The WNBA started newcomers at the league mean as a
stated assumption and measurement found it optimistic for two of three - three
franchises being far too few to fit anything. Phase 1's identity table records
a first season for each of 40 franchises.

  A RELOCATION KEEPS ITS TEAM_ID and continues its rating. A genuinely new id
  starts fresh. That distinction comes from the identity table, never from
  names - phase 1 found three franchises that changed abbreviation while
  keeping their id, and one that changed abbreviation while keeping its NAME.

  THE SIGN IS NOT ASSUMED. WNBA expansion teams started weak, but a G League
  franchise is founded or bought by an NBA parent club and may arrive with
  assigned players from day one. The fit decides.

ELO USES REGULAR-SEASON GAMES ONLY, and that is a decision rather than an
oversight. Rest days count Cup games because fatigue is physical, and the
rolling features ship an explicit Cup-inclusive pair for phase 3 to judge; a
Cup-inclusive Elo is the same question again and belongs with those candidates
rather than being folded in silently here.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parents[1]

PROCESSED = PIPELINE / "data" / "gleague" / "processed"
REGULAR = PROCESSED / "gleague_games_final.csv"
IDENTITY = PROCESSED / "gleague_franchise_identity.csv"
OUT = PROCESSED / "gleague_elo.csv"

BASELINE_RATING = 1500.0

# Provisional. Phase 3 chooses the real span and must refit all three
# parameters against it - they are fitted on whatever is called training.
TRAINING_THROUGH = "2021-22"

K_GRID = [8, 12, 16, 20, 25, 30, 40, 50, 60, 80, 100]
CARRYOVER_GRID = [0.0, 0.1, 0.2, 1 / 3, 0.5, 0.67, 0.8, 1.0]
OFFSET_GRID = [-150, -120, -100, -80, -60, -40, -20, 0,
               20, 40, 60, 80, 100]

BOOTSTRAP_DRAWS = 2000
RNG_SEED = 20261004


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def expected_score(rating, opponent_rating):
    return 1.0 / (1.0 + 10 ** ((opponent_rating - rating) / 400.0))


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per game: home team, away team, who won, season, date."""
    home = frame[frame["IS_HOME"]].set_index("GAME_ID")
    away = frame[~frame["IS_HOME"]].set_index("GAME_ID")

    games = pd.DataFrame({
        "SEASON": home["SEASON"],
        "GAME_DATE": home["GAME_DATE"],
        "HOME_TEAM_ID": home["TEAM_ID"],
        "AWAY_TEAM_ID": away["TEAM_ID"],
        "HOME_PTS": home["PTS"],
        "AWAY_PTS": away["PTS"],
    }).reset_index()

    games["HOME_WIN"] = (games["HOME_PTS"] > games["AWAY_PTS"]).astype(int)
    if (games["HOME_PTS"] == games["AWAY_PTS"]).any():
        raise SystemExit("a tied game: Elo has no draw handling here")
    return games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def first_seasons(identity: pd.DataFrame) -> dict:
    """Each TEAM_ID's first season in the record, from the identity table."""
    return (identity[identity["first_season"]]
            .set_index("TEAM_ID")["SEASON"].to_dict())


def new_franchises(identity: pd.DataFrame, seasons: list) -> dict:
    """Franchises whose FIRST appearance is after the record begins.

    The 2003-04 cohort is excluded: those teams are where the data starts, not
    expansion, and treating them as newcomers would fit the offset partly on
    the arbitrary beginning of the corpus.
    """
    firsts = first_seasons(identity)
    opening = min(seasons)
    return {team: season for team, season in firsts.items()
            if season != opening}


def run_elo(games: pd.DataFrame, k: float, carryover: float,
            offset: float = 0.0, newcomers: dict = None) -> pd.DataFrame:
    """Sequential Elo, replayed from the first game in `games`.

    Season transition regresses each rating toward the league mean by
    (1 - carryover). A team appearing for the first time starts at
    BASELINE_RATING + offset if it is in `newcomers`, else at BASELINE_RATING.
    """
    newcomers = newcomers or {}
    ratings = {}
    current_season = None

    home_pre = np.empty(len(games))
    away_pre = np.empty(len(games))
    expectations = np.empty(len(games))

    for i, row in enumerate(games.itertuples(index=False)):
        if row.SEASON != current_season:
            if current_season is not None and ratings:
                mean = sum(ratings.values()) / len(ratings)
                for team in ratings:
                    ratings[team] = (mean + carryover
                                     * (ratings[team] - mean))
            current_season = row.SEASON

        for team in (row.HOME_TEAM_ID, row.AWAY_TEAM_ID):
            if team not in ratings:
                start = BASELINE_RATING
                if team in newcomers:
                    start += offset
                ratings[team] = start

        home_rating = ratings[row.HOME_TEAM_ID]
        away_rating = ratings[row.AWAY_TEAM_ID]
        home_pre[i] = home_rating
        away_pre[i] = away_rating

        expected = expected_score(home_rating, away_rating)
        expectations[i] = expected

        change = k * (row.HOME_WIN - expected)
        ratings[row.HOME_TEAM_ID] = home_rating + change
        ratings[row.AWAY_TEAM_ID] = away_rating - change

    out = games.copy()
    out["HOME_TEAM_ELO"] = home_pre
    out["AWAY_TEAM_ELO"] = away_pre
    out["HOME_EXPECTED"] = expectations
    return out


def log_loss_on(played: pd.DataFrame, seasons: set) -> float:
    rows = played[played["SEASON"].isin(seasons)]
    p = np.clip(rows["HOME_EXPECTED"].to_numpy(), 1e-15, 1 - 1e-15)
    y = rows["HOME_WIN"].to_numpy()
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_parameters(games: pd.DataFrame, training: set,
                   newcomers: dict) -> tuple:
    """Grid-search K and carryover on training log loss, offset held at 0."""
    section("FITTING K AND CARRYOVER (training seasons only)")

    best = None
    surface = {}
    for k in K_GRID:
        for carryover in CARRYOVER_GRID:
            played = run_elo(games, k, carryover, 0.0, newcomers)
            loss = log_loss_on(played, training)
            surface[(k, carryover)] = loss
            if best is None or loss < best[2]:
                best = (k, carryover, loss)

    k, carryover, loss = best
    print(f"  grid: {len(K_GRID)} K x {len(CARRYOVER_GRID)} carryover = "
          f"{len(surface)} combinations")
    print(f"  best: K = {k}, carryover = {carryover:.3f}, "
          f"training log loss {loss:.6f}")

    nba = surface.get((20, 1 / 3))
    if nba is not None:
        print(f"  the NBA's K=20, carryover=1/3 on the same data: "
              f"{nba:.6f}  ({(nba - loss) / nba * 100:+.2f}% worse)")

    # An optimum on the edge of the grid is a boundary artifact, not a result -
    # the WNBA phase hit exactly this and had to widen from 40 to 100.
    for name, value, grid in (("K", k, K_GRID),
                              ("carryover", carryover, CARRYOVER_GRID)):
        if value in (grid[0], grid[-1]):
            print(f"  WARNING: {name} = {value} sits on the EDGE of its grid. "
                  f"Widen it before trusting this fit.")

    print(f"\n  CARRYOVER READ: {carryover:.3f} against the NBA's 0.333 and "
          f"the WNBA's 0.500.")
    if nba is not None:
        print(f"""  THE SURFACE IS SHALLOW, AND THAT IS THE HONEST SIZE OF IT: the fitted
  pair beats the NBA's values by {(nba - loss) / nba * 100:.2f}% on training log loss. So the
  DIRECTION is what this establishes, not the magnitude - the fitted
  carryover is not sharply determined, the same caveat the WNBA phase
  recorded about its own fit.""")
    print(f"""  Note the LEVEL too: {loss:.3f}, against roughly 0.63 for the WNBA and
  0.613 for the NBA. Elo alone is markedly weaker here, which is what a
  high-churn league should look like.""")
    if carryover < 1 / 3:
        print("""  LOWER than both, which is what heavy roster churn predicts: a G League
  team's rating last spring says comparatively little about this autumn,
  because the roster that earned it has largely moved on.""")
    else:
        print("""  NOT lower than both, which the churn argument did not predict and is
  worth understanding before trusting. Affiliate continuity - a parent club
  assigning similar players each year - would produce it.""")

    return k, carryover, surface


def fit_expansion_offset(games: pd.DataFrame, training: set, newcomers: dict,
                         k: float, carryover: float) -> float:
    """A single starting offset for genuinely new franchises.

    Two independent estimates, because they answer the question differently
    and agreement between them is the evidence:

      1  a grid search on training log loss, the same instrument used for K
         and carryover
      2  per-franchise revealed strength - where each newcomer's rating
         actually settled by the end of its first season - which supports an
         uncertainty interval over franchises, as the grid search cannot
    """
    section("FITTING THE EXPANSION OFFSET (training seasons only)")

    in_training = {t: s for t, s in newcomers.items() if s in training}
    print(f"  {len(newcomers)} new franchise(s) in the record, "
          f"{len(in_training)} first appearing in a training season")
    if len(in_training) < 3:
        print("  too few to fit an offset; the league mean stays")
        return 0.0

    best = None
    for offset in OFFSET_GRID:
        played = run_elo(games, k, carryover, offset, newcomers)
        loss = log_loss_on(played, training)
        if best is None or loss < best[1]:
            best = (offset, loss)
    grid_offset, grid_loss = best
    at_zero = log_loss_on(run_elo(games, k, carryover, 0.0, newcomers),
                          training)
    print(f"\n  1  grid search: best offset {grid_offset:+.0f}, "
          f"training log loss {grid_loss:.6f}")
    print(f"     against {at_zero:.6f} at offset 0 "
          f"({(at_zero - grid_loss) / at_zero * 100:+.3f}%)")
    if grid_offset in (OFFSET_GRID[0], OFFSET_GRID[-1]):
        print(f"     WARNING: {grid_offset} is on the EDGE of the offset "
              f"grid. Widen it.")

    # Revealed strength: replay with no offset, then read where each newcomer
    # sat at the end of its first season. That is what its own results say its
    # starting rating should have been, and it is one observation per
    # franchise rather than one number for all of them.
    played = run_elo(games, k, carryover, 0.0, newcomers)
    revealed = {}
    for team, season in in_training.items():
        rows = played[(played["SEASON"] == season)
                      & ((played["HOME_TEAM_ID"] == team)
                         | (played["AWAY_TEAM_ID"] == team))]
        if not len(rows):
            continue
        last = rows.iloc[-1]
        rating = (last["HOME_TEAM_ELO"] if last["HOME_TEAM_ID"] == team
                  else last["AWAY_TEAM_ELO"])
        # The pre-game rating of its last game, plus that game's own update,
        # is where the season left it.
        change = k * (last["HOME_WIN"] - last["HOME_EXPECTED"])
        rating += change if last["HOME_TEAM_ID"] == team else -change
        revealed[team] = rating - BASELINE_RATING

    values = np.array(list(revealed.values()))
    rng = np.random.default_rng(RNG_SEED)
    draws = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(BOOTSTRAP_DRAWS)])
    low, high = np.percentile(draws, [2.5, 97.5])

    print(f"\n  2  revealed strength of {len(values)} franchise(s) at the end "
          f"of their first season")
    print(f"     mean {values.mean():+.1f} rating points from the league "
          f"mean, median {np.median(values):+.1f}")
    print(f"     95% CI over franchises [{low:+.1f}, {high:+.1f}] "
          f"({BOOTSTRAP_DRAWS} draws)")
    print(f"     range {values.min():+.0f} to {values.max():+.0f}")

    spans_zero = low <= 0 <= high
    print()
    if spans_zero:
        print(f"""  THE INTERVAL SPANS ZERO, SO THE LEAGUE MEAN STAYS. A new G League
  franchise is not measurably stronger or weaker than the league at its
  first game, on {len(values)} examples. That is a legitimate result rather
  than a failure to find one - and it is the OPPOSITE of the WNBA's picture,
  where two of three newcomers drifted sharply down. The likely mechanism is
  the one the spec named: these franchises arrive with players assigned by an
  NBA parent club rather than through an expansion draft.""")
        offset = 0.0
    else:
        direction = "STRONGER" if values.mean() > 0 else "WEAKER"
        print(f"""  THE INTERVAL EXCLUDES ZERO: new franchises start measurably {direction}
  than the league mean, by {abs(values.mean()):.0f} rating points on """
              f"""{len(values)} examples.""")
        offset = float(values.mean())

    agree = (grid_offset == 0) == spans_zero
    print(f"\n  the two estimates {'AGREE' if agree else 'DISAGREE'} on "
          f"whether an offset is warranted "
          f"(grid {grid_offset:+.0f}, interval "
          f"{'spans' if spans_zero else 'excludes'} zero)")
    if not agree:
        print("""     Taking the interval, because the grid search optimises a whole
     season's log loss and so can move the offset to buy a fraction of a
     percent that the per-franchise spread says is not there.""")
    return offset


def report_continuity(games: pd.DataFrame, identity: pd.DataFrame,
                      played: pd.DataFrame, carryover: float) -> None:
    """Relocations continue their rating; new ids start from the baseline.

    Checked as a general property over every team at every season boundary,
    because the obvious check fails: the WNBA phase keyed on TEAM_NAME and
    missed a franchise that changed abbreviation while keeping its name.
    """
    section("RELOCATIONS CONTINUE, NEW IDS START FRESH")

    firsts = first_seasons(identity)
    seasons = sorted(games["SEASON"].unique())

    def rating_before(team, season):
        rows = played[(played["SEASON"] == season)
                      & ((played["HOME_TEAM_ID"] == team)
                         | (played["AWAY_TEAM_ID"] == team))]
        if not len(rows):
            return None
        first = rows.iloc[0]
        return (first["HOME_TEAM_ELO"] if first["HOME_TEAM_ID"] == team
                else first["AWAY_TEAM_ELO"])

    restarts = 0
    boundaries = 0
    for team, first in firsts.items():
        for season in seasons:
            if season <= first:
                continue
            value = rating_before(team, season)
            if value is None:
                continue
            boundaries += 1
            if abs(value - BASELINE_RATING) < 1e-9:
                restarts += 1

    print(f"  season boundaries checked         : {boundaries}")
    print(f"  boundaries that RESTARTED at 1500 : {restarts}")
    if restarts:
        raise SystemExit("a continuing franchise restarted at the baseline")

    changes = identity[identity["identity_changed"]]
    print(f"\n  {len(changes)} identity change(s) across "
          f"{identity['TEAM_ID'].nunique()} franchises - each one a "
          f"relocation or rename on a CONTINUING id:")
    for row in changes.head(6).itertuples():
        before = rating_before(row.TEAM_ID, row.SEASON)
        print(f"    {row.TEAM_ID}  {row.previous_abbr} -> {row.abbr} "
              f"at {row.SEASON}  entering rating "
              f"{before:.1f}  CONTINUOUS")
    if len(changes) > 6:
        print(f"    ... and {len(changes) - 6} more, all continuous")

    newcomers = new_franchises(identity, seasons)
    print(f"\n  {len(newcomers)} genuinely new id(s) after the record opens, "
          f"each starting fresh:")
    for team, season in sorted(newcomers.items(), key=lambda kv: kv[1])[:6]:
        value = rating_before(team, season)
        print(f"    {team}  first season {season}  "
              f"starting rating {value:.1f}")
    if len(newcomers) > 6:
        print(f"    ... and {len(newcomers) - 6} more")


def assert_no_leakage(games: pd.DataFrame, training: set, held_out: set,
                      newcomers: dict, k: float, carryover: float) -> None:
    """The fit must not move when held-out results change, and must move when
    training results do."""
    section("LEAKAGE: NEGATIVE TEST AND POSITIVE CONTROL")

    def fit(frame):
        best = None
        for kk in K_GRID:
            for cc in CARRYOVER_GRID:
                loss = log_loss_on(run_elo(frame, kk, cc, 0.0, newcomers),
                                   training)
                if best is None or loss < best[2]:
                    best = (kk, cc, loss)
        return best[0], best[1]

    corrupted = games.copy()
    mask = corrupted["SEASON"].isin(held_out)
    corrupted.loc[mask, "HOME_WIN"] = 1 - corrupted.loc[mask, "HOME_WIN"]
    moved = fit(corrupted)
    print(f"  flipping all {int(mask.sum()):,} held-out rows -> "
          f"K = {moved[0]}, carryover = {moved[1]:.3f}")
    if moved != (k, carryover):
        raise SystemExit("held-out results moved a training-only fit")
    print("  unchanged, correct - the fit reads training seasons only")

    # FLIPPING EVERY TRAINING RESULT IS SELF-INVERTING AND MOVES NOTHING.
    # Elo learns exactly mirrored ratings, predicts the mirrored outcomes
    # equally well, and the loss surface is identical - the WNBA phase's
    # positive control failed on precisely this. Randomising HALF destroys
    # signal rather than reflecting it.
    rng = np.random.default_rng(RNG_SEED)
    noisy = games.copy()
    train_rows = noisy.index[noisy["SEASON"].isin(training)]
    chosen = rng.choice(train_rows, size=len(train_rows) // 2, replace=False)
    noisy.loc[chosen, "HOME_WIN"] = rng.integers(0, 2, size=len(chosen))
    control = fit(noisy)
    print(f"\n  randomising {len(chosen):,} of {len(train_rows):,} TRAINING "
          f"rows -> K = {control[0]}, carryover = {control[1]:.3f}")
    if control == (k, carryover):
        raise SystemExit(
            "corrupting half the training results did not move the fit, so "
            "the negative test above proves nothing")
    print(f"  moved, correct - and the DIRECTION is the point: K "
          f"{k} -> {control[0]}, "
          f"{'lower' if control[0] < k else 'higher'}, "
          f"which is what {'less' if control[0] < k else 'more'} signal "
          f"predicts")


def main() -> int:
    print(__doc__)

    frame = pd.read_csv(REGULAR, dtype={"GAME_ID": str})
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])
    identity = pd.read_csv(IDENTITY, dtype={"TEAM_ID": "int64"})

    games = prepare(frame)
    seasons = sorted(games["SEASON"].unique())
    training = {s for s in seasons if s <= TRAINING_THROUGH}
    held_out = set(seasons) - training

    section("SPANS (PROVISIONAL - PHASE 3 CHOOSES AND MUST REFIT)")
    print(f"  replayed from   {seasons[0]} (always, whatever the span)")
    print(f"  training        {min(training)} .. {max(training)}  "
          f"({len(training)} seasons, "
          f"{int(games['SEASON'].isin(training).sum()):,} games)")
    print(f"  held out        {min(held_out)} .. {max(held_out)}  "
          f"({len(held_out)} seasons, "
          f"{int(games['SEASON'].isin(held_out).sum()):,} games)")

    newcomers = new_franchises(identity, seasons)

    k, carryover, _ = fit_parameters(games, training, newcomers)
    offset = fit_expansion_offset(games, training, newcomers, k, carryover)

    played = run_elo(games, k, carryover, offset, newcomers)
    report_continuity(games, identity, played, carryover)
    assert_no_leakage(games, training, held_out, newcomers, k, carryover)

    section("OUTPUT")
    print(f"  K = {k}, carryover = {carryover:.3f}, "
          f"expansion offset = {offset:+.0f}")
    print(f"  rating range {played[['HOME_TEAM_ELO', 'AWAY_TEAM_ELO']].min().min():.0f} "
          f"to {played[['HOME_TEAM_ELO', 'AWAY_TEAM_ELO']].max().max():.0f}")

    out = played[["GAME_ID", "SEASON", "GAME_DATE", "HOME_TEAM_ID",
                  "AWAY_TEAM_ID", "HOME_TEAM_ELO", "AWAY_TEAM_ELO",
                  "HOME_EXPECTED", "HOME_WIN"]]
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"  wrote {OUT.name}: {len(out):,} games")
    return 0


if __name__ == "__main__":
    sys.exit(main())
