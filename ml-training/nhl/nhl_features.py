r"""NHL features, every one of them knowable before puck drop.

THE A3 RULE, APPLIED FROM THE FIRST LINE. A3 cost this project a retrain
because four shipped features were functions of what happened DURING the
game rather than of what was known before it. So every column below carries
the reason it is pre-game, and everything the corpus offers that is not
pre-game is named in EXCLUDED with the reason.

WHAT IS AVAILABLE BEFORE A GAME. Only completed games STRICTLY BEFORE the
fixture's own date - which is what `History.cut` means, and what the
truncation test checks by recomputing every training row from that history
alone, through this same code.

EXCLUDED, AND WHY
-----------------
GOALIE OF RECORD. The parsed rows carry the goalie credited with the
decision. It is decided BY the game - A3's trap exactly. A *confirmed
starting* goalie would be pre-game, and is the single most predictive piece
of information an NHL bettor uses; Phase 0b found no source for it under this
project's data rules, so the honest answer is to have neither rather than to
use the post-game one and discover it at a retrain.

THE FIXTURE'S OWN GOALS_FOR / GOALS_AGAINST / MARGIN / RESULT / GAME_END.
The outcome, and the targets.

DERIVED / END_DISPUTED. Notes about how the row was reconciled, written when
the corpus was built. A reconciliation note is not a fact about the teams,
and a feature reading one would be learning which games Wikipedia's two
copies disagreed about.

STANDINGS. The standings tables are season-to-date aggregates whose value AT
THE TIME of an early game cannot be recovered from the article as it stands
now. Using today's value for a game in October is time travel. They stay a
validation channel, which is what Phase 1 built them for.

PUCK DROP TIME. Phase 1 measured that no article carries a usable one, so
features that would need it (a west-coast late start, a matinee) cannot be
built at all. Recorded rather than approximated.

ARIZONA -> UTAH: CARRY, AND IT IS STRUCTURAL
--------------------------------------------
The team, roster, staff and contracts moved together in 2024, so the rating
and the form a side brings into its first Utah game are the ones it earned in
Arizona. The alternative - reset Utah to the baseline as an expansion team
would be - is the Vegas/Seattle treatment, and Utah is not an expansion team.

THIS COULD NOT BE DECIDED ON ERROR. The relocation falls in 2024-25, and
2024-25 and 2025-26 are the TEST seasons, so any comparison would have been
made on test data. It is therefore recorded as a STRUCTURAL decision taken
WITHOUT an error comparison, and Utah's own test-season numbers are reported
afterwards as DESCRIPTION, never as the justification.
"""
import bisect
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DEFAULT_TABLE = (PROJECT / "data-pipeline" / "data" / "nhl" / "processed"
                 / "nhl_games_final.csv")

# ---------------------------------------------------------------- identity
# THE FRANCHISE A RELOCATION CARRIES INTO. Authored, because the identity
# table records Arizona and Utah as two franchise ids and says nothing about
# one becoming the other - it cannot be derived from the data. A map rather
# than a pair, because the NHL has relocated before and will again, and
# `check_relocations` makes a wrong pair fail rather than carry silently.
#
#   1614000032  ARI  Phoenix/Arizona Coyotes, through 2023-24
#   1614000033  UTA  Utah Hockey Club / Mammoth, from 2024-25
RELOCATIONS = {1614000032: 1614000033}

# ---------------------------------------------------------------- windows
ROLLING_WINDOWS = (5, 10)

# A rest gap longer than this reads as "a long break" rather than as a number
# of days. Capping matters because an uncapped gap makes the first game after
# the February break - one 20-day gap in a season - an outlier no training
# row resembles, and a season opener a 170-day one.
REST_DAYS_CAP = 7

# ----------------------------------------------------------------- Elo
# DEFAULTS ONLY, and deliberately not the fitted values. The fitted triple
# comes from nhl_elo_tuning.py, which searches on the TRAINING seasons and
# scores on validation, and it is recorded in the manifest - not here, where
# it would become a number nobody re-derives.
ELO_BASELINE = 1500.0
ELO_DEFAULTS = {"k": 20.0, "home_advantage": 35.0, "carryover": 0.70}

# ------------------------------------------------------------------ split
# DEFINED ONCE. common.TEST_SEASONS is the NBA's lesson: a split written out
# twice drifts, and the half nobody re-reads is the half that leaks. The
# labels carry an EN DASH, which is what the source writes and what the
# corpus stores; spelling one with a hyphen selects nothing, silently.
TEST_SEASONS = ("2024–25", "2025–26")
VALIDATION_SEASONS = ("2023–24",)


def refuse_undeclared_seasons(frame):
    """A season later than the test seasons would land in TRAINING, silently.

    `split` defines training by SUBTRACTION, so any season nobody named ends
    up in it - and the default table carries the OPEN 2026-27 season, which
    would have been fitted on. Fitting on the future is the plainest leak
    there is, and it raises rather than being filtered quietly: a table
    holding a season nobody declared is a question for whoever built it.

    IT MUST RUN BEFORE THE COMPLETENESS FILTER, AND THE FIRST VERSION DID
    NOT. Placed inside `split` - which is called on the already-filtered
    frame - it reported nothing and exit 0, because every one of 2026-27's
    119 fixtures is incomplete this early in the season and the filter had
    already removed them. It looked like a working guard and was an accident
    of the date: by late November those rows become complete and would have
    been swept into training with nothing said. A guard that is correct only
    while the season is young is the misleading green this project keeps
    tallying, so the call site is the fix, not the condition.
    """
    last_declared = max(int(str(s)[:4]) for s in TEST_SEASONS)
    later = sorted({s for s in set(frame["SEASON"])
                    if int(str(s)[:4]) > last_declared})
    if later:
        raise ValueError(
            f"the table carries season(s) {later}, later than the declared "
            f"test seasons {list(TEST_SEASONS)}; defining training by "
            f"subtraction would fit on them. Pass a table of completed "
            f"seasons, or declare them in nhl_features")

# ==========================================================================
# loading
# ==========================================================================

def load_games(table=None, seasons=None, apply_relocations=True):
    """The team-game table: one row per team per game, dates as dates.

    RELOCATION APPLIED HERE, so everything downstream sees one franchise and
    no later step has to remember to.
    """
    games = pd.read_csv(table or DEFAULT_TABLE, low_memory=False)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"]).dt.date
    games["SEASON"] = games["SEASON"].astype(str)
    # IS_HOME ARRIVES AS 0/1 AND MUST BE MADE BOOLEAN. Masking a frame with
    # an int64 column raises in pandas, so it is converted once, here, rather
    # than at four call sites that each have to remember.
    games["IS_HOME"] = games["IS_HOME"].astype(int).astype(bool)
    if seasons is not None:
        games = games[games["SEASON"].isin(set(seasons))]
    if apply_relocations:
        check_relocations(games)
        games = games.copy()
        games["TEAM_ID"] = games["TEAM_ID"].replace(RELOCATIONS)
        games["OPPONENT_ID"] = games["OPPONENT_ID"].replace(RELOCATIONS)
    return (games.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"])
            .reset_index(drop=True))


def check_relocations(games):
    """A relocation pair whose two sides overlap in time is not a relocation.

    THE CHECK THAT MAKES THE AUTHORED MAP SAFE. If both ids appear in the
    same season they are two teams, and merging them would fuse two real
    franchises' histories - so it raises rather than carrying.
    """
    for old, new in RELOCATIONS.items():
        old_seasons = set(games.loc[games["TEAM_ID"] == old, "SEASON"])
        new_seasons = set(games.loc[games["TEAM_ID"] == new, "SEASON"])
        shared = old_seasons & new_seasons
        if shared:
            raise ValueError(
                f"franchise {old} and {new} are mapped as a relocation but "
                f"both played in {sorted(shared)}; that is two teams, not "
                f"one moving")


def season_order(games):
    """Seasons in chronological order, from the label's start year."""
    return sorted(set(games["SEASON"]), key=lambda s: int(str(s)[:4]))


def as_fixtures(games):
    """One row per GAME: both ids, both scores, and the targets.

    BUILT FROM IS_HOME, NOT FROM ROW ORDER. Phase 1 guarantees exactly one
    home side per game - its plant (j) is what checks that - so this is a
    join, and a game with two home rows raises here instead of having one of
    them picked arbitrarily.
    """
    home = games[games["IS_HOME"]].set_index("GAME_ID")
    away = games[~games["IS_HOME"]].set_index("GAME_ID")
    if home.index.has_duplicates or away.index.has_duplicates:
        raise ValueError("a game id carries more than one home or away row")
    shared = home.index.intersection(away.index)
    if len(shared) != len(home) or len(shared) != len(away):
        raise ValueError(
            f"{len(home)} home row(s) and {len(away)} away row(s) share only "
            f"{len(shared)} game id(s); Phase 1 guarantees one of each per "
            f"game, so this is a corpus fault rather than something to work "
            f"around")
    shared = list(shared)
    out = pd.DataFrame({
        "GAME_ID": shared,
        "SEASON": home.loc[shared, "SEASON"].to_numpy(),
        "GAME_DATE": home.loc[shared, "GAME_DATE"].to_numpy(),
        "HOME_ID": home.loc[shared, "TEAM_ID"].to_numpy(),
        "AWAY_ID": away.loc[shared, "TEAM_ID"].to_numpy(),
        "HOME_GOALS": home.loc[shared, "GOALS_FOR"].to_numpy(),
        "AWAY_GOALS": away.loc[shared, "GOALS_FOR"].to_numpy(),
        "GAME_END": home.loc[shared, "GAME_END"].to_numpy(),
    })
    out["HOME_WIN"] = (out["HOME_GOALS"] > out["AWAY_GOALS"]).astype(int)
    out["HOME_MARGIN"] = out["HOME_GOALS"] - out["AWAY_GOALS"]
    out["TOTAL_GOALS"] = out["HOME_GOALS"] + out["AWAY_GOALS"]
    return (out.sort_values(["GAME_DATE", "GAME_ID"])
            .reset_index(drop=True))


# ==========================================================================
# Elo
# ==========================================================================

def _regress(ratings, carryover):
    """Pull every rating toward the league baseline between seasons."""
    for team in list(ratings):
        ratings[team] = ELO_BASELINE + carryover * (ratings[team]
                                                    - ELO_BASELINE)


def _update(ratings, home_id, away_id, home_goals, away_goals, k,
            home_advantage):
    """One game's Elo update. Returns the two PRE-game ratings.

    SHARED BY THE BULK REPLAY AND THE AS-OF REPLAY on purpose: the truncation
    test is only a test of this code if both paths run the same arithmetic.

    THERE ARE NO TIES IN THIS ERA, so the outcome is binary and a shootout
    win is a win - which is how the source records it and how the market
    settles it. The alternative (0.6/0.4 for an OT or SO result, as some
    public Elos use) would be a second parameter fitted on 13 seasons, and
    section 5 measures whether OT/SO is worth modelling at all first.
    """
    home = ratings.setdefault(home_id, ELO_BASELINE)
    away = ratings.setdefault(away_id, ELO_BASELINE)
    expected = 1.0 / (1.0 + 10.0 ** (
        -(home + home_advantage - away) / 400.0))
    actual = 1.0 if home_goals > away_goals else 0.0
    change = k * (actual - expected)
    ratings[home_id] = home + change
    ratings[away_id] = away - change
    return home, away


def replay_elo(fixtures, k, home_advantage, carryover):
    """The rating each side brought INTO each fixture, plus the final state.

    PRE-GAME BY CONSTRUCTION. The rating recorded per fixture is the one each
    team held BEFORE the game; the update is applied after it is recorded,
    which is the whole reason this returns the pre-game arrays separately
    from the ratings.

    CARRY-OVER IS FITTED, NOT ASSUMED. 1.0 carries a rating untouched into
    the next autumn, 0.0 resets everyone to the baseline. The NHL's draft and
    cap make some regression right; how much is a parameter.
    """
    ratings = {}
    season_seen = None
    pre_home = np.empty(len(fixtures), dtype=float)
    pre_away = np.empty(len(fixtures), dtype=float)

    for index, row in enumerate(fixtures.itertuples()):
        if row.SEASON != season_seen:
            if season_seen is not None:
                _regress(ratings, carryover)
            season_seen = row.SEASON
        pre_home[index], pre_away[index] = _update(
            ratings, row.HOME_ID, row.AWAY_ID, row.HOME_GOALS,
            row.AWAY_GOALS, k, home_advantage)

    return pre_home, pre_away, ratings


def elo_as_of(fixtures, date, season, k, home_advantage, carryover):
    """Ratings as they stood before any game on `date`, for `season`.

    THE TRUNCATION TEST'S INDEPENDENT PATH. It replays only games strictly
    before `date` and then applies the season carry-over if the last game it
    saw belongs to an earlier season - which is exactly what replay_elo does
    when it reaches the first fixture of a new season, and it is why `season`
    has to be passed in rather than inferred from the games. Without that
    step a season opener's rating would be last April's, and the test would
    report a disagreement that is really a missing transform.
    """
    ratings = {}
    season_seen = None
    for row in fixtures.itertuples():
        if row.GAME_DATE >= date:
            break
        if row.SEASON != season_seen:
            if season_seen is not None:
                _regress(ratings, carryover)
            season_seen = row.SEASON
        _update(ratings, row.HOME_ID, row.AWAY_ID, row.HOME_GOALS,
                row.AWAY_GOALS, k, home_advantage)

    if season_seen is not None and season_seen != season:
        _regress(ratings, carryover)
    return ratings


def elo_win_probability(home_elo, away_elo, home_advantage):
    """A rating pair as a home win probability."""
    return 1.0 / (1.0 + 10.0 ** (
        -(np.asarray(home_elo, dtype=float) + home_advantage
          - np.asarray(away_elo, dtype=float)) / 400.0))


# ==========================================================================
# history
# ==========================================================================

class History:
    """Each team's completed games, answerable as of any date.

    STRICTLY BEFORE IS THE ONLY QUESTION IT ANSWERS, which is the point: the
    only way to read a game's own row out of this object is to ask for a date
    after it, and nothing in the feature code does.
    """

    def __init__(self, games):
        self.dates = {}
        self.seasons = {}
        self.goals_for = {}
        self.goals_against = {}
        self.wins = {}
        for team, frame in games.groupby("TEAM_ID"):
            frame = frame.sort_values(["GAME_DATE", "GAME_ID"])
            self.dates[team] = list(frame["GAME_DATE"])
            self.seasons[team] = list(frame["SEASON"])
            self.goals_for[team] = frame["GOALS_FOR"].to_numpy(dtype=float)
            self.goals_against[team] = frame["GOALS_AGAINST"].to_numpy(
                dtype=float)
            self.wins[team] = (
                frame["GOALS_FOR"].to_numpy()
                > frame["GOALS_AGAINST"].to_numpy()).astype(float)

    def cut(self, team, date):
        """How many of this team's games are STRICTLY before `date`."""
        dates = self.dates.get(team)
        if not dates:
            return 0
        return bisect.bisect_left(dates, date)

    def season_start(self, team, season):
        """Index of this team's first game in `season`, or None."""
        seasons = self.seasons.get(team)
        if not seasons:
            return None
        try:
            return seasons.index(season)
        except ValueError:
            return None


# ==========================================================================
# rest
# ==========================================================================

REST_COLUMNS = ("REST_DAYS", "IS_BACK_TO_BACK", "IS_THIRD_IN_FOUR")


def rest_features(history, team, date, season, carry_across_seasons):
    """Days since this team's previous game, and the two hockey flags.

    PRE-GAME: the previous game's date is known the moment it is played, and
    the fixture's own date is on the published schedule weeks ahead.

    THE SEASON OPENER IS DEFINED EXPLICITLY, which is the G League's lesson.
    Its `rest_for` is season-scoped and leaves a genuine opener NaN, and that
    NaN is why every 2026-27 G League fixture refused until each team had one
    game on record. Here it is a parameter, so section 4 can measure it:

      * season-scoped (carry_across_seasons=False): an opener has no previous
        game this season, so REST_DAYS is NaN and the row is INCOMPLETE.
        Honest, and it makes 32 fixtures a season unservable.
      * carrying (True): the gap is measured to the last game of the previous
        season and then CAPPED, so an opener reads as a long break rather
        than as nothing. A 170-day summer and a 20-day February break both
        become "longer than the cap", which is all the feature can usefully
        say about either, and is true of both.

    THIRD IN FOUR NIGHTS is included because it is specifically a hockey
    thing: a back-to-back is common and survivable, a third game in four
    nights is the schedule spot teams rest a starter for.
    """
    cut = history.cut(team, date)
    if cut == 0:
        return {name: np.nan for name in REST_COLUMNS}

    floor = 0
    if not carry_across_seasons:
        start = history.season_start(team, season)
        if start is None or start >= cut:
            return {name: np.nan for name in REST_COLUMNS}
        floor = start

    previous = history.dates[team][floor:cut]
    if not previous:
        return {name: np.nan for name in REST_COLUMNS}

    raw = (date - previous[-1]).days
    # The fixture itself is the third game, so two of this team's own games
    # inside the trailing three nights makes it three in four.
    within_three = sum(1 for d in previous if (date - d).days <= 3)
    return {
        "REST_DAYS": float(min(raw, REST_DAYS_CAP)),
        "IS_BACK_TO_BACK": float(raw <= 1),
        "IS_THIRD_IN_FOUR": float(within_three >= 2),
    }


# ==========================================================================
# rolling form
# ==========================================================================

ROLLING_METRICS = ("GOALS_FOR", "GOALS_AGAINST", "GOAL_DIFF", "WIN_SHARE")


def rolling_form(history, team, date, season, window, carry_across_seasons):
    """Trailing means over this team's previous `window` completed games.

    PRE-GAME: every row read is a completed game whose date is strictly
    before the fixture's, so the window is closed before the puck drops.

    HOW A WINDOW CROSSES A SEASON BOUNDARY IS A PARAMETER, and the four
    leagues already shipped disagree on purpose. The WNBA and G League carry
    across, because their seasons are short enough that the warm-up would
    cost a tenth of the rows. The NBA's are within-season, because 82 games
    make the warm-up cheap and a summer of roster turnover makes last April's
    form a poor guide. The NHL also plays 82, which argues for within-season
    - but it is section 4's question, measured rather than asserted, so both
    are built and neither is the default by assumption.

    AN INCOMPLETE WINDOW IS NaN, NEVER A PARTIAL MEAN. A mean of three games
    in a column named for ten is a different quantity wearing the same name -
    the unknown-is-never-a-zero rule in its less obvious form.
    """
    cut = history.cut(team, date)
    floor = 0
    if not carry_across_seasons:
        start = history.season_start(team, season)
        if start is None:
            return {f"ROLL{window}_{m}": np.nan for m in ROLLING_METRICS}
        floor = start
    if cut - floor < window:
        return {f"ROLL{window}_{m}": np.nan for m in ROLLING_METRICS}

    lo = cut - window
    goals_for = history.goals_for[team][lo:cut]
    goals_against = history.goals_against[team][lo:cut]
    wins = history.wins[team][lo:cut]
    return {
        f"ROLL{window}_GOALS_FOR": float(goals_for.mean()),
        f"ROLL{window}_GOALS_AGAINST": float(goals_against.mean()),
        f"ROLL{window}_GOAL_DIFF": float((goals_for - goals_against).mean()),
        f"ROLL{window}_WIN_SHARE": float(wins.mean()),
    }


# ==========================================================================
# the feature row
# ==========================================================================

def feature_columns(windows=ROLLING_WINDOWS):
    """Every feature column, in a fixed order, DERIVED from the definitions.

    NOT A HAND-WRITTEN LIST. Three of this project's recorded breakages were
    a literal that went stale - a season list, a row count and a reference
    probability - and a feature list is the same shape of mistake waiting for
    the first added window.

    HOME/AWAY IS IN THE FRAMING, NOT IN A COLUMN. Every row is home-versus-
    away, so a constant `IS_HOME = 1` would be collinear with the intercept
    and carry no information; the home effect is the intercept in the linear
    models and the `home_advantage` parameter in Elo. Stated here because
    "where did home/away go" is the first question a reader will have.
    """
    columns = ["HOME_ELO", "AWAY_ELO", "ELO_DIFF"]
    for side in ("HOME", "AWAY"):
        for name in REST_COLUMNS:
            columns.append(f"{side}_{name}")
        for window in windows:
            for metric in ROLLING_METRICS:
                columns.append(f"{side}_ROLL{window}_{metric}")
    return columns


KEY_COLUMNS = ("GAME_ID", "SEASON", "GAME_DATE", "HOME_ID", "AWAY_ID")
TARGET_COLUMNS = ("HOME_WIN", "HOME_MARGIN", "TOTAL_GOALS")


def row_features(history, home_id, away_id, date, season, home_elo, away_elo,
                 carry_rolling=False, carry_rest=False,
                 windows=ROLLING_WINDOWS):
    """One fixture's features. THE ONLY PLACE THEY ARE DEFINED.

    The bulk builder and the truncation test both go through here, which is
    what makes the test a test: if it recomputed the features its own way it
    would be checking a second implementation rather than this one.
    """
    row = {
        "HOME_ELO": home_elo,
        "AWAY_ELO": away_elo,
        "ELO_DIFF": home_elo - away_elo,
    }
    for side, team in (("HOME", home_id), ("AWAY", away_id)):
        for name, value in rest_features(history, team, date, season,
                                         carry_rest).items():
            row[f"{side}_{name}"] = value
        for window in windows:
            for name, value in rolling_form(history, team, date, season,
                                            window, carry_rolling).items():
                row[f"{side}_{name}"] = value
    return row


def build_features(games, elo=None, carry_rolling=False, carry_rest=False,
                   windows=ROLLING_WINDOWS):
    """One row per fixture: keys, features and targets."""
    params = dict(ELO_DEFAULTS, **(elo or {}))
    fixtures = as_fixtures(games)
    history = History(games)
    pre_home, pre_away, _final = replay_elo(
        fixtures, params["k"], params["home_advantage"], params["carryover"])

    rows = []
    for index, fixture in enumerate(fixtures.itertuples()):
        row = {name: getattr(fixture, name) for name in KEY_COLUMNS}
        row.update(row_features(
            history, fixture.HOME_ID, fixture.AWAY_ID, fixture.GAME_DATE,
            fixture.SEASON, pre_home[index], pre_away[index],
            carry_rolling=carry_rolling, carry_rest=carry_rest,
            windows=windows))
        for name in TARGET_COLUMNS:
            row[name] = getattr(fixture, name)
        row["GAME_END"] = fixture.GAME_END
        rows.append(row)

    columns = (list(KEY_COLUMNS) + feature_columns(windows)
               + list(TARGET_COLUMNS) + ["GAME_END"])
    return pd.DataFrame(rows, columns=columns)


def is_complete(frame, windows=ROLLING_WINDOWS):
    """Rows with every feature present - the set a model would train on."""
    return frame[feature_columns(windows)].notna().all(axis=1)
