"""NFL phase 2: turn `nfl_games_final.csv` into one row per game with features.

Output: `data/processed/nfl_model_dataset.csv`, gitignored like every other NFL
table (see nfl/DATA_LICENSE.md).

ONE FEATURE FUNCTION, FOR TRAINING AND FOR SERVING. `features_for(fixture,
history)` reads a history holding only games STRICTLY BEFORE the fixture, and
knows nothing about the fixture's own result. The training dataset is built by
walking games in order and calling that same function before appending each
result, so phase 4 serves whatever phase 3 selected by calling it again with a
real fixture. `verify_nfl_features.py` rebuilds the history from scratch for 50
sampled games and requires float equality against the stored row.

Run from the project root:
    python data-pipeline/nfl/preprocessing/build_nfl_model_dataset.py
"""
import argparse
import collections
import json
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
NFL = HERE.parent
PIPELINE = NFL.parent          # data-pipeline/, so the NFL's data sits
                               # under data/nfl/ like the other three
                               # leagues' - which is what lets one
                               # DATA_DIR point at either the repo or a
                               # served snapshot with no per-league
                               # special case (served_data.py)
sys.path.insert(0, str(HERE))

import nfl_divisions as D               # noqa: E402
import nfl_franchises as F              # noqa: E402

PROCESSED = PIPELINE / "data" / "nfl" / "processed"
GAMES_PATH = PROCESSED / "nfl_games_final.csv"
OUT_PATH = PROCESSED / "nfl_model_dataset.csv"
MANIFEST_PATH = PROCESSED / "nfl_feature_manifest.json"

# =========================================================================
# THE TEST SET, DECLARED BEFORE ANY FEATURE WAS BUILT (phase 2 spec section 0)
# =========================================================================
# No parameter in this phase may be fitted on these seasons, and this phase
# reports no score on them. `refuse_test_rows` is called by every fitting
# function, so the firewall is enforced rather than remembered.
TEST_SEASONS = (2024, 2025)
# 2026 is NEITHER training nor test: it is the season being served. Its played
# games are in the output, flagged `served`, so phase 4 has them to replay from
# and nothing trains on a half-finished season.
SERVED_SEASON_IS_CURRENT = True
FIRST_SEASON = F.FIRST_SEASON          # 2012, a scope decision


def split_role(season, test_seasons=TEST_SEASONS):
    if season in test_seasons:
        return "test"
    if season > max(test_seasons):
        return "served"
    return "train"


class TestSeasonLeak(RuntimeError):
    """A fitting function was handed a test-season row."""


def refuse_test_rows(games, who, test_seasons=TEST_SEASONS):
    """The firewall. Every fitting function calls this on its input."""
    offending = sorted({g["season"] for g in games} & set(test_seasons))
    if offending:
        raise TestSeasonLeak(
            f"{who} was given {offending} rows. The test set "
            f"{list(test_seasons)} was declared before any feature was built "
            f"and no parameter may be fitted on it.")
    return games


# =========================================================================
# FORM WINDOWS
# =========================================================================
# `season` windows use only the current season, `carry` windows take the last N
# games whatever season they fall in. A 17-game season leaves a within-season
# window empty for weeks, and the carried windows are what made the WNBA and G
# League datasets usable in week 1 - so both are built and phase 3 chooses.
WINDOWS = {
    "ROLL3": (3, "season"),
    "ROLL5": (5, "season"),
    "CARRY5": (5, "carry"),
    "CARRY8": (8, "carry"),
}
METRICS = ("PF", "PA", "MARGIN", "WIN_RATE")

# =========================================================================
# REST
# =========================================================================
# MEASURED, not assumed. The rest-day distribution over 6,974 non-opener
# team-games is 4d 455, 5d 21, 6d 580, 7d 4,491, 8d 471, 9d 28, 10d 415, 11d 64,
# 12d 2, 13d 53, 14d 353, 15d 38, 16d 2, 17d 1.
#
# SHORT_WEEK at <=5 days matches the data: 4 days is Sunday-to-Thursday and
# there is a clean break before 6 days, which is the ordinary week after a
# Monday night game rather than a short one.
SHORT_WEEK_MAX_DAYS = 5

# OFF_BYE COMES FROM THE WEEK GAP, NOT FROM A DAY THRESHOLD, AND THE SPEC'S
# ">= 12 days" WOULD BE WRONG ON ONE ROW. Cross-tabulated, week gap 1 spans
# 4-12 days and week gap 2 spans 12-17, overlapping only at 12:
#     2020 DAL  wk12 2020-11-26 -> wk13 2020-12-08   12 days, week gap 1
#     2021 NE   wk13 2021-12-06 -> wk15 2021-12-18   12 days, week gap 2
# Dallas got twelve days because its week-13 game was rescheduled into the
# following Tuesday, not because it had a bye. A bye IS a scheduled week with no
# game, so the week gap is the definition rather than a proxy for it.
BYE_WEEK_GAP = 2

# =========================================================================
# ELO
# =========================================================================
BASELINE_RATING = 1500.0

EloParams = collections.namedtuple(
    "EloParams", "k carryover home_advantage mov")
# `carryover` is the fraction of the way back to the league mean applied at a
# season boundary, the same sense the WNBA and G League use, so the three
# leagues' fitted values are comparable. 0.0 carries a rating forward untouched.

# The MOV multiplier is the standard log-margin form: a blowout moves a rating
# further, damped when the winner was already favoured, so a strong team beating
# a weak one by 30 is worth less than an upset by 30.
MOV_SHAPE = 2.2
MOV_SCALE = 0.001

# K reaches 100 because the first run put the plain variant's optimum on 50 with
# the grid ending there - a boundary artifact rather than a result, which is the
# trap the WNBA phase hit and `fit_elo` now reports as a GRID EDGE either way.
K_GRID = (8, 12, 16, 20, 24, 28, 32, 40, 50, 60, 70, 85, 100)
CARRYOVER_GRID = (0.0, 0.1, 0.2, 1.0 / 3.0, 0.4, 0.5, 0.6)
HOME_GRID = (0, 20, 35, 50, 65, 80)


def expected_score(rating, opponent_rating):
    return 1.0 / (1.0 + 10.0 ** (-(rating - opponent_rating) / 400.0))


def mov_multiplier(margin, winner_rating_edge):
    """Standard log-margin multiplier, damped by the winner's rating edge."""
    return (math.log(abs(margin) + 1.0)
            * (MOV_SHAPE / ((winner_rating_edge * MOV_SCALE) + MOV_SHAPE)))


def regress_toward_mean(rating, carryover):
    return rating + carryover * (BASELINE_RATING - rating)


def win_value(result):
    return {"W": 1.0, "T": 0.5, "L": 0.0}[result]


class EloLedger:
    """One Elo state under one parameter set. Read-only queries never mutate."""

    def __init__(self, params):
        self.params = params
        self.rating = {}
        self.last_season = {}

    def rating_entering(self, franchise_id, season):
        """The rating a team carries INTO a game, season boundary applied.

        Applied on read rather than on a season-change event, so a caller that
        only reads cannot leave the ledger half-advanced.
        """
        current = self.rating.get(franchise_id, BASELINE_RATING)
        if franchise_id not in self.last_season:
            return current
        if self.last_season[franchise_id] != season:
            return regress_toward_mean(current, self.params.carryover)
        return current

    def home_probability(self, home_id, away_id, season, neutral_site):
        home = self.rating_entering(home_id, season)
        away = self.rating_entering(away_id, season)
        edge = 0.0 if neutral_site else self.params.home_advantage
        return expected_score(home + edge, away)

    def update(self, game):
        home_id, away_id = game["home_franchise_id"], game["away_franchise_id"]
        season = game["season"]
        home = self.rating_entering(home_id, season)
        away = self.rating_entering(away_id, season)
        edge = 0.0 if game["neutral_site"] else self.params.home_advantage
        expected_home = expected_score(home + edge, away)
        actual_home = win_value(game["home_result"])

        shift = self.params.k * (actual_home - expected_home)
        if self.params.mov:
            margin = game["home_points"] - game["away_points"]
            if margin == 0:
                multiplier = mov_multiplier(0, 0.0)
            else:
                # The edge is measured from the WINNER's point of view.
                winner_edge = (home + edge - away) if margin > 0 else (
                    away - home - edge)
                multiplier = mov_multiplier(margin, winner_edge)
            shift *= multiplier

        self.rating[home_id] = home + shift
        self.rating[away_id] = away - shift
        self.last_season[home_id] = season
        self.last_season[away_id] = season


# =========================================================================
# HISTORY
# =========================================================================

class NflHistory:
    """Append-only history of played games. The only input `features_for` has.

    Holds one game log per franchise and one Elo ledger per named variant.
    """

    def __init__(self, elo_configs):
        self.elo_configs = dict(elo_configs)
        self.ledgers = {name: EloLedger(params)
                        for name, params in self.elo_configs.items()}
        self.log = collections.defaultdict(list)
        self.count = 0

    def add(self, game):
        for side, own, other in (("home", "home", "away"),
                                 ("away", "away", "home")):
            self.log[game[f"{own}_franchise_id"]].append({
                "season": game["season"],
                "week": game["week"],
                "date": game["date"],
                "pf": game[f"{own}_points"],
                "pa": game[f"{other}_points"],
                "result": game[f"{own}_result"],
            })
        for ledger in self.ledgers.values():
            ledger.update(game)
        self.count += 1

    def team_log(self, franchise_id):
        return self.log.get(franchise_id, ())


def window_metrics(entries):
    """PF, PA, margin and win rate over a complete window of game entries."""
    n = len(entries)
    return {
        "PF": sum(e["pf"] for e in entries) / n,
        "PA": sum(e["pa"] for e in entries) / n,
        "MARGIN": sum(e["pf"] - e["pa"] for e in entries) / n,
        "WIN_RATE": sum(win_value(e["result"]) for e in entries) / n,
    }


def window_pool(entries, in_season, scope):
    """Which games a window draws on: the season, or everything before it.

    A named seam rather than an inline conditional, so the verifier can replace
    it to prove the within-season windows really are within-season. Patching
    WINDOWS instead would change what the CHECK believes the scopes are, and the
    plant would pass by redefining the question.
    """
    return in_season if scope == "season" else entries


def side_features(franchise_id, season, week, date, neutral_site, history):
    """Every feature for one side of one fixture."""
    out = {}
    entries = history.team_log(franchise_id)
    in_season = [e for e in entries if e["season"] == season]

    for name, (size, scope) in WINDOWS.items():
        pool = window_pool(entries, in_season, scope)
        if len(pool) >= size:
            values = window_metrics(pool[-size:])
        else:
            # Incomplete by design, as every other league's early rows are.
            values = {metric: float("nan") for metric in METRICS}
        for metric in METRICS:
            out[f"{name}_{metric}"] = values[metric]

    # rest, season-scoped: an offseason is not rest, so a season opener has no
    # rest value rather than a capped one. The WNBA settled this the same way.
    if in_season:
        previous = in_season[-1]
        rest_days = (date - previous["date"]).days
        week_gap = week - previous["week"]
        out["REST_DAYS"] = float(rest_days)
        out["SHORT_WEEK"] = float(rest_days <= SHORT_WEEK_MAX_DAYS)
        out["OFF_BYE"] = float(week_gap >= BYE_WEEK_GAP)
        out["SEASON_OPENER"] = 0.0
    else:
        out["REST_DAYS"] = float("nan")
        # NOT unknown-as-zero: a team opening a season definitively did not come
        # off a short week and did not come off a bye - it came off an
        # offseason, which SEASON_OPENER is what records. Only the number of
        # days is unanswerable within a season, and that is the NaN.
        out["SHORT_WEEK"] = 0.0
        out["OFF_BYE"] = 0.0
        out["SEASON_OPENER"] = 1.0

    for name, ledger in history.ledgers.items():
        out[name] = ledger.rating_entering(franchise_id, season)

    # NO PER-SIDE `HOME` COLUMN, AND THE SPEC ASKED FOR ONE. In a one-row-per-
    # game layout it degenerates: the away side's value is 0 on every row ever
    # written, so `AWAY_HOME` is constant by construction and the no-constant-
    # feature guard caught it. Whether the home team actually has home advantage
    # is a property of the GAME, not of a side, and `NEUTRAL_SITE` already
    # carries it - the two would be perfectly collinear. In a long,
    # one-row-per-team layout the spec's version would be the right one. That is
    # also why this function takes no `is_home`: nothing about a side's own
    # windows, rest or rating depends on which end of the fixture it is.
    return out


def features_for(fixture, history):
    """THE one feature function. `history` holds only games before `fixture`.

    Nothing here reads the fixture's own result, and nothing reads a game that
    is not already in `history` - which is what makes the dependency rule in
    section 4 of the phase-2 spec true rather than hoped for.
    """
    season, week, date = fixture["season"], fixture["week"], fixture["date"]
    neutral = bool(fixture["neutral_site"])
    row = {}
    for prefix, key in (("HOME", "home_franchise_id"),
                        ("AWAY", "away_franchise_id")):
        side = side_features(fixture[key], season, week, date, neutral,
                             history)
        for name, value in side.items():
            row[f"{prefix}_{name}"] = value

    row["DIVISION_GAME"] = float(D.is_division_game(
        fixture["home_franchise_id"], fixture["away_franchise_id"]))
    row["WEEK"] = float(week)
    row["NEUTRAL_SITE"] = float(neutral)
    for name, ledger in history.ledgers.items():
        row[f"{name}_PROB"] = ledger.home_probability(
            fixture["home_franchise_id"], fixture["away_franchise_id"],
            season, neutral)
    return row


def feature_columns(elo_names):
    names = []
    for prefix in ("HOME", "AWAY"):
        for window in WINDOWS:
            for metric in METRICS:
                names.append(f"{prefix}_{window}_{metric}")
        names += [f"{prefix}_REST_DAYS", f"{prefix}_SHORT_WEEK",
                  f"{prefix}_OFF_BYE", f"{prefix}_SEASON_OPENER"]
        names += [f"{prefix}_{name}" for name in elo_names]
    names += ["DIVISION_GAME", "WEEK", "NEUTRAL_SITE"]
    names += [f"{name}_PROB" for name in elo_names]
    return names


# =========================================================================
# LOADING
# =========================================================================

def load_games(path=GAMES_PATH):
    """One record per game, from the two team rows phase 1 writes."""
    frame = pd.read_csv(path, parse_dates=["date"])
    games, bad = {}, []
    for game_id, pair in frame.groupby("game_id"):
        if len(pair) != 2:
            bad.append((game_id, len(pair)))
            continue
        home = pair[pair.is_home == 1]
        away = pair[pair.is_home == 0]
        if len(home) != 1 or len(away) != 1:
            bad.append((game_id, "home/away not one each"))
            continue
        home, away = home.iloc[0], away.iloc[0]
        # Recomputed from points rather than read from the margin column, and
        # cross-checked below, so the label cannot inherit a stale derivation.
        games[game_id] = {
            "game_id": game_id,
            "season": int(home["season"]),
            "week": int(home["week"]),
            "date": home["date"].to_pydatetime(),
            "home_franchise_id": int(home["franchise_id"]),
            "away_franchise_id": int(away["franchise_id"]),
            "neutral_site": int(home["neutral_site"]),
            "home_points": int(home["points_for"]),
            "away_points": int(away["points_for"]),
            "home_result": str(home["result"]),
            "away_result": str(away["result"]),
            "stored_margin": int(home["margin"]),
        }
    if bad:
        raise RuntimeError(f"{len(bad)} game(s) are not two clean rows: "
                           f"{bad[:5]}")
    ordered = sorted(games.values(), key=lambda g: (g["date"], g["game_id"]))
    return ordered


def labels_for(game):
    margin = game["home_points"] - game["away_points"]
    tie = game["home_result"] == "T"
    return {
        "HOME_PTS": game["home_points"],
        "AWAY_PTS": game["away_points"],
        "HOME_MARGIN": margin,
        "TOTAL_PTS": game["home_points"] + game["away_points"],
        # A TIE IS EXCLUDED FROM THE WINNER LABEL AND KEPT FOR THE OTHERS. So a
        # served winner probability means P(home wins | not tied), the same
        # qualifier the Q1 and 1H markets already carry.
        "HOME_WIN": "" if tie else int(margin > 0),
        "IS_TIE": int(tie),
    }


# =========================================================================
# ELO FITTING - TRAINING SEASONS ONLY
# =========================================================================

def elo_log_loss(games, params, score_seasons=None):
    """Replay `games` in order; mean log loss over the scored subset.

    Each game is scored on its PRE-game rating, so replaying through the scored
    window is not leakage - a game never contributes to its own prediction.
    Ties are excluded from the objective, for the same reason they are excluded
    from the label: there is no binary outcome to score.
    """
    ledger = EloLedger(params)
    total, n = 0.0, 0
    for game in games:
        if game["home_result"] != "T" and (
                score_seasons is None
                or game["season"] in score_seasons):
            p = ledger.home_probability(
                game["home_franchise_id"], game["away_franchise_id"],
                game["season"], game["neutral_site"])
            p = min(max(p, 1e-15), 1 - 1e-15)
            actual = 1.0 if game["home_result"] == "W" else 0.0
            total += -(actual * math.log(p) + (1 - actual) * math.log(1 - p))
            n += 1
        ledger.update(game)
    return total / n if n else float("nan")


def fit_elo_including_test(games, mov, who="fit_elo_including_test"):
    """The ONE way to fit Elo on rows that include the test seasons.

    Phase 3's production fit is defined as training on everything through the
    last completed season, so its Elo has no holdout - and `fit_elo` must keep
    refusing test rows, because every selection step goes through it. Rather
    than let a caller pass `test_seasons=()` and weaken the firewall by a
    keyword, the opt-out is a separate function with a name that says what it
    is: greppable, impossible to reach by a typo, and asserted by
    `verify_nfl_selection.py` to have exactly one caller.
    """
    return _fit_elo_unguarded(games, mov, who)


def fit_elo(train_games, mov, who="fit_elo"):
    """Grid search K, carryover and home advantage on the training games."""
    refuse_test_rows(train_games, who)
    return _fit_elo_unguarded(train_games, mov, who)


def _fit_elo_unguarded(train_games, mov, who):
    best, best_loss = None, float("inf")
    for k in K_GRID:
        for carryover in CARRYOVER_GRID:
            for home in HOME_GRID:
                params = EloParams(k, carryover, home, mov)
                loss = elo_log_loss(train_games, params)
                if loss < best_loss:
                    best, best_loss = params, loss
    edges = []
    if best.k in (K_GRID[0], K_GRID[-1]):
        edges.append(f"K={best.k}")
    if best.carryover in (CARRYOVER_GRID[0], CARRYOVER_GRID[-1]):
        edges.append(f"carryover={best.carryover:.3f}")
    if best.home_advantage in (HOME_GRID[0], HOME_GRID[-1]):
        edges.append(f"home={best.home_advantage}")
    return best, best_loss, edges


def rolling_origin_folds(train_seasons, count=4):
    """The last `count` training seasons, each validated on itself."""
    seasons = sorted(train_seasons)
    return [(seasons[:seasons.index(v)], v) for v in seasons[-count:]]


# =========================================================================
# BUILD
# =========================================================================

def prior_counts(game, history):
    """How many games each side had played before this one, for the guards."""
    out = {}
    for prefix, key in (("HOME", "home_franchise_id"),
                        ("AWAY", "away_franchise_id")):
        entries = history.team_log(game[key])
        out[prefix] = (
            len(entries),
            sum(1 for e in entries if e["season"] == game["season"]),
        )
    return out


def build_rows(games, elo_configs):
    """Walk games in order, reading features before appending each result."""
    history = NflHistory(elo_configs)
    rows, priors = [], {}
    for game in games:
        priors[game["game_id"]] = prior_counts(game, history)
        row = {
            "game_id": game["game_id"],
            "season": game["season"],
            "week": game["week"],
            "date": game["date"].date().isoformat(),
            "home_franchise_id": game["home_franchise_id"],
            "away_franchise_id": game["away_franchise_id"],
            "split_role": split_role(game["season"]),
        }
        row.update(features_for(game, history))
        row.update(labels_for(game))
        rows.append(row)
        history.add(game)
    return rows, history, priors


def write_manifest(elo_configs, fold_table, frame, path=MANIFEST_PATH):
    """Everything phase 3 selects from and phase 4 must replay under.

    THE PARAMETERS LIVE HERE RATHER THAN IN THE DATASET'S COLUMNS, because a
    column is a value and phase 4 needs the rule that produced it. The WNBA and
    G League serving paths replay Elo from the raw table under their manifest's
    parameters for exactly this reason - a precomputed column that disagrees
    with the parameters beside it is a drift surface with no guard on it.
    """
    manifest = {
        "phase": "nfl-2-features",
        "built_from": GAMES_PATH.name,
        "rows": len(frame),
        "test_seasons": list(TEST_SEASONS),
        "test_set_declared": "before any feature was built; no parameter in "
                             "phase 2 is fitted on it and phase 2 reports no "
                             "score on it",
        "split_roles": {role: int((frame.split_role == role).sum())
                        for role in ("train", "test", "served")},
        "windows": {name: {"size": size, "scope": scope}
                    for name, (size, scope) in WINDOWS.items()},
        "window_metrics": list(METRICS),
        "rest": {
            "short_week_max_days": SHORT_WEEK_MAX_DAYS,
            "bye_week_gap": BYE_WEEK_GAP,
            "bye_from_week_gap_not_days": "a 12-day threshold misreads 2020 "
                                          "DAL wk12->wk13, twelve days from a "
                                          "reschedule rather than a bye",
            "season_opener_rest_is_null": "an offseason is not rest",
        },
        "elo": {
            name: {
                "k": params.k,
                "carryover": params.carryover,
                "carryover_meaning": "fraction of the way back to "
                                     f"{BASELINE_RATING} at a season boundary",
                "home_advantage": params.home_advantage,
                "margin_of_victory_multiplier": params.mov,
                "baseline_rating": BASELINE_RATING,
                "fitted_on": "training seasons only",
            }
            for name, params in elo_configs.items()
        },
        "elo_mov_shape": {"shape": MOV_SHAPE, "scale": MOV_SCALE},
        "elo_folds": [
            {"variant": variant, "validation_season": validation,
             "k": params.k, "carryover": params.carryover,
             "home_advantage": params.home_advantage,
             "train_log_loss": round(train_loss, 6),
             "validation_log_loss": round(validation_loss, 6)}
            for variant, validation, params, train_loss, validation_loss, _
            in fold_table
        ],
        "feature_columns": feature_columns(list(elo_configs)),
        "labels": ["HOME_PTS", "AWAY_PTS", "HOME_MARGIN", "TOTAL_PTS",
                   "HOME_WIN", "IS_TIE"],
        "home_win_meaning": "ties are excluded (blank), so a served winner "
                            "probability means P(home wins | not tied)",
        "not_in_this_phase": ["quarterback or any player-level input",
                              "injuries", "weather", "travel distance",
                              "betting lines", "quarter-based features"],
        "known_largest_gap": "quarterback availability",
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                    encoding="utf-8")
    return manifest


# =========================================================================
# GUARDS ON THE OUTPUT
# =========================================================================

def check_key_coverage(rows, games):
    ids = [r["game_id"] for r in rows]
    counts = collections.Counter(ids)
    repeated = {g: c for g, c in counts.items() if c != 1}
    missing = {g["game_id"] for g in games} - set(ids)
    return (not repeated and not missing,
            f"{len(ids)} rows for {len(games)} games, "
            f"repeated {len(repeated)}, missing {len(missing)}")


def check_all_null_rows(rows, elo_names, priors):
    """An all-null windowed row is allowed only where no window COULD be full.

    THE SPEC'S RULE WAS "a franchise's first game in the data", AND THAT IS
    NARROWER THAN THE WINDOWS ALLOW. The smallest within-season window is 3
    games and the smallest carried one is 5, so a row stays all-null until a
    side reaches those - which is the whole of 2012 weeks 1-3, 48 games, of
    which only 16 are anybody's debut. The rule checked here is the property
    the spec's version was approximating: all-null is permitted only when
    BOTH sides were short of every window, and is a defect anywhere else.
    """
    windowed = [c for c in feature_columns(elo_names)
                if any(c.endswith(f"_{m}") for m in METRICS)]
    min_season = min(size for size, scope in WINDOWS.values()
                     if scope == "season")
    min_carry = min(size for size, scope in WINDOWS.values()
                    if scope == "carry")

    allowed, offenders = 0, []
    for row in rows:
        if not all(isinstance(row[c], float) and math.isnan(row[c])
                   for c in windowed):
            continue
        counts = priors[row["game_id"]]
        short = all(in_season < min_season and overall < min_carry
                    for overall, in_season in counts.values())
        if short:
            allowed += 1
        else:
            offenders.append(row["game_id"])
    return (not offenders,
            f"{allowed} all-null row(s), every one with both sides short of "
            f"every window (<{min_season} in season, <{min_carry} overall); "
            f"{len(offenders)} elsewhere")


def check_no_constant_feature(frame, elo_names):
    constant = [c for c in feature_columns(elo_names)
                if frame[c].nunique(dropna=True) <= 1]
    return not constant, f"constant: {constant or 'none'}"


def check_division_counts(games):
    """Since 2002 every team plays its three rivals twice: 6 per team, 96 per
    season. This is what makes an error in the authored table fatal rather than
    silent - a misplaced franchise breaks the count for two divisions at once.
    """
    per_season = collections.defaultdict(collections.Counter)
    for game in games:
        if D.is_division_game(game["home_franchise_id"],
                              game["away_franchise_id"]):
            per_season[game["season"]][game["home_franchise_id"]] += 1
            per_season[game["season"]][game["away_franchise_id"]] += 1
    detail, ok = [], True
    for season in sorted(per_season):
        counts = per_season[season]
        values = sorted(set(counts.values()))
        total = sum(counts.values()) // 2
        expected = D.DIVISION_GAMES_PER_TEAM_PER_SEASON
        complete = season <= max(TEST_SEASONS)       # 2026 is part-played
        good = (values == [expected] and total == 96) if complete else True
        ok = ok and good
        if not good or not complete:
            detail.append(f"{season}: per-team {values}, {total} games")
    return ok, ("every completed season 96 games, 6 per team"
                + (f"; {'; '.join(detail)}" if detail else ""))


def check_no_double_header(games):
    """A team never plays twice on one date, which is why ordering games within
    a date cannot change any feature."""
    seen = collections.defaultdict(set)
    clashes = []
    for game in games:
        for key in ("home_franchise_id", "away_franchise_id"):
            stamp = (game[key], game["date"].date())
            if stamp in seen[game[key]]:
                clashes.append(stamp)
            seen[game[key]].add(stamp)
    return not clashes, f"{len(clashes)} clash(es)"


# =========================================================================
# THE DEPENDENCY RULE (phase 2 spec section 4) - a finding for phase 4
# =========================================================================

def dependency_report(games, fixtures_path=PROCESSED / "nfl_fixtures.csv"):
    """How many unplayed fixtures are predictable under the dependency rule.

    THE RULE: a fixture is predictable once both teams' previous games are in
    history. Every feature above reads only each team's own earlier games, and
    Elo updates only on games already played, so a Sunday fixture's features are
    final as soon as both teams' prior games are recorded - whatever happens on
    the Thursday in between. That is a different rule from the NBA's
    MAX_DAYS_AHEAD = 1, which is about dates.
    """
    if not fixtures_path.exists():
        return None
    fixtures = pd.read_csv(fixtures_path, parse_dates=["date"])
    season = int(fixtures["season"].max())
    rows = fixtures[fixtures.season == season]

    # Each team's NEXT scheduled fixture is the one whose features are already
    # final. Anything beyond it waits on a game that has not been played.
    next_week = {}
    for _, fixture in rows.sort_values(["date", "game_id"]).iterrows():
        for key in ("home_franchise_id", "away_franchise_id"):
            next_week.setdefault(int(fixture[key]), int(fixture["week"]))

    predictable, waiting = 0, collections.Counter()
    for _, fixture in rows.iterrows():
        home = int(fixture["home_franchise_id"])
        away = int(fixture["away_franchise_id"])
        week = int(fixture["week"])
        if next_week.get(home) == week and next_week.get(away) == week:
            predictable += 1
        else:
            waiting[week] += 1
    return {"season": season, "fixtures": len(rows),
            "predictable_now": predictable, "waiting_by_week": waiting}


# =========================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-count", type=int, default=4)
    parser.add_argument("--quick", action="store_true",
                        help="skip the per-fold Elo refit; it does not change "
                             "the shipped columns")
    args = parser.parse_args()

    games = load_games()
    drift = [g["game_id"] for g in games
             if g["stored_margin"] != g["home_points"] - g["away_points"]]
    if drift:
        print(f"REFUSING: {len(drift)} game(s) where the stored margin "
              f"disagrees with the points: {drift[:5]}")
        return 1

    seasons = sorted({g["season"] for g in games})
    train_seasons = [s for s in seasons if split_role(s) == "train"]
    test_seasons = [s for s in seasons if split_role(s) == "test"]
    served_seasons = [s for s in seasons if split_role(s) == "served"]
    counted = collections.Counter(split_role(g["season"]) for g in games)

    print("=" * 74)
    print("NFL PHASE 2: FEATURES")
    print("=" * 74)
    print(f"  games            {len(games):,} over {seasons[0]}..{seasons[-1]}")
    print(f"  train            {train_seasons[0]}..{train_seasons[-1]} "
          f"({counted['train']:,} games)")
    print(f"  test  DECLARED   {test_seasons} ({counted['test']:,} games) "
          f"- nothing fitted on it, no score reported on it")
    print(f"  served           {served_seasons} ({counted['served']:,} games "
          f"played so far)")

    train_games = [g for g in games if split_role(g["season"]) == "train"]

    # ---------------------------------------------------------- Elo, per fold
    folds = rolling_origin_folds(train_seasons, args.fold_count)
    fold_table = []
    if not args.quick:
        print()
        print("ELO, REFITTED PER ROLLING-ORIGIN FOLD (training seasons only)")
        for variant, mov in (("plain", False), ("MOV", True)):
            for fold_train, validation in folds:
                subset = [g for g in train_games
                          if g["season"] in set(fold_train)]
                params, train_loss, edges = fit_elo(
                    subset, mov, who=f"fit_elo[{variant}/{validation}]")
                scored = [g for g in train_games
                          if g["season"] in set(fold_train) | {validation}]
                validation_loss = elo_log_loss(
                    scored, params, score_seasons={validation})
                fold_table.append((variant, validation, params, train_loss,
                                   validation_loss, edges))
        print(f"  {'variant':<8}{'fold':<7}{'K':>4}{'carry':>8}{'home':>6}"
              f"{'train LL':>11}{'val LL':>9}  grid edge")
        for variant, validation, params, tl, vl, edges in fold_table:
            print(f"  {variant:<8}{validation:<7}{params.k:>4}"
                  f"{params.carryover:>8.3f}{params.home_advantage:>6}"
                  f"{tl:>11.4f}{vl:>9.4f}  {','.join(edges) or '-'}")
        print()
        for variant in ("plain", "MOV"):
            losses = [vl for v, _, _, _, vl, _ in fold_table if v == variant]
            distinct = {(p.k, round(p.carryover, 3), p.home_advantage)
                        for v, _, p, _, _, _ in fold_table if v == variant}
            print(f"  {variant:<6} mean validation log loss "
                  f"{sum(losses) / len(losses):.4f}   "
                  f"{len(distinct)} distinct fitted triple(s) over "
                  f"{len(folds)} folds")
        if any(len({(p.k, round(p.carryover, 3), p.home_advantage)
                    for v, _, p, _, _, _ in fold_table if v == variant}) == 1
               for variant in ("plain", "MOV")):
            print("  a variant landing on one triple across every fold means "
                  "fold-to-fold spread is")
            print("  NOT evidence the refit is live; the two controls in the "
                  "verifier are.")

    # ------------------------------------------- the shipped Elo columns
    print()
    print("ELO FITTED ON ALL TRAINING SEASONS, for the shipped columns")
    elo_configs = {}
    for name, mov in (("ELO", False), ("ELO_MOV", True)):
        params, loss, edges = fit_elo(train_games, mov, who=f"fit_elo[{name}]")
        elo_configs[name] = params
        print(f"  {name:<9} K={params.k:<3} carryover={params.carryover:.3f} "
              f"home={params.home_advantage:<3} training log loss {loss:.4f}"
              f"{'   GRID EDGE: ' + ','.join(edges) if edges else ''}")

    # ---------------------------------------------------------------- build
    rows, history, priors = build_rows(games, elo_configs)
    frame = pd.DataFrame(rows)
    elo_names = list(elo_configs)

    print()
    print("GUARDS")
    checks = [
        ("key coverage", check_key_coverage(rows, games)),
        ("all-null rows only where no window could be full",
         check_all_null_rows(rows, elo_names, priors)),
        ("no constant feature", check_no_constant_feature(frame, elo_names)),
        ("division counts reconcile", check_division_counts(games)),
        ("no team plays twice on a date", check_no_double_header(games)),
    ]
    failed = 0
    for label, (ok, detail) in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
        failed += not ok

    try:
        fit_elo([g for g in games if g["season"] == test_seasons[0]], False,
                who="firewall probe")
        print("  [FAIL] test-season firewall did not fire")
        failed += 1
    except TestSeasonLeak:
        print(f"  [PASS] test-season firewall: a {test_seasons[0]} row raises "
              f"TestSeasonLeak")

    if failed:
        print(f"\nREFUSING to write: {failed} guard(s) failed")
        return 1

    # ------------------------------------------------------------- retention
    print()
    print("WINDOW RETENTION (both sides complete)")
    print(f"  {'window':<9}{'overall':>10}{'weeks 1-4':>12}")
    early = frame[frame.WEEK <= 4]
    for window in WINDOWS:
        cols = [f"{p}_{window}_{m}" for p in ("HOME", "AWAY") for m in METRICS]
        whole = frame[cols].notna().all(axis=1).mean()
        first4 = early[cols].notna().all(axis=1).mean()
        print(f"  {window:<9}{whole:>9.1%}{first4:>12.1%}")

    rest_nan = frame["HOME_REST_DAYS"].isna() | frame["AWAY_REST_DAYS"].isna()
    print(f"  REST_DAYS is NaN on {int(rest_nan.sum())} game(s) "
          f"({rest_nan.mean():.1%}), every one a season opener. The carried "
          f"windows make")
    print(f"  week 1 scoreable on FORM; rest is the one column that does not, "
          f"which is phase 3's to decide.")

    # ------------------------------------------------------------- the rest
    print()
    print("REST, MEASURED")
    rest = pd.concat([frame["HOME_REST_DAYS"], frame["AWAY_REST_DAYS"]])
    print(f"  rest days   min {rest.min():.0f}  median {rest.median():.0f}  "
          f"max {rest.max():.0f}   NaN {int(rest.isna().sum())}")
    for name in ("SHORT_WEEK", "OFF_BYE", "SEASON_OPENER"):
        series = pd.concat([frame[f"HOME_{name}"], frame[f"AWAY_{name}"]])
        print(f"  {name:<14}{int(series.sum()):>6} of {len(series)} "
              f"team-games ({series.mean():.1%})")

    # ------------------------------------------------------------- the labels
    print()
    print("LABELS")
    ties = int(frame["IS_TIE"].sum())
    decided = frame[frame.HOME_WIN != ""]
    print(f"  ties               {ties} game(s): HOME_WIN blank, HOME_MARGIN "
          f"0, TOTAL_PTS kept")
    print(f"  HOME_WIN present   {len(decided)} of {len(frame)}")
    print(f"  neutral-site games {int(frame['NEUTRAL_SITE'].sum())}")
    print(f"  division games     {int(frame['DIVISION_GAME'].sum())} "
          f"({frame['DIVISION_GAME'].mean():.1%})")
    print(f"  home win rate      "
          f"{decided['HOME_WIN'].astype(int).mean():.1%}  (ties excluded)")

    # --------------------------------------------------------- the dependency
    report = dependency_report(games)
    if report:
        print()
        print("THE DEPENDENCY RULE (a finding for phase 4, not code here)")
        print(f"  a fixture is predictable once both teams' previous games are "
              f"in history")
        print(f"  {report['season']} unplayed fixtures : {report['fixtures']}")
        print(f"  predictable today            : {report['predictable_now']}")
        waiting = report["waiting_by_week"]
        if waiting:
            print(f"  waiting on an earlier game   : "
                  f"{sum(waiting.values())} across weeks "
                  f"{min(waiting)}..{max(waiting)}")

    # ------------------------------------------------------------------ write
    ordered = (["game_id", "season", "week", "date", "home_franchise_id",
                "away_franchise_id", "split_role"]
               + feature_columns(elo_names)
               + ["HOME_PTS", "AWAY_PTS", "HOME_MARGIN", "TOTAL_PTS",
                  "HOME_WIN", "IS_TIE"])
    missing = [c for c in ordered if c not in frame.columns]
    extra = [c for c in frame.columns if c not in ordered]
    if missing or extra:
        print(f"\nREFUSING to write: column set disagrees. "
              f"missing {missing}, unexpected {extra}")
        return 1
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = OUT_PATH.with_suffix(".partial")
    frame[ordered].to_csv(partial, index=False)
    partial.replace(OUT_PATH)
    manifest = write_manifest(elo_configs, fold_table, frame)
    print()
    print(f"  wrote {OUT_PATH.name}: {len(frame):,} rows x {len(ordered)} "
          f"columns ({len(feature_columns(elo_names))} features)")
    print(f"  wrote {MANIFEST_PATH.name}: the windows, the rest thresholds, the "
          f"fitted Elo parameters and {len(manifest['elo_folds'])} fold record(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
