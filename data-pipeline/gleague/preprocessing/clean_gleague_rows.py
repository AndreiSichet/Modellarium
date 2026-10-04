"""Row-level cleaning shared by the G League validator and builder.

Shared deliberately: a table cleaned by a rule the validation never saw is a
table nobody has checked. The NBA pipeline learned this the other way round -
its builder drops 10 corrupted-MATCHUP games that its validator passes - and
here the two use one function.

FOUR DEFECT CLASSES, each with a structural signature rather than a score
threshold. A threshold would be the kind of hardcoded expectation this project
distrusts; all four of these are derived from the data's own shape.

  1  UNIDENTIFIABLE ROWS - null TEAM_ID, abbreviation and name, PTS 0. Five
     rows across 23 seasons. They also carry every WL value that is not W or
     L ('O', 'T'), which is how they were first noticed.

  2  PARTIAL STAT LINES - the same (GAME_ID, TEAM_ID) appearing twice, where
     one row carries a valid WL and the other carries NaN and a fragment of a
     score. Eleven team-games.

     THESE ARE SUMMED, NOT DE-DUPLICATED, AND THE DATA SETTLED IT. The first
     implementation kept the WL-bearing row and discarded the other, which
     looked obviously right. The WL-versus-PTS cross-check then failed on two
     games - and summing agrees with WL on 11 of 11 where discarding agrees on
     only 9. On game 2020800105 the surviving row read ANA 93 (W) against REN
     106; summed it reads 109 against 106, which is what the W says happened.
     So they are two halves of one stat line, not a duplicate and a fragment.

  3  UNKNOWN WL - values outside {W, L}, which after class 1 is removed means
     one game whose rows all read 'O'. Normalised to NaN: the margin is still
     exactly derivable from PTS, and an unusable cross-check should read as
     absent rather than as a value.

  4  IRRECONCILABLE GAMES - WL contradicting PTS after summing. Two games in
     9,632. Dropped, because a game whose own record disagrees about who won
     cannot be labelled - and NAMED in KNOWN_UNUSABLE_GAMES below, so a
     fourth one appearing is fatal rather than quietly absorbed.

  5  AMBIGUOUS HOME AND AWAY - both rows of a game carrying the same MATCHUP
     string, so neither reads as the home side. One game. This is the same
     class the NBA pipeline already drops: build_games_table.py discards 10
     games / 20 rows whose MATCHUP is corrupted and identical on both rows,
     which is also why it derives OPPONENT by pairing on GAME_ID rather than
     by parsing MATCHUP text.
"""

import pandas as pd

VALID_WL = {"W", "L"}

# THE THREE GAMES THIS CORPUS CANNOT LABEL, NAMED WITH THEIR EVIDENCE.
#
# Why a named set rather than a silent drop: a negative test showed the
# cleaner absorbing a deliberately corrupted game into the same bucket as
# these, leaving the validator green. That made it unable to tell the
# corpus's three broken games from three hundred newly broken ones - and a
# misleading green is worse than a red. Any game dropped for these reasons
# and NOT listed here is now fatal.
#
# Why a literal is acceptable where the quarter/half list was not: that one
# recorded fetch failures, which a retry could cure, so it needed the
# fetcher's own durable log as its source. These are games whose own two rows
# contradict each other in a file that will never change. The expiry risk is
# handled the other way round instead - the validator fails if a listed game
# has gone missing or has stopped being unusable, so a dead entry cannot sit
# here unnoticed.
KNOWN_UNUSABLE_GAMES = {
    # AUS 92 marked L against ROA 89 marked W - the loser outscored the
    # winner, so WL and PTS cannot both be right and neither can be chosen.
    "2020400101": "WL contradicts PTS (AUS 92 L vs ROA 89 W)",
    # ABQ 99 marked W against TUL 102 marked L - the same defect reversed.
    "2020500004": "WL contradicts PTS (ABQ 99 W vs TUL 102 L)",
    # Three rows, every WL 'O', and both surviving MATCHUP strings written
    # from the away perspective ('ROA @ ABQ'), so no row claims to be home.
    "2020400057": "no valid WL on any row and no identifiable home side",
    # Huntsville Flight 95 - North Charleston Lowgators 95, 2004-01-21, with
    # no WL on either side. Basketball has no draws, so the record is
    # incomplete and nothing recovers the winner.
    "2020300058": "a tied score (95-95) with no WL on either side",
}


# Summed across a team's partial lines. Everything else is taken from the row
# carrying the valid WL, which is the main line.
COUNTING_STATS = [
    "PTS", "REB", "AST", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
    "OREB", "DREB", "STL", "BLK", "TOV", "PF",
]


def clean_season(raw: pd.DataFrame) -> dict:
    """One season's rows, cleaned, with every drop counted and listed."""
    report = {}

    # --- class 1
    unidentifiable = raw[raw["TEAM_ID"].isna()]
    frame = raw[raw["TEAM_ID"].notna()].copy()
    report["unidentifiable_rows"] = len(unidentifiable)

    # --- class 3, before class 2 needs to read WL
    outside = frame["WL"].notna() & ~frame["WL"].isin(VALID_WL)
    report["unknown_wl_rows"] = int(outside.sum())
    frame.loc[outside, "WL"] = pd.NA

    # --- class 2
    duplicated = frame.duplicated(subset=["GAME_ID", "TEAM_ID"], keep=False)
    report["partial_line_rows"] = int(duplicated.sum())

    if duplicated.any():
        frame = _sum_partial_lines(frame)

    # --- class 4
    irreconcilable = _irreconcilable_games(frame)
    report["irreconcilable_games"] = irreconcilable
    if irreconcilable:
        frame = frame[~frame["GAME_ID"].isin(irreconcilable)]

    # --- class 5
    ambiguous = _ambiguous_home_away(frame)
    report["ambiguous_home_away_games"] = ambiguous
    if ambiguous:
        frame = frame[~frame["GAME_ID"].isin(ambiguous)]

    # Every game dropped for contradicting itself must be one of the three
    # this corpus is known to contain. A fourth is a change in the data, not
    # a defect class, and the validator fails on it.
    report["unlisted_drops"] = [
        g for g in list(irreconcilable) + list(ambiguous)
        if str(g) not in KNOWN_UNUSABLE_GAMES
    ]

    report["clean"] = frame.reset_index(drop=True)
    return report


def _sum_partial_lines(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse each (GAME_ID, TEAM_ID) to one row, summing counting stats.

    Identity and context columns come from the row carrying a valid WL where
    one exists, so MATCHUP and the team's labels are taken from the main line
    rather than from a fragment.
    """
    def collapse(group):
        if len(group) == 1:
            return group.iloc[0]

        main = group[group["WL"].isin(VALID_WL)]
        base = (main.iloc[0] if len(main) else group.iloc[0]).copy()

        for column in COUNTING_STATS:
            if column in group.columns:
                base[column] = group[column].sum(min_count=1)
        return base

    collapsed = (frame.groupby(["GAME_ID", "TEAM_ID"], sort=False,
                               group_keys=False)
                 .apply(collapse, include_groups=True))
    return pd.DataFrame(collapsed).reset_index(drop=True)


def _irreconcilable_games(frame: pd.DataFrame) -> list:
    """Games whose WL contradicts their PTS, or that cannot be judged.

    A game with WL absent on either side is NOT irreconcilable in general -
    the margin is still exactly derivable from PTS. Only an active
    contradiction is.

    A TIE IS THE ONE EXCEPTION, AND IT HOLDS WHATEVER WL SAYS. Basketball has
    no draws; overtime decides. So equal scores mean the record is incomplete
    and nothing recovers who won - the margin derives to 0, which is not a
    real result, and HOME_WIN cannot be formed at all. Found by phase 2's Elo
    input gate rather than here, because the WL-validity guard below used to
    skip a tie whose WL was absent on both sides.
    """
    bad = []
    for game_id, group in frame.groupby("GAME_ID"):
        if len(group) != 2:
            continue
        a, b = group.iloc[0], group.iloc[1]
        if a["PTS"] == b["PTS"]:
            bad.append(game_id)
            continue
        if not ({a["WL"], b["WL"]} <= VALID_WL):
            continue
        if sorted([a["WL"], b["WL"]]) != ["L", "W"]:
            bad.append(game_id)
            continue
        if a["PTS"] == b["PTS"]:
            bad.append(game_id)
            continue
        winner = a if a["PTS"] > b["PTS"] else b
        if winner["WL"] != "W":
            bad.append(game_id)
    return bad


def schedule_shape(frame: pd.DataFrame) -> dict:
    """Whether the completeness rule applies to this season, and the result.

    THE RULE IS ONE-DIRECTIONAL AND THE G LEAGUE BREAKS IT, which the WNBA
    never did. "N teams short by S implies N*S/2 absent games" describes a
    balanced schedule with games missing. Three G League seasons have a team
    playing MORE than the mode - 2014-15 at 50-51, 2019-20 at 41-44, 2021-22
    at 31-35 - and a balanced-schedule-minus-absences cannot produce that.

    So there are three outcomes rather than two, and the third is derived from
    the data rather than named in a list: a season whose maximum exceeds its
    mode is UNBALANCED and the rule does not apply to it. That is reported
    loudly, never passed silently, and the row-count identity is asserted on
    it regardless.
    """
    per_team = frame.groupby("TEAM_ID")["GAME_ID"].nunique()
    games = frame["GAME_ID"].nunique()
    teams = len(per_team)
    modal = int(per_team.mode().iloc[0])
    low, high = int(per_team.min()), int(per_team.max())

    shape = {
        "teams": teams, "games": games, "modal": modal,
        "min": low, "max": high,
        # Always true if every game has exactly two sides. Does NOT catch a
        # wholly absent game, which is why it cannot replace the rule below.
        "identity_holds": bool(per_team.sum() == 2 * games),
    }

    if low == high:
        expected = teams * modal / 2
        shape.update(verdict="uniform", expected=expected,
                     ok=bool(games == expected),
                     detail=f"{teams} x {modal} / 2 = {expected:.0f}")
    elif high == modal:
        expected = teams * modal / 2
        shortfall = int((modal - per_team[per_team < modal]).sum())
        missing = expected - games
        shape.update(verdict="short-only", expected=expected,
                     ok=bool(shortfall / 2 == missing),
                     detail=f"shortfall {shortfall} team-games against "
                            f"{missing:.0f} absent -> {shortfall}/2 "
                            f"{'==' if shortfall / 2 == missing else '!='} "
                            f"{missing:.0f}")
    else:
        above = int((per_team > modal).sum())
        shape.update(verdict="unbalanced", expected=None, ok=None,
                     detail=f"{above} team(s) played more than the mode "
                            f"({modal}), so a balanced schedule minus "
                            f"absences cannot produce this distribution")
    return shape


def _ambiguous_home_away(frame: pd.DataFrame) -> list:
    """Games where MATCHUP does not identify exactly one home side.

    MATCHUP is written from the row team's perspective, so a paired game must
    have exactly one row containing 'vs.'. A game where both rows share the
    same string - the NBA's corrupted-MATCHUP pattern - yields zero or two,
    and there is no way to tell which team hosted.
    """
    bad = []
    for game_id, group in frame.groupby("GAME_ID"):
        if len(group) != 2:
            continue
        if int(group["MATCHUP"].str.contains("vs.", regex=False).sum()) != 1:
            bad.append(game_id)
    return bad
