"""Games and teams the standings channel cannot resolve, NAMED with evidence.

A BLANKET TOLERANCE IS THE WRONG INSTRUMENT. One wide enough for the real
failures is wide enough to admit a planted error, which is the whole point of
the channel: the standings' goals-for and goals-against are the only thing
that sees a score changed WITHOUT changing the winner, because the NHL
publishes no period scores.

So the residue is listed, and anything NOT listed is fatal.

THE REVERSE GUARD IS IN `validate_nhl_games.py`: a listed entry that no
longer fails, or no longer exists, is also fatal. Without it this file rots
into a tolerance under another name - the thing the G League's
`KNOWN_UNUSABLE_GAMES` pattern exists to prevent (NOTES_HISTORY.md §47).

Both maps were populated from a real build in discovery mode, never
pre-filled. The build applies **54 repairs across 14 seasons** and leaves
exactly what is below: two games and three team rows, all in two seasons.
Round 4's probe flagged 47 teams and resolved 33 of 39 goals failures; the
difference is the parser, not the rule - Pittsburgh's `{{small|OT}}` labels,
the `@` away marker and two articles' table forms were all costing rows.
"""

# "<season> <ISO date>" -> why this game cannot be resolved.
KNOWN_UNUSABLE_GAMES = {
    # INDETERMINATE. Buffalo's and Chicago's copies disagree about this game,
    # and NO assignment of either reading reconciles both teams with the
    # standings - so the error is not confined to this game. Chicago's diff
    # spans two results (one regulation, one overtime) while its OTHER two
    # candidate games, both against Pittsburgh (2015-01-21 and 2015-02-15),
    # have copies that AGREE with each other. This source therefore cannot
    # tell a shared error in both copies from a wrong standings row, which is
    # round 4's "indeterminate" class.
    "2014–15 2015-04-03":
        "indeterminate: copies disagree and no assignment reconciles "
        "Buffalo, Chicago and Pittsburgh with the 2014-15 standings",

    # AMBIGUOUS. Two assignments reconcile Anaheim and Calgary equally well,
    # so believing either copy is defensible and neither is demonstrated.
    # Recorded rather than picked, because picking would invent a fact.
    "2022–23 2023-03-10":
        "ambiguous: two assignments reconcile Anaheim and Calgary with the "
        "2022-23 standings",
}

# season label -> {abbreviation} whose standings row this corpus cannot
# reconcile. Named per TEAM because a diff is per team, and the evidence for
# each is the diff itself, which the build prints.
STANDINGS_EXCEPTIONS = {
    # The three teams the 2015-04-03 game above leaves unreconciled:
    #   BUF  w +1  l -1  gf +1  ga -1
    #   CHI  w -2  l +1  otl +1  gf -2  ga +2
    #   PIT  w +1  otl -1  gf +1  ga -1
    # Chicago's two-game shortfall is what makes the set irreducible: one of
    # its games against Pittsburgh would have to be wrong in BOTH copies.
    "2014–15": {"BUF", "CHI", "PIT"},
}
