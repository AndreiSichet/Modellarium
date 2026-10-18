# NHL pipeline — phase 1: ingestion and validation

A fifth league, and the second that is not basketball. Parallel to the other
four rather than shared with them, for the reason the WNBA, G League and NFL
READMEs already give: generalising against one working example is how shared
code acquires the wrong abstraction. Nothing here imports from the NBA, WNBA,
G League or NFL pipelines.

**Licence and storage: see `DATA_LICENSE.md`. No NHL table is committed.**

Everything this phase was built against was measured first, in
`data-pipeline/probes/NOTES_round4_nhl.md` and summarised in
`NOTES_HISTORY.md` §12. Where a number here looks oddly specific, that is
where it came from.

## Run order

```
python data-pipeline/nhl/ingestion/fetch_nhl_seasons.py          # 2013-14..current
python data-pipeline/nhl/preprocessing/build_nhl_tables.py       # the 5 tables
python data-pipeline/nhl/preprocessing/validate_nhl_games.py     # read-only gate
python data-pipeline/nhl/preprocessing/verify_nhl_validator.py   # negative suite
```

## What makes this league different from the other four

**Season titles use an EN DASH.** `2021–22 Montreal Canadiens season`, not
`2021-22`. Every title is built through one helper so the dash appears in one
place; a hyphen returns a page that does not exist, or worse a redirect.

**There is no stable team id**, as for the NFL. Identity comes from the
opponent cell's **link target**, resolved through an authored table in
`preprocessing/nhl_franchises.py`. Ids are project-assigned in
`1614000001..1614000033`, disjoint from the four ranges already in use.
**Arizona and Utah are separate franchise ids**, as the league records them —
whether Elo carries across is a phase 2 decision and belongs there, not here.
Link targets are *validated* against the table and never trusted: two in the
corpus are not NHL franchises at all (`Dallas Mavericks`, and a misspelled
`Colorado Avalance`).

**A third outcome exists.** `RESULT ∈ {W, L, OTL}` and
`GAME_END ∈ {REG, OT, SO}` are separate columns, because they answer different
questions: an OTL is a loss that earns a point, and a shootout is not an
overtime. **Zero ties in 15 seasons and zero OT/SO games with a margin other
than 1**, across 7,434 rows — which is the invariant that proves the
one-extra-goal convention, and also the dominant fact about modelling this
sport: about half of all one-goal games are decided in OT or a shootout.
`GAME_END` is carried so a later phase can treat those differently.

**The game key is `({both teams}, date ± 1 day)`.** Measured: it pairs all
2,624 rows of 2021-22 with **0 unpaired**, where `(date, home, away)` leaves 4
singletons — because two games' two copies disagree about the date by one day.
The two copies never disagree about *which* team was home.

**Columns are located by name, never by position.** 73 distinct literal header
tuples collapse to 6 forms; the names are in the markup, either as positional
parameters of `{{Game log section start}}` or as an ordinary `! # !! Date !!`
header row. A table whose required columns cannot be named is **fatal, not
guessed**.

**Two containers, and neither is "the" markup**: 10,003 rows sit in templates
and 28,490 in raw nested `{| class="toccolours collapsible"` wikitables.

**Whose score comes first is read from the column names.** An `Opponent`
column means our team's goals come first; a `Visitor`/`Home` pair means the
visitor's do.

## The standings channel, which no other league here has

`Template:<season> NHL standings` carries `gf` and `ga` per team. **A score
altered without changing the winner moves those two numbers and nothing
else**, so this is the only channel that can see that edit at all — the NHL's
answer to the NFL's quarter-by-quarter box score, which this sport does not
publish.

The whole-league template exists only from **2020-21**; before that the same
data is split one template per division, named in the season article's
Standings section, so that article is read to find them.

**A trap inside it: the tiebreak column changes meaning.** `RW` (regulation
wins) from 2020-21, **`ROW`** (regulation *and overtime* wins) before it. The
header says which, and comparing a ROW column against regulation wins made all
30 teams of 2016-17 look wrong by 3 to 9. The reader takes the meaning from
the header and the builder does not use the column at all.

**It is a repair rule, not only a detector**, which is what makes a non-zero
baseline usable. Crossed with the two-copy pairing and solved by exhaustive
search over which copy to believe, 33 of round 4's 39 goals failures have a
**unique** repair. Those are applied and recorded in `nhl_repairs.csv` as
**derived**; the residue is named in `nhl_standings_exceptions.py` with its
evidence and the reverse guard. **A blanket tolerance is the wrong
instrument** — one wide enough for the real failures is wide enough to admit a
planted error.

## Completed seasons are strict; the open season quarantines

Phase 1 of a league whose current season is edited continuously cannot be
fatal on a single disagreement, or the daily refresh stops for one bad row
(`CLAUDE.md`'s 22 October item). So:

- **Completed season**: a disagreement between the two copies is resolved by
  the standings or the game is named. Nothing is silently dropped.
- **Open season**: a disagreement is **quarantined** — out of the games table,
  into `nhl_quarantine.csv` with both copies and the reason, reported, **not
  fatal**. A quarantined game whose copies come to agree leaves quarantine on
  the next build, and that movement is reported too.
- **Open season standings are advisory**, with the sign pattern classified:
  all-negative across several teams is lag, mixed-sign on one team is a
  probable error localised to a game.
- **A row that fails to parse is fatal in either case.** Quarantine is for
  disagreement between two readable copies, never for markup nobody can read.

## Scope

**2013-14 through 2026-27.** 2012-13 is excluded: 11 of its 30 articles have
no game log at all, so its failures are a coverage problem rather than a data
one. `FIRST_SEASON` is a scope literal and carries a comment saying so;
nothing else in this pipeline is a literal that can expire.
