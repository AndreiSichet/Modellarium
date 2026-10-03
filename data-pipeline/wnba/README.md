# WNBA pipeline

A second, **parallel** pipeline. It deliberately duplicates parts of the NBA
one rather than generalising it.

## Why duplicated rather than shared

The NBA pipeline is run by hand on the morning of **21 October 2026**, the day
after opening night, and it is the one thing in this project that must work
that morning. Generalising `fetch_games.py` or adding a league parameter to
`validate_games.py` would put that run at risk for no benefit to it.

There is a second reason, which outlasts the deadline: **generalising against
one working example is how shared code acquires the wrong abstraction.** The
two leagues already differ in ways that would have to be parameters —

| | NBA | WNBA |
|---|---|---|
| Season label | `2025-26`, spans two calendar years | `2026`, one calendar year |
| Season boundary | October | May |
| Game-id league prefix | `00` | `10` |
| Games per team | 82, stable | 28 → 44, changed five times since 1997 |
| Teams | 30, stable | 8 → 15, still expanding |

— and only once the WNBA pipeline works is it knowable which of those are
genuinely shared shape and which are coincidence. Extracting the common parts
is a later job, done against two working examples.

**What is imported from the NBA pipeline, and the rule:** pure utilities with
no league assumptions only. Today that is `RateLimiter`. Anything encoding a
season format, a league id or a schedule length stays out.

## Layout

```
wnba/
├── ingestion/fetch_wnba_games.py        LeagueGameFinder, league_id '10'
├── preprocessing/validate_wnba_games.py read-only gate
└── preprocessing/build_wnba_games_table.py  writes the output table
```

```
data/wnba/raw/wnba_games_<season>.csv        gitignored
data/wnba/processed/wnba_games_final.csv     gitignored FOR NOW - see below
```

**Three scripts, not two.** The phase-1 spec named a fetcher and a validator
but required an output table neither obviously owns. The NBA pipeline keeps
`validate_games.py` a read-only gate and builds the table in a separate
`build_games_table.py`, so this follows that: a failed validation must not be
able to leave a half-written output behind.

## Scope: 2015 through the current season

`FIRST_SEASON` is a **scope decision, not a fact that drifts**, the same as
`fetch_games.FIRST_SEASON_START_YEAR`. History reaches back to **1997** and
that is worth knowing, but the 1997 league had eight teams and a different
game. More seasons is not automatically better when the era has shifted, and
starting at 2015 makes the two leagues' datasets comparable in span. Earlier
seasons are a separate experiment, not a default.

## One deliberate divergence from the NBA table

The NBA's `OPPONENT` column holds the opponent's **abbreviation**. This table
carries `OPPONENT_TEAM_ID` as well, and that is the authoritative pairing key.

The reason is measured rather than hypothetical: three WNBA franchises changed
abbreviation while keeping their `TEAM_ID` — `PHO→PHX`, and two relocations,
`SAN→LVA` (San Antonio → Las Vegas) and `TUL→DAL` (Tulsa → Dallas). Keying
anything on abbreviation across a season boundary would split one franchise
into two and silently corrupt every rolling window and Elo rating that crosses
the change. `OPPONENT` is kept for readability and parity; `OPPONENT_TEAM_ID`
is what joins.

## Tracking: gitignored now, committed in phase 4

The NBA's `games_final.csv` is committed because the inference image copies it
and CI reads it. The WNBA equivalent will need the same treatment **once
serving exists**, and not before — committing a file nothing consumes is the
reverse of the problem this project has had three times (see CLAUDE.md §4's
standing rule). Revisit at phase 4, and apply `git check-ignore -v` in the same
change that adds it to any `COPY` list.

## Serving reality

The WNBA plays May to September. As of 3 October 2026 the regular season is
complete and the only unplayed fixtures are **playoffs**, which this project's
regular-season-only design does not serve. So this pipeline can be built and
verified fully against history now; it cannot serve a live prediction until
roughly **May 2027**.
