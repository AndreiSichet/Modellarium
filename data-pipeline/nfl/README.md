# NFL pipeline — phase 1: ingestion and validation

A fourth league, and the first that is not basketball. Parallel to the other
three rather than shared with them, for the reason the WNBA and G League
READMEs already give: generalising against one working example is how shared
code acquires the wrong abstraction. Nothing here imports from the NBA, WNBA or
G League pipelines.

**Licence and storage: see `DATA_LICENSE.md`. No NFL table is committed.**

## Run order

```
python data-pipeline/nfl/ingestion/fetch_nfl_seasons.py        # 2012..current
python data-pipeline/nfl/preprocessing/validate_nfl_games.py   # read-only gate
python data-pipeline/nfl/preprocessing/build_nfl_tables.py     # the 4 tables
python data-pipeline/nfl/preprocessing/verify_nfl_validator.py # negative suite
```

## What makes this league different from the other three

**There is no stable team id.** The other three key on `TEAM_ID` from
`nba_api`. Here identity comes from the opponent cell's **link target** — the
season article it points at — resolved through an authored table in
`preprocessing/nfl_franchises.py`. `franchise_id` values are project-assigned
in `1613000001..1613000032`, chosen to be disjoint from the three nba_api
ranges already in use. Five renames or relocations fall inside 2012-2026.

**There is no game-type field.** The other three filter on the third digit of
the game id. Here preseason, regular season and postseason are separate tables,
separated by section heading — measured across 2006-2026 as exactly two
spellings, `Regular season` and `Schedule` — with the postseason table
identified by its header reading `Round`. Two cross-checks back that up: the
recap slug's own `reg`/`pre` token, and a content test, since some articles put
both schedules under one `Schedule` heading and a heading cannot then tell them
apart.

**The game key is `(date, {home, away})`.** Measured: it pairs every row of
every completed season 2012-2025 with zero singletons. A week-based key fails
on 28 rows of 2020, where rescheduling moved games between weeks and the two
articles disagree. **Unplayed fixtures are keyed on week instead**, because a
flexed fixture has no settled date — one article writes `December 26/27` and
the other `December 27`.

**A fifth season boundary.** `SEASON_START_MONTH = 9`: an NFL season is
labelled by the year it starts in. Do not harmonise it with the other four;
they answer different questions, as `CLAUDE.md` §20.26 records.

## Four row states, not two

| state | meaning |
|---|---|
| `played` | a parseable result, kicked off more than five hours ago |
| `fixture` | no result yet — goes to `nfl_fixtures.csv` |
| `voided` | a result cell that is not a score: the 2022 Bills-Bengals game reads `No contest` |
| `in_progress` | a score present on a game that kicked off under five hours ago |

`voided` and `in_progress` belong to neither table. A score written during a
game must never enter history, which is what the five-hour rule is for.

## Access

Action API only, one request per season for all 32 titles. Completed seasons
are **frozen by revision id** and never refetched; only the current season and
the one before it are. See `DATA_LICENSE.md` for the rate, the User-Agent and
the robots.txt reading.

## A reading, stated as one

`en.wikipedia.org/robots.txt` disallows `/w/` for `User-agent: *`, while
Wikimedia's User-Agent policy is written for `api.php` clients and asks them to
identify themselves with contact information. This pipeline reads the
User-Agent policy as the governing document for API access, and robots.txt as
addressing crawlers of index.php URLs. That is a reading of two documents that
do not refer to each other, not a clause, and it is recorded here as such.

## Source defects found and handled

Each of these was found by measurement, not anticipated:

| defect | where | handling |
|---|---|---|
| TemplateStyles CSS inside a date cell | throughout | stripped before parsing; phase 0 measured it dropping 11 of 256 games |
| footnote templates inside a venue cell | 2020 SF, 2021 NO | footnotes dropped whole, not unwrapped — unwrapping put a paragraph about a COVID stadium ban into the venue |
| a piped link inside `{{nowrap}}` | 2014, 2018 | brace-aware unwrap; a first-pipe capture truncated `[[NFL on Thanksgiving Day\|November 27]]` |
| a stray leading pipe in a cell | 2018 CHI | stripped at the cell boundary |
| `! + style="..."\|Week` | 2015 OAK | tolerated; MediaWiki renders it, so a strict parser sees an empty article |
| `rowspan="2" ! \| NFL.com<br>recap` | 2012-era | stray `!` tolerated; `<br>` becomes a space before cleaning |
| a two-level header (`colspan="2" \| Results` over `! Score !! Record`) | 2014 NE | colspan expanded and the continuation row's names substituted |
| a header missing a column | 2024 WAS | repaired only against the canonical order, and the repair is then **tested** against the rows |
| a table opened by `{{NFL Schedule Start}}` | 6 articles, 2024-25 | the template's column order is supplied, with the row width guarded |
| an opponent bolded but not linked | 3 rows per season | the display name is resolved through the same authored table |
| home/away disagreeing between the two articles | 2025 wk7 | resolved from the recap slug, and reported |
| a defective game-summary box | 2012 GB/TEN | `KNOWN_BAD_BOXES`, with both-direction staleness guards |
