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

---

# NFL pipeline — phase 2: features

```
python data-pipeline/nfl/preprocessing/nfl_divisions.py            # the table's own checks
python data-pipeline/nfl/preprocessing/build_nfl_model_dataset.py  # the dataset
python ml-training/nfl/verify_nfl_features.py                      # 10 checks + 10 plants
```

Output, gitignored like everything else here: `nfl_model_dataset.csv` (one row
per played regular-season game) and `nfl_feature_manifest.json` (the windows,
the rest thresholds, the fitted Elo parameters, the fold records). **Phase 4
must replay Elo from the manifest rather than read a stored column**, for the
reason the WNBA and G League serving paths already do: a precomputed column
sitting beside parameters it does not agree with is a drift surface with no
guard on it.

## The test set was declared before any feature was built

**2024 and 2025, 544 games.** `refuse_test_rows` is called by every fitting
function, so handing one a test-season row raises `TestSeasonLeak` rather than
quietly fitting on it. **2026 is neither**: it is the season being served, and
its played games carry `split_role = served` so nothing trains on a
half-finished season.

## One feature function, for training and for serving

`features_for(fixture, history)` is given a history holding only games strictly
before the fixture and never sees the fixture's own result. The training dataset
is built by walking games in order and calling that same function before
appending each result. The verifier rebuilds the history from scratch for 50
sampled games and requires **float equality** against the stored row.

## What the measurements changed about the plan

| the plan | what the data said |
|---|---|
| `OFF_BYE` at ">= 12 days" | the **week gap** is the definition; 12 days misreads 2020 DAL, twelve days from a COVID reschedule rather than a bye |
| a per-side `HOME` column | degenerates in a one-row-per-game layout: `AWAY_HOME` is 0 on every row ever written, so `NEUTRAL_SITE` carries it instead |
| all-null rows only at a franchise's debut | too narrow by construction — the smallest window is 3 games, so 2012 weeks 1-3 are all-null and only 16 of those 48 games are anybody's debut |

## Why the carried windows exist

Retention, both sides complete, overall and in **weeks 1-4** — the window in
which an NFL season is actually served:

| window | overall | weeks 1-4 |
|---|---|---|
| `ROLL3` | 80.7% | 24.0% |
| `ROLL5` | 67.5% | **0.0%** |
| `CARRY5` | 97.8% | **93.4%** |
| `CARRY8` | 96.5% | 93.4% |

`ROLL5` at 0.0% is structural rather than unlucky: five prior games inside a
season cannot exist before week 6. A 17-game season is why phase 3 is given both
kinds rather than one.

## The quarterback is the biggest known gap, and it is not in this phase

No player-level input of any kind: no quarterback, no injuries, no weather, no
travel distance, no betting lines, no quarter-based features. **Quarterback
availability is the largest single omission for this sport** and is recorded
here rather than left implicit — one position change moves an NFL team's
expected margin more than any team-level trailing average in this dataset will
capture, and a starter ruled out on the Friday is invisible to every column
written. The NBA's availability work (`CLAUDE.md` §15) is the shape a fix would
take and it is a separate phase, with the same live-source problem: a played
game's starter is knowable afterwards, and an unplayed game's is not.

## The prediction rule is a dependency, not a date

The NBA serves `MAX_DAYS_AHEAD = 1`, a rule about dates. Every feature here
reads only each team's own earlier games, and Elo updates only on games already
played, so:

> a fixture is predictable once **both teams' previous games are in history**

A Sunday fixture's features are final as soon as both sides' prior games are
recorded, whatever happens on the Thursday in between. Measured on the 208
unplayed 2026 fixtures: **15 are predictable now** — the whole of the next
week's slate — and the remaining 193 become so a week at a time. So this rule
yields a full slate at once where a date rule yields a single day. **A finding
for phase 4, not code in the inference service.**
