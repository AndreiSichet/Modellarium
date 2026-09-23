# Modellarium — Project Insights

**A personal sportsbook.** Predictions for NBA games, quarters, halves and
player statistics, produced by models trained from scratch on public data.

Written: 21 September 2026

---

## Who this document is for

Anyone who needs to understand this project without already knowing it, and
without necessarily being a programmer. Every technical term is explained the
first time it appears, and Chapter 2 is a glossary.

After reading this you should be able to explain: what was built, why each
piece exists, how the pieces talk to each other, how a number gets from a
model to a screen, how to run it, and how to prove it works.

This is the only document of its kind in the project. It absorbed an earlier
`backend_insights` guide that covered the Java backend alone — chapters 9 to
13 are that document, brought up to date, and nothing in it was dropped.

## A note on the code itself

The code carries very few comments. That is deliberate, and this document is
the reason: explanations belong somewhere a reader can find them in order, not
scattered across 137 files where each one is only visible to whoever happens
to open that file. The comments that remain are the ones that would be needed
*while editing that exact line* — a sign convention, a unit, a subtle ordering
requirement.

---

## Table of contents

1. [The project in one page](#1-the-project-in-one-page)
2. [Glossary](#2-glossary)
3. [How the whole app works](#3-how-the-whole-app-works)
4. [The data pipeline](#4-the-data-pipeline)
5. [The models](#5-the-models)
6. [The continuous training pipeline](#6-the-continuous-training-pipeline)
7. [The inference service](#7-the-inference-service)
8. [The injury sidecar](#8-the-injury-sidecar)
9. [The backend — what it is, and the database](#9-the-backend--what-it-is-and-the-database)
10. [The backend — every file, explained one by one](#10-the-backend--every-file-explained-one-by-one)
11. [The backend — how a request travels through the system](#11-the-backend--how-a-request-travels-through-the-system)
12. [The backend — the seven endpoints, exact contracts](#12-the-backend--the-seven-endpoints-exact-contracts)
13. [The backend — the schedule sync job](#13-the-backend--the-schedule-sync-job)
14. [Frontend insights](#14-frontend-insights)
15. [Running everything](#15-running-everything)
16. [Testing everything](#16-testing-everything)
17. [Problems found and fixed](#17-problems-found-and-fixed)
18. [Design decisions and why](#18-design-decisions-and-why)
19. [The accuracy experiments — six ideas tried, six rejected](#19-the-accuracy-experiments--six-ideas-tried-six-rejected)
20. [Known limitations and what comes next](#20-known-limitations-and-what-comes-next)
21. [Quick reference](#21-quick-reference)

---

## 1. The project in one page

### What it does

Modellarium predicts the outcome of NBA basketball games before they are
played. For any game it can reach, it produces **eighteen separate numbers**
across three families.

**Family one — the whole game (seven numbers, the original set)**

1. **Moneyline** — the probability that the home team wins (0 to 1)
2. **Spread** — by how many points the home team wins or loses
3. **Totals** — the combined points scored by both teams
4. **Rebound margin** — the home team's rebounds minus the away team's
5. **Total rebounds** — both teams' rebounds added together
6. **Assist margin** — the same idea, for assists
7. **Total assists** — the same idea, for assists

**Family two — the first quarter and the first half (six numbers)**

Each period gets three: who leads, by how much, and the combined score. A
quarter is twelve minutes and a half is twenty-four, so these are harder to
predict than a whole game — and measurably so (Chapter 5). Two of the six
carry warnings that travel with the number all the way to the screen; see
Chapter 12, because this is the part most easily got wrong.

**Family three — individual player statistics (five numbers per player)**

Points, rebounds, assists, three-pointers made, and the three added together.
Returned for up to ten players per team, so a single request can produce a
hundred numbers.

Nothing is scraped from a bookmaker and nothing is guessed. Every number comes
from models trained on eleven seasons of real NBA results.

### What it is made of

Six independent pieces, each of which can be understood on its own:

1. **The data pipeline** (Python) — downloads eleven seasons of NBA data and
   turns it into a table a model can learn from.
2. **The models** (Python, scikit-learn and XGBoost) — the trained files that
   turn a row of features into a prediction.
3. **The inference service** (Python, FastAPI) — a small web service that
   loads the models and answers prediction requests.
4. **The injury sidecar** (Python) — a second small service whose only job is
   turning the NBA's injury-report PDF into structured data.
5. **The backend** (Java, Spring Boot, PostgreSQL) — the application server.
   It stores predictions, caches the fixture list, and is the only thing the
   website talks to.
6. **The frontend** (React) — the website.

### Why it exists

It is a portfolio project, but not only that. Genuine predictive accuracy was
a goal from the start, not just a working demonstration — which is why a
substantial part of this document is about measurements that *failed* to
improve anything, and why those results were kept rather than quietly dropped.

Basketball was chosen over tennis for a specific reason: the author's
employer's machine-learning team was already building a tennis model, and this
project needed to be unambiguously independent — public data and original code
only.

### The single most important fact about it today

**The data ends on 12 April 2026** — the last day of the 2025-26 regular
season. The 2026-27 season opens on 20 October 2026.

The models can only score a game **one day past the newest data they have**
(Chapter 5 explains why this limit is structural rather than a bug). So right
now there is no real game the app can predict, and the website says so plainly
rather than showing an empty page. This resolves itself when the pipeline is
re-run with new-season data.

---

## 2. Glossary

Terms are grouped by the part of the system they belong to, not listed
alphabetically. Read this if any word below is unfamiliar; everything later
assumes these.

### Machine-learning terms

**Feature** — one input number a model learns from. "The home team's average
points over its last ten games" is a feature. This project's main model uses
38 of them.

**Label** (or *target*) — the answer the model is trying to predict, known for
past games because they were played. "Did the home team win?" is a label.

**Training** — showing a model thousands of past games, each with its features
and its known label, until it can guess the label from the features alone.

**Model** — the file that results. Give it 38 numbers, it gives you one back.

**Inference** — using a trained model to make a prediction. The opposite of
training.

**Rolling average** — an average over a moving window of recent games. "Points
over the last five games" is a rolling average; it changes as games are played.

**Elo rating** — a single number summarising a team's strength, which goes up
when it wins and down when it loses, by an amount depending on how surprising
the result was. Borrowed from chess.

**Leakage** — accidentally letting a model see the answer among its inputs.
Predicting a game's total points using that game's own points is leakage. It
produces an amazing-looking model that is worthless. Preventing it is a
recurring concern in Chapter 4.

**Baseline** — the simplest possible prediction, used as a yardstick. If a
model cannot beat "always predict the home team wins", it has learned nothing.

**MAE** (mean absolute error) — the average size of a prediction's miss,
ignoring direction. An MAE of 10.7 on margins means predictions are out by
about 10.7 points on average. Lower is better.

**Log loss** — a score for probability predictions that punishes confident
wrong answers much harder than uncertain ones. Lower is better. It is the
honest way to score a model that outputs a percentage.

**Holdout / test set** — games deliberately kept back from training, used to
measure how a model does on data it has never seen. Scoring a model on games
it trained on flatters it, sometimes enormously (Chapter 6 has a case where
this made a model look 12% better than it was).

### Web and network terms

**API** — "Application Programming Interface". A set of addresses (URLs) that
another program can call to ask for data or make something happen. Unlike a
website, the answer is data, not a page to look at.

**Endpoint** — one specific address in an API. `GET /api/teams` is an endpoint
that returns the list of teams.

**HTTP / GET / POST** — HTTP is the language programs use to talk over the web.
GET means "give me some data, change nothing". POST means "here is some data,
create something with it". This backend has four GETs and three POSTs.

**HTTP status code** — a three-digit number in every response saying how it
went:

| Code | Meaning |
|---|---|
| 200 | success |
| 400 | you sent something wrong (the caller's fault) |
| 404 | not found |
| 500 | the server broke (our fault) |
| 502 | a service we depend on answered, but badly |
| 503 | a service we depend on is unavailable |

Distinguishing 400 from 500 matters: a 400 tells the caller to fix their
request, a 500 tells them to report a bug.

**JSON** — a plain-text way of writing structured data, understood by nearly
every language. For example:
`{ "id": 1610612737, "name": "Atlanta Hawks", "abbreviation": "ATL" }`

**Service** — a program that runs continuously waiting to answer requests.
This project has four.

**Container / Docker** — a way of packaging a program with everything it needs
to run, so it behaves the same on any machine. Each of the five parts of the
running system is a container.

### Database terms

**Database / PostgreSQL / table / row / column** — a database stores
information permanently on disk. PostgreSQL (often "Postgres") is the database
program used here. Data lives in **tables**, which are like spreadsheets:
**columns** are the fields (name, date), **rows** are the individual records.

**Primary key** — the column that uniquely identifies a row in a table. No two
rows may share one.

**Foreign key** — a column that points at a row in another table. The `game`
table has a `home_team_id` column pointing at a row in `team`. The database
refuses to store a game whose team does not exist, which is how it keeps the
data honest.

**Transaction** — a group of database changes treated as one all-or-nothing
unit. Either every change happens or none does. Prevents half-finished states,
such as a game being saved but its prediction not.

**Idempotent** — an operation that produces the same result whether it runs
once or twenty times. The schedule sync is idempotent: running it repeatedly
does not create duplicate fixtures.

### Java and Spring terms

**Entity** — a Java class that mirrors one database table. One object in
memory = one row in the table. There are six: `Team`, `Game`, `Prediction`,
`Player`, `QuarterHalfPrediction`, `PlayerPropPrediction`.

**ORM / JPA / Hibernate** — "Object-Relational Mapping": technology that
automatically translates between Java objects and database rows, so nobody has
to write database commands by hand. JPA is the standard; Hibernate is the
implementation Spring Boot uses.

**Repository** — an interface listing the database questions we want to ask.
Spring writes the actual code. Remarkably, declaring a method named
`findByPlayedFalseOrderByGameDateAsc()` makes Spring read that *name* and
generate a query for "games that are not played, sorted by date". No code is
written by hand.

> **Caution, learned the hard way:** this name-reading has limits, and it does
> not warn you when you exceed them. See 17.9.

**Service class** — a class holding business logic, the actual
decision-making steps. Sits between the controller and the repositories.

**Controller** — the class that receives web requests and returns responses.
It maps a URL to a piece of code.

**DTO ("Data Transfer Object")** — a simple object built purely to be sent
over the network, deliberately separate from the entity. The entity is shaped
for the database; the DTO is shaped for whoever consumes the API. Keeping them
separate means changing the database does not automatically change the public
API, and vice versa.

**Lazy loading** — when a `Game` is loaded, its two `Team`s are *not* fetched
immediately. A placeholder goes there instead, and the real data is fetched
only if someone asks. This avoids wasted work. The catch: the fetch only works
while the database session is still open. Ask too late and you get an error
(17.2).

**N+1 query problem** — fetching a list with one query, then firing one more
query per item in that list. Twenty items means twenty-one queries. It is
invisible when the list is short and painful when it is long, which is exactly
why it tends to be discovered late. This backend had one, and it became real
the moment the fixture list started being cached (17.8).

**Entity graph** — a way of telling Hibernate "when you load these rows, load
their relations at the same time", turning a lazy load per item into one
combined query. The cure for one flavour of N+1.

**Scheduled job / cron expression** — code that runs automatically on a timer
rather than because somebody asked. A cron expression is the compact notation
describing when: `0 0 */6 * * *` means "at zero seconds, zero minutes, every
sixth hour".

**Spring Boot** — the Java framework doing the heavy lifting: starting a web
server, connecting to the database, converting Java objects to JSON, and
wiring classes together.

**Maven / `pom.xml`** — Maven is the build tool that compiles the Java code
and downloads libraries. `pom.xml` is its configuration file.

**Lombok** — a library that generates repetitive Java code automatically.
Writing `@Data` on a class makes Lombok generate all the get/set methods
invisibly, keeping the file short.

**Jackson** — the library that converts between Java objects and JSON. It
matters here because Spring Boot 4 upgraded to Jackson 3, which moved to
different package names — a detail that caused a compile failure (17.10).

### General

**Stale** — computed from data that is no longer current. The app tracks and
displays this rather than hiding it.

---

## 3. How the whole app works

This chapter follows a single prediction from raw data to the screen. Later
chapters go deeper on each step.

### 3.1 The six parts, and how they connect

```
    +-------------------+
    |  data-pipeline    |  Python. Downloads raw NBA results and turns them
    |                   |  into clean tables. Calculates "features" - the
    |                   |  inputs the models learn from, such as a team's
    |                   |  scoring average over its last 10 games, days of
    |                   |  rest, and Elo rating. Also does this for quarter
    |                   |  scores and for individual players.
    +-------------------+
             |
             |  writes CSV files (spreadsheet-like text files)
             v
    +-------------------+
    |  ml-training      |  Python. Trains the models on those CSVs. Produces
    |                   |  THREE sets: 7 whole-game models, 6 quarter/half
    |                   |  models, and 10 player-prop files.
    +-------------------+
             |
             |  saves models to ml-training/models*/
             v
    +-------------------+
    | inference-service |  Python (FastAPI) on port 8000. Loads all the
    |                   |  models into memory once at startup. Given two team
    |                   |  IDs and a date, builds the inputs and returns
    |                   |  predictions as JSON.
    +-------------------+
             ^
             |  asks, only when predicting: "who is unavailable today?"
             |
    +-------------------+
    |  injury-service   |  Python (FastAPI) on port 8001. The smallest part.
    |                   |  The NBA publishes its injury report as a PDF, and
    |                   |  reading a PDF needs a Java program - so this is the
    |                   |  only container in the system with Java installed.
    |                   |
    |                   |  Its whole job is turning that PDF into a list of
    |                   |  players and statuses. It does NOT work out what
    |                   |  that means for a prediction; the inference service
    |                   |  does that, because it already holds the player
    |                   |  history needed to interpret it.
    |                   |
    |                   |  If it is switched off, predictions still work. They
    |                   |  simply say "roster availability unknown" rather
    |                   |  than failing - which is also what happens between
    |                   |  seasons, when the NBA publishes nothing.
    +-------------------+
             |
             |  HTTP (a network call), JSON in and JSON out
             v
    +-------------------+
    |  backend          |  Java (Spring Boot) on port 8080. Calls the
    |                   |  inference service, stores results in a PostgreSQL
    |                   |  database, caches the NBA fixture list, and exposes
    |                   |  a clean API for the website.
    +-------------------+
             |
             |  HTTP, JSON
             v
    +-------------------+
    |  frontend         |  React. The website people actually use.
    +-------------------+
```

**Why split it up like this?** Each part can be worked on, restarted or
replaced without touching the others. The models can be retrained without
stopping the backend. The website can be rewritten without touching the
models. It also means each part uses the language best suited to it: Python
for machine learning, Java for a reliable business-logic and database layer.

**Important:** the backend cannot predict anything by itself. If the inference
service on port 8000 is not running, prediction requests fail. This is by
design, not a bug. **Browsing, however, still works** with the Python service
completely down, because the fixture list lives in the database — that was
verified deliberately (16.3).

**The backend does not talk to the injury sidecar.** Only the inference
service does. The sidecar is drawn above so that someone looking at the
running system does not find a container the diagram never mentions.

### 3.2 The two timescales

The system runs on two clocks, and confusing them is the main source of
misunderstanding.

**The slow clock — training.** Historical games are downloaded, turned into
features, and used to train models. This happens rarely: once when the project
was built, and thereafter on a weekly schedule that only acts when enough new
games have arrived. It takes hours and produces model files.

**The fast clock — serving.** A visitor opens the website and asks about a
game. Features for that specific unplayed game are computed on the spot, fed
to the already-trained models, and the answer comes back in well under a
second.

The connection between them is a set of files. Training writes models to disk;
serving reads them. Nothing else crosses between the two.

### 3.3 The end-to-end journey

```
    NBA public API
          |
          |  (slow clock, Chapter 4)
          v
   Data pipeline  ──────────────>  games_final.csv       (game history)
    11 seasons                     model_dataset.csv     (training table)
          |
          |  (slow clock, Chapter 5)
          v
   Model training ──────────────>  ml-training/models/*.json          (7)
                                   ml-training/models_quarter_half/   (6)
                                   ml-training/models_player_props/   (10)

    ─────────────────────────────────────────────────────────────────────

   Browser
      |  http://localhost:3000
      v
   Frontend (React)
      |  http://localhost:8080/api      (fast clock, Chapter 14)
      v
   Backend (Spring Boot)  <────────>  PostgreSQL         (Chapters 9-13)
      |  http://inference-service:8000
      v
   Inference service (FastAPI)  ────>  injury-service     (Chapters 7, 8)
      |
      |  loads at startup: games_final.csv + all 23 model artifacts
      v
   A prediction
```

### 3.4 What actually happens when someone opens the site

**Step 1 — the page loads.** The browser fetches the React application from
nginx. React decides which page to render from the address.

**Step 2 — the frontend asks for two things at once.** It calls the backend
for the fixture list (`GET /api/games/schedule`) and for data freshness
(`GET /api/health`). It needs both: a list of games is useless without knowing
which of them are recent enough to be scored.

**Step 3 — the backend answers the fixture list from its own database.** It
does *not* call the NBA API here. A background job refreshes the fixture cache
every six hours, so a click costs a database query rather than a multi-second
call to an external service.

**Step 4 — the backend passes the freshness question through.** `/api/health`
is deliberately *not* cached. It reports the date the model data ends, and
caching that would make the website compute "what is predictable" from an
out-of-date answer — wrong at exactly the moment it matters most, the first
request after new data arrives.

**Step 5 — the frontend decides what is reachable.** It computes the cutoff as
`dataAsOf + 1 day` and keeps only fixtures on that date.

**Step 6 — one prediction request per reachable game.** For each, the frontend
POSTs to `/api/predictions`. The backend looks up both teams, **calls the
inference service before writing anything to the database**, finds or creates
a row for the game, stores the prediction, and returns a summary. The ordering
is load-bearing and was a real bug once (17.3).

**Step 7 — the inference service does the actual work.** It already has the
models and the game history in memory from startup. It rebuilds the 38
features for this unplayed game (Chapter 7), asks each of the seven models for
its number, and returns all seven plus freshness metadata.

**Step 8 — the screen fills in.** Rows appear, grouped by league, capped at
five per league with a link to the rest.

**Step 9 — clicking a game opens the detail page**, which shows seven tabs
over all three prediction families. The full-game numbers are *reused* from
step 6 rather than re-requested; quarter/half is fetched once on arrival;
player props are fetched only if a player tab is opened.

### 3.5 The five running containers

| Container | Port | What it holds |
|---|---|---|
| `postgres` | 5432 | The database |
| `inference-service` | 8000 | Models, game history, feature computation |
| `injury-service` | 8001 | The PDF parser and a Java runtime |
| `backend` | 8080 | The application server |
| `frontend` | 3000 | nginx serving the built React bundle |

They find each other by **service name** on a shared private network — the
backend reaches the database at `postgres`, not `localhost`. The one exception
is the address the *browser* uses, which must be a real host address because
the browser is outside that network.

---

## 4. The data pipeline

Location: `data-pipeline/`. Language: Python (pandas, numpy, `nba_api`).

The pipeline's job is to turn "eleven seasons of NBA games" into a single
table where each row is one game and each column is either a feature or a
label. Every script reads files and writes files; they are run in order, and
each one can be re-run without harm.

### 4.1 The guiding principle: a model must never see the future

Almost every design decision in this chapter comes from one rule. When
building the features for a game played on 3 March, the pipeline may use only
information that existed *before tip-off on 3 March*.

This sounds obvious and is easy to break by accident. The usual way is a
rolling average: if "the last five games" includes the game being predicted,
the model is being handed part of its own answer. The pipeline prevents this
with a one-game shift applied before every rolling calculation — the single
most important line in the whole feature-building step.

### 4.2 Ingestion — getting the raw data

Four scripts in `data-pipeline/ingestion/`:

**`fetch_games.py`** — pulls the team-level result of every game across eleven
seasons (2015-16 to 2025-26). One file per season.

> **The trap that shaped this script.** The NBA API will happily return
> preseason games, All-Star games, playoffs, play-in games and NBA Cup finals
> mixed in with the regular season. They look identical in the output. The
> request must explicitly restrict itself to the regular season, or the models
> train on exhibition games where nobody is trying. The game's identifier
> encodes its type in its third digit — `2` means regular season — which is
> how the filter is verified.

**`fetch_player_boxscores.py`** — one request per game for the individual
player statistics. That is **13,199 requests**, against 11 for the whole
team-level pull.

At that volume the script needs properties the others do not:

  - **Resumable.** One file per game; an existing file is skipped. Stop it
    with Ctrl+C and start it again tomorrow and it continues.
  - **Crash-safe writes.** Each file is written under a temporary name and
    renamed only once complete, so an interrupted write cannot leave a
    half-file that a later run mistakes for finished.
  - **Rate-limited globally.** Several requests run in parallel to hide
    network latency, but all of them pass through one shared limiter, so the
    NBA API sees exactly one request per second no matter how many workers
    there are.
  - **Failure-tolerant.** A failed game is logged and skipped, never fatal. A
    re-run retries only the failures.

> **Two traps here.** First, the game identifier must be the zero-padded
> ten-character string; passing the number drops the leading zeros and the API
> returns *an empty result rather than an error* — a silent failure. Second,
> the API version changed midway through the original run, and the new version
> returns `0` for every statistic of a player who did not play, where the old
> one returned blank. Zero and "did not play" are different facts; the script
> normalises them back apart before writing.

**`fetch_quarter_scores.py`** — period-by-period scores, for the quarter and
half models. Three games out of 13,199 cannot be retrieved: the NBA API raises
an internal error on them, reproducibly. That is 0.02% of the data and is
accepted rather than worked around.

**`fetch_team_advanced_stats.py`** — pace and efficiency statistics. Built,
validated, and ultimately **not used** — see Chapter 5.

### 4.3 Validation — refusing to build on bad data

Three scripts whose only job is to check. They write nothing and they fail
loudly.

**`validate_games.py`** checks that the schema is consistent across seasons,
that no critical value is missing, that dates parse and fall inside their
season, that the win/loss column only ever says W or L, and — the one that
matters most — that **every game identifier appears exactly twice**, once for
each team. A game appearing once means a team's row was lost.

**`validate_player_boxscores.py`** runs over all 13,199 files. Its centrepiece
check is a cross-validation rather than a self-consistency check: **each
team's player points, summed, must equal that team's final score** in the
already-trusted team-level data. A brand-new data source is checked against
one that has been correct for months. All 13,199 games pass.

**`validate_quarter_scores.py`** has an unusual shape for a good reason. The
period scores contain **no overtime data** — on an overtime game the four
quarters sum to the regulation score, not the final score. So the obvious
check, "quarters must sum to the final", would fail on about 5% of games and
report a known structural gap as a data-quality problem. Instead the validator
splits games into two populations: exact equality for regulation games, a
bounded shortfall for overtime ones, and *quarters summing to more than the
final* as a hard failure in either case.

> **A real defect found here.** On one game the API's own reported total
> disagrees with its own period breakdown by a single point. The pipeline
> treats the trusted team-level score as the authority, which resolves the
> disagreement and lands the overtime classification at 12,486 regulation and
> 710 overtime games — matching an independent signal (minutes played) with
> zero disagreements.

### 4.4 Feature building — the seven steps in order

Each script reads the previous one's output.

**1. `build_games_table.py`** — combines all seasons into one table. Derives
which team was at home, and who the opponent was.

> The opponent is derived by **pairing the two rows that share a game
> identifier**, not by reading the text of the matchup description. About ten
> games have a corrupted description that is identical on both teams' rows —
> probably rescheduled fixtures — and text-parsing them would silently assign
> the wrong opponent. Those ten games are dropped. Result: 26,398 rows, one
> per team per game.

**2. `build_rolling_features.py`** — adds trailing five- and ten-game averages
for win percentage, points, plus-minus, field-goal percentage, rebounds,
assists and turnovers.

> Two things here are load-bearing. The averages **reset each season**,
> because a team in October is not a continuation of the team in April. And
> each is computed with a **one-game shift**, so a game's own result can never
> enter its own features. Early-season rows are therefore empty by design, and
> the count of empty rows is verified: 30 teams × 11 seasons × window size.

**3. `build_rest_days.py`** — days since that team's previous game, and
whether it is the second night of a back-to-back.

> Rest is capped at seven days so the summer break does not register as
> "extremely well rested". And a back-to-back is `rest == 1`, **not**
> `rest == 0` — two games on consecutive calendar days are one day apart, never
> zero. That was a real bug, caught here.

**4. `build_elo_ratings.py`** — a genuine sequential Elo rating for each team,
computed by walking the games in chronological order. Every team starts at
1500; the rating moves after each game. Between seasons each team's rating is
regressed one third of the way back toward 1500, reflecting roster turnover.

> This is the one step that cannot be vectorised — each game's rating depends
> on the previous game's result — so it is a plain loop. It was verified by
> recomputing samples by hand and matching to six decimal places.
>
> Its output, **`games_final.csv`, is the most important file in the
> project**: it is what the live inference service reads at startup.

**5. `build_player_rolling_minutes.py`** — per-player trailing averages over
their last ten *appearances* (not calendar games — a player who misses three
weeks resumes from where he left off).

**6. `build_team_availability.py`** — for each game, how many of a team's
regular rotation players did not appear, and how many minutes those absences
represent.

**7. `build_final_dataset.py`** — reshapes everything into one row per game,
with `HOME_` and `AWAY_` prefixed columns, and merges in the quarter/half and
advanced-stat columns. Result: **13,199 rows, 111 columns**.

> **The merges are strict on purpose.** If a game arrives without quarter data,
> the merge raises rather than quietly leaving the column empty — because a
> confident-looking number computed from a shortened window is worse than a
> crash. There is one deliberate exception: the advanced-stats merge warns and
> carries an empty value instead, because nothing downstream uses those
> columns.
>
> The set of games allowed to be missing is not hardcoded. It is read from the
> fetcher's own failure log, so it tracks reality instead of expiring silently
> the first time a different game fails.

**A sanity check worth knowing:** the finished dataset shows the home team
winning **56.4%** of games. That is the real, long-observed NBA home-court
advantage. Getting it out of the data by accident is a strong sign the
pipeline is correct.

---

## 5. The models

Location: `ml-training/`.

### 5.1 The headline finding

Three completely different kinds of model were tried on the full-game markets:

1. **A closed-form formula** — the win probability implied by the two teams'
   Elo ratings, with no learning at all.
2. **Linear and logistic regression** — the simplest real models.
3. **XGBoost** — gradient-boosted decision trees, a far more expressive family,
   and then a tuning search on top of it.

All three land in the same accuracy band on every one of the seven targets.

| Moneyline approach | Accuracy | Log loss |
|---|---|---|
| Always predict the home team | 0.5458 | — |
| Elo formula, no learning | 0.6674 | 0.6130 |
| Logistic regression | 0.6773 | 0.6045 |
| XGBoost | 0.6754 | 0.6067 |
| XGBoost, tuned | no meaningful improvement | |

**This is a ceiling in the information, not a failure of modelling.** When a
formula, a linear model and a tuned tree ensemble all agree, the limit is what
the features know, not how cleverly they are combined. The conclusion —
recorded rather than buried — was to stop tuning and go looking for new
information instead.

### 5.2 What broke the ceiling, and what did not

**Player availability: worked.** Adding four columns describing who was
missing improved margin prediction by **5.8%** — the first genuine break in
the ceiling.

| Target | Before | After | Change |
|---|---|---|---|
| **Margin (spread)** | 11.4013 | **10.7368** | **−5.83%** |
| Assist margin | 5.4732 | 5.3928 | −1.47% |
| Rebound margin | 7.5778 | 7.5095 | −0.90% |
| Total points | 15.2685 | 15.2322 | −0.24% |
| Moneyline (log loss) | 0.6067 | **0.5979** | −1.45% |

Confirmed three independent ways: the old result was deterministic and on
record three times at exactly 11.4013; a direct side-by-side comparison on
identical settings reproduced the gain; and the feature-importance breakdown
shows availability accounting for 15.8% of the model's total decision weight,
with "how many home players are out" the **third most important feature
overall**.

> **Five more experiments were run after these two, and all five were
> rejected. They are in [Chapter 19](#19-the-accuracy-experiments--six-ideas-tried-six-rejected),
> along with the diagnostic that finally made it possible to say which
> rejections were informative and which were not.**

**Advanced pace and efficiency statistics: did not work**, and the reason is
interesting. Adding all five metrics made margin prediction *worse*. The
hypothesis was that three of them (offensive, defensive and net rating)
duplicate information the model already has through points and plus-minus,
and were diluting it. Dropping those three and keeping only pace and shooting
efficiency restored the result exactly. So the correct conclusion is not
"advanced stats do not matter" but "**this encoding of them adds nothing
beyond what raw points and possessions already imply**" — a transferable
insight about *why* the ceiling holds.

### 5.3 What ships

**Twenty-three model artifacts in three directories:**

| Directory | Contents | Kind |
|---|---|---|
| `ml-training/models/` | 7 full-game models | XGBoost |
| `ml-training/models_quarter_half/` | 6 period models + manifest | Linear/logistic |
| `ml-training/models_player_props/` | 10 player models + manifest | Hybrid |

They are **siblings, never nested**. The inference service refuses to start
unless `models/` contains exactly the seven files it expects, so putting the
other sixteen there would break the running application at startup.

**The quarter/half models are linear, and that is the evidence rather than a
shortcut.** On all six period targets, XGBoost was *slightly worse* than
linear regression — between 0.2% and 0.9% worse, despite training on 1,266
more games. When a more expressive model with more data loses, the information
is not there to extract.

Three consequences follow from shipping linear models:

  - Each artifact bundles its own data scaler, so the two cannot drift apart.
  - **They cannot handle missing values at all.** Early in a season, when a
    team's recent form does not yet exist, they *refuse* rather than
    predicting badly — and the caller must say why.
  - The two winner models were trained only on periods that had a winner, so
    their output means `P(home leads | not tied)`. A tied quarter is a push,
    not a wrong prediction. This qualifier is displayed on screen, not hidden.

**The player-prop models are a hybrid**, the only one in the project. Neither
half is more accurate — they differ by under 1% — but they differ enormously
in *coverage*:

| | Games it can score |
|---|---|
| Linear | 225,060 of 280,943 |
| XGBoost | **280,943 of 280,943** |

The ~56,000 rows only XGBoost can handle are rookies, call-ups and players
returning from long absence. Refusing to predict for them is worse than
predicting slightly less well, so both ship and a routing rule in the manifest
— not the caller — decides which answers each request: complete history goes
to the linear model, anything else to XGBoost.

### 5.4 The period ordering, from four unrelated measurements

First quarters are less predictable than first halves, which are less
predictable than full games. That is unsurprising as a hypothesis. What makes
it a *result* is that four measurements derived from completely different
mechanisms produce it independently, with no exceptions:

| Measurement | Q1 | 1H | Full game |
|---|---|---|---|
| Correlation with the final margin | +0.475 | +0.674 | 1.0 |
| Home wins, having led the period | 71.5% | 77.9% | — |
| Best model accuracy | 0.5796 | 0.6343 | 0.6777 |
| Model complexity at convergence | lowest | middle | highest |

The last row is the strongest of the four, because it is the least direct: it
measures how much structure the optimiser still found worth fitting, and it
was never told which period it was looking at.

**A related real-world fact falls out of the same data:** home advantage
*accumulates*. The home team wins 51.8% of first quarters, 52.9% of first
halves, and 54.6% of games. Whatever produces home advantage builds over the
course of a game rather than being present at tip-off.

### 5.5 The one-day limit

The inference service will not predict a game more than **one day** past the
newest game in its data. This is permanent and structural, not a staleness
workaround.

The reason is the rest-days feature. Rest is measured from the last game in
the dataset. If there are unplayed fixtures between that game and the one
being predicted, the rest figure is measured against the wrong game and is
simply wrong. Rather than serve a confidently incorrect number, the service
refuses.

Everything visible in the application follows from this: the website shows
only today's games, and a game that cannot be scored has no page.

---

## 6. The continuous training pipeline

Location: `ml-training/continuous_retrain.py`, `model_evaluation.py`,
`verify_continuous_retrain.py`, and
`.github/workflows/continuous-retrain.yml`.

This is the only part of the project that runs on its own schedule and can
propose a change to what is in production.

### 6.1 What it does, and what it deliberately does not

Once a week it checks whether enough new games have arrived. If so it
refreshes the data, retrains the seven full-game models, compares the new
models against the current ones, and — if the comparison passes — **opens a
pull request**.

**It never deploys.** Merging is a human decision. The new models are written
to a directory nothing serves.

### 6.2 The gate is a regression guard, not an improvement bar

Every other comparison in this project asks *"does this new idea earn its
place?"* and holds the answer to a demanding bar. This one asks something
different: *"does the model still work, now that there is more data?"*

Retraining is not a hypothesis. So the gate is deliberately loose and
one-directional: a candidate may be slightly worse and still be promoted,
because month-to-month noise is real and a fresh model reflecting current
reality is worth a fraction of a percent. What it may not do is get
*meaningfully* worse.

### 6.3 The correctness fix at the centre of it

The obvious way to build this gate is to load the current production models
and score them on recent games. **That is provably wrong here**, and a
deliberately adversarial test found it rather than code review.

The production models are trained on the *entire* dataset with no holdout —
correctly so, because model selection was already finished by then. Which
means any recent window is part of their training data. Scored that way,
production looked like it achieved an error of **9.58** on margins against the
**10.74** it genuinely achieves on unseen games. It was being graded on its own
homework, and every candidate honestly holding data back looked 5-12% worse by
comparison.

The fix: the gate does not score the shipped model files at all. It takes the
production model's **architecture**, re-fits it on exactly the candidate's
training data, and scores both on games neither has seen. After the fix, all
seven targets moved from "appears to have regressed 5-12%" to within **±1.06%**.

### 6.4 Exit codes are a contract

```
0   a verdict was reached: promoted, or nothing worth retraining
1   a verdict was reached, and the gate refused to promote
2   no verdict was reached - it crashed on the way there
```

**1 and 2 were the same code until the first real run.** An unhandled error
exits 1, exactly like an honest refusal — so a crash during the data refresh
was indistinguishable from a completed comparison, and the wrapper reported it
with the reassuring message *"candidate rejected, the guard worked as
intended"*. The reassuring message, for the one case that had not happened.

The distinction that matters is not "did something go wrong". It is **"was a
comparison actually made"**. A refusal is a measurement. A crash is the absence
of one, and says nothing whatever about model quality.

### 6.5 Why it runs on a self-hosted machine

The job needs 13,199 downloaded box-score files that are not stored in version
control. A cloud runner starts from an empty disk and would re-download all of
them at one request per second — roughly 3.7 hours, every week, with 13,199
calls to the NBA API each time. The whole resumable design assumes a disk that
persists.

Running it on a persistent machine also sidesteps two other problems rather
than mitigating them: bandwidth limits on large stored files, and an unknown
datacentre address making tens of thousands of API calls.

**Cloud-runner defaults are actively wrong on a persistent machine**, and two
had to be turned off. The default checkout behaviour deletes untracked files —
which would delete 61 MB across 39,594 files at the start of every run and
recreate the exact cold start that self-hosting exists to avoid. Silently, with
the only visible symptom being a job that suddenly takes hours.

### 6.6 How it was proven before real data existed

Real new games do not exist until October. A "detect new data" check would
correctly find nothing every time, so waiting would mean shipping a completely
unexercised pipeline and finding its bugs at the worst possible moment.

`verify_continuous_retrain.py` picks a date in the middle of real data,
pretends that is today, and lets the games that genuinely happened afterwards
stand in for new ones. **Nothing is mocked**: the truncated dataset is real,
the "old" model is really trained on it, and the candidate is judged by the
same gate the production job uses, invoked as a separate process so the
new-data trigger has to fire on its own.

---

## 7. The inference service

Location: `inference-service/app.py`. Framework: FastAPI.

### 7.1 What it is

A small Python web service that loads every model and the entire game history
**once at startup**, and then answers prediction requests from memory. It is
the only part of the system that knows how to turn a matchup into a set of
features.

### 7.2 Its endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Model counts per family, the date the data ends, and whether that is stale |
| `GET /schedule` | Upcoming fixtures from the NBA API, regular season only |
| `POST /predict` | The seven full-game markets |
| `POST /predict/quarter-half` | The six period markets |
| `POST /predict/player-props` | Both teams' player boards |

### 7.3 It refuses to start if anything is missing

The service holds an explicit list of the seven models it expects. If the
files on disk do not match that list exactly, **it refuses to boot**. This is
deliberate: a service that starts successfully and then produces wrong answers
is far worse than one that does not start.

This guard has already earned its place twice — once catching seven model
files that had never actually been saved into version control, and once
confirming that sixteen newly added models had correctly gone into sibling
directories rather than breaking the running application.

### 7.4 Rebuilding features for a game that has not been played

This is the hardest part of the service. The models expect 38 numbers in a
specific order, computed exactly the way the training pipeline computed them.
A mismatch between how a feature is built for training and how it is built for
serving is one of the most damaging and least visible bugs in machine
learning.

The defence is simple and absolute: the serving code **imports every constant
and formula directly from the pipeline scripts** rather than restating them.
The Elo starting rating, the K-factor, the season-boundary rule, the rest-days
cap, the list of rolling metrics and window sizes — all imported. There is no
second copy to drift.

Verification: the serving code was asked to rebuild the features for 200
historical games and compared against what the pipeline produced for the same
games. They match to about 1 part in 10 trillion — floating-point noise,
nothing more. The check deliberately includes the awkward cases: games with
incomplete recent form, season-opening games where Elo has just been
regressed, and back-to-backs.

### 7.5 A dependency that no code imports

`scikit-learn` is a required dependency even though nothing in the service
imports it. Loading an XGBoost model through its friendlier interface — which
this service does deliberately, because that interface validates feature names
— runs initialisation code that fails without scikit-learn present.

This is invisible during development, because the shared development
environment already has it for training. It would surface only as a startup
crash in a clean container.

### 7.6 A pandas trap, hit once

When the upcoming-fixture list is empty, attaching a derived column to the
empty table **re-inflates it** into a full table of blank rows. The fix is to
derive columns *before* filtering rather than after. It failed only when
nothing was scheduled — which is to say, only in the off-season, which is
exactly the current state.

---

## 8. The injury sidecar

Location: `injury-service/`.

### 8.1 Why a separate service exists

The availability features (Chapter 5) need to know who is expected to miss
tomorrow's game. For past games that comes from the box score. For a game that
has not been played, the only source is the NBA's official injury report —
**published as a PDF, several times a day, and never archived**.

Parsing that PDF requires a library that runs a Java program underneath. The
inference service is a Python image with no Java runtime, and adding one to it
would mean carrying a Java installation into an image whose actual job is
arithmetic.

### 8.2 The boundary is the parse, not the feature

The obvious split would be a service that returns the four finished
availability numbers. That is the wrong one.

Exactly **one step** needs Java: turning the PDF into rows. Everything
afterwards — matching player names, looking up their typical minutes,
aggregating into counts — is ordinary data work over a file the inference
service already carries.

```
injury-service      PDF  ->  JSON     (the Java runtime lives here, and only here)
inference-service   JSON ->  features (matching and aggregation, where the data is)
```

Putting the feature computation in the sidecar would have duplicated an 80 MB
data file into a second image and created a second place where the matching
logic could drift away from what the models were trained on.

### 8.3 Three findings that each cost a wrong turn

**The index page cannot be scraped.** The NBA's injury-report page visibly
lists links to every report, but none of them are in the page a program
receives — that list is inserted by JavaScript after loading. A link-scraper
finds only the unrelated PDFs baked into the page and will cheerfully download
a committee brochure instead.

**The report addresses are predictable**, so the index page is skipped
entirely. Reports appear roughly hourly but not on a fixed minute, so the
service walks backwards in fifteen-minute steps until it finds one.

**A missing report answers "forbidden", not "not found."** Code that treats
only "not found" as an expected miss would report every ordinary gap as an
unexpected error.

### 8.4 Two policies, both deliberate

**Five report statuses map to a binary.** The models learned from a played /
did-not-play signal, so *Out* and *Doubtful* map to absent, and *Questionable*,
*Probable* and *Available* map to present. **An unrecognised status raises an
error** rather than defaulting to present, so a sixth status introduced by the
NBA fails loudly at the cheapest possible place to notice it.

**"Not yet submitted" is kept separate and never counted as zero absences.**
Teams file at different times, so a report routinely contains games where one
or both teams have not reported. Those rows carry no player and no status.
Treating them as "nobody is out" would be the same unknown-as-zero mistake
this project has now hit four separate times. A prediction for a team with an
unfiled report should refuse, not quietly score that team as healthy.

### 8.5 No report is a normal answer

When nothing is published — between seasons, or overnight — the service
returns a successful response saying so, not an error. Nothing being published
is an ordinary state of the world rather than a failure of the service.

The inference service treats an unknown roster as a *supported* state: the
prediction still completes, with the four availability features blank and the
other 34 intact.

---

## 9. The backend — what it is, and the database

Location: `backend/`. Java 21 on Spring Boot 4, with PostgreSQL.

### 9.1 What the backend is for

The backend does **not** do the predicting. It is the layer that receives
requests from the website, asks the Python service for the numbers, stores
what comes back, and returns a clean answer. Think of it as the office that
takes the order, files the paperwork and replies — not the factory.

Concretely it does four things the inference service should not: it
**remembers** predictions, it **caches** the fixture list, it **translates**
naming conventions between the Python side and the browser side, and it is the
**single address** the website talks to.

### 9.2 The database

```
Database name: basketball_predictor
Runs on:       localhost, port 5432 (the PostgreSQL default)
Username:      postgres
```

The tables are created **automatically** by Hibernate from the Java entity
classes when the application starts. Nobody wrote the table definitions by
hand. This is controlled by `spring.jpa.hibernate.ddl-auto=update` in
`application.properties`.

> **A warning about which database you are looking at.** This machine can have
> **two** PostgreSQL servers both claiming port 5432: the Windows service
> `postgresql-x64-17`, and the one inside Docker. A backend started on the host
> silently connects to the **Windows** one. Checking the Docker one and finding
> tables missing does not mean they were never created — it means you looked in
> the wrong place. This cost real time; when someone says "check the database",
> ask which.

#### Table: `team` — 30 rows, all NBA teams

| Column | Type | Meaning |
|---|---|---|
| `id` | bigint | The real NBA team ID, e.g. 1610612737 for Atlanta |
| `name` | text | "Atlanta Hawks" |
| `abbreviation` | text | "ATL" |

The id is **not** auto-generated. The NBA already assigns every team a
permanent unique number, and the pipeline data uses those numbers. Reusing
them means a row here lines up with the data files with no translation step.

These 30 rows are inserted automatically at startup by `TeamSeeder` if the
table is empty. That is permanent behaviour, not a one-off script: a fresh
Docker volume starts with an empty database, and without these rows nothing
works.

#### Table: `game` — hundreds of rows, the cached fixture list

| Column | Type | Meaning |
|---|---|---|
| `id` | bigint | Auto-generated by the database, 1, 2, 3… |
| `nba_game_id` | bigint | The official NBA game ID. Still NULL. |
| `home_team_id` | bigint | Points at `team.id` |
| `away_team_id` | bigint | Points at `team.id` |
| `game_date` | date | The day it is played |
| `played` | boolean | false for a fixture that has not happened |

This table used to be empty until somebody asked for a prediction. The
schedule sync job now fills it in advance — a recent run cached 451 fixtures.
A row still holds no predictions until one is requested.

Crucially, both paths create rows through the **same code** (`GameLookup`), so
a fixture cached by the job and the same fixture predicted by a user are one
row, not two.

#### Table: `prediction` — the seven whole-game numbers

| Column | Type | Meaning |
|---|---|---|
| `id` | bigint | Auto-generated |
| `game_id` | bigint | Points at `game.id` |
| `home_win_probability` | float | 0 to 1 |
| `home_margin` | float | Positive = home team wins by that much |
| `total_points` | float | |
| `rebound_margin` | float | |
| `total_rebounds` | float | |
| `assist_margin` | float | |
| `total_assists` | float | |
| `data_as_of` | date | Newest game the models could see |
| `stale` | boolean | True if that date is more than 2 days old |
| `predicted_at` | timestamp | When this row was created |

**Predictions are never overwritten. Asking twice creates two rows.**

#### Table: `player` — created on demand

| Column | Type | Meaning |
|---|---|---|
| `id` | bigint | The real NBA player ID |
| `name` | text | "Jayson Tatum" |

**Not seeded in bulk, unlike teams, and that is deliberate.** There are 30
teams and they never change; there are thousands of players and any one
request concerns about twenty of them. A row appears the first time a
prediction actually returns that player, and is reused afterwards. The table
therefore grows to exactly what has been asked for.

#### Table: `quarter_half_prediction` — the six Q1 / first-half numbers

| Column | Type | Meaning |
|---|---|---|
| `id` | bigint | Auto-generated |
| `game_id` | bigint | Points at `game.id` — the **same** game |
| `q1_spread` | float | Home margin in the first quarter |
| `q1_total` | float | Combined first-quarter points |
| `q1_winner_probability` | float | See the warning in 12.2 |
| `half1_spread` | float | |
| `half1_total` | float | |
| `half1_winner_probability` | float | |
| `data_as_of` | date | |
| `stale` | boolean | |
| `predicted_at` | timestamp | |

**What is deliberately not stored here, and the rule behind it:** a value that
would be identical on every row this table will ever hold is a constant, not
data. The two winner numbers each come with a confidence label and a warning
about what the probability actually means. Those describe the **model**, not
this request — they would be the same on row one and row ten thousand — so
they are attached when the response is built, not written to disk.

#### Table: `player_prop_prediction` — five numbers per player per request

| Column | Type | Meaning |
|---|---|---|
| `id` | bigint | Auto-generated |
| `game_id` | bigint | Points at `game.id` |
| `player_id` | bigint | Points at `player.id` |
| `team_id` | bigint | Which side he was on **for this call** |
| `predicted_points` | float | |
| `predicted_rebounds` | float | |
| `predicted_assists` | float | |
| `predicted_threes_made` | float | |
| `predicted_pra` | float | Points + rebounds + assists |
| `model_used` | text | "linear" or "xgb" |
| `predicted_at` | timestamp | |

`team_id` is stored per row so that a later trade does not rewrite history:
the row says who he was playing for when the prediction was made.

**`model_used` IS stored, unlike the quarter/half confidence labels, and the
contrast is the test for whether something belongs in a table.** Which of two
models answered varies per player and per request, because it depends on how
much recent history that player had at that moment. It is a fact about this
row. A confidence label is not.

**What is not stored:** whether the roster availability was known, and the
accompanying caveat text. Those describe a whole team for one request, so
writing them here would repeat one value across ten rows and let the copies
drift apart. They are assembled at the response layer.

Up to 20 rows are written per request (ten players, two teams).

---

## 10. The backend — every file, explained one by one

All Java files live under
`backend/src/main/java/com/andreisichet/basketball_predictor/`.

### `BasketballPredictorApplication.java` — the starting point

Starts everything. Carries two annotations:

- `@SpringBootApplication` — the standard one.
- `@EnableScheduling` — **without it, every timed job in the application is
  silently inert**: no error, no warning, the method simply never runs. That
  failure mode is why it is worth naming here.

### `model/` — the entities, one class per database table

| File | What it is |
|---|---|
| `Team.java` | 30 rows, real NBA ids, never generated |
| `Game.java` | One fixture. Auto-generated id |
| `Prediction.java` | The seven whole-game numbers |
| `Player.java` | Real NBA player id, created on demand |
| `QuarterHalfPrediction.java` | The six Q1/1H numbers, hanging off the **same** `Game` row as everything else |
| `PlayerPropPrediction.java` | One player's five numbers, plus which model produced them |

### `repository/` — the database queries

| File | Queries |
|---|---|
| `TeamRepository.java` | Standard operations only |
| `GameRepository.java` | Three: unplayed games soonest-first (with an `@EntityGraph` so both teams load in the same query, 17.8); unplayed games inside a date window, used by the browse endpoint; find one game by teams + date |
| `PredictionRepository.java` | Latest prediction for one game, and all predictions for a **batch** of games |
| `PlayerRepository.java` | Standard operations only |
| `QuarterHalfPredictionRepository.java` | Standard operations only |
| `PlayerPropPredictionRepository.java` | Standard operations only |

### `service/` — the decision-making

**`InferenceClient.java` — the most important class in the backend.** The
single place that talks to the Python service. Five methods, one per Python
endpoint, each doing the network call and the same translation of failures:

| What Python did | What we say |
|---|---|
| replied 400 | **400** — the caller's input was wrong, not our bug |
| replied 500 | **502** |
| was unreachable | **503** |

Before this existed, that rule lived inside one service. Copying it into two
more is how one endpoint quietly starts returning 500 where the others return
400.

> It is genuinely single **since `getHealth()` moved here**. For two sessions
> the class documented itself as the one point of contact while a second
> network client sat beside it in `ScheduleService`. The docstring was only
> made true once it was.

**`GameLookup.java`** — finds a team or rejects the request, and
finds-or-creates a `Game`. **One game row per matchup per date, whichever
markets are being predicted.** This is what guarantees the scheduler and a
user's request land on the same row.

| File | Role |
|---|---|
| `PredictionService.java` | The original seven models |
| `QuarterHalfPredictionService.java` | The six Q1/1H markets |
| `PlayerPropPredictionService.java` | Both teams' player boards, and the find-or-create for `Player` rows |
| `GameService.java` | Lists unplayed games with their latest prediction. Uses two queries instead of one-per-game (17.8) |
| `ScheduleService.java` | Serves the cached fixture list and the live freshness check. Holds no network client of its own |
| `ScheduleSyncService.java` | Fetches the fixture list and caches it |

### `config/` — startup and timers

| File | Role |
|---|---|
| `TeamSeeder.java` | Inserts the 30 teams if the table is empty |
| `WebConfig.java` | Allows the website (a different address) to call this API — without it the browser blocks everything |
| `ScheduleSyncJob.java` | **When** the schedule sync runs: every six hours, and once at startup. Nothing about **how** — that is `ScheduleSyncService`'s job |

### `controller/` — the web addresses

| File | Endpoints |
|---|---|
| `PredictionController.java` | Three POST endpoints |
| `GameController.java` | Upcoming games, and the fixture list |
| `TeamController.java` | The 30 teams |
| `HealthController.java` | Freshness of the underlying data |

### `dto/` — the shapes sent over the network

Two groups, and the split is deliberate.

**Talking to Python** (field names in `under_score` style, as Python writes):
`InferenceRequest`, `InferenceResponse`, `InferenceHealth`,
`InferenceScheduledGame`, `InferenceQuarterHalfResponse`,
`InferencePlayerPropsResponse`.

**Talking to the website** (field names in `camelCase`, as JavaScript
expects): `TeamDto`, `PredictionDto`, `GameSummaryDto`, `HealthDto`,
`ScheduledGameDto`, `QuarterHalfSummaryDto`, `PlayerPropsResponseDto`.

Keeping them apart means one naming convention survives across the whole
public API even though the service behind it uses another.

---

## 11. The backend — how a request travels through the system

### 11.1 A prediction request

Someone sends:

```
POST http://localhost:8080/api/predictions
{ "homeTeamId": 1610612737,
  "awayTeamId": 1610612738,
  "gameDate": "2026-04-13" }
```

| Step | What happens |
|---|---|
| 1 | Spring routes it to `PredictionController` and turns the JSON into a `PredictionRequest` object |
| 2 | The controller hands it to `PredictionService` and does nothing else |
| 3 | A database transaction begins |
| 4 | `GameLookup` finds both teams. If either is unknown: stop, 400, nothing saved |
| 5 | `InferenceClient` POSTs to `http://localhost:8000/predict`, field names converted to the underscore style Python expects |
| 6 | The Python service computes the inputs, runs the seven models, and returns the numbers plus `data_as_of` and `stale` |
| 7 | If Python replied 400, `InferenceClient` turns that into our 400 with Python's own explanation attached. **Nothing has been written yet** — that ordering is the point (17.3) |
| 8 | `GameLookup` finds the existing game or creates one. Since the sync job caches fixtures in advance, it usually **finds** one |
| 9 | A `Prediction` row is saved |
| 10 | A `GameSummaryDto` is assembled **while the session is still open** |
| 11 | The transaction commits |
| 12 | Spring converts the DTO to JSON, status 200 |

The quarter/half and player-prop endpoints follow the identical shape. The
only differences are which Python endpoint is called, which table is written,
and what the response looks like.

### 11.2 A browse request

When someone opens the website, the page makes **two calls at once**:

```
GET /api/games/schedule?daysAhead=14      the fixture list
GET /api/health                           how fresh the data is
```

**The fixture list comes from the database.** One query for the games in the
window, one for the 30 teams to attach names, done. No network call to Python
at all.

**The freshness check still goes live to Python on every request, and must.**
`data_as_of` tracks the pipeline's data. Caching it alongside the fixtures
would mean the website deciding which games are predictable from an
out-of-date cutoff — and it would go wrong at exactly the moment it matters
most, the first request after the pipeline is rerun.

Before this arrangement both calls reached Python, and the fixture one was a
multi-second call to the NBA. Now one small fast call remains. **Verified by
watching the Python service's own logs during a click: one `/health` hit, zero
`/schedule` hits.**

---

## 12. The backend — the seven endpoints, exact contracts

Base address while running locally: `http://localhost:8080`

### 12.1 `POST /api/predictions` — the seven whole-game numbers

**Request:**
```json
{ "homeTeamId": 1610612737, "awayTeamId": 1610612738,
  "gameDate": "2026-04-13" }
```

**Response 200:**
```json
{ "id": 4,
  "homeTeamAbbreviation": "ATL",
  "awayTeamAbbreviation": "BOS",
  "gameDate": "2026-04-13",
  "played": false,
  "latestPrediction": { "...the seven numbers, dataAsOf, stale, predictedAt" } }
```

**Errors:** 400 for same team / unknown team / date too far ahead (with
Python's own reason), 503 if Python is unreachable.

> **Note:** this endpoint returns the game id under the name `id`, while the
> two below use `gameId`. That inconsistency is known and deliberately left
> alone — the website already reads `id`, and renaming a shipped field to win
> tidiness would break a working page for no gain.

### 12.2 `POST /api/predictions/quarter-half` — the six Q1 / first-half numbers

**Request:** identical to 12.1.

**Response 200:**
```json
{ "gameId": 4,
  "homeTeamAbbreviation": "ATL",
  "awayTeamAbbreviation": "BOS",
  "gameDate": "2026-04-13",
  "prediction": {
    "q1Spread": -1.12,  "q1Total": 59.26,
    "q1WinnerProbability": 0.447,
    "q1WinnerConfidence": "low",
    "q1WinnerInterpretation": "P(home leads | not tied)",
    "half1Spread": -1.17, "half1Total": 117.89,
    "half1WinnerProbability": 0.454,
    "half1WinnerConfidence": "medium",
    "half1WinnerInterpretation": "P(home leads | not tied)",
    "dataAsOf": "2026-04-12", "stale": true, "daysBehind": 139,
    "predictedAt": "..." } }
```

**The four extra fields are required and never omitted.** This is the part of
the API most easily got wrong, so it is worth being blunt about why.

`"q1WinnerConfidence": "low"` is not decoration. That model is right about 58%
of the time against a 52% baseline — genuinely better than guessing, but close
enough to it that showing the number next to the others without a label would
present a near-coin-flip as an equal peer.

`"P(home leads | not tied)"` is a statement about what the number **means**. A
quarter can end level; a game cannot. These two models were trained only on
periods that had a winner, so the probability is conditional on the period
being decided. It is **not** the same quantity as the whole-game win
probability, and must not be displayed as though it were.

Neither field is read from the database — see 9.2.

### 12.3 `POST /api/predictions/player-props` — both teams' player boards

**Request:** identical to 12.1.

**Response 200 (abbreviated):**
```json
{ "gameId": 4, "gameDate": "2026-04-13",
  "homeTeam": {
    "teamId": 1610612737, "teamAbbreviation": "ATL",
    "availabilityKnown": false,
    "availabilityNote": "21 players ... AVAILABILITY UNKNOWN - no injury
                         report was available, so nobody has been excluded.
                         This is not a clean bill of health.",
    "players": [
      { "playerId": 1630559, "playerName": "...",
        "predictedPoints": 22.48, "predictedRebounds": 3.39,
        "predictedAssists": 3.71, "predictedThreesMade": 3.54,
        "predictedPra": 29.58, "modelUsed": "linear" } ] },
  "awayTeam": { "...same shape..." },
  "dataAsOf": "2026-04-12", "stale": true, "daysBehind": 139,
  "predictedAt": "..." }
```

**`availabilityNote` is always present, in both states.** Today it always
carries the unknown-availability warning, because the NBA publishes nothing
between seasons. The field is deliberately modelled as "what is known about
availability" rather than "the warning", so that when that gap closes it
carries the good news instead. **A field that only exists while something is
broken gets deleted the moment it is fixed, and then there is nowhere for the
fixed state to go.**

Up to ten players per team.

### 12.4 `GET /api/games/schedule?daysAhead=14` — the fixture list

**Response 200:** a list of
```json
{ "homeTeamId": 1610612752, "homeTeamAbbr": "NYK",
  "homeTeamName": "New York Knicks",
  "awayTeamId": 1610612755, "awayTeamAbbr": "PHI",
  "awayTeamName": "Philadelphia 76ers",
  "gameDate": "2026-10-20" }
```

**Same contract as before, different source.** This used to call the NBA
through Python on every request; it now reads the cached table. The response
was compared field-for-field, old versus new, on 451 fixtures: identical.

**One real behavioural change:** it can only return what the sync job has
cached, which is 120 days by default. Asking for a year out used to reach the
live API and get a year; now it gets 120 days.

An empty list is a valid answer, not an error — there genuinely are no NBA
fixtures in the next fourteen days during the off-season.

### 12.5 `GET /api/games/upcoming` — stored games with their latest prediction

**Response 200:** a list of `GameSummaryDto`, same shape as 12.1's response.

Returns hundreds of rows rather than a handful, because the sync job fills the
table. That is what turned its long-documented N+1 query into a measured 486
database queries per call; it is now 2. See 17.8.

> **This endpoint currently has no caller.** The frontend rebuild removed the
> browse view that used it. The endpoint and both its regression tests are
> still live and still correct — they are simply guarding something nothing
> calls. See 19.13.

### 12.6 `GET /api/teams` — all 30 teams

**Response 200:**
```json
[ { "id": 1610612737, "name": "Atlanta Hawks", "abbreviation": "ATL" } ]
```

### 12.7 `GET /api/health` — how fresh the underlying data is

**Response 200:**
```json
{ "status": "ok",
  "modelsLoaded": { "team": 7, "quarter_half": 6, "player_props": 10 },
  "dataAsOf": "2026-04-12",
  "daysBehind": 139,
  "stale": true }
```

`modelsLoaded` **used to be a single number** and is now a breakdown per model
family. That change on the Python side is what broke this endpoint once — see
17.7. The breakdown exists so that a failure names **which** family failed to
load, instead of showing a total that still looks plausible when one family
loaded twice and another not at all.

503 if the Python service is unreachable. Note this endpoint genuinely depends
on Python, while 12.4 no longer does.

---

## 13. The backend — the schedule sync job

### What it does

Every six hours, and once at startup, it asks the Python service for the NBA
fixture list and makes sure every fixture has a row in the `game` table. It
writes nothing else — no predictions, no player rows.

### Why it exists

The browse page used to reach the live NBA API through Python on every single
button press. That is a multi-second call to somebody else's server, repeated
per user, to fetch a season calendar that changes about as often as one would
expect a calendar to. Fetching it on a timer and serving it from Postgres
turns a slow dependency into a fast local read.

### The two triggers, and why both

A six-hour timer alone would leave a fresh deployment — or a fresh Docker
volume, which starts with an empty database — serving an empty fixture list
for up to six hours. That is a visible step backwards from the old always-live
behaviour. So startup runs one sync immediately, **in addition** to the timer.

The two were confirmed to be genuinely independent code paths rather than one
masquerading as the other, by checking which internal thread each ran on: the
startup one on `main`, the timed ones on `scheduling-1`.

### Why it never crashes

If the Python service is restarting when a cycle fires, that cycle logs a
warning and does nothing. Six hours later it tries again. Taking the whole
application down because a dependency was briefly unavailable would be a far
worse outcome than one skipped refresh.

### Why running it twice is harmless

It creates a fixture through the same find-or-create used by every prediction
endpoint, so a fixture already present is simply found. A real run logs
`451 new` the first time and `0 new, 451 already present` every time after.

### What it skips, every single cycle

**Six fixtures are skipped as "unknown team", and this is correct rather than a
fault.** The NBA schedule includes tournament knock-out and play-in games whose
participants are not yet decided; those come back with a placeholder team id of
zero. A game that is real but not yet fully determined cannot be shown or
predicted, so it is skipped — and the count stays in the log line rather than
being silently dropped, because a skip that happens every cycle deserves to
stay legible.

### Settings (`application.properties`)

```
schedule.sync.days-ahead=120        how far ahead to cache
schedule.sync.cron=0 0 */6 * * *    when to run
```

The cadence is a setting rather than a constant so it can be shortened to
every minute for a smoke test without rebuilding anything — which is exactly
how it was verified.

---

## 14. Frontend insights

Location: `frontend/`. React, created with create-react-app.

### 10.1 The shape of the site

Four routes, all sharing one two-column layout: a sports rail on the left, the
content on the right.

| Address | Page |
|---|---|
| `/` | About — what the project is and how it works |
| `/predictions` | **General** — today's games across every sport |
| `/predictions/:sport` | **Upcoming** — today's games for one sport |
| `/predictions/:sport/:league` | **League** — everything available for one league |
| `/predictions/:sport/:league/:gameId` | **Game detail** — seven tabs over all 18 markets |

General and Upcoming are **the same component** with a different scope. Two
components would be two copies of the subsections, the shortcuts, the cap and
the links — and the first edit to either is where they start telling the user
different things.

### 10.2 Everything is fetched once, in one place

The data is fetched by the shared layout, not by the pages. Because all four
routes match that layout, moving between them does not remount it and does not
refetch.

**This matters more than it sounds.** Every game on screen costs a POST that
writes a database row, because the prediction endpoint is append-only by
design. A page that refetched on navigation would turn browsing into
row-writing.

The game detail page takes this further: its full-game numbers are **reused**
from what the layout already fetched, because re-requesting them would return
the same seven numbers and write a second row. Only the period markets are
fetched on arrival, and player props only if a player tab is opened — twenty
players by five statistics is heavier than the other two calls combined, and
many visits never open that tab.

> **This rate is a known cost, flagged where it happens.** The real fix is a
> backend change — a batch endpoint, or a path that returns a recent existing
> prediction instead of always writing — not a cache in the browser, which
> would hide the rate without changing it.

### 10.3 Saying what is not known

A theme running through the whole interface: **a prediction without its caveat
misleads**, and a redesign is exactly when caveats get dropped for looking
cluttered.

Three survive to the screen as visible elements, never as hover-only tooltips:

  - **The confidence label** on each period market. The first-quarter winner
    model ships labelled *low* deliberately. It is genuinely better than
    chance, but close enough to a coin flip that rendering it identically to
    the others would misrepresent it.
  - **The conditional qualifier** on both winner markets — the literal text
    `P(home leads | not tied)`. A tied quarter is a push, not a wrong
    prediction, and the probability means nothing without that condition
    stated.
  - **The availability note**, rendered **verbatim**, exactly as the API
    returns it. It says the absence of an injury report is not a clean bill of
    health. Softening it would invert its meaning, and an unfiltered roster
    would read as a confirmed lineup.

A **stale badge** appears whenever the underlying data is old, which is the
same instinct one layer up.

### 10.4 The design language

A single file defines every colour, typeface, size and spacing step. **No
component may hardcode any of them.** That is not tidiness — it is what makes
it possible to change the whole visual language in one place, and a single
hardcoded colour is what makes that no longer true.

There are exactly two exemptions, both principled: the token file itself, and
the team colour table, because published team colours are facts about the
world rather than design choices.

Contrast was measured rather than eyeballed:

| | On the page background | Verdict |
|---|---|---|
| Body text | 15.87:1 | passes comfortably |
| Secondary text | 4.67:1 | passes |
| Accent, as text | 3.64:1 | **failed** — a darker variant was added at 4.82:1 |
| Accent, as an underline | 3.64:1 | passes (different threshold applies) |

Two accent tokens rather than one darkened one, because the thresholds
genuinely differ: an underline or a focus ring is judged at a lower bar than
text, and darkening those too would dull the identity to fix a problem they do
not have.

### 10.5 Accessibility

  - **Landmarks.** One main region per page, wrapping the *content column* —
    the sports rail sits beside it as a navigation region, not inside it.
  - **The detail tabs are a real tab list.** Arrow keys move between them,
    Home and End jump to the ends, and only the active tab is in the keyboard
    tab order — so pressing Tab moves *past* the bar rather than through all
    seven buttons.
  - **The league shortcut bar is deliberately not a tab list**, and labelling
    it as one would state something false. Its buttons scroll the page to a
    section; they reveal and hide nothing.
  - **Team badges are hidden from screen readers**, which is correct only
    because the team is always named in text beside them.

### 10.6 What tests can and cannot see

The test environment has **no layout engine**. Elements have no size, nothing
overflows, nothing scrolls, and sticky positioning has no meaning. A layout
bug cannot fail a test — not "does not currently", *cannot*.

This is not a hypothetical limitation. Two pages shipped visibly broken while
every test passed and the compiled stylesheet contained exactly the right
rules. In one case a sticky element's rule was correct the whole time, and a
*different* rule on its parent had removed the room it needed to move.

So the discipline is: tests cover behaviour, text and structure; layout,
scrolling and anything involving the production web server are checked in a
browser, and reports say which was which.

---

## 15. Running everything

There are two ways: everything in Docker (one command), or each piece by hand.
Docker is easier; by hand is better for developing one part.

### Option A — everything at once, with Docker

From the project root:

```
docker compose up --build
```

Five containers start: PostgreSQL, the inference service, the injury sidecar,
the backend, and the website. When it settles:

```
website   http://localhost:3000
backend   http://localhost:8080
Python    http://localhost:8000
injury    http://localhost:8001
```

> **Important:** if the Windows PostgreSQL service is running, it already owns
> port 5432 and the container's database will not be the one the host can
> reach. See the warning in 9.2.

### Option B — by hand

**Prerequisites**

- PostgreSQL running, with a database named `basketball_predictor`.
  `psql` on this machine: `C:\Program Files\PostgreSQL\17\bin\psql`
- Java installed (this machine has Java 25).
- Python installed, with the inference service's requirements installed.
- The trained model files present in `ml-training/models*/`.

**Step 1 — start the Python inference service**

```
cd inference-service
python -m uvicorn app:app --port 8000
```

Confirm it is alive at `http://localhost:8000/health`. Expected — note that
`models_loaded` is an object, not a number:

```json
{"status":"ok",
 "models_loaded":{"team":7,"quarter_half":6,"player_props":10},
 "data_as_of":"2026-04-12","days_behind":162,"stale":true}
```

Leave this terminal running.

**Step 2 — start the backend**

A second terminal. The database password comes from an environment variable,
never from a file in the repository.

PowerShell:
```
cd backend
$env:DB_PASSWORD = "postgres"
.\mvnw spring-boot:run
```

Git Bash:
```
cd backend
DB_PASSWORD=postgres ./mvnw spring-boot:run
```

Wait for `Started BasketballPredictorApplication`, then a moment later the
startup schedule sync:

```
Running the initial schedule sync so a fresh database is not empty.
Schedule sync: 457 fixtures fetched (120 days ahead), 451 new, ...
```

**Step 3 — start the website**

```
cd frontend
npm start
```

**Step 4 — the injury sidecar, if you want availability data**

```
docker compose up -d injury-service
```

Predictions work without it; they simply report availability as unknown.

**Stopping.** Ctrl+C in each terminal.

> Ctrl+C on Maven sometimes leaves the Java process running and holding port
> 8080. The next start then fails with "Port 8080 was already in use". To
> clear it, in PowerShell:
> ```
> Get-NetTCPConnection -LocalPort 8080 -State Listen |
>   ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
> ```

### Seeing a populated site out of season

Because no real game is currently predictable, the frontend carries a set of
development fixtures behind an environment switch:

```
cd frontend
$env:REACT_APP_DEV_FIXTURES = "1"
npm start
```

These are shaped exactly like the backend's real responses and deliberately
exercise the awkward cases: a nine-game spread across three dates so the
five-soonest cap is visible, a team identifier that is not in the colour table
so the fallback badge appears, and both halves of the player-prop hybrid so
the model tag cannot silently become a constant.

**The switch is an environment variable rather than a code flag or a web
address**, because a production build made without it cannot take that path at
all.

### Regenerating everything from scratch

Run the data-pipeline scripts in the order of Chapter 4, then the training
scripts. This takes hours, most of it waiting on rate-limited downloads.

---

## 16. Testing everything

### 16.1 What runs automatically

Four independent jobs on every push. They are deliberately parallel rather
than chained — they fail for different reasons, and chaining would let one
slow job hide the others' results.

| Job | What it proves |
|---|---|
| **backend** | The full application starts against a real database, and all tests pass |
| **frontend** | 88 tests pass and the production build compiles |
| **inference-service** | The service genuinely boots and reports all models loaded |
| **docker-build** | Every container image still builds from a clean checkout |

The inference job is a **real boot**, not an import check. It is a regression
test for two bugs this project actually had: a missing dependency that made
model loading fail, and a broken startup entry point where nothing answered at
all.

### 16.2 The backend's automated tests

```
cd backend
DB_PASSWORD=postgres ./mvnw test
```

There are **fourteen**. Only some need a running PostgreSQL; the
continuous-integration pipeline provides one.

**`contextLoads`** — starts the whole application against a real database.
Proves every class wires together and every setting is valid. This is the test
that catches "it does not even start".

**`InferenceWireShapeTest`** — ten tests, no database needed. The largest
group, and the one guarding the failure this document keeps returning to: the
Java side and the Python side describing the same response differently, with
nothing to notice.

There are **five** Java classes that read a Python response, one per endpoint
the backend calls. Each has a test here that feeds it a real captured response
and checks the values arrive intact.

Three things about it are deliberate and easy to undo by accident:

- **The fixtures are real**, captured from a running service with `curl` and
  saved exactly as they arrived, in `src/test/resources/fixtures/`. A fixture
  typed out from the Java class would change whenever that class changed, so
  it could never notice the very drift it exists to catch. This is not
  hypothetical: it is why the 17.7 bug went unnoticed.
- **The tests look deep inside.** Two of these responses nest two or three
  levels down — a list of teams, each holding a list of players, each holding
  a set of predicted numbers. A test that only checked the outer object would
  pass while the inside arrived empty, which is the quietest way this could
  break. So the player test checks a named player's five actual numbers, three
  levels in.
- **They use the same JSON reader the running application uses**, not a fresh
  one built for the test. A test that reads JSON differently from production
  is testing something other than production. An earlier version did exactly
  that; the two readers were measured and happen to behave identically today,
  so nothing was actually broken — but a test that validates the wrong thing
  cannot tell you when that stops being true.

**What they catch and what they do not**, decided on purpose: if Python
**removes** a field or **changes its type**, these tests fail. If Python
**adds** a field, they pass. That is the right way round — a new field is a
harmless addition the backend should tolerate, while the two destructive
changes are the ones that have actually caused outages.

**`GameServiceUpcomingDateTest`** — two tests. Checks that "upcoming games"
means games in the future. Nothing ever marks a game as played, so "not
played" had quietly stopped meaning "still to come" once hundreds of fixtures
were being cached. Includes a boundary case pinning that a game dated **today**
still counts as upcoming, since the two spellings of that rule are one word
apart.

**`GameServiceQueryCountTest`** — asks for the upcoming-games list and counts
how many database queries that took, using Hibernate's own statistics. Fails
if the count grows with the number of games. Exists because that endpoint
quietly went from 2 queries to 486 without anything failing (17.8).

It asserts a small **ceiling** rather than an exact number: what must hold is
that the count does not scale, and pinning it to exactly 2 would break on
unrelated changes to how Hibernate batches.

### 16.3 The checks worth repeating after any change

These were each run deliberately, and each caught or confirmed something.

**a) The numbers must match Python exactly.** Call the Python service directly
and call the backend, then compare every number. **106 values were compared
this way across the two newer endpoints: zero differences.** The backend must
never round, average or reinterpret — it stores and forwards.

**b) All three prediction types must share one game row.** Call all three
endpoints for the same matchup and confirm the `game` table gained exactly one
row, not three.

**c) A rejected request must leave nothing behind.** Send a date too far ahead
to each endpoint and confirm the game, player and prediction counts are all
unchanged. This is a real bug that happened once (17.3), so it is checked
rather than assumed.

**d) Asking twice must not duplicate players.** Call the player-props endpoint
twice for the same matchup: the prediction rows should double, and the player
rows should not move at all.

**e) The browse page must survive Python being down.** Stop the Python service
and call `/api/games/schedule`: it should still return the cached fixtures.
`/api/health` should return 503.

### 16.4 Checking the backend by hand

```
curl http://localhost:8080/api/teams
curl http://localhost:8080/api/health
curl "http://localhost:8080/api/games/schedule?daysAhead=120"

curl -X POST http://localhost:8080/api/predictions ^
  -H "Content-Type: application/json" ^
  -d "{\"homeTeamId\":1610612737,\"awayTeamId\":1610612738,\"gameDate\":\"2026-04-13\"}"
```

The same body works for `/api/predictions/quarter-half` and
`/api/predictions/player-props`.

### 16.5 The test counts

| Area | Tests |
|---|---|
| Frontend | 88 across 7 files |
| Backend | 14 |
| Data pipeline | Validation scripts, run manually over the full corpus |
| Models | Verification harnesses, run manually |

### 16.6 The most valuable tests in the project

They are not the ones with the highest counts.

**The feature-replay check.** Rebuilds features for 200 historical games and
compares against what the pipeline produced. This is the check that would
catch training and serving drifting apart — the most damaging and least
visible failure in the whole system.

**The wire-shape tests.** Covered in 16.2.

**The query-count test.** Asserts a *ceiling* rather than an improvement,
which is the only reason the second hidden performance problem was found.

**The simulated-cutoff harness.** Proves the retraining pipeline works using
history in place of the future, with nothing mocked.

---

## 17. Problems found and fixed

These are the most valuable part of this document. Each was a real fault, each
was silent, and each would have been much harder to diagnose later.

### 17.1 Lombok generated nothing on Java 25

**Symptom:** the very first compile failed with `constructor Team cannot be
applied to given types; required: no arguments`.

**Cause:** Java 23 and later no longer run code generators found among the
project's libraries. Lombok is such a generator. It was present, but the
compiler ignored it, so none of the getters, setters or constructors it was
supposed to write existed.

**Why it is confusing:** the error points at the line that *uses* the missing
constructor, not at the real cause. Nothing mentions Lombok.

**Fix:** name Lombok explicitly in the `maven-compiler-plugin`'s
`annotationProcessorPaths` in `pom.xml`.

### 17.2 "Open Session In View" was hiding a real problem

**Background:** Spring Boot has a feature, on by default, that keeps the
database session open for the entire duration of a web request — including
while the response is being converted to JSON. It exists to make lazy loading
"just work".

**The problem:** it makes broken code look correct. The POST endpoint
originally returned the database `Prediction` object directly. Converting that
to JSON required loading the `Game`, and then both `Team`s. Those loads
happened during JSON conversion, silently, only because the session was still
open.

**Fix:** `spring.jpa.open-in-view=false`, plus returning DTOs instead of
database objects.

**What happened immediately:** the POST endpoint started returning 500 with
"Could not initialize proxy — no session". **That failure was not caused by the
change. The change revealed it.**

The setting is now permanently off, so any future code that forgets to load
what it needs fails immediately and loudly rather than working by luck.

### 17.3 Failed requests were leaving junk games in the database

**Symptom:** the first call to `GET /api/games/upcoming` returned **three**
games when only one had ever been successfully created. Two were nonsense —
one for a date the system had refused, and one for "Boston versus Boston".

**Cause:** the original order of operations created the game row **first** and
called the Python service **second**. When Python rejected the request, the
game row had already been written and stayed there.

**Fix, two layers:**
1. Call the Python service **before** creating anything.
2. Mark the whole operation `@Transactional`.

**This is now enforced in one place.** When the two newer prediction endpoints
arrived, the ordering rule was written into `GameLookup` and all three
services follow the same sequence: validate, call Python, then write. Three
separate copies of that ordering is exactly how one of them would eventually
get it wrong.

### 17.4 The database could not retrofit auto-generated IDs

`Game.id` was changed from "supply your own value" to "let the database
generate it". The automatic schema updater (`ddl-auto=update`) can **add**
tables and columns, but it cannot **alter** an existing column to become
auto-generating.

**Fix:** confirmed the tables were empty, dropped them, let Hibernate recreate
them, then **checked** the result rather than assuming it.

**Still relevant:** the three tables added later were brand new, which is the
case `ddl-auto=update` handles perfectly. The limitation only bites on
*changing* something.

### 17.5 Two Spring Boot 4 migration surprises

**a)** `RestClient.Builder` could not be found. In Spring Boot 4 the tooling
for *calling* other services moved into a separate package. Fixed by adding
`spring-boot-starter-restclient`.

**b)** The property for including error messages was renamed. The old
`server.error.include-message` is deprecated; the correct name is now
`spring.web.error.include-message`.

### 17.6 Team abbreviations were not where expected

The plan was to load teams from `games_final.csv`, which turned out to contain
team IDs and names but **not** abbreviations. The raw season files do contain
all three, and were checked for consistency across eleven seasons before being
relied on: exactly 30 distinct combinations, no team ever changing name or
abbreviation.

### 17.7 The health endpoint broke because Python changed shape

**Symptom:** the website's fixture-fetching failed entirely, while every
prediction endpoint kept working perfectly. The backend log said
`Error while extracting response for type InferenceHealth`.

**Cause:** the Python service's `/health` used to report `models_loaded` as a
single number. It now reports a breakdown per model family. The Java class
receiving that response still declared the field as a plain integer. Jackson
cannot put an object into an integer, so the whole response failed to parse.

**Why only the browse page broke:** that is the only part of the website that
needs the freshness figures before it can decide which fixtures are reachable.
Predictions never touch `/health`.

**Why it was invisible until runtime:** nothing connects the two sides at
compile time. The Java code compiled, the application started, and the
mismatch only appeared when a real response arrived. **A compiler cannot see
across an HTTP boundary.**

**Fix:** change the field to a map. Then, more importantly, add a test so the
shape is asserted somewhere. That test was deliberately checked by temporarily
putting the bug back: it failed with exactly the production error, which is
**the only way to know a guard actually guards**.

### 17.8 486 database queries for one request

**Symptom:** none. Nothing failed. `GET /api/games/upcoming` simply became
slow.

**Cause:** it fetched the games in one query, then fired one more query per
game to find that game's latest prediction. This was a known, accepted
trade-off — with a handful of games it was 2 or 16 queries.

**What changed:** the schedule sync job began caching hundreds of fixtures.
The same untouched code went to a measured **486 queries for a single call**.

> **This is worth dwelling on: the code did not change and did not break.** A
> decision made elsewhere — cache the fixture list — turned an accepted
> trade-off into a real problem. Accepted trade-offs need re-checking when
> their assumptions move.

**Fix, in two stages, and the second was found by accident:**

| | Queries |
|---|---|
| Before | **486** |
| Fetch every latest prediction for the batch in one query | 32 |
| Also load each game's two teams alongside, with an `@EntityGraph` | **2** |

The remaining 30 were a **second, quieter N+1**: loading each game's two teams
one at a time. It was bounded at 30 (the number of teams) rather than growing
with the games, which is exactly why it had been hidden behind the louder one.

**Stage 2 was only found because the new test asserted a small ceiling rather
than "better than before".** A looser assertion would have passed at 32 and
left it there.

### 17.9 A repository method name that would have compiled and lied

While fixing 17.8, the obvious method name was `findTopByGameIdIn(ids)` — read
it aloud and it sounds like "the top prediction for each of these games".

**It is not.** Spring Data's `Top` limits the **whole result set**, not each
group. That method would have compiled, run without error, and returned
exactly **one** prediction for the entire batch — silently giving every game
but one a blank prediction.

There is no way to express "newest row per group" in a Spring Data method name
at all. The working version fetches every prediction for the batch, newest
first, and picks the first one seen per game.

**Why this is worth recording:** the convenience of Spring reading method names
is genuinely remarkable, and that is exactly what makes its limits dangerous.
It does not warn you when a name means something other than what it reads like.

### 17.10 Spring Boot 4 uses Jackson 3, which moved

Writing the test for 17.7 failed to compile:
`package com.fasterxml.jackson.databind does not exist`.

**Cause:** Spring Boot 4 upgraded to Jackson 3, which moved its main package
from `com.fasterxml.jackson.databind` to `tools.jackson.databind`. Only the
**annotations** stayed on the old name, which is why every existing file in
this project still imports `com.fasterxml.jackson.annotation` and looks like
it is using the old version.

Jackson 3 also handles dates without the extra module the old one needed.

**A small process note:** the code editor flagged this immediately and it was
initially dismissed as a stale editor index. The build proved the editor
right. Worth a moment's check before assuming tooling is wrong.

### 17.11 Only one of five response shapes was actually guarded

After 17.7, a test was added pinning the shape of the health response. That
covered **one of the five** places the backend reads a Python response. The
other four had nothing, and any of them could have broken the same silent way.

There was also a subtler problem with the test itself: it built its own JSON
reader rather than using the one the running application uses. Those are two
different objects that can be configured differently, so the test was checking
something *adjacent to* production rather than production.

**How the list was found.** Not by reading the folder of classes, which holds
both directions — responses coming in from Python, and the backend's own
replies going out to the website. Only the incoming ones can be broken by a
change on the Python side. So the list was taken from the one class that
actually talks to Python, by looking at which types it asks for. That gave
**five**, not the six an earlier note had claimed: the sixth was the *request*
body, which the backend sends rather than receives.

**Two things that went wrong while building it, both worth keeping:**

- **The first capture was not faithful.** PowerShell read the response as
  Western European text when it was actually UTF-8, which turned the player
  name "Vít Krejčí" into garbled characters. Saved that way, the fixture would
  have been a record of a bug rather than of the service. `curl` writes the
  bytes exactly as they arrived.
- **The first attempt at deliberately breaking each test** — to prove it would
  notice — reported that none of them noticed. That was the checking script
  reading the wrong summary line, not the tests failing to work. Once
  corrected, four of five caught their sabotage immediately. The fifth needed
  a better sabotage: the field being removed was one the test expects to be
  empty, and **a test that expects nothing cannot tell you when nothing
  arrives.** Breaking a field that carries real text made it fail correctly.

**The lesson: a guard that has never been shown to fail has not been shown to
work.**

### 17.12 Unknown treated as zero — four separate times

The same mistake in four unrelated places across the Python side:

1. A blank minutes field read as "played zero minutes".
2. An API version change that returned `0` for players who did not appear,
   where the previous one returned blank.
3. Injury-report rows for teams that have not filed, which would count as
   "nobody is out".
4. Tied quarters, where coding a tie as "away team led" would assert something
   false about 611 games.

**Zero and unknown are different facts.** Every one of these produces a
confident, plausible, wrong number rather than an error.

### 17.13 Green that means nothing

- A crash and an honest refusal shared an exit code, so a crash was reported
  with a reassuring message.
- A data guard printed a *warning* and continued, which leaves the step green
  — and a green step at the top of a job that then runs for three hours is
  indistinguishable from having no guard at all.
- A test asserted a value that could not change, so it could not fail.
- A test's helper picked the wrong line of output and reported five failing
  checks as passing.

**A misleading green is worse than a red.** Every one of these was converted
into a hard failure.

### 17.14 Files that existed locally but not in version control

Three times, a genuine runtime dependency sat inside an ignore rule that was
correct when the folder held only disposable output. Container builds read the
local disk and do not care what is committed; the automated build is the first
thing that only ever sees what *is*.

**The standing rule: any file added to a container build must have its
version-control status checked in the same change.** For large files there is
a second half — without an extra flag the automated build receives a tiny
placeholder instead of the file, and the failure looks like a corrupt file
rather than a missing one.

### 17.15 The same footgun in two places, resolved differently

A data-grouping function that appears to return "the last row for each group"
actually returns *each column's last non-blank value independently* — so it can
assemble a row that never existed.

In one place this was a real bug, admitting a player to a roster on a
minutes-figure carried over from the previous season. Fixed.

In the other it was **measured rather than assumed**: only one column can be
affected, the team and date provably never mix, and 257 of 1,578 players are
touched. Left in place with a dated decision point, because changing it alters
the inputs of shipped models and the better moment to decide is when the live
path can be tested end to end.

### 17.16 An operating-system behaviour that looked like a configuration error

Containers were configured to restart automatically. After a machine restart
they came back. After a *shutdown and power-on*, they did not.

Windows Fast Startup does not fully shut down — it hibernates, which gives the
container system a clean window in which it *stops* the containers. That sets
the same flag a manual stop sets, and the chosen restart policy honoured it.

Diagnosed from the boot-type event log rather than guessed: the two tests split
exactly along it. **The lesson is that "restart the machine" and "turn it off
and on again" are different code paths.**

### 17.17 Two pages shipped visibly broken with every test green

The test environment has **no layout engine**, so a layout bug *cannot* fail a
test. Two pages shipped broken while every test passed and the compiled
stylesheet contained exactly the right rules:

- An SPA-routing gap that exists only in the production web server, because
  the development server does that job for free.
- A sticky-positioning rule that was correct the whole time, while a
  *different* rule on its parent had removed the room it needed to move.

**Checking that a declaration is present is not checking the box it lands in.**

---

## 18. Design decisions and why

**Why six separate pieces rather than one program?** Each has a different
dependency footprint. The training stack is large and needed only for
training; the Java runtime is needed only for parsing a PDF. Combining them
would mean every container carries everything. It also lets each part use the
language best suited to it.

**Why real NBA IDs for teams and players, but generated IDs for games?** Teams
and players already have permanent, official, unique numbers, and the pipeline
data is keyed by them. Reusing them removes an entire class of translation
bugs. Games are different: this app invents game records for matchups, and
those have no official number yet. So the database assigns one, and the
`nba_game_id` column waits for the real value.

**Why are teams loaded in bulk but players created on demand?** There are 30
teams, they never change, and nothing works without them — so they are
inserted at startup. There are thousands of players, they change constantly,
and any one request concerns about twenty. Shipping and maintaining a list of
thousands to support twenty would be work with no payoff.

**Why three separate prediction endpoints instead of one?** The request bodies
are identical, which is exactly the argument someone would make for merging
them. The responses are not: seven plain numbers, six numbers carrying
confidence labels and conditional-probability warnings, and two nested rosters
of five numbers each. A single endpoint returning all of that would be mostly
empty fields whichever way it was called, and every caller would need to know
which parts to ignore.

**Why extract `InferenceClient` and `GameLookup` when the third caller
arrived?** Two copies of something is a judgement call. Three is a pattern, and
the point at which the copies start drifting. Concretely: three copies of "find
the game or create it" is how one fixture ends up with three rows, and three
copies of the error translation is how one endpoint quietly starts returning
500 where the others return 400.

**Why keep every prediction instead of overwriting?** A prediction is a
statement made at a moment in time with the information available then.
Overwriting would destroy the ability to ask "what did we think a week ago,
and were we right?".

**Why store `data_as_of` and `stale`?** Because a prediction from stale data
looks exactly like a good one. Recording the freshness makes an invisible
weakness visible, permanently.

**What decides whether something gets stored in a table?** A value that would
be identical on every row the table will ever hold is a constant, not data.

- The confidence label on a quarter-winner prediction is the same for every
  request ever made. **Not stored**; attached to the response.
- Which of two models produced a player's numbers depends on how much history
  that player had at that moment. **Stored.**
- Whether roster availability was known describes a whole team for one request,
  so storing it per player would repeat one value across ten rows and let the
  copies drift. **Not stored.**

**Why does the schedule come from the database but freshness come live?** A
season calendar changes rarely, so caching it costs nothing and saves a slow
third-party call on every click. The freshness figure tracks the underlying
pipeline data and changes precisely when it matters. Caching that would mean
the website deciding what is predictable from an out-of-date cutoff, and it
would be wrong at the worst possible moment — the first request after new data
lands.

**Why is the Python service called before anything is saved?** Because most
failures come from that call. Doing it first means the common failure path
never touches the database at all.

**Why is the URL of the Python service a setting, not in the code?** It changes
between a developer's machine and a real deployment. Configuration belongs in
configuration. The same now applies to the schedule sync's cadence and horizon.

**Why do failed and empty stay different states everywhere?** "The service did
not answer" and "there are no games" look identical on screen if a failure
falls through to the empty state — and the empty state is reassuring, and
false.

**Why are unreachable things disabled with a reason rather than hidden?** The
same instinct as the stale badge: say so, do not disappear.

**Why were results that did not work kept and written down?** Two whole feature
families were built, validated, measured and rejected. The measurements are
more useful than the code, because they explain *why* the accuracy ceiling
holds rather than just confirming that it does.

**Why was tuning stopped deliberately?** Once three independent model families
converged, further tuning was recognised as the wrong instrument and effort
moved to finding new information instead.

---

## 19. The accuracy experiments — six ideas tried, six rejected

Chapter 5 described the ceiling: three completely different kinds of model all
land in the same accuracy band, so the limit is what the features know rather
than how cleverly they are combined. The obvious response is to go looking for
new information.

Six attempts have now been made. **All six were rejected**, and this chapter is
the record of what each one was, what it found, and — the part that took longest
to learn — **why the six failures are not all the same kind of failure.**

Two of them worked in the sense that mattered most: they cost a few hours and
prevented a few weeks.

### 19.1 The short version

| # | Idea | Result | Kind of failure |
|---|---|---|---|
| 1 | Advanced pace and efficiency stats | worse, then flat | model already had it |
| 2 | Calibration of the win probabilities | already good; nothing to fix | not a failure — a measurement |
| 3 | Win probability derived from the margin model | tie | model already had it |
| 4 | Weighting absences by player quality | nothing | information was genuinely new |
| 5 | Shot location and shot quality | tie, then stopped | information was genuinely new |
| 6 | Travel, time zones and schedule density | tie | **both, and it could tell which** |

Numbers 1 and 3 failed because the information was already in the model under
another name. Numbers 4, 5 and 6 failed **despite the information being
genuinely new**, which is the more significant result — it says the ceiling is
not a matter of feature coverage.

### 19.2 Is the model's confidence trustworthy? (Calibration)

Before adding anything, a different question: when the model says a team has a
70% chance of winning, do those teams actually win about 70% of the time?

This had never been measured. Accuracy and log loss were the only scores on
record, and neither answers it. The question came from a review of published
NBA prediction research, where one paper found that selecting bets on
**calibration** rather than accuracy turned a 35% loss into a 35% gain — a
result entirely about whether a probability means what it says.

**The answer: the probabilities are already good.**

| Measure | Value | |
|---|---|---|
| Brier score | **0.2063** | better than the 0.221–0.225 published range for honestly-evaluated NBA models |
| Of the loss, how much is bad calibration | **2.3%** | |
| Of the loss, how much is inability to separate winners from losers | **97.7%** | |

That second split is the useful part. A prediction can be wrong two ways: the
probabilities can be systematically miscalibrated, or the model can simply fail
to tell good teams from bad ones. **Almost all of the loss is the second kind.**

Two standard techniques for fixing calibration after the fact were tried. One
improved the score by 0.2% while making calibration slightly *worse*; the other
was worse on everything. **Neither was adopted**, and the decomposition above
explains why there was never much to win.

There is still a real pattern in what miscalibration remains: the model is
**underconfident at the extremes**. Where it says 17% it should say 8%; where it
says 84% it should say 90%. Its probabilities sit a little too close to 50/50.
That shape is exactly what a calibrator should be able to exploit — and it
could not, because the games at those extremes are too few to learn from.

**One trap worth recording**, because it recurs throughout this chapter: the
first attempt scored the *shipped* model and got 0.7217 accuracy — far better
than its honest 0.6698. The shipped models are deliberately trained on every
game with nothing held back, so the test games are inside their training data.
They were being graded on their own homework. Every experiment below therefore
trains a fresh model on the historical portion only.

### 19.3 Could the margin model predict the winner better? (Derived probability)

Since calibration was ruled out, only **discrimination** was left worth chasing —
telling winners from losers more sharply.

There is one place the project visibly throws information away. The win/loss
model is trained on a single bit: did the home team win. A two-point win and a
thirty-point win are identical to it. The margin model sees the full number on
the same fixture. If that extra detail survives being converted back into a
probability, the converted version should discriminate better.

This is also how bookmakers actually work — the spread is priced first and the
moneyline is derived from it, not the other way round.

**It does not.** The converted probability ranks fixtures very slightly *worse*,
and a resampling test says even that difference is noise.

The cleanest measure here is one that cannot be gamed. The conversion from
margin to probability is a smoothly increasing function, so it cannot reorder
anything — whatever scale is chosen, the *ranking* of fixtures is identical.
That means a measure of pure ranking quality answers the question by itself,
and **no choice of scale could rescue a ranking loss.** There was none to
rescue: 0.7343 against the classifier's 0.7364, with a confidence interval
spanning zero.

**The number that nearly fooled us.** The converted version wins **15 more
games** on raw accuracy. That looks like a finding and is not:

- A paired statistical test on the 207 games where the two disagree gives 96
  against 111 — well inside chance.
- Every version of the conversion scores **identically** on accuracy, whatever
  scale is used, because converting a margin to a probability crosses the 50%
  line exactly when the margin crosses zero. **Accuracy only ever tests the
  sign of the predicted margin**, and is blind to everything else the method
  does.

Reporting those 15 games as a win would have been reporting a coin flip.

### 19.4 Does it matter *which* players are missing? (Player impact)

The availability features count how many players are out and weight each
absence by that player's recent **minutes**. Minutes are a proxy for a coach's
trust, not for contribution — so a superstar and a rotation player missing the
same 32 minutes currently count identically.

**This was the best-founded of the six**, because the model has no
representation of player quality anywhere: team averages describe outcomes, and
the rating system is team-level by construction. The gap was real.

Three replacements were tried: recent scoring production, production per minute,
and a crude "how much better is the team with this player than without" measure.

**None of them did anything.** The best moved margin prediction by 0.26% — and a
resampling test puts that inside noise, so it is not even a small real gain that
missed the bar.

**What that says is more interesting than the null itself.** Minutes are a
coach's revealed judgement of who matters, accumulated over a season. Cruder
than production as a measure of quality, but evidently not worse for the
question actually being asked — which is not "how good is this player" but
"how much does this team lose without him".

The likeliest remaining explanation is the *shape* of the feature rather than
the weight: all three variants **add up** a number across a team's absentees,
and adding up can make one missing superstar and three missing bench players
look the same. Testing that needs a different feature, not a different weight.

### 19.5 Were those shots good shots? (Shot location)

This was the one idea carrying genuinely new information, and it is worth being
precise about why.

A box score records that a team made 42 of 90 shots. It does not record whether
those were open layups or contested long jumpers. So the team's recent scoring
average cannot tell a team that is genuinely good from a team that has been
shooting unsustainably well. Shot location fills that gap directly.

**Two things were done before committing to it**, and both were deliberate
choices to spend a little to avoid spending a lot.

**First, a ten-minute feasibility probe.** The data comes from an endpoint the
NBA is known to restrict, and public reports said older seasons are served more
reliably than recent ones — which would bite precisely on the two seasons used
for testing. Five games were requested, one from each of five seasons spanning
the whole range. All five returned complete data.

The probe's real design point: **an empty response and a genuine "no data" look
identical**, so the shot counts were cross-checked against the number of shots
each team is independently known to have taken. All five matched exactly.

**Second, a five-season subset instead of all eleven.** Two hours of downloading
rather than four, on the reasoning that if the idea has nothing on five seasons
it will have nothing on eleven.

**Result: nothing.** The two-feature version changed margin prediction by
+0.09% with an interval spanning zero. The four-feature version was **reliably
worse** — 0.77% worse with an interval excluding zero, which is not a null but a
genuine regression from adding correlated columns.

**One measurement made this comparison honest, and it is easy to skip.** A model
trained on five seasons is worse than one trained on eleven, regardless of any
new feature. Comparing against the eleven-season number would have shown shot
features "losing" **2.69%** before they did anything at all. Both sides were
therefore trained on the identical five seasons, and that baseline was computed
and written down *before* any variant was scored.

**What it would take to revisit this.** Not more seasons — the measurement says
the signal is not there. The plausible gap is that zone-level conversion rates
are too coarse: every three-pointer from the same area is treated alike, when an
open catch-and-shoot and a contested step-back are not. Separating those needs
defender distance and shot-clock data, which this source does not carry. That is
a different data source, not a longer download of this one.

### 19.6 Does the schedule grind teams down? (Travel and density)

The model knows **how long since the last game**. It does not know:

- how far the team flew to get here
- whether they crossed time zones, and in which direction
- how long they have been away from home
- **how many games they have played this week**, as opposed to when the last one
  was

The last point is the sharpest. "One day of rest" is identical whether it is the
second game of a road trip or the fourth game in six nights.

Eight features were built from arena coordinates and time zones — no new
downloading at all, since the schedule was already on hand.

**Result: three ties.** Density alone, travel alone, and all eight together all
land within half a percent of the baseline with every interval spanning zero.
Unlike the shot-quality experiment, the widest version was *not* reliably worse,
so this is a clean null rather than a dilution result.

**But this experiment did something the previous five could not: it said what
kind of null it was.**

Before scoring anything, each new feature was correlated against the one the
model already had — days since the last game:

| New feature | Correlation with rest | What that means |
|---|---|---|
| Distance travelled | +0.115 | genuinely new |
| Time zones crossed | −0.003 | genuinely new |
| Games into a road trip | −0.084 | genuinely new |
| **Games in the last 7 days** | **−0.652** | substantially overlapping |

**These two halves are not equally informative.**

- Travel, time zones and road-trip length are genuinely new information. Their
  null is **strong evidence**: the model was handed something it did not have
  and could not use it. Distance flown does not move NBA outcomes at a scale
  this model can detect.
- Games-in-seven-days is **weaker evidence**. The hypothesis was that density
  and recency are different quantities. At −0.65 that is partly wrong at the
  premise — density overlaps rest substantially without being a mere restatement
  of it. Its null does not establish that density is useless, only that whatever
  it adds is not usable.

**That diagnostic is the one thing from these six experiments most worth
reusing.** Run it before scoring, and a null becomes classifiable rather than
arguable.

### 19.7 What makes these results trustworthy

Every experiment above follows the same discipline, and each rule exists because
something went wrong without it.

**The baseline is re-earned every time.** Each experiment retrains the standard
model from scratch and checks it reproduces the recorded numbers exactly before
any comparison is reported. If it does not, the script refuses to print a
comparison at all. Without this, a comparison could be against a subtly
different model and look perfectly reasonable.

**Small differences get a confidence interval.** Three of these experiments
produced a headline number under 1%, and in every case a resampling test showed
it spanned zero. Two of them would have been written up as findings without it.

**Guards are tested by breaking things deliberately.** A check that has never
been shown to fail has not been shown to work. So a feature claimed to use only
past games is verified twice: corrupting *future* games must not change it, and
corrupting *past* games **must**. Without that second half, a feature that reads
no data at all passes the first test perfectly — which has happened here twice.

**Vacuous passes are reported as such.** During the shot-quality work two checks
could not run at all on the sample available, because every game in it came from
one season. They printed "NOT RUN — unexercised, not verified" rather than a
green tick, and were exercised properly later.

**Guards catching their own author.** The travel experiment's own check rejected
its first two probes. Both times the *check* was right and the probe was wrong:
"games into a road trip" cannot respond at a home game, and "time zones crossed"
cannot when two consecutive venues share a zone. Neither is a defect; the fix
was to test each feature where its test can mean something.

### 19.8 What six rejections actually establish

Not that the model is finished, and not that nothing will ever help. Something
narrower and more useful:

**The ceiling is not about feature coverage.** Three of the six added genuinely
new information — player quality, shot location, travel — and none of it moved
the result. The limit is not that the model is missing a column.

The one candidate never tested is **market odds**, and it is categorically
different from everything above. Every rejected idea was derived from game data
this project already holds — a different view of the same box scores. A betting
line is not that. It is an aggregate of what thousands of other people think,
including information no box score contains: who is tired, who is carrying a
knock, what the crowd will be like. It is the only remaining input that is not a
rearrangement of what is already here.

That is also why it is the honest test of the whole project. Predicting outcomes
accurately is one thing; predicting them better than the market already does is
the only measure that would mean the model knows something the world does not.

---

## 20. Known limitations and what comes next

These are understood and accepted, not oversights.

1. **Nothing is predictable until the season starts.** The data ends 12 April
   2026, the season opens 20 October 2026, and the model horizon is one day.
   The site explains this rather than showing an empty page.

2. **Predictions only one day ahead.** The models use "days since last game" as
   an input. For a fixture further out, with unplayed games in between, that
   would be measured against the wrong previous game. The Python service
   enforces the limit. This is now the **only** reason most listed fixtures are
   unpredictable, since the fixture list itself is no longer the blocker.

3. **The data is currently stale.** Every prediction made today is flagged
   `stale = true`. Correct behaviour, honestly reported.

4. **Roster availability is unknown, and will stay unknown into November.** A
   player's typical-minutes figure does not exist until they have played eleven
   games. So the season opening unblocks *fixtures*, not availability — the
   honest calendar is mid-November for half of players and late December for
   most. The packaging is finished and tested; the calendar is the remaining
   blocker.

5. **The fixture list is only as wide as the sync horizon** — 120 days by
   default. A setting, easily raised, but a real change worth knowing about.

6. **`played` is never set to `true`** (half fixed). Nothing yet notices that a
   game has finished. What *was* fixed: `/api/games/upcoming` no longer trusts
   the flag on its own — it asks for games dated today or later, so a fixture
   that has simply aged past its date drops out. The remaining half is tracked
   with the pipeline rerun.

7. **The game history is baked into the container image**, so re-running the
   pipeline means rebuilding. Fine while the data is a static artifact; a
   mounted volume is the real answer once the pipeline runs on a schedule.

8. **Three games are permanently missing** from the quarter/half data — 0.02%
   of the corpus, accepted rather than worked around.

9. **The Python service pins only its direct dependencies.** Every library that
   could silently change a prediction is pinned to an exact version. The
   libraries those depend on in turn are not, so a rebuild can still pick up a
   different version of something indirect. Pinning everything would turn the
   file into a lockfile, which is more machinery than this project uses
   anywhere else.

10. **`ddl-auto=update` will not scale.** It only ever adds. It never renames,
    removes or alters. The standard replacement is a migration tool such as
    Flyway, deliberately deferred while the schema is still changing.

11. **A database transaction is held open across the call to Python.** This buys
    all-or-nothing safety, at the cost of occupying a database connection while
    waiting on the network. It matters slightly more now that the player-props
    call is the heaviest of the three.

12. **No authentication, no rate limiting.** Anyone who can reach the port can
    call any endpoint. Acceptable on a local machine, not acceptable if exposed
    publicly.

13. **One backend endpoint has no caller.** `GET /api/games/upcoming` lost its
    last user when the website was restructured. The endpoint and its two
    regression tests are still live and still correct — they are simply
    guarding something nothing calls. Recorded here as a deliberate decision to
    make rather than deleted quietly as a side effect of unrelated work.

14. **Test coverage is better but still thin.** Fourteen backend tests, and
    every one guards something that actually went wrong. All five Python
    response shapes are pinned, which closes the largest gap. What is still
    missing is the layer in between: nothing exercises a prediction endpoint
    end to end — website request, database write, response — so the wiring
    between the pieces is still checked by hand. **The shapes are guarded; the
    journey is not.**

15. **Two field names disagree.** The original endpoint returns the game id as
    `id`; the two newer ones use `gameId`. Known, and deliberately not fixed —
    renaming a field the website already reads would break a working page to
    win tidiness.

16. **Two databases can claim port 5432.** The Windows PostgreSQL service and
    the Docker one. A backend started on the host reaches the Windows one. Not
    a code problem, but it has cost real debugging time.

### The roadmap, in priority order

1. **Serve the availability features.** The 5.8% margin improvement is proven,
   built and packaged. It needs the season, and then a decision about one
   inconsistency between how a player's minutes are looked up at training time
   versus serving time.
2. **Market odds** — both to measure genuine edge against the market and to use
   the market's own line as an input. Historical coverage from most providers
   only goes back to about 2019, so full eleven-season coverage will not exist.
   **Chapter 19 raised this from "next on the list" to "the only candidate
   left":** six ideas have now been tested and rejected, and every one of them
   was a different view of game data the project already holds. A betting line
   is the only remaining input that is not a rearrangement of what is already
   here.
3. **Cloud deployment**, and deployment on merge. The last two unfinished items
   from the original definition of done.
4. **Drift monitoring across a season** — the remaining half of real
   operational maturity. The automated retraining and its promotion gate
   already exist.

---

## 21. Quick reference

### Start everything

```
docker compose up --build
```

### Start the pieces by hand

```
cd inference-service && python -m uvicorn app:app --port 8000

cd backend
$env:DB_PASSWORD = "postgres"
.\mvnw spring-boot:run

cd frontend && npm start
```

### Run the tests

```
cd backend   && DB_PASSWORD=postgres ./mvnw test
cd frontend  && npm test -- --watchAll=false
```

### Endpoints

```
GET  /api/teams                        all 30 teams
GET  /api/health                       data freshness + models loaded
GET  /api/games/schedule?daysAhead=14  cached NBA fixtures
GET  /api/games/upcoming               stored games + latest prediction
POST /api/predictions                  the 7 whole-game numbers
POST /api/predictions/quarter-half     the 6 Q1 / first-half numbers
POST /api/predictions/player-props     5 numbers per player, both teams
```

### Tables

```
team                      30 rows, seeded at startup
game                      cached fixtures + anything predicted
prediction                7 whole-game numbers per request
player                    created on demand
quarter_half_prediction   6 numbers per request
player_prop_prediction    5 numbers per player per request
```

### Useful database checks

```
psql -U postgres -d basketball_predictor
  select count(*) from game;
  select count(*) from player;
  select tablename from pg_tables where schemaname='public';
```

### Free port 8080 (PowerShell)

```
Get-NetTCPConnection -LocalPort 8080 -State Listen |
  ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

### The numbers worth remembering

| | |
|---|---|
| Seasons of data | 11 (2015-16 to 2025-26) |
| Games | 13,199 |
| Team-game rows | 26,398 |
| Player-game rows | 280,943 |
| Features in the main model | 38 |
| Predicted numbers | 18 (7 game, 6 period, 5 player) |
| Model artifacts | 23 |
| Data ends | 12 April 2026 |
| Prediction horizon | 1 day past the newest data |
| Home win rate | 56.4% |

### Key accuracy figures

| Target | Best result | Beaten by |
|---|---|---|
| Moneyline | 0.5979 log loss | availability features |
| Margin | 10.74 average error | availability features (−5.8%) |
| Total points | 15.23 average error | nothing tried |
| First-quarter winner | 0.5796 accuracy | nothing — ships labelled *low confidence* |
| Player points | 4.73 average error | nothing — a player's own average is nearly all of it |

### Files that matter most

| File | Why |
|---|---|
| `data-pipeline/data/processed/games_final.csv` | The game history the live service reads |
| `ml-training/common.py` | The feature list — the contract between training and serving |
| `ml-training/live_features.py` | Rebuilds features for an unplayed game |
| `inference-service/app.py` | The service that answers predictions |
| `backend/.../InferenceClient.java` | The one place Java talks to Python |
| `frontend/src/styles/tokens.css` | Every colour, size and typeface |
| `CLAUDE.md` | The full engineering history, including everything tried and rejected |
