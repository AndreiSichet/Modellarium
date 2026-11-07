# Modellarium

A personal sportsbook: five leagues, self-trained models, no bookmaker data.

For every fixture it can reach, Modellarium predicts the markets that league's
models actually earned the right to serve — a winner, a margin, totals, and for
the NBA a first quarter, a first half and five stats per player. Every number
comes from models trained in this repository on real results. Nothing is
scraped from a bookmaker, and no market is shown unless it beat a no-skill
baseline on seasons the model never saw.

Five leagues are served: **NBA**, **WNBA**, **NBA G League**, **NFL** and
**NHL**.

---

## What it predicts

| League | Markets | Model family | History |
|---|---|---|---|
| **NBA** | 7 full-game, 6 quarter/half, 5 per player | XGBoost (team), linear (Q1/1H), hybrid (props) | 13,199 games, 11 seasons |
| **WNBA** | 3 — winner, margin, total points | Linear | 12 seasons, 2015–2026 |
| **G League** | 3 — winner, margin, total points | Linear | 23 seasons, 2003–04 on |
| **NFL** | 3 — winner, margin, total points | XGBoost (margin), linear (total, winner) | 14 seasons, 2012–2025 |
| **NHL** | 2 — winner, puck line | Elo alone (winner), linear (puck line) | 13 seasons, 2013–14 on |

The NBA's three families in full:

| Family | Markets |
|---|---|
| **Full game** | Home win probability, home margin, total points, rebound margin and total, assist margin and total |
| **First quarter and first half** | Winner, margin and total for each period — Q1 and 1H only, not all four quarters |
| **Player props** | Points, rebounds, assists, threes made, and the three combined — up to 10 players a team |

**Markets that did not earn a place are declared, not hidden.** NHL totals are
not served: a constant predictor beat the fitted model on the test seasons, so
the page says so and gives the measurement. The quarter and half winner
probabilities are conditional on the period not being tied, and the
first-quarter winner is labelled low confidence because it measurably is.

---

## Accuracy

Every model figure below is read from that model's own manifest or provenance
file, and every one is paired with the baseline it has to beat — a number on
its own says nothing about whether a model is any good. The NBA's comparators
come from `train_baseline.py`, which reads the same `TEST_SEASONS` literal
production's split reads, so the two are scored on the same games.

### NBA — held out on the 2024 and 2025 seasons

Selection split: train `SEASON < 2023`, validate 2023, test 2024–2025. The
shipped weights are then refitted on every game, so these are the selection
run's held-out scores. Recorded 2026-10-24 in `ml-training/models/provenance.json`.

| Market | Shipped (XGBoost) | Linear baseline | Other baselines |
|---|---|---|---|
| Moneyline | **0.6735** acc / **0.6091** log loss | 0.6777 / 0.6041 | always-home 0.5458 acc; Elo alone 0.6674 / 0.6130 |
| Margin (spread) | **11.43** MAE | 11.25 | — |
| Total points | **15.25** MAE | 15.26 | — |
| Rebound margin / total | **7.56** / **7.36** MAE | 7.50 / 7.32 | — |
| Assist margin / total | **5.48** / **5.85** MAE | 5.36 / 5.69 | — |

**Read that table honestly: the tuned XGBoost models do not beat a linear fit
on most of these markets.** They are level on totals and marginally behind on
the rest, and all of the gaps are small. That is the information ceiling in
the section below, stated with numbers rather than asserted — and it is why
the project stopped tuning. XGBoost ships because it is what the selection run
chose on validation, not because it pulled ahead.

Quarter and half, by selection MAE and accuracy against their own naive
baselines:

| Market | Naive | Model | Confidence |
|---|---|---|---|
| Q1 spread | 7.03 | **6.61** | medium |
| Q1 total | 6.88 | **6.58** | medium |
| 1H spread | 9.59 | **8.91** | medium |
| 1H total | 10.20 | **9.78** | medium |
| Q1 winner | 0.5184 | **0.5796** | **low** — shown, and labelled |
| 1H winner | 0.5291 | **0.6343** | medium |

Player props ship as 10 artifacts over 5 targets — a linear model for players
with a complete rolling history and XGBoost for everyone else, routed by the
manifest. Trained on 225,060 and 280,943 player-games respectively. **The
manifest records no held-out score for them**, so none is quoted here.

### WNBA — held out on 2025 and 2026, 589 games

| Market | Model | Naive | Elo alone |
|---|---|---|---|
| Moneyline | **0.6130** log loss | 0.6878 | **0.6046** |
| Margin | **10.29** MAE | 12.06 | — |
| Total points | **14.48** MAE | 15.06 | — |

Elo alone beats the fitted moneyline model here. That is in the manifest and
is not hidden; the gap has a confidence interval spanning zero, so neither is
presented as the winner.

### G League — held out on 2024-25 and 2025-26, ~1,080 games

| Market | Model | Naive | Elo alone |
|---|---|---|---|
| Moneyline | **0.6671** log loss | 0.6898 | 0.6774 |
| Margin | **11.21** MAE | 12.30 | — |
| Total points | **15.00** MAE | 15.36 | — |

The least predictable of the three basketball leagues, which its Elo level said
in advance. Every candidate configuration tied on validation, so the shipped
one is **a pick rather than a winner**, and the manifest says so.

### NFL — held out on 2024 and 2025, 544 games

| Market | Model | Naive | Elo alone | Verdict |
|---|---|---|---|---|
| Winner | **0.6519** accuracy / **0.6296** log loss / **0.2201** Brier | 0.5359 / 0.6914 | 0.6501 / 0.6265 | **ties Elo alone** |
| Margin | **10.23** MAE / 13.19 RMSE | 11.07 | 10.11 | **ties Elo alone** |
| Total points | **10.24** MAE | 10.51 | — | beats naive |

Both tie verdicts travel on the wire and are rendered in the app. A model that
merely matches a rating formula is reported as matching it.

### NHL — held out on 2024-25 and 2025-26, 2,624 games

| Market | Model | No skill | Gain |
|---|---|---|---|
| Winner (Elo alone) | **0.6816** log loss / **0.2443** Brier / 0.5469 accuracy | 0.6896 | +0.0080 |
| Puck line (home wins by 2+) | **0.6252** log loss / **0.2171** Brier / 0.6688 accuracy | 0.6325 | +0.0074 |

**The winner's gain does not confirm on test.** Its 95% interval against no
skill is `[-0.0158, +0.0001]` and crosses zero; in one of the two test seasons
Elo is worse than no skill. It ships because the decision rule was validation
and validation was applied — and the app says so beside the number, in a
caveat written for a reader. The puck line's interval excludes zero.

The NHL's entire signal is three Elo columns. A calibrator was built for the
winner and **rejected by its own pre-registered rule**: neither Platt nor
isotonic improved calibration on validation, and both made log loss worse.

---

## A note on accuracy

Three independent model families — a closed-form Elo formula, linear and
logistic regression, and tuned XGBoost — converge on the same accuracy band on
every full-game target. That is an information ceiling in the feature set, not
a modelling problem, and it is treated as one: further tuning is not pursued.
Seven accuracy experiments were run and all seven rejected; each is written up.

**Nothing has broken through that ceiling, and one thing that appeared to was
measured and removed.** Until October 2026 the NBA models used four features
describing who was absent from each game. They were worth about 5.8% on margin
error — and they were derived from the game's own box score, so they describe
who *actually played*. A prediction made the day before cannot know that.

Measured four ways, the mismatch cost more than the feature was ever worth:
trained on real participation but served a pre-game proxy, margin error was
**12.05** — worse than never having built the feature at all. The four columns
were dropped, all seven models retrained, and the 11.43 above is what the
shipped models score. The previously published figure was 10.76, and it was not
achievable live. **Replacing a better number with a true one is the point.**

---

## How it works

```
Wikipedia / nba_api  →  per-league pipeline  →  snapshot on a volume
                                                        │
                        models baked into the image      ▼
                                      └──────→  inference-service (FastAPI)
                                                        │
                                   Spring Boot backend ──┘
                                                        │
                                            React frontend
```

**Models are baked into the image; data is not.** Deliberate: data advances
every day, a model only through a reviewed promotion — so a data refresh can
never change which model serves.

**The prediction rule differs per league, and the frontend dispatches on a
field the API declares rather than reimplementing it.** The three basketball
leagues use a *date* rule: one day past the newest result. The NFL and NHL use
a *dependency* rule — a fixture becomes predictable once both teams' previous
games are in history, which makes a whole week's slate available at once. For
the NHL the dependency rule offers 71% of a season the morning before, against
the date rule's 0%.

**Two leagues' data is CC BY-SA 4.0** from English Wikipedia — the NFL's and
the NHL's. No such table is committed, raw or processed; attribution travels on
the wire and is rendered on every page that shows the data. Wikipedia is read
only through the Action API, rate-limited, with the project's own user agent.

---

## Tech stack

| Layer | Technology |
|---|---|
| Data pipeline | Python 3.14 — pandas, numpy, nba_api |
| Training and tracking | Python — scikit-learn, XGBoost, MLflow |
| Model serving | FastAPI, uvicorn |
| Backend | Java 21 target on JDK 25, Spring Boot 4.1, PostgreSQL 17, Flyway |
| Frontend | React 19 (create-react-app), plain CSS with design tokens |
| Containers | Docker and Compose — five services |
| CI | GitHub Actions — four parallel jobs, including a real service boot |
| Scheduled jobs | Daily data refresh and a weekly retrain, on a self-hosted runner |

Direct dependencies are pinned in all three `requirements.txt` files.

---

## Repository structure

```
basketball-predictor/
├── data-pipeline/          Ingestion and feature engineering (Python)
│   ├── ingestion/          Games, player box scores, quarter scores,
│   │                       team advanced stats, live injury report
│   ├── preprocessing/      Validation, rolling features, rest days, Elo,
│   │                       and the final dataset build
│   ├── wnba/ gleague/      One folder per added league, parallel to the
│   ├── nfl/  nhl/          NBA's pipeline rather than sharing it
│   └── data/               raw/ and processed/ (mostly gitignored)
├── ml-training/            Baselines, XGBoost, tuning, finalisation
│   ├── models/                 7 NBA team models + provenance
│   ├── models_quarter_half/    6 models + manifest
│   ├── models_player_props/    10 artifacts + routing manifest
│   ├── models_wnba/            3 models + manifest
│   ├── models_gleague/         3 models + manifest
│   ├── models_nfl/             3 models + manifest
│   ├── models_nhl/             1 artifact + manifest — the winner is a
│   │                           formula, so it has no file
│   ├── daily_refresh.py        Build → validate → swap → verify → roll back
│   ├── continuous_retrain.py   Weekly retrain behind a promotion gate
│   └── verify_*.py             The checks, each with its own plants
├── injury-service/         FastAPI — turns the injury-report PDF into JSON
├── inference-service/      FastAPI — /health, schedules, per-league /predict
├── backend/                Spring Boot REST API + PostgreSQL
├── frontend/               React — three sports, five leagues, per-league
│                           game pages whose tabs follow that league's markets
├── .github/workflows/      ci.yml, daily-refresh.yml, continuous-retrain.yml
├── PROJECT_INSIGHTS.md     How the whole system works, in detail
└── docker-compose.yml
```

---

## Running it

```bash
docker compose up --build
```

Five services: `postgres`, `injury-service`, `inference-service`, `backend`,
`frontend`. The app is at `http://localhost:3000`.

### The served history lives on a volume, not in the image

`inference-service` reads every league's history from a read-only bind mount,
located through `DATA_DIR`. It **refuses to boot** if that is unset or
incomplete — deliberately, with no fallback to a copy inside the image, because
two indistinguishable sources of served data is a hazard this project has
already paid for once.

Seed it once:

```bash
python ml-training/seed_served_volume.py
```

**The committed tables under `data-pipeline/data/` are not what the app
serves.** They remain for two things: seeding that volume the first time, and
CI's inference boot check. `GET /health` reports which snapshot is being
served, so "the refresh ran" and "the service picked it up" stay separately
observable.

### The daily refresh

`daily-refresh` runs each morning, rebuilds every league's history, validates
it, smoke-tests that live features still compute, swaps the snapshot in and
restarts the inference service — **rolling back if the new snapshot will not
boot**. Predictions advance on their own; no pipeline run has to be remembered.

Four outcomes, all legible: **advanced**, **nothing new** (an off-day or a
league between seasons — green, not a failure), **partial** (some leagues
advanced, the rest still serving their previous tables) and **failed** (red,
with the previous snapshot still serving).

**Only open seasons are fetched** — the current one and the one before.
Completed seasons are reused from disk, so historical rows are fixed between
snapshots rather than rebuilt each morning from a source free to answer
differently. That is not hypothetical: the first scheduled run failed because a
2005-06 G League season came back with its rows in a different order.

The cost is that the job no longer notices if a source corrects old data, so
that is available on demand: run it with **`full_refetch`** and it re-fetches
every season and reports how the result differs from what is being served,
**swapping nothing**. Adopting a historical change is a decision, not something
a morning job should take on its own.

---

## Rebuilding data and models

Pipeline scripts run in dependency order — ingestion, validation, features,
Elo, then the final dataset — followed by the training scripts. Both fetchers
skip what is already on disk, so a rerun pays only for genuinely new games.

The weekly retrain job owns **models only** and **never deploys**: candidates
land in a directory nothing serves, and a promotion opens a pull request for a
human to merge. Its gate scores the shipped models as shipped on the most
recent 500 games they have never seen, and refuses any candidate that is more
than 2% worse on any market or that gives up half a market's edge over its
no-skill baseline. A 500-game window is measured, not chosen: at 100 games a
candidate identical to production is refused by sampling noise 26.5% of the
time, at 500 games 0.1%.

---

## Documentation

**[PROJECT_INSIGHTS.md](PROJECT_INSIGHTS.md)** explains the whole system in
plain language and in detail: how a prediction travels from raw data to the
screen, each league's pipeline, the models and what was measured, the
continuous-training pipeline, the inference service, the injury sidecar, the
backend, the frontend, how to run and test everything, the problems found along
the way, and **the accuracy experiments that were run and rejected** — what
each tried, what it found, and why the failures are not all the same kind of
failure.

Start there. This README is the short version.
