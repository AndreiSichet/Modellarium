# Modellarium

A personal sportsbook, in active development.

For each NBA game it predicts the winner, the margin, and the totals for points, rebounds and assists — the same again for the first quarter and the first half — plus five stats for each player expected to play. Every number comes from models trained here on eleven seasons of real results. Nothing is scraped from a bookmaker.

**The WNBA was added as a second league in October 2026**, with its own pipeline, its own models, its own three markets and its own pages in the app.

---

## What it predicts

Three families, all reachable in the running app.

| Family | Markets | Notes |
|---|---|---|
| **Full game** | 7 | Home win probability, home margin, total points, rebound margin/total, assist margin/total |
| **First quarter and first half** | 6 | Winner, margin and total for each period. Q1 and 1H only, not all four quarters |
| **Player props** | 5 per player | Points, rebounds, assists, three-pointers made, and the three combined. Up to 10 players per team |
| **WNBA** | 3 | Winner, margin and total points. Twelve seasons, its own models, selected and scored separately |

Two of the markets carry qualifiers that travel all the way to the screen rather than being hidden: the quarter and half winner probabilities are conditional on the period not being tied, and the first-quarter winner is labelled low confidence because it measurably is. A third was considered for the WNBA winner and dropped — the gap it would have described has a confidence interval spanning zero, so it is recorded in the model's manifest rather than shown as if it were a finding.

## Scope, stated plainly

**Predictions are for the next game only.** A game's features are built from each team's form, rest and Elo *going into* it, and a fixture two days out may have an unplayed game in between that changes those inputs. The schedule view lists fixtures months ahead; anything beyond the next matchday is shown with a visible reason rather than hidden.

**The data currently ends 2026-04-12**, the close of the 2025-26 regular season. The app says so on screen and marks itself stale. Fresh predictions resume once the pipeline is rerun with 2026-27 data.

**Roster availability is not yet applied at serving time.** Absence features measurably improve the spread prediction and are trained into the models, but computing them live needs an injury-report fetch the container cannot currently run. Player boards say so instead of implying a clean bill of health.

---

## Tech stack

| Layer | Technology |
|---|---|
| Data pipeline | Python — pandas, numpy, nba_api |
| Model training & tracking | Python — scikit-learn, XGBoost, MLflow |
| Model serving | FastAPI, uvicorn |
| Backend | Java 21 (target) on JDK 25, Spring Boot 4.1, PostgreSQL 17 |
| Frontend | React 19 (create-react-app), plain CSS |
| Containerisation | Docker + Compose, all five services |
| CI | GitHub Actions — four parallel jobs |
| Continuous retraining | GitHub Actions on a self-hosted Windows runner, weekly |
| Not yet built | Cloud deployment, CD |

Direct dependencies are pinned in all three `requirements.txt` files.

## Documentation

**[PROJECT_INSIGHTS.md](PROJECT_INSIGHTS.md)** explains the whole system in
detail and in plain language: how a prediction travels from raw NBA data to
the screen, the data pipeline, the models and what was measured, the
continuous-training pipeline, the inference service, the injury sidecar, the
backend (database, every file, every endpoint), the frontend, how to run and
test everything, the problems found along the way, and **the six accuracy
experiments that were run and rejected** — what each one tried, what it found,
and why the failures are not all the same kind of failure.

Start there. This README is the short version.

## Repository structure

```
basketball-predictor/
├── data-pipeline/          Ingestion and feature engineering (Python)
│   ├── ingestion/          Games, player box scores, quarter scores,
│   │                       team advanced stats, live injury report
│   ├── preprocessing/      Validation, rolling features, rest days, Elo,
│   │                       availability, and the final dataset build
│   ├── wnba/               The second league's own pipeline, parallel
│   │                       to the NBA's rather than sharing it
│   └── data/               raw/ and processed/ (mostly gitignored)
├── ml-training/            Baselines, XGBoost, tuning, finalisation
│   ├── models/             7 production models (committed)
│   ├── models_quarter_half/    6 models + manifest
│   ├── models_player_props/    10 artifacts + routing manifest
│   ├── wnba/                   Selection, and the live feature module
│   ├── models_wnba/            3 models + manifest
│   ├── continuous_retrain.py   Scheduled retrain with a promotion gate
│   └── model_evaluation.py     Shared scoring
├── injury-service/         FastAPI — turns the injury-report PDF into JSON
│                           (the only service carrying a Java runtime)
├── inference-service/      FastAPI — /health, schedules, four /predict routes
├── backend/                Spring Boot REST API + PostgreSQL
│   └── src/                   Spring Boot application
├── frontend/               React — four routes, both leagues, a game detail
│                           page whose tabs follow the league's markets
├── infra/                  Terraform (not started)
├── .github/workflows/      ci.yml, continuous-retrain.yml
├── PROJECT_INSIGHTS.md     How the whole system works, in detail
└── docker-compose.yml
```

## Running it

Everything, in one command:

```bash
docker compose up --build
```

Five services: `postgres`, `injury-service`, `inference-service`, `backend`,
`frontend`. The app is at `http://localhost:3000`.

### The served game history lives on a volume, not in the image

`inference-service` reads every league's history from a read-only bind mount
at `D:\modellarium-data`, located through `DATA_DIR`. It **refuses to boot**
if that is unset or incomplete — deliberately, with no fallback to a copy
inside the image, because two indistinguishable sources of served data is a
hazard this project has already paid for once elsewhere.

Seed it once:

```bash
python ml-training/seed_served_volume.py
```

**The committed tables under `data-pipeline/data/` are no longer what the app
serves.** They remain in the repo for exactly two things: seeding that volume
the first time, and CI's inference boot check. A reader who sees a committed
`games_final.csv` and concludes it is what production reads would be wrong.

`GET /health` reports which snapshot is being served, so "the refresh ran" and
"the service picked it up" are separately observable.

### Opening night, and every day after

**Trigger `daily-refresh` once manually, then leave it.** It runs every
morning at 07:00 UTC, rebuilds each league's history, validates the result,
swaps it in and restarts the inference service — rolling back if the new
snapshot will not boot. Predictions then advance on their own, which they did
not before: `MAX_DAYS_AHEAD` is 1, so a hand-built image yields predictions
for exactly one day.

Three outcomes, all legible: **advanced**, **nothing new** (an off-day or a
league between seasons — green, not a failure), and **failed** (red, with the
previous snapshot still serving).


## Rebuilding the data and models

The pipeline scripts run in dependency order — ingestion, validation, features, Elo, then the final dataset — followed by the training scripts. Both fetchers skip what is already on disk, so a rerun pays only for genuinely new games.

The retrain job automates this weekly and **never deploys**: candidate models land in a directory nothing serves, and promotion opens a pull request for a human to merge.

That job owns **models only**. Served *data* advances daily through
`daily-refresh` and the volume above, which is why the retrain's habit of
rebuilding the tables and discarding them is no longer a gap.

---

## Status

Working end to end, locally and under Docker Compose.

- [x] Data pipeline — ingestion, validation, feature engineering, final dataset (13,199 games, 38 features)
- [x] Model training — baselines, XGBoost, tuning, finalised production models
- [x] Live feature computation for an unplayed matchup
- [x] Inference service — three prediction endpoints, health and schedule
- [x] Backend — Spring Boot, PostgreSQL persistence, cached fixture sync
- [x] Frontend — browse and detail views over all three NBA prediction families
- [x] A second league — WNBA pipeline, models, inference, backend and frontend
- [x] Docker and Compose — all five services
- [x] CI — four parallel jobs, all green
- [x] Continuous retraining with a promotion gate
- [ ] Cloud deployment
- [ ] CD on merge

### Next

Serving roster availability is the highest-value remaining work — it is the only change so far that measurably beat the accuracy ceiling — but it cannot be verified until the 2026-27 season opens and real injury reports exist. Cloud deployment and CD are the last two items on the original milestone list.

The WNBA has no automated data refresh — the NBA's weekly retrain has no WNBA counterpart, so its data only advances when the pipeline is run by hand. That matters from May 2027, and the interface is written so as not to promise otherwise.

Longer term: market odds as a measuring stick, drift monitoring across a season, and a second sport.

---

## A note on accuracy

Three independent model families — a closed-form Elo formula, linear/logistic regression, and tuned XGBoost — converge on the same accuracy band on every full-game target. That is an information ceiling in the feature set, not a modelling problem, and it is treated as one: further tuning is not pursued.

The one thing that has broken through it is player availability, worth roughly 5.8% on spread error. It is trained in and waiting on the serving work above.

Predictions are honest about their own weakness where they have one. A market that scores barely better than a base rate ships labelled as such rather than being quietly dropped or presented as equal to the rest.
