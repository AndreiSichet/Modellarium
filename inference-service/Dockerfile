# Build from the REPOSITORY ROOT, not from this directory:
#
#   docker build -f inference-service/Dockerfile -t basketball-inference .
#
# The context has to be the root because this service deliberately imports
# from two sibling folders rather than duplicating feature logic:
# ml-training/{common,live_features}.py and three scripts in
# data-pipeline/preprocessing/. live_features.py locates them with
# Path(__file__).resolve().parents[1], so the sibling layout has to survive
# intact inside the image - /app is what parents[1] resolves to from both
# /app/inference-service/app.py and /app/ml-training/live_features.py.

FROM python:3.14-slim

WORKDIR /app

# Dependencies before code, so editing app.py does not reinstall xgboost.
COPY inference-service/requirements.txt ./inference-service/requirements.txt
RUN pip install --no-cache-dir -r inference-service/requirements.txt

# The sibling layout, reproduced exactly. Only the files this service
# actually reaches for - not the whole pipeline or the training scripts.
COPY inference-service/ ./inference-service/
COPY ml-training/common.py ml-training/live_features.py \
     ml-training/injury_availability.py ./ml-training/
COPY ml-training/models/ ./ml-training/models/
COPY data-pipeline/preprocessing/build_elo_ratings.py \
     data-pipeline/preprocessing/build_rest_days.py \
     data-pipeline/preprocessing/build_rolling_features.py \
     ./data-pipeline/preprocessing/

# Quarter/half and player props. train_baseline.py and
# train_quarter_half_baseline.py are here not because anything trains in
# this image, but because the feature modules import the column lists and
# the season split from the scripts that DEFINED them rather than keeping a
# second copy - the same reason live_features.py reaches into the pipeline.
COPY ml-training/train_baseline.py \
     ml-training/train_quarter_half_baseline.py \
     ml-training/live_quarter_half_features.py \
     ml-training/live_player_features.py \
     ./ml-training/
COPY ml-training/models_quarter_half/ ./ml-training/models_quarter_half/
COPY ml-training/models_player_props/ ./ml-training/models_player_props/
COPY data-pipeline/preprocessing/build_quarter_half_rolling.py \
     data-pipeline/preprocessing/build_player_dataset.py \
     data-pipeline/preprocessing/build_player_rolling_minutes.py \
     ./data-pipeline/preprocessing/
# build_player_rolling_minutes imports the canonical box-score schema and
# has_played() from the fetcher; nothing in the image ever calls the API.
COPY data-pipeline/ingestion/fetch_player_boxscores.py ./data-pipeline/ingestion/

# The history tables the service serves from. Baked into the image, so a
# pipeline rerun means a rebuild - see the note in CLAUDE.md section 11.
#
# SIZE WARNING: player_boxscores_with_rolling.csv is 77 MB, more than ten
# times games_final.csv, and it is loaded into memory at startup. This is
# the single largest thing this image carries and the reason it is no
# longer small. A mounted volume or an object-store fetch is the real
# answer once the pipeline runs on a schedule.
COPY data-pipeline/data/processed/games_final.csv \
     data-pipeline/data/processed/quarter_half_raw.csv \
     data-pipeline/data/processed/player_boxscores_with_rolling.csv \
     ./data-pipeline/data/processed/

# WNBA. A fourth model family, and the first league that is not the NBA, so
# it gets its own directories throughout rather than extra columns in the
# NBA's - the same parallel-not-shared arrangement the pipeline uses.
#
# wnba_games_final.csv is 746 KB against the NBA's 6 MB, and it was
# gitignored as disposable pipeline output until this line existed. That is
# the fourth time a genuine runtime dependency has been found inside a broad
# ignore rule, so it was checked in the same change that added this COPY
# rather than by a red CI run - CLAUDE.md section 4's standing rule.
#
# NAME COLLISION, AND IT IS THE WRONG WAY ROUND. Despite the name, this is
# NOT the WNBA counterpart of the NBA's games_final.csv: it is the RAW long
# table, the counterpart of games_master.csv. The file carrying rolling,
# rest and Elo is wnba_games_final_features.csv, and it is deliberately NOT
# here. Serving replays all three from this raw table under the parameters
# in models_wnba/manifest.json, so there is no precomputed feature column
# in the image that could drift from what the models were fitted on.
#
# The directory copy is deliberate: the registry check reads manifest.json
# and globs the directory, so adding a fourth WNBA target needs no change
# here. It also carries the test-once receipt and the selection tables,
# which nothing in the image reads but which are the artifacts' provenance.
COPY ml-training/wnba/live_wnba_features.py ./ml-training/wnba/
COPY data-pipeline/wnba/preprocessing/build_wnba_rolling_features.py \
     data-pipeline/wnba/preprocessing/build_wnba_rest_days.py \
     data-pipeline/wnba/preprocessing/build_wnba_elo.py \
     ./data-pipeline/wnba/preprocessing/
# fetch_wnba_games is here for current_season() alone - the WNBA's May season
# boundary is defined there and restating it would be a second copy. It is
# imported lazily inside season_of() so nba_api is not pulled in at startup,
# and as with fetch_player_boxscores above, nothing in the image calls the API.
COPY data-pipeline/wnba/ingestion/fetch_wnba_games.py ./data-pipeline/wnba/ingestion/
COPY ml-training/models_wnba/ ./ml-training/models_wnba/
COPY data-pipeline/data/wnba/processed/wnba_games_final.csv \
     ./data-pipeline/data/wnba/processed/

WORKDIR /app/inference-service

EXPOSE 8000

# --host 0.0.0.0 is load-bearing. uvicorn defaults to 127.0.0.1, which
# inside a container is the container's own loopback: the port would be
# published and nothing would ever answer on it. It works locally today
# only because dev calls are made from the same machine.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
