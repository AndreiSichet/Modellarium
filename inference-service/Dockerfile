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
COPY ml-training/served_data.py ./ml-training/
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

# NO HISTORY TABLE IS COPIED INTO THIS IMAGE ANY MORE.
#
# All six - the NBA's three, the WNBA's one and the G League's two - arrive
# through a read-only bind mount and are located via DATA_DIR. The service
# REFUSES TO BOOT if that is unset or incomplete, with no fallback to a copy
# inside the image: a fallback would let it serve either of two snapshots
# with nothing saying which, which is the two-Postgres-on-5432 hazard
# rebuilt somewhere new.
#
# This also took 77 MB of player box scores out of the image - previously
# the largest thing it carried - and ended the "a pipeline rerun means a
# rebuild" note in CLAUDE.md section 11.
#
# MODELS STAY BAKED IN, DELIBERATELY. Data advances daily; a model advances
# only through a promotion, which already ends in a rebuild. Keeping them
# apart means a data refresh can never change which model is serving, and a
# model change stays a reviewed event.

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

# --- G League, the fifth model family -------------------------------------
#
# THE SHOWCASE CUP TABLE IS THE ONE THAT LOOKS OPTIONAL AND IS NOT. REST_DAYS
# is computed across the Cup boundary, so a team's first regular-season game
# of a Cup season reads rest from its last Cup game. Serving without this
# table would give that row a value the models never saw for it - one game
# per team per season, silently, and invisible to any check that does not
# look specifically at a post-Cup opener.
#
# Both tables sat inside data-pipeline/data/gleague/processed/* and were
# un-ignored in the SAME change that added these lines. 2.8 MB and 356 KB, so
# no LFS: the 50 MB threshold in CLAUDE.md section 4 is not close.
#
# The features table and the model dataset are deliberately NOT copied.
# Serving replays rolling, rest and Elo from the raw tables under
# manifest.json's parameters, so the image carries no precomputed feature
# column that could drift - and here it certainly would, since phase 2's
# diagnostic Elo fitted carryover 0.200 while the artifacts were fitted
# under 0.333.
COPY ml-training/gleague/live_gleague_features.py ./ml-training/gleague/
COPY data-pipeline/gleague/preprocessing/build_gleague_elo.py \
     data-pipeline/gleague/preprocessing/clean_gleague_rows.py \
     ./data-pipeline/gleague/preprocessing/
# fetch_gleague_games is here for its season helpers alone - the G League's
# NOVEMBER boundary, which is neither the NBA's October nor the WNBA's May.
# Restating it in app.py would be a fourth copy of a rule four scripts
# already disagree about on purpose.
COPY data-pipeline/gleague/ingestion/fetch_gleague_games.py \
     ./data-pipeline/gleague/ingestion/
COPY ml-training/models_gleague/ ./ml-training/models_gleague/

# --- NFL, the sixth model family and the first sport that is not basketball -
#
# NO NFL TABLE IS COPIED, AND UNLIKE THE OTHER THREE THAT IS A LICENCE
# DECISION AS WELL AS A VOLUME ONE. The NFL history is derived from English
# Wikipedia and is CC BY-SA 4.0; this repository is not licensed to
# redistribute a derived database, so the three tables are never committed and
# arrive only on the volume. CI, which has no volume, boots against a
# SYNTHETIC set that make_synthetic_nfl_tables.py generates from invented
# scores - and live_nfl_features.refuse_synthetic makes a production boot
# refuse that set rather than serve noise as results.
#
# THE FEATURE FUNCTION IS IMPORTED, NOT COPIED, which is why the pipeline's
# own builder is in the image. build_nfl_model_dataset.py carries
# features_for, NflHistory and EloLedger, and serving calls them exactly as
# the training walk did; a second implementation here is the thing that would
# drift. nfl_divisions and nfl_franchises come with it because it imports
# them, and parse_nfl_wikitext comes for kickoff_utc alone - the zone map and
# the Arizona no-daylight-saving exception live there, and /schedule/nfl
# converts with that function rather than a second copy of the same rule.
#
# Nothing in this image fetches: wiki_api.py is deliberately absent, so the
# builder's import of it would fail if it had one - it does not.
#
# Git-tracking checked in this same change, CLAUDE.md section 4's standing
# rule: all five .py files are committed, and the three tables are ignored on
# purpose and must stay ignored.
COPY ml-training/nfl/live_nfl_features.py ./ml-training/nfl/
COPY data-pipeline/nfl/preprocessing/build_nfl_model_dataset.py      data-pipeline/nfl/preprocessing/nfl_divisions.py      data-pipeline/nfl/preprocessing/nfl_franchises.py      data-pipeline/nfl/preprocessing/parse_nfl_wikitext.py      data-pipeline/nfl/preprocessing/make_synthetic_nfl_tables.py      ./data-pipeline/nfl/preprocessing/
COPY ml-training/models_nfl/ ./ml-training/models_nfl/

WORKDIR /app/inference-service

EXPOSE 8000

# --host 0.0.0.0 is load-bearing. uvicorn defaults to 127.0.0.1, which
# inside a container is the container's own loopback: the port would be
# published and nothing would ever answer on it. It works locally today
# only because dev calls are made from the same machine.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
