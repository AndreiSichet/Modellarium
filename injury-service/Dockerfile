# BASE IMAGE: python:3.13-slim, and the version number is the interesting
# part. Three candidates were tried; two were ruled out by measurement
# rather than by reasoning.
#
#   eclipse-temurin:*-jre-alpine — the smallest JRE base, and disqualified
#   before size enters into it. Alpine uses musl, and pandas publishes
#   manylinux wheels against glibc, so pip would build it from source: a long
#   build, a toolchain baked into the image, and a numeric stack compiled
#   differently from every other container here.
#
#   python:3.14-slim — the obvious choice, matching inference-service, and it
#   FAILS. nbainjuries pins jpype1<1.6.0, and JPype1 1.5.2 ships manylinux
#   wheels for cp311/312/313 but NOT cp314. pip therefore compiles it, and a
#   slim image has no compiler:
#
#       error: [Errno 2] No such file or directory: '/usr/bin/g++'
#       ERROR: Failed building wheel for jpype1
#
#   Checked per-version with `pip download --only-binary=:all:` rather than
#   inferred from that error, so the boundary is known and not guessed.
#
#   python:3.13-slim — a wheel exists, so no compiler is needed and none is
#   installed. This is the one.
#
# WHY A DIFFERENT PYTHON VERSION IS ACCEPTABLE HERE SPECIFICALLY, since
# CLAUDE.md §22 makes a great deal of not letting environments drift apart.
# That concern is about NUMERIC reproducibility: a pandas or xgboost change
# under a trained model alters predictions silently. This service does no
# numeric work. It turns a PDF into strings, imports nothing from the rest of
# the repo, and returns JSON. Nothing it computes can move a prediction.
#
# The alternative that would keep 3.14 is a multi-stage build with g++ in the
# builder. That is strictly more machinery to avoid a version difference that
# costs nothing here - and it would put a source-compiled jpype1 in the image
# where a tested wheel is available, which is worse for reproducibility, not
# better.
FROM python:3.13-slim

WORKDIR /app

# default-jre-headless, NOT default-jre. tabula-py only ever runs a
# command-line extraction; the headless package omits the GUI and font
# libraries a desktop JRE carries. Nothing here opens a window.
#
# --no-install-recommends matters more than usual on this package: the
# recommended set pulls in fonts and X libraries that would roughly undo the
# headless saving.
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Requirements before source, so editing app.py does not reinstall pandas
# and the JVM bridge.
#
# CONTEXT IS injury-service/, not the repo root. Unlike inference-service -
# which imports from ml-training/ and data-pipeline/ and therefore has to be
# built from the root - this service is self-contained, so it follows the
# backend/ and frontend/ convention of a narrow context. That also means the
# root .dockerignore does not apply here; there is a local one.
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

# --host 0.0.0.0 for the same reason the inference service needs it:
# uvicorn's default 127.0.0.1 is the container's own loopback, so the port
# would be published and nothing would answer.
#
# Port 8001, beside inference-service's 8000.
EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
