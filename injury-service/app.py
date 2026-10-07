"""Turn today's NBA injury-report PDF into JSON. That is the whole job."""

import os
import time
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI

from fetch_current_injury_report import (
    NoReportAvailable,
    SourceUnavailable,
    get_current_injury_status,
)

# THE WIRE CONTRACT FOR AVAILABILITY STATE. These three strings are read by
# inference-service, which adds two of its own (`unreachable`, when it cannot
# reach THIS service, and `not_applicable`, for the two leagues whose models
# have no availability features). Kept as literals in both places because the
# two images share no code; changing one means changing the other.
STATE_AVAILABLE = "available"
STATE_NONE_PUBLISHED = "none_published"
STATE_SOURCE_FAILED = "source_failed"
STATE_NEVER_FETCHED = "never_fetched"

app = FastAPI(title="injury-service")

# A DIAGNOSTIC ANCHOR FOR THE LIVE LOOKUP, normally unset. With it set, "now"
# for the un-parameterised /injury-report becomes this timestamp, so a specific
# published report can be replayed through the whole path - which is the only
# way to exercise the `used` state out of season, when the NBA publishes
# nothing. Deliberately an env var rather than a query parameter: a caller
# cannot reach it by accident, and a production deployment that never sets it
# behaves exactly as before.
LIVE_AS_OF_OVERRIDE = os.environ.get("INJURY_LIVE_AS_OF") or None

CACHE_TTL_SECONDS = 15 * 60

FAILURE_CACHE_TTL_SECONDS = 5 * 60

_cache: dict = {}

def _records(frame: pd.DataFrame) -> list:
    """A DataFrame as JSON-safe rows."""
    if frame is None or frame.empty:
        return []
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")

@app.get("/injury-report")
def injury_report(as_of: str = None):
    """The newest published report, parsed."""
    key = as_of or "live"

    if key in _cache:
        cached_at, payload = _cache[key]
        ttl = (FAILURE_CACHE_TTL_SECONDS if not payload["report_available"]
               else CACHE_TTL_SECONDS)
        if time.monotonic() - cached_at < ttl:
            return {**payload, "cached": True}

    try:
        effective = as_of or LIVE_AS_OF_OVERRIDE
        parsed = datetime.fromisoformat(effective) if effective else None
        report = get_current_injury_status(as_of=parsed)
        payload = {
            "report_available": True,
            "report_state": STATE_AVAILABLE,
            "reason": None,
            "timestamp": report.timestamp.isoformat(),
            "players": _records(report.players),
            "pending": _records(report.pending),
        }
    # BEFORE NoReportAvailable, which it subclasses - the other order would
    # make the distinction unreachable, which is the bug being fixed.
    except SourceUnavailable as error:
        payload = {
            "report_available": False,
            "report_state": STATE_SOURCE_FAILED,
            "reason": str(error),
            "timestamp": None,
            "players": [],
            "pending": [],
        }
    except NoReportAvailable as error:
        payload = {
            "report_available": False,
            "report_state": STATE_NONE_PUBLISHED,
            "reason": str(error),
            "timestamp": None,
            "players": [],
            "pending": [],
        }

    payload["observed_at"] = datetime.now(timezone.utc).isoformat()

    _cache[key] = (time.monotonic(), payload)
    return {**payload, "cached": False}


@app.get("/state")
def state():
    """The last observed source state, from cache only - never fetches.

    EXISTS SO /health CAN ASK WITHOUT PAYING FOR A FETCH. Resolving a report
    cold costs seconds to tens of seconds; a health endpoint that did that
    would be the kind of slow nobody forgives. So this reports what is already
    known and says `never_fetched` when nothing is, which is honest rather than
    a guess.
    """
    live = _cache.get("live")
    if live is None:
        return {"state": STATE_NEVER_FETCHED, "observed_at": None,
                "report_timestamp": None, "reason": None,
                "age_seconds": None}

    cached_at, payload = live
    return {
        "state": payload.get("report_state"),
        "observed_at": payload.get("observed_at"),
        "report_timestamp": payload.get("timestamp"),
        "reason": payload.get("reason"),
        "age_seconds": round(time.monotonic() - cached_at, 1),
        "live_as_of_override": LIVE_AS_OF_OVERRIDE,
    }
