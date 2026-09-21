"""Turn today's NBA injury-report PDF into JSON. That is the whole job."""

import time
from datetime import datetime

import pandas as pd
from fastapi import FastAPI

from fetch_current_injury_report import (
    NoReportAvailable,
    get_current_injury_status,
)

app = FastAPI(title="injury-service")

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
        parsed = datetime.fromisoformat(as_of) if as_of else None
        report = get_current_injury_status(as_of=parsed)
        payload = {
            "report_available": True,
            "reason": None,
            "timestamp": report.timestamp.isoformat(),
            "players": _records(report.players),
            "pending": _records(report.pending),
        }
    except NoReportAvailable as error:
        payload = {
            "report_available": False,
            "reason": str(error),
            "timestamp": None,
            "players": [],
            "pending": [],
        }

    _cache[key] = (time.monotonic(), payload)
    return {**payload, "cached": False}
