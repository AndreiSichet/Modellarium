"""
Turn today's NBA injury-report PDF into JSON. That is the whole job.

WHY THIS IS A SEPARATE SERVICE, and the boundary is drawn narrower than
"availability lives here". Exactly one step in the availability path needs a
Java runtime: parsing the PDF. nbainjuries delegates to tabula-py, which
shells out to Java through jpype. Everything downstream of the parse - name
reconciliation with its accent folding and traded-player handling, the
ROLL10_MIN lookup, the aggregation into ABSENT_COUNT and
WEIGHTED_ABSENT_MIN - is pure pandas over player_boxscores_with_rolling.csv,
which the inference image already carries.

So this service returns the PARSED REPORT, not computed feature values. The
tempting alternative - have the sidecar return the four numbers - would drag
the 80.6 MB player-history CSV into a second image and create a second place
where the reconciliation could drift out of step with what the models were
trained on. The split is: PDF -> JSON here, JSON -> features there.

NOT IN THE JAVA BACKEND, which already has a JRE. nbainjuries is Python, and
reimplementing the parse in Java means rebuilding the thing that was hard
enough to need a purpose-built library in the first place: the PDF separates
columns by position rather than by any character, and wrapped Reason text
extracts out of visual order.

Run:  uvicorn app:app --port 8001
"""

import time
from datetime import datetime

import pandas as pd
from fastapi import FastAPI

from fetch_current_injury_report import (
    NoReportAvailable,
    get_current_injury_status,
)

app = FastAPI(title="injury-service")

# THE CACHE FOR THE FETCH LIVES HERE, and only here.
#
# It used to sit in injury_availability.get_live_availability(), wrapped
# around fetch-plus-reconcile together. With the fetch moved out, the two
# halves have different costs in different processes and each keeps its own
# cache:
#
#   here                     the PDF fetch and parse. Seconds on success;
#                            ~40s on failure, because a missing report means
#                            probing 48 quarter-hour candidates (measured
#                            48.84s before negative caching, 0.01s after).
#
#   inference-service        the reconciliation. An 80.6 MB CSV read and a
#                            1,578-player groupby, measured at 2.1s with no
#                            incidental memoisation (2.11s then 2.09s on a
#                            repeat call).
#
# That is one cache per expensive operation, not a duplicated one. What
# would be duplication - two caches over the same fetch - cannot arise,
# because after this split inference-service performs no fetch at all.
#
# One fetch here serves every caller, which is the other reason it belongs
# on this side: several prediction requests in a minute cost one PDF.
CACHE_TTL_SECONDS = 15 * 60

# Shorter, deliberately. A missing report is a transient state during a
# publishing gap, and the next probe should be cheap but not instant.
FAILURE_CACHE_TTL_SECONDS = 5 * 60

_cache: dict = {}


def _records(frame: pd.DataFrame) -> list:
    """A DataFrame as JSON-safe rows.

    NaN is not valid JSON, and pandas emits it freely here - a pending row
    has no player name, and Reason is often empty. Converted to null rather
    than to "" or 0, because the distinction between "no value" and "an
    empty value" is the one this whole availability path exists to keep.
    """
    if frame is None or frame.empty:
        return []
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


@app.get("/injury-report")
def injury_report(as_of: str = None):
    """The newest published report, parsed.

    `as_of` takes an ISO timestamp and is what makes this testable against a
    known report rather than only against whatever is published right now.

    NO REPORT IS A 200, NOT AN ERROR. Nothing is published between seasons or
    across an overnight gap, and that is an ordinary state of the world
    rather than a failure of this service. It answers
    `report_available: false` with a reason, so a caller has to look at the
    flag rather than infer emptiness from a 500 - the same reasoning that
    made NoReportAvailable an exception instead of an empty frame.
    """
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
