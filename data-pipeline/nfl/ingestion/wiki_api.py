"""MediaWiki Action API client for the NFL pipeline.

The Action API is the only channel this pipeline uses. Article HTML is not
fetched at all, and `get` refuses any URL that is not the API endpoint, so the
rule is enforced rather than remembered.

Wikipedia's robots.txt disallows `/w/` for `User-agent: *` while Wikimedia's
User-Agent policy is written for `api.php` clients and asks them to identify
themselves. This pipeline reads the User-Agent policy as the governing
document for API access and robots.txt as addressing crawlers of index.php
URLs. That is a reading, recorded as one in nfl/README.md.
"""
import time

import requests

API = "https://en.wikipedia.org/w/api.php"

# Identifies the client and where to complain, per the User-Agent policy.
USER_AGENT = (
    "Modellarium/0.1 "
    "(https://github.com/AndreiSichet/Modellarium; NFL results pipeline) "
    f"python-requests/{requests.__version__}"
)

# Phase 0 measured 429s above roughly one content request every three
# seconds, so that is the ceiling rather than a guess.
CONTENT_INTERVAL_SECONDS = 3.0
CHEAP_INTERVAL_SECONDS = 1.0
MAXLAG_SECONDS = 5
MAX_ATTEMPTS = 5
BACKOFF_SECONDS = 5.0


class NotTheApi(RuntimeError):
    """Something tried to fetch a URL that is not the Action API."""


def assert_api_only(url):
    """The one permitted endpoint. Called on every request, so the claim
    'zero requests to article HTML' is enforced rather than remembered."""
    if url != API:
        raise NotTheApi(
            f"this pipeline reads the Action API only, never article HTML; "
            f"refused {url!r}")
    return url


class Throttled(RuntimeError):
    """The API asked us to slow down more times than we will retry."""


class ApiStats:
    """Per-run counters, so a throttled run reports it instead of hiding it."""

    def __init__(self):
        self.requests = 0
        self.content_requests = 0
        self.cheap_requests = 0
        self.http_429 = 0
        self.maxlag_errors = 0
        self.retry_after_waits = 0.0
        self.bytes = 0

    def summary(self):
        return (f"{self.requests} API request(s): "
                f"{self.content_requests} content, {self.cheap_requests} "
                f"revision-id; {self.bytes:,} bytes; "
                f"{self.http_429} HTTP 429, "
                f"{self.maxlag_errors} maxlag error(s), "
                f"{self.retry_after_waits:.0f}s spent waiting")


STATS = ApiStats()
_last_request_at = 0.0


def _wait(interval):
    global _last_request_at
    gap = time.monotonic() - _last_request_at
    if gap < interval:
        time.sleep(interval - gap)
    _last_request_at = time.monotonic()


def get(params, heavy, session=None, stats=None):
    """One API call. `heavy` picks the rate ceiling and the counter.

    Raises rather than returning a partial result: a throttled response that
    a caller treats as "no data" is indistinguishable from a missing page,
    which is exactly how a whole run of seasons once read as absent.
    """
    stats = stats or STATS
    http = session or requests
    query = dict(params)
    query.setdefault("format", "json")
    query.setdefault("formatversion", 2)
    query.setdefault("maxlag", MAXLAG_SECONDS)
    wait = BACKOFF_SECONDS

    for attempt in range(1, MAX_ATTEMPTS + 1):
        _wait(CONTENT_INTERVAL_SECONDS if heavy else CHEAP_INTERVAL_SECONDS)
        response = http.get(assert_api_only(API), params=query, timeout=60,
                            headers={"User-Agent": USER_AGENT})
        stats.requests += 1
        stats.bytes += len(response.content)
        if heavy:
            stats.content_requests += 1
        else:
            stats.cheap_requests += 1

        # The API signals its own overload two different ways.
        retry_after = response.headers.get("Retry-After")
        lagged = False
        if response.status_code == 429:
            stats.http_429 += 1
        elif response.status_code == 200:
            try:
                body = response.json()
            except ValueError:
                body = {}
            if body.get("error", {}).get("code") == "maxlag":
                stats.maxlag_errors += 1
                lagged = True
            else:
                response.raise_for_status()
                return body
        else:
            response.raise_for_status()

        if attempt == MAX_ATTEMPTS:
            break
        pause = float(retry_after) if retry_after and retry_after.isdigit() \
            else wait
        stats.retry_after_waits += pause
        time.sleep(pause)
        wait *= 2

    raise Throttled(
        f"the API throttled {MAX_ATTEMPTS} attempts "
        f"({stats.http_429} x 429, {stats.maxlag_errors} x maxlag). "
        f"Nothing was written.")


def revision_ids(titles, session=None, stats=None):
    """Cheap change-detection: {title: (revid, timestamp)} for up to 50."""
    body = get(dict(action="query", prop="revisions",
                    rvprop="ids|timestamp", titles="|".join(titles)),
               heavy=False, session=session, stats=stats)
    out = {}
    for page in body.get("query", {}).get("pages", []):
        revs = page.get("revisions")
        out[page["title"]] = ((revs[0]["revid"], revs[0]["timestamp"],
                               page.get("pageid")) if revs else None)
    return out


def wikitext(titles, session=None, stats=None):
    """{title: (wikitext, revid, timestamp, pageid)} for up to 50 titles."""
    body = get(dict(action="query", prop="revisions", rvprop="content|ids|"
                    "timestamp", rvslots="main", titles="|".join(titles)),
               heavy=True, session=session, stats=stats)
    out = {}
    for page in body.get("query", {}).get("pages", []):
        revs = page.get("revisions")
        if not revs:
            out[page["title"]] = None
            continue
        rev = revs[0]
        out[page["title"]] = (rev["slots"]["main"]["content"], rev["revid"],
                              rev["timestamp"], page.get("pageid"))
    return out
