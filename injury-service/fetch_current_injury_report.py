"""Fetch the NBA's official injury report as of right now, parsed into rows."""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

EASTERN = ZoneInfo("America/New_York")

STEP_MINUTES = 15
MAX_STEPS = 48
DELAY_BETWEEN_PROBES_SECONDS = 0.5

ABSENT_STATUSES = frozenset({"Out", "Doubtful"})
PRESENT_STATUSES = frozenset({"Questionable", "Probable", "Available"})

PENDING_MARKER = "NOT YET SUBMITTED"

STATUS_COLUMN = "Current Status"
PLAYER_COLUMN = "Player Name"
REASON_COLUMN = "Reason"

class NoReportAvailable(Exception):
    """No injury report exists in the searched window."""

class SourceUnavailable(NoReportAvailable):
    """The NBA's server could not be reached at all.

    A SUBCLASS DELIBERATELY. Every caller that catches NoReportAvailable keeps
    degrading exactly as before - availability resolves to NaN either way, and
    this must not turn an upstream outage into a 500 on /predict. What the
    subclass buys is that the two can be TOLD APART by anything that cares,
    which before this nothing could: check_reportvalid() returns False for a
    404, a 403 and a dead network alike, so the walk-back loop fell through to
    NoReportAvailable and reported "Expected between seasons - the NBA
    publishes these only around game days" for what was a connectivity
    failure. Measured 2026-10-07: a blackholed host produced a response
    byte-identically shaped to the real offseason one.
    """

@dataclass(frozen=True)
class InjuryReport:
    """One parsed report."""

    timestamp: datetime
    players: pd.DataFrame
    pending: pd.DataFrame

    @property
    def absent(self) -> pd.DataFrame:
        return self.players[self.players["IS_ABSENT"]]

def normalize_player_name(name: str) -> str:
    """Turn 'Porter Jr., Michael' into 'Michael Porter Jr.'."""
    if not isinstance(name, str) or "," not in name:
        return name.strip() if isinstance(name, str) else name
    surname, forename = name.split(",", 1)
    return f"{forename.strip()} {surname.strip()}".strip()

SOURCE_PROBE_TIMEOUT_SECONDS = 20

def probe_source(timestamp: datetime) -> tuple:
    """(reachable, detail) for one candidate URL - did the HOST answer at all.

    The library cannot answer this. check_reportvalid() catches
    URLRetrievalError AND bare Exception and returns False for both, so "no
    report at this minute" and "the network is down" arrive identically. One
    level down, validate_injrepurl() DOES raise a typed URLRetrievalError
    carrying the original requests exception on .reason, which is the
    distinction this reads.

    The URL still comes from the library (injury.gen_url), so §15's reason for
    using it at all - it owns the URL format and stays correct when that format
    changes - is preserved. Only the classification is ours.

    Measured 2026-10-07: a missing report gives HTTPError 403 (the status §15
    recorded), a blackholed host gives ConnectionError with no response.
    """
    try:
        from nbainjuries import _constants, _parser, injury
        from nbainjuries._exceptions import URLRetrievalError
    except Exception as error:  # noqa: BLE001
        # A library reshuffle must not break fetching - it only costs the
        # classification, so say so rather than guessing which state it is.
        return True, f"cannot classify ({type(error).__name__})"

    try:
        _parser.validate_injrepurl(
            injury.gen_url(timestamp),
            headers=_constants.requestheaders,
            timeout=SOURCE_PROBE_TIMEOUT_SECONDS,
        )
        return True, "report present"
    except URLRetrievalError as error:
        cause = getattr(error, "reason", None)
        status = getattr(getattr(cause, "response", None), "status_code", None)
        if status is not None:
            return True, f"host answered {status}"
        return False, f"{type(cause).__name__}: {cause}"
    except Exception as error:  # noqa: BLE001
        return True, f"cannot classify ({type(error).__name__})"

def find_latest_report(as_of: datetime = None, max_steps: int = MAX_STEPS) -> datetime:
    """Newest report at or before `as_of`, searching backward."""
    from nbainjuries import injury

    if as_of is None:
        as_of = datetime.now(EASTERN)

    anchor = as_of.replace(tzinfo=None) if as_of.tzinfo else as_of
    anchor = anchor.replace(
        minute=(anchor.minute // STEP_MINUTES) * STEP_MINUTES, second=0, microsecond=0
    )

    # Classify ONCE, before the walk-back. If the host cannot be reached, no
    # earlier timestamp will fix that, so 48 probes would be 25s of futile
    # waiting ending in the wrong explanation.
    reachable, detail = probe_source(anchor)
    if not reachable:
        raise SourceUnavailable(
            f"The NBA's report server could not be reached ({detail}). This is "
            f"NOT the offseason case - the host did not answer at all, so "
            f"whether a report exists for "
            f"{anchor:%Y-%m-%d %I:%M %p} ET is unknown rather than no."
        )

    for step in range(max_steps):
        candidate = anchor - timedelta(minutes=STEP_MINUTES * step)
        # check_reportvalid swallows everything and returns False, so the bare
        # except that used to sit here was unreachable. probe_source above is
        # what tells a dead network from a missing report.
        if injury.check_reportvalid(candidate):
            return candidate
        time.sleep(DELAY_BETWEEN_PROBES_SECONDS)

    raise NoReportAvailable(
        f"No injury report in the {max_steps * STEP_MINUTES / 60:.0f} hours before "
        f"{anchor:%Y-%m-%d %I:%M %p} ET. Expected between seasons - the NBA "
        f"publishes these only around game days."
    )

def apply_status_policy(players: pd.DataFrame) -> pd.DataFrame:
    """Collapse the five statuses into IS_ABSENT."""
    statuses = players[STATUS_COLUMN].astype(str).str.strip()

    unknown = sorted(set(statuses) - ABSENT_STATUSES - PRESENT_STATUSES)
    if unknown:
        raise ValueError(
            f"Unrecognised status value(s): {unknown}. The five documented "
            f"statuses are {sorted(ABSENT_STATUSES | PRESENT_STATUSES)}. Decide "
            f"explicitly how a new one maps rather than letting it default."
        )

    players = players.copy()
    players["IS_ABSENT"] = statuses.isin(ABSENT_STATUSES)
    players["PLAYER_NAME_NORMALIZED"] = players[PLAYER_COLUMN].map(normalize_player_name)
    return players

def get_current_injury_status(as_of: datetime = None) -> InjuryReport:
    """The newest available report, split into real rows and pending teams."""
    from nbainjuries import injury

    timestamp = find_latest_report(as_of=as_of)
    frame = injury.get_reportdata(timestamp, return_df=True)

    is_pending = (
        frame[PLAYER_COLUMN].isna()
        | (frame[PLAYER_COLUMN].astype(str).str.strip() == "")
        | frame[REASON_COLUMN].astype(str).str.upper().str.contains(PENDING_MARKER)
    )

    pending = frame[is_pending].copy()
    players = apply_status_policy(frame[~is_pending].copy())

    return InjuryReport(timestamp=timestamp, players=players, pending=pending)

def main():
    print("Looking for the most recent injury report...\n")

    try:
        report = get_current_injury_status()
    except NoReportAvailable as error:
        print("NO REPORT CURRENTLY AVAILABLE")
        print(f"  {error}")
        print("\nThis is correct behaviour, not a failure: nothing is published")
        print("between seasons. Nothing stale was returned.")
        return 0

    print(f"Report: {report.timestamp:%Y-%m-%d %I:%M %p} ET")
    print(f"  player rows : {len(report.players):,}")
    print(f"  pending rows: {len(report.pending):,}")

    print("\n  status breakdown:")
    counts = report.players[STATUS_COLUMN].value_counts()
    for status, count in counts.items():
        bucket = "absent" if status in ABSENT_STATUSES else "present"
        print(f"    {status:<14} {count:>4}   -> {bucket}")
    print(f"    {'IS_ABSENT True':<14} {int(report.players['IS_ABSENT'].sum()):>4}")

    if not report.pending.empty:
        print(f"\n  NOT YET SUBMITTED - unknown, NOT zero absences:")
        for matchup, group in report.pending.groupby("Matchup"):
            teams = ", ".join(sorted(group["Team"].astype(str)))
            print(f"    {matchup}: {teams}")

    if not report.players.empty:
        first_matchup = report.players["Matchup"].iloc[0]
        sample = report.players[report.players["Matchup"] == first_matchup]
        print(f"\n  parsed table for {first_matchup}:")
        print(sample[["Team", PLAYER_COLUMN, "PLAYER_NAME_NORMALIZED",
                      STATUS_COLUMN, "IS_ABSENT"]].to_string(index=False))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
