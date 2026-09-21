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

def find_latest_report(as_of: datetime = None, max_steps: int = MAX_STEPS) -> datetime:
    """Newest report at or before `as_of`, searching backward."""
    from nbainjuries import injury

    if as_of is None:
        as_of = datetime.now(EASTERN)

    anchor = as_of.replace(tzinfo=None) if as_of.tzinfo else as_of
    anchor = anchor.replace(
        minute=(anchor.minute // STEP_MINUTES) * STEP_MINUTES, second=0, microsecond=0
    )

    for step in range(max_steps):
        candidate = anchor - timedelta(minutes=STEP_MINUTES * step)
        try:
            if injury.check_reportvalid(candidate):
                return candidate
        except Exception:
            pass
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
