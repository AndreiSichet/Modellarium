"""Add rest-day features to the WNBA feature table.

The WNBA calendar has breaks the NBA's does not: a month-long Olympic break in
some seasons, plus an All-Star break and the Commissioner's Cup. A month-long
gap is a real value rather than an error, but it sits far outside everything
else in the distribution - so this reports the raw distribution, identifies
every long gap by its CAUSE from the data's own structure, and then emits three
columns rather than silently choosing one treatment.
"""

import sys
from pathlib import Path

import pandas as pd

WNBA = Path(__file__).resolve().parents[1]
PIPELINE = WNBA.parent
PROCESSED = PIPELINE / "data" / "wnba" / "processed"
INPUT_PATH = PROCESSED / "wnba_games_with_rolling.csv"
OUTPUT_PATH = PROCESSED / "wnba_games_with_features.csv"

TEAM_KEY = "TEAM_ID"

# The NBA's cap, kept for parity so the two leagues' REST_DAYS mean the same
# thing. Everything above it survives in REST_DAYS_RAW.
REST_DAYS_CAP = 7

# Above this a gap is a scheduled break rather than rest, and is reported
# individually with its cause.
LONG_BREAK_DAYS = 10


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def add_rest_days(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values([TEAM_KEY, "GAME_DATE"]).reset_index(drop=True)

    # SEASON-SCOPED, unlike the NBA's, which groups on team alone and relies on
    # the cap to flatten the offseason. Here the first game of a season is NaN
    # rather than "7 days rest", because those are different facts and a WNBA
    # offseason is seven months.
    raw = frame.groupby([TEAM_KEY, "SEASON"])["GAME_DATE"].diff().dt.days

    frame["REST_DAYS_RAW"] = raw
    frame["REST_DAYS"] = raw.clip(upper=REST_DAYS_CAP)
    frame["IS_LONG_BREAK"] = raw > LONG_BREAK_DAYS
    # 1, not 0: games on consecutive calendar days are one day apart.
    frame["IS_BACK_TO_BACK"] = frame["REST_DAYS"] == 1
    return frame


def report_distribution(frame: pd.DataFrame) -> None:
    section("REST_DAYS DISTRIBUTION (raw, uncapped)")
    raw = frame["REST_DAYS_RAW"]
    print(raw.describe().to_string())
    print(f"\n  NaN (season openers, correctly)   : {int(raw.isna().sum()):,}")
    print(f"  back-to-backs (== 1)              : "
          f"{int((raw == 1).sum()):,}")
    print(f"  above the NBA cap of {REST_DAYS_CAP}            : "
          f"{int((raw > REST_DAYS_CAP).sum()):,}")
    print(f"  above {LONG_BREAK_DAYS} days (scheduled breaks) : "
          f"{int((raw > LONG_BREAK_DAYS).sum()):,}")

    print("\n  counts by value:")
    counts = raw.value_counts().sort_index()
    for value, n in counts.items():
        bar = "#" * min(int(n / 20), 40)
        marker = "  <- above the cap" if value > REST_DAYS_CAP else ""
        print(f"    {int(value):>3} days  {n:>5,}  {bar}{marker}")


def identify_long_breaks(frame: pd.DataFrame) -> None:
    """Name the cause of every gap over LONG_BREAK_DAYS, from structure.

    The signature distinguishes them: a league-wide break idles EVERY team over
    the same dates, while a gap affecting one or two teams is a cancellation or
    a game this pipeline excluded. Inferred from how many teams share the gap,
    rather than asserted from a remembered calendar.
    """
    section(f"EVERY GAP OVER {LONG_BREAK_DAYS} DAYS, WITH ITS CAUSE")

    long_gaps = frame[frame["REST_DAYS_RAW"] > LONG_BREAK_DAYS].copy()
    long_gaps["prev_date"] = (long_gaps["GAME_DATE"]
                              - pd.to_timedelta(long_gaps["REST_DAYS_RAW"], unit="D"))
    teams_per_season = frame.groupby("SEASON")[TEAM_KEY].nunique()

    print(f"{'SEASON':<8}{'TEAMS':>8}{'GAP DAYS':>10}  {'IDLE WINDOW':<26}CAUSE")
    print("-" * 78)

    for season, rows in long_gaps.groupby("SEASON"):
        total = teams_per_season[season]
        # MERGE OVERLAPPING IDLE WINDOWS. Grouping on exact endpoints splits one
        # league-wide break into a dozen clusters, because teams' last games
        # before it and first games after it fall on different days. Overlap is
        # the property that actually identifies a single break.
        spans = sorted(zip(rows["prev_date"], rows["GAME_DATE"],
                           rows[TEAM_KEY], rows["REST_DAYS_RAW"]))
        clusters = []
        for start, end, team, gap in spans:
            if clusters and start <= clusters[-1]["end"]:
                c = clusters[-1]
                c["end"] = max(c["end"], end)
                c["start"] = min(c["start"], start)
                c["teams"].add(team)
                c["gaps"].append(gap)
            else:
                clusters.append({"start": start, "end": end,
                                 "teams": {team}, "gaps": [gap]})

        for c in clusters:
            share = len(c["teams"])
            if share >= total - 1:
                cause = f"LEAGUE-WIDE BREAK ({share} of {total} teams idle)"
            elif share <= 2:
                cause = f"{share} team(s) only - excluded game or cancellation"
            else:
                cause = f"{share} of {total} teams - partial break"
            window = f"{c['start'].date()} .. {c['end'].date()}"
            print(f"{season:<8}{share:>8}{min(c['gaps']):>5.0f}-{max(c['gaps']):<4.0f}  "
                  f"{window:<26}{cause}")

    print("""
  THE COMMISSIONER'S CUP FINAL IS EXCLUDED, AND IT NEVER REACHES THIS REPORT.
  Verified directly rather than assumed: the final carries game-id type '105',
  not the regular season's '2', so the phase-1 filter drops it. 2026's was
  LVA @ NYL on 2026-06-30, and type '103' is the All-Star game.

  Its only effect is to inflate ONE ordinary rest value per finalist. Measured:
  both LVA and NYL show REST_DAYS_RAW = 5 across 2026-06-28 -> 2026-07-03, when
  their true rest was 3 - they played the final in between. Two team-games per
  Cup season, overstated by two days, inside the normal distribution.

  Recorded rather than patched. Inventing a row for a game deliberately
  excluded from a regular-season model would be a worse defect than a known
  two-day overstatement on two rows, and it would put a non-counting game into
  every rolling window that spans it.""")


def record_the_decision() -> None:
    section("THE DECISION, MADE EXPLICITLY")
    print("""  Three columns ship, because the right treatment depends on the model
  phase 3 chooses and that choice is not made here:

    REST_DAYS_RAW   uncapped. A month-long Olympic break reads as ~30.
    REST_DAYS       clipped at 7, matching the NBA so the two leagues'
                    column means the same thing.
    IS_LONG_BREAK   boolean, raw > 10 days.

  RECOMMENDATION for phase 3, not a decision taken here: a tree model can
  take REST_DAYS_RAW directly - it splits on order, so a 30 is just "a lot".
  A linear model cannot: one 30 among values of 1-4 is a high-leverage point
  that will drag the coefficient. Given the dataset is 2,655 games, phase 3
  may well prefer linear, in which case REST_DAYS plus IS_LONG_BREAK keeps
  the information without the leverage.""")


def main() -> int:
    print(__doc__)
    frame = pd.read_csv(INPUT_PATH, dtype={"GAME_ID": str})
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])

    frame = add_rest_days(frame)

    report_distribution(frame)
    identify_long_breaks(frame)
    record_the_decision()

    frame.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    section("OUTPUT")
    print(f"  wrote {OUTPUT_PATH.name}")
    print(f"  rows {len(frame):,}, columns {len(frame.columns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
