"""THROWAWAY PROBE. Can we get per-shot location data across all 11 seasons?

Not wired into any pipeline, not imported by anything, writes nothing. Delete
once the answer is recorded.

One question only: does shotchartdetail serve our full season range, or does it
filter the recent seasons that are our test window? Public reporting says the
NBA restricts this endpoint by IP, rate limit and browser fingerprinting, and
that older seasons are served more reliably than recent ones - so a probe that
sampled only 2016 would pass and tell us nothing.
"""

import sys
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import shotchartdetail

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_player_boxscores import RateLimiter  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GAMES_FINAL_PATH = DATA_DIR / "processed" / "games_final.csv"
PLAYERS_PATH = DATA_DIR / "processed" / "player_boxscores_with_rolling.csv"

# The same ceiling every other fetcher in this pipeline respects. Five requests.
REQUESTS_PER_SECOND = 1.0
rate_limiter = RateLimiter(REQUESTS_PER_SECOND)

# Deliberately spanning the range rather than a convenient corner: the last two
# are the test window, which is exactly where filtering would bite.
TARGET_SEASONS = [2015, 2018, 2021, 2024, 2025]
SEASON_LABEL = {2015: "2015-16", 2018: "2018-19", 2021: "2021-22",
                2024: "2024-25", 2025: "2025-26"}

REQUIRED_FIELDS = ["LOC_X", "LOC_Y", "SHOT_MADE_FLAG", "SHOT_ZONE_BASIC",
                   "SHOT_DISTANCE"]

# FGA, not the 'PTS' default: PTS returns made shots only, so the cross-check
# against known field goal attempts would fail by construction.
CONTEXT_MEASURE = "FGA"

TIMEOUT_SECONDS = 30


def pick_games() -> list:
    """One real mid-season GAME_ID per target season.

    Real ids from games_final.csv rather than invented ones: an invalid id
    returns an empty result set that looks identical to a block, and that
    ambiguity would make the whole probe worthless. Mid-season rather than
    opening night, to avoid any first-game oddity.
    """
    games = pd.read_csv(GAMES_FINAL_PATH,
                        usecols=["GAME_ID", "GAME_DATE", "SEASON"])
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])

    picked = []
    for season in TARGET_SEASONS:
        subset = (games[games["SEASON"] == season]
                  .drop_duplicates("GAME_ID")
                  .sort_values(["GAME_DATE", "GAME_ID"])
                  .reset_index(drop=True))
        if subset.empty:
            raise SystemExit(f"no games found for season {season}")
        row = subset.iloc[len(subset) // 2]
        picked.append({
            "season": season,
            "label": SEASON_LABEL[season],
            "game_id": str(int(row["GAME_ID"])).zfill(10),
            "date": row["GAME_DATE"].date(),
        })
    return picked


def known_fga(game_ids: list) -> dict:
    """Field goal attempts per game, summed from the box scores we already have.

    This is what makes the probe meaningful. An empty result and a successful
    request for a game with no data look identical, so rows-returned on its own
    proves nothing - and a game returning SOME rows but materially fewer than
    its known FGA is worse than an outright block, because it would produce a
    feature that looks fine and is wrong.
    """
    players = pd.read_csv(PLAYERS_PATH, usecols=["GAME_ID", "FGA"],
                          dtype={"GAME_ID": str}, low_memory=False)
    players["FGA"] = pd.to_numeric(players["FGA"], errors="coerce")
    wanted = players[players["GAME_ID"].isin(game_ids)]
    return wanted.groupby("GAME_ID")["FGA"].sum().astype(int).to_dict()


def probe_one(game_id: str) -> dict:
    """One request. Never retries, never works around a refusal."""
    rate_limiter.acquire()
    started = time.perf_counter()
    try:
        endpoint = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=0,
            game_id_nullable=game_id,
            context_measure_simple=CONTEXT_MEASURE,
            season_type_all_star="Regular Season",
            timeout=TIMEOUT_SECONDS,
        )
        frames = endpoint.get_data_frames()
        elapsed = time.perf_counter() - started
    except Exception as error:  # noqa: BLE001 - the probe reports, never retries
        return {
            "status": type(error).__name__,
            "detail": str(error)[:120],
            "seconds": time.perf_counter() - started,
            "rows": None,
            "frame": None,
        }

    shots = frames[0] if frames else pd.DataFrame()
    return {"status": "200", "detail": "", "seconds": elapsed,
            "rows": len(shots), "frame": shots}


def describe_fields(frame) -> tuple:
    """Are the location fields present AND populated, not merely present?"""
    if frame is None or frame.empty:
        return "n/a", []

    missing = [c for c in REQUIRED_FIELDS if c not in frame.columns]
    if missing:
        return f"MISSING {','.join(missing)}", []

    empty = [c for c in REQUIRED_FIELDS if frame[c].isna().all()]
    if empty:
        return f"ALL-NULL {','.join(empty)}", []

    zones = sorted(str(z) for z in frame["SHOT_ZONE_BASIC"].dropna().unique())
    return "yes", zones


def main():
    print("=" * 78)
    print("SHOT CHART FEASIBILITY PROBE - throwaway, writes nothing")
    print("=" * 78)

    games = pick_games()
    fga = known_fga([g["game_id"] for g in games])

    print(f"\nSampling {len(games)} real games spanning "
          f"{games[0]['label']} to {games[-1]['label']}, at "
          f"{REQUESTS_PER_SECOND:.0f} req/s.")
    print(f"context_measure_simple={CONTEXT_MEASURE!r} - the default 'PTS' "
          "returns made shots only,\nwhich could never match a field-goal-"
          "attempt count.\n")

    results = []
    for game in games:
        print(f"  requesting {game['label']}  {game['game_id']}  "
              f"({game['date']}) ...", flush=True)
        outcome = probe_one(game["game_id"])
        fields, zones = describe_fields(outcome["frame"])
        results.append({**game, **outcome, "fields": fields, "zones": zones,
                        "fga": fga.get(game["game_id"])})

    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"{'SEASON':<9}{'GAME_ID':<12}{'STATUS':<18}{'SECS':>6}"
          f"{'ROWS':>7}{'KNOWN FGA':>11}{'LOC FIELDS':>12}")
    print("-" * 78)
    for r in results:
        rows = "-" if r["rows"] is None else f"{r['rows']:,}"
        known = "?" if r["fga"] is None else f"{r['fga']:,}"
        print(f"{r['label']:<9}{r['game_id']:<12}{r['status']:<18}"
              f"{r['seconds']:>6.1f}{rows:>7}{known:>11}{r['fields']:>12}")

    for r in results:
        if r["detail"]:
            print(f"\n  {r['label']} error detail: {r['detail']}")

    print("\n" + "=" * 78)
    print("ROWS AGAINST KNOWN FGA")
    print("=" * 78)
    print("Shot chart rows are field goal attempts, so these should agree")
    print("closely. A game returning rows but materially fewer than its known")
    print("FGA is silently partial data - worse than a clean block.\n")
    for r in results:
        if r["rows"] is None or r["fga"] is None:
            print(f"  {r['label']:<9} no comparison possible")
            continue
        gap = r["rows"] - r["fga"]
        pct = gap / r["fga"] * 100 if r["fga"] else float("nan")
        verdict = ("exact" if gap == 0 else
                   "close" if abs(pct) <= 1.0 else
                   "PARTIAL" if gap < 0 else "MORE THAN EXPECTED")
        print(f"  {r['label']:<9}{r['rows']:>6,} rows against "
              f"{r['fga']:>6,} FGA   {gap:>+5} ({pct:>+6.2f}%)   {verdict}")

    zones = {z for r in results for z in r["zones"]}
    if zones:
        print(f"\nDistinct SHOT_ZONE_BASIC values seen ({len(zones)}):")
        for z in sorted(zones):
            print(f"  - {z}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    served = [r for r in results if r["rows"] is not None and r["rows"] > 0]
    complete = [r for r in served
                if r["fga"] and abs(r["rows"] - r["fga"]) / r["fga"] <= 0.01
                and r["fields"] == "yes"]
    failed = [r for r in results if r not in served]
    partial = [r for r in served if r not in complete]

    if len(complete) == len(results):
        print("ALL FIVE SEASONS RETURN COMPLETE DATA.")
        print("The idea is live. Next step is a full ingestion spec:")
        print("~13,199 requests at 1 req/s, roughly 3.7 hours, resumable by")
        print("file existence like the two fetchers that already work that way.")
    elif complete and failed:
        print("SPLIT - some seasons served, some not.")
        print(f"  served complete: {[r['label'] for r in complete]}")
        print(f"  failed:          {[r['label'] for r in failed]}")
        recent = [r for r in failed if r["season"] >= 2024]
        if recent:
            print("\nTHE IDEA IS DEAD IN THIS FORM. The failures include")
            print(f"{[r['label'] for r in recent]}, which IS the test window -")
            print("a feature we cannot compute for 2024-25 and 2025-26 cannot")
            print("be evaluated against the frozen 10.7368 spread baseline.")
    elif partial and not failed:
        print("SERVED BUT PARTIAL - the dangerous outcome.")
        print(f"  {[r['label'] for r in partial]} returned rows that do not")
        print("  match known FGA. Silently incomplete data would produce a")
        print("  feature that looks fine and is wrong.")
    else:
        print("BLOCKED OUTRIGHT.")
        print("Recording this alongside the hosted-runner finding as durable")
        print("knowledge about what this data source will and will not serve.")
        print("No workaround attempted - no fingerprint spoofing, no headless")
        print("browser, no proxy rotation. A refusal is an answer.")

    print("\nThis probe wrote nothing. Delete it once the answer is recorded.")


if __name__ == "__main__":
    main()
