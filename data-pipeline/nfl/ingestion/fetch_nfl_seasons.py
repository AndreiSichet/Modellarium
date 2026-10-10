"""Fetch NFL team-season wikitext, one file per article, frozen by revision.

History is frozen by revision id: a completed season is fetched once and never
again unless asked, and its manifest is the proof. Only open seasons - the
current one and the one before it, from the franchise table's own September
boundary - are refetched. That is the rule the daily-refresh failure taught,
built in rather than retrofitted.

Change detection runs first and costs one cheap request per season: content is
pulled only for articles whose revision id moved.

Run from the project root with the root .venv active:
    python data-pipeline/nfl/ingestion/fetch_nfl_seasons.py
    python data-pipeline/nfl/ingestion/fetch_nfl_seasons.py --seasons 2022
    python data-pipeline/nfl/ingestion/fetch_nfl_seasons.py --refetch-frozen
"""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NFL = HERE.parent
PIPELINE = NFL.parent          # data-pipeline/, so the NFL's data sits
                               # under data/nfl/ like the other three
                               # leagues' - which is what lets one
                               # DATA_DIR point at either the repo or a
                               # served snapshot with no per-league
                               # special case (served_data.py)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(NFL / "preprocessing"))

import nfl_franchises as F          # noqa: E402
import wiki_api                     # noqa: E402

RAW = PIPELINE / "data" / "nfl" / "raw"
MANIFEST_NAME = "manifest.json"


def season_dir(season):
    return RAW / str(season)


def manifest_path(season):
    return season_dir(season) / MANIFEST_NAME


def load_manifest(season):
    path = manifest_path(season)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def article_file(season, link_target):
    safe = link_target.replace(" ", "_").replace(".", "")
    return season_dir(season) / f"{safe}.wiki"


def write_article(season, link_target, text):
    """Atomic: write to .partial then rename, as the other fetchers do."""
    path = article_file(season, link_target)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    with open(partial, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    partial.replace(path)
    return path


def fetch_season(season, force=False, stats=None):
    """Returns (verdict, per-article detail). Writes the manifest if changed."""
    targets = F.teams_for(season)
    titles = [F.article_title(t, season) for t in targets]
    by_title = dict(zip(titles, targets))
    manifest = load_manifest(season)
    frozen = season not in F.open_seasons() and not force

    on_disk_ok = {}
    for title, target in by_title.items():
        entry = manifest.get(title)
        path = article_file(season, target)
        if entry and path.exists():
            text = path.read_text(encoding="utf-8")
            if sha256(text) == entry.get("sha256"):
                on_disk_ok[title] = entry

    if frozen and len(on_disk_ok) == len(titles):
        return "frozen", {"articles": len(titles), "fetched": 0,
                          "reused": len(titles)}

    # One cheap request tells us which revisions moved.
    live = wiki_api.revision_ids(titles, stats=stats)
    missing_pages = [t for t, v in live.items() if v is None]
    if missing_pages:
        raise RuntimeError(
            f"{season}: the API returned no revision for "
            f"{missing_pages} - a missing article is never skipped")

    need = []
    for title in titles:
        revid, ts, pageid = live[title]
        entry = on_disk_ok.get(title)
        if entry is None or entry.get("revid") != revid:
            need.append(title)

    fetched = 0
    if need:
        got = wiki_api.wikitext(need, stats=stats)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for title in need:
            payload = got.get(title)
            if payload is None:
                raise RuntimeError(f"{season}: no content returned for {title}")
            text, revid, ts, pageid = payload
            write_article(season, by_title[title], text)
            manifest[title] = dict(title=title, pageid=pageid, revid=revid,
                                   revision_timestamp=ts, fetched_at=now,
                                   sha256=sha256(text), bytes=len(text),
                                   link_target=by_title[title])
            fetched += 1
        manifest_path(season).parent.mkdir(parents=True, exist_ok=True)
        tmp = manifest_path(season).with_suffix(".partial")
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=1, sort_keys=True)
        tmp.replace(manifest_path(season))

    verdict = ("fetched" if fetched == len(titles) else
               "updated" if fetched else "unchanged")
    return verdict, {"articles": len(titles), "fetched": fetched,
                     "reused": len(titles) - fetched}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", nargs="*", type=int,
                    help="default: every season from 2012 to the current one")
    ap.add_argument("--refetch-frozen", action="store_true",
                    help="ignore the freeze and re-read completed seasons")
    args = ap.parse_args()

    seasons = args.seasons or F.seasons_through()
    stats = wiki_api.ApiStats()
    print(f"NFL wikitext fetch: seasons {seasons[0]}..{seasons[-1]} "
          f"({len(seasons)} seasons)")
    print(f"  open (refetched)  : {sorted(F.open_seasons())}")
    print(f"  frozen by revision: everything earlier"
          + ("  [OVERRIDDEN by --refetch-frozen]"
             if args.refetch_frozen else ""))
    print(f"  user agent        : {wiki_api.USER_AGENT}")
    print()

    results = {}
    for season in seasons:
        verdict, detail = fetch_season(season, force=args.refetch_frozen,
                                       stats=stats)
        results[season] = (verdict, detail)
        print(f"  {season}  {verdict:<9} "
              f"{detail['fetched']:>2} fetched, {detail['reused']:>2} reused")

    print()
    # THE PREFIX IS READ BY ml-training/daily_refresh.py, which keeps
    # only marked lines from a successful fetch step and discards the
    # rest of its output. Changing the literal here without changing
    # FETCH_STATS_PREFIX there makes this line vanish from the refresh
    # log; the refresh prints a count of reporting fetch steps so that
    # shows up rather than passing unnoticed.
    print(f"  FETCH-STATS: {stats.summary()}")
    if stats.http_429 or stats.maxlag_errors:
        print(f"  NOTE: the API throttled this run "
              f"({stats.http_429} x 429, {stats.maxlag_errors} x maxlag). "
              f"Reported rather than absorbed.")
    else:
        print("  no 429 and no maxlag error")
    print(f"  content requests per season: "
          f"{stats.content_requests / max(len(seasons), 1):.2f}")
    return results


if __name__ == "__main__":
    main()
