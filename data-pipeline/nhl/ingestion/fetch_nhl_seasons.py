"""Fetch NHL team-season wikitext and standings templates, one season at a
time, through the Action API only.

Completed seasons are frozen by revision id and never refetched; the open
seasons are the current one and the one before it. A season whose raw
directory is absent is fetched regardless of that rule - which is the half
§52 had to learn twice, once in each direction: the WNBA fetcher skipped on
presence alone so a live season could never update, and the G League's
required a file that could not exist so 18 seasons re-fetched every morning.

Writes `data-pipeline/data/nhl/raw/<season>/`, gitignored: one file per
article, one per standings template, and a manifest recording title, page id,
revision id, revision timestamp, fetch time and sha256 - which is the
attribution trail CC BY-SA asks for.
"""
import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "preprocessing"))

import wiki_api as api                                       # noqa: E402
import nhl_franchises as F                                   # noqa: E402

# One root for every league's data, which is what lets `served_data` point at
# either the repo copy or a served snapshot with a single environment
# variable. The NFL's phase 4 had to move here after the fact (gaps entry
# 80); this league starts here, so no checkout ever holds the other path.
RAW = HERE.parent.parent / "data" / "nhl" / "raw"

# The whole-league standings template exists only from this season; before it
# the same data is one template per division, named in the season article.
FIRST_WHOLE_LEAGUE_STANDINGS = 2020

STANDINGS_IN_ARTICLE = re.compile(
    r"\{\{\s*(\d{4}–\d{2} NHL [^}|\n]*?standings)\s*[|}]", re.I)


def season_dir(start_year):
    """A filesystem-safe directory name. The EN DASH stays in TITLES only."""
    return "%d-%02d" % (start_year, (start_year + 1) % 100)


def slug(title):
    return re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")[:120]


def open_seasons(today=None):
    """The current season and the one before it.

    The one before is open too because late corrections happen, and because
    the boundary has to be crossed safely: in early October the current
    season has no games yet, so the previous one is still the live table.
    """
    seasons = F.seasons_through(today)
    return set(seasons[-2:])


def division_standings_titles(start_year, session, stats):
    """The per-division template names, READ FROM the season article.

    Never guessed: the division names change (the 2012-13 lockout season
    splits by conference), and the article is the only place that says which
    templates carry that season's table. The WILD CARD templates are
    deliberately excluded - measured, they repeat teams already in the
    division templates, 10 of them in 2016-17, so including both would count
    those teams twice.
    """
    article = F.season_article_title(start_year)
    got = api.wikitext([article], session=session, stats=stats)
    text = (got.get(article) or (None,))[0]
    if not text:
        raise RuntimeError(
            "%r is absent, so this season's standings templates cannot be "
            "named. They are read from the article rather than guessed."
            % article)
    names = []
    for found in STANDINGS_IN_ARTICLE.findall(text):
        name = found.strip()
        if re.search(r"wild\s*card", name, re.I):
            continue
        if name not in names:
            names.append(name)
    if not names:
        raise RuntimeError(
            "%r names no division standings template. The pattern or the "
            "article changed; this is fatal rather than skipped." % article)
    return ["Template:" + n for n in names]


def standings_titles(start_year, session, stats):
    if start_year >= FIRST_WHOLE_LEAGUE_STANDINGS:
        return ["Template:%s NHL standings" % F.season_label(start_year)]
    return division_standings_titles(start_year, session, stats)


def read_manifest(directory):
    path = directory / "manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("pages", {})


def write_page(directory, title, text, revid, stamp, pageid, entries):
    """One article or template, plus its provenance."""
    if text.lstrip().upper().startswith("#REDIRECT"):
        # The hyphen-vs-en-dash trap, made fatal. A redirect page's content
        # parses to zero games and raises nothing, which is exactly the kind
        # of silent emptiness this project has been bitten by.
        raise RuntimeError(
            "%r is a REDIRECT page, not an article. Its content is "
            "%r. Check season_label's EN DASH."
            % (title, text.strip()[:60]))
    name = slug(title) + ".wikitext"
    (directory / name).write_text(text, encoding="utf-8", newline="\n")
    entries[title] = dict(file=name, pageid=pageid, revid=revid,
                          revision_timestamp=stamp,
                          fetched_at=dt.datetime.now(dt.timezone.utc)
                          .isoformat(),
                          sha256=hashlib.sha256(text.encode("utf-8"))
                          .hexdigest(),
                          bytes=len(text.encode("utf-8")))


def fetch_season(start_year, session, stats, force=False):
    """Returns (fetched, skipped, reason)."""
    directory = RAW / season_dir(start_year)
    titles = F.titles_for(start_year)
    previous = read_manifest(directory)
    is_open = start_year in open_seasons()
    absent = not directory.is_dir() or not previous

    if not (force or is_open or absent):
        return 0, len(titles), "completed and frozen"

    directory.mkdir(parents=True, exist_ok=True)
    entries = dict(previous)
    article_titles = sorted(titles)

    # Cheap change detection BEFORE any content request.
    current = api.revision_ids(article_titles, session=session, stats=stats)
    missing = [t for t, v in current.items() if v is None]
    if missing:
        raise RuntimeError(
            "these titles do not exist, so the authored franchise table and "
            "the corpus disagree: %s" % missing)

    stale = [t for t in article_titles
             if force or t not in previous
             or previous[t].get("revid") != current[t][0]
             or not (directory / previous[t]["file"]).is_file()]

    fetched = 0
    for i in range(0, len(stale), 50):
        batch = stale[i:i + 50]
        got = api.wikitext(batch, session=session, stats=stats)
        for title in batch:
            value = got.get(title)
            if value is None:
                raise RuntimeError("no content returned for %r" % title)
            text, revid, stamp, pageid = value
            write_page(directory, title, text, revid, stamp, pageid, entries)
            fetched += 1

    # The standings template(s). Always refetched for an open season, because
    # the page carries a date rather than `complete` and can lag the logs.
    st_titles = standings_titles(start_year, session, stats)
    need = [t for t in st_titles
            if force or is_open or t not in entries
            or not (directory / entries[t]["file"]).is_file()]
    if need:
        got = api.wikitext(need, session=session, stats=stats)
        for title in need:
            value = got.get(title)
            if value is None:
                raise RuntimeError(
                    "the standings template %r is absent. It is named by the "
                    "season article, so this is a real change rather than a "
                    "guess that missed." % title)
            text, revid, stamp, pageid = value
            write_page(directory, title, text, revid, stamp, pageid, entries)
            fetched += 1

    (directory / "manifest.json").write_text(json.dumps(dict(
        season=F.season_label(start_year),
        season_start_year=start_year,
        is_open_season=is_open,
        standings_titles=st_titles,
        source="English Wikipedia via the MediaWiki Action API",
        licence="CC BY-SA 4.0",
        pages=entries), indent=1), encoding="utf-8", newline="\n")
    return fetched, len(article_titles) - len(stale), \
        "open season" if is_open else ("absent" if absent else "forced")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", default="",
                        help="comma-separated start years; default all")
    parser.add_argument("--force", action="store_true",
                        help="refetch even completed seasons")
    args = parser.parse_args(argv)

    seasons = ([int(s) for s in args.seasons.split(",") if s.strip()]
               or F.seasons_through())
    print("NHL raw fetch: %d season(s), %s .. %s"
          % (len(seasons), F.season_label(seasons[0]),
             F.season_label(seasons[-1])))
    print("  open season(s): %s"
          % ", ".join(sorted(F.season_label(y) for y in open_seasons())))
    print("  writing to %s" % RAW)

    import requests
    session = requests.Session()
    totals = [0, 0]
    for year in seasons:
        fetched, skipped, why = fetch_season(year, session, api.STATS,
                                            force=args.force)
        totals[0] += fetched
        totals[1] += skipped
        print("  %-8s %3d fetched, %3d unchanged   (%s)"
              % (F.season_label(year), fetched, skipped, why))

    print()
    print("  %d page(s) fetched, %d unchanged" % tuple(totals))
    # The stats line is printed whether or not anything was throttled, so a
    # 429 cannot vanish by being reported only on failure.
    print("  " + api.STATS.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
