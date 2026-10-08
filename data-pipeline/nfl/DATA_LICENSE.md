# NFL data: source, licence, and why none of it is in this repository

## Source

Every NFL row in this project comes from **English Wikipedia team-season
articles**, read as wikitext through the MediaWiki Action API
(`https://en.wikipedia.org/w/api.php`). No other source is used. No article
HTML is fetched; `ingestion/wiki_api.assert_api_only` refuses any URL that is
not the API endpoint, so the claim is enforced rather than remembered.

Phase 0 established that every purpose-built NFL feed is closed to this
project — nfl.com's Terms of Use prohibit "systematic retrieval of data …
whether to create or compile … a database"; ESPN/Disney's prohibit automated
collection and dataset building; Pro Football Reference returns HTTP 403 on
its own `robots.txt` behind a JavaScript challenge; and the nflverse chain
terminates in an unlicensed repository fed by a personal site that states
nothing at all. Wikipedia is the one source whose terms permit this use.

## Licence

Wikipedia content is **CC BY-SA 4.0**. That grant covers commercial use and
requires **attribution** and **share-alike** on derivative works.

Attribution trail: every raw file's provenance is recorded in the per-season
`manifest.json` beside it — article title, page id, **revision id**, revision
timestamp, fetch time and sha256. A revision id identifies exactly which
version of which article a row came from. Reader-facing attribution appears on
the NFL pages in phase 5.

## Why none of it is committed

**Every NFL table stays out of the repository, raw and processed alike** —
`.gitignore` carries `data-pipeline/nfl/data/` with no un-ignoring exception,
unlike the NBA, WNBA and G League tables, several of which are committed
because they are runtime dependencies.

The reason is the licence, not size. Share-alike on a derived database
published in a public repository is a commitment the owner has deliberately
declined to make here. The data lives on disk, and later on the served volume,
only. If a future phase needs an NFL table at runtime it arrives on the
volume, never through git.

So there is nothing to un-ignore, and nothing here contradicts the standing
rule in `CLAUDE.md` §4 about Dockerfile `COPY` lists: no NFL file is in any
image.

## Access discipline

- One request every three seconds for content, measured against the 429s
  phase 0 saw above roughly that rate.
- `maxlag=5`, and a 429 or maxlag error is honoured, backed off, counted, and
  reported in the run summary rather than absorbed.
- A User-Agent that identifies the client and where to complain, per
  Wikimedia's User-Agent policy:
  `Modellarium/0.1 (https://github.com/AndreiSichet/Modellarium; NFL results
  pipeline) python-requests/<version>`.
- Completed seasons are frozen by revision id and never refetched; only the
  current season and the one before it are.
