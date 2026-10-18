# NHL data: source, licence, and why none of it is in this repository

## Source

Every NHL row in this project comes from **English Wikipedia team-season
articles**, read as wikitext through the MediaWiki Action API
(`https://en.wikipedia.org/w/api.php`), plus one **standings template** per
season for the goals-for and goals-against reconciliation. No other source is
used. No article HTML is fetched; `ingestion/wiki_api.assert_api_only` refuses
any URL that is not the API endpoint, so the claim is enforced rather than
remembered.

Rounds 3 and 4 of the probe established that the purpose-built hockey feeds are
closed to this project, and the source gate was not reopened for Phase 1.
Wikipedia is the one source whose terms permit this use.

## Licence

Wikipedia content is **CC BY-SA 4.0**. That grant covers commercial use and
requires **attribution** and **share-alike** on derivative works.

Attribution trail: every raw file's provenance is recorded in the per-season
`manifest.json` beside it — article title, page id, **revision id**, revision
timestamp, fetch time and sha256. A revision id identifies exactly which
version of which article a row came from.

**One NHL-specific attribution note.** Some rows in `nhl_games_final.csv`
carry `DERIVED = 1`. Those are games where the two articles disagreed and the
season's standings template resolved which copy to believe — so the value is
not stated by any single article. `nhl_repairs.csv` records, per game, which
copy was believed and the standings evidence for it. A derived row is still
wholly a derivative of CC BY-SA material; the flag exists so no later phase
mistakes it for something an article asserts.

## Why none of it is committed

**Every NHL table stays out of the repository, raw and processed alike** —
`.gitignore` carries `data-pipeline/data/nhl/` with no un-ignoring exception,
unlike the NBA, WNBA and G League tables, several of which are committed
because they are runtime dependencies.

The reason is the licence, not size. Share-alike on a derived database
published in a public repository is a commitment the owner has deliberately
declined to make here. The data lives on disk only. If a future phase needs an
NHL table at runtime it arrives on the served volume, never through git.

**The data root is `data-pipeline/data/nhl/`, beside the other four
leagues.** That is what lets `served_data` point at either the repo copy or a
served snapshot with one environment variable, because the snapshot layout
mirrors the repo's. The NFL's phase 4 had to move here after the fact and
keeps its old path ignored as well, since a checkout already held data there
(gaps entry 80); this league starts here, so there is only ever one path and
nothing to orphan.

So there is nothing to un-ignore, and nothing here contradicts the standing
rule in `CLAUDE.md` about Dockerfile `COPY` lists: no NHL file is in any image.

## Access discipline

- One request every three seconds for content, and one per second for the
  cheap revision-id checks.
- `maxlag=5`; a 429 or maxlag error is honoured, backed off, **counted, and
  printed in the step's final stats line** rather than absorbed.
- A User-Agent that identifies the client and where to complain, per
  Wikimedia's User-Agent policy:
  `Modellarium/0.1 (https://github.com/AndreiSichet/Modellarium; NHL results
  pipeline) python-requests/<version>`.
- **Completed seasons are frozen by revision id and never refetched.** Only
  the current season and the one before it are, and a season whose raw file is
  absent is fetched regardless — the rule the daily refresh learned the hard
  way (`NOTES_HISTORY.md` §52).
