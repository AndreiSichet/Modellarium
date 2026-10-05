# Round-two league probes — G League, NBL, Liga ACB (2026-10-04)

Throwaway. Not committed. Two of the three terminated at the terms gate with
**no data request made**, so there is no script for them — only this trace.
Same handling as the historical-odds probe, which also wrote no script.

---

## NBA G League — VIABLE

`probe_gleague.py` in this directory. Terms question settled by the
hypothesis holding: league id `20` answers on `stats.nba.com` through the same
`nba_api` library the NBA (`00`) and WNBA (`10`) pipelines already call, so no
new party holds the data and no new terms apply.

### Coverage starts at 2003-04, not at the league's founding

The NBDL began in 2001-02, so an empty response for that season is a finding
about the API rather than an absence of a league. Narrowed to the season:

| season | games | teams |
|---|---|---|
| 2001-02 | **0** | - |
| 2002-03 | **0** | - |
| 2003-04 | 138 | 6 |
| 2004-05 | 144 | 6 |
| 2005-06 | 192 | 8 |

So **23 seasons, 2003-04 to 2025-26** — twice the NBA's eleven. The first two
seasons of the league's existence are not served. 2003-04 reconciles exactly:
6 teams x 46 games / 2 = 138.

### Two more findings worth carrying

**The NBA's game-type digit meanings DO NOT transfer.** Digit `5` is the
play-in for the NBA; for the G League it is the **Showcase Cup**, and the
arithmetic proves it rather than the name: 31 teams x 14 games / 2 = **217**,
exactly the count returned, and its dates (2025-11-07 .. 2025-12-16) fall
*before* the regular season (2025-12-19 .. 2026-03-28). A play-in runs after.

`season_type_nullable='Regular Season'` and `digit == '2'` agree exactly at
558 games, so the existing filter discipline transfers even though the digit
*meanings* do not.

**2019-20 does not reconcile, and that is the rule being the wrong instrument
rather than a data defect.** 597 games, 28 teams, 41-44 per team; the
shortfall is 13 team-games against a 5-game deficit, so 13/2 != 5. The
reconciliation rule assumes a uniform intended schedule with some games
missing. A season cancelled mid-flight has no uniform intended total - teams
had simply played different numbers when it stopped. The WNBA's 2018 forfeit
reconciled because that season was otherwise uniform. No branch was added;
the finding is reported instead.

---

## Australian NBL — NOT OBTAINABLE

**No data request was made to any NBL or Genius Sports endpoint.** The chain
was traced by reading source code on GitHub, then the terms were read. That
order is the point.

### The provenance chain

```
nblR (R package, JaseZiv/nblR)
  |  downloads .rds files from
  v
JaseZiv/nblr_data  GitHub releases      <- no LICENSE file; API reports
  |                                        "license": None
  |  which were scraped from TWO upstreams:
  |
  +-> https://fibalivestats.dcd.shared.geniussports.com/data/{match_id}/data.json
  |      backfill/get_games_lists.R line 15
  |      -> Genius Sports (FIBA LiveStats)
  |
  +-> https://apicdn.nbl.com.au/nbl/custom/api/genius?route=competitions/{id}/matches
         backfill/league.R lines 10, 24
         -> nbl.com.au, whose own route is literally named "genius"
```

So this is the **second and third situations combined**: a commercial stats
provider powering the league's site, pre-scraped and rehosted with no licence.
`nblR`'s own package licence covers its R code. It cannot grant rights to
Genius Sports' or the NBL's data, which is the principle that decided
EuroLeague and the Kaggle odds dataset.

The FIBA LiveStats shape is visible in the parsing code before any URL is
read - `resp[["tm"]][["1"]][["pl"]]` in `backfill/helpers.R` is that format's
signature.

### robots.txt — permissive, and not the deciding document

`https://www.nbl.com.au/robots.txt` (301 from the apex):

```
User-agent: *
Disallow: /nbl-tipping
Disallow: /nbl-tipping-
Disallow: /angel-
Disallow: /test-
```

`https://www.geniussports.com/robots.txt`: `Allow: /*`.

Neither forbids crawling. **robots.txt is not a licence**, and the terms say
something different.

### The terms, quoted verbatim

`https://www.nbl.com.au/pages/nbl-experience-terms-and-conditions`. Scope is
the site itself, not a sub-product:

> "Access to and use of this website and related services available through
> this website is subject to these terms and conditions which include our
> Privacy Policy"

> "Any reproduction or redistribution of this website or the Content is
> prohibited and may result in civil and criminal penalties. **You must not
> copy the Content to any other server, location or support** for publication,
> reproduction or distribution is expressly prohibited."

> "You may store, print and display the content supplied **solely for your own
> personal use**. You are not permitted to publish, manipulate, distribute or
> otherwise reproduce, in any format, any of the content or copies of the
> content supplied to you or which appears on this website nor may you use any
> such content **in connection with any business or commercial enterprise**."

> "You may not use this website, or any of its Content, to further any
> commercial purpose"

Copying match results and box scores onto this project's own server and
serving them from a deployed app is the named prohibited act, twice over. The
`/legal` page is a privacy policy and says nothing about content use; this is
the governing document.

**Verdict: not obtainable.** Same category as EuroLeague.

---

## Liga ACB — NOT OBTAINABLE

**No data request was made.** Only `robots.txt` and the legal notice were
read.

### robots.txt — permissive as written, and oddly shaped

`https://acb.com/robots.txt` enumerates fifteen AI crawlers - GPTBot,
ClaudeBot, Claude-SearchBot, Claude-User, PerplexityBot, CCBot and others -
then:

```
User-agent: *
Allow: /
```

Consecutive `User-agent` lines with no rules between them form **one group**,
so as published the file allows all of them. Worth recording that the operator
enumerated AI agents at all, since that is usually the preamble to a
`Disallow`; but a file must be read as written, not as guessed at, and as
written it permits crawling.

Again: not the deciding document.

### The legal notice, quoted verbatim

`https://acb.com/es/liga/aviso-legal` (ACEB, S.A.U.):

> "ACEB, S.A.U. autoriza al Usuario para visualizar la información que se
> contiene en este sitio web, así como para efectuar reproducciones privadas
> (simple actividad de descarga y almacenamiento en sus sistemas informáticos),
> **siempre y cuando los elementos sean destinados únicamente al uso personal**,
> así como su utilización **exclusivamente con fines periodísticos**"

> "**Se prohíbe**, fuera de las finalidades establecidas expresamente en el
> párrafo anterior, la utilización del contenido del Sitio Web, y su
> **distribución**, modificación, cesión a terceros, así como la
> **reproducción**, transformación o **comunicación pública**, mediante
> cualquier medio y tecnología, requiriéndose para ello el **consentimiento
> previo y expreso** de ACEB, S.A.U."

In English: viewing and *private* reproduction are permitted **only** for
personal use or journalism. Outside those purposes, use, distribution,
reproduction and **public communication by any medium or technology** are
prohibited without ACEB's prior express consent.

A publicly deployed prediction app is public communication of derived content,
and is neither personal use nor journalism. This is almost word for word the
position EuroLeague failed on.

**Verdict: not obtainable.** No scraping package changes this - a wrapper
cannot grant rights its source does not hold.

---

## Decision

Per the spec's rule: **G League is built first, after the 21st.** The NBL and
the ACB are closed on terms, and no substitute league was probed to make up
the count.
