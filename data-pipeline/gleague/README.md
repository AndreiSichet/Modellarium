# G League pipeline — phase 1 (ingestion and validation)

A third parallel pipeline, beside `data-pipeline/` (NBA) and
`data-pipeline/wnba/`. Ingestion and validation only; no features, no models,
no serving.

```
gleague/
├── ingestion/fetch_gleague_games.py        23 seasons, two competitions
└── preprocessing/
    ├── clean_gleague_rows.py               shared by the two below
    ├── validate_gleague_games.py           311 checks
    ├── build_gleague_games_table.py        the two output tables
    └── verify_gleague_validator.py         negative tests, 12 of 12
```

Outputs, both gitignored — nothing reads them at runtime until serving exists:

| file | rows | games |
|---|---|---|
| `data/gleague/processed/gleague_games_final.csv` | 19,258 | 9,629 |
| `data/gleague/processed/gleague_showcase_games.csv` | 2,384 | 1,192 |
| `data/gleague/processed/gleague_franchise_identity.csv` | 452 | — |

---

## THE SHARED-CODE EXTRACTION IS DEFERRED — 2026-10-04

**This is the third copy of the same pipeline shape, and three copies is
normally where this project extracts shared code.** `InferenceClient`,
`GameLookup`, `model_evaluation.py` and `trailing_mean()` were all extracted at
the third call site, for the reason recorded in §8 and §24 of `CLAUDE.md`:
three copies of "find the game or make one" is how one fixture ends up with
three rows.

It is not extracted here, deliberately, and the reason is different from the
reason the WNBA pipeline gave for not generalising the NBA's:

1. **The NBA pipeline runs by hand on 21 October.** Refactoring the thing that
   must work on a fixed date, three weeks before the date, trades a certainty
   for a convenience.

2. **Two of the three differences only became visible once the third existed,
   and they are not the differences the second one predicted.** The WNBA's
   README expected season label, season boundary, league prefix, team count
   and games per team. Those did vary. But the G League added two that no
   amount of looking at two leagues would have shown:

   - **a second competition.** The Showcase Cup needs its own table, and the
     NBA and WNBA shapes have nowhere to put one.
   - **a different defect profile.** The G League needs five row-level
     cleaning classes the other two do not have at all, and its
     `PLUS_MINUS` is wrong six times more often.

   A shared `fetch_league_games(league_id, ...)` extracted from the first two
   would have had to be reopened for both.

**What to extract, when it is extracted:** the season-derivation trio
(`current_season_start_year` / `season_label` / `seasons_through`), which is
genuinely identical across all three apart from one integer, and the
atomic-write and resume-by-existence pattern. **What not to extract:** the
cleaning, which is per-league by nature.

**Do not unify the season boundaries.** There are now four and they answer
four different questions:

| constant | value | maps |
|---|---|---|
| `build_rolling_features.derive_season` | month ≥ 8 | a game date → its NBA season |
| `fetch_games.SEASON_START_MONTH` | 10 | today → which NBA seasons exist |
| `fetch_wnba_games.SEASON_START_MONTH` | 5 | today → which WNBA seasons exist |
| `fetch_gleague_games.SEASON_START_MONTH` | 11 | today → which G League seasons exist |

---

## What phase 1 found

**Coverage is 23 seasons, 2003-04 to 2025-26** — twice the NBA's eleven. The
NBDL began in 2001-02 and those first two seasons return zero rows, which is a
property of the API rather than an absence of a league, so `2003` is where the
source stops rather than a scope decision.

**`PLUS_MINUS` is wrong on 6.79% of games where it is populated** (602 of
8,865), against 1.02% for the NBA and 1.24% for the WNBA — same endpoint, same
defect, an order of magnitude more of it. On the Showcase Cup it is worse
still: 347 of 1,192, **14.6%**. It is also entirely absent before 2005-06. The
builder recomputes the margin from `PTS` and refuses to write if the derived
sign disagrees with `WL`.

**Five row-level defect classes**, each identified by a structural signature
rather than a score threshold — see `clean_gleague_rows.py`, which the
validator and the builder share so the output cannot be cleaned by a rule the
validation never saw.

**The completeness rule is one-directional and this league breaks it.**
`N teams short by S ⇒ N·S/2 absent games` describes a balanced schedule with
games missing. Three seasons have a team playing *more* than the mode, which
that cannot produce, so `schedule_shape()` returns a three-way verdict and the
third is derived from the distribution rather than named in a list.
