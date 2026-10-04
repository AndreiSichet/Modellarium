# G League pipeline — phases 1 and 2

A third parallel pipeline, beside `data-pipeline/` (NBA) and
`data-pipeline/wnba/`. Ingestion, validation and features; no models, no
serving.

```
gleague/
├── ingestion/fetch_gleague_games.py            23 seasons, two competitions
└── preprocessing/
    ├── clean_gleague_rows.py                   shared by validator + builder
    ├── validate_gleague_games.py               341 checks
    ├── build_gleague_games_table.py            the two long tables
    ├── verify_gleague_validator.py             negative tests, 13 of 13
    ├── build_gleague_rest_days.py              counts Cup games
    ├── build_gleague_rolling_features.py       FIVE candidates
    ├── build_gleague_elo.py                    K, carryover, expansion offset
    ├── build_gleague_model_dataset.py          the wide output
    └── verify_gleague_features.py              26 of 26
```

Run in that order. Outputs all gitignored — nothing reads them at runtime
until serving exists:

| file | rows | games |
|---|---|---|
| `processed/gleague_games_final.csv` | 19,256 | 9,628 |
| `processed/gleague_showcase_games.csv` | 2,384 | 1,192 |
| `processed/gleague_franchise_identity.csv` | 452 | — |
| `processed/gleague_rest_days.csv` | 19,256 | — |
| `processed/gleague_rolling_features.csv` | 19,256 | — |
| `processed/gleague_elo.csv` | — | 9,628 |
| **`processed/gleague_model_dataset.csv`** | — | **9,628 x 88** |

## Phase 2 at a glance

**Five rolling candidates, and phase 3 chooses.** `ROLL5` (within season),
`CARRY5`/`CARRY10` (carried across seasons), and `CUP5`/`CUP10` — carried
*and* counting the Showcase Cup, which no other league here can offer.
Retention: 86% / 97% / 95% / 97% / 95%.

**The Cup feeds features and never labels.** It reaches rest days (fatigue is
physical) and the `CUP*` windows; no Cup game is ever a target row, asserted
by game-id type digit.

**Elo: K=20, carryover=0.200**, lower than the NBA's 1/3 and the WNBA's 0.5 as
roster churn predicts — though it beats the NBA's values by only 0.03%, so the
direction is established and the magnitude is not. The level is the
informative number: **0.672** against ~0.63 (WNBA) and 0.613 (NBA).

**The expansion offset is fitted and the league mean stays.** 31 new
franchises, revealed strength mean −18.1 with a 95% CI of [−41.3, +3.6], and
an independent grid search picking +0. The opposite of the WNBA's picture.

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

**Six row-level defect classes**, each identified by a structural signature
rather than a score threshold — see `clean_gleague_rows.py`, which the
validator and the builder share so the output cannot be cleaned by a rule the
validation never saw.

**The sixth was found by phase 2, not by validation**, and it is worth
knowing why: a tied game (`2020300058`, 95-95, no `WL` on either side) is
perfectly well-formed — two rows, two teams, one date, no nulls — and passes
the `WL`-agrees-with-`PTS` check vacuously because `WL` is absent. It is wrong
only against a fact about basketball that no row-level check encodes, and
Elo's input gate is what refused it.

**The completeness rule is one-directional and this league breaks it.**
`N teams short by S ⇒ N·S/2 absent games` describes a balanced schedule with
games missing. Three seasons have a team playing *more* than the mode, which
that cannot produce, so `schedule_shape()` returns a three-way verdict and the
third is derived from the distribution rather than named in a list.
