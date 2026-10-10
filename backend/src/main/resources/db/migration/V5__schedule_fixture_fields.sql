-- Four facts a cached fixture needs to carry, all nullable, all NFL-only
-- today and basketball-null forever.
--
-- WHY THESE LIVE ON `game` RATHER THAN BEING FETCHED PER REQUEST. The obvious
-- alternative is for ScheduleService to enrich NFL rows from a live
-- /schedule/nfl call. That would undo the whole point of caching fixtures:
-- §21 moved the schedule into this table so browsing survives the inference
-- service being down, and a per-request upstream call puts that dependency
-- straight back. So the sync stores them and the read path is unchanged at
-- two queries.
--
-- `predictable` IS NULLABLE AND THAT IS A THIRD STATE, NOT A MISSING BOOLEAN.
-- Basketball decides predictability from a date - one day after that league's
-- cutoff - which the client computes. The NFL's rule is a dependency (both
-- teams' previous games in history), measured in the inference service and not
-- derivable from any date, so the answer travels with the fixture. A
-- basketball fixture is neither predictable nor unpredictable by this column:
-- it has no server-side answer, and `false` would be a lie the client would
-- then have to second-guess. Hence Boolean, not boolean - the same convention
-- §38 found across all 45 existing columns, where every NOT NULL column is a
-- Java primitive and every nullable one an object type.
--
-- `week` is unquoted deliberately: it is a non-reserved keyword in Postgres,
-- so Hibernate's generated `week` matches and ddl-auto=validate passes. It is
-- the only one of the four that could have collided, which is why it is worth
-- a line rather than a surprise at boot.
--
-- No backfill. Every existing row is basketball and must stay null; the NFL's
-- rows acquire values on the next schedule sync, which runs at startup.

ALTER TABLE game ADD COLUMN predictable BOOLEAN;
ALTER TABLE game ADD COLUMN kickoff_utc TIMESTAMP WITH TIME ZONE;
ALTER TABLE game ADD COLUMN flex BOOLEAN;
ALTER TABLE game ADD COLUMN week INTEGER;
