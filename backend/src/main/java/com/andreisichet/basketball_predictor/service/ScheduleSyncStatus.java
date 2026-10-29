package com.andreisichet.basketball_predictor.service;

import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import org.springframework.stereotype.Component;

import com.andreisichet.basketball_predictor.service.ScheduleSyncService.SyncOutcome;
import com.andreisichet.basketball_predictor.service.ScheduleSyncService.SyncResult;

/**
 * What the schedule sync last did, so /health can say it rather than the log.
 *
 * THIS EXISTS BECAUSE THE FAILURE WAS INVISIBLE. The startup sync failed at
 * every cold start - measured 2026-10-29, all four leagues failing 59-82ms
 * after the listener fired, because uvicorn binds its socket only after the
 * models are loaded and so nothing was listening yet - and the only trace was
 * four warn lines in the backend's own container log. Every freshness field
 * /health already carried described the DATA, and all of them read identically
 * whether the fixtures were cached this morning or never cached at all.
 *
 * WRITTEN FROM A BACKGROUND THREAD, READ FROM A REQUEST THREAD, so the whole
 * state is one immutable record behind an AtomicReference rather than a set of
 * mutable fields. A reader therefore sees a consistent snapshot: the four
 * per-league results and the summary that describes them can never be observed
 * half-updated, which is exactly the shape that makes a status page lie.
 */
@Component
public class ScheduleSyncStatus {

    /** One league's last outcome, with the time it was recorded. */
    public record LeagueStatus(String result, String detail, Instant at) {
    }

    /**
     * The whole state, as /health renders it.
     *
     * {@code lastResult} is one of four words:
     * <ul>
     *   <li>{@code never_run} - nothing has been recorded. On a healthy boot
     *       this is visible only for the moment before the first attempt.</li>
     *   <li>{@code retrying} - at least one league is NOT_READY and the
     *       startup loop has attempts left.</li>
     *   <li>{@code ok} - every league settled successfully.</li>
     *   <li>{@code gave_up} - at least one league is unresolved and nothing
     *       will try again. THIS COVERS TWO CASES ON PURPOSE: the retry cap
     *       was reached, or a league failed for a reason retrying cannot fix.
     *       They are one word because they mean the same thing to whoever
     *       reads the field - the fixtures are not cached and no further
     *       attempt is coming - and the {@code leagues} map below still
     *       distinguishes NOT_READY from FAILED, so nothing is lost.</li>
     * </ul>
     */
    public record Snapshot(
            String lastResult,
            Instant lastAttemptAt,
            Instant lastSuccessAt,
            int startupAttempts,
            long startupWaitedSeconds,
            Map<String, LeagueStatus> leagues) {
    }

    private static final Snapshot NEVER_RUN = new Snapshot(
            "never_run", null, null, 0, 0L, Map.of());

    private final AtomicReference<Snapshot> state =
            new AtomicReference<>(NEVER_RUN);

    /** The state as it stands, for the health endpoint. */
    public Snapshot snapshot() {
        return state.get();
    }

    /** One league finished an attempt. */
    public void record(SyncOutcome outcome) {
        Instant now = Instant.now();
        state.updateAndGet(current -> {
            Map<String, LeagueStatus> leagues =
                    new LinkedHashMap<>(current.leagues());
            leagues.put(outcome.league(), new LeagueStatus(
                    outcome.result().name(), outcome.detail(), now));
            boolean succeeded = outcome.result() == SyncResult.SYNCED
                    || outcome.result() == SyncResult.NOTHING_TO_CACHE;
            return new Snapshot(
                    current.lastResult(),
                    now,
                    succeeded ? now : current.lastSuccessAt(),
                    current.startupAttempts(),
                    current.startupWaitedSeconds(),
                    leagues);
        });
    }

    /** The startup loop is still waiting on at least one league. */
    public void startupRetrying(int attempts, long waitedSeconds) {
        summarise("retrying", attempts, waitedSeconds);
    }

    /** The startup loop is over, however it ended. */
    public void startupFinished(int attempts, long waitedSeconds) {
        summarise(verdict(), attempts, waitedSeconds);
    }

    /**
     * A six-hourly pass is over.
     *
     * KEEPS startupAttempts AND startupWaitedSeconds AS THEY WERE, because
     * those describe the boot and a scheduled pass is not one. Writing 1 and 0
     * here would erase the only record that the cold start took five attempts
     * - which is the single fact this whole block exists to carry.
     *
     * THIS IS THE ONE WAY THE SIX-HOURLY PATH IS NOT LITERALLY UNCHANGED. Its
     * logging is identical: the warn and info lines are emitted from exactly
     * where they were, with the same text and at the same level. It does now
     * update this record, because the alternative is a /health block that
     * describes a boot hours ago and never moves again - stale but true, which
     * is the reading this project keeps catching.
     */
    public void cycleFinished() {
        Snapshot current = state.get();
        summarise(verdict(), current.startupAttempts(),
                current.startupWaitedSeconds());
    }

    /** ok unless some league is still NOT_READY or outright FAILED. */
    private String verdict() {
        boolean unresolved = state.get().leagues().values().stream().anyMatch(
                league -> SyncResult.NOT_READY.name().equals(league.result())
                        || SyncResult.FAILED.name().equals(league.result()));
        return unresolved ? "gave_up" : "ok";
    }

    private void summarise(String word, int attempts, long waitedSeconds) {
        state.updateAndGet(current -> new Snapshot(
                word,
                current.lastAttemptAt(),
                current.lastSuccessAt(),
                attempts,
                waitedSeconds,
                current.leagues()));
    }
}
