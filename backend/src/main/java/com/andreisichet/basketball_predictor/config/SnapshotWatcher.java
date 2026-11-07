package com.andreisichet.basketball_predictor.config;

import java.util.concurrent.atomic.AtomicReference;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import com.andreisichet.basketball_predictor.dto.InferenceHealth;
import com.andreisichet.basketball_predictor.service.InferenceClient;

/**
 * Re-sync the schedule when the served snapshot moves, not only on the clock.
 *
 * THE GAP THIS CLOSES WAS MEASURED, NOT IMAGINED. The sync runs at 00, 06, 12
 * and 18 UTC; the daily refresh swaps the pointer at about 08:30-08:45. So
 * every morning there is a window between the swap and the 12:00 tick in
 * which the backend's {@code game} rows describe yesterday's snapshot. On
 * 2026-11-05 at 09:00 UTC that window was at its worst: inference served five
 * predictable NHL fixtures and the backend held none at all, because the
 * 06:00 sync ran two and a half hours before the 08:43 swap. Replayed across
 * every reading since the NHL went live, one of six diverged - and all six
 * were inside two days, because only four of the seven retained snapshots
 * carry NHL tables at all.
 *
 * WHY POLLING AND NOT A PUSH. The refresh already restarts
 * {@code inference-service} on a swap, but nothing tells the backend, and
 * giving the refresh a second thing to notify would put a deploy-time
 * dependency between a Python job and a Java service that currently share
 * only a volume and an HTTP contract. A poll needs no coordination and fails
 * safe: if it cannot reach inference it simply tries again later.
 *
 * WHAT IT COSTS. One {@code GET /health} per interval, which is budgeted to
 * {@code HEALTH_BUDGET_SECONDS} and measured at about 0.09 s, plus one full
 * sync on the days the snapshot moves - measured at 2.4 s for all five
 * leagues, warm. That is at most one extra sync a day, and it is the reason
 * the 00:00 and 18:00 ticks could be retired later if anyone wants to; they
 * are left alone here because removing a safety net is a separate decision
 * from adding a trigger.
 *
 * WHAT IT DOES NOT DO. It never syncs on the first poll. At that point the
 * startup sync has already run against whatever snapshot is current, so the
 * first observation is a baseline and not a change; syncing on it would mean
 * every container start paid for two full syncs. It also treats a missing
 * {@code served_data} block as "cannot tell" rather than as a change, so an
 * inference service too old to send it causes no churn.
 */
@Component
public class SnapshotWatcher {
    private static final Logger log =
            LoggerFactory.getLogger(SnapshotWatcher.class);

    private final InferenceClient inference;
    private final ScheduleSyncJob job;
    private final boolean enabled;

    /**
     * The snapshot the cache was last known to describe.
     *
     * An AtomicReference rather than a plain field because the scheduler's
     * pool and the startup thread are different threads, and a torn read
     * here would mean either a missed re-sync or a duplicated one.
     */
    private final AtomicReference<String> lastSeen = new AtomicReference<>();

    public SnapshotWatcher(
            InferenceClient inference,
            ScheduleSyncJob job,
            @Value("${schedule.sync.snapshot-watch.enabled:true}")
            boolean enabled) {
        this.inference = inference;
        this.job = job;
        this.enabled = enabled;
    }

    /**
     * Five minutes by default, and the number is a ceiling on staleness.
     *
     * fixedDelay rather than fixedRate: the delay is measured from the end of
     * the previous run, so a slow {@code /health} or a 2.4 s sync cannot
     * cause runs to queue up behind each other. That is the same failure mode
     * as the Docker healthcheck loop measured on 6 November, where a 5 s
     * interval against a handler slower than its own timeout fed itself to
     * 128% CPU.
     */
    @Scheduled(
            fixedDelayString = "${schedule.sync.snapshot-watch.ms:300000}",
            initialDelayString = "${schedule.sync.snapshot-watch.ms:300000}")
    public void check() {
        if (!enabled) {
            return;
        }
        String snapshot = currentSnapshot();
        if (snapshot == null) {
            return;
        }
        String previous = lastSeen.getAndSet(snapshot);
        if (previous == null) {
            log.debug("Snapshot watch starting from {}", snapshot);
            return;
        }
        if (previous.equals(snapshot)) {
            return;
        }
        log.info("Served snapshot moved {} -> {}; re-syncing the schedule "
                + "rather than waiting for the next six-hourly tick.",
                previous, snapshot);
        job.run();
    }

    /**
     * The served snapshot, or null when it cannot be read.
     *
     * EVERY FAILURE IS "CANNOT TELL", DELIBERATELY. Inference being down,
     * answering slowly or answering with a body this backend cannot parse are
     * all reasons to leave the cache alone and look again in five minutes -
     * none of them is evidence that the data moved. Throwing here would also
     * kill the scheduled task for the life of the process, which would turn
     * one bad minute into a permanently disabled watcher.
     */
    private String currentSnapshot() {
        try {
            InferenceHealth health = inference.getHealth();
            InferenceHealth.ServedData served =
                    health == null ? null : health.servedData();
            return served == null ? null : served.snapshot();
        } catch (RuntimeException error) {
            log.debug("Snapshot watch could not read /health: {}",
                    error.toString());
            return null;
        }
    }

    /** What the watcher believes is cached, for tests and for /api/health. */
    public String lastSeenSnapshot() {
        return lastSeen.get();
    }
}
