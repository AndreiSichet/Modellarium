package com.andreisichet.basketball_predictor.config;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import com.andreisichet.basketball_predictor.service.InferenceReadiness;
import com.andreisichet.basketball_predictor.service.ScheduleSyncService;
import com.andreisichet.basketball_predictor.service.ScheduleSyncService.SyncOutcome;
import com.andreisichet.basketball_predictor.service.ScheduleSyncService.SyncResult;
import com.andreisichet.basketball_predictor.service.ScheduleSyncStatus;

/** When the schedule cache refreshes. */
@Component
public class ScheduleSyncJob {
    private static final Logger log = LoggerFactory.getLogger(ScheduleSyncJob.class);

    /**
     * The first wait, then doubling, to a total of five minutes.
     *
     * DERIVED FROM MEASUREMENT, WITH A MARGIN THE MEASUREMENT CANNOT EAT.
     * Two cold starts were timed on 2026-10-29. On a `docker compose restart`
     * the backend's startup sync fired 0.5s after Spring reported ready and
     * inference answered its first 200 THIRTEEN POINT TWO seconds after its
     * own start, a 6.3s gap; on a simulated host wake, where the daemon
     * starts all five containers at once with no dependency ordering at all,
     * inference was ready in 11.5s and the gap was 3.7s. So the thing to
     * survive is on the order of ten seconds.
     *
     * Five minutes is about 23x the worse of those two, and the margin is
     * wide on purpose rather than generous: the backend's own startup has
     * been observed anywhere from 5.8s to 28.4s on this machine, a 5x spread,
     * and inference loads 32 model artifacts plus an 80.6 MB player history
     * off a volume that another writer may be swapping at that moment. A cap
     * sized to the measurement would turn an ordinary slow morning into four
     * uncached leagues.
     *
     * The schedule is 5s, 10s, 20s, 40s, 80s, then whatever is left of the
     * 300s - so seven attempts, the last at t=300s. Doubling rather than a
     * fixed interval because the common case is over in one retry and should
     * cost one wait, while the rare slow case should not cost sixty log
     * lines.
     */
    static final long INITIAL_WAIT_MS = 5_000L;
    static final long MAX_TOTAL_WAIT_MS = 300_000L;

    /** Sleeping, injected so a test does not have to take five minutes. */
    interface Waiter {
        void await(long millis) throws InterruptedException;
    }

    /**
     * One league: the label its log lines use, the key its outcome carries,
     * and the call.
     *
     * TWO NAMES, BECAUSE THE TWO EXISTING ONES DISAGREE AND NEITHER MOVES.
     * The warn line below has always said "G League" and ScheduleSyncService
     * has always tagged that league "GLEAGUE"; using one for both would
     * either change a log line or key the status map differently depending on
     * whether the service returned or the wrapper caught - two entries for
     * one league, which is the sort of thing a status page states falsely.
     */
    private record League(String label, String key,
                          Supplier<SyncOutcome> sync) {
    }

    private final ScheduleSyncService scheduleSyncService;
    private final ScheduleSyncStatus status;
    private final Waiter waiter;
    private final boolean background;
    private final long initialWaitMs;
    private final long maxTotalWaitMs;

    /**
     * @Autowired IS REQUIRED HERE, not decoration.
     *
     * This class has two constructors, and Spring only picks one for you when
     * there is exactly one - with two and no annotation it looks for a no-arg
     * constructor, finds none, and the whole context fails to start. The
     * second constructor exists for tests, so this one has to say it is the
     * injectable one.
     *
     * The negative default means "use the derived constant". A test that must
     * not have a background retry running during a measurement sets
     * schedule.sync.startup.max-total-wait-ms=0, which makes the startup
     * listener behave exactly as it did before this file grew a retry: one
     * attempt per league and no waiting.
     */
    @Autowired
    public ScheduleSyncJob(
            ScheduleSyncService scheduleSyncService,
            ScheduleSyncStatus status,
            @Value("${schedule.sync.startup.initial-wait-ms:-1}")
            long initialWaitMs,
            @Value("${schedule.sync.startup.max-total-wait-ms:-1}")
            long maxTotalWaitMs) {
        this(scheduleSyncService, status, Thread::sleep, true,
                orDerived(initialWaitMs, INITIAL_WAIT_MS),
                orDerived(maxTotalWaitMs, MAX_TOTAL_WAIT_MS));
    }

    /**
     * A negative property means "the derived default"; zero means zero.
     *
     * Absence has to be distinguishable from 0, because 0 is the value a test
     * sets on purpose to switch the retry off. An absent property defaulting
     * to 0 would silently disable the retry everywhere - a green startup that
     * does nothing, which is this project's most-repeated failure shape.
     */
    static long orDerived(long configured, long derived) {
        return configured < 0L ? derived : configured;
    }

    ScheduleSyncJob(ScheduleSyncService scheduleSyncService,
                    ScheduleSyncStatus status,
                    Waiter waiter,
                    boolean background) {
        this(scheduleSyncService, status, waiter, background,
                INITIAL_WAIT_MS, MAX_TOTAL_WAIT_MS);
    }

    ScheduleSyncJob(ScheduleSyncService scheduleSyncService,
                    ScheduleSyncStatus status,
                    Waiter waiter,
                    boolean background,
                    long initialWaitMs,
                    long maxTotalWaitMs) {
        this.scheduleSyncService = scheduleSyncService;
        this.status = status;
        this.waiter = waiter;
        this.background = background;
        this.initialWaitMs = initialWaitMs;
        this.maxTotalWaitMs = maxTotalWaitMs;
    }

    /**
     * All four leagues, each through the proxy and each in its own
     * transaction.
     *
     * Called from here rather than from inside ScheduleSyncService, because a
     * method on that class calling its own sibling would bypass Spring's
     * proxy and run both leagues in one transaction - which is exactly the
     * independence this split exists to provide. Each call is also wrapped
     * individually, so one league's failure cannot stop another's sync on the
     * same tick or take the startup listener down with it.
     */
    private List<League> leagues() {
        return List.of(
                new League("NBA", "NBA", scheduleSyncService::syncNba),
                new League("WNBA", "WNBA", scheduleSyncService::syncWnba),
                new League("G League", "GLEAGUE",
                        scheduleSyncService::syncGleague),
                new League("NFL", "NFL", scheduleSyncService::syncNfl));
    }

    /**
     * Every six hours. DOES NOT RETRY, AND THAT IS DELIBERATE.
     *
     * A scheduled tick runs against a stack that has been up for hours, so a
     * failure here is a real fault rather than a race with a cold start - and
     * the next tick is six hours away, which is already a retry. Retrying
     * here would also mean a tick could still be sleeping when the next one
     * fires.
     *
     * Its logging is unchanged: the same warn line below for a throw, and
     * ScheduleSyncService's own lines are emitted from exactly where they
     * were.
     */
    @Scheduled(cron = "${schedule.sync.cron:0 0 */6 * * *}")
    public void run() {
        for (League league : leagues()) {
            status.record(attempt(league));
        }
        status.cycleFinished();
    }

    /**
     * On startup, retrying the leagues that failed because nothing answered.
     *
     * RUNS ON ITS OWN THREAD. ApplicationReadyEvent is published after the
     * web server is accepting connections, so the API is already up - but the
     * publishing thread is the one finishing the bootstrap, and holding it
     * for up to five minutes would leave the process in a state no further
     * runner and no shutdown hook can act on. A daemon thread also cannot
     * keep a stopping JVM alive.
     */
    @EventListener(ApplicationReadyEvent.class)
    public void syncOnStartup() {
        if (!background) {
            retryUntilSettled();
            return;
        }
        Thread thread = new Thread(this::retryUntilSettled,
                "schedule-sync-startup");
        thread.setDaemon(true);
        thread.start();
    }

    /**
     * Attempt every league, then only the ones worth attempting again.
     *
     * RETRIES ON THE ORIGINAL FAILURE, NOT ON THE MAPPED STATUS.
     * InferenceClient turns every RestClientException into a 503 and
     * inference's own 503 into a 502, so the status code cannot tell a
     * refused connection from a bad request. InferenceReadiness walks the
     * cause chain instead; see its docstring.
     *
     * PER LEAGUE, NOT PER RUN. A league that synced is not asked again, and a
     * league that failed for a reason retrying cannot fix is not asked again
     * either - so a genuine NFL fault cannot cost the NBA four extra fetches,
     * which is the same per-league isolation the daily refresh job needed.
     */
    void retryUntilSettled() {
        log.info("Running the initial schedule sync so a fresh database is not empty.");

        List<League> pending = leagues();
        long nextWaitMs = initialWaitMs;
        long waitedMs = 0L;
        int attempts = 0;

        while (true) {
            attempts++;
            List<League> again = new ArrayList<>();
            for (League league : pending) {
                SyncOutcome outcome = attempt(league);
                status.record(outcome);
                if (outcome.retryable()) {
                    again.add(league);
                }
            }

            if (again.isEmpty()) {
                status.startupFinished(attempts, waitedMs / 1000L);
                if (attempts > 1) {
                    log.info("Initial schedule sync settled on attempt {} "
                            + "after waiting {}s for the inference service.",
                            attempts, waitedMs / 1000L);
                }
                return;
            }

            long remainingMs = maxTotalWaitMs - waitedMs;
            if (remainingMs <= 0L) {
                status.startupFinished(attempts, waitedMs / 1000L);
                log.error("Initial schedule sync gave up after {} attempts "
                        + "over {}s; {} league(s) still not ready: {}. The "
                        + "six-hourly sync will try again.",
                        attempts, waitedMs / 1000L, again.size(),
                        again.stream().map(League::label).toList());
                return;
            }

            long waitMs = Math.min(nextWaitMs, remainingMs);
            log.debug("Initial schedule sync attempt {}: {} league(s) not "
                    + "ready ({}), retrying in {}s.",
                    attempts, again.size(),
                    again.stream().map(League::label).toList(),
                    waitMs / 1000L);
            status.startupRetrying(attempts, waitedMs / 1000L);

            try {
                waiter.await(waitMs);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                status.startupFinished(attempts, waitedMs / 1000L);
                log.warn("Initial schedule sync interrupted after {} "
                        + "attempt(s).", attempts);
                return;
            }

            waitedMs += waitMs;
            nextWaitMs *= 2L;
            pending = again;
        }
    }

    /**
     * One league, with the throw it could still produce turned into an
     * outcome.
     *
     * ScheduleSyncService catches its own fetch failure, so this almost never
     * fires - but @Transactional can still throw on commit, and this wrapper
     * was here before the file grew a retry. The warn text is the one it
     * always used.
     */
    private SyncOutcome attempt(League league) {
        try {
            return league.sync().get();
        } catch (Exception error) {
            log.warn("{} schedule sync failed: {}",
                    league.label(), error.getMessage());
            return new SyncOutcome(league.key(),
                    InferenceReadiness.isNotReady(error)
                            ? SyncResult.NOT_READY
                            : SyncResult.FAILED,
                    error.getMessage());
        }
    }
}
