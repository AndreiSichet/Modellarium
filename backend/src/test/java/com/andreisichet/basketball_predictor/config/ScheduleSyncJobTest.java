package com.andreisichet.basketball_predictor.config;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.HttpServerErrorException;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClient;
import org.springframework.web.server.ResponseStatusException;

import com.andreisichet.basketball_predictor.dto.InferenceScheduledGame;
import com.andreisichet.basketball_predictor.service.InferenceClient;
import com.andreisichet.basketball_predictor.service.ScheduleSyncService;
import com.andreisichet.basketball_predictor.service.ScheduleSyncStatus;

/**
 * The startup schedule sync survives a cold start, and does not retry what
 * retrying cannot fix.
 *
 * WHAT IS REAL HERE AND WHAT IS A STUB. The job, ScheduleSyncService's
 * syncLeague, the outcome classification and InferenceReadiness's walk of the
 * cause chain are all the production code. Only two things are replaced: the
 * inference client, which throws on cue, and sleeping, which is recorded
 * instead of performed - otherwise the give-up case would take five real
 * minutes.
 *
 * THE EXCEPTIONS THE STUB THROWS ARE THE SHAPE InferenceClient REALLY
 * PRODUCES: a ResponseStatusException whose cause is the original
 * RestClientException. That the real client produces exactly that shape, and
 * that adding the cause changed neither its status nor its message, is proved
 * separately by InferenceClientCauseTest - the two tests are only useful
 * together, because this one would pass against a wrapping that threw the
 * cause away.
 *
 * The repositories are null on purpose. Every outcome exercised here is
 * reached before syncLeague touches one: a throw from the fetch, or an empty
 * fixture list. A null that was wrongly reached would be a
 * NullPointerException, which is a louder failure than a mock returning
 * something plausible.
 */
class ScheduleSyncJobTest {

    /** Sleeping, recorded rather than done. */
    private static final class RecordingWaiter implements ScheduleSyncJob.Waiter {
        private final List<Long> waits = new ArrayList<>();

        @Override
        public void await(long millis) {
            waits.add(millis);
        }

        long totalSeconds() {
            return waits.stream().mapToLong(Long::longValue).sum() / 1000L;
        }
    }

    /**
     * An inference client with a script per league.
     *
     * Each league gets a queue of answers. An answer is either an exception to
     * throw or an empty fixture list, which syncLeague reports as
     * NOTHING_TO_CACHE - a real success that touches no repository. When a
     * queue runs dry the last answer repeats, so "never ready" needs one entry
     * rather than seven.
     */
    private static final class ScriptedClient extends InferenceClient {
        private final Map<String, Deque<RuntimeException>> script =
                new LinkedHashMap<>();
        private final Map<String, Integer> calls = new LinkedHashMap<>();
        private final Map<String, RuntimeException> last = new LinkedHashMap<>();

        ScriptedClient() {
            super(RestClient.builder(), "http://localhost:1");
        }

        /** null means "succeed with no fixtures". */
        ScriptedClient script(String league, RuntimeException... answers) {
            Deque<RuntimeException> queue = new ArrayDeque<>();
            for (RuntimeException answer : answers) {
                queue.add(answer == null ? NONE : answer);
            }
            script.put(league, queue);
            return this;
        }

        private static final RuntimeException NONE =
                new RuntimeException("sentinel: succeed");

        private List<InferenceScheduledGame> answer(String league) {
            calls.merge(league, 1, Integer::sum);
            Deque<RuntimeException> queue =
                    script.getOrDefault(league, new ArrayDeque<>());
            RuntimeException answer = queue.isEmpty()
                    ? last.getOrDefault(league, NONE)
                    : queue.poll();
            last.put(league, answer);
            if (answer != NONE) {
                throw answer;
            }
            return List.of();
        }

        int calls(String league) {
            return calls.getOrDefault(league, 0);
        }

        @Override
        public List<InferenceScheduledGame> fetchSchedule(int daysAhead) {
            return answer("NBA");
        }

        @Override
        public List<InferenceScheduledGame> fetchWnbaSchedule(int daysAhead) {
            return answer("WNBA");
        }

        @Override
        public List<InferenceScheduledGame> fetchGleagueSchedule(int daysAhead) {
            return answer("GLEAGUE");
        }

        @Override
        public List<InferenceScheduledGame> fetchNflSchedule() {
            return answer("NFL");
        }
    }

    private static final String[] LEAGUES = {"NBA", "WNBA", "GLEAGUE", "NFL"};

    /** What InferenceClient.unreachable() produces for a refused connection. */
    private static ResponseStatusException refused() {
        return new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE,
                "Inference service unreachable: Connection refused",
                new ResourceAccessException("I/O error: Connection refused"));
    }

    /** What it produces for a read timeout. */
    private static ResponseStatusException timedOut() {
        return new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE,
                "Inference service unreachable: Read timed out",
                new ResourceAccessException("I/O error: Read timed out"));
    }

    /** Inference answering 503 for itself, mapped by rejected() to 502. */
    private static ResponseStatusException inferenceUnavailable() {
        return new ResponseStatusException(HttpStatus.BAD_GATEWAY,
                "Schedule lookup failed: not ready",
                HttpServerErrorException.create(
                        HttpStatusCode.valueOf(503), "Service Unavailable",
                        org.springframework.http.HttpHeaders.EMPTY,
                        new byte[0], null));
    }

    /** A bad request. Retrying cannot help. */
    private static ResponseStatusException badRequest() {
        return new ResponseStatusException(HttpStatus.BAD_REQUEST,
                "Schedule lookup failed: days_ahead must be positive",
                HttpClientErrorException.create(
                        HttpStatusCode.valueOf(400), "Bad Request",
                        org.springframework.http.HttpHeaders.EMPTY,
                        new byte[0], null));
    }

    /** Inference answering 500. A real fault, not a cold start. */
    private static ResponseStatusException serverFault() {
        return new ResponseStatusException(HttpStatus.BAD_GATEWAY,
                "Schedule lookup failed: boom",
                HttpServerErrorException.create(
                        HttpStatusCode.valueOf(500), "Internal Server Error",
                        org.springframework.http.HttpHeaders.EMPTY,
                        new byte[0], null));
    }

    private record Harness(ScheduleSyncJob job, ScriptedClient client,
                           ScheduleSyncStatus status,
                           RecordingWaiter waiter) {
        ScheduleSyncStatus.Snapshot snapshot() {
            return status.snapshot();
        }
    }

    /**
     * background = false, so syncOnStartup runs on this thread.
     *
     * The production constructor starts a daemon thread, which a test would
     * have to join - and a test that joins a thread with a real five-minute
     * sleep in it is a test nobody runs. The thread is exercised by the real
     * stack runs instead, where it is the only thing that could have made the
     * retry work at all.
     */
    private static Harness harness(ScriptedClient client) {
        ScheduleSyncStatus status = new ScheduleSyncStatus();
        RecordingWaiter waiter = new RecordingWaiter();
        ScheduleSyncService service = new ScheduleSyncService(
                client, null, null, null, 120);
        ScheduleSyncJob job = new ScheduleSyncJob(service, status, waiter,
                false);
        return new Harness(job, client, status, waiter);
    }

    // ---------------------------------------------------------------- cases

    @Test
    void readyAtOnceSyncsEveryLeagueExactlyOnce() {
        Harness h = harness(new ScriptedClient());

        h.job().syncOnStartup();

        ScheduleSyncStatus.Snapshot snapshot = h.snapshot();
        assertThat(snapshot.lastResult()).isEqualTo("ok");
        assertThat(snapshot.startupAttempts()).isEqualTo(1);
        assertThat(snapshot.startupWaitedSeconds()).isZero();
        assertThat(h.waiter().waits).isEmpty();
        for (String league : LEAGUES) {
            assertThat(h.client().calls(league)).isEqualTo(1);
            assertThat(snapshot.leagues().get(league).result())
                    .isEqualTo("NOTHING_TO_CACHE");
        }
    }

    @Test
    void notReadyThreeTimesThenReadySettlesOnTheFourthAttempt() {
        ScriptedClient client = new ScriptedClient();
        for (String league : LEAGUES) {
            client.script(league, refused(), refused(), refused(), null);
        }
        Harness h = harness(client);

        h.job().syncOnStartup();

        ScheduleSyncStatus.Snapshot snapshot = h.snapshot();
        assertThat(snapshot.lastResult()).isEqualTo("ok");
        assertThat(snapshot.startupAttempts()).isEqualTo(4);
        // 5s, 10s, 20s - doubling, three waits for four attempts.
        assertThat(h.waiter().waits)
                .containsExactly(5_000L, 10_000L, 20_000L);
        assertThat(snapshot.startupWaitedSeconds()).isEqualTo(35L);
        for (String league : LEAGUES) {
            assertThat(h.client().calls(league)).isEqualTo(4);
            assertThat(snapshot.leagues().get(league).result())
                    .isEqualTo("NOTHING_TO_CACHE");
        }
    }

    @Test
    void neverReadyGivesUpAtTheCapAndSaysSo() {
        ScriptedClient client = new ScriptedClient();
        for (String league : LEAGUES) {
            client.script(league, refused());
        }
        Harness h = harness(client);

        h.job().syncOnStartup();

        ScheduleSyncStatus.Snapshot snapshot = h.snapshot();
        assertThat(snapshot.lastResult()).isEqualTo("gave_up");
        // 5 + 10 + 20 + 40 + 80 = 155, then the remaining 145 rather than 160:
        // the last wait is clipped to the cap instead of overshooting it.
        assertThat(h.waiter().waits).containsExactly(
                5_000L, 10_000L, 20_000L, 40_000L, 80_000L, 145_000L);
        assertThat(h.waiter().totalSeconds()).isEqualTo(300L);
        assertThat(snapshot.startupAttempts()).isEqualTo(7);
        assertThat(snapshot.startupWaitedSeconds()).isEqualTo(300L);
        for (String league : LEAGUES) {
            assertThat(h.client().calls(league)).isEqualTo(7);
            assertThat(snapshot.leagues().get(league).result())
                    .isEqualTo("NOT_READY");
        }
        assertThat(snapshot.lastSuccessAt()).isNull();
    }

    @Test
    void aBadRequestIsNotRetried() {
        ScriptedClient client = new ScriptedClient();
        client.script("NBA", badRequest());
        Harness h = harness(client);

        h.job().syncOnStartup();

        ScheduleSyncStatus.Snapshot snapshot = h.snapshot();
        assertThat(h.client().calls("NBA")).isEqualTo(1);
        assertThat(h.waiter().waits).isEmpty();
        assertThat(snapshot.startupAttempts()).isEqualTo(1);
        assertThat(snapshot.leagues().get("NBA").result()).isEqualTo("FAILED");
        // One league unresolved is enough for the summary to refuse to say ok,
        // even though the other three synced.
        assertThat(snapshot.lastResult()).isEqualTo("gave_up");
        assertThat(snapshot.leagues().get("WNBA").result())
                .isEqualTo("NOTHING_TO_CACHE");
    }

    @Test
    void aServerFaultThatIsNot503IsNotRetried() {
        ScriptedClient client = new ScriptedClient();
        client.script("NFL", serverFault());
        Harness h = harness(client);

        h.job().syncOnStartup();

        assertThat(h.client().calls("NFL")).isEqualTo(1);
        assertThat(h.waiter().waits).isEmpty();
        assertThat(h.snapshot().leagues().get("NFL").result())
                .isEqualTo("FAILED");
    }

    @Test
    void inferencesOwn503IsRetriedEvenThoughItArrivesAs502() {
        ScriptedClient client = new ScriptedClient();
        client.script("NBA", inferenceUnavailable(), null);
        Harness h = harness(client);

        h.job().syncOnStartup();

        ScheduleSyncStatus.Snapshot snapshot = h.snapshot();
        assertThat(h.client().calls("NBA")).isEqualTo(2);
        assertThat(snapshot.leagues().get("NBA").result())
                .isEqualTo("NOTHING_TO_CACHE");
        assertThat(snapshot.lastResult()).isEqualTo("ok");
    }

    @Test
    void aTimeoutIsRetried() {
        ScriptedClient client = new ScriptedClient();
        client.script("GLEAGUE", timedOut(), null);
        Harness h = harness(client);

        h.job().syncOnStartup();

        assertThat(h.client().calls("GLEAGUE")).isEqualTo(2);
        assertThat(h.snapshot().lastResult()).isEqualTo("ok");
    }

    @Test
    void onlyTheLeagueThatIsNotReadyIsAskedAgain() {
        ScriptedClient client = new ScriptedClient();
        client.script("NFL", refused(), refused(), null);
        Harness h = harness(client);

        h.job().syncOnStartup();

        // THE POINT OF THE WHOLE RETRY BEING PER LEAGUE. Three leagues
        // answered on the first attempt and are never asked again; only the
        // NFL is retried, so a slow or broken league costs nothing to the
        // others.
        assertThat(h.client().calls("NFL")).isEqualTo(3);
        assertThat(h.client().calls("NBA")).isEqualTo(1);
        assertThat(h.client().calls("WNBA")).isEqualTo(1);
        assertThat(h.client().calls("GLEAGUE")).isEqualTo(1);
        assertThat(h.snapshot().lastResult()).isEqualTo("ok");
    }

    @Test
    void theSixHourlyPassDoesNotRetryAtAll() {
        ScriptedClient client = new ScriptedClient();
        for (String league : LEAGUES) {
            client.script(league, refused());
        }
        Harness h = harness(client);

        h.job().run();

        assertThat(h.waiter().waits).isEmpty();
        for (String league : LEAGUES) {
            assertThat(h.client().calls(league)).isEqualTo(1);
        }
        ScheduleSyncStatus.Snapshot snapshot = h.snapshot();
        assertThat(snapshot.lastResult()).isEqualTo("gave_up");
        // AND IT DOES NOT CLAIM A STARTUP IT DID NOT RUN. These two describe
        // the boot; a scheduled pass leaves them where they were.
        assertThat(snapshot.startupAttempts()).isZero();
        assertThat(snapshot.startupWaitedSeconds()).isZero();
    }

    @Test
    void aSixHourlyPassAfterAGoodStartupKeepsTheStartupRecord() {
        ScriptedClient client = new ScriptedClient();
        for (String league : LEAGUES) {
            client.script(league, refused(), null);
        }
        Harness h = harness(client);

        h.job().syncOnStartup();
        assertThat(h.snapshot().startupAttempts()).isEqualTo(2);

        h.job().run();

        ScheduleSyncStatus.Snapshot snapshot = h.snapshot();
        assertThat(snapshot.lastResult()).isEqualTo("ok");
        assertThat(snapshot.startupAttempts()).isEqualTo(2);
        assertThat(snapshot.startupWaitedSeconds()).isEqualTo(5L);
    }

    @Test
    void theDerivedScheduleIsFiveSecondsDoublingToFiveMinutes() {
        // The derivation is in the constants' javadoc: 23x the worst measured
        // readiness of 13.2s, with room for the 5x spread this machine's own
        // Spring startup has shown. Pinned so an edit to either number is a
        // decision rather than a typo.
        assertThat(ScheduleSyncJob.INITIAL_WAIT_MS).isEqualTo(5_000L);
        assertThat(ScheduleSyncJob.MAX_TOTAL_WAIT_MS).isEqualTo(300_000L);
    }

    @Test
    void anAbsentPropertyIsNotTheSameAsZero() {
        // -1 is how "absent" arrives, because 0 is a value a test sets on
        // purpose to switch the retry off. Collapsing the two would disable
        // the retry everywhere while every test still passed.
        assertThat(ScheduleSyncJob.orDerived(-1L, 300_000L)).isEqualTo(300_000L);
        assertThat(ScheduleSyncJob.orDerived(0L, 300_000L)).isZero();
        assertThat(ScheduleSyncJob.orDerived(1_234L, 300_000L)).isEqualTo(1_234L);
    }

    @Test
    void aZeroCapMeansOneAttemptPerLeagueAndNoWaiting() {
        ScriptedClient client = new ScriptedClient();
        for (String league : LEAGUES) {
            client.script(league, refused());
        }
        ScheduleSyncStatus status = new ScheduleSyncStatus();
        RecordingWaiter waiter = new RecordingWaiter();
        ScheduleSyncJob job = new ScheduleSyncJob(
                new ScheduleSyncService(client, null, null, null, 120),
                status, waiter, false, 5_000L, 0L);

        job.syncOnStartup();

        assertThat(waiter.waits).isEmpty();
        assertThat(status.snapshot().startupAttempts()).isEqualTo(1);
        for (String league : LEAGUES) {
            assertThat(client.calls(league)).isEqualTo(1);
        }
    }

    @Test
    void nothingRecordedYetReadsAsNeverRun() {
        ScheduleSyncStatus status = new ScheduleSyncStatus();

        ScheduleSyncStatus.Snapshot snapshot = status.snapshot();

        assertThat(snapshot.lastResult()).isEqualTo("never_run");
        assertThat(snapshot.lastAttemptAt()).isNull();
        assertThat(snapshot.lastSuccessAt()).isNull();
        assertThat(snapshot.leagues()).isEmpty();
    }
}
