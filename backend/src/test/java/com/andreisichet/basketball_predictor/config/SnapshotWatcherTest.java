package com.andreisichet.basketball_predictor.config;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.web.client.ResourceAccessException;

import com.andreisichet.basketball_predictor.dto.InferenceHealth;
import com.andreisichet.basketball_predictor.service.InferenceClient;

/**
 * The snapshot watcher re-syncs when the data moves and at no other time.
 *
 * EVERY CASE HERE IS ABOUT NOT SYNCING. Syncing when the snapshot moved is
 * one line and one test; the four ways to sync when it did not are what make
 * a five-minute poll either harmless or a self-inflicted load generator, so
 * they get the other five.
 */
class SnapshotWatcherTest {

    private static InferenceHealth healthWith(String snapshot) {
        return new InferenceHealth(
                "ok", null, null, 0, false, null, null, null, null,
                snapshot == null ? null
                        : new InferenceHealth.ServedData(snapshot));
    }

    private record Fixture(SnapshotWatcher watcher,
                           InferenceClient inference,
                           ScheduleSyncJob job) {
    }

    private static Fixture fixture(boolean enabled) {
        InferenceClient inference = mock(InferenceClient.class);
        ScheduleSyncJob job = mock(ScheduleSyncJob.class);
        return new Fixture(new SnapshotWatcher(inference, job, enabled),
                inference, job);
    }

    @Test
    @DisplayName("the first poll is a baseline, never a sync")
    void firstPollDoesNotSync() {
        Fixture f = fixture(true);
        when(f.inference().getHealth()).thenReturn(healthWith("2026-11-07T0830"));

        f.watcher().check();

        // The startup sync has already run against this snapshot. Treating
        // the first observation as a change would make every container start
        // pay for two full syncs.
        verify(f.job(), never()).run();
        assertThat(f.watcher().lastSeenSnapshot()).isEqualTo("2026-11-07T0830");
    }

    @Test
    @DisplayName("an unchanged snapshot does not sync, however often it polls")
    void unchangedDoesNotSync() {
        Fixture f = fixture(true);
        when(f.inference().getHealth()).thenReturn(healthWith("2026-11-07T0830"));

        for (int i = 0; i < 12; i++) {
            f.watcher().check();
        }

        verify(f.job(), never()).run();
    }

    @Test
    @DisplayName("a moved snapshot syncs exactly once, then settles")
    void movedSyncsOnce() {
        Fixture f = fixture(true);
        when(f.inference().getHealth())
                .thenReturn(healthWith("2026-11-06T1249"))
                .thenReturn(healthWith("2026-11-07T0830"))
                .thenReturn(healthWith("2026-11-07T0830"))
                .thenReturn(healthWith("2026-11-07T0830"));

        f.watcher().check();   // baseline
        f.watcher().check();   // the swap
        f.watcher().check();
        f.watcher().check();

        // Once for the move and not again: the new snapshot becomes the
        // baseline, so a swap costs one sync rather than one per poll until
        // the next swap.
        verify(f.job(), times(1)).run();
        assertThat(f.watcher().lastSeenSnapshot()).isEqualTo("2026-11-07T0830");
    }

    @Test
    @DisplayName("inference being unreachable is 'cannot tell', not a change")
    void unreachableDoesNotSync() {
        Fixture f = fixture(true);
        when(f.inference().getHealth())
                .thenReturn(healthWith("2026-11-07T0830"))
                .thenThrow(new ResourceAccessException("connection refused"))
                .thenReturn(healthWith("2026-11-07T0830"));

        f.watcher().check();
        f.watcher().check();
        f.watcher().check();

        // And the baseline must SURVIVE the failure: if the throw had reset
        // it to null, the next good poll would read as a first poll and the
        // watcher would be permanently one swap behind.
        verify(f.job(), never()).run();
        assertThat(f.watcher().lastSeenSnapshot()).isEqualTo("2026-11-07T0830");
    }

    @Test
    @DisplayName("an inference with no served_data block causes no churn")
    void missingBlockDoesNotSync() {
        Fixture f = fixture(true);
        when(f.inference().getHealth())
                .thenReturn(healthWith("2026-11-07T0830"))
                .thenReturn(healthWith(null))
                .thenReturn(healthWith(null));

        f.watcher().check();
        f.watcher().check();
        f.watcher().check();

        verify(f.job(), never()).run();
        assertThat(f.watcher().lastSeenSnapshot()).isEqualTo("2026-11-07T0830");
    }

    @Test
    @DisplayName("disabled means it never even asks")
    void disabledAsksNothing() {
        Fixture f = fixture(false);

        f.watcher().check();
        f.watcher().check();

        verify(f.inference(), never()).getHealth();
        verify(f.job(), never()).run();
        assertThat(f.watcher().lastSeenSnapshot()).isNull();
    }

    @Test
    @DisplayName("a null health body is tolerated like any other unknown")
    void nullHealthDoesNotSync() {
        Fixture f = fixture(true);
        when(f.inference().getHealth()).thenReturn(null);

        f.watcher().check();

        verify(f.job(), never()).run();
        assertThat(f.watcher().lastSeenSnapshot()).isNull();
    }

    @Test
    @DisplayName("the sync's own failure does not poison the baseline")
    void aFailingSyncStillAdvancesTheBaseline() {
        Fixture f = fixture(true);
        when(f.inference().getHealth())
                .thenReturn(healthWith("2026-11-06T1249"))
                .thenReturn(healthWith("2026-11-07T0830"));
        // ScheduleSyncJob.run() records per-league outcomes and does not
        // throw, but if a future change made it throw, the watcher must not
        // retry it every five minutes forever. The baseline is advanced
        // BEFORE the sync is attempted, so a throw costs one missed re-sync
        // and not an unbounded loop; the six-hourly tick is the backstop.
        org.mockito.Mockito.doThrow(new IllegalStateException("boom"))
                .when(f.job()).run();

        f.watcher().check();
        try {
            f.watcher().check();
        } catch (IllegalStateException expected) {
            // the scheduler logs and the next tick continues
        }

        assertThat(f.watcher().lastSeenSnapshot()).isEqualTo("2026-11-07T0830");
        verify(f.job(), times(1)).run();
        verify(f.inference(), times(2)).getHealth();
        verify(f.job(), never()).syncOnStartup();
    }
}
