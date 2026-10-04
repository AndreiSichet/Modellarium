package com.andreisichet.basketball_predictor.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import com.andreisichet.basketball_predictor.service.ScheduleSyncService;

/** When the schedule cache refreshes. */
@Component
public class ScheduleSyncJob {
    private static final Logger log = LoggerFactory.getLogger(ScheduleSyncJob.class);

    private final ScheduleSyncService scheduleSyncService;

    public ScheduleSyncJob(ScheduleSyncService scheduleSyncService) {
        this.scheduleSyncService = scheduleSyncService;
    }

    /** Every six hours. */
    @Scheduled(cron = "${schedule.sync.cron:0 0 */6 * * *}")
    public void run() {
        syncBothLeagues();
    }

    @EventListener(ApplicationReadyEvent.class)
    public void syncOnStartup() {
        log.info("Running the initial schedule sync so a fresh database is not empty.");
        syncBothLeagues();
    }

    /**
     * Both leagues, each through the proxy and each in its own transaction.
     *
     * Called from here rather than from inside ScheduleSyncService, because a
     * method on that class calling its own sibling would bypass Spring's
     * proxy and run both leagues in one transaction - which is exactly the
     * independence this split exists to provide. The WNBA call is also
     * wrapped, so a failure there cannot stop the NBA sync on a later tick or
     * take the startup listener down with it.
     */
    private void syncBothLeagues() {
        try {
            scheduleSyncService.syncNba();
        } catch (Exception error) {
            log.warn("NBA schedule sync failed: {}", error.getMessage());
        }
        try {
            scheduleSyncService.syncWnba();
        } catch (Exception error) {
            log.warn("WNBA schedule sync failed: {}", error.getMessage());
        }
    }
}
