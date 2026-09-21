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
        scheduleSyncService.sync();
    }

    @EventListener(ApplicationReadyEvent.class)
    public void syncOnStartup() {
        log.info("Running the initial schedule sync so a fresh database is not empty.");
        scheduleSyncService.sync();
    }
}
