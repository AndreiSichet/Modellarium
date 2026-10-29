package com.andreisichet.basketball_predictor.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.andreisichet.basketball_predictor.dto.HealthDto;
import com.andreisichet.basketball_predictor.dto.HealthDto.ScheduleSyncDto;
import com.andreisichet.basketball_predictor.service.ScheduleService;
import com.andreisichet.basketball_predictor.service.ScheduleSyncStatus;

/** Pass-through to the inference service's freshness report. */
@RestController
@RequestMapping("/api/health")
public class HealthController {
    private final ScheduleService scheduleService;
    private final ScheduleSyncStatus scheduleSyncStatus;

    public HealthController(ScheduleService scheduleService,
                            ScheduleSyncStatus scheduleSyncStatus) {
        this.scheduleService = scheduleService;
        this.scheduleSyncStatus = scheduleSyncStatus;
    }

    /**
     * UNCHANGED, INCLUDING ITS 503. This calls inference and lets the failure
     * propagate, so when inference is down this endpoint is down. That was
     * considered and kept: an endpoint answering 200 while the service it
     * reports on is unreachable is the misleading green this project keeps
     * removing, and quietening a real red to expose a new field would be a
     * bad trade. The sub-resource below is the additive answer instead.
     */
    @GetMapping
    public HealthDto health() {
        return scheduleService.getHealth();
    }

    /**
     * Whether this backend cached its fixtures, answerable when inference is
     * not.
     *
     * THE ONE TIME THE BLOCK IS INTERESTING IS THE ONE TIME /api/health
     * CANNOT SERVE IT. During the startup retry window `lastResult` reads
     * `retrying`, and if the retry gives up it reads `gave_up` - and in both
     * of those states inference is by definition not answering, so the
     * pass-through above is a 503 and the block is unreadable. Until this
     * existed the backend's container log was the only instrument.
     *
     * TOUCHES NOTHING. It reads an in-memory snapshot and makes no outbound
     * call, so it cannot fail for the reason it exists to report on. The
     * same object is still carried on /api/health's `scheduleSync` field, so
     * nothing is duplicated and no existing client changes.
     */
    @GetMapping("/schedule-sync")
    public ScheduleSyncDto scheduleSync() {
        return ScheduleSyncDto.from(scheduleSyncStatus.snapshot());
    }
}
