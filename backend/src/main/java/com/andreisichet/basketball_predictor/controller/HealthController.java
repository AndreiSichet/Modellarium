package com.andreisichet.basketball_predictor.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.andreisichet.basketball_predictor.dto.HealthDto;
import com.andreisichet.basketball_predictor.service.ScheduleService;

/** Pass-through to the inference service's freshness report. */
@RestController
@RequestMapping("/api/health")
public class HealthController {
    private final ScheduleService scheduleService;

    public HealthController(ScheduleService scheduleService) {
        this.scheduleService = scheduleService;
    }

    @GetMapping
    public HealthDto health() {
        return scheduleService.getHealth();
    }
}
