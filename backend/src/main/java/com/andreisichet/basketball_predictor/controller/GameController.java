package com.andreisichet.basketball_predictor.controller;

import java.util.List;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.andreisichet.basketball_predictor.dto.ScheduledGameDto;
import com.andreisichet.basketball_predictor.service.ScheduleService;

@RestController
@RequestMapping("/api/games")
public class GameController {
    private final ScheduleService scheduleService;

    public GameController(ScheduleService scheduleService) {
        this.scheduleService = scheduleService;
    }

    /** Real NBA fixtures, straight from the schedule. */
    @GetMapping("/schedule")
    public List<ScheduledGameDto> schedule(
            @RequestParam(defaultValue = "14") int daysAhead) {
        return scheduleService.getSchedule(daysAhead);
    }
}
