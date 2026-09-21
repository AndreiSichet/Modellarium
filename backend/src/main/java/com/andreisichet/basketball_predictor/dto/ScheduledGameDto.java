package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;

/** An upcoming fixture offered to the client as a candidate to predict. */
public record ScheduledGameDto(
        Long homeTeamId,
        String homeTeamAbbr,
        String homeTeamName,
        Long awayTeamId,
        String awayTeamAbbr,
        String awayTeamName,
        LocalDate gameDate) {
}
