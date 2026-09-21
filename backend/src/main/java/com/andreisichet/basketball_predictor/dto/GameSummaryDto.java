package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;

/** A game as listed by the API. */
public record GameSummaryDto(
        Long id,
        String homeTeamAbbreviation,
        String awayTeamAbbreviation,
        LocalDate gameDate,
        boolean played,
        PredictionDto latestPrediction) {
}
