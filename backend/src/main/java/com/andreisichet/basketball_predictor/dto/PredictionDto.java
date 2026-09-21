package com.andreisichet.basketball_predictor.dto;

import java.time.Instant;
import java.time.LocalDate;

import com.andreisichet.basketball_predictor.model.Prediction;

/** Model output as sent to clients. */
public record PredictionDto(
        double homeWinProbability,
        double homeMargin,
        double totalPoints,
        double reboundMargin,
        double totalRebounds,
        double assistMargin,
        double totalAssists,
        LocalDate dataAsOf,
        boolean stale,
        Instant predictedAt) {
    public static PredictionDto from(Prediction prediction) {
        return new PredictionDto(
                prediction.getHomeWinProbability(),
                prediction.getHomeMargin(),
                prediction.getTotalPoints(),
                prediction.getReboundMargin(),
                prediction.getTotalRebounds(),
                prediction.getAssistMargin(),
                prediction.getTotalAssists(),
                prediction.getDataAsOf(),
                prediction.isStale(),
                prediction.getPredictedAt());
    }
}
