package com.andreisichet.basketball_predictor.dto;

import java.time.Instant;
import java.time.LocalDate;

import com.andreisichet.basketball_predictor.model.QuarterHalfPrediction;

/** What POST /api/predictions/quarter-half returns. */
public record QuarterHalfSummaryDto(
        Long gameId,
        String homeTeamAbbreviation,
        String awayTeamAbbreviation,
        LocalDate gameDate,
        Prediction prediction) {
    public record Prediction(
            double q1Spread,
            double q1Total,
            double q1WinnerProbability,
            String q1WinnerConfidence,
            String q1WinnerInterpretation,
            double half1Spread,
            double half1Total,
            double half1WinnerProbability,
            String half1WinnerConfidence,
            String half1WinnerInterpretation,
            LocalDate dataAsOf,
            boolean stale,
            int daysBehind,
            Instant predictedAt) {
        /** Numbers from the persisted row, qualifiers from the live payload. */
        public static Prediction of(
                QuarterHalfPrediction saved,
                InferenceQuarterHalfResponse inference,
                InferenceQuarterHalfResponse.Market q1Winner,
                InferenceQuarterHalfResponse.Market half1Winner) {
            return new Prediction(
                    saved.getQ1Spread(),
                    saved.getQ1Total(),
                    saved.getQ1WinnerProbability(),
                    q1Winner.confidence(),
                    q1Winner.interpretation(),
                    saved.getHalf1Spread(),
                    saved.getHalf1Total(),
                    saved.getHalf1WinnerProbability(),
                    half1Winner.confidence(),
                    half1Winner.interpretation(),
                    saved.getDataAsOf(),
                    saved.isStale(),
                    inference.daysBehind(),
                    saved.getPredictedAt());
        }
    }
}
