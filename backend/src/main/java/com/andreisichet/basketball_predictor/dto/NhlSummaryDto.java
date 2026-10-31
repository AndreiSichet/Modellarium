package com.andreisichet.basketball_predictor.dto;

import java.time.Instant;
import java.time.LocalDate;

import com.andreisichet.basketball_predictor.model.NhlPrediction;

/**
 * What the backend returns for an NHL prediction.
 *
 * <p>TOTALS ARE DECLARED, NOT OMITTED. {@code totalsServed} is false and
 * {@code totalsReasonCode} carries {@code no_skill_over_constant}, so a
 * client renders "not offered, and here is why" instead of leaving a gap a
 * reader has to interpret.
 *
 * <p>THE WINNER'S CAVEAT IS ALWAYS PRESENT. Its test interval against a
 * constant includes zero, and the project's rule is that the figure never
 * travels without the sentence.
 */
public record NhlSummaryDto(
        Long gameId,
        String homeTeamAbbreviation,
        String awayTeamAbbreviation,
        LocalDate gameDate,
        Prediction prediction) {

    public record Prediction(
            double homeWinProbability,
            double homeWinsByTwoOrMore,
            String winnerModel,
            String winnerCaveat,
            String puckLineDefinition,
            boolean totalsServed,
            String totalsReasonCode,
            String totalsDetail,
            String season,
            String predictionRule,
            LocalDate dataAsOf,
            boolean stale,
            int daysBehind,
            String source,
            Instant predictedAt) {

        public static Prediction of(
                NhlPrediction saved,
                InferenceNhlResponse inference,
                InferenceNhlResponse.Market winner,
                InferenceNhlResponse.Market puckLine,
                InferenceNhlResponse.Market totals) {
            return new Prediction(
                    saved.getHomeWinProbability(),
                    saved.getHomeWinsByTwoOrMore(),
                    winner == null ? null : winner.family(),
                    winner == null ? null : winner.caveat(),
                    puckLine == null ? null : puckLine.definition(),
                    totals != null && totals.served(),
                    totals == null ? null : totals.reasonCode(),
                    totals == null ? null : totals.detail(),
                    inference.season(),
                    inference.predictionRule(),
                    saved.getDataAsOf(),
                    saved.isStale(),
                    inference.daysBehind(),
                    inference.source(),
                    saved.getPredictedAt());
        }
    }
}
