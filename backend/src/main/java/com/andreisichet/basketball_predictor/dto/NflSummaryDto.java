package com.andreisichet.basketball_predictor.dto;

import java.time.Instant;
import java.time.LocalDate;
import java.util.List;

import com.andreisichet.basketball_predictor.model.NflPrediction;

/**
 * The client-facing NFL response.
 *
 * CARRIES THE LICENCE, WHICH NO OTHER LEAGUE'S DTO DOES. The NFL history is
 * derived from English Wikipedia and is CC BY-SA 4.0. The three basketball
 * leagues' data comes from nba_api under different terms and carries no
 * attribution requirement, so this field exists here and nowhere else rather
 * than being added to all four for symmetry. Phase 5 renders it; it is on the
 * wire so a client cannot show the number without being able to show where it
 * came from.
 *
 * Carries the winner interpretation and the per-market notes too. Those are
 * constants about which model produced a number, so they are NOT stored on the
 * entity - they are attached here from the live inference payload, which is
 * the same arrangement the quarter/half qualifiers use.
 */
public record NflSummaryDto(
        Long gameId,
        String homeTeamAbbreviation,
        String awayTeamAbbreviation,
        LocalDate gameDate,
        Prediction prediction) {

    public record Prediction(
            double homeWinProbability,
            double homeMargin,
            double totalPoints,
            String homeWinInterpretation,
            String winnerModel,
            String marginModel,
            String totalsModel,
            String winnerNote,
            String marginNote,
            String totalsNote,
            List<String> imputedFeatures,
            int season,
            int week,
            LocalDate dataAsOf,
            boolean stale,
            int daysBehind,
            String source,
            Instant predictedAt) {

        public static Prediction of(
                NflPrediction saved,
                InferenceNflResponse inference,
                InferenceNflResponse.Market winner,
                InferenceNflResponse.Market margin,
                InferenceNflResponse.Market totals) {
            return new Prediction(
                    saved.getHomeWinProbability(),
                    saved.getHomeMargin(),
                    saved.getTotalPoints(),
                    inference.homeWinInterpretation(),
                    winner.modelUsed(),
                    margin.modelUsed(),
                    totals.modelUsed(),
                    winner.note(),
                    margin.note(),
                    totals.note(),
                    // The three markets share a feature list prefix, so the
                    // winner's imputed set is the one a client needs to see.
                    // Flattened rather than nested per market because the
                    // reason a feature is imputed is a property of the FIXTURE
                    // - a season opener has no rest value for anybody - not of
                    // one model's view of it.
                    winner.imputed(),
                    inference.season(),
                    saved.getWeek(),
                    saved.getDataAsOf(),
                    saved.isStale(),
                    inference.daysBehind(),
                    inference.source(),
                    saved.getPredictedAt());
        }
    }
}
