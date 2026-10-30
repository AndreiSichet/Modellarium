package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;

/** A game as listed by the API. */
public record GameSummaryDto(
        Long id,
        String homeTeamAbbreviation,
        String awayTeamAbbreviation,
        LocalDate gameDate,
        boolean played,
        PredictionDto latestPrediction,
        /**
         * How much of this season each side has played, when it is known.
         *
         * BESIDE latestPrediction, NOT INSIDE IT, and that is the whole
         * design decision. PredictionDto is built from the PERSISTED
         * Prediction entity, so putting these there would mean a migration
         * and a column - and the context is a fact about NOW rather than
         * about the prediction: the same stored prediction is early-season
         * today and ordinary in three weeks. So it rides alongside, and is
         * null on every path that reads a prediction back from the database
         * rather than making one.
         *
         * `id` is still first and `latestPrediction` still carries exactly
         * the fields it carried - section 21's outage came from RETYPING a
         * field on a response like this one, and the lesson was to add
         * rather than change.
         */
        EarlySeasonDto earlySeason) {

    /** The note's inputs, as the browser reads them. */
    public record EarlySeasonDto(
            int needed,
            int homeGamesPlayed,
            int awayGamesPlayed,
            LocalDate fullWindowDate) {
        static EarlySeasonDto from(InferenceResponse.EarlySeason source) {
            return source == null ? null : new EarlySeasonDto(
                    source.needed(),
                    source.homeGamesPlayed(),
                    source.awayGamesPlayed(),
                    source.fullWindowDate());
        }

        /** Public so PredictionService can map without reaching inside. */
        public static EarlySeasonDto of(InferenceResponse inference) {
            return inference == null ? null : from(inference.earlySeason());
        }
    }
}
