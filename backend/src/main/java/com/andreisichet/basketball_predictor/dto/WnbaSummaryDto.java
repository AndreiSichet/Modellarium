package com.andreisichet.basketball_predictor.dto;

import java.time.Instant;
import java.time.LocalDate;

import com.andreisichet.basketball_predictor.model.WnbaPrediction;

/** What POST /api/predictions/wnba returns. */
public record WnbaSummaryDto(
        Long gameId,
        String homeTeamAbbreviation,
        String awayTeamAbbreviation,
        LocalDate gameDate,
        Prediction prediction) {
    public record Prediction(
            double homeWinProbability,
            double homeMargin,
            double totalPoints,
            String moneylineWindow,
            String spreadWindow,
            String totalsWindow,
            String moneylineCaveat,
            LocalDate dataAsOf,
            boolean stale,
            int daysBehind,
            Instant predictedAt) {
        /**
         * Numbers from the persisted row, qualifiers from the live payload.
         *
         * The windows and the caveat are facts about WHICH MODEL produced a
         * number, identical on every row this table will ever hold, so by the
         * rule in CLAUDE.md section 8 they are constants rather than data and
         * are not persisted. moneylineCaveat is never omitted and never
         * abbreviated: phase 3 measured Elo alone beating this model on the
         * test seasons, and a prediction without its caveat misleads.
         */
        public static Prediction of(
                WnbaPrediction saved,
                InferenceWnbaResponse inference,
                InferenceWnbaResponse.Market moneyline,
                InferenceWnbaResponse.Market spread,
                InferenceWnbaResponse.Market totals) {
            return new Prediction(
                    saved.getHomeWinProbability(),
                    saved.getHomeMargin(),
                    saved.getTotalPoints(),
                    moneyline.window(),
                    spread.window(),
                    totals.window(),
                    moneyline.caveat(),
                    saved.getDataAsOf(),
                    saved.isStale(),
                    inference.daysBehind(),
                    saved.getPredictedAt());
        }
    }
}
