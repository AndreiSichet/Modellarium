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
            LocalDate dataAsOf,
            boolean stale,
            int daysBehind,
            Instant predictedAt) {
        /**
         * Numbers from the persisted row, provenance from the live payload.
         *
         * The windows are facts about WHICH MODEL produced a number,
         * identical on every row this table will ever hold, so by the rule in
         * CLAUDE.md section 8 they are constants rather than data and are not
         * persisted.
         *
         * THE MANIFEST'S CAVEAT IS NOT CARRIED AT ALL. It records bare Elo
         * scoring 0.6046 on test against this model's 0.6130, which was
         * served for a day as a client-readable weakness. A paired bootstrap
         * puts that difference at +0.008301, 95% interval
         * [-0.005932, +0.022443], spanning zero on all ten seeds - so it is
         * not a reliable difference, and carrying it invited clients to show
         * a null result as a finding. It stays in models_wnba/manifest.json.
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
                    saved.getDataAsOf(),
                    saved.isStale(),
                    inference.daysBehind(),
                    saved.getPredictedAt());
        }
    }
}
