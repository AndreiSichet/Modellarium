package com.andreisichet.basketball_predictor.dto;

import java.time.Instant;
import java.time.LocalDate;

import com.andreisichet.basketball_predictor.model.GleaguePrediction;

/**
 * The client-facing G League response.
 *
 * NO ENGINEERING NOTE IS CARRIED, and the manifest has one that invites it:
 * it records that every candidate tied on validation, so the shipped
 * configuration is "defensible, not demonstrated". That is true, useful to
 * whoever retrains this, and NOT a property of any one prediction - which is
 * the test CLAUDE.md section 8 sets for what a response may carry. The WNBA
 * phase shipped exactly that kind of note over the wire and had to withdraw
 * it from the frontend, this DTO and the Python response in turn.
 *
 * The three windows ARE carried, because a client comparing two markets
 * should be able to see which window produced each. That is a fact about this
 * response rather than a caveat about the project - and all three currently
 * read CARRY10, which is itself worth being able to observe.
 */
public record GleagueSummaryDto(
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
            String season,
            LocalDate dataAsOf,
            boolean stale,
            int daysBehind,
            Instant predictedAt) {

        public static Prediction of(
                GleaguePrediction saved,
                InferenceGleagueResponse inference,
                InferenceGleagueResponse.Market moneyline,
                InferenceGleagueResponse.Market spread,
                InferenceGleagueResponse.Market totals) {
            return new Prediction(
                    saved.getHomeWinProbability(),
                    saved.getHomeMargin(),
                    saved.getTotalPoints(),
                    moneyline.window(),
                    spread.window(),
                    totals.window(),
                    inference.season(),
                    saved.getDataAsOf(),
                    saved.isStale(),
                    inference.daysBehind(),
                    saved.getPredictedAt());
        }
    }
}
