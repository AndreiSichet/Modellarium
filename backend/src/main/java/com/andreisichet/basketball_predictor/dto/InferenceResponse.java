package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;

import com.fasterxml.jackson.annotation.JsonProperty;

/** Response from the Python inference service. */
public record InferenceResponse(
        @JsonProperty("data_as_of") LocalDate dataAsOf,
        boolean stale,
        @JsonProperty("days_behind") int daysBehind,
        Predictions predictions,
        /**
         * How much of this season each side has played. ADDED 2026-10-30,
         * LAST AND NULLABLE.
         *
         * Nullable because inference omits it when it cannot be computed,
         * and because every snapshot of this record older than today has no
         * such field - Jackson's tolerance for an absent key is what makes
         * this additive, and that tolerance is proved by the running system
         * rather than only by this comment: NflFreshness declares 8 fields
         * where the wire already sends 11.
         */
        @JsonProperty("early_season") EarlySeason earlySeason) {

    /** The three numbers the early-season note needs, and nothing else. */
    public record EarlySeason(
            int needed,
            @JsonProperty("home_games_played") int homeGamesPlayed,
            @JsonProperty("away_games_played") int awayGamesPlayed,
            @JsonProperty("full_window_date") LocalDate fullWindowDate) {
    }
    public record Predictions(
            @JsonProperty("home_win_probability") double homeWinProbability,
            @JsonProperty("home_margin") double homeMargin,
            @JsonProperty("total_points") double totalPoints,
            @JsonProperty("rebound_margin") double reboundMargin,
            @JsonProperty("total_rebounds") double totalRebounds,
            @JsonProperty("assist_margin") double assistMargin,
            @JsonProperty("total_assists") double totalAssists) {
    }
}
