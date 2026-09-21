package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;

import com.fasterxml.jackson.annotation.JsonProperty;

/** One fixture as the Python service returns it from GET /schedule. */
public record InferenceScheduledGame(
        @JsonProperty("home_team_id") Long homeTeamId,
        @JsonProperty("away_team_id") Long awayTeamId,
        @JsonProperty("game_date") LocalDate gameDate) {
}
