package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;

import com.fasterxml.jackson.annotation.JsonProperty;

/** Body sent to the Python inference service. */
public record InferenceRequest(
        @JsonProperty("home_team_id") Long homeTeamId,
        @JsonProperty("away_team_id") Long awayTeamId,
        @JsonProperty("game_date") LocalDate gameDate) {
}
