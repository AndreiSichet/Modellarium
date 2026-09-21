package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonProperty;

/** Response from the Python service's GET /health. */
public record InferenceHealth(
        String status,
        @JsonProperty("models_loaded") Map<String, Integer> modelsLoaded,
        @JsonProperty("data_as_of") LocalDate dataAsOf,
        @JsonProperty("days_behind") int daysBehind,
        boolean stale) {
}
