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
        boolean stale,
        LeagueFreshness wnba,
        LeagueFreshness gleague) {
    /**
     * The WNBA's own cutoff, which is a different date from the NBA's.
     *
     * Declared rather than ignored because the frontend decides which
     * fixtures are predictable from dataAsOf, and MAX_DAYS_AHEAD is one day.
     * Reading the NBA's cutoff for a WNBA fixture would disable every WNBA
     * game whenever the two leagues' data ended on different dates - which is
     * the normal case, since their seasons barely overlap.
     *
     * Nullable on purpose: an older inference service does not send it, and
     * Jackson ignores unknown properties rather than failing, so a backend
     * deployed ahead of the Python side gets null here instead of a 500.
     */
    public record LeagueFreshness(
            @JsonProperty("data_as_of") LocalDate dataAsOf,
            @JsonProperty("days_behind") int daysBehind,
            boolean stale) {
    }
}
