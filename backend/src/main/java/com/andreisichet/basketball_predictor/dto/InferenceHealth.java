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
        LeagueFreshness gleague,
        NflFreshness nfl) {
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

    /**
     * The NFL's freshness, plus the two fields no other league has.
     *
     * THIS BLOCK HAS BEEN SILENTLY DROPPED TWICE, which is why it is declared
     * in the same change that adds it on the Python side and why the
     * regression gate names added fields per body. The WNBA's block was lost
     * here first, then the G League's; both times the Python service was
     * sending it and these records were discarding it, and both times the
     * gate's "fields added" line is what showed it rather than anything
     * failing.
     *
     * {@code predictableFixtures} is the field that makes this league
     * different. The other three are served under MAX_DAYS_AHEAD = 1, so a
     * client can compute what is predictable from the cutoff: cutoff plus one
     * day. The NFL's rule is a DEPENDENCY - both teams' previous games in
     * history - so it cannot be computed from a date at all, and a client
     * that guessed cutoff plus one would be wrong about every fixture. The
     * count comes from the service that knows.
     *
     * {@code synthetic} is true only in CI, where the real CC BY-SA tables
     * cannot exist and an invented set is served so the container boots.
     */
    public record NflFreshness(
            @JsonProperty("data_as_of") LocalDate dataAsOf,
            @JsonProperty("days_behind") int daysBehind,
            boolean stale,
            @JsonProperty("predictable_fixtures") int predictableFixtures,
            int fixtures,
            @JsonProperty("prediction_rule") String predictionRule,
            String source,
            boolean synthetic) {
    }
}
