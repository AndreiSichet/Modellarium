package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * The inference service's /predict/nhl body.
 *
 * <p>A MARKET CAN SAY IT IS NOT SERVED, which no other league's market needs.
 * Totals are declared unserved for the NHL with the code
 * {@code no_skill_over_constant}, so {@code value} is null there and
 * {@code served} is false. A client that cannot tell "we do not offer this"
 * from "this response forgot it" will eventually show one as the other, so
 * the false case is carried explicitly rather than by absence.
 *
 * <p>{@code caveat} is the winner's, from the manifest. Its test interval
 * against a constant INCLUDES ZERO, so the figure travels with the sentence
 * saying the gain is not established on test. That is the opposite direction
 * from the WNBA's withdrawn moneyline caveat: there the interval spanning
 * zero was the reason to drop a claim, here it is what the caveat is about.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record InferenceNhlResponse(
        @JsonProperty("data_as_of") LocalDate dataAsOf,
        boolean stale,
        @JsonProperty("days_behind") int daysBehind,
        String season,
        String home,
        String away,
        Map<String, Market> markets,
        @JsonProperty("prediction_rule") String predictionRule,
        String source) {

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Market(
            boolean served,
            Double value,
            String family,
            String definition,
            String caveat,
            @JsonProperty("reason_code") String reasonCode,
            String detail) {
    }
}
