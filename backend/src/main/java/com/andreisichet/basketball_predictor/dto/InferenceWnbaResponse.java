package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonProperty;

/** Wire shape of POST /predict/wnba on the Python service. */
public record InferenceWnbaResponse(
        @JsonProperty("data_as_of") LocalDate dataAsOf,
        boolean stale,
        @JsonProperty("days_behind") int daysBehind,
        int season,
        Map<String, Market> markets) {
    /**
     * The Python side no longer sends a `caveat`, and this record no longer
     * declares one. Unknown properties are ignored, so an older inference
     * service that still sends it is tolerated rather than rejected - which
     * is the right way round, and the same reasoning the wire-shape tests
     * record: an added field is backwards compatible, a removed or retyped
     * one is a break.
     */
    public record Market(
            double value,
            String metric,
            String window) {
    }

    /**
     * One market by name, or a loud failure.
     *
     * A Map rather than the List the quarter/half response uses, because the
     * Python side keys these by target name and the WNBA's target set is a
     * selection output rather than a fixed market list. The same
     * named-lookup-or-throw shape is kept: a missing key means the response
     * changed, and that must not read as a zero.
     */
    public Market market(String name) {
        Market found = markets == null ? null : markets.get(name);
        if (found == null) {
            throw new IllegalStateException(
                    "Inference service returned no WNBA market named " + name
                            + ". Its response shape has changed.");
        }
        return found;
    }
}
