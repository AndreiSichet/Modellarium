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
    public record Market(
            double value,
            String metric,
            String window,
            String caveat) {
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
