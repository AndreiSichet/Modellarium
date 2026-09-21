package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonProperty;

/** Wire shape of POST /predict/quarter-half on the Python service. */
public record InferenceQuarterHalfResponse(
        @JsonProperty("data_as_of") LocalDate dataAsOf,
        boolean stale,
        @JsonProperty("days_behind") int daysBehind,
        List<Market> predictions) {
    public record Market(
            String market,
            double value,
            String confidence,
            String interpretation) {
    }

    /** One market by name. */
    public Market market(String name) {
        return predictions.stream()
                .filter(entry -> entry.market().equals(name))
                .findFirst()
                .orElseThrow(() -> new IllegalStateException(
                        "Inference service returned no market named " + name
                                + ". Its response shape has changed."));
    }
}
