package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * The inference service's /predict/gleague body.
 *
 * {@code season} is a STRING here and an int in the WNBA's record, and that is
 * not an inconsistency to harmonise: a G League season spans two calendar
 * years and is labelled "2025-26", while a WNBA season sits inside one and is
 * labelled 2025. Declaring this int would fail deserialisation outright.
 */
public record InferenceGleagueResponse(
        @JsonProperty("data_as_of") LocalDate dataAsOf,
        boolean stale,
        @JsonProperty("days_behind") int daysBehind,
        String season,
        Map<String, Market> markets) {

    public record Market(
            double value,
            String metric,
            String window) {
    }

    public Market market(String name) {
        Market found = markets == null ? null : markets.get(name);
        if (found == null) {
            throw new IllegalStateException(
                    "Inference service returned no G League market named "
                            + name + ". Its response shape has changed.");
        }
        return found;
    }
}
