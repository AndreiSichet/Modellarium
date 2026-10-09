package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * The inference service's /predict/nfl body.
 *
 * {@code season} is an int here, as the WNBA's is and unlike the G League's
 * String: an NFL season sits inside one calendar year and is labelled by the
 * year it starts in, so 2026 is the whole label. Declaring it String would
 * work and would then disagree with what the Python side sends.
 *
 * {@code week} has no counterpart in the other three records. It is a feature
 * the NFL models read and the unit an NFL result is discussed in, so it comes
 * over the wire rather than being derived from a date here.
 */
public record InferenceNflResponse(
        @JsonProperty("data_as_of") LocalDate dataAsOf,
        boolean stale,
        @JsonProperty("days_behind") int daysBehind,
        int season,
        int week,
        Map<String, Market> markets,
        @JsonProperty("home_win_interpretation") String homeWinInterpretation,
        String source) {

    /**
     * One market's value and provenance.
     *
     * {@code note} carries the manifest's per-market verdict - "TIES Elo
     * alone" for winner and margin. Declared rather than dropped, and the
     * reasoning is worth stating because the WNBA went the other way: it
     * shipped a moneyline caveat over the wire, measured the difference, found
     * the interval spanned zero and withdrew the field from three layers. This
     * is not that. A tie against Elo alone is a measured, reported result in
     * the test receipt, and it describes what this number IS - a model whose
     * discrimination is not distinguishable from a one-feature formula's.
     *
     * {@code imputed} names the features the artifact's own imputer supplied
     * for this row. A property of this response rather than of the project:
     * a season opener has no REST_DAYS because an offseason is not rest, and a
     * client should be able to see the row was completed rather than observed.
     */
    public record Market(
            double value,
            String metric,
            @JsonProperty("model_used") String modelUsed,
            String note,
            List<String> imputed) {
    }

    public Market market(String name) {
        Market found = markets == null ? null : markets.get(name);
        if (found == null) {
            throw new IllegalStateException(
                    "Inference service returned no NFL market named " + name
                            + ". Its response shape has changed.");
        }
        return found;
    }
}
