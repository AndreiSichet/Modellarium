package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;
import java.util.Map;

/** Dataset freshness as sent to clients. */
public record HealthDto(
        String status,
        Map<String, Integer> modelsLoaded,
        LocalDate dataAsOf,
        int daysBehind,
        boolean stale) {
    public static HealthDto from(InferenceHealth health) {
        return new HealthDto(
                health.status(),
                health.modelsLoaded(),
                health.dataAsOf(),
                health.daysBehind(),
                health.stale());
    }
}
