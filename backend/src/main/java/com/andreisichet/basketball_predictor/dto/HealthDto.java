package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;
import java.util.Map;

/** Dataset freshness as sent to clients. */
public record HealthDto(
        String status,
        Map<String, Integer> modelsLoaded,
        LocalDate dataAsOf,
        int daysBehind,
        boolean stale,
        LeagueFreshnessDto wnba,
        LeagueFreshnessDto gleague) {
    /** Per-league freshness in the camelCase shape clients read. */
    public record LeagueFreshnessDto(
            LocalDate dataAsOf,
            int daysBehind,
            boolean stale) {
        static LeagueFreshnessDto from(InferenceHealth.LeagueFreshness source) {
            return source == null ? null : new LeagueFreshnessDto(
                    source.dataAsOf(), source.daysBehind(), source.stale());
        }
    }

    /**
     * The top-level dataAsOf stays the NBA's.
     *
     * This endpoint took the browse view down once by retyping modelsLoaded
     * (CLAUDE.md section 21), and the lesson was to add rather than change.
     * So the WNBA's cutoff arrives as a new nested field and every existing
     * client keeps reading exactly what it read before.
     */
    public static HealthDto from(InferenceHealth health) {
        return new HealthDto(
                health.status(),
                health.modelsLoaded(),
                health.dataAsOf(),
                health.daysBehind(),
                health.stale(),
                LeagueFreshnessDto.from(health.wnba()),
                LeagueFreshnessDto.from(health.gleague()));
    }
}
