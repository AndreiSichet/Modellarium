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
        LeagueFreshnessDto gleague,
        NflFreshnessDto nfl) {
    /**
     * The NFL's freshness, carrying the two fields its rule needs.
     *
     * {@code predictableFixtures} and {@code predictionRule} reach the client
     * because the frontend cannot work out what is predictable for this
     * league the way it does for the other three. There, predictability is
     * cutoff plus one day and the browser computes it. Here it is a
     * dependency on games being recorded, so the answer has to travel.
     *
     * {@code synthetic} is NOT carried, and that is a decision rather than a
     * third silent drop. /health reports it; this record does not. It is an
     * operational fact about which table set the inference service booted
     * against - true only in CI, where the real CC BY-SA tables cannot exist -
     * and nothing in the browser needs it. The same shape was a real defect
     * twice, when these records dropped the WNBA's freshness block and then
     * the G League's, so it is written down here rather than left to be
     * rediscovered: the regression gate still names added fields per body, so
     * if this ever becomes needed the asymmetry is visible.
     */
    public record NflFreshnessDto(
            LocalDate dataAsOf,
            int daysBehind,
            boolean stale,
            int predictableFixtures,
            int fixtures,
            String predictionRule,
            String source) {
        static NflFreshnessDto from(InferenceHealth.NflFreshness source) {
            return source == null ? null : new NflFreshnessDto(
                    source.dataAsOf(), source.daysBehind(), source.stale(),
                    source.predictableFixtures(), source.fixtures(),
                    source.predictionRule(), source.source());
        }
    }

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
                LeagueFreshnessDto.from(health.gleague()),
                NflFreshnessDto.from(health.nfl()));
    }
}
