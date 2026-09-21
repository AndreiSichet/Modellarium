package com.andreisichet.basketball_predictor.dto;

import java.time.Instant;
import java.time.LocalDate;
import java.util.List;

import com.andreisichet.basketball_predictor.model.PlayerPropPrediction;

/** Both sides' prop boards for one fixture, plus the usual freshness block. */
public record PlayerPropsResponseDto(
        Long gameId,
        LocalDate gameDate,
        TeamBoard homeTeam,
        TeamBoard awayTeam,
        LocalDate dataAsOf,
        boolean stale,
        int daysBehind,
        Instant predictedAt) {
    /** One side's board. */
    public record TeamBoard(
            Long teamId,
            String teamAbbreviation,
            boolean availabilityKnown,
            String availabilityNote,
            List<PlayerLine> players) {
    }

    /** One player's five predictions. */
    public record PlayerLine(
            Long playerId,
            String playerName,
            double predictedPoints,
            double predictedRebounds,
            double predictedAssists,
            double predictedThreesMade,
            double predictedPra,
            String modelUsed) {
        public static PlayerLine from(PlayerPropPrediction saved) {
            return new PlayerLine(
                    saved.getPlayer().getId(),
                    saved.getPlayer().getName(),
                    saved.getPredictedPoints(),
                    saved.getPredictedRebounds(),
                    saved.getPredictedAssists(),
                    saved.getPredictedThreesMade(),
                    saved.getPredictedPra(),
                    saved.getModelUsed());
        }
    }
}
