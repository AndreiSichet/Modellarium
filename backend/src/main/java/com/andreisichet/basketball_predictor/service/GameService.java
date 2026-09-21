package com.andreisichet.basketball_predictor.service;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.andreisichet.basketball_predictor.dto.GameSummaryDto;
import com.andreisichet.basketball_predictor.dto.PredictionDto;
import com.andreisichet.basketball_predictor.model.Game;
import com.andreisichet.basketball_predictor.model.Prediction;
import com.andreisichet.basketball_predictor.repository.GameRepository;
import com.andreisichet.basketball_predictor.repository.PredictionRepository;

@Service
public class GameService {
    private final GameRepository gameRepository;
    private final PredictionRepository predictionRepository;

    public GameService(GameRepository gameRepository, PredictionRepository predictionRepository) {
        this.gameRepository = gameRepository;
        this.predictionRepository = predictionRepository;
    }

    /** Upcoming games, soonest first, each with its newest prediction. */
    @Transactional(readOnly = true)
    public List<GameSummaryDto> getUpcomingGames() {
        List<Game> games = gameRepository
                .findByPlayedFalseAndGameDateGreaterThanEqualOrderByGameDateAsc(
                        LocalDate.now());

        if (games.isEmpty()) {
            return List.of();
        }

        Map<Long, Prediction> latestByGameId = latestPredictions(games);

        return games.stream()
                .map(game -> toSummary(game, latestByGameId.get(game.getId())))
                .toList();
    }

    /** The newest prediction for each of these games, in one query. */
    private Map<Long, Prediction> latestPredictions(List<Game> games) {
        List<Long> gameIds = games.stream().map(Game::getId).toList();

        return predictionRepository.findByGameIdInOrderByPredictedAtDesc(gameIds).stream()
                .collect(Collectors.toMap(
                        prediction -> prediction.getGame().getId(),
                        Function.identity(),
                        (newest, older) -> newest));
    }

    private GameSummaryDto toSummary(Game game, Prediction latest) {
        return new GameSummaryDto(
                game.getId(),
                game.getHomeTeam().getAbbreviation(),
                game.getAwayTeam().getAbbreviation(),
                game.getGameDate(),
                game.isPlayed(),
                latest == null ? null : PredictionDto.from(latest));
    }
}
