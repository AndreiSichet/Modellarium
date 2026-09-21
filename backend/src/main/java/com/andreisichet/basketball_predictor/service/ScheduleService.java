package com.andreisichet.basketball_predictor.service;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.andreisichet.basketball_predictor.dto.HealthDto;
import com.andreisichet.basketball_predictor.dto.ScheduledGameDto;
import com.andreisichet.basketball_predictor.model.Game;
import com.andreisichet.basketball_predictor.model.Team;
import com.andreisichet.basketball_predictor.repository.GameRepository;
import com.andreisichet.basketball_predictor.repository.TeamRepository;

/** Serves upcoming fixtures and dataset freshness. */
@Service
public class ScheduleService {
    private final GameRepository gameRepository;
    private final TeamRepository teamRepository;
    private final InferenceClient inferenceClient;

    public ScheduleService(
            GameRepository gameRepository,
            TeamRepository teamRepository,
            InferenceClient inferenceClient) {
        this.gameRepository = gameRepository;
        this.teamRepository = teamRepository;
        this.inferenceClient = inferenceClient;
    }

    /** Cached fixtures inside the requested window, with team names attached. */
    @Transactional(readOnly = true)
    public List<ScheduledGameDto> getSchedule(int daysAhead) {
        LocalDate today = LocalDate.now();
        List<Game> games = gameRepository
                .findByPlayedFalseAndGameDateBetweenOrderByGameDateAsc(
                        today, today.plusDays(daysAhead));

        if (games.isEmpty()) {
            return List.of();
        }

        Map<Long, Team> teamsById = teamRepository.findAll().stream()
                .collect(Collectors.toMap(Team::getId, Function.identity()));

        return games.stream()
                .map(game -> toDto(game, teamsById))
                .toList();
    }

    /** Live on every call, never cached. */
    public HealthDto getHealth() {
        return HealthDto.from(inferenceClient.getHealth());
    }

    private ScheduledGameDto toDto(Game game, Map<Long, Team> teamsById) {
        Team home = teamsById.get(game.getHomeTeam().getId());
        Team away = teamsById.get(game.getAwayTeam().getId());

        return new ScheduledGameDto(
                home.getId(),
                home.getAbbreviation(),
                home.getName(),
                away.getId(),
                away.getAbbreviation(),
                away.getName(),
                game.getGameDate());
    }
}
