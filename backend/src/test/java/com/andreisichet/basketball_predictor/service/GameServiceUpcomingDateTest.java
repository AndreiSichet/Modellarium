package com.andreisichet.basketball_predictor.service;

import static org.assertj.core.api.Assertions.assertThat;

import java.time.LocalDate;
import java.util.List;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.TestPropertySource;

import com.andreisichet.basketball_predictor.dto.GameSummaryDto;
import com.andreisichet.basketball_predictor.model.Game;
import com.andreisichet.basketball_predictor.model.Team;
import com.andreisichet.basketball_predictor.repository.GameRepository;
import com.andreisichet.basketball_predictor.repository.TeamRepository;

/** "Upcoming" must mean today or later, not merely "not marked played". */
@SpringBootTest
@TestPropertySource(properties = {
        "schedule.sync.cron=0 0 0 29 2 ?",
})
class GameServiceUpcomingDateTest {
    @Autowired
    private GameService gameService;

    @Autowired
    private GameRepository gameRepository;

    @Autowired
    private TeamRepository teamRepository;

    private Game inserted;

    @AfterEach
    void removeInsertedGame() {
        if (inserted != null) {
            gameRepository.delete(inserted);
            inserted = null;
        }
    }

    @Test
    void aPastFixtureThatWasNeverMarkedPlayedIsNotUpcoming() {
        List<Team> teams = teamRepository.findAll();
        assertThat(teams)
                .as("the team table must be seeded for this test to mean anything")
                .hasSizeGreaterThanOrEqualTo(2);

        LocalDate yesterday = LocalDate.now().minusDays(1);

        inserted = new Game();
        inserted.setHomeTeam(teams.get(0));
        inserted.setAwayTeam(teams.get(1));
        inserted.setGameDate(yesterday);
        inserted.setPlayed(false);
        inserted = gameRepository.save(inserted);

        List<GameSummaryDto> upcoming = gameService.getUpcomingGames();

        assertThat(upcoming)
                .as("a fixture dated %s with played=false must not be reported "
                        + "as upcoming", yesterday)
                .noneMatch(game -> game.id().equals(inserted.getId()));

        assertThat(upcoming)
                .as("nothing dated before today may appear at all")
                .allMatch(game -> !game.gameDate().isBefore(LocalDate.now()));
    }

    @Test
    void aFixtureLaterTodayIsStillUpcoming() {
        List<Team> teams = teamRepository.findAll();

        inserted = new Game();
        inserted.setHomeTeam(teams.get(0));
        inserted.setAwayTeam(teams.get(1));
        inserted.setGameDate(LocalDate.now());
        inserted.setPlayed(false);
        inserted = gameRepository.save(inserted);

        List<GameSummaryDto> upcoming = gameService.getUpcomingGames();

        assertThat(upcoming)
                .as("a fixture dated today must still count as upcoming")
                .anyMatch(game -> game.id().equals(inserted.getId()));
    }
}
