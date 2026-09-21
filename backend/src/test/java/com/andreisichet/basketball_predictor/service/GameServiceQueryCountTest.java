package com.andreisichet.basketball_predictor.service;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.hibernate.SessionFactory;
import org.hibernate.stat.Statistics;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.TestPropertySource;

import com.andreisichet.basketball_predictor.dto.GameSummaryDto;

import jakarta.persistence.EntityManagerFactory;

/** Pins GET /api/games/upcoming to a constant number of queries. */
@SpringBootTest
@TestPropertySource(properties = {
        "spring.jpa.properties.hibernate.generate_statistics=true",

        "schedule.sync.cron=0 0 0 29 2 ?",
})
class GameServiceQueryCountTest {
    private static final int QUERY_CEILING = 6;

    @Autowired
    private GameService gameService;

    @Autowired
    private EntityManagerFactory entityManagerFactory;

    @Test
    void upcomingGamesDoesNotScaleQueriesWithGameCount() {
        Statistics statistics = entityManagerFactory
                .unwrap(SessionFactory.class)
                .getStatistics();
        statistics.clear();

        List<GameSummaryDto> games = gameService.getUpcomingGames();

        long queries = statistics.getPrepareStatementCount();

        assertThat(queries)
                .as("one call returning %d games issued %d queries - it must not "
                        + "scale with the number of games", games.size(), queries)
                .isLessThanOrEqualTo(QUERY_CEILING);
    }
}
