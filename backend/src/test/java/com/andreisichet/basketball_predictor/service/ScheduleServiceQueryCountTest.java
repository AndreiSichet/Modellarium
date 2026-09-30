package com.andreisichet.basketball_predictor.service;

import static org.assertj.core.api.Assertions.assertThat;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;

import org.hibernate.SessionFactory;
import org.hibernate.stat.Statistics;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.TestPropertySource;

import com.andreisichet.basketball_predictor.dto.ScheduledGameDto;
import com.andreisichet.basketball_predictor.model.Game;
import com.andreisichet.basketball_predictor.model.Team;
import com.andreisichet.basketball_predictor.repository.GameRepository;
import com.andreisichet.basketball_predictor.repository.TeamRepository;

import jakarta.persistence.EntityManagerFactory;

/**
 * Pins GET /api/games/schedule to a query count that does not grow with the
 * number of fixtures.
 *
 * Replaces GameServiceQueryCountTest, which was deleted along with
 * /api/games/upcoming. The risk did not go with it - it moved. That endpoint
 * went from 2 queries to a measured 486 without its code changing, because
 * caching fixtures moved the assumption underneath it; batching took it to 32
 * and an @EntityGraph on the team associations to 2.
 *
 * ScheduleService avoids the same Team-proxy N+1 a different way, with one
 * TeamRepository.findAll() into a map, relying on getHomeTeam().getId() not
 * initialising a lazy proxy. Nothing else pins that. Fetching teams per game
 * would reintroduce ~690 queries with no error and a page that still renders.
 *
 * Measured on this endpoint: 2 queries for 729 fixtures, and 2 for 7,290.
 * Ten times the fixtures, the same two queries.
 */
@SpringBootTest
@TestPropertySource(properties = {
        "spring.jpa.properties.hibernate.generate_statistics=true",

        // 29 February: never fires. Keeps the sync job from issuing queries
        // in the middle of a measurement.
        "schedule.sync.cron=0 0 0 29 2 ?",
})
class ScheduleServiceQueryCountTest {
    /**
     * A ceiling, not an exact count. What must hold is that the count does not
     * scale with fixture count; pinning the measured 2 would turn an unrelated
     * Hibernate batching change into a failure, and a test that fires for the
     * wrong reason gets relaxed rather than investigated.
     */
    private static final int QUERY_CEILING = 6;

    private static final int DAYS_AHEAD = 120;
    private static final int SMALL_BATCH = 40;

    /**
     * The large batch is PROPORTIONAL, not a constant. The table already
     * holds the synced fixtures, which dominate any fixed batch: 40 against
     * an ambient ~690 is a 1.06x increase, which would prove almost nothing
     * about scaling. Seeding 9x whatever the first measurement saw gives an
     * order of magnitude whatever the environment happens to hold.
     */
    private static final int SCALE_FACTOR = 9;

    @Autowired
    private ScheduleService scheduleService;

    @Autowired
    private GameRepository gameRepository;

    @Autowired
    private TeamRepository teamRepository;

    @Autowired
    private EntityManagerFactory entityManagerFactory;

    private final List<Long> seeded = new ArrayList<>();

    @AfterEach
    void removeSeededFixtures() {
        gameRepository.deleteAllById(seeded);
        seeded.clear();
    }

    @Test
    void scheduleDoesNotScaleQueriesWithFixtureCount() {
        // Deliberately NOT @Transactional. A shared persistence context would
        // serve teams from its first-level cache and hide exactly the per-game
        // lookup this test exists to catch.
        int smallFixtures = seedFixtures(SMALL_BATCH);
        long smallQueries = queriesForOneSchedule();

        int largeFixtures = seedFixtures(smallFixtures * SCALE_FACTOR);
        long largeQueries = queriesForOneSchedule();

        assertThat(smallQueries)
                .as("%d fixtures cost %d queries", smallFixtures, smallQueries)
                .isLessThanOrEqualTo(QUERY_CEILING);

        assertThat(largeFixtures)
                .as("the second measurement must be an order of magnitude larger, "
                        + "or it says nothing about scaling")
                .isGreaterThanOrEqualTo(smallFixtures * SCALE_FACTOR);

        assertThat(largeQueries)
                .as("%d fixtures cost %d queries - a %dx increase in fixtures "
                        + "must not increase the query count",
                        largeFixtures, largeQueries, largeFixtures / smallFixtures)
                .isLessThanOrEqualTo(QUERY_CEILING)
                .isEqualTo(smallQueries);
    }

    /** One call to the endpoint's service method, counting the SQL it issues. */
    private long queriesForOneSchedule() {
        Statistics statistics = entityManagerFactory
                .unwrap(SessionFactory.class)
                .getStatistics();
        statistics.clear();

        List<ScheduledGameDto> schedule = scheduleService.getSchedule(DAYS_AHEAD);

        // The team join is the point: a DTO carries both teams' names, so a
        // count that ignored them would pass while the lookups regressed.
        assertThat(schedule).isNotEmpty();
        assertThat(schedule.getFirst().homeTeamName()).isNotBlank();
        assertThat(schedule.getFirst().awayTeamName()).isNotBlank();

        return statistics.getPrepareStatementCount();
    }

    /** Add fixtures inside the requested window; returns the total now visible. */
    private int seedFixtures(int count) {
        List<Team> teams = teamRepository.findAll();
        LocalDate start = LocalDate.now().plusDays(1);

        List<Game> batch = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            Game game = new Game();
            game.setHomeTeam(teams.get(i % teams.size()));
            game.setAwayTeam(teams.get((i + 1) % teams.size()));
            game.setGameDate(start.plusDays(i % (DAYS_AHEAD - 1)));
            game.setPlayed(false);
            batch.add(game);
        }
        gameRepository.saveAll(batch).forEach(game -> seeded.add(game.getId()));

        return scheduleService.getSchedule(DAYS_AHEAD).size();
    }
}
