package com.andreisichet.basketball_predictor.service;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.andreisichet.basketball_predictor.dto.InferenceScheduledGame;
import com.andreisichet.basketball_predictor.model.Team;
import com.andreisichet.basketball_predictor.repository.GameRepository;
import com.andreisichet.basketball_predictor.repository.TeamRepository;

/** Caches upcoming NBA fixtures into the games table. */
@Service
public class ScheduleSyncService {
    private static final Logger log = LoggerFactory.getLogger(ScheduleSyncService.class);

    private final InferenceClient inferenceClient;
    private final GameLookup gameLookup;
    private final GameRepository gameRepository;
    private final TeamRepository teamRepository;
    private final int daysAhead;

    public ScheduleSyncService(
            InferenceClient inferenceClient,
            GameLookup gameLookup,
            GameRepository gameRepository,
            TeamRepository teamRepository,
            @Value("${schedule.sync.days-ahead:120}") int daysAhead) {
        this.inferenceClient = inferenceClient;
        this.gameLookup = gameLookup;
        this.gameRepository = gameRepository;
        this.teamRepository = teamRepository;
        this.daysAhead = daysAhead;
    }

    /** Fetch the upcoming schedule and make sure every fixture has a row. */
    @Transactional
    public void sync() {
        List<InferenceScheduledGame> fixtures;

        try {
            fixtures = inferenceClient.fetchSchedule(daysAhead);
        } catch (Exception error) {
            log.warn("Schedule sync skipped - could not fetch fixtures: {}", error.getMessage());
            return;
        }

        if (fixtures.isEmpty()) {
            log.info("Schedule sync: 0 fixtures returned for the next {} days, nothing to cache.",
                    daysAhead);
            return;
        }

        Map<Long, Team> teamsById = teamRepository.findAll().stream()
                .collect(Collectors.toMap(Team::getId, Function.identity()));

        long before = gameRepository.count();
        int skipped = 0;

        for (InferenceScheduledGame fixture : fixtures) {
            Team home = teamsById.get(fixture.homeTeamId());
            Team away = teamsById.get(fixture.awayTeamId());

            if (home == null || away == null) {
                skipped++;
                continue;
            }

            gameLookup.findOrCreateGame(home, away, fixture.gameDate());
        }

        long created = gameRepository.count() - before;
        log.info("Schedule sync: {} fixtures fetched ({} days ahead), {} new, {} already present{}.",
                fixtures.size(),
                daysAhead,
                created,
                fixtures.size() - skipped - created,
                skipped > 0 ? ", " + skipped + " skipped (unknown team)" : "");
    }
}
