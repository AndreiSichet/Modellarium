package com.andreisichet.basketball_predictor.service;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.IntFunction;
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

/** Caches upcoming fixtures into the games table, one league at a time. */
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

    /**
     * The NBA schedule. A SEPARATE TRANSACTION from the WNBA's.
     *
     * The two leagues are synced by two public methods rather than one, and
     * that is the whole point: @Transactional is proxy-based, so one method
     * calling the other on `this` would share a single transaction and a
     * WNBA failure would roll back the NBA fixtures written before it. The
     * job calls both through the proxy, so each gets its own transaction and
     * its own fetch failure handling.
     */
    @Transactional
    public void syncNba() {
        syncLeague("NBA", inferenceClient::fetchSchedule);
    }

    /**
     * The WNBA schedule, independently.
     *
     * Expected to cache NOTHING for most of the year, and that is correct
     * rather than broken: the WNBA regular season runs May to September, and
     * the type-digit filter on the Python side keeps playoff fixtures out
     * because no model here has seen one. Measured in October 2026 - all 17
     * unplayed WNBA games were playoffs, so this syncs 0 fixtures.
     */
    @Transactional
    public void syncWnba() {
        syncLeague("WNBA", inferenceClient::fetchWnbaSchedule);
    }

    private void syncLeague(
            String league, IntFunction<List<InferenceScheduledGame>> fetch) {
        List<InferenceScheduledGame> fixtures;

        try {
            fixtures = fetch.apply(daysAhead);
        } catch (Exception error) {
            log.warn("{} schedule sync skipped - could not fetch fixtures: {}",
                    league, error.getMessage());
            return;
        }

        if (fixtures.isEmpty()) {
            log.info("{} schedule sync: 0 fixtures returned for the next {} days, "
                    + "nothing to cache.", league, daysAhead);
            return;
        }

        // Scoped to the league being synced. The two id ranges are disjoint,
        // so a fixture from the wrong league would simply fail to resolve and
        // be counted as skipped rather than quietly written against the other
        // league's team rows.
        Map<Long, Team> teamsById = teamRepository.findByLeagueOrderByNameAsc(league)
                .stream()
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
        log.info("{} schedule sync: {} fixtures fetched ({} days ahead), {} new, "
                + "{} already present{}.",
                league,
                fixtures.size(),
                daysAhead,
                created,
                fixtures.size() - skipped - created,
                skipped > 0 ? ", " + skipped + " skipped (unknown team)" : "");
    }
}
