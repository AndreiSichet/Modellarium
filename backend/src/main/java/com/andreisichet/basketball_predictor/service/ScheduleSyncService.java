package com.andreisichet.basketball_predictor.service;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.andreisichet.basketball_predictor.dto.InferenceScheduledGame;
import com.andreisichet.basketball_predictor.model.Game;
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

    /**
     * The G League schedule, independently.
     *
     * THE FIRST NEW LEAGUE WITH FIXTURES TO CACHE RIGHT NOW. The NBA's and
     * the WNBA's were both out of season when they were added; the G League's
     * 2026-27 season is already scheduled, so this populates on the first
     * tick rather than waiting months - which means a mistake here reaches a
     * live browse view in weeks rather than next May.
     *
     * Showcase Cup fixtures must not arrive, and the filter is on the Python
     * side: Cup games carry type digit 5 and `upcoming_regular_season` keeps
     * only digit 2. A cached Cup fixture would show as predictable and the
     * backend would then reject it, because no shipped model has seen one.
     */
    @Transactional
    public void syncGleague() {
        syncLeague("GLEAGUE", inferenceClient::fetchGleagueSchedule);
    }

    /**
     * The NFL schedule, independently.
     *
     * IGNORES daysAhead, AND IT IS THE FIRST LEAGUE THAT DOES. The other
     * three proxy a live nba_api call that genuinely takes a horizon. The
     * NFL's fixtures arrive in the served snapshot, so the list is whatever
     * remains of the season; the horizon still applies downstream, where
     * ScheduleService filters the cached rows by date exactly as it does for
     * the others. The lambda therefore discards its argument rather than
     * passing one the endpoint would ignore.
     *
     * CACHES EVERY REMAINING FIXTURE, NOT ONLY THE PREDICTABLE ONES. 193 of
     * 208 are not predictable yet under the dependency rule, and they become
     * so a week at a time - so caching only the predictable 15 would mean the
     * browse view could show no fixture until the week it is played. Which
     * ones can be predicted is a serving question, answered by
     * /health's nfl.predictable_fixtures and by the per-fixture flag on
     * /schedule/nfl, not by what is in the game table.
     */
    @Transactional
    public void syncNfl() {
        syncLeague("NFL", ignoredHorizon -> inferenceClient.fetchNflSchedule());
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
            log.info("{} schedule sync: 0 fixtures returned{}, "
                    + "nothing to cache.", league, horizon(league));
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
        int updated = 0;

        for (InferenceScheduledGame fixture : fixtures) {
            Team home = teamsById.get(fixture.homeTeamId());
            Team away = teamsById.get(fixture.awayTeamId());

            if (home == null || away == null) {
                skipped++;
                continue;
            }

            Game game = gameLookup.findOrCreateGame(home, away,
                    fixture.gameDate());
            if (applyFixtureFields(game, fixture)) {
                gameRepository.save(game);
                updated++;
            }
        }

        long created = gameRepository.count() - before;
        log.info("{} schedule sync: {} fixtures fetched{}, {} new, "
                + "{} already present{}.",
                league,
                fixtures.size(),
                horizon(league),
                created,
                fixtures.size() - skipped - created,
                skipped > 0 ? ", " + skipped + " skipped (unknown team)" : "");
        if (updated > 0) {
            log.info("{} schedule sync: {} fixture(s) had their "
                    + "predictable/kickoff/flex/week updated.", league, updated);
        }
    }

    /**
     * Copy the NFL's per-fixture fields onto the row; true if anything moved.
     *
     * ONLY ON CHANGE, BECAUSE THIS RUNS EVERY SIX HOURS. Writing all four
     * columns unconditionally would mean ~1,200 pointless UPDATEs per cycle
     * for rows that had not moved. In the steady state this writes nothing;
     * it writes when a fixture becomes predictable, when a flexed kickoff is
     * finally set, or on the first sync after the columns were added.
     *
     * The three basketball leagues send none of these, so every field stays
     * null and this returns false for them - which is why the sync does not
     * need to know which league it is looking at.
     */
    private boolean applyFixtureFields(Game game,
                                       InferenceScheduledGame fixture) {
        boolean changed = false;
        if (!Objects.equals(game.getPredictable(), fixture.predictable())) {
            game.setPredictable(fixture.predictable());
            changed = true;
        }
        if (!Objects.equals(game.getKickoffUtc(), fixture.kickoffUtc())) {
            game.setKickoffUtc(fixture.kickoffUtc());
            changed = true;
        }
        if (!Objects.equals(game.getFlex(), fixture.flex())) {
            game.setFlex(fixture.flex());
            changed = true;
        }
        if (!Objects.equals(game.getWeek(), fixture.week())) {
            game.setWeek(fixture.week());
            changed = true;
        }
        return changed;
    }

    /**
     * The horizon clause, or nothing for a league that has none.
     *
     * THE NFL'S FETCH IGNORES daysAhead, so printing it was a report claiming
     * something untrue: the first rehearsal logged "208 fixtures fetched (120
     * days ahead)" when the fetch had asked for no horizon at all and the
     * number happened to be every remaining fixture of the season. A log line
     * that states a parameter the call did not use is the kind of report
     * people later reason from.
     */
    private String horizon(String league) {
        return "NFL".equals(league) ? " from the served snapshot"
                : " (" + daysAhead + " days ahead)";
    }
}
