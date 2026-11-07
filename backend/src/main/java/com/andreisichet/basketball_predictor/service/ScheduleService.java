package com.andreisichet.basketball_predictor.service;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import tools.jackson.core.JacksonException;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.ObjectReader;

import lombok.extern.slf4j.Slf4j;

import com.andreisichet.basketball_predictor.dto.HealthDto;
import com.andreisichet.basketball_predictor.dto.InferenceScheduledGame;
import com.andreisichet.basketball_predictor.dto.ScheduledGameDto;
import com.andreisichet.basketball_predictor.model.Game;
import com.andreisichet.basketball_predictor.model.Team;
import com.andreisichet.basketball_predictor.repository.GameRepository;
import com.andreisichet.basketball_predictor.repository.TeamRepository;

/** Serves upcoming fixtures and dataset freshness. */
@Service
@Slf4j
public class ScheduleService {

    /**
     * Reader for the `blocked_by` column, built once.
     *
     * A TYPED READER RATHER THAN A SHARED ObjectMapper CALL, so the generic
     * type is fixed here instead of at each call site, where an erased
     * `List` would let anything through.
     */
    /**
     * READ BACK AS THE RECORD THAT WAS WRITTEN, then mapped.
     *
     * This read {@code ScheduledGameDto.Blocker} while ScheduleSyncService
     * writes {@code InferenceScheduledGame.Blocker}, and the two spell one
     * field differently: the inbound record carries
     * {@code @JsonProperty("franchise_id")}, which Jackson applies when
     * SERIALISING too, so the stored text says {@code franchise_id} and the
     * outbound record's {@code franchiseId} never matched it. Measured on a
     * probe stack: every blocker came back with {@code franchiseId: null}.
     *
     * It was invisible because the frontend resolves the team from
     * {@code side} and only falls back to the id when the side is neither
     * home nor away - which never happens - and because the wire test fed
     * the DTO a fixture written by hand rather than bytes this service had
     * stored. One definition on both sides is the fix; two records agreeing
     * by convention is how they came apart.
     */
    private static final ObjectReader BLOCKERS = new ObjectMapper()
            .readerForListOf(InferenceScheduledGame.Blocker.class);

    private final GameRepository gameRepository;
    private final TeamRepository teamRepository;
    private final InferenceClient inferenceClient;
    private final ScheduleSyncStatus scheduleSyncStatus;

    public ScheduleService(
            GameRepository gameRepository,
            TeamRepository teamRepository,
            InferenceClient inferenceClient,
            ScheduleSyncStatus scheduleSyncStatus) {
        this.gameRepository = gameRepository;
        this.teamRepository = teamRepository;
        this.inferenceClient = inferenceClient;
        this.scheduleSyncStatus = scheduleSyncStatus;
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

    /**
     * Live on every call, never cached.
     *
     * THE INFERENCE CALL COMES FIRST, AND IT STILL THROWS. A 503 here when
     * inference is unreachable is the existing contract and is not softened:
     * the schedule-sync block is an extra field on the success path, not an
     * excuse to answer 200 while the thing this endpoint reports on is down.
     * The consequence - that the block cannot be read during the retry
     * window, which is when it is most interesting - is recorded in
     * CLAUDE.md rather than fixed by making a red look green.
     */
    public HealthDto getHealth() {
        return HealthDto.from(inferenceClient.getHealth(),
                scheduleSyncStatus.snapshot());
    }

    private ScheduledGameDto toDto(Game game, Map<Long, Team> teamsById) {
        Team home = teamsById.get(game.getHomeTeam().getId());
        Team away = teamsById.get(game.getAwayTeam().getId());

        // Both sides are in the same league by construction - the sync
        // resolves a fixture's teams from one league's rows - so the home
        // team's league is the fixture's.
        return new ScheduledGameDto(
                home.getLeague() == null
                        ? null : home.getLeague().toLowerCase(),
                game.getId(),
                home.getId(),
                home.getAbbreviation(),
                home.getName(),
                away.getId(),
                away.getAbbreviation(),
                away.getName(),
                game.getGameDate(),
                game.getPredictable(),
                game.getKickoffUtc(),
                game.getFlex(),
                game.getWeek(),
                game.getReasonCode(),
                readBlockers(game.getBlockedBy()));
    }

    /**
     * The stored JSON blocker list, or an empty list.
     *
     * NEVER NULL AND NEVER A THROW. This is a browse path: a fixture whose
     * stored detail is unreadable must still appear in the schedule, without
     * its explanation, rather than taking the list down. The column is
     * written by this application's own sync from a validated shape, so a
     * parse failure means something changed underneath - it is logged once
     * per request at most and the row degrades to the code alone, which the
     * client already has a generic sentence for.
     *
     * An unparsable value is NOT silently treated as "nothing blocks this":
     * `reasonCode` travels beside it and still says the fixture is blocked.
     */
    // PACKAGE-PRIVATE SO A TEST CAN CALL THE REAL ONE. The first
    // version of that test built its own ObjectMapper and round-
    // tripped the record directly, which passes whatever this
    // method does - proved by planting the old reader and watching
    // the test stay green. A check that cannot fail is not a check.
    List<ScheduledGameDto.Blocker> readBlockers(String json) {
        if (json == null || json.isBlank()) {
            return List.of();
        }
        try {
            List<InferenceScheduledGame.Blocker> stored =
                    BLOCKERS.readValue(json);
            return stored.stream()
                    .map(blocker -> new ScheduledGameDto.Blocker(
                            blocker.side(), blocker.franchiseId(),
                            blocker.code(), blocker.pending()))
                    .toList();
        } catch (JacksonException error) {
            // JACKSON 3'S EXCEPTIONS ARE UNCHECKED, so this catch is a
            // deliberate choice rather than a compiler requirement - and
            // the choice is that a browse page must not 500 over an
            // explanatory field.
            log.warn("Unreadable blocked_by JSON, falling back to the code "
                    + "alone: {}", error.getMessage());
            return List.of();
        }
    }
}
