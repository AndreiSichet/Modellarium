package com.andreisichet.basketball_predictor.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import com.andreisichet.basketball_predictor.dto.InferenceScheduledGame;
import com.andreisichet.basketball_predictor.dto.ScheduledGameDto;
import com.andreisichet.basketball_predictor.repository.GameRepository;
import com.andreisichet.basketball_predictor.repository.TeamRepository;

/**
 * What ScheduleSyncService stores, ScheduleService reads back intact.
 *
 * THE ONE SHAPE A FIXTURE FILE CANNOT TEST, and the reason this class exists
 * in the service package rather than beside the other wire tests. Every
 * other test of these DTOs feeds them JSON written by hand, which can only
 * prove the DTO parses what its author believed; it cannot notice that the
 * thing at the other end of the column writes something else. Here the bytes
 * come from the writer the sync actually uses and go into the reader the
 * browse endpoint actually uses.
 *
 * IT CAUGHT A REAL NULL. ScheduleSyncService serialises
 * {@code InferenceScheduledGame.Blocker}, whose components carry
 * {@code @JsonProperty("franchise_id")} - and Jackson applies those when
 * WRITING as well as reading, so the stored text said {@code franchise_id}
 * while the reader asked for {@code ScheduledGameDto.Blocker.franchiseId}.
 * Measured on a probe stack: every blocker on every row came back with
 * {@code franchiseId: null}, and nothing showed it, because the frontend
 * resolves the team from {@code side} and only consults the id on a branch
 * that never runs.
 *
 * AND THE FIRST VERSION OF THIS TEST WAS VACUOUS. It built its own
 * ObjectMapper and round-tripped the record against itself, so it passed
 * whatever ScheduleService did - proved by planting the old reader back and
 * watching it stay green. {@code readBlockers} is package-private for
 * exactly this reason: the test has to call the real one.
 */
class ScheduleServiceBlockerTest {

    private static ScheduleService service() {
        return new ScheduleService(
                mock(GameRepository.class),
                mock(TeamRepository.class),
                mock(InferenceClient.class),
                mock(ScheduleSyncStatus.class));
    }

    /** Exactly what ScheduleSyncService.writeBlockers would store. */
    private static String stored(InferenceScheduledGame.Blocker... blockers)
            throws Exception {
        return new tools.jackson.databind.ObjectMapper()
                .writer().writeValueAsString(List.of(blockers));
    }

    @Test
    @DisplayName("a stored blocker reads back with every field intact")
    void everyFieldSurvives() throws Exception {
        String json = stored(
                new InferenceScheduledGame.Blocker(
                        "home", 1614000014L, "previous_game_pending",
                        List.of("2026-11-06")),
                new InferenceScheduledGame.Blocker(
                        "away", 1614000023L, "season_opener", List.of()));

        List<ScheduledGameDto.Blocker> back = service().readBlockers(json);

        assertThat(back).hasSize(2);
        // The field the old reader silently dropped.
        assertThat(back.get(0).franchiseId()).isEqualTo(1614000014L);
        assertThat(back.get(1).franchiseId()).isEqualTo(1614000023L);
        assertThat(back.get(0).side()).isEqualTo("home");
        assertThat(back.get(0).code()).isEqualTo("previous_game_pending");
        assertThat(back.get(0).pending()).containsExactly("2026-11-06");
        assertThat(back.get(1).pending()).isEmpty();
    }

    @Test
    @DisplayName("no field comes back null that was written non-null")
    void nothingIsSilentlyDropped() throws Exception {
        String json = stored(new InferenceScheduledGame.Blocker(
                "away", 1614000028L, "previous_game_pending",
                List.of("2026-11-05", "2026-11-06")));

        ScheduledGameDto.Blocker back = service().readBlockers(json).get(0);

        // Stated as "nothing is null" as well as by field, because the
        // defect was a null and a future rename would reintroduce it in a
        // field this test does not happen to name.
        assertThat(back.side()).isNotNull();
        assertThat(back.franchiseId()).isNotNull();
        assertThat(back.code()).isNotNull();
        assertThat(back.pending()).isNotNull().hasSize(2);
    }

    @Test
    @DisplayName("an unreadable value degrades to no blockers, not a throw")
    void unreadableIsEmpty() {
        // reasonCode travels beside this and still says the fixture is
        // blocked, so losing the detail costs a sentence and not the page.
        assertThat(service().readBlockers("{not json")).isEmpty();
        assertThat(service().readBlockers("")).isEmpty();
        assertThat(service().readBlockers(null)).isEmpty();
    }
}
