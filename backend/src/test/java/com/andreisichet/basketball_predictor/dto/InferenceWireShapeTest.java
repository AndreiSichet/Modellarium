package com.andreisichet.basketball_predictor.dto;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.json.JsonTest;

import tools.jackson.core.type.TypeReference;
import tools.jackson.databind.ObjectMapper;

/** Pins the wire shape of every inference-service response the backend deserialises. */
@JsonTest
class InferenceWireShapeTest {
    @Autowired
    private ObjectMapper mapper;

    /** All fixtures come from this matchup, the one used throughout the project. */
    private static final LocalDate DATA_AS_OF = LocalDate.of(2026, 4, 12);
    private static final int DAYS_BEHIND = 146;

    @Nested
    class Health {
        @Test
        void deserialisesThePerFamilyModelBreakdown() {
            InferenceHealth health =
                    mapper.readValue(Fixture.read("health.json"), InferenceHealth.class);

            assertThat(health.status()).isEqualTo("ok");
            assertThat(health.dataAsOf()).isEqualTo(DATA_AS_OF);
            assertThat(health.daysBehind()).isEqualTo(DAYS_BEHIND);
            assertThat(health.stale()).isTrue();

            assertThat(health.modelsLoaded()).containsExactlyInAnyOrderEntriesOf(
                    Map.of("team", 7, "quarter_half", 6, "player_props", 10));
        }

        @Test
        void carriesTheBreakdownThroughToTheClientDto() {
            InferenceHealth health =
                    mapper.readValue(Fixture.read("health.json"), InferenceHealth.class);

            HealthDto dto = HealthDto.from(health);

            assertThat(dto.modelsLoaded()).isEqualTo(health.modelsLoaded());
            assertThat(dto.dataAsOf()).isEqualTo(DATA_AS_OF);
            assertThat(dto.stale()).isTrue();
        }
    }

    @Nested
    class Schedule {
        @Test
        void deserialisesTheListAndEveryFieldInAnElement() {
            List<InferenceScheduledGame> games = mapper.readValue(
                    Fixture.read("schedule.json"), new TypeReference<>() {
                    });

            assertThat(games).hasSize(14);

            InferenceScheduledGame first = games.get(0);
            assertThat(first.homeTeamId()).isEqualTo(1610612765L);
            assertThat(first.awayTeamId()).isEqualTo(1610612738L);
            assertThat(first.gameDate()).isEqualTo(LocalDate.of(2026, 10, 20));

            assertThat(games).allSatisfy(game -> {
                assertThat(game.homeTeamId()).isNotNull();
                assertThat(game.awayTeamId()).isNotNull();
                assertThat(game.gameDate()).isNotNull();
            });
        }
    }

    @Nested
    class FullGame {
        @Test
        void deserialisesAllSevenPredictionsInsideTheNestedObject() {
            InferenceResponse response =
                    mapper.readValue(Fixture.read("predict.json"), InferenceResponse.class);

            assertThat(response.dataAsOf()).isEqualTo(DATA_AS_OF);
            assertThat(response.stale()).isTrue();
            assertThat(response.daysBehind()).isEqualTo(DAYS_BEHIND);

            InferenceResponse.Predictions p = response.predictions();
            assertThat(p).isNotNull();
            assertThat(p.homeWinProbability()).isEqualTo(0.42541608214378357);
            assertThat(p.homeMargin()).isEqualTo(1.4149187803268433);
            assertThat(p.totalPoints()).isEqualTo(232.93231201171875);
            assertThat(p.reboundMargin()).isEqualTo(1.5239256620407104);
            assertThat(p.totalRebounds()).isEqualTo(89.01725769042969);
            assertThat(p.assistMargin()).isEqualTo(1.986781120300293);
            assertThat(p.totalAssists()).isEqualTo(50.692169189453125);
        }

        @Test
        void homeWinProbabilityStillMatchesTheRecordedReferenceValue() {
            InferenceResponse response =
                    mapper.readValue(Fixture.read("predict.json"), InferenceResponse.class);

            assertThat(response.predictions().homeWinProbability())
                    .isEqualTo(0.42541608214378357, within(1e-15));
        }
    }

    @Nested
    class QuarterHalf {
        @Test
        void deserialisesSixMarketsAsAListNotSixNamedFields() {
            InferenceQuarterHalfResponse response = mapper.readValue(
                    Fixture.read("predict-quarter-half.json"),
                    InferenceQuarterHalfResponse.class);

            assertThat(response.dataAsOf()).isEqualTo(DATA_AS_OF);
            assertThat(response.daysBehind()).isEqualTo(DAYS_BEHIND);
            assertThat(response.predictions()).hasSize(6);

            assertThat(response.predictions())
                    .extracting(InferenceQuarterHalfResponse.Market::market)
                    .containsExactly(
                            "q1_home_margin", "q1_total_points",
                            "half1_home_margin", "half1_total_points",
                            "q1_home_win_probability", "half1_home_win_probability");
        }

        @Test
        void carriesTheQualifiersThatMakeAWinnerMarketReadable() {
            InferenceQuarterHalfResponse response = mapper.readValue(
                    Fixture.read("predict-quarter-half.json"),
                    InferenceQuarterHalfResponse.class);

            InferenceQuarterHalfResponse.Market q1Winner =
                    response.market("q1_home_win_probability");
            assertThat(q1Winner.value()).isEqualTo(0.4469873035961624);
            assertThat(q1Winner.confidence()).isEqualTo("low");
            assertThat(q1Winner.interpretation()).isEqualTo("P(home leads | not tied)");

            InferenceQuarterHalfResponse.Market q1Margin = response.market("q1_home_margin");
            assertThat(q1Margin.value()).isEqualTo(-1.124112908717173);
            assertThat(q1Margin.confidence()).isEqualTo("medium");
            assertThat(q1Margin.interpretation()).isNull();
        }
    }

    @Nested
    class PlayerProps {
        @Test
        void deserialisesThreeLevelsDownToANamedPlayersValues() {
            InferencePlayerPropsResponse response = mapper.readValue(
                    Fixture.read("predict-player-props.json"),
                    InferencePlayerPropsResponse.class);

            assertThat(response.dataAsOf()).isEqualTo(DATA_AS_OF);
            assertThat(response.teams()).hasSize(2);

            InferencePlayerPropsResponse.TeamBoard home = response.board(true);
            assertThat(home.teamId()).isEqualTo(1610612737L);
            assertThat(home.isHome()).isTrue();
            assertThat(home.availabilityKnown()).isFalse();
            assertThat(home.availabilityNote()).contains("AVAILABILITY UNKNOWN");
            assertThat(home.players()).hasSize(10);

            InferencePlayerPropsResponse.PlayerLine first = home.players().get(0);
            assertThat(first.playerId()).isEqualTo(1629638L);
            assertThat(first.playerName()).isEqualTo("Nickeil Alexander-Walker");
            assertThat(first.modelUsed()).isEqualTo("linear");
            assertThat(first.value("PTS")).isEqualTo(22.477595340281667);
            assertThat(first.value("REB")).isEqualTo(3.388430796609306);
            assertThat(first.value("AST")).isEqualTo(3.711617752945157);
            assertThat(first.value("FG3M")).isEqualTo(3.535938657494989);
            assertThat(first.value("PRA")).isEqualTo(29.577643889836146);
        }

        @Test
        void bothBoardsArriveAndTheAwaySideIsNotAHomeCopy() {
            InferencePlayerPropsResponse response = mapper.readValue(
                    Fixture.read("predict-player-props.json"),
                    InferencePlayerPropsResponse.class);

            InferencePlayerPropsResponse.TeamBoard away = response.board(false);
            assertThat(away.teamId()).isEqualTo(1610612738L);
            assertThat(away.isHome()).isFalse();
            assertThat(away.players()).hasSize(10);
            assertThat(away.players().get(0).playerName()).isEqualTo("Jaylen Brown");
            assertThat(away.players().get(0).value("PTS")).isEqualTo(28.10797718167896);

            assertThat(response.teams())
                    .extracting(InferencePlayerPropsResponse.TeamBoard::isHome)
                    .containsExactlyInAnyOrder(true, false);
        }

        @Test
        void everyPlayerLineIsFullyPopulated() {
            InferencePlayerPropsResponse response = mapper.readValue(
                    Fixture.read("predict-player-props.json"),
                    InferencePlayerPropsResponse.class);

            assertThat(response.teams()).allSatisfy(board ->
                    assertThat(board.players()).allSatisfy(player -> {
                        assertThat(player.playerId()).isNotNull();
                        assertThat(player.playerName()).isNotBlank();
                        assertThat(player.modelUsed()).isNotBlank();
                        for (String target : List.of("PTS", "REB", "AST", "FG3M", "PRA")) {
                            assertThat(player.value(target)).isNotNaN();
                        }
                    }));
        }
    }
}
