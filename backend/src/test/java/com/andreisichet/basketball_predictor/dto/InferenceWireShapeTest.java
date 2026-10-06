package com.andreisichet.basketball_predictor.dto;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.time.Instant;
import java.time.LocalDate;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.json.JsonTest;

import com.andreisichet.basketball_predictor.model.GleaguePrediction;
import com.andreisichet.basketball_predictor.model.WnbaPrediction;

import tools.jackson.core.type.TypeReference;
import tools.jackson.databind.ObjectMapper;

/** Pins the wire shape of every inference-service response the backend deserialises. */
@JsonTest
class InferenceWireShapeTest {
    @Autowired
    private ObjectMapper mapper;

    /** All fixtures come from this matchup, the one used throughout the project. */
    private static final LocalDate DATA_AS_OF = LocalDate.of(2026, 4, 12);
    // No DAYS_BEHIND literal. It is DATE-DERIVED - computed from
    // datetime.now() against a fixed cutoff - so it increments every midnight
    // and moves on every recapture. It was pinned at 146 and the A7 recapture
    // made it 177; bumping the number would only re-arm the same trap. The
    // health assertion below already does it this way, and the regression gate
    // classifies the field as volatile for the same reason.
    private static final int DAYS_BEHIND_FLOOR = 100;

    /**
     * health.json was recaptured when the WNBA family was added, so its
     * daysBehind is from that later moment while the four prediction fixtures
     * keep theirs. The numbers are both static records of a real response; no
     * test cross-references the two, so they are allowed to differ rather
     * than one being edited to match the other.
     */
    private static final LocalDate WNBA_DATA_AS_OF = LocalDate.of(2026, 9, 24);
    private static final LocalDate GLEAGUE_DATA_AS_OF = LocalDate.of(2026, 3, 28);

    @Nested
    class Health {
        @Test
        void deserialisesThePerFamilyModelBreakdown() {
            InferenceHealth health =
                    mapper.readValue(Fixture.read("health.json"), InferenceHealth.class);

            assertThat(health.status()).isEqualTo("ok");
            assertThat(health.dataAsOf()).isEqualTo(DATA_AS_OF);
            // daysBehind is DATE-DERIVED, so it is checked for type and
            // plausibility rather than pinned to a literal. It is computed
            // from datetime.now() against a fixed cutoff, so it increments
            // every midnight and changes whenever this fixture is recaptured
            // - which is exactly what broke this assertion when the G League
            // block was added and the capture moved 174 -> 175.
            //
            // The regression gate classifies the same field as volatile for
            // the same reason. Pinning it here would be pinning the CAPTURE
            // DATE, which is not what a wire-shape test is for; dataAsOf
            // above is the stable value and is pinned.
            assertThat(health.daysBehind()).isGreaterThan(0);
            assertThat(health.stale()).isTrue();

            assertThat(health.modelsLoaded()).containsExactlyInAnyOrderEntriesOf(
                    Map.of("team", 7, "quarter_half", 6, "player_props", 10,
                            "wnba", 3, "gleague", 3));
        }

        @Test
        void deserialisesTheWnbaBlockAsItsOwnCutoff() {
            InferenceHealth health =
                    mapper.readValue(Fixture.read("health.json"), InferenceHealth.class);

            assertThat(health.wnba()).isNotNull();
            assertThat(health.wnba().dataAsOf()).isEqualTo(WNBA_DATA_AS_OF);
            assertThat(health.wnba().stale()).isTrue();

            // The point of the nested block: a DIFFERENT date from the NBA's.
            // If these were ever equal the test would still pass, so the
            // inequality is asserted rather than left implied by two literals.
            assertThat(health.wnba().dataAsOf()).isNotEqualTo(health.dataAsOf());
        }

        @Test
        void carriesTheBreakdownThroughToTheClientDto() {
            InferenceHealth health =
                    mapper.readValue(Fixture.read("health.json"), InferenceHealth.class);

            HealthDto dto = HealthDto.from(health);

            assertThat(dto.modelsLoaded()).isEqualTo(health.modelsLoaded());
            assertThat(dto.dataAsOf()).isEqualTo(DATA_AS_OF);
            assertThat(dto.stale()).isTrue();

            // The NBA cutoff stays top-level and the WNBA's arrives beside
            // it, which is the additive shape section 21's outage argued for.
            assertThat(dto.wnba()).isNotNull();
            assertThat(dto.wnba().dataAsOf()).isEqualTo(WNBA_DATA_AS_OF);
            assertThat(dto.gleague()).isNotNull();
            assertThat(dto.gleague().dataAsOf()).isEqualTo(GLEAGUE_DATA_AS_OF);
        }

        @Test
        void deserialisesTheGleagueBlockAsItsOwnCutoff() {
            InferenceHealth health =
                    mapper.readValue(Fixture.read("health.json"), InferenceHealth.class);

            assertThat(health.gleague()).isNotNull();
            assertThat(health.gleague().dataAsOf()).isEqualTo(GLEAGUE_DATA_AS_OF);
            assertThat(health.gleague().stale()).isTrue();

            // THREE DISTINCT CUTOFFS, ASSERTED PAIRWISE. The regression gate
            // caught this block being dropped by these records while the
            // Python side sent it, exactly as it caught the WNBA's a day
            // earlier - so the inequalities are asserted rather than left
            // implied by three literals that happen to differ.
            assertThat(health.gleague().dataAsOf()).isNotEqualTo(health.dataAsOf());
            assertThat(health.gleague().dataAsOf())
                    .isNotEqualTo(health.wnba().dataAsOf());
        }
    }

    @Nested
    class Gleague {
        @Test
        void deserialisesAllThreeMarketsAndTheirProvenance() {
            InferenceGleagueResponse response = mapper.readValue(
                    Fixture.read("predict-gleague.json"),
                    InferenceGleagueResponse.class);

            assertThat(response.dataAsOf()).isEqualTo(GLEAGUE_DATA_AS_OF);
            assertThat(response.stale()).isTrue();
            assertThat(response.markets())
                    .containsOnlyKeys("moneyline", "spread", "totals");

            // SEASON IS A STRING HERE AND AN INT FOR THE WNBA, and that is
            // the shape difference most likely to be "tidied" into a single
            // type. A G League season spans two calendar years and is
            // labelled "2025-26"; a WNBA season sits inside one. Declaring
            // this int would fail deserialisation outright.
            assertThat(response.season()).isEqualTo("2025-26");

            // Pinned, not merely present: these three were reproduced by
            // float equality against phase 3's own artifacts offline, so a
            // drift here means the serving path moved.
            assertThat(response.market("moneyline").value())
                    .isEqualTo(0.5908580792747549);
            assertThat(response.market("spread").value())
                    .isEqualTo(2.927269925841589);
            assertThat(response.market("totals").value())
                    .isEqualTo(244.46527901729075);

            // All three ship CARRY10 after the serving tie-break switched two
            // CUP picks to their CARRY equivalents. Asserted per market
            // rather than once, because they are three independent selection
            // outputs that happen to agree.
            assertThat(response.market("moneyline").window()).isEqualTo("CARRY10");
            assertThat(response.market("spread").window()).isEqualTo("CARRY10");
            assertThat(response.market("totals").window()).isEqualTo("CARRY10");
        }

        @Test
        void theResponseCarriesNoEngineeringNote() {
            // The G League manifest records that every candidate tied on
            // validation, so the shipped configuration is "defensible, not
            // demonstrated". True, useful to whoever retrains this, and NOT a
            // property of any one prediction - so it must not reach the wire.
            // The WNBA phase shipped that kind of note and had to withdraw it
            // from three layers.
            //
            // Asserted on the RAW JSON rather than through the record,
            // because the record does not declare such a field - reading it
            // back would check that something this class cannot see is
            // absent, which passes whatever Python sends.
            String body = Fixture.read("predict-gleague.json");

            assertThat(body).doesNotContain("tied_with");
            assertThat(body).doesNotContain("defensible");
            assertThat(body).doesNotContain("demonstrated");
            assertThat(body).doesNotContain("caveat");

            // And the fixture is a real capture rather than an edited one.
            assertThat(body).contains("\"window\":\"CARRY10\"");
        }

        @Test
        void aRenamedMarketThrowsRatherThanReadingAsZero() {
            // The negative test. A market Python renames must fail loudly,
            // not arrive as 0.0 and be stored as a prediction - which is what
            // a plain map lookup would do.
            String renamed = Fixture.read("predict-gleague.json")
                    .replace("\"totals\"", "\"total_points\"");

            InferenceGleagueResponse response =
                    mapper.readValue(renamed, InferenceGleagueResponse.class);

            assertThatThrownBy(() -> response.market("totals"))
                    .isInstanceOf(IllegalStateException.class)
                    .hasMessageContaining("no G League market named totals");
        }

        @Test
        void mapsThroughToTheClientDto() {
            InferenceGleagueResponse response = mapper.readValue(
                    Fixture.read("predict-gleague.json"),
                    InferenceGleagueResponse.class);

            GleaguePrediction saved = new GleaguePrediction();
            saved.setHomeWinProbability(response.market("moneyline").value());
            saved.setHomeMargin(response.market("spread").value());
            saved.setTotalPoints(response.market("totals").value());
            saved.setDataAsOf(response.dataAsOf());
            saved.setStale(response.stale());
            saved.setPredictedAt(Instant.parse("2026-10-04T00:00:00Z"));

            GleagueSummaryDto.Prediction dto = GleagueSummaryDto.Prediction.of(
                    saved,
                    response,
                    response.market("moneyline"),
                    response.market("spread"),
                    response.market("totals"));

            assertThat(dto.homeWinProbability()).isEqualTo(0.5908580792747549);
            assertThat(dto.homeMargin()).isEqualTo(2.927269925841589);
            assertThat(dto.totalPoints()).isEqualTo(244.46527901729075);
            assertThat(dto.season()).isEqualTo("2025-26");
            assertThat(dto.moneylineWindow()).isEqualTo("CARRY10");
            assertThat(dto.dataAsOf()).isEqualTo(GLEAGUE_DATA_AS_OF);
            // daysBehind comes from the inference body rather than the saved
            // row, which is the one field here that could silently read 0.
            assertThat(dto.daysBehind()).isEqualTo(190);
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
            assertThat(response.daysBehind()).isGreaterThan(DAYS_BEHIND_FLOOR);

            InferenceResponse.Predictions p = response.predictions();
            assertThat(p).isNotNull();
            assertThat(p.homeWinProbability()).isEqualTo(0.4191701412200928);
            assertThat(p.homeMargin()).isEqualTo(1.314629077911377);
            assertThat(p.totalPoints()).isEqualTo(230.9997100830078);
            assertThat(p.reboundMargin()).isEqualTo(0.7266454696655273);
            assertThat(p.totalRebounds()).isEqualTo(88.9955825805664);
            assertThat(p.assistMargin()).isEqualTo(2.744473457336426);
            assertThat(p.totalAssists()).isEqualTo(51.03422546386719);
        }

        @Test
        void homeWinProbabilityStillMatchesTheRecordedReferenceValue() {
            InferenceResponse response =
                    mapper.readValue(Fixture.read("predict.json"), InferenceResponse.class);

            assertThat(response.predictions().homeWinProbability())
                    .isEqualTo(0.4191701412200928, within(1e-15));
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
            assertThat(response.daysBehind()).isGreaterThan(DAYS_BEHIND_FLOOR);
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

    @Nested
    class Wnba {
        private static final LocalDate GAME_DATE = LocalDate.of(2026, 9, 25);

        @Test
        void deserialisesAllThreeMarketsAndTheirProvenance() {
            InferenceWnbaResponse response = mapper.readValue(
                    Fixture.read("predict-wnba.json"), InferenceWnbaResponse.class);

            assertThat(response.dataAsOf()).isEqualTo(WNBA_DATA_AS_OF);
            assertThat(response.stale()).isTrue();
            assertThat(response.season()).isEqualTo(2026);
            assertThat(response.markets()).containsOnlyKeys("moneyline", "spread", "totals");

            // Phoenix Mercury at home against the Las Vegas Aces. The values
            // are pinned, not just checked for presence: they were reproduced
            // bit-identically by the offline cross-check against phase 3's own
            // dataset, so a drift here means the serving path moved.
            assertThat(response.market("moneyline").value()).isEqualTo(0.18193019489666626);
            assertThat(response.market("spread").value()).isEqualTo(-8.403725674288566);
            assertThat(response.market("totals").value()).isEqualTo(178.40689601339275);

            // Two targets chose CARRY5 and one chose CARRY10. A single window
            // would make this field look decorative; it is not.
            assertThat(response.market("moneyline").window()).isEqualTo("CARRY5");
            assertThat(response.market("spread").window()).isEqualTo("CARRY5");
            assertThat(response.market("totals").window()).isEqualTo("CARRY10");
        }

        @Test
        void theResponseCarriesNoCaveatAtAll() {
            // THE FIELD IS ABSENT FROM THE WIRE, NOT NULL ON IT. Asserted on
            // the raw JSON rather than through the record, because the record
            // no longer declares the field - so reading it back would be
            // checking that a thing this class cannot see is not there, which
            // would pass whatever Python sends.
            String body = Fixture.read("predict-wnba.json");

            assertThat(body).doesNotContain("caveat");
            assertThat(body).doesNotContain("Elo alone");

            // And the fixture is a real capture, not an edited one: it still
            // carries everything the response does send.
            assertThat(body).contains("\"window\":\"CARRY10\"");
        }

        @Test
        void mapsThroughToTheClientDtoWithoutIt() {
            InferenceWnbaResponse response = mapper.readValue(
                    Fixture.read("predict-wnba.json"), InferenceWnbaResponse.class);

            WnbaPrediction saved = new WnbaPrediction();
            saved.setHomeWinProbability(response.market("moneyline").value());
            saved.setHomeMargin(response.market("spread").value());
            saved.setTotalPoints(response.market("totals").value());
            saved.setDataAsOf(response.dataAsOf());
            saved.setStale(response.stale());
            saved.setPredictedAt(Instant.parse("2026-10-03T00:00:00Z"));

            WnbaSummaryDto.Prediction dto = WnbaSummaryDto.Prediction.of(
                    saved,
                    response,
                    response.market("moneyline"),
                    response.market("spread"),
                    response.market("totals"));

            assertThat(dto.homeWinProbability()).isEqualTo(0.18193019489666626);
            assertThat(dto.totalsWindow()).isEqualTo("CARRY10");

            // The windows are what remains of the provenance, and they are
            // genuinely per-target rather than decorative.
            assertThat(dto.moneylineWindow()).isEqualTo("CARRY5");
            assertThat(dto.spreadWindow()).isEqualTo("CARRY5");
        }

        @Test
        void aMissingMarketFailsLoudlyRatherThanReadingAsZero() {
            // The negative test. If Python renamed or dropped a market, a
            // lenient lookup would hand the entity a 0.0 and the row would
            // claim a 0% win probability and a 0-point total - numbers that
            // look like predictions. Same reasoning as the quarter/half
            // response throwing on an unknown market name.
            InferenceWnbaResponse response = mapper.readValue(
                    Fixture.read("predict-wnba.json").replace("\"spread\"", "\"handicap\""),
                    InferenceWnbaResponse.class);

            assertThatThrownBy(() -> response.market("spread"))
                    .isInstanceOf(IllegalStateException.class)
                    .hasMessageContaining("no WNBA market named spread")
                    .hasMessageContaining("response shape has changed");
        }
    }
}
