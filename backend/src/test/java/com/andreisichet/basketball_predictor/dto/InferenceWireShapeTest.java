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
import com.andreisichet.basketball_predictor.service.ScheduleSyncStatus;
import com.andreisichet.basketball_predictor.model.NflPrediction;
import com.andreisichet.basketball_predictor.model.WnbaPrediction;

import tools.jackson.core.type.TypeReference;
import tools.jackson.databind.ObjectMapper;

/** Pins the wire shape of every inference-service response the backend deserialises. */
@JsonTest
class InferenceWireShapeTest {
    @Autowired
    private ObjectMapper mapper;

    /** A double quote, so the rename negative below needs no escaping. */
    private static final String QUOTE = String.valueOf((char) 34);

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
    private static final LocalDate NFL_DATA_AS_OF = LocalDate.of(2026, 10, 5);

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
                            "wnba", 3, "gleague", 3, "nfl", 3));
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

            // The second argument is the schedule-sync block, which is
            // the backend's own state rather than anything on the inference
            // wire. A fresh ScheduleSyncStatus is the never_run state, which
            // is what this endpoint reports in the moment before the first
            // attempt.
            HealthDto dto = HealthDto.from(health,
                    new ScheduleSyncStatus().snapshot());

            assertThat(dto.modelsLoaded()).isEqualTo(health.modelsLoaded());
            assertThat(dto.dataAsOf()).isEqualTo(DATA_AS_OF);
            assertThat(dto.stale()).isTrue();

            // The NBA cutoff stays top-level and the WNBA's arrives beside
            // it, which is the additive shape section 21's outage argued for.
            assertThat(dto.wnba()).isNotNull();
            assertThat(dto.wnba().dataAsOf()).isEqualTo(WNBA_DATA_AS_OF);
            assertThat(dto.gleague()).isNotNull();
            assertThat(dto.gleague().dataAsOf()).isEqualTo(GLEAGUE_DATA_AS_OF);
            assertThat(dto.nfl()).isNotNull();
            assertThat(dto.nfl().dataAsOf()).isEqualTo(NFL_DATA_AS_OF);
            // The two fields the NFL's rule needs and no other league has.
            // Carried because the frontend cannot compute predictability from
            // a cutoff for this league the way it does for the other three.
            assertThat(dto.nfl().predictableFixtures()).isPositive();
            assertThat(dto.nfl().predictionRule())
                    .isEqualTo("both teams' previous games in history");

            // ADDITIVE, like the WNBA's block before it. Every assertion
            // above this line is the one it was, which is the point: section
            // 21's outage came from RETYPING a field on this record, so a new
            // field arrives beside the old ones and nothing existing moves.
            assertThat(dto.scheduleSync()).isNotNull();
            assertThat(dto.scheduleSync().lastResult()).isEqualTo("never_run");
            assertThat(dto.scheduleSync().leagues()).isEmpty();
            assertThat(dto.scheduleSync().lastSuccessAt()).isNull();
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

        @Test
        void deserialisesTheNflBlockWithItsOwnRule() {
            InferenceHealth health =
                    mapper.readValue(Fixture.read("health.json"), InferenceHealth.class);

            assertThat(health.nfl()).isNotNull();
            assertThat(health.nfl().dataAsOf()).isEqualTo(NFL_DATA_AS_OF);

            // FOUR DISTINCT CUTOFFS, ASSERTED PAIRWISE, for the reason the G
            // League's block records: these records have silently dropped a
            // league's freshness block twice while the Python side was
            // sending it, so the inequalities are asserted rather than left
            // implied by four literals that happen to differ today.
            assertThat(health.nfl().dataAsOf()).isNotEqualTo(health.dataAsOf());
            assertThat(health.nfl().dataAsOf())
                    .isNotEqualTo(health.wnba().dataAsOf());
            assertThat(health.nfl().dataAsOf())
                    .isNotEqualTo(health.gleague().dataAsOf());

            // The dependency rule's own fields. predictableFixtures is a
            // small fraction of fixtures and must be, because a fixture
            // becomes predictable only once both teams' previous games are
            // recorded - so asserting it is merely positive would also pass
            // on a service that had quietly started offering the whole
            // season at once.
            assertThat(health.nfl().fixtures()).isGreaterThan(100);
            assertThat(health.nfl().predictableFixtures()).isPositive();
            assertThat(health.nfl().predictableFixtures())
                    .isLessThan(health.nfl().fixtures());

            // THE LICENCE TRAVELS. The NFL history is CC BY-SA 4.0 and no
            // other league here carries an attribution requirement, so this
            // field exists for this league and nowhere else.
            assertThat(health.nfl().source())
                    .isEqualTo("English Wikipedia, CC BY-SA 4.0");

            // Real tables, not CI's invented ones. If this is ever true in a
            // captured fixture, a CC BY-SA table reached a checkout - which
            // is a licence problem rather than a test failure.
            assertThat(health.nfl().synthetic()).isFalse();
        }
    }

    @Nested
    class Nfl {
        @Test
        void deserialisesAllThreeMarketsAndTheirProvenance() {
            InferenceNflResponse response = mapper.readValue(
                    Fixture.read("predict-nfl.json"), InferenceNflResponse.class);

            assertThat(response.dataAsOf()).isEqualTo(NFL_DATA_AS_OF);
            assertThat(response.stale()).isTrue();
            assertThat(response.daysBehind()).isGreaterThan(0);
            assertThat(response.season()).isEqualTo(2026);

            // WEEK HAS NO COUNTERPART IN THE OTHER THREE RECORDS. It is a
            // feature the models actually read and the unit an NFL result is
            // discussed in, so it travels rather than being derived here.
            assertThat(response.week()).isEqualTo(5);

            assertThat(response.markets()).containsOnlyKeys(
                    "winner", "margin", "total");

            InferenceNflResponse.Market winner = response.market("winner");
            assertThat(winner.value()).isCloseTo(0.4390723587536829, within(1e-12));
            assertThat(winner.metric()).isEqualTo("log_loss");
            assertThat(winner.modelUsed()).isEqualTo("linear/mov/elo_context");
            assertThat(winner.note()).isEqualTo("TIES Elo alone");
            assertThat(winner.imputed()).isEmpty();

            assertThat(response.market("margin").value())
                    .isCloseTo(-2.7356462478637695, within(1e-12));
            assertThat(response.market("margin").modelUsed())
                    .isEqualTo("xgboost/mov/elo_context");
            assertThat(response.market("total").value())
                    .isCloseTo(43.34370323002624, within(1e-12));

            // The qualifier the winner probability MEANS. Ties are excluded
            // from the label, so this is not the same quantity as a
            // basketball moneyline and must not be displayed as if it were.
            assertThat(response.homeWinInterpretation())
                    .isEqualTo("P(home wins | not tied)");
            assertThat(response.source())
                    .isEqualTo("English Wikipedia, CC BY-SA 4.0");
        }

        @Test
        void mapsThroughToTheClientDtoWithTheLicenceAndTheNotes() {
            InferenceNflResponse inference = mapper.readValue(
                    Fixture.read("predict-nfl.json"), InferenceNflResponse.class);

            NflPrediction saved = new NflPrediction();
            saved.setHomeWinProbability(inference.market("winner").value());
            saved.setHomeMargin(inference.market("margin").value());
            saved.setTotalPoints(inference.market("total").value());
            saved.setWeek(inference.week());
            saved.setDataAsOf(inference.dataAsOf());
            saved.setStale(inference.stale());
            saved.setPredictedAt(Instant.parse("2026-10-09T07:00:00Z"));

            NflSummaryDto.Prediction dto = NflSummaryDto.Prediction.of(
                    saved, inference, inference.market("winner"),
                    inference.market("margin"), inference.market("total"));

            assertThat(dto.homeWinProbability())
                    .isCloseTo(0.4390723587536829, within(1e-12));
            assertThat(dto.week()).isEqualTo(5);
            assertThat(dto.season()).isEqualTo(2026);
            assertThat(dto.homeWinInterpretation())
                    .isEqualTo("P(home wins | not tied)");
            assertThat(dto.winnerModel()).isEqualTo("linear/mov/elo_context");
            assertThat(dto.marginModel()).isEqualTo("xgboost/mov/elo_context");
            assertThat(dto.winnerNote()).isEqualTo("TIES Elo alone");
            assertThat(dto.marginNote()).isEqualTo("TIES Elo alone");
            assertThat(dto.source()).isEqualTo("English Wikipedia, CC BY-SA 4.0");
        }

        @Test
        void throwsWhenAMarketIsRenamedRatherThanReadingItAsZero() {
            // The same negative the other leagues' records carry, and the
            // reason is sharper than "fail loudly": a missing double
            // deserialises to 0.0, and a 0.0 home-win probability is a
            // servable-looking number rather than an obvious absence.
            String renamed = Fixture.read("predict-nfl.json")
                    .replace(QUOTE + "winner" + QUOTE,
                             QUOTE + "home_winner" + QUOTE);

            InferenceNflResponse response =
                    mapper.readValue(renamed, InferenceNflResponse.class);

            assertThatThrownBy(() -> response.market("winner"))
                    .isInstanceOf(IllegalStateException.class)
                    .hasMessageContaining("no NFL market named winner");
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

        @Test
        void leavesTheFourNflOnlyFieldsNullForBasketball() {
            List<InferenceScheduledGame> games = mapper.readValue(
                    Fixture.read("schedule.json"), new TypeReference<>() {
                    });

            // NULL, NOT false AND NOT ZERO. The three basketball leagues send
            // none of these: they decide predictability from a date the client
            // computes, have no kickoff time and no week. `false` would say
            // "this fixture cannot be predicted", which is a different and
            // wrong claim - the client must be able to see that there is no
            // server-side answer at all, which is why these are object types.
            assertThat(games).allSatisfy(game -> {
                assertThat(game.predictable()).isNull();
                assertThat(game.kickoffUtc()).isNull();
                assertThat(game.flex()).isNull();
                assertThat(game.week()).isNull();
            });
        }
    }

    @Nested
    class NflSchedule {
        /**
         * CAPTURED FROM A CONTAINER BOOTED ON THE SYNTHETIC NFL TABLES, so it
         * is a real response and carries no Wikipedia data.
         *
         * The NFL history is CC BY-SA 4.0 and no NFL table is committed, which
         * would normally rule out committing a captured schedule too - a
         * fixture list is source data. Phase 4's synthetic generator resolves
         * it: every venue is the string "Synthetic Stadium", every date and
         * score is from a seeded PRNG, and the only things shared with the
         * real tables are the schema and the 32 project-authored franchise
         * ids. So this is captured rather than hand-written - which is what
         * lets it catch a field rename - without redistributing anything.
         *
         * Its one limitation, stated: the generator emits no TBD kickoff, so
         * this fixture cannot pin a null `kickoff_utc`. The real schedule has
         * 24 of those and the frontend covers that case in Nfl.test.js.
         */
        @Test
        void deserialisesPredictableKickoffFlexAndWeek() {
            List<InferenceScheduledGame> games = mapper.readValue(
                    Fixture.read("schedule-nfl.json"), new TypeReference<>() {
                    });

            assertThat(games).hasSize(32);

            InferenceScheduledGame first = games.get(0);
            assertThat(first.homeTeamId()).isEqualTo(1613000001L);
            assertThat(first.awayTeamId()).isEqualTo(1613000002L);
            assertThat(first.gameDate()).isEqualTo(LocalDate.of(2026, 9, 13));
            assertThat(first.week()).isEqualTo(1);
            assertThat(first.flex()).isFalse();
            assertThat(first.predictable()).isTrue();

            // AN INSTANT, NOT A STRING. The wire carries an offset datetime;
            // if this stopped parsing, every NFL row would render with no
            // kickoff and nothing would fail.
            assertThat(first.kickoffUtc())
                    .isEqualTo(Instant.parse("2026-09-13T20:00:00Z"));

            // BOTH VALUES OF predictable ARE PRESENT, so this fixture pins the
            // field rather than one of its two states.
            assertThat(games).anySatisfy(
                    game -> assertThat(game.predictable()).isTrue());
            assertThat(games).anySatisfy(
                    game -> assertThat(game.predictable()).isFalse());

            assertThat(games).allSatisfy(game -> {
                assertThat(game.week()).isNotNull();
                assertThat(game.predictable()).isNotNull();
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

    /**
     * The IN-SEASON captures, taken 2026-10-21 - the day after the NBA season
     * opened, and the first day since this project was built on which an
     * injury report was actually published.
     *
     * THESE SIT BESIDE THE OFFSEASON FIXTURES AND DO NOT REPLACE THEM. Every
     * fixture above was captured with no injury report available, which is a
     * real state the service returns for roughly seven months of the year and
     * must stay under test. What had never been under test is the other half:
     * a non-stale cutoff, a published report, and the two refusals that only
     * happen when a season is a day old.
     *
     * Four shapes here are reachable at no other time of year:
     *
     *   1. stale=false with daysBehind=1. Every other fixture is stale with
     *      daysBehind in the hundreds, so "not stale" had never been read off
     *      the wire at all.
     *   2. availability.nba.state = "used" with a real report_timestamp. The
     *      offseason fixture carries "none_published" and a null stamp.
     *   3. The quarter/half models REFUSING, because no team has a complete
     *      rolling window two days into a season. The offseason fixture is a
     *      successful six-market response.
     *   4. An EMPTY player-props board, for the same reason one step further
     *      on: no player has usable history yet.
     *
     * Nothing here pins a clock value. daysBehind, age_seconds and
     * observed_at all move with the calendar rather than with the code, and
     * the reasoning is the same one recorded on DAYS_BEHIND_FLOOR above.
     */
    @Nested
    class InSeason {
        /** The cutoff is the day BEFORE the fixture: the NBA rule is dataAsOf + 1. */
        private static final LocalDate IN_SEASON_DATA_AS_OF = LocalDate.of(2026, 10, 20);
        private static final LocalDate IN_SEASON_GAME_DATE = LocalDate.of(2026, 10, 21);

        /** Orlando hosting Atlanta, one of the eleven fixtures on the rule date. */
        private static final long ORL = 1610612753L;
        private static final long ATL = 1610612737L;

        @Test
        void healthIsNotStaleAndCarriesTheSameModelCountAsTheOffseason() {
            InferenceHealth health = mapper.readValue(
                    Fixture.read("health-inseason.json"), InferenceHealth.class);

            assertThat(health.status()).isEqualTo("ok");
            assertThat(health.dataAsOf()).isEqualTo(IN_SEASON_DATA_AS_OF);

            // The assertion the offseason fixtures cannot make. Theirs is
            // isTrue() with daysBehind over a hundred; this is the other
            // branch of the same field, read off a real response for the
            // first time.
            assertThat(health.stale()).isFalse();
            assertThat(health.daysBehind()).isGreaterThan(0);
            assertThat(health.daysBehind()).isLessThan(DAYS_BEHIND_FLOOR);

            // A season opening must not change which models serve. Data
            // advances daily; a model only through a reviewed promotion.
            assertThat(health.modelsLoaded()).containsExactlyInAnyOrderEntriesOf(
                    Map.of("team", 7, "quarter_half", 6, "player_props", 10,
                            "wnba", 3, "gleague", 3, "nfl", 3));

            // The two out-of-season leagues are still stale, and say so
            // against their OWN cutoffs rather than the NBA's.
            assertThat(health.wnba().dataAsOf()).isEqualTo(WNBA_DATA_AS_OF);
            assertThat(health.wnba().stale()).isTrue();
            assertThat(health.gleague().dataAsOf()).isEqualTo(GLEAGUE_DATA_AS_OF);
            assertThat(health.gleague().stale()).isTrue();
        }

        @Test
        @SuppressWarnings("unchecked")
        void theAvailabilityBlockCarriesAPublishedReportRatherThanNonePublished() {
            Map<String, Object> body = mapper.readValue(
                    Fixture.read("health-inseason.json"),
                    new TypeReference<Map<String, Object>>() {
                    });
            Map<String, Object> availability =
                    (Map<String, Object>) body.get("availability");
            Map<String, Object> nba = (Map<String, Object>) availability.get("nba");

            // THE SHAPE E12 EXISTED FOR. Offseason this reads
            // "none_published" with a null report_timestamp and a detail
            // sentence explaining the absence; in season it resolves.
            assertThat(nba.get("state")).isEqualTo("used");
            assertThat(nba.get("report_timestamp")).isEqualTo("2026-10-21T04:15:00");
            assertThat(nba.get("for_date")).isEqualTo("2026-10-21");

            // detail is the null half of a two-field pair: it explains a
            // non-resolving state and is null when there is nothing to
            // explain. An empty string here would read as an explanation.
            assertThat(nba.get("detail")).isNull();

            // Present, deliberately not pinned - these are clock values.
            assertThat(nba).containsKeys("observed_at", "age_seconds", "source");

            // The other three leagues name themselves out of scope. This is
            // the project's "an unknown is never a zero" rule on the wire: a
            // league with no availability features says so, rather than
            // reporting a resolved state or a 0 that reads as "nobody out".
            for (String league : List.of("wnba", "gleague", "nfl")) {
                Map<String, Object> other =
                        (Map<String, Object>) availability.get(league);
                assertThat(other.get("state")).isEqualTo("not_applicable");
                assertThat(other.get("detail")).asString().isNotEmpty();
            }

            // InferenceHealth declares neither availability nor served_data,
            // so this asserts the thing that actually matters to the backend:
            // a populated block it does not read does not break the parse. It
            // is NOT a claim that the backend surfaces availability from
            // /health - it does not, and the only availability that reaches a
            // browser comes through the player-props board below.
            assertThat(body).containsKey("served_data");
            InferenceHealth parsed = mapper.readValue(
                    Fixture.read("health-inseason.json"), InferenceHealth.class);
            assertThat(parsed.dataAsOf()).isEqualTo(IN_SEASON_DATA_AS_OF);
        }

        @Test
        void aFixtureOnTheRuleDateScoresAllSevenMarketsAndIsNotStale() {
            InferenceResponse response = mapper.readValue(
                    Fixture.read("predict-inseason.json"), InferenceResponse.class);

            assertThat(response.dataAsOf()).isEqualTo(IN_SEASON_DATA_AS_OF);
            assertThat(response.stale()).isFalse();
            assertThat(response.daysBehind()).isLessThan(DAYS_BEHIND_FLOOR);

            InferenceResponse.Predictions p = response.predictions();
            assertThat(p).isNotNull();
            assertThat(p.homeWinProbability()).isEqualTo(0.5410619974136353);
            assertThat(p.homeMargin()).isEqualTo(0.8396134376525879);
            assertThat(p.totalPoints()).isEqualTo(221.41098022460938);
            assertThat(p.reboundMargin()).isEqualTo(-0.1728011518716812);
            assertThat(p.totalRebounds()).isEqualTo(90.6750717163086);
            assertThat(p.assistMargin()).isEqualTo(0.4007718563079834);
            assertThat(p.totalAssists()).isEqualTo(47.22257995605469);

            // A negative margin market, which the offseason fixture does not
            // have: every one of its seven values is positive, so a sign bug
            // on reboundMargin could not have been caught by it.
            assertThat(p.reboundMargin()).isNegative();
        }

        @Test
        @SuppressWarnings("unchecked")
        void theQuarterHalfModelsRefuseAnIncompleteWindowRatherThanImputing() {
            // A 400, not a prediction. Captured because it is only reachable
            // in the first days of a season and the offseason fixture beside
            // it is a successful response - so the refusal had never been
            // recorded in any form.
            Map<String, Object> body = mapper.readValue(
                    Fixture.read("predict-quarter-half-refused-inseason.json"),
                    new TypeReference<Map<String, Object>>() {
                    });

            assertThat(body).containsOnlyKeys("detail");
            String detail = (String) body.get("detail");

            // The message has to name the team, the shortfall and the reason,
            // because a refusal a caller cannot act on is as bad as a wrong
            // number. All three are asserted rather than the message as a
            // whole, which would pin the wording.
            assertThat(detail).contains(String.valueOf(ORL));
            assertThat(detail).contains("has played 0 game(s) with quarter data");
            assertThat(detail).contains("12 feature(s) would be NaN");
            assertThat(detail).contains("cannot score a partial row");
            assertThat(detail).contains("refuse the fixture rather than imputing");

            // And it must not read as a zero-valued prediction.
            assertThat(body).doesNotContainKeys("predictions", "data_as_of");
        }

        @Test
        void thePlayerPropsBoardIsEmptyThisEarlyAndSaysWhyRatherThanReturningZeroes() {
            InferencePlayerPropsResponse response = mapper.readValue(
                    Fixture.read("predict-player-props-inseason.json"),
                    InferencePlayerPropsResponse.class);

            assertThat(response.dataAsOf()).isEqualTo(IN_SEASON_DATA_AS_OF);
            assertThat(response.teams()).hasSize(2);

            // BOTH boards are empty two days into a season, and this is the
            // shape a browser gets today. It is NOT the populated-roster
            // shape: a player needs eleven appearances before availability
            // carries any weight, which is a mid-November measurement and a
            // separate backlog item. What is captured here is the honest
            // empty board, which had also never been under test.
            for (boolean home : List.of(true, false)) {
                InferencePlayerPropsResponse.TeamBoard board = response.board(home);
                assertThat(board.players()).isEmpty();
                assertThat(board.availabilityKnown()).isFalse();
                assertThat(board.availabilityNote())
                        .isEqualTo("no players with usable history");
            }

            assertThat(response.board(true).teamId()).isEqualTo(ORL);
            assertThat(response.board(false).teamId()).isEqualTo(ATL);

            // The note is a DIFFERENT sentence from the offseason one, which
            // says "AVAILABILITY UNKNOWN - no injury report was available".
            // Two distinct reasons for an unknown, and neither is a zero.
            assertThat(response.board(true).availabilityNote())
                    .doesNotContain("AVAILABILITY UNKNOWN");
        }

        @Test
        void theScheduleRunsWellPastTheRuleDateAndOnlyTheFirstDayIsPredictable() {
            List<InferenceScheduledGame> games = mapper.readValue(
                    Fixture.read("schedule-inseason.json"), new TypeReference<>() {
                    });

            assertThat(games).hasSize(118);

            // The service hands over the whole horizon; MAX_DAYS_AHEAD = 1
            // decides what is offerable, and that decision is the client's.
            // So most of this list is deliberately not predictable, and a
            // reader that assumed otherwise would offer 118 fixtures on a day
            // when the data supports eleven.
            assertThat(games).allSatisfy(game -> {
                assertThat(game.gameDate()).isAfterOrEqualTo(IN_SEASON_GAME_DATE);
                assertThat(game.homeTeamId()).isNotNull();
                assertThat(game.awayTeamId()).isNotNull();
            });

            long onTheRuleDate = games.stream()
                    .filter(game -> IN_SEASON_GAME_DATE.equals(game.gameDate()))
                    .count();
            assertThat(onTheRuleDate).isEqualTo(11);

            assertThat(games.stream().map(InferenceScheduledGame::gameDate).distinct().count())
                    .isEqualTo(14);
        }
    }
}
