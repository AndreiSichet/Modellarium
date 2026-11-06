package com.andreisichet.basketball_predictor;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

/**
 * The guard that keeps the suite off the production database.
 *
 * NO CONTEXT AND NO CONNECTION. Everything here is a string, deliberately:
 * proving the refusal by pointing a real context at the production database
 * would mean doing the thing the refusal exists to prevent, and if the guard
 * were broken the proof would be the damage.
 */
class TestDatabaseGuardTest {

    @Test
    void theProductionDatabaseIsRefusedBeforeAnythingConnects() {
        assertThatThrownBy(() -> TestDatabase.refuse(
                "jdbc:postgresql://localhost:5432/basketball_predictor"))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("REFUSING")
                // THE MESSAGE HAS TO SAY WHAT TO DO. A refusal that only
                // says no sends the reader to work out the alternative, and
                // the alternative is one command.
                .hasMessageContaining("basketball_predictor_test")
                .hasMessageContaining("CREATE DATABASE");
    }

    @Test
    void parametersDoNotDisguiseIt() {
        // `.../basketball_predictor?ssl=true` is still production, and a
        // naive endsWith check would have let it through.
        assertThatThrownBy(() -> TestDatabase.refuse(
                "jdbc:postgresql://localhost:5432/basketball_predictor?ssl=true"))
                .isInstanceOf(IllegalStateException.class);
    }

    @Test
    void theTestDatabaseIsAllowedWhereverItLives() {
        // THE NAME IS THE RULE, NOT THE HOST. CI's service container and the
        // local compose container are both localhost:5432, so a host-based
        // rule could not tell them apart - which is exactly why CI's
        // database was renamed rather than special-cased.
        assertThatCode(() -> TestDatabase.refuse(
                "jdbc:postgresql://localhost:5432/basketball_predictor_test"))
                .doesNotThrowAnyException();
        assertThatCode(() -> TestDatabase.refuse(
                "jdbc:postgresql://postgres:5432/basketball_predictor_test"))
                .doesNotThrowAnyException();
    }

    @Test
    void aDatabaseWhoseNameMerelyContainsTheProductionOneIsAllowed() {
        // `basketball_predictor_test` CONTAINS `basketball_predictor`, so a
        // substring test would refuse the very database the tests are
        // supposed to use. This is the control for that mistake.
        assertThat(TestDatabase.databaseName(
                "jdbc:postgresql://localhost:5432/basketball_predictor_test"))
                .isEqualTo("basketball_predictor_test")
                .isNotEqualTo(TestDatabase.PRODUCTION);
    }

    @Test
    void theUrlTheSuiteActuallyUsesIsNotTheProductionOne() {
        // THE REAL CALL, not a constructed string: this is what
        // @DynamicPropertySource hands Spring, so if the defaults ever
        // changed to the production name this fails here rather than in a
        // migration.
        String url = TestDatabase.url();
        assertThat(TestDatabase.databaseName(url))
                .isNotEqualTo(TestDatabase.PRODUCTION);
        assertThat(url).startsWith("jdbc:postgresql://");
    }

    @Test
    void theGuardReadsTheNameAndNotTheWholeUrl() {
        assertThat(TestDatabase.databaseName(
                "jdbc:postgresql://host:5432/db?a=1&b=2")).isEqualTo("db");
        assertThat(TestDatabase.databaseName("jdbc:postgresql://host:5432/"))
                .isNull();
        assertThat(TestDatabase.databaseName(null)).isNull();
    }
}
