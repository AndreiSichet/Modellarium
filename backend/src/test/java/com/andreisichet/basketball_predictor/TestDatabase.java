package com.andreisichet.basketball_predictor;

/**
 * The database the Spring tests are allowed to touch, and the refusal that
 * keeps them off the production one.
 *
 * WHAT THESE TESTS DO TO A DATABASE, which is why this exists at all. Two
 * classes load a full application context - {@code
 * BasketballPredictorApplicationTests} and {@code
 * ScheduleServiceQueryCountTest} - and on context load:
 *
 * <ul>
 *   <li><b>MIGRATE.</b> Flyway runs every pending migration, with
 *       {@code baseline-on-migrate=true}. A migration cannot be un-applied.</li>
 *   <li><b>INSERT.</b> The query-count test seeds fixtures - 40, then nine
 *       times whatever the first measurement saw, so up to a few hundred
 *       {@code game} rows.</li>
 *   <li><b>DELETE.</b> Its {@code @AfterEach} removes those rows BY ID.
 *       Nothing is truncated and nothing else is touched - but a run killed
 *       between the insert and the {@code @AfterEach} leaves them, which is
 *       this project's recorded "an interrupted run does not execute its
 *       finally" lesson.</li>
 * </ul>
 *
 * {@code ddl-auto=validate}, so Hibernate changes no schema itself.
 *
 * THE HAZARD WAS NOT HYPOTHETICAL. {@code application.properties} points at
 * {@code jdbc:postgresql://${DB_HOST:localhost}:5432/basketball_predictor} -
 * the production database name, defaulting to localhost, which is the compose
 * container the running app uses. With {@code DB_PASSWORD} set, a local
 * {@code mvn verify} migrates and writes to the live application database.
 * On 6 November it only failed to because the password was absent and the
 * connection never opened.
 *
 * AND THE URL ALONE COULD NOT TELL CI APART, which is the part worth
 * recording: CI's service container was ALSO {@code basketball_predictor} on
 * {@code localhost:5432}, so "refuse the production URL" would have refused
 * CI as well. The fix is therefore a NAME, not a heuristic - the test
 * database is {@code basketball_predictor_test} everywhere, CI's service
 * container included, and the production name is refused outright. A guess
 * about which localhost is which would have been the kind of silent fallback
 * this project has twice had to remove.
 */
public final class TestDatabase {

    /** The production database, which these tests must never reach. */
    public static final String PRODUCTION = "basketball_predictor";

    /** Their own, which CI's service container is also named. */
    public static final String TEST = "basketball_predictor_test";

    private TestDatabase() {
    }

    /**
     * The JDBC url the tests use, refusing the production database.
     *
     * Supplied through {@code @DynamicPropertySource} rather than asserted
     * in a {@code @BeforeAll}, because this has to run BEFORE Flyway: the
     * migration happens while the context refreshes, and a check that runs
     * after the context is built is a check that runs after the damage.
     */
    public static String url() {
        String host = System.getenv().getOrDefault("DB_HOST", "localhost");
        String port = System.getenv().getOrDefault("DB_PORT", "5432");
        String name = System.getenv().getOrDefault("TEST_DB_NAME", TEST);
        return refuse(String.format(
                "jdbc:postgresql://%s:%s/%s", host, port, name));
    }

    /**
     * The url, or an exception naming what to do instead.
     *
     * MATCHED ON THE DATABASE NAME, not on the host. "localhost" says
     * nothing: the production container and CI's are both there. What
     * distinguishes them is which database the url ends in, which is a fact
     * rather than an inference.
     */
    public static String refuse(String url) {
        String name = databaseName(url);
        if (PRODUCTION.equals(name)) {
            throw new IllegalStateException(
                    "REFUSING TO RUN THE TEST SUITE AGAINST THE PRODUCTION "
                    + "DATABASE.\n"
                    + "  url: " + url + "\n"
                    + "  These tests run Flyway migrations and insert "
                    + "fixtures. A migration cannot be un-applied.\n"
                    + "  Use " + TEST + " instead:\n"
                    + "    docker exec -e PGPASSWORD=postgres "
                    + "basketball-predictor-postgres-1 \\\n"
                    + "      psql -U postgres -c 'CREATE DATABASE "
                    + TEST + "'\n"
                    + "  CI's service container is already named " + TEST
                    + "; set TEST_DB_NAME to override.");
        }
        return url;
    }

    /**
     * The database a postgres jdbc url names, or null.
     *
     * Anything after a '?' is a parameter list, not part of the name - so
     * {@code .../basketball_predictor?ssl=true} is still production.
     */
    public static String databaseName(String url) {
        if (url == null) {
            return null;
        }
        int slash = url.lastIndexOf('/');
        if (slash < 0 || slash == url.length() - 1) {
            return null;
        }
        String tail = url.substring(slash + 1);
        int question = tail.indexOf('?');
        return question < 0 ? tail : tail.substring(0, question);
    }
}
