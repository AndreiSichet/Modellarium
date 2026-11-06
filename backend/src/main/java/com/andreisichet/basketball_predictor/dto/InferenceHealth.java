package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonProperty;

/** Response from the Python service's GET /health. */
public record InferenceHealth(
        String status,
        @JsonProperty("models_loaded") Map<String, Integer> modelsLoaded,
        @JsonProperty("data_as_of") LocalDate dataAsOf,
        @JsonProperty("days_behind") int daysBehind,
        boolean stale,
        LeagueFreshness wnba,
        LeagueFreshness gleague,
        NflFreshness nfl,
        NhlFreshness nhl) {
    /**
     * The WNBA's own cutoff, which is a different date from the NBA's.
     *
     * Declared rather than ignored because the frontend decides which
     * fixtures are predictable from dataAsOf, and MAX_DAYS_AHEAD is one day.
     * Reading the NBA's cutoff for a WNBA fixture would disable every WNBA
     * game whenever the two leagues' data ended on different dates - which is
     * the normal case, since their seasons barely overlap.
     *
     * Nullable on purpose: an older inference service does not send it, and
     * Jackson ignores unknown properties rather than failing, so a backend
     * deployed ahead of the Python side gets null here instead of a 500.
     */
    public record LeagueFreshness(
            @JsonProperty("data_as_of") LocalDate dataAsOf,
            @JsonProperty("days_behind") int daysBehind,
            boolean stale) {
    }

    /**
     * The NFL's freshness, plus the two fields no other league has.
     *
     * THIS BLOCK HAS BEEN SILENTLY DROPPED TWICE, which is why it is declared
     * in the same change that adds it on the Python side and why the
     * regression gate names added fields per body. The WNBA's block was lost
     * here first, then the G League's; both times the Python service was
     * sending it and these records were discarding it, and both times the
     * gate's "fields added" line is what showed it rather than anything
     * failing.
     *
     * {@code predictableFixtures} is the field that makes this league
     * different. The other three are served under MAX_DAYS_AHEAD = 1, so a
     * client can compute what is predictable from the cutoff: cutoff plus one
     * day. The NFL's rule is a DEPENDENCY - both teams' previous games in
     * history - so it cannot be computed from a date at all, and a client
     * that guessed cutoff plus one would be wrong about every fixture. The
     * count comes from the service that knows.
     *
     * {@code synthetic} is true only in CI, where the real CC BY-SA tables
     * cannot exist and an invented set is served so the container boots.
     */
    /**
     * The NHL's block, and it is the first that can say IT IS NOT SERVED.
     *
     * <p>The NHL's five tables are OPTIONAL in a snapshot, so a service
     * booted on one built before phase 3 answers {@code served: false} with
     * {@code reason_code: not_yet_served} and nothing else. That is a state,
     * not a fault - it is what lets a new image boot on an old snapshot - so
     * every other field here is an object type and may be absent.
     *
     * <p>DECLARES FEWER FIELDS THAN THE WIRE SENDS, deliberately, exactly as
     * {@code NflFreshness} does. The wire also carries the quarantine, the
     * standings class, the per-market served flags, the blocked-fixture
     * counts and the winner's full caveat. Jackson's tolerance for unknown
     * properties is proven by the running system rather than only by this
     * comment; what the backend forwards is what a client has a use for
     * today, and phase 4's frontend is what will want the caveat's reader
     * form.
     */
    public record NhlFreshness(
            boolean served,
            @JsonProperty("reason_code") String reasonCode,
            @JsonProperty("data_as_of") LocalDate dataAsOf,
            @JsonProperty("days_behind") Integer daysBehind,
            Boolean stale,
            @JsonProperty("predictable_fixtures") Integer predictableFixtures,
            Integer fixtures,
            @JsonProperty("prediction_rule") String predictionRule,
            String source,

            /**
             * The winner market's block, read ONLY for its caveat.
             *
             * The caveat is not decoration. The NHL's winner is Elo alone
             * and its test interval crosses zero, so every surface showing
             * the probability has to show the caveat with it - and the
             * words have to come from here rather than from a copy in the
             * frontend, which would go on saying something the model no
             * longer supports.
             */
            Winner winner) {
    }

    /** The winner market, as /health reports it. */
    public record Winner(
            String family,
            String artifact,
            Caveat caveat) {
    }

    /**
     * Three strings for three audiences, and the short one carries no
     * figure by design: a caveat with a number in it expires at the next
     * retrain while still looking authoritative.
     */
    public record Caveat(
            /**
             * `short` is a Java keyword, so the component cannot be named
             * after the JSON field and the mapping is explicit. The
             * annotation goes on the COMPONENT, not on an accessor: Jackson
             * binds a record through its canonical constructor, so an
             * annotated getter would be read on the way out and ignored on
             * the way in.
             */
            @JsonProperty("short") String shortText,
            String reader,
            @JsonProperty("confirmed_on_test") Boolean confirmedOnTest) {
    }

    public record NflFreshness(
            @JsonProperty("data_as_of") LocalDate dataAsOf,
            @JsonProperty("days_behind") int daysBehind,
            boolean stale,
            @JsonProperty("predictable_fixtures") int predictableFixtures,
            int fixtures,
            @JsonProperty("prediction_rule") String predictionRule,
            String source,
            boolean synthetic) {
    }
}
