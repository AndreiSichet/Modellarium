package com.andreisichet.basketball_predictor.dto;

import java.time.Instant;
import java.time.LocalDate;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonProperty;

/** One fixture as the Python service returns it from GET /schedule. */
public record InferenceScheduledGame(
        @JsonProperty("home_team_id") Long homeTeamId,
        @JsonProperty("away_team_id") Long awayTeamId,
        @JsonProperty("game_date") LocalDate gameDate,

        /**
         * Whether this fixture can be predicted right now. NFL only; the three
         * basketball leagues send nothing and this stays null.
         *
         * THE SERVER DECIDES, AND THAT IS THE WHOLE POINT. Basketball
         * predictability is a date rule - one day after that league's cutoff -
         * which a client can compute. The NFL's is a dependency: both teams'
         * previous games must be in history, which is measured in the
         * inference service and cannot be derived from a date. A second copy
         * of that rule in JavaScript would drift from the one that refuses
         * requests, so the answer travels with the fixture instead.
         */
        @JsonProperty("predictable") Boolean predictable,

        /**
         * Kickoff in UTC, or null for a fixture whose time is not yet set.
         * NFL only. The client renders it in the viewer's own zone; a null
         * shows the date alone rather than a guessed time.
         */
        @JsonProperty("kickoff_utc") Instant kickoffUtc,

        /** Whether the kickoff may still move. NFL only. */
        @JsonProperty("flex") Boolean flex,

        /** The NFL week this fixture belongs to, for the Week N headings. */
        @JsonProperty("week") Integer week,

        /**
         * Why this fixture is not predictable, as a stable code. Null when
         * it is predictable, and null for the three basketball leagues,
         * whose rule is a date the client computes.
         *
         * The inference row carries an English `reason` beside this. It is
         * deliberately NOT read: the sentence belongs to the reader and gets
         * reworded, and a client that parses it breaks silently when it is.
         */
        @JsonProperty("reason_code") String reasonCode,

        /**
         * Which sides block it and what each is waiting for.
         *
         * A LIST BECAUSE BOTH TEAMS USUALLY BLOCK - 1,074 of 1,083 NHL
         * fixtures on 2026-11-06. The code above comes from a rule that
         * returns on the first side it finds, which is right for a refusal
         * and half an answer on a schedule.
         */
        @JsonProperty("blocked_by") List<Blocker> blockedBy) {

    /**
     * One side that blocks a fixture. THE SAME KEYS AS THE REFUSAL BODY, so
     * a client that has handled one shape has handled both.
     */
    public record Blocker(
            @JsonProperty("side") String side,
            @JsonProperty("franchise_id") Long franchiseId,
            @JsonProperty("code") String code,
            @JsonProperty("pending") List<String> pending) {
    }
}
