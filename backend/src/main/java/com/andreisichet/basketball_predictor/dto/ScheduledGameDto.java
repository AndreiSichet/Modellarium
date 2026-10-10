package com.andreisichet.basketball_predictor.dto;

import java.time.Instant;
import java.time.LocalDate;

/** An upcoming fixture offered to the client as a candidate to predict. */
public record ScheduledGameDto(
        /**
         * Which league this fixture belongs to, lower-cased to match the
         * frontend's own league slugs ("nba", "wnba", "gleague").
         *
         * ADDED WHEN THE G LEAGUE BECAME THE FIRST NEW LEAGUE WITH FIXTURES
         * TO CACHE. This endpoint has never been league-scoped - it returns
         * every unplayed fixture in the window - and that was invisible while
         * the only other added league cached nothing: the WNBA's regular
         * season was over, so its sync wrote 0 rows and no NBA list could
         * ever contain one. The G League's 2026-27 season is already
         * scheduled, so 228 of its fixtures are in that list now.
         *
         * Additive on purpose rather than a ?league= parameter: the NBA's
         * response must keep every field a client already reads, and a
         * parameter with a default is one typo away from changing what the
         * NBA list contains.
         */
        String league,
        Long homeTeamId,
        String homeTeamAbbr,
        String homeTeamName,
        Long awayTeamId,
        String awayTeamAbbr,
        String awayTeamName,
        LocalDate gameDate,

        /**
         * Whether this fixture can be predicted now, or null for a league that
         * decides that from a date instead.
         *
         * THREE STATES, NOT TWO. The three basketball leagues send null here
         * and keep their date rule - one day after that league's cutoff, which
         * the client computes. The NFL's rule is a dependency measured in the
         * inference service (both teams' previous games in history) and cannot
         * be derived from any date, so the answer travels with the fixture
         * rather than being reimplemented in JavaScript where it would drift
         * from the code that refuses the request.
         *
         * STALENESS, STATED: the sync runs every six hours and at startup, so
         * a fixture that becomes predictable after the morning refresh can
         * appear up to six hours late. It cannot err the other way - a fixture
         * never stops being predictable before it is played - so the failure
         * mode is a late offer, never a rejected one.
         */
        Boolean predictable,

        /**
         * Kickoff in UTC, null when the time is not yet set. NFL only; the
         * client renders it in the viewer's own zone and shows the date alone
         * when this is null, rather than inventing a time.
         */
        Instant kickoffUtc,

        /** Whether the kickoff may still move. NFL only. */
        Boolean flex,

        /** The NFL week, for the Week N headings on the league page. */
        Integer week) {
}
