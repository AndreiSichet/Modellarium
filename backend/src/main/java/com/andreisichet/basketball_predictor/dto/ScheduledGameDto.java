package com.andreisichet.basketball_predictor.dto;

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
        LocalDate gameDate) {
}
