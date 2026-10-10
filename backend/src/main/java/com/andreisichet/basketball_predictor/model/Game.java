package com.andreisichet.basketball_predictor.model;

import java.time.Instant;
import java.time.LocalDate;

import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/** One game, upcoming or completed. */
@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Game {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    /** Real nba_api GAME_ID, null until the game has one. */
    private Long nbaGameId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "home_team_id")
    private Team homeTeam;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "away_team_id")
    private Team awayTeam;

    private LocalDate gameDate;

    private boolean played;

    /**
     * Whether this fixture can be predicted now. NFL only; null for the three
     * basketball leagues, which decide it from a date instead.
     *
     * NULLABLE AND AN OBJECT TYPE ON PURPOSE. A primitive would map to NOT
     * NULL (§38's observation that every nullable column here is an object
     * type and every NOT NULL one a primitive), and "false" is not what a
     * basketball fixture means - it has no server-side answer at all, which is
     * a third state the client must be able to see.
     */
    private Boolean predictable;

    /** Kickoff in UTC, null when the time is not yet set. NFL only. */
    private Instant kickoffUtc;

    /** Whether the kickoff may still move. NFL only. */
    private Boolean flex;

    /** The NFL week, for the Week N headings on the league page. */
    private Integer week;
}
