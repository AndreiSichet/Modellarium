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

/**
 * One stored G League prediction: three markets for one fixture.
 *
 * Its own table rather than columns on {@code prediction}, for the reason the
 * WNBA's records: three markets against the NBA's seven would mean four
 * always-null columns and nothing in the row saying which league it was. The
 * {@code Game} it hangs off IS reused, which is safe because the three
 * leagues' team-id ranges are disjoint - asserted at startup rather than
 * assumed.
 *
 * What is deliberately NOT stored: the per-market window. It is identical on
 * every row this table will ever hold - all three markets ship CARRY10 - so
 * by the rule in CLAUDE.md section 8 it is a constant about which model
 * produced a number, attached at response-build time rather than duplicated
 * per row and left free to drift.
 */
@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
public class GleaguePrediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "game_id")
    private Game game;

    private double homeWinProbability;
    private double homeMargin;
    private double totalPoints;

    private LocalDate dataAsOf;
    private boolean stale;
    private Instant predictedAt;
}
