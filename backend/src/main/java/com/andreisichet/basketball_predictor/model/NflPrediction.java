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
 * One stored NFL prediction: three markets for one fixture.
 *
 * Its own table rather than columns on {@code prediction}, as the WNBA's and
 * the G League's are. The {@code Game} it hangs off IS reused, which is safe
 * because the four leagues' team-id ranges are disjoint - asserted at startup
 * rather than assumed.
 *
 * {@code week} is the one field here no sibling entity has. It varies per row,
 * it is a feature the models actually read, and an NFL result is discussed by
 * week rather than by date - so it is data rather than a constant about which
 * model produced a number. Season is recoverable from the game's date and is
 * deliberately not duplicated.
 *
 * What is NOT stored, by the rule in CLAUDE.md section 8: the winner
 * interpretation, the per-market model identifier and the manifest's
 * "TIES Elo alone" verdict. Each is identical on every row this table will
 * ever hold, so each is attached at response-build time from the live payload
 * rather than duplicated per row and left free to drift.
 */
@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
public class NflPrediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "game_id")
    private Game game;

    private double homeWinProbability;
    private double homeMargin;
    private double totalPoints;

    private int week;

    private LocalDate dataAsOf;
    private boolean stale;
    private Instant predictedAt;
}
