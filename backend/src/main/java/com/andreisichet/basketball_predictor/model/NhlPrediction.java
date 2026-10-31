package com.andreisichet.basketball_predictor.model;

import java.time.Instant;
import java.time.LocalDate;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.PrePersist;
import jakarta.persistence.Table;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * One stored NHL prediction: two markets, the cutoff and the staleness.
 *
 * <p>TWO MARKETS AND NO TOTALS COLUMN. Totals are not served for the NHL -
 * a constant predictor beat the fitted model on test - so the column does
 * not exist rather than existing and being null. A nullable third market
 * would invite someone to fill it without re-reading why it was empty.
 *
 * <p>THE CAVEAT IS NOT STORED, AND THAT IS THE SAME RULE V4 APPLIED. The
 * winner's test interval against a constant includes zero, so every response
 * carries a caveat saying the gain is not established on test - but that
 * sentence is identical on every row this table will ever hold, so it is a
 * constant about the model and is attached when the response is built.
 */
@Entity
@Table(name = "nhl_prediction")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class NhlPrediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "game_id")
    private Game game;

    @Column(name = "home_win_probability", nullable = false)
    private double homeWinProbability;

    @Column(name = "home_wins_by_two_or_more", nullable = false)
    private double homeWinsByTwoOrMore;

    @Column(name = "data_as_of")
    private LocalDate dataAsOf;

    @Column(nullable = false)
    private boolean stale;

    @Column(name = "predicted_at")
    private Instant predictedAt;

    @PrePersist
    void stampPredictedAt() {
        if (predictedAt == null) {
            predictedAt = Instant.now();
        }
    }
}
