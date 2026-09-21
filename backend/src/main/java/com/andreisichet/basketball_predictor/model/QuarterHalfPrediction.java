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

/** One set of Q1 / first-half model outputs for one game. */
@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
public class QuarterHalfPrediction {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "game_id")
    private Game game;

    private double q1Spread;

    private double q1Total;

    /** P(home leads Q1 | Q1 is not tied). */
    private double q1WinnerProbability;

    private double half1Spread;

    private double half1Total;

    /** P(home leads at half | the half is not tied). */
    private double half1WinnerProbability;

    private LocalDate dataAsOf;

    private boolean stale;

    private Instant predictedAt;
}
