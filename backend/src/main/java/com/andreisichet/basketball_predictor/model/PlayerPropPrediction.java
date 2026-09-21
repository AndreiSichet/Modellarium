package com.andreisichet.basketball_predictor.model;

import java.time.Instant;

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

/** One player's five prop predictions for one game. */
@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
public class PlayerPropPrediction {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "game_id")
    private Game game;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "player_id")
    private Player player;

    /** The side this player was on for this prediction. */
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "team_id")
    private Team team;

    private double predictedPoints;

    private double predictedRebounds;

    private double predictedAssists;

    private double predictedThreesMade;

    /** Points + rebounds + assists, the common combined market. */
    private double predictedPra;

    /** "linear" or "xgb" - which half of the hybrid answered. */
    private String modelUsed;

    private Instant predictedAt;
}
