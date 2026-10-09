package com.andreisichet.basketball_predictor.repository;

import org.springframework.data.jpa.repository.JpaRepository;

import com.andreisichet.basketball_predictor.model.NflPrediction;

/** Plain JpaRepository; no custom finders until a GET endpoint needs one. */
public interface NflPredictionRepository
        extends JpaRepository<NflPrediction, Long> {
}
