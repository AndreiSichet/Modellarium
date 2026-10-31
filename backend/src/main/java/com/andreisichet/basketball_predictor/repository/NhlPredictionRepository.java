package com.andreisichet.basketball_predictor.repository;

import org.springframework.data.jpa.repository.JpaRepository;

import com.andreisichet.basketball_predictor.model.NhlPrediction;

public interface NhlPredictionRepository
        extends JpaRepository<NhlPrediction, Long> {
}
