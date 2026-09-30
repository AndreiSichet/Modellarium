package com.andreisichet.basketball_predictor.service;

import java.time.Instant;

import org.springframework.stereotype.Service;
import org.springframework.transaction.support.TransactionTemplate;

import com.andreisichet.basketball_predictor.dto.GameSummaryDto;
import com.andreisichet.basketball_predictor.dto.InferenceRequest;
import com.andreisichet.basketball_predictor.dto.InferenceResponse;
import com.andreisichet.basketball_predictor.dto.PredictionDto;
import com.andreisichet.basketball_predictor.dto.PredictionRequest;
import com.andreisichet.basketball_predictor.model.Game;
import com.andreisichet.basketball_predictor.model.Prediction;
import com.andreisichet.basketball_predictor.model.Team;
import com.andreisichet.basketball_predictor.repository.PredictionRepository;

/** The seven full-game models. */
@Service
public class PredictionService {
    private final GameLookup gameLookup;
    private final PredictionRepository predictionRepository;
    private final InferenceClient inferenceClient;
    private final TransactionTemplate transactionTemplate;

    public PredictionService(
            GameLookup gameLookup,
            PredictionRepository predictionRepository,
            InferenceClient inferenceClient,
            TransactionTemplate transactionTemplate) {
        this.gameLookup = gameLookup;
        this.predictionRepository = predictionRepository;
        this.inferenceClient = inferenceClient;
        this.transactionTemplate = transactionTemplate;
    }

    /**
     * The inference call runs OUTSIDE any transaction, and the writes inside a
     * short one.
     *
     * Ordering is load-bearing and unchanged: inference first, so a request the
     * service rejects leaves no orphan Game row. What changed is that a pooled
     * database connection is no longer held across the HTTP round trip - which
     * costs nothing locally and costs a connection per concurrent request once
     * the inference service is remote, slow, or timing out.
     *
     * TransactionTemplate rather than an extracted @Transactional method: that
     * annotation is proxy-based, so calling it on `this` would bypass the proxy
     * and silently run with no transaction at all.
     */
    public GameSummaryDto predict(PredictionRequest request) {
        Team homeTeam = gameLookup.requireTeam(request.homeTeamId());
        Team awayTeam = gameLookup.requireTeam(request.awayTeamId());

        // Inference BEFORE any write: a rejected request must not leave an
        // orphan game row behind.
        InferenceResponse inference = inferenceClient.predict(
                new InferenceRequest(
                        request.homeTeamId(), request.awayTeamId(), request.gameDate()));

        return transactionTemplate.execute(status -> {
            Game game = gameLookup.findOrCreateGame(
                    homeTeam, awayTeam, request.gameDate());
            Prediction prediction =
                    predictionRepository.save(toPrediction(game, inference));

            return new GameSummaryDto(
                    game.getId(),
                    homeTeam.getAbbreviation(),
                    awayTeam.getAbbreviation(),
                    game.getGameDate(),
                    game.isPlayed(),
                    PredictionDto.from(prediction));
        });
    }

    private Prediction toPrediction(Game game, InferenceResponse inference) {
        InferenceResponse.Predictions values = inference.predictions();

        Prediction prediction = new Prediction();
        prediction.setGame(game);
        prediction.setHomeWinProbability(values.homeWinProbability());
        prediction.setHomeMargin(values.homeMargin());
        prediction.setTotalPoints(values.totalPoints());
        prediction.setReboundMargin(values.reboundMargin());
        prediction.setTotalRebounds(values.totalRebounds());
        prediction.setAssistMargin(values.assistMargin());
        prediction.setTotalAssists(values.totalAssists());
        prediction.setDataAsOf(inference.dataAsOf());
        prediction.setStale(inference.stale());
        prediction.setPredictedAt(Instant.now());
        return prediction;
    }
}
