package com.andreisichet.basketball_predictor.service;

import org.springframework.stereotype.Service;
import org.springframework.transaction.support.TransactionTemplate;

import com.andreisichet.basketball_predictor.dto.InferenceNhlResponse;
import com.andreisichet.basketball_predictor.dto.InferenceRequest;
import com.andreisichet.basketball_predictor.dto.NhlSummaryDto;
import com.andreisichet.basketball_predictor.dto.PredictionRequest;
import com.andreisichet.basketball_predictor.model.Game;
import com.andreisichet.basketball_predictor.model.NhlPrediction;
import com.andreisichet.basketball_predictor.model.Team;
import com.andreisichet.basketball_predictor.repository.NhlPredictionRepository;

/**
 * Stores and returns one NHL prediction.
 *
 * <p>THE INFERENCE CALL HAPPENS BEFORE THE TRANSACTION OPENS, which is the
 * shape every league here follows: a slow or failing upstream must not hold
 * a database transaction open, and a refusal must not leave a half-written
 * row behind.
 *
 * <p>A MARKET THAT IS NOT SERVED IS NOT AN ERROR. Totals come back with
 * {@code served: false} and a reason code, and this service passes that
 * through rather than treating a missing value as a fault - the NHL serves
 * two markets by decision, not by accident.
 */
@Service
public class NhlPredictionService {

    private static final String WINNER = "winner";
    private static final String PUCK_LINE = "puck_line";
    private static final String TOTALS = "totals";

    private final GameLookup gameLookup;
    private final NhlPredictionRepository predictionRepository;
    private final InferenceClient inferenceClient;
    private final TransactionTemplate transactionTemplate;

    public NhlPredictionService(
            GameLookup gameLookup,
            NhlPredictionRepository predictionRepository,
            InferenceClient inferenceClient,
            TransactionTemplate transactionTemplate) {
        this.gameLookup = gameLookup;
        this.predictionRepository = predictionRepository;
        this.inferenceClient = inferenceClient;
        this.transactionTemplate = transactionTemplate;
    }

    public NhlSummaryDto predict(PredictionRequest request) {
        Team homeTeam = gameLookup.requireTeam(request.homeTeamId());
        Team awayTeam = gameLookup.requireTeam(request.awayTeamId());

        InferenceNhlResponse inference = inferenceClient.predictNhl(
                new InferenceRequest(
                        request.homeTeamId(), request.awayTeamId(),
                        request.gameDate()));

        InferenceNhlResponse.Market winner = market(inference, WINNER);
        InferenceNhlResponse.Market puckLine = market(inference, PUCK_LINE);
        InferenceNhlResponse.Market totals = market(inference, TOTALS);

        return transactionTemplate.execute(status -> {
            Game game = gameLookup.findOrCreateGame(
                    homeTeam, awayTeam, request.gameDate());
            NhlPrediction saved = predictionRepository.save(
                    toEntity(game, inference, winner, puckLine));
            return new NhlSummaryDto(
                    game.getId(),
                    homeTeam.getAbbreviation(),
                    awayTeam.getAbbreviation(),
                    game.getGameDate(),
                    NhlSummaryDto.Prediction.of(
                            saved, inference, winner, puckLine, totals));
        });
    }

    private static InferenceNhlResponse.Market market(
            InferenceNhlResponse inference, String name) {
        return inference.markets() == null
                ? null : inference.markets().get(name);
    }

    /**
     * AN UNSERVED MARKET'S VALUE IS NEVER STORED AS A ZERO. Both stored
     * columns are NOT NULL primitives, so a market the service declined to
     * serve cannot be written as 0.0 and read back later as a prediction -
     * it raises instead. That is the unknown-is-never-a-zero rule at the
     * one boundary where a double would have silently accepted it.
     */
    private static NhlPrediction toEntity(
            Game game, InferenceNhlResponse inference,
            InferenceNhlResponse.Market winner,
            InferenceNhlResponse.Market puckLine) {
        requireServed(WINNER, winner);
        requireServed(PUCK_LINE, puckLine);
        NhlPrediction prediction = new NhlPrediction();
        prediction.setGame(game);
        prediction.setHomeWinProbability(winner.value());
        prediction.setHomeWinsByTwoOrMore(puckLine.value());
        prediction.setDataAsOf(inference.dataAsOf());
        prediction.setStale(inference.stale());
        return prediction;
    }

    private static void requireServed(
            String name, InferenceNhlResponse.Market market) {
        if (market == null || !market.served() || market.value() == null) {
            throw new IllegalStateException(
                    "the inference service did not serve the NHL " + name
                    + " market, so there is no value to store. A stored 0.0"
                    + " would read back as a prediction.");
        }
    }
}
