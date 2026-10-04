package com.andreisichet.basketball_predictor.service;

import java.time.Instant;

import org.springframework.stereotype.Service;
import org.springframework.transaction.support.TransactionTemplate;

import com.andreisichet.basketball_predictor.dto.InferenceRequest;
import com.andreisichet.basketball_predictor.dto.InferenceWnbaResponse;
import com.andreisichet.basketball_predictor.dto.PredictionRequest;
import com.andreisichet.basketball_predictor.dto.WnbaSummaryDto;
import com.andreisichet.basketball_predictor.model.Game;
import com.andreisichet.basketball_predictor.model.Team;
import com.andreisichet.basketball_predictor.model.WnbaPrediction;
import com.andreisichet.basketball_predictor.repository.WnbaPredictionRepository;

/** The three WNBA markets. */
@Service
public class WnbaPredictionService {
    private static final String MONEYLINE = "moneyline";
    private static final String SPREAD = "spread";
    private static final String TOTALS = "totals";

    private final GameLookup gameLookup;
    private final WnbaPredictionRepository predictionRepository;
    private final InferenceClient inferenceClient;
    private final TransactionTemplate transactionTemplate;

    public WnbaPredictionService(
            GameLookup gameLookup,
            WnbaPredictionRepository predictionRepository,
            InferenceClient inferenceClient,
            TransactionTemplate transactionTemplate) {
        this.gameLookup = gameLookup;
        this.predictionRepository = predictionRepository;
        this.inferenceClient = inferenceClient;
        this.transactionTemplate = transactionTemplate;
    }

    /**
     * The inference call runs OUTSIDE any transaction, and the writes inside a
     * short one - the same shape as the other three prediction services.
     *
     * Ordering is load-bearing: inference first, so a request the Python side
     * rejects leaves no orphan Game row. That matters more here than for the
     * NBA, because the WNBA models cannot score an incomplete feature row at
     * all and so reject a larger share of requests - a team early in its
     * season has no window yet, and the service answers 400 rather than
     * returning a number.
     *
     * TransactionTemplate rather than an extracted @Transactional method: that
     * annotation is proxy-based, so calling it on `this` would bypass the proxy
     * and silently run with no transaction at all.
     */
    public WnbaSummaryDto predict(PredictionRequest request) {
        Team homeTeam = gameLookup.requireTeam(request.homeTeamId());
        Team awayTeam = gameLookup.requireTeam(request.awayTeamId());

        InferenceWnbaResponse inference = inferenceClient.predictWnba(
                new InferenceRequest(
                        request.homeTeamId(), request.awayTeamId(), request.gameDate()));

        return transactionTemplate.execute(status -> {
            Game game = gameLookup.findOrCreateGame(
                    homeTeam, awayTeam, request.gameDate());
            WnbaPrediction saved = predictionRepository.save(toEntity(game, inference));

            return new WnbaSummaryDto(
                    game.getId(),
                    homeTeam.getAbbreviation(),
                    awayTeam.getAbbreviation(),
                    game.getGameDate(),
                    WnbaSummaryDto.Prediction.of(
                            saved,
                            inference,
                            inference.market(MONEYLINE),
                            inference.market(SPREAD),
                            inference.market(TOTALS)));
        });
    }

    private WnbaPrediction toEntity(Game game, InferenceWnbaResponse inference) {
        WnbaPrediction prediction = new WnbaPrediction();
        prediction.setGame(game);
        prediction.setHomeWinProbability(inference.market(MONEYLINE).value());
        prediction.setHomeMargin(inference.market(SPREAD).value());
        prediction.setTotalPoints(inference.market(TOTALS).value());
        prediction.setDataAsOf(inference.dataAsOf());
        prediction.setStale(inference.stale());
        prediction.setPredictedAt(Instant.now());
        return prediction;
    }
}
