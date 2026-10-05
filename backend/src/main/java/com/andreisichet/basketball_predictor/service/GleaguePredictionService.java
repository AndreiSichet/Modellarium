package com.andreisichet.basketball_predictor.service;

import java.time.Instant;

import org.springframework.stereotype.Service;
import org.springframework.transaction.support.TransactionTemplate;

import com.andreisichet.basketball_predictor.dto.InferenceRequest;
import com.andreisichet.basketball_predictor.dto.InferenceGleagueResponse;
import com.andreisichet.basketball_predictor.dto.PredictionRequest;
import com.andreisichet.basketball_predictor.dto.GleagueSummaryDto;
import com.andreisichet.basketball_predictor.model.Game;
import com.andreisichet.basketball_predictor.model.Team;
import com.andreisichet.basketball_predictor.model.GleaguePrediction;
import com.andreisichet.basketball_predictor.repository.GleaguePredictionRepository;

/** The three G League markets. */
/**
 * The G League's three markets for one fixture.
 *
 * A sibling of the NBA and WNBA services rather than a league parameter on
 * one of them. The request bodies match and the responses do not, so a union
 * type would be mostly nulls whichever way it was called - and all three
 * share the same validate-then-call-then-write ordering, which is what keeps
 * the anti-orphan rule in one place.
 *
 * THE INFERENCE CALL HAPPENS BEFORE ANY WRITE, AND OUTSIDE THE TRANSACTION.
 * Before: a rejected request must not leave a Game row for a fixture that
 * never got a prediction, which was a real bug once. Outside: the call is a
 * synchronous HTTP round trip and holding a pooled connection across it costs
 * one connection per concurrent request - and a TransactionTemplate rather
 * than an extracted @Transactional method, because that annotation is
 * proxy-based and calling it on `this` would silently run with no transaction
 * at all.
 */
@Service
public class GleaguePredictionService {
    private static final String MONEYLINE = "moneyline";
    private static final String SPREAD = "spread";
    private static final String TOTALS = "totals";

    private final GameLookup gameLookup;
    private final GleaguePredictionRepository predictionRepository;
    private final InferenceClient inferenceClient;
    private final TransactionTemplate transactionTemplate;

    public GleaguePredictionService(
            GameLookup gameLookup,
            GleaguePredictionRepository predictionRepository,
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
     * NBA, because the G League models cannot score an incomplete feature row at
     * all and so reject a larger share of requests - a team early in its
     * season has no window yet, and the service answers 400 rather than
     * returning a number.
     *
     * TransactionTemplate rather than an extracted @Transactional method: that
     * annotation is proxy-based, so calling it on `this` would bypass the proxy
     * and silently run with no transaction at all.
     */
    public GleagueSummaryDto predict(PredictionRequest request) {
        Team homeTeam = gameLookup.requireTeam(request.homeTeamId());
        Team awayTeam = gameLookup.requireTeam(request.awayTeamId());

        InferenceGleagueResponse inference = inferenceClient.predictGleague(
                new InferenceRequest(
                        request.homeTeamId(), request.awayTeamId(), request.gameDate()));

        return transactionTemplate.execute(status -> {
            Game game = gameLookup.findOrCreateGame(
                    homeTeam, awayTeam, request.gameDate());
            GleaguePrediction saved = predictionRepository.save(toEntity(game, inference));

            return new GleagueSummaryDto(
                    game.getId(),
                    homeTeam.getAbbreviation(),
                    awayTeam.getAbbreviation(),
                    game.getGameDate(),
                    GleagueSummaryDto.Prediction.of(
                            saved,
                            inference,
                            inference.market(MONEYLINE),
                            inference.market(SPREAD),
                            inference.market(TOTALS)));
        });
    }

    private GleaguePrediction toEntity(Game game, InferenceGleagueResponse inference) {
        GleaguePrediction prediction = new GleaguePrediction();
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
