package com.andreisichet.basketball_predictor.service;

import java.time.Instant;

import org.springframework.stereotype.Service;
import org.springframework.transaction.support.TransactionTemplate;

import com.andreisichet.basketball_predictor.dto.InferenceNflResponse;
import com.andreisichet.basketball_predictor.dto.InferenceRequest;
import com.andreisichet.basketball_predictor.dto.NflSummaryDto;
import com.andreisichet.basketball_predictor.dto.PredictionRequest;
import com.andreisichet.basketball_predictor.model.Game;
import com.andreisichet.basketball_predictor.model.NflPrediction;
import com.andreisichet.basketball_predictor.model.Team;
import com.andreisichet.basketball_predictor.repository.NflPredictionRepository;

/**
 * The NFL's three markets for one fixture.
 *
 * A sibling of the other three prediction services rather than a league
 * parameter on one of them, for the reason theirs record: the request bodies
 * match and the responses do not, so a union type would be mostly nulls
 * whichever way it was called.
 *
 * THE INFERENCE CALL HAPPENS BEFORE ANY WRITE, AND OUTSIDE THE TRANSACTION.
 * Before: a rejected request must not leave a Game row for a fixture that
 * never got a prediction, which was a real bug once. That matters more here
 * than for any other league, because the NFL refuses on a rule the caller
 * cannot easily pre-compute - a fixture whose teams have an earlier unplayed
 * game is refused, and 193 of 208 fixtures are in that state today. So most
 * requests a naive client makes will be rejected, and every one of them must
 * leave the database exactly as it was.
 *
 * Outside: the call is a synchronous HTTP round trip and holding a pooled
 * connection across it costs one connection per concurrent request. A
 * TransactionTemplate rather than an extracted @Transactional method, because
 * that annotation is proxy-based and calling it on {@code this} would silently
 * run with no transaction at all.
 */
@Service
public class NflPredictionService {
    private static final String WINNER = "winner";
    private static final String MARGIN = "margin";
    private static final String TOTAL = "total";

    private final GameLookup gameLookup;
    private final NflPredictionRepository predictionRepository;
    private final InferenceClient inferenceClient;
    private final TransactionTemplate transactionTemplate;

    public NflPredictionService(
            GameLookup gameLookup,
            NflPredictionRepository predictionRepository,
            InferenceClient inferenceClient,
            TransactionTemplate transactionTemplate) {
        this.gameLookup = gameLookup;
        this.predictionRepository = predictionRepository;
        this.inferenceClient = inferenceClient;
        this.transactionTemplate = transactionTemplate;
    }

    public NflSummaryDto predict(PredictionRequest request) {
        Team homeTeam = gameLookup.requireTeam(request.homeTeamId());
        Team awayTeam = gameLookup.requireTeam(request.awayTeamId());

        InferenceNflResponse inference = inferenceClient.predictNfl(
                new InferenceRequest(
                        request.homeTeamId(), request.awayTeamId(),
                        request.gameDate()));

        return transactionTemplate.execute(status -> {
            Game game = gameLookup.findOrCreateGame(
                    homeTeam, awayTeam, request.gameDate());
            NflPrediction saved =
                    predictionRepository.save(toEntity(game, inference));

            return new NflSummaryDto(
                    game.getId(),
                    homeTeam.getAbbreviation(),
                    awayTeam.getAbbreviation(),
                    game.getGameDate(),
                    NflSummaryDto.Prediction.of(
                            saved,
                            inference,
                            inference.market(WINNER),
                            inference.market(MARGIN),
                            inference.market(TOTAL)));
        });
    }

    private NflPrediction toEntity(Game game, InferenceNflResponse inference) {
        NflPrediction prediction = new NflPrediction();
        prediction.setGame(game);
        prediction.setHomeWinProbability(inference.market(WINNER).value());
        prediction.setHomeMargin(inference.market(MARGIN).value());
        prediction.setTotalPoints(inference.market(TOTAL).value());
        prediction.setWeek(inference.week());
        prediction.setDataAsOf(inference.dataAsOf());
        prediction.setStale(inference.stale());
        prediction.setPredictedAt(Instant.now());
        return prediction;
    }
}
