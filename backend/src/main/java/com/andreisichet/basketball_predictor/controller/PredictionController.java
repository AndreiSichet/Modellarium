package com.andreisichet.basketball_predictor.controller;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.andreisichet.basketball_predictor.dto.GameSummaryDto;
import com.andreisichet.basketball_predictor.dto.PlayerPropsResponseDto;
import com.andreisichet.basketball_predictor.dto.PredictionRequest;
import com.andreisichet.basketball_predictor.dto.QuarterHalfSummaryDto;
import com.andreisichet.basketball_predictor.dto.WnbaSummaryDto;
import com.andreisichet.basketball_predictor.service.PlayerPropPredictionService;
import com.andreisichet.basketball_predictor.service.PredictionService;
import com.andreisichet.basketball_predictor.service.QuarterHalfPredictionService;
import com.andreisichet.basketball_predictor.service.WnbaPredictionService;

@RestController
@RequestMapping("/api/predictions")
public class PredictionController {
    private final PredictionService predictionService;
    private final QuarterHalfPredictionService quarterHalfPredictionService;
    private final PlayerPropPredictionService playerPropPredictionService;
    private final WnbaPredictionService wnbaPredictionService;

    public PredictionController(
            PredictionService predictionService,
            QuarterHalfPredictionService quarterHalfPredictionService,
            PlayerPropPredictionService playerPropPredictionService,
            WnbaPredictionService wnbaPredictionService) {
        this.predictionService = predictionService;
        this.quarterHalfPredictionService = quarterHalfPredictionService;
        this.playerPropPredictionService = playerPropPredictionService;
        this.wnbaPredictionService = wnbaPredictionService;
    }

    @PostMapping
    public GameSummaryDto create(@RequestBody PredictionRequest request) {
        return predictionService.predict(request);
    }

    @PostMapping("/quarter-half")
    public QuarterHalfSummaryDto createQuarterHalf(@RequestBody PredictionRequest request) {
        return quarterHalfPredictionService.predict(request);
    }

    @PostMapping("/player-props")
    public PlayerPropsResponseDto createPlayerProps(@RequestBody PredictionRequest request) {
        return playerPropPredictionService.predict(request);
    }

    /**
     * A FOURTH SIBLING, not a league parameter on the first.
     *
     * Same reasoning as quarter-half and player-props: the request bodies
     * match but the responses do not. The WNBA prices three markets with
     * their own windows, so a shared endpoint would return a mostly-null
     * union whichever league it was called for. It is also the honest shape
     * given the Python side splits the same way.
     */
    @PostMapping("/wnba")
    public WnbaSummaryDto createWnba(@RequestBody PredictionRequest request) {
        return wnbaPredictionService.predict(request);
    }
}
