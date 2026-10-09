package com.andreisichet.basketball_predictor.controller;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.andreisichet.basketball_predictor.dto.GameSummaryDto;
import com.andreisichet.basketball_predictor.dto.GleagueSummaryDto;
import com.andreisichet.basketball_predictor.dto.NflSummaryDto;
import com.andreisichet.basketball_predictor.dto.PlayerPropsResponseDto;
import com.andreisichet.basketball_predictor.dto.PredictionRequest;
import com.andreisichet.basketball_predictor.dto.QuarterHalfSummaryDto;
import com.andreisichet.basketball_predictor.dto.WnbaSummaryDto;
import com.andreisichet.basketball_predictor.service.PlayerPropPredictionService;
import com.andreisichet.basketball_predictor.service.GleaguePredictionService;
import com.andreisichet.basketball_predictor.service.NflPredictionService;
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
    private final GleaguePredictionService gleaguePredictionService;
    private final NflPredictionService nflPredictionService;

    public PredictionController(
            PredictionService predictionService,
            QuarterHalfPredictionService quarterHalfPredictionService,
            PlayerPropPredictionService playerPropPredictionService,
            WnbaPredictionService wnbaPredictionService,
            GleaguePredictionService gleaguePredictionService,
            NflPredictionService nflPredictionService) {
        this.predictionService = predictionService;
        this.quarterHalfPredictionService = quarterHalfPredictionService;
        this.playerPropPredictionService = playerPropPredictionService;
        this.wnbaPredictionService = wnbaPredictionService;
        this.gleaguePredictionService = gleaguePredictionService;
        this.nflPredictionService = nflPredictionService;
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

    /**
     * The G League's three markets. A fourth sibling, not a league parameter.
     *
     * No existing endpoint gained a league argument, which is the same call
     * the WNBA phase made: the three leagues' responses carry different
     * markets and different season labels - "2025-26" here against 2025 for
     * the WNBA - so a shared endpoint would return a mostly-null union and
     * would have to vary its own field types by league.
     */
    /**
     * The NFL's three markets.
     *
     * REFUSES FAR MORE OFTEN THAN ITS SIBLINGS, and a 400 here is usually
     * correct rather than a client error worth logging as one. The NFL is
     * served under a DEPENDENCY rule - a fixture is predictable once both
     * teams' previous games are in history - so 193 of 208 remaining 2026
     * fixtures are refused today, each with a reason naming which side's
     * previous game is missing. The inference service's message passes
     * through unchanged, which is the only reason that reason is useful.
     */
    @PostMapping("/nfl")
    public NflSummaryDto createNfl(@RequestBody PredictionRequest request) {
        return nflPredictionService.predict(request);
    }

    @PostMapping("/gleague")
    public GleagueSummaryDto createGleague(@RequestBody PredictionRequest request) {
        return gleaguePredictionService.predict(request);
    }
}
