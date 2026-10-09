package com.andreisichet.basketball_predictor.service;

import java.util.List;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.server.ResponseStatusException;

import com.andreisichet.basketball_predictor.dto.InferenceHealth;
import com.andreisichet.basketball_predictor.dto.InferencePlayerPropsResponse;
import com.andreisichet.basketball_predictor.dto.InferenceGleagueResponse;
import com.andreisichet.basketball_predictor.dto.InferenceNflResponse;
import com.andreisichet.basketball_predictor.dto.InferenceQuarterHalfResponse;
import com.andreisichet.basketball_predictor.dto.InferenceRequest;
import com.andreisichet.basketball_predictor.dto.InferenceResponse;
import com.andreisichet.basketball_predictor.dto.InferenceScheduledGame;
import com.andreisichet.basketball_predictor.dto.InferenceWnbaResponse;

/** The one place that talks to the Python inference service. */
@Component
public class InferenceClient {
    private static final ParameterizedTypeReference<List<InferenceScheduledGame>> SCHEDULE_TYPE =
            new ParameterizedTypeReference<>() {
            };

    private final RestClient client;

    public InferenceClient(
            RestClient.Builder builder,
            @Value("${inference.service.url}") String inferenceServiceUrl) {
        this.client = builder.baseUrl(inferenceServiceUrl).build();
    }

    /** The seven full-game models. */
    public InferenceResponse predict(InferenceRequest body) {
        return post("/predict", body, InferenceResponse.class);
    }

    /** The six Q1 / first-half models. */
    public InferenceQuarterHalfResponse predictQuarterHalf(InferenceRequest body) {
        return post("/predict/quarter-half", body, InferenceQuarterHalfResponse.class);
    }

    /** The three WNBA models: moneyline, spread and totals. */
    public InferenceWnbaResponse predictWnba(InferenceRequest body) {
        return post("/predict/wnba", body, InferenceWnbaResponse.class);
    }

    /** The three G League models: moneyline, spread and totals. */
    public InferenceGleagueResponse predictGleague(InferenceRequest body) {
        return post("/predict/gleague", body, InferenceGleagueResponse.class);
    }

    /** Upcoming G League regular-season fixtures. */
    public List<InferenceScheduledGame> fetchGleagueSchedule(int daysAhead) {
        return fetchSchedule("/schedule/gleague", daysAhead);
    }

    /** The three NFL models: winner, margin and total. */
    public InferenceNflResponse predictNfl(InferenceRequest body) {
        return post("/predict/nfl", body, InferenceNflResponse.class);
    }

    /**
     * Unplayed NFL fixtures.
     *
     * TAKES NO daysAhead, AND THE OTHER THREE DO. Those proxy a live nba_api
     * call that is genuinely parameterised by a horizon. The NFL's fixtures
     * come out of the served snapshot, so the list is whatever remains of the
     * season - bounded by the schedule rather than by a window - and passing a
     * horizon would be a parameter the endpoint ignores. Date filtering still
     * happens downstream in ScheduleService, against the cached rows, exactly
     * as it does for the others.
     *
     * The richer fields /schedule/nfl returns - week, kickoff in UTC, the
     * predictable flag - are deliberately not read here. The sync only needs
     * the pair and the date to create a Game row; a client that wants the rest
     * asks the inference service directly, which is what phase 5 will do.
     */
    public List<InferenceScheduledGame> fetchNflSchedule() {
        try {
            List<InferenceScheduledGame> fixtures = client.get()
                    .uri("/schedule/nfl")
                    .retrieve()
                    .body(SCHEDULE_TYPE);
            return fixtures == null ? List.of() : fixtures;
        } catch (RestClientResponseException error) {
            throw rejected("NFL schedule lookup failed", error);
        } catch (RestClientException error) {
            throw unreachable(error);
        }
    }

    /** Both teams' prop boards in one call. */
    public InferencePlayerPropsResponse predictPlayerProps(InferenceRequest body) {
        return post("/predict/player-props", body, InferencePlayerPropsResponse.class);
    }

    /** Upcoming WNBA regular-season fixtures. */
    public List<InferenceScheduledGame> fetchWnbaSchedule(int daysAhead) {
        return fetchSchedule("/schedule/wnba", daysAhead);
    }

    /** Upcoming regular-season fixtures, straight from nba_api. */
    public List<InferenceScheduledGame> fetchSchedule(int daysAhead) {
        return fetchSchedule("/schedule", daysAhead);
    }

    private List<InferenceScheduledGame> fetchSchedule(String path, int daysAhead) {
        try {
            List<InferenceScheduledGame> fixtures = client.get()
                    .uri(builder -> builder.path(path)
                            .queryParam("days_ahead", daysAhead)
                            .build())
                    .retrieve()
                    .body(SCHEDULE_TYPE);
            return fixtures == null ? List.of() : fixtures;
        } catch (RestClientResponseException error) {
            throw rejected("Schedule lookup failed", error);
        } catch (RestClientException error) {
            throw unreachable(error);
        }
    }

    /** How fresh the underlying pipeline data is. */
    public InferenceHealth getHealth() {
        try {
            return client.get()
                    .uri("/health")
                    .retrieve()
                    .body(InferenceHealth.class);
        } catch (RestClientResponseException error) {
            throw rejected("Health check failed", error);
        } catch (RestClientException error) {
            throw unreachable(error);
        }
    }

    private <T> T post(String path, InferenceRequest body, Class<T> responseType) {
        try {
            return client.post()
                    .uri(path)
                    .body(body)
                    .retrieve()
                    .body(responseType);
        } catch (RestClientResponseException error) {
            throw rejected("Inference service rejected the request", error);
        } catch (RestClientException error) {
            throw unreachable(error);
        }
    }

    /** The service answered, but with a failure. */
    private ResponseStatusException rejected(String context, RestClientResponseException error) {
        HttpStatus status = error.getStatusCode().is4xxClientError()
                ? HttpStatus.BAD_REQUEST
                : HttpStatus.BAD_GATEWAY;
        return new ResponseStatusException(
                status, context + ": " + error.getResponseBodyAsString());
    }

    /** Nothing answered at all. */
    private ResponseStatusException unreachable(RestClientException error) {
        return new ResponseStatusException(
                HttpStatus.SERVICE_UNAVAILABLE,
                "Inference service unreachable: " + error.getMessage());
    }
}
