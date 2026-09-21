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
import com.andreisichet.basketball_predictor.dto.InferenceQuarterHalfResponse;
import com.andreisichet.basketball_predictor.dto.InferenceRequest;
import com.andreisichet.basketball_predictor.dto.InferenceResponse;
import com.andreisichet.basketball_predictor.dto.InferenceScheduledGame;

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

    /** Both teams' prop boards in one call. */
    public InferencePlayerPropsResponse predictPlayerProps(InferenceRequest body) {
        return post("/predict/player-props", body, InferencePlayerPropsResponse.class);
    }

    /** Upcoming regular-season fixtures, straight from nba_api. */
    public List<InferenceScheduledGame> fetchSchedule(int daysAhead) {
        try {
            List<InferenceScheduledGame> fixtures = client.get()
                    .uri(builder -> builder.path("/schedule")
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
