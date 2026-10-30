package com.andreisichet.basketball_predictor.service;

import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.client.UnknownContentTypeException;
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

import tools.jackson.core.type.TypeReference;
import tools.jackson.databind.ObjectMapper;

/** The one place that talks to the Python inference service. */
@Component
public class InferenceClient {
    private static final ParameterizedTypeReference<List<InferenceScheduledGame>> SCHEDULE_TYPE =
            new ParameterizedTypeReference<>() {
            };

    private static final TypeReference<Map<String, Object>> BODY_TYPE =
            new TypeReference<>() {
            };

    private final RestClient client;

    /**
     * ITS OWN MAPPER, NOT THE INJECTED ONE, AND THAT IS DELIBERATE.
     * Deserialising a refusal body must not be able to fail because the
     * application's mapper was configured for something else - and this
     * constructor is called from tests that build the client directly with no
     * Spring context at all. The only thing read is a flat map of a refusal's
     * own fields.
     */
    private final ObjectMapper refusalMapper = new ObjectMapper();

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
        } catch (UnknownContentTypeException error) {
            throw unreadable(error);
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
        } catch (UnknownContentTypeException error) {
            throw unreadable(error);
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
        } catch (UnknownContentTypeException error) {
            throw unreadable(error);
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
        } catch (UnknownContentTypeException error) {
            throw unreadable(error);
        } catch (RestClientException error) {
            throw unreachable(error);
        }
    }

    /**
     * The service answered, but with a failure.
     *
     * THE CAUSE IS PASSED, AND THE STATUS AND REASON ARE UNCHANGED. Adding a
     * cause to a {@code ResponseStatusException} alters neither
     * {@code getStatusCode()} nor {@code getReason()}, and Spring builds the
     * response body from those two - so every status code and message this API
     * returns is byte-identical to before. What it changes is that
     * {@link InferenceReadiness} can now see what actually went wrong, which
     * the mapped status cannot express: a 4xx and a malformed body both have
     * to be told apart from a service that is merely still starting.
     */
    private ResponseStatusException rejected(String context, RestClientResponseException error) {
        HttpStatus status = error.getStatusCode().is4xxClientError()
                ? HttpStatus.BAD_REQUEST
                : HttpStatus.BAD_GATEWAY;
        String body = error.getResponseBodyAsString();
        String reason = context + ": " + body;

        // THE SAME STATUS AND THE SAME REASON EITHER WAY. A refusal that
        // carries a `code` becomes an InferenceRefusal so its fields can
        // reach the response body through InferenceErrorAttributes; one that
        // does not stays exactly the ResponseStatusException it has always
        // been. Nothing about the status or the message depends on which
        // branch is taken, which is what makes this additive.
        Map<String, Object> fields = refusalFields(body);
        if (fields != null && fields.get("code") != null) {
            return new InferenceRefusal(status, reason, fields, error);
        }
        return new ResponseStatusException(status, reason, error);
    }

    /**
     * A refusal body as a flat map, or null when it is not one.
     *
     * NULL ON ANY DOUBT, AND NEVER A THROW. This runs while already handling
     * a failure, so a body that is HTML, empty, or JSON of the wrong shape
     * must degrade to "no structured fields" rather than replacing a useful
     * 400 with a parse error. An unknown is not a zero: a null here means
     * "nothing classified this", which is what the frontend treats as a
     * code-less refusal.
     */
    private Map<String, Object> refusalFields(String body) {
        if (body == null || body.isBlank() || body.charAt(0) != '{') {
            return null;
        }
        try {
            return refusalMapper.readValue(body, BODY_TYPE);
        } catch (RuntimeException ignored) {
            return null;
        }
    }

    /**
     * The service answered, and the backend could not read what it said.
     *
     * 502, NOT THE 503 THIS USED TO BE, AND THE OLD ANSWER WAS UNTRUE.
     * {@code UnknownContentTypeException} extends {@code RestClientException}
     * but NOT {@code RestClientResponseException}, so it fell through to
     * {@code unreachable} and a service that had answered perfectly well was
     * reported as *"Inference service unreachable"* with a 503 - the wrong
     * diagnosis in the first place an operator looks. 502 is what the other
     * "inference answered badly" case already uses, which is
     * {@code rejected}'s non-4xx branch.
     *
     * NOTHING ELSE MOVED. A refused connection, a timeout and any other I/O
     * failure are still 503 with the same message; 4xx is still 400 and
     * other 5xx still 502, both with their existing text.
     * {@code InferenceClientCauseTest} asserts each of those unchanged.
     *
     * AND IT IS STILL NOT A COLD START. {@link InferenceReadiness} matches on
     * {@code ResourceAccessException}, never on {@code RestClientException},
     * so a body the backend cannot parse was never retryable and is not
     * retryable now - the status it is reported under was never what the
     * retry read.
     */
    private ResponseStatusException unreadable(UnknownContentTypeException error) {
        return new ResponseStatusException(
                HttpStatus.BAD_GATEWAY,
                "Inference service returned a response the backend could "
                        + "not read: " + error.getMessage(), error);
    }

    /** Nothing answered at all. */
    private ResponseStatusException unreachable(RestClientException error) {
        return new ResponseStatusException(
                HttpStatus.SERVICE_UNAVAILABLE,
                "Inference service unreachable: " + error.getMessage(), error);
    }
}
