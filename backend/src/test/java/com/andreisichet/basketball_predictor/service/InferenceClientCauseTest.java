package com.andreisichet.basketball_predictor.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withServerError;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withStatus;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;

import org.junit.jupiter.api.Test;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.client.UnknownContentTypeException;
import org.springframework.web.server.ResponseStatusException;

/**
 * Passing the cause changed no status code and no message.
 *
 * WHY THIS TEST EXISTS AND NOT JUST THE CLASSIFIER'S. InferenceReadiness
 * needs the original exception, so InferenceClient now passes it as the cause.
 * That is a change to the one class every API error in this backend goes
 * through, and the requirement was explicit: the responses must be identical.
 * A ResponseStatusException builds its response from getStatusCode() and
 * getReason(), and a third constructor argument touches neither - but
 * "touches neither" is a claim about Spring's internals, so it is asserted
 * here against the real client rather than trusted.
 *
 * The two paths are checked separately because they are two different catch
 * blocks: rejected() for a service that answered with a failure, unreachable()
 * for one that did not answer.
 */
class InferenceClientCauseTest {

    private static final String BASE = "http://inference.test:8000";

    private record Harness(InferenceClient client, MockRestServiceServer server) {
    }

    private static Harness harness() {
        RestClient.Builder builder = RestClient.builder();
        MockRestServiceServer server = MockRestServiceServer.bindTo(builder).build();
        return new Harness(new InferenceClient(builder, BASE), server);
    }

    @Test
    void a4xxKeepsItsStatusAndMessageAndGainsTheCause() {
        Harness h = harness();
        h.server().expect(requestTo(BASE + "/health"))
                .andRespond(withStatus(HttpStatus.BAD_REQUEST)
                        .contentType(MediaType.APPLICATION_JSON)
                        .body("{\"detail\":\"bad\"}"));

        assertThatThrownBy(() -> h.client().getHealth())
                .isInstanceOf(ResponseStatusException.class)
                .satisfies(thrown -> {
                    ResponseStatusException error = (ResponseStatusException) thrown;
                    // UNCHANGED: a 4xx from inference is reported as 400 with
                    // the body appended to the context string.
                    assertThat(error.getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
                    assertThat(error.getReason())
                            .isEqualTo("Health check failed: {\"detail\":\"bad\"}");
                    // NEW, AND THE ONLY THING THAT IS: the cause is reachable.
                    assertThat(error.getCause())
                            .isInstanceOf(RestClientResponseException.class);
                    assertThat(InferenceReadiness.isNotReady(error)).isFalse();
                });
    }

    @Test
    void a5xxKeepsItsStatusAndMessageAndGainsTheCause() {
        Harness h = harness();
        h.server().expect(requestTo(BASE + "/health"))
                .andRespond(withServerError().body("boom"));

        assertThatThrownBy(() -> h.client().getHealth())
                .isInstanceOf(ResponseStatusException.class)
                .satisfies(thrown -> {
                    ResponseStatusException error = (ResponseStatusException) thrown;
                    assertThat(error.getStatusCode()).isEqualTo(HttpStatus.BAD_GATEWAY);
                    assertThat(error.getReason())
                            .isEqualTo("Health check failed: boom");
                    assertThat(error.getCause())
                            .isInstanceOf(RestClientResponseException.class);
                    // 500 is not 503, so this is not a cold start.
                    assertThat(InferenceReadiness.isNotReady(error)).isFalse();
                });
    }

    @Test
    void inferencesOwn503KeepsIts502MappingAndIsStillClassifiedNotReady() {
        Harness h = harness();
        h.server().expect(requestTo(BASE + "/health"))
                .andRespond(withStatus(HttpStatus.SERVICE_UNAVAILABLE)
                        .body("loading"));

        assertThatThrownBy(() -> h.client().getHealth())
                .isInstanceOf(ResponseStatusException.class)
                .satisfies(thrown -> {
                    ResponseStatusException error = (ResponseStatusException) thrown;
                    // THE MAPPING IS NOT SOFTENED. rejected() sends anything
                    // that is not 4xx to BAD_GATEWAY, and that is still what
                    // the API returns - which is exactly why the retry cannot
                    // read the mapped status.
                    assertThat(error.getStatusCode()).isEqualTo(HttpStatus.BAD_GATEWAY);
                    assertThat(error.getReason())
                            .isEqualTo("Health check failed: loading");
                    assertThat(InferenceReadiness.isNotReady(error)).isTrue();
                });
    }

    @Test
    void anUnreachableServiceKeepsIts503AndMessageAndGainsTheCause() {
        // A real socket, refused. Nothing mocks this path: the whole point is
        // that the pre-ready behaviour is a refused connection rather than an
        // HTTP status, which is what a mocked response cannot reproduce.
        InferenceClient client = new InferenceClient(
                RestClient.builder(), "http://127.0.0.1:1");

        assertThatThrownBy(client::getHealth)
                .isInstanceOf(ResponseStatusException.class)
                .satisfies(thrown -> {
                    ResponseStatusException error = (ResponseStatusException) thrown;
                    assertThat(error.getStatusCode())
                            .isEqualTo(HttpStatus.SERVICE_UNAVAILABLE);
                    assertThat(error.getReason())
                            .startsWith("Inference service unreachable: ");
                    assertThat(error.getCause())
                            .isInstanceOf(ResourceAccessException.class);
                    assertThat(InferenceReadiness.isNotReady(error)).isTrue();
                });
    }

    @Test
    void aMalformedBodyIsNowAHonest502AndNotAnUntrue503() {
        // THE WHOLE OF C1. UnknownContentTypeException extends
        // RestClientException but not RestClientResponseException, so until
        // 2026-10-29 it fell through to unreachable() and a service that had
        // answered perfectly well was reported as "Inference service
        // unreachable" with a 503. A JSON endpoint answering text/html is the
        // ordinary way to get here - a proxy error page, or a FastAPI
        // traceback rendered as HTML.
        Harness h = harness();
        h.server().expect(requestTo(BASE + "/health"))
                .andRespond(withSuccess("<html>not json at all</html>",
                        MediaType.TEXT_HTML));

        assertThatThrownBy(() -> h.client().getHealth())
                .isInstanceOf(ResponseStatusException.class)
                .satisfies(thrown -> {
                    ResponseStatusException error = (ResponseStatusException) thrown;
                    assertThat(error.getStatusCode()).isEqualTo(HttpStatus.BAD_GATEWAY);
                    assertThat(error.getReason())
                            .startsWith("Inference service returned a response "
                                    + "the backend could not read: ");
                    // AND IT NO LONGER CLAIMS THE SERVICE WAS UNREACHABLE.
                    assertThat(error.getReason()).doesNotContain("unreachable");
                    assertThat(error.getCause())
                            .isInstanceOf(UnknownContentTypeException.class);
                    // STILL NOT A COLD START. The retry reads the cause, not
                    // the status, so this was never retryable and is not now.
                    assertThat(InferenceReadiness.isNotReady(error)).isFalse();
                });
    }

    @Test
    void theMalformedBodyMappingAppliesToEveryCallSite() {
        // FOUR CATCH BLOCKS, NOT ONE. getHealth, both schedule fetchers and
        // post each have their own try - a fix applied to one of them would
        // leave the other three reporting 503, and the schedule fetch is the
        // call the startup sync actually makes.
        for (String path : new String[] {"/schedule?days_ahead=120",
                                         "/schedule/nfl"}) {
            Harness h = harness();
            h.server().expect(requestTo(BASE + path))
                    .andRespond(withSuccess("<html>nope</html>",
                            MediaType.TEXT_HTML));

            assertThatThrownBy(() -> {
                if (path.contains("nfl")) {
                    h.client().fetchNflSchedule();
                } else {
                    h.client().fetchSchedule(120);
                }
            })
                    .isInstanceOf(ResponseStatusException.class)
                    .satisfies(thrown -> assertThat(
                            ((ResponseStatusException) thrown).getStatusCode())
                            .isEqualTo(HttpStatus.BAD_GATEWAY));
        }
    }

    @Test
    void aRefusedConnectionIsStillA503AndStillSaysUnreachable() {
        // THE CONTROL FOR C1. The point of the change was that one case
        // stopped claiming "unreachable"; the case that genuinely IS
        // unreachable must keep saying so, at the same status, with the same
        // words. Without this the fix could have moved both.
        InferenceClient client = new InferenceClient(
                RestClient.builder(), "http://127.0.0.1:1");

        assertThatThrownBy(client::getHealth)
                .satisfies(thrown -> {
                    ResponseStatusException error = (ResponseStatusException) thrown;
                    assertThat(error.getStatusCode())
                            .isEqualTo(HttpStatus.SERVICE_UNAVAILABLE);
                    assertThat(error.getReason())
                            .startsWith("Inference service unreachable: ");
                });
    }

    @Test
    void theReasonIsWhatSpringRendersAsTheBody() {
        // getReason() is not decoration: ResponseStatusException's
        // getBody() builds the ProblemDetail from the status and the reason,
        // and that is what reaches the browser. Asserting it here is what
        // makes the three tests above statements about the API's responses
        // rather than about an exception's fields.
        Harness h = harness();
        h.server().expect(requestTo(BASE + "/health"))
                .andRespond(withStatus(HttpStatus.BAD_REQUEST).body("nope"));

        assertThatThrownBy(() -> h.client().getHealth())
                .satisfies(thrown -> {
                    ResponseStatusException error = (ResponseStatusException) thrown;
                    assertThat(error.getBody().getStatus()).isEqualTo(400);
                    assertThat(error.getBody().getDetail())
                            .isEqualTo("Health check failed: nope");
                });
    }
}
