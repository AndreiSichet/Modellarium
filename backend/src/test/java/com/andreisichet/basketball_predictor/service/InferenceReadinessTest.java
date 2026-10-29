package com.andreisichet.basketball_predictor.service;

import static org.assertj.core.api.Assertions.assertThat;

import java.net.ConnectException;
import java.net.SocketTimeoutException;

import org.junit.jupiter.api.Test;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.MediaType;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.HttpServerErrorException;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.UnknownContentTypeException;
import org.springframework.web.server.ResponseStatusException;

/**
 * One case per exception class the inference client can produce.
 *
 * THE CLASSIFICATION IS THE WHOLE RISK. Retrying the wrong thing is not a
 * harmless extra request: a bad request retried seven times is seven
 * identical failures and five minutes of a thread, and inference's own 503
 * NOT retried is the cold-start bug left in place. The mapped status cannot
 * tell these apart - InferenceClient sends every RestClientException to 503
 * and inference's own 503 to 502 - which is why this reads the cause.
 */
class InferenceReadinessTest {

    /** As InferenceClient.unreachable() wraps it. */
    private static ResponseStatusException unreachable(Throwable cause) {
        return new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE,
                "Inference service unreachable: " + cause.getMessage(), cause);
    }

    /** As InferenceClient.rejected() wraps it. */
    private static ResponseStatusException rejected(HttpStatus mapped,
                                                    Throwable cause) {
        return new ResponseStatusException(mapped, "context: body", cause);
    }

    @Test
    void aRefusedConnectionIsNotReady() {
        ResourceAccessException cause = new ResourceAccessException(
                "I/O error on GET request", new ConnectException(
                        "Connection refused: no further information"));

        assertThat(InferenceReadiness.isNotReady(unreachable(cause))).isTrue();
    }

    @Test
    void aReadTimeoutIsNotReady() {
        ResourceAccessException cause = new ResourceAccessException(
                "I/O error on GET request",
                new SocketTimeoutException("Read timed out"));

        assertThat(InferenceReadiness.isNotReady(unreachable(cause))).isTrue();
    }

    @Test
    void inferencesOwn503IsNotReadyAlthoughItIsMappedTo502() {
        HttpServerErrorException cause = HttpServerErrorException.create(
                HttpStatusCode.valueOf(503), "Service Unavailable",
                HttpHeaders.EMPTY, new byte[0], null);
        ResponseStatusException wrapped =
                rejected(HttpStatus.BAD_GATEWAY, cause);

        // The mapped status is 502, which says nothing about readiness.
        assertThat(wrapped.getStatusCode().value()).isEqualTo(502);
        assertThat(InferenceReadiness.isNotReady(wrapped)).isTrue();
    }

    @Test
    void aBadRequestIsNotNotReady() {
        HttpClientErrorException cause = HttpClientErrorException.create(
                HttpStatusCode.valueOf(400), "Bad Request",
                HttpHeaders.EMPTY, new byte[0], null);

        assertThat(InferenceReadiness.isNotReady(
                rejected(HttpStatus.BAD_REQUEST, cause))).isFalse();
    }

    @Test
    void aNotFoundIsNotNotReady() {
        HttpClientErrorException cause = HttpClientErrorException.create(
                HttpStatusCode.valueOf(404), "Not Found",
                HttpHeaders.EMPTY, new byte[0], null);

        assertThat(InferenceReadiness.isNotReady(
                rejected(HttpStatus.BAD_REQUEST, cause))).isFalse();
    }

    @Test
    void aFiveHundredIsNotNotReady() {
        HttpServerErrorException cause = HttpServerErrorException.create(
                HttpStatusCode.valueOf(500), "Internal Server Error",
                HttpHeaders.EMPTY, new byte[0], null);

        assertThat(InferenceReadiness.isNotReady(
                rejected(HttpStatus.BAD_GATEWAY, cause))).isFalse();
    }

    @Test
    void aGatewayTimeoutFromInferenceIsNotNotReady() {
        // 504 is a 5xx that is not 503. Retrying is not obviously wrong, but
        // the rule is exactly 503 - so this pins the boundary rather than
        // leaving "5xx-ish" to be read into it later.
        HttpServerErrorException cause = HttpServerErrorException.create(
                HttpStatusCode.valueOf(504), "Gateway Timeout",
                HttpHeaders.EMPTY, new byte[0], null);

        assertThat(InferenceReadiness.isNotReady(
                rejected(HttpStatus.BAD_GATEWAY, cause))).isFalse();
    }

    @Test
    void aMalformedBodyIsNotNotReady() {
        // REPORTED AS UNREACHABLE/503 TODAY, AND THAT IS A RECORDED DEFECT
        // RATHER THAN A DECISION. UnknownContentTypeException extends
        // RestClientException but not RestClientResponseException, so
        // InferenceClient's second catch takes it and calls it unreachable.
        // What this asserts is that the defect cannot make the retry loop
        // spin: matching on ResourceAccessException rather than on
        // RestClientException keeps it out of the retry set whatever status
        // it is reported under.
        UnknownContentTypeException cause = new UnknownContentTypeException(
                String.class, MediaType.TEXT_HTML, 200, "OK",
                HttpHeaders.EMPTY, new byte[0]);

        assertThat(InferenceReadiness.isNotReady(unreachable(cause))).isFalse();
    }

    @Test
    void theCauseIsFoundThroughMoreThanOneWrapper() {
        // The real chain is two deep already (ResponseStatusException ->
        // ResourceAccessException -> ConnectException). A third wrapper is
        // what a future @Transactional or proxy layer would add, and the walk
        // has to survive it.
        ResourceAccessException cause = new ResourceAccessException(
                "I/O error", new ConnectException("Connection refused"));

        assertThat(InferenceReadiness.isNotReady(
                new IllegalStateException("outer", unreachable(cause))))
                .isTrue();
    }

    @Test
    void anUnrelatedFailureIsNotNotReady() {
        assertThat(InferenceReadiness.isNotReady(
                new IllegalStateException("something else"))).isFalse();
    }

    @Test
    void aCyclicCauseChainTerminatesRatherThanSpinning() {
        // NOT A SELF-CAUSE, BECAUSE THAT CANNOT BE BUILT. Throwable.initCause
        // throws IllegalArgumentException if the cause is the throwable itself
        // - which is how the first version of this guard turned out to be
        // unreachable code guarding nothing. A two-exception cycle IS
        // constructible, and it is what the depth bound actually protects
        // against. A hang in the startup listener is the worst outcome in
        // this file, so it is pinned.
        RuntimeException first = new RuntimeException("first");
        RuntimeException second = new RuntimeException("second", first);
        first.initCause(second);

        assertThat(InferenceReadiness.isNotReady(first)).isFalse();
    }

    @Test
    void nullIsNotNotReady() {
        assertThat(InferenceReadiness.isNotReady(null)).isFalse();
    }
}
