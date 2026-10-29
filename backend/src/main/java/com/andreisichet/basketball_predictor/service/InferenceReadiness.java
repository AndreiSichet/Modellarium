package com.andreisichet.basketball_predictor.service;

import org.springframework.http.HttpStatus;
import org.springframework.web.client.HttpServerErrorException;
import org.springframework.web.client.ResourceAccessException;

/**
 * Is this failure "the inference service is not ready yet", or a real error?
 *
 * WHY THIS CLASSIFIES THE ORIGINAL EXCEPTION AND NOT THE STATUS CODE.
 * {@link InferenceClient} maps <em>every</em> {@code RestClientException} to a
 * 503, so by the time a caller sees a {@code ResponseStatusException} the
 * status says nothing about what went wrong: a refused connection, a read
 * timeout and a response body it could not parse all arrive as 503. Worse,
 * inference answering 503 for itself arrives as <em>502</em>, because
 * {@code rejected()} sends anything that is not 4xx to BAD_GATEWAY. Deciding
 * what to retry from the mapped status would therefore retry a malformed body
 * forever and refuse to retry inference's own 503.
 *
 * So the cause chain is what is examined. {@code InferenceClient} now passes
 * the original exception as the cause for exactly this purpose - it changes no
 * status and no message, only what is reachable from the exception.
 *
 * <p>Retried, because the service may simply not be up yet:
 * <ul>
 *   <li>{@link ResourceAccessException} - a refused connection, an I/O error
 *       or a timeout. This is what a cold start produces: measured on
 *       2026-10-29, all four leagues failed within 59-82ms of each other,
 *       far too fast to be a timeout, because uvicorn runs its lifespan
 *       before it binds a socket and so there is nothing listening until the
 *       models are loaded.</li>
 *   <li>{@link HttpServerErrorException} with status 503 - inference itself
 *       saying "not available". Nothing emits this today; it is retried
 *       because a readiness-aware inference service would.</li>
 * </ul>
 *
 * <p>Not retried, because repeating them cannot help:
 * <ul>
 *   <li>any 4xx - a bad request stays bad</li>
 *   <li>any other 5xx - a real server fault</li>
 *   <li>a malformed body, which arrives as
 *       {@code UnknownContentTypeException}. That extends
 *       {@code RestClientException} but <em>not</em>
 *       {@code RestClientResponseException}, so it is reported as
 *       unreachable/503 today - a separate recorded defect, deliberately not
 *       fixed here. Matching on {@code ResourceAccessException} rather than on
 *       {@code RestClientException} is what keeps it out of the retry set
 *       regardless.</li>
 * </ul>
 */
public final class InferenceReadiness {

    private InferenceReadiness() {
    }

    /**
     * True when the failure means "not ready yet" rather than "wrong".
     *
     * The whole cause chain is walked, not just the top, because the caller
     * sees the {@code ResponseStatusException} that wraps the original.
     */
    public static boolean isNotReady(Throwable error) {
        // BOUNDED, BECAUSE A CAUSE CHAIN CAN BE A CYCLE. Throwable refuses to
        // let an exception be its own cause, so the obvious self-cause guard
        // is unreachable code - but A causing B causing A is constructible and
        // would spin here forever. A hang in the startup listener is worse
        // than any wrong answer, so the walk is capped. Real chains are two or
        // three deep.
        Throwable current = error;
        for (int depth = 0; current != null && depth < MAX_CAUSE_DEPTH; depth++) {
            if (current instanceof ResourceAccessException) {
                return true;
            }
            if (current instanceof HttpServerErrorException server
                    && server.getStatusCode().value()
                            == HttpStatus.SERVICE_UNAVAILABLE.value()) {
                return true;
            }
            current = current.getCause();
        }
        return false;
    }

    private static final int MAX_CAUSE_DEPTH = 20;
}
