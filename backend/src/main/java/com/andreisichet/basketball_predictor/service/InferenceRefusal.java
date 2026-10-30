package com.andreisichet.basketball_predictor.service;

import java.util.LinkedHashMap;
import java.util.Map;

import org.springframework.http.HttpStatusCode;
import org.springframework.web.server.ResponseStatusException;

/**
 * Inference refused, and said why in a form a client can branch on.
 *
 * WHAT THIS EXISTS TO CARRY. A refusal from inference is a 400 whose body is
 * now {@code {"detail": …, "code": "insufficient_history", "needed": 10,
 * "home_games_played": 5, …}}. Until this class, the backend put that whole
 * body into the message string and nothing else - so the browser received
 * {@code "Inference service rejected the request: {\"detail\":\"HOME team …"}
 * and the Quarters &amp; Halves tab printed it onto the page under the heading
 * "the service did not answer". The service had answered; it had refused, and
 * said precisely why.
 *
 * THE STATUS AND THE MESSAGE DO NOT MOVE. This extends
 * {@code ResponseStatusException} and passes the same status and the same
 * reason the old code did, so {@code getStatusCode()} and {@code getReason()}
 * are byte-identical and anything reading them keeps working. The structured
 * fields ride alongside, and reach the response body through
 * {@link InferenceErrorAttributes}.
 *
 * WHY A SEPARATE TYPE RATHER THAN A FIELD ON THE OLD ONE. The error-attributes
 * hook has to be able to tell "this error carries fields a client should see"
 * from every other failure in the application, and a type is the one way to do
 * that which cannot be true by accident. Every other error's body is
 * untouched, which is what keeps this additive.
 */
public class InferenceRefusal extends ResponseStatusException {

    /**
     * The refusal's own fields, exactly as inference sent them minus
     * {@code detail}.
     *
     * {@code detail} is deliberately excluded: it is already the message, and
     * sending it twice under two names would be two places for one sentence
     * to drift.
     */
    private final Map<String, Object> fields;

    public InferenceRefusal(HttpStatusCode status, String reason,
                            Map<String, Object> fields, Throwable cause) {
        super(status, reason, cause);
        Map<String, Object> copy = new LinkedHashMap<>();
        if (fields != null) {
            fields.forEach((key, value) -> {
                if (!"detail".equals(key) && value != null) {
                    copy.put(key, value);
                }
            });
        }
        this.fields = Map.copyOf(copy);
    }

    /** Unmodifiable, and never null. */
    public Map<String, Object> getFields() {
        return fields;
    }

    /** The machine-readable reason, or null when inference sent none. */
    public String getCode() {
        Object code = fields.get("code");
        return code == null ? null : String.valueOf(code);
    }
}
