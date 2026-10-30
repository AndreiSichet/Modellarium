package com.andreisichet.basketball_predictor.config;

import java.util.Map;
import java.util.Set;

import org.springframework.boot.web.error.ErrorAttributeOptions;
import org.springframework.boot.webmvc.error.DefaultErrorAttributes;
import org.springframework.stereotype.Component;
import org.springframework.web.context.request.WebRequest;

import com.andreisichet.basketball_predictor.service.InferenceRefusal;

/**
 * Puts a refusal's structured fields into the error body, and nothing else.
 *
 * WHY THE DEFAULT BODY NEEDED EXTENDING AT ALL. This backend answers an error
 * with Spring's standard attributes - {@code timestamp, status, error,
 * message, path} - and the frontend reads {@code message}. Inference's
 * refusals now carry a {@code code} and the numbers behind it, and there was
 * nowhere for them to go: the only channel was the message string, which is
 * how the browser came to display raw JSON.
 *
 * ADDITIVE, AND NARROWLY SO. Every attribute Spring already produced is left
 * exactly as it was, including {@code status} and {@code message}; fields are
 * added only when the error is an {@link InferenceRefusal}. Any other failure
 * in the application - a 404, a bad request, a database error - renders
 * byte-identically to before, which is the whole reason this keys on a type
 * rather than on "is there a JSON body in the message".
 *
 * {@code message} IS NOT REWRITTEN. It still reads "Inference service
 * rejected the request: {…}", because something may be reading it and because
 * the gate compares bodies. The frontend prefers {@code code} when it is
 * there and falls back to {@code message} when it is not, so the prose stays
 * as the human-readable last resort it has always been.
 */
@Component
public class InferenceErrorAttributes extends DefaultErrorAttributes {

    /**
     * Attribute names a refusal may never contribute, present or not.
     *
     * THE FIRST VERSION USED putIfAbsent AND THAT WAS NOT ENOUGH - caught by
     * this class's own test, which fed in a refusal claiming
     * {@code "status": 999} and watched 999 reach the body. putIfAbsent only
     * protects a key Spring has already set, and which attributes Spring sets
     * depends on the dispatch: on a real error it sets all of these, in a
     * unit test it sets fewer. "Protected because Spring usually got there
     * first" is not a guarantee, so the names are listed.
     *
     * Taken from DefaultErrorAttributes' own contract rather than guessed:
     * timestamp, status, error, exception, message, errors, trace, path.
     */
    private static final Set<String> RESERVED = Set.of(
            "timestamp", "status", "error", "exception", "message",
            "errors", "trace", "path");

    @Override
    public Map<String, Object> getErrorAttributes(WebRequest request,
                                                  ErrorAttributeOptions options) {
        Map<String, Object> attributes = super.getErrorAttributes(request, options);

        Throwable error = getError(request);
        for (Throwable current = error; current != null;
                current = current.getCause() == current ? null : current.getCause()) {
            if (current instanceof InferenceRefusal refusal) {
                // RESERVED NAMES ARE DROPPED, not merely deferred to. A
                // refusal cannot contribute `status` or `message` whether or
                // not Spring has already set them, so the response contract
                // holds on every dispatch rather than on the usual one.
                refusal.getFields().forEach((key, value) -> {
                    if (!RESERVED.contains(key)) {
                        attributes.putIfAbsent(key, value);
                    }
                });
                break;
            }
        }
        return attributes;
    }
}
