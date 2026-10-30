package com.andreisichet.basketball_predictor.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withStatus;

import java.util.Map;

import org.junit.jupiter.api.Test;
import org.springframework.boot.web.error.ErrorAttributeOptions;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestClient;
import org.springframework.web.context.request.ServletWebRequest;
import org.springframework.web.server.ResponseStatusException;

import com.andreisichet.basketball_predictor.config.InferenceErrorAttributes;
import com.andreisichet.basketball_predictor.dto.InferenceRequest;

/**
 * A refusal's structured fields reach the response body, and nothing else
 * about an error body moves.
 *
 * THE DEFECT THIS PINS. Inference refuses a Q1/1H fixture with a 400 whose
 * body carries {@code code}, {@code needed} and each side's games played. The
 * backend used to put that whole body into the message string, so the browser
 * received raw JSON and the tab rendered it under "the service did not
 * answer". Measured 2026-10-30: all ten NBA fixtures of the day refused, and
 * the last one cannot succeed before 2026-11-13.
 *
 * WHAT MUST NOT MOVE, and is asserted here rather than assumed: the status
 * code, the message, and the shape of every error that is NOT a refusal.
 */
class InferenceRefusalTest {

    private static final String BASE = "http://inference.test:8000";

    /** The real Q1/1H refusal body, trimmed. */
    private static final String REFUSAL_BODY = """
            {"detail":"HOME team 1610612744 has played 5 game(s) with quarter \
            data this season - not enough for a complete window.",\
            "code":"insufficient_history","needed":10,"home_games_played":5,\
            "away_games_played":5,"short_side":"home",\
            "short_team_id":1610612744}""";

    private record Harness(InferenceClient client, MockRestServiceServer server) {
    }

    private static Harness harness() {
        RestClient.Builder builder = RestClient.builder();
        MockRestServiceServer server = MockRestServiceServer.bindTo(builder).build();
        return new Harness(new InferenceClient(builder, BASE), server);
    }

    private static InferenceRequest request() {
        return new InferenceRequest(1610612744L, 1610612747L,
                java.time.LocalDate.of(2026, 10, 30));
    }

    @Test
    void aCodedRefusalBecomesAnInferenceRefusalCarryingItsFields() {
        Harness h = harness();
        h.server().expect(requestTo(BASE + "/predict/quarter-half"))
                .andRespond(withStatus(HttpStatus.BAD_REQUEST)
                        .contentType(MediaType.APPLICATION_JSON)
                        .body(REFUSAL_BODY));

        assertThatThrownBy(() -> h.client().predictQuarterHalf(request()))
                .isInstanceOf(InferenceRefusal.class)
                .satisfies(thrown -> {
                    InferenceRefusal refusal = (InferenceRefusal) thrown;
                    assertThat(refusal.getCode()).isEqualTo("insufficient_history");
                    assertThat(refusal.getFields())
                            .containsEntry("needed", 10)
                            .containsEntry("home_games_played", 5)
                            .containsEntry("away_games_played", 5)
                            .containsEntry("short_side", "home");
                    // `detail` IS NOT DUPLICATED: it is already the message,
                    // and two names for one sentence is two places to drift.
                    assertThat(refusal.getFields()).doesNotContainKey("detail");
                });
    }

    @Test
    void theStatusAndMessageAreExactlyWhatTheyWereBefore() {
        // THE CONTROL FOR THE WHOLE CHANGE. A refusal is still a 400 whose
        // reason is the context plus the body verbatim - which is what
        // InferenceClientCauseTest pins for every other response, and what
        // the regression gate compares.
        Harness h = harness();
        h.server().expect(requestTo(BASE + "/predict/quarter-half"))
                .andRespond(withStatus(HttpStatus.BAD_REQUEST)
                        .contentType(MediaType.APPLICATION_JSON)
                        .body(REFUSAL_BODY));

        assertThatThrownBy(() -> h.client().predictQuarterHalf(request()))
                .satisfies(thrown -> {
                    ResponseStatusException error = (ResponseStatusException) thrown;
                    assertThat(error.getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
                    assertThat(error.getReason())
                            .isEqualTo("Inference service rejected the request: "
                                    + REFUSAL_BODY);
                    assertThat(error.getBody().getStatus()).isEqualTo(400);
                });
    }

    @Test
    void aRefusalWithNoCodeStaysAPlainResponseStatusException() {
        // THE THIRD FRONTEND CASE'S BACKEND HALF. A 4xx that nothing
        // classified must NOT arrive looking classified, or the tab would
        // show the insufficient-history sentence for an unrelated refusal.
        Harness h = harness();
        h.server().expect(requestTo(BASE + "/predict/quarter-half"))
                .andRespond(withStatus(HttpStatus.BAD_REQUEST)
                        .contentType(MediaType.APPLICATION_JSON)
                        .body("{\"detail\":\"something else went wrong\"}"));

        assertThatThrownBy(() -> h.client().predictQuarterHalf(request()))
                .isInstanceOf(ResponseStatusException.class)
                .isNotInstanceOf(InferenceRefusal.class)
                .satisfies(thrown -> assertThat(
                        ((ResponseStatusException) thrown).getStatusCode())
                        .isEqualTo(HttpStatus.BAD_REQUEST));
    }

    @Test
    void aNonJsonErrorBodyDoesNotBecomeAParseFailure() {
        // Parsing runs while already handling a failure, so a body that is
        // HTML must degrade to "no structured fields" rather than replacing a
        // useful 400 with a mapper exception.
        Harness h = harness();
        h.server().expect(requestTo(BASE + "/predict/quarter-half"))
                .andRespond(withStatus(HttpStatus.BAD_REQUEST)
                        .contentType(MediaType.TEXT_HTML)
                        .body("<html>nginx</html>"));

        assertThatThrownBy(() -> h.client().predictQuarterHalf(request()))
                .isInstanceOf(ResponseStatusException.class)
                .isNotInstanceOf(InferenceRefusal.class);
    }

    @Test
    void theErrorAttributesMergeTheFieldsAndLeaveSpringsOwnAlone() {
        InferenceErrorAttributes attributes = new InferenceErrorAttributes();
        MockHttpServletRequest servletRequest = new MockHttpServletRequest();
        ServletWebRequest request = new ServletWebRequest(servletRequest);

        InferenceRefusal refusal = new InferenceRefusal(
                HttpStatus.BAD_REQUEST, "Inference service rejected the request: {}",
                Map.of("code", "insufficient_history", "needed", 10), null);
        attributes.resolveException(servletRequest, null, null, refusal);

        Map<String, Object> body = attributes.getErrorAttributes(
                request, ErrorAttributeOptions.defaults());

        assertThat(body).containsEntry("code", "insufficient_history")
                .containsEntry("needed", 10);
        // SPRING'S OWN ATTRIBUTES ARE UNTOUCHED. putIfAbsent, not put: a
        // refusal field must never silently replace part of the contract.
        assertThat(body).containsKeys("timestamp", "status", "error");
    }

    @Test
    void aRefusalFieldCannotOverwriteOneOfSpringsAttributes() {
        InferenceErrorAttributes attributes = new InferenceErrorAttributes();
        MockHttpServletRequest servletRequest = new MockHttpServletRequest();
        ServletWebRequest request = new ServletWebRequest(servletRequest);

        // 418, NOT 999, AND THAT IS A HARNESS FIX. The first version used
        // 999 and went red reporting `status=999` - but 999 is Spring's OWN
        // sentinel for "this dispatch carries no status attribute", which
        // DefaultErrorAttributes writes alongside `error="None"`. The
        // assertion could not tell the refusal's value from Spring's, so it
        // failed against working code. Any value Spring does not use itself
        // makes the test about the filter again.
        InferenceRefusal refusal = new InferenceRefusal(
                HttpStatus.BAD_REQUEST, "reason",
                Map.of("code", "x", "status", 418, "message", "mine"), null);
        attributes.resolveException(servletRequest, null, null, refusal);

        Map<String, Object> body = attributes.getErrorAttributes(
                request, ErrorAttributeOptions.defaults());

        // DROPPED OUTRIGHT, not merely lost to Spring's own value. The
        // first version of this assertion was isNotEqualTo(999) and it went
        // red: putIfAbsent had added 999, because this mock dispatch sets no
        // `status` attribute for it to defer to. The reserved-name list is
        // what makes the guarantee hold on every dispatch.
        assertThat(body).doesNotContainEntry("status", 418);
        assertThat(body).doesNotContainEntry("message", "mine");
        assertThat(body).containsEntry("code", "x");
    }

    @Test
    void anErrorThatIsNotARefusalGetsNoExtraFields() {
        // THE ADDITIVE GUARANTEE. Every other failure in the application -
        // a 404, a bad parameter, a database error - must render exactly as
        // it did before this class existed.
        InferenceErrorAttributes attributes = new InferenceErrorAttributes();
        MockHttpServletRequest servletRequest = new MockHttpServletRequest();
        ServletWebRequest request = new ServletWebRequest(servletRequest);

        attributes.resolveException(servletRequest, null, null,
                new IllegalStateException("unrelated"));

        Map<String, Object> body = attributes.getErrorAttributes(
                request, ErrorAttributeOptions.defaults());

        assertThat(body).doesNotContainKeys("code", "needed",
                "home_games_played");
    }
}
