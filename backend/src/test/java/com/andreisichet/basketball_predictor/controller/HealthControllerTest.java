package com.andreisichet.basketball_predictor.controller;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;
import org.springframework.http.HttpStatus;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.server.ResponseStatusException;

import com.andreisichet.basketball_predictor.dto.HealthDto.ScheduleSyncDto;
import com.andreisichet.basketball_predictor.service.ScheduleService;
import com.andreisichet.basketball_predictor.service.ScheduleSyncService.SyncOutcome;
import com.andreisichet.basketball_predictor.service.ScheduleSyncService.SyncResult;
import com.andreisichet.basketball_predictor.service.ScheduleSyncStatus;

/**
 * The schedule-sync block is readable when inference is not.
 *
 * WHY THIS IS A UNIT TEST AND NOT A SLICE TEST. What is being asserted is
 * that one endpoint makes no outbound call and the other still fails - which
 * is a property of the controller's two methods, not of the HTTP plumbing.
 * A @WebMvcTest would need the whole service graph mocked to say the same
 * thing, and the mapping of @GetMapping("/schedule-sync") onto a path is
 * Spring's, not this project's.
 */
class HealthControllerTest {

    /**
     * A ScheduleService whose inference call fails exactly as the real one
     * does when nothing is listening.
     *
     * The real failure is a ResponseStatusException built by
     * InferenceClient.unreachable - 503, with a ResourceAccessException as
     * its cause - so that is what is thrown here rather than a bare
     * RuntimeException. InferenceClientCauseTest is what proves the real
     * client produces this shape.
     */
    private static final class DownInference extends ScheduleService {
        DownInference() {
            super(null, null, null, null);
        }

        @Override
        public com.andreisichet.basketball_predictor.dto.HealthDto getHealth() {
            throw new ResponseStatusException(
                    HttpStatus.SERVICE_UNAVAILABLE,
                    "Inference service unreachable: Connection refused",
                    new ResourceAccessException("I/O error: Connection refused"));
        }
    }

    private static ScheduleSyncStatus retryingStatus() {
        ScheduleSyncStatus status = new ScheduleSyncStatus();
        status.record(new SyncOutcome("NBA", SyncResult.NOT_READY,
                "503 SERVICE_UNAVAILABLE \"Inference service unreachable\""));
        status.startupRetrying(1, 0L);
        return status;
    }

    @Test
    void theSubResourceAnswersWhileInferenceIsDown() {
        ScheduleSyncStatus status = retryingStatus();
        HealthController controller =
                new HealthController(new DownInference(), status);

        ScheduleSyncDto block = controller.scheduleSync();

        // THE ONE STATE THE OLD ENDPOINT COULD NEVER REPORT. `retrying` only
        // exists while inference is not answering, which is exactly when the
        // pass-through is a 503.
        assertThat(block).isNotNull();
        assertThat(block.lastResult()).isEqualTo("retrying");
        assertThat(block.leagues().get("NBA").result()).isEqualTo("NOT_READY");
        assertThat(block.lastSuccessAt()).isNull();
    }

    @Test
    void andTheExistingEndpointStillFailsTheSameWay() {
        // THE CONTROL, AND THE POINT OF THE WHOLE DESIGN. Adding a readable
        // sub-resource must not quieten the red: /api/health still raises the
        // 503 it always raised, with the message it always had. An endpoint
        // answering 200 while inference is down would be the misleading green
        // this project keeps catching.
        HealthController controller =
                new HealthController(new DownInference(), retryingStatus());

        assertThatThrownBy(controller::health)
                .isInstanceOf(ResponseStatusException.class)
                .satisfies(thrown -> {
                    ResponseStatusException error = (ResponseStatusException) thrown;
                    assertThat(error.getStatusCode())
                            .isEqualTo(HttpStatus.SERVICE_UNAVAILABLE);
                    assertThat(error.getReason())
                            .isEqualTo("Inference service unreachable: "
                                    + "Connection refused");
                });
    }

    @Test
    void theSubResourceReportsNeverRunRatherThanAnEmptyBody() {
        // An unknown is not a zero. Before the first attempt there is no
        // result, and the field says so in a word rather than being absent.
        HealthController controller = new HealthController(
                new DownInference(), new ScheduleSyncStatus());

        ScheduleSyncDto block = controller.scheduleSync();

        assertThat(block.lastResult()).isEqualTo("never_run");
        assertThat(block.leagues()).isEmpty();
        assertThat(block.startupAttempts()).isZero();
    }

    @Test
    void theSubResourceMakesNoOutboundCall() {
        // PROVED BY CONSTRUCTION: a ScheduleService that throws on ANY use.
        // If scheduleSync() ever reaches it - a refactor routing the block
        // through the pass-through, say - this fails rather than quietly
        // becoming unavailable again at the moment it is needed.
        ScheduleService explodes = new ScheduleService(null, null, null, null) {
            @Override
            public com.andreisichet.basketball_predictor.dto.HealthDto getHealth() {
                throw new AssertionError(
                        "the schedule-sync sub-resource called inference");
            }
        };
        HealthController controller =
                new HealthController(explodes, retryingStatus());

        assertThat(controller.scheduleSync().lastResult())
                .isEqualTo("retrying");
    }
}
