/**
 * What a tab shows when it has no data.
 *
 * NEVER "THE SERVICE DID NOT ANSWER" FOR A 4xx. A 4xx means the service
 * answered and declined, which is the opposite of not answering, and saying
 * so was the single most misleading sentence in the app: on 2026-10-30 all
 * ten NBA fixtures of the day refused the Q1/1H markets and every one of them
 * produced that heading, followed by the refusal's JSON printed onto the page.
 *
 * THE RAW BODY GOES BEHIND A TOGGLE. It is worth keeping - it is often the
 * only thing that identifies an unclassified refusal - but it is for whoever
 * goes looking, not for the first thing a reader sees.
 */
function TabStatus({ status, error, failure, onRetry, what }) {
  if (status === 'loading' || status === 'idle') {
    return <p className="predictions-muted">Loading…</p>;
  }

  const httpStatus = failure?.status ?? null;
  const refused = typeof httpStatus === 'number' && httpStatus >= 400 && httpStatus < 500;

  return (
    <div className="predictions-message">
      <h2 className="predictions-message-title">
        {refused ? `No ${what} for this fixture` : `Could not load the ${what}`}
      </h2>
      <p className="predictions-message-body">
        {refused
          ? `The service answered and declined to predict this one. That is not the
             same as a failure, and the other tabs on this page are unaffected.`
          : `The request failed, so this is not the same as there being nothing to
             show — it means the service did not answer. Other tabs on this page
             are unaffected.`}
      </p>
      {error ? (
        <details className="predictions-error-details">
          <summary>Technical details</summary>
          <p className="predictions-error-detail">{error}</p>
        </details>
      ) : null}
      <button type="button" className="predictions-retry" onClick={onRetry}>
        Try again
      </button>
    </div>
  );
}

export default TabStatus;
