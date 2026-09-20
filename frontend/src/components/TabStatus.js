/**
 * Loading and error, PER TAB rather than per page.
 *
 * The Game tab being rendered does not mean the player tabs are, and a
 * failed player-props call must not blank a Game tab that already
 * succeeded. Each domain carries its own status for that reason, and this
 * renders whichever one it is handed.
 *
 * THE RETRY IS SCOPED TOO. It re-runs only the call that failed, so
 * recovering from a broken player fetch does not re-POST the quarter/half
 * markets and write a second row for them.
 */
function TabStatus({ status, error, onRetry, what }) {
  if (status === 'loading' || status === 'idle') {
    // A line, not a spinner. The call returns fast enough that a spinner
    // appears and vanishes as a flash of noise.
    return <p className="predictions-muted">Loading…</p>;
  }

  return (
    <div className="predictions-message">
      <h2 className="predictions-message-title">Could not load the {what}</h2>
      <p className="predictions-message-body">
        The request failed, so this is not the same as there being nothing to
        show — it means the service did not answer. Other tabs on this page
        are unaffected.
      </p>
      {error ? <p className="predictions-error-detail">{error}</p> : null}
      <button type="button" className="predictions-retry" onClick={onRetry}>
        Try again
      </button>
    </div>
  );
}

export default TabStatus;
