function TabStatus({ status, error, onRetry, what }) {
  if (status === 'loading' || status === 'idle') {
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
