import { formatValue } from '../predictionFormat';
import { MarketGroup, Metric } from './GameTab';
import TabStatus from './TabStatus';

function QuarterHalfTab({ state, onRetry, health }) {
  if (state.status !== 'ready') {
    return (
      <TabStatus
        status={state.status}
        error={state.error}
        onRetry={onRetry}
        what="quarter and half markets"
      />
    );
  }

  const markets = state.data;

  if (!markets) {
    return (
      <div className="predictions-message">
        <h2 className="predictions-message-title">No quarter or half prediction</h2>
        <p className="predictions-message-body">
          These six models cannot score a game whose teams do not yet have a
          complete run of recent form — unlike the full-game models, they
          refuse rather than predicting worse.
        </p>
        {health?.dataAsOf ? (
          <p className="predictions-message-meta">
            Model data is current to {health.dataAsOf}.
          </p>
        ) : null}
      </div>
    );
  }

  return (
    <div className="market-groups">
      <MarketGroup title="First quarter">
        <Metric
          label="Home win probability"
          tag={<ConfidenceTag confidence={markets.q1WinnerConfidence} />}
          value={percent(markets.q1WinnerProbability)}
          note={markets.q1WinnerInterpretation}
        />
        <Metric label="Home margin" value={formatValue(markets.q1Spread)} />
        <Metric label="Total points" value={formatValue(markets.q1Total)} />
      </MarketGroup>

      <MarketGroup title="First half">
        <Metric
          label="Home win probability"
          tag={<ConfidenceTag confidence={markets.half1WinnerConfidence} />}
          value={percent(markets.half1WinnerProbability)}
          note={markets.half1WinnerInterpretation}
        />
        <Metric label="Home margin" value={formatValue(markets.half1Spread)} />
        <Metric label="Total points" value={formatValue(markets.half1Total)} />
      </MarketGroup>

      <p className="market-tie-note">
        A quarter or half can end level, and a tie is a push rather than a
        wrong call. The two win probabilities above are conditional on the
        period being decided — which is what their qualifier says.
      </p>

      {markets.stale ? (
        <p className="market-freshness">
          <span className="stale-badge">STALE</span>
          Computed from data as of {markets.dataAsOf}
        </p>
      ) : null}
    </div>
  );
}

function percent(probability) {
  return typeof probability === 'number'
    ? `${(probability * 100).toFixed(1)}%`
    : '—';
}

function ConfidenceTag({ confidence }) {
  if (!confidence) return null;
  return (
    <span className={`confidence-tag confidence-tag--${confidence}`}>
      {confidence} confidence
    </span>
  );
}

export default QuarterHalfTab;
