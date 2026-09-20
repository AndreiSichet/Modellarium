import { formatValue } from '../predictionFormat';
import { MarketGroup, Metric } from './GameTab';
import TabStatus from './TabStatus';

/**
 * The six Q1 / first-half markets.
 *
 * THREE QUALIFIER SURFACES ARE REQUIRED HERE, NOT DECORATIVE, and a
 * redesign is exactly the moment they get dropped for looking cluttered.
 * Each one exists because the number beside it misleads without it:
 *
 * 1. CONFIDENCE, per market, from the API. `q1_winner` ships as "low"
 *    deliberately - it scores 0.5796 against a 0.5184 always-home baseline,
 *    genuinely better than chance but close enough to it that rendering it
 *    identically to the others would misrepresent it. The label is the
 *    honest presentation of a weak model, not an apology for one.
 *
 * 2. INTERPRETATION, on both winner markets: the literal string
 *    "P(home leads | not tied)". These two models were trained only on
 *    periods that HAD a winner, so the probability is conditional and is
 *    not the same quantity as the full-game win probability one tab over.
 *    Displayed, never a tooltip - a caveat you have to hover to find is a
 *    caveat most readers never see.
 *
 * 3. THE TIE NOTE, near the winner markets. A tied quarter is a push, not
 *    a wrong prediction, and 4.6% of first quarters end level. Without this
 *    the conditional probability above has no visible reason to exist.
 */
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

/**
 * Rendered from the API's own value rather than from a threshold computed
 * here. Which band a model falls into is a fact about how it scored, and
 * the manifest that ships with the models is where that was decided.
 */
function ConfidenceTag({ confidence }) {
  if (!confidence) return null;
  return (
    <span className={`confidence-tag confidence-tag--${confidence}`}>
      {confidence} confidence
    </span>
  );
}

export default QuarterHalfTab;
