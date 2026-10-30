import { formatValue } from '../predictionFormat';
import { MarketGroup, Metric } from './GameTab';
import TabStatus from './TabStatus';

/**
 * The refusal this tab exists to explain.
 *
 * Inference sends `code: "insufficient_history"` with `needed` - the rolling
 * window, read from the model's own feature definitions and never a literal -
 * and each side's games this season. Measured 2026-10-30: every NBA fixture
 * refused, the teams had 3 to 6 games against a window of 10, and the last
 * one cannot succeed before 2026-11-13. So this is what a reader meets for
 * the first fortnight of a season, every season.
 */
function InsufficientHistory({ fields, game, health }) {
  const needed = fields?.needed;
  const home = fields?.home_games_played;
  const away = fields?.away_games_played;
  const homeName = game?.homeTeamAbbr || game?.homeTeamName || 'the home team';
  const awayName = game?.awayTeamAbbr || game?.awayTeamName || 'the away team';

  return (
    <div className="predictions-message">
      <h2 className="predictions-message-title">Not enough of this season yet</h2>
      <p className="predictions-message-body">
        {typeof needed === 'number'
          ? `Quarter and half predictions start once both teams have played ${needed} games this season.`
          : 'Quarter and half predictions start once both teams have played enough games this season.'}
        {typeof home === 'number' && typeof away === 'number'
          ? ` ${homeName} has played ${home}, ${awayName} ${away}.`
          : ''}
      </p>
      <p className="predictions-message-body">
        These six models are linear and cannot score a partial row, so they
        refuse rather than predicting worse. The other tabs on this page are
        unaffected.
      </p>
      {health?.dataAsOf ? (
        <p className="predictions-message-meta">
          Model data is current to {health.dataAsOf}.
        </p>
      ) : null}
    </div>
  );
}

function QuarterHalfTab({ state, onRetry, health, game }) {
  // THE CODE IS CHECKED BEFORE THE STATUS, because an insufficient-history
  // refusal arrives as an error and would otherwise fall straight through to
  // "could not load". A refusal with any OTHER code, or none, keeps the
  // honest failure message - a wrong explanation is worse than a vague one.
  if (state.status === 'error' && state.failure?.code === 'insufficient_history') {
    return (
      <InsufficientHistory
        fields={state.failure.fields}
        game={game}
        health={health}
      />
    );
  }

  if (state.status !== 'ready') {
    return (
      <TabStatus
        status={state.status}
        error={state.error}
        failure={state.failure}
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
