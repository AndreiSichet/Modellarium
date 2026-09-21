import { formatSpread, formatValue, formatWin } from '../predictionFormat';
import { teamFor } from '../data/teams';

function GameTab({ game, health }) {
  const prediction = game.prediction;

  if (!prediction) return <NoPrediction health={health} />;

  const home = teamFor(game.homeTeamId);
  const away = teamFor(game.awayTeamId);
  const { homeWinProbability: probability, homeMargin: margin } = prediction;

  return (
    <div className="market-groups">
      <MarketGroup title="Outcome">
        <TeamMetric
          label="Win probability"
          homeLabel={home.abbr}
          awayLabel={away.abbr}
          homeValue={formatWin(probability, true)}
          awayValue={formatWin(probability, false)}
        />
        <TeamMetric
          label="Spread"
          homeLabel={home.abbr}
          awayLabel={away.abbr}
          homeValue={formatSpread(margin, true)}
          awayValue={formatSpread(margin, false)}
          note="A negative number means that side is favoured by it."
        />
      </MarketGroup>

      <MarketGroup title="Totals">
        <Metric label="Total points" value={formatValue(prediction.totalPoints)} />
      </MarketGroup>

      <MarketGroup title="Rebounds">
        <Metric label="Rebound margin" value={formatValue(prediction.reboundMargin)} />
        <Metric label="Total rebounds" value={formatValue(prediction.totalRebounds)} />
      </MarketGroup>

      <MarketGroup title="Assists">
        <Metric label="Assist margin" value={formatValue(prediction.assistMargin)} />
        <Metric label="Total assists" value={formatValue(prediction.totalAssists)} />
      </MarketGroup>

      {prediction.stale ? (
        <p className="market-freshness">
          <span className="stale-badge">STALE</span>
          Computed from data as of {prediction.dataAsOf}
        </p>
      ) : null}
    </div>
  );
}

export function MarketGroup({ title, children }) {
  return (
    <section className="market-group">
      <h2 className="market-group-title">{title}</h2>
      <dl className="market-list">{children}</dl>
    </section>
  );
}

export function Metric({ label, value, tag, note }) {
  return (
    <div className="market-row">
      <dt className="market-label">
        {label}
        {tag}
      </dt>
      <dd className="market-value">
        {value}
        {note ? <span className="market-note">{note}</span> : null}
      </dd>
    </div>
  );
}

function TeamMetric({ label, homeLabel, awayLabel, homeValue, awayValue, note }) {
  return (
    <div className="market-row">
      <dt className="market-label">{label}</dt>
      <dd className="market-value">
        <span className="market-side">
          <span className="market-side-team">{awayLabel}</span>
          <span className="market-side-value">{awayValue}</span>
        </span>
        <span className="market-side">
          <span className="market-side-team">{homeLabel}</span>
          <span className="market-side-value">{homeValue}</span>
        </span>
        {note ? <span className="market-note">{note}</span> : null}
      </dd>
    </div>
  );
}

function NoPrediction({ health }) {
  return (
    <div className="predictions-message">
      <h2 className="predictions-message-title">No prediction for this game</h2>
      <p className="predictions-message-body">
        Predictions need recent form to work from, so they become available
        once each team has played enough games for that form to be computed —
        around ten games into a season.
      </p>
      {health?.dataAsOf ? (
        <p className="predictions-message-meta">
          Model data is current to {health.dataAsOf}.
        </p>
      ) : null}
    </div>
  );
}

export default GameTab;
