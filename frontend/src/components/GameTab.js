import { formatSpread, formatValue, formatWin } from '../predictionFormat';
import { teamFor } from '../data/teams';
import { dataAsOfFor } from '../dates';

function GameTab({ game, health, league }) {
  const prediction = game.prediction;
  const markets = league?.markets ?? {};

  if (!prediction) return <NoPrediction health={health} league={league} />;

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
          // THE QUALIFIER, ON THE SAME SURFACE THE Q1/1H WINNERS USE. The
          // NFL's winner model is trained on decided games, so its
          // probability means P(home wins | not tied) and a reader shown a
          // bare percentage is shown a different quantity. Absent for
          // basketball, where the full-game moneyline has no such condition,
          // so those pages render exactly as before.
          note={prediction.homeWinInterpretation}
          model={prediction.winnerModel}
        />
        <TeamMetric
          label="Spread"
          homeLabel={home.abbr}
          awayLabel={away.abbr}
          homeValue={formatSpread(margin, true)}
          awayValue={formatSpread(margin, false)}
          note="A negative number means that side is favoured by it."
          model={prediction.marginModel}
        />
      </MarketGroup>

      <MarketGroup title="Totals">
        <Metric
          label="Total points"
          value={formatValue(prediction.totalPoints)}
          model={prediction.totalsModel}
        />
      </MarketGroup>

      {/* Only the groups this league serves. The WNBA has no rebound or
          assist markets, and an empty group is a heading over nothing. */}
      {markets.rebounds ? (
        <MarketGroup title="Rebounds">
          <Metric label="Rebound margin" value={formatValue(prediction.reboundMargin)} />
          <Metric label="Total rebounds" value={formatValue(prediction.totalRebounds)} />
        </MarketGroup>
      ) : null}

      {markets.assists ? (
        <MarketGroup title="Assists">
          <Metric label="Assist margin" value={formatValue(prediction.assistMargin)} />
          <Metric label="Total assists" value={formatValue(prediction.totalAssists)} />
        </MarketGroup>
      ) : null}

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

export function Metric({ label, value, tag, note, model }) {
  return (
    <div className="market-row">
      <dt className="market-label">
        {label}
        {tag}
      </dt>
      <dd className="market-value">
        {value}
        {note ? <span className="market-note">{note}</span> : null}
        {model ? <span className="market-model">{model}</span> : null}
      </dd>
    </div>
  );
}

function TeamMetric({
  label, homeLabel, awayLabel, homeValue, awayValue, note, model,
}) {
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
        {model ? <span className="market-model">{model}</span> : null}
      </dd>
    </div>
  );
}

function NoPrediction({ health, league }) {
  // The explanation and the cutoff both come from the league. The sentence
  // that used to be here named a ten-game warm-up, which is a property of
  // within-season rolling windows and false of the WNBA's carried ones.
  const dataAsOf = dataAsOfFor(health, league);

  return (
    <div className="predictions-message">
      <h2 className="predictions-message-title">No prediction for this game</h2>

      {(league?.seasonNote ?? []).map((line) => (
        <p className="predictions-message-body" key={line}>
          {line}
        </p>
      ))}

      {dataAsOf ? (
        <p className="predictions-message-meta">
          {league ? `${league.label} model` : 'Model'} data is current to{' '}
          {dataAsOf}.
        </p>
      ) : null}
    </div>
  );
}

export default GameTab;
