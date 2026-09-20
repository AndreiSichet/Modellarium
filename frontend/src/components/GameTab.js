import { formatSpread, formatValue, formatWin } from '../predictionFormat';
import { teamFor } from '../data/teams';

/**
 * The seven team-level markets, GROUPED RATHER THAN LISTED FLAT.
 *
 * Seven numbers in one column is a readout; four groups is a page. The
 * grouping is also the honest one - outcome, totals, rebounds and assists
 * are four different questions, and rebound margin has more in common with
 * rebound total than with win probability.
 *
 * SPREAD SIGNS COME FROM predictionFormat, the same helper the game rows on
 * the previous page use. Reimplementing the flip here is exactly how the
 * list and the detail page would come to disagree about the same game.
 */
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

      {/*
        THE FRESHNESS BADGE SURVIVES THE REDESIGN. It has said the same
        thing since v1: these numbers were computed from data that ends on
        a date, and if that date is old the prediction is old with it.
        Stale numbers that look fresh are the failure this whole project
        keeps designing against.
      */}
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

/** One number about the game as a whole. */
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

/**
 * One market with a value per side.
 *
 * Both sides are shown rather than the home figure alone: the away number
 * is derivable, but making the reader derive it is how a 44.9% gets read as
 * the away team's chance.
 */
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

/**
 * A game reachable through the list always has a prediction - the list is
 * built from them. This covers the case where one is reachable some other
 * way, and it explains rather than blanking.
 */
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
