import { Link } from 'react-router-dom';

import { formatSpread, formatWin } from '../predictionFormat';
import { formatKickoff } from '../dates';
import { nameOf } from '../data/teams';
import { refusalText, scheduleReason } from '../refusalText';
import TeamBadge from './TeamBadge';
import './GameRow.css';

function TeamLine({ teamId, name, abbr, spread, win, cells }) {
  // EITHER THE TWO NAMED CELLS OR A DECLARED LIST. The four original
  // leagues pass spread and win by name and render exactly as before; a
  // league with  passes the cells it declared. One component so
  // the team name, badge and grid placement cannot drift between the two.
  const values = cells ?? [spread, win];

  return (
    <>
      <div className="game-row-team">
        <TeamBadge teamId={teamId} abbr={abbr} />
        <span className="game-row-team-name">{nameOf(teamId, name)}</span>
      </div>
      {values.map((value, index) => (
        // eslint-disable-next-line react/no-array-index-key
        <div className="game-row-cell" key={index}>{value}</div>
      ))}
    </>
  );
}

/**
 * A fixture the service declined to predict.
 *
 * ITS OWN ROW RATHER THAN AN ABSENCE. The row is still there, still named,
 * and says why - so a reader can see that a fixture exists and that the
 * models chose not to score it, which is different from the fixture being
 * missing and very different from the page having failed.
 */
function NotPredictableRow({ game }) {
  const when = formatKickoff(game);
  // THE ROW'S OWN REASON FIRST. A schedule row for a dependency-rule league
  // carries `reasonCode` and `blockedBy`, which name the team and the date;
  // `game.refusal` is only set once a prediction has actually been attempted
  // and refused, which on a browse list is almost never. The schedule
  // sentence is built from data and never from the API's prose - see
  // scheduleReason, and the generic fallback it uses instead.
  const why = scheduleReason(game) ?? refusalText(game.refusal);

  return (
    <li className="game-row game-row--not-predictable">
      <div className="game-row-unpredictable">
        <div className="game-row-teams-plain">
          <TeamBadge teamId={game.awayTeamId} abbr={game.awayTeamAbbr} />
          <span className="game-row-team-name">
            {nameOf(game.awayTeamId, game.awayTeamName)}
          </span>
          <span className="game-row-at">at</span>
          <TeamBadge teamId={game.homeTeamId} abbr={game.homeTeamAbbr} />
          <span className="game-row-team-name">
            {nameOf(game.homeTeamId, game.homeTeamName)}
          </span>
        </div>
        {/* `when` is an object - text, flex, timed - not a string. The
            first version of this row rendered it directly and React raised
            "Objects are not valid as a React child". A JSX comment cannot
            carry the key list in braces either: the first brace closes the
            expression container and Babel fails to parse the file. The
            predicted row above has always destructured it; so does this. */}
        <div className="game-row-when">
          {when.text}
          {when.flex ? (
            <span className="game-row-flex" title="The kickoff time may move">
              flex
            </span>
          ) : null}
        </div>
        <p className="game-row-refusal">
          <span className="game-row-refusal-tag">Not yet predictable</span>
          {why ? <span className="game-row-refusal-why">{why}</span> : null}
        </p>
      </div>
    </li>
  );
}

/** What each declarable row market is called and how it is read. */
const ROW_MARKETS = {
  win: {
    heading: 'Win',
    home: (p) => formatWin(p.homeWinProbability, true),
    away: (p) => formatWin(p.homeWinProbability, false),
  },
  // HOME ONLY, AND THE BLANK IS DELIBERATE. The model predicts "home wins
  // by 2 or more" and nothing about the away side; 1 - p would assert a
  // market that is not served, which is the same discipline that keeps the
  // NHL's totals row saying "not offered" instead of hiding.
  puckLine: {
    heading: 'By 2+',
    home: (p) => formatWin(p.homeWinsByTwoOrMore, true),
    away: () => null,
  },
};

/**
 * A row for a league that declares which markets it shows.
 *
 * The default row is Spread / Win / Total, which is the three basketball
 * leagues and the NFL. A league with `rowMarkets` gets exactly those
 * columns and no empty cells for markets it does not serve.
 */
function DeclaredMarketsRow({ game, league }) {
  const { prediction } = game;
  const columns = league.rowMarkets
    .map((name) => ROW_MARKETS[name])
    .filter(Boolean);
  const when = formatKickoff(game);

  return (
    <li className="game-row">
      <div
        className="game-row-grid game-row-grid--declared"
        style={{ '--row-market-count': columns.length }}
      >
        <div className="game-row-headings" aria-hidden="true">
          <span />
          {columns.map((column) => (
            <span className="game-row-heading" key={column.heading}>
              {column.heading}
            </span>
          ))}
        </div>

        <TeamLine
          teamId={game.homeTeamId}
          name={game.homeTeamName}
          abbr={game.homeTeamAbbr}
          cells={columns.map((column) => column.home(prediction))}
        />
        <TeamLine
          teamId={game.awayTeamId}
          name={game.awayTeamName}
          abbr={game.awayTeamAbbr}
          cells={columns.map((column) => column.away(prediction))}
        />
      </div>

      <div className="game-row-footer">
        <span className="game-row-when">{when.text}</span>
        <Link
          className="game-row-more"
          to={`/predictions/${league.sport}/${league.slug}/${game.gameId}`}
        >
          More on this
        </Link>
      </div>
    </li>
  );
}

function GameRow({ game, league }) {
  const { prediction } = game;

  // BEFORE ANY FIELD OF `prediction` IS READ. Every line below assumes it
  // exists, so a refused fixture has to branch here or the row throws and
  // takes the list with it - which is the same failure the per-fixture catch
  // in PredictionsLayout exists to prevent, one layer down.
  if (!prediction) {
    return <NotPredictableRow game={game} />;
  }

  // A LEAGUE WHOSE ROW IS A DIFFERENT SHAPE SAYS SO, rather than this
  // function guessing from which fields happen to be present. The three
  // columns below are Spread / Win / Total, and the NHL serves none of them
  // in that form: no spread, no total, and a puck line defined as "home wins
  // by 2 or more". Reading `prediction.totalPoints.toFixed(1)` on an NHL row
  // threw and took the whole list with it - the same failure the
  // `!prediction` branch above exists to prevent, one field deeper.
  if (Array.isArray(league?.rowMarkets)) {
    return <DeclaredMarketsRow game={game} league={league} />;
  }

  const margin = prediction.homeMargin;
  const probability = prediction.homeWinProbability;

  // THE SAME SLOT THE DATE ALREADY OCCUPIED, not a new column. A basketball
  // fixture carries no kickoff, so this is its bare ISO date exactly as
  // before and no empty time cell appears on those rows.
  const when = formatKickoff(game);

  return (
    <li className="game-row">
      <div className="game-row-grid">
        <div className="game-row-headings" aria-hidden="true">
          <span />
          <span className="game-row-heading">Spread</span>
          <span className="game-row-heading">Win</span>
          <span className="game-row-heading">Total</span>
        </div>

        <TeamLine
          teamId={game.homeTeamId}
          name={game.homeTeamName}
          abbr={game.homeTeamAbbr}
          spread={formatSpread(margin, true)}
          win={formatWin(probability, true)}
        />

        <div className="game-row-total">{prediction.totalPoints.toFixed(1)}</div>

        <TeamLine
          teamId={game.awayTeamId}
          name={game.awayTeamName}
          abbr={game.awayTeamAbbr}
          spread={formatSpread(margin, false)}
          win={formatWin(probability, false)}
        />
      </div>

      <div className="game-row-footer">

        <span className="game-row-when">
          {when.text}
          {when.flex ? (
            <span className="game-row-flex" title="The kickoff time may move">
              flex
            </span>
          ) : null}
        </span>
        <Link
          className="game-row-more"
          to={`/predictions/${league.sport}/${league.slug}/${game.gameId}`}
        >
          More on this
        </Link>
      </div>
    </li>
  );
}

export default GameRow;
