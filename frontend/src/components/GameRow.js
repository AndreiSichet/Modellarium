import { Link } from 'react-router-dom';

import { formatSpread, formatWin } from '../predictionFormat';
import { formatKickoff } from '../dates';
import { nameOf } from '../data/teams';
import { refusalText } from '../refusalText';
import TeamBadge from './TeamBadge';
import './GameRow.css';

function TeamLine({ teamId, name, abbr, spread, win }) {
  return (
    <>
      <div className="game-row-team">
        <TeamBadge teamId={teamId} abbr={abbr} />
        <span className="game-row-team-name">{nameOf(teamId, name)}</span>
      </div>
      <div className="game-row-cell">{spread}</div>
      <div className="game-row-cell">{win}</div>
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
  const why = refusalText(game.refusal);

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

function GameRow({ game, league }) {
  const { prediction } = game;

  // BEFORE ANY FIELD OF `prediction` IS READ. Every line below assumes it
  // exists, so a refused fixture has to branch here or the row throws and
  // takes the list with it - which is the same failure the per-fixture catch
  // in PredictionsLayout exists to prevent, one layer down.
  if (!prediction) {
    return <NotPredictableRow game={game} />;
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
