import { Link } from 'react-router-dom';

import { formatSpread, formatWin } from '../predictionFormat';
import { formatKickoff } from '../dates';
import { teamFor } from '../data/teams';
import TeamBadge from './TeamBadge';
import './GameRow.css';

function TeamLine({ teamId, name, abbr, spread, win }) {
  return (
    <>
      <div className="game-row-team">
        <TeamBadge teamId={teamId} abbr={abbr} />
        <span className="game-row-team-name">{name || teamFor(teamId).name}</span>
      </div>
      <div className="game-row-cell">{spread}</div>
      <div className="game-row-cell">{win}</div>
    </>
  );
}

function GameRow({ game, league }) {
  const { prediction } = game;
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
