import { Link } from 'react-router-dom';

import { formatSpread, formatWin } from '../predictionFormat';
import { teamFor } from '../data/teams';
import TeamBadge from './TeamBadge';
import './GameRow.css';

function TeamLine({ teamId, name, spread, win }) {
  return (
    <>
      <div className="game-row-team">
        <TeamBadge teamId={teamId} />
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
          spread={formatSpread(margin, true)}
          win={formatWin(probability, true)}
        />

        <div className="game-row-total">{prediction.totalPoints.toFixed(1)}</div>

        <TeamLine
          teamId={game.awayTeamId}
          name={game.awayTeamName}
          spread={formatSpread(margin, false)}
          win={formatWin(probability, false)}
        />
      </div>

      <div className="game-row-footer">

        <span className="game-row-when">{game.gameDate}</span>
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
