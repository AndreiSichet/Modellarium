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

/**
 * THE LINK NEEDS ITS SPORT AND LEAGUE, AND IS GIVEN THEM RATHER THAN
 * GUESSING. Phase 3 hardcoded /predictions/basketball/game/<id>, which was
 * a placeholder standing in for a route that did not exist. The route is
 * /predictions/:sport/:league/:gameId now, and on General a row can belong
 * to any league - so deriving the sport from the current URL would send
 * every row to whichever sport the reader happened to be looking at.
 *
 * Both come from the subsection that renders the list, which already knows
 * which league it is.
 */
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

        {/*
          THE TOTAL SPANS BOTH LINES, and that is a claim about the data
          rather than a layout preference. Spread and win are per-team;
          the predicted total is one number about the game. A sportsbook
          prints O/U twice because it is selling two sides of a bet —
          Modellarium is not, so repeating the number would invent a
          symmetry that does not exist.
        */}
        <div className="game-row-total">{prediction.totalPoints.toFixed(1)}</div>

        <TeamLine
          teamId={game.awayTeamId}
          name={game.awayTeamName}
          spread={formatSpread(margin, false)}
          win={formatWin(probability, false)}
        />
      </div>

      <div className="game-row-footer">
        {/*
          THE DATE, NOT A TIP-OFF TIME. ScheduledGameDto carries gameDate as
          a LocalDate and the inference service's /schedule returns
          game_date alone — there is no time anywhere in the chain, so
          rendering one would mean inventing it.
        */}
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
