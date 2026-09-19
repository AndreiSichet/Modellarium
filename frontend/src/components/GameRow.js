import { Link } from 'react-router-dom';

import { teamFor } from '../data/teams';
import TeamBadge from './TeamBadge';
import './GameRow.css';

/**
 * SPREAD SIGN — the single easiest thing in this phase to get backwards,
 * and it fails silently because both versions look plausible.
 *
 * `homeMargin` is the predicted home margin: positive means the home team
 * is expected to win by that much. Betting convention gives the FAVOURITE a
 * NEGATIVE spread, so the sign flips on the way to the screen. A homeMargin
 * of +1.41 displays as -1.4 for home and +1.4 for away.
 *
 * Pinned by a test rather than trusted to review.
 */
export function formatSpread(homeMargin, isHome) {
  const value = isHome ? -homeMargin : homeMargin;
  // toFixed(1) on -0.04 gives "-0.0"; normalising through +0 avoids
  // printing a negative zero.
  const rounded = Number((value + 0).toFixed(1)) + 0;
  return rounded > 0 ? `+${rounded.toFixed(1)}` : rounded.toFixed(1);
}

/** `homeWinProbability` is the home figure; away is its complement. */
export function formatWin(homeWinProbability, isHome) {
  const value = isHome ? homeWinProbability : 1 - homeWinProbability;
  return `${(value * 100).toFixed(1)}%`;
}

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

function GameRow({ game }) {
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
          to={`/predictions/basketball/game/${game.gameId ?? ''}`}
        >
          More on this
        </Link>
      </div>
    </li>
  );
}

export default GameRow;
