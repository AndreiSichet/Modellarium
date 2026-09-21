import { formatValue } from '../predictionFormat';
import TabStatus from './TabStatus';
import TeamBadge from './TeamBadge';

function PlayerStatTab({ stat, state, onRetry, game }) {
  if (state.status !== 'ready') {
    return (
      <TabStatus
        status={state.status}
        error={state.error}
        onRetry={onRetry}
        what="player predictions"
      />
    );
  }

  const board = state.data;

  if (!board?.homeTeam && !board?.awayTeam) {
    return (
      <div className="predictions-message">
        <h2 className="predictions-message-title">No player predictions</h2>
        <p className="predictions-message-body">
          Player props need each player's own recent history, which does not
          exist until they have played enough games this season.
        </p>
      </div>
    );
  }

  return (
    <div className="player-boards">

      <RosterBoard team={board.awayTeam} stat={stat} teamId={game.awayTeamId} />
      <RosterBoard team={board.homeTeam} stat={stat} teamId={game.homeTeamId} />
    </div>
  );
}

function RosterBoard({ team, stat, teamId }) {
  if (!team) return null;

  const players = team.players
    .slice()
    .sort((a, b) => (b[stat.field] ?? 0) - (a[stat.field] ?? 0));

  return (
    <section className="player-board" aria-label={`${team.teamAbbreviation} ${stat.short}`}>
      <header className="player-board-head">
        <TeamBadge teamId={teamId} />
        <h2 className="player-board-title">{team.teamAbbreviation}</h2>
        <span className="player-board-stat">{stat.short}</span>
      </header>

      {!team.availabilityKnown && team.availabilityNote ? (
        <p className="player-board-note">{team.availabilityNote}</p>
      ) : null}

      <ul className="player-list">
        {players.map((player) => (
          <li className="player-row" key={player.playerId}>
            <span className="player-row-name">{player.playerName}</span>
            <span className={`model-tag model-tag--${player.modelUsed}`}>
              {player.modelUsed}
            </span>
            <span className="player-row-value">{formatValue(player[stat.field])}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

export default PlayerStatTab;
