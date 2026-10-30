import { formatValue } from '../predictionFormat';
import TabStatus from './TabStatus';
import TeamBadge from './TeamBadge';

/** Does this board have nobody to show, and does it say why? */
function isEmpty(team) {
  return !team || !(team.players?.length > 0);
}

/** The API's own words when it has them, ours when it does not. */
function reasonFor(...teams) {
  for (const team of teams) {
    if (team?.availabilityNote) return team.availabilityNote;
  }
  return null;
}

function PlayerStatTab({ stat, state, onRetry, game }) {
  if (state.status !== 'ready') {
    return (
      <TabStatus
        status={state.status}
        error={state.error}
        failure={state.failure}
        onRetry={onRetry}
        what="player predictions"
      />
    );
  }

  const board = state.data;

  // BOTH BOARDS EMPTY. This branch used to require both boards to be ABSENT
  // (`!board?.homeTeam && !board?.awayTeam`), and the real response has both
  // PRESENT with empty rosters - so the explanation could never appear and a
  // reader got two headed panels with a one-line note and nothing else.
  // Measured 2026-10-30: all ten fixtures of the day were in exactly that
  // state, every board carrying `availability_code: "no_usable_history"`.
  if (isEmpty(board?.homeTeam) && isEmpty(board?.awayTeam)) {
    const note = reasonFor(board?.homeTeam, board?.awayTeam);
    return (
      <div className="predictions-message">
        <h2 className="predictions-message-title">No player predictions yet</h2>
        <p className="predictions-message-body">
          Player props need each player's own recent history, which does not
          exist until they have played enough games this season.
        </p>
        {note ? <p className="predictions-message-meta">{note}</p> : null}
      </div>
    );
  }

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

  const players = (team.players ?? [])
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

      {/* ONE SIDE EMPTY, THE OTHER NOT. A board with nobody on it is said in
          a sentence rather than left as an empty list under a heading - which
          is what a reader saw before, with no indication whether the roster
          was empty or the request had half failed. */}
      {players.length === 0 ? (
        <p className="player-board-empty">
          No player predictions for this team yet — none of its players has
          enough of this season recorded.
        </p>
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
