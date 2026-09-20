import { formatValue } from '../predictionFormat';
import TabStatus from './TabStatus';
import TeamBadge from './TeamBadge';

/**
 * One stat, both rosters, sorted best first.
 *
 * ONE STAT PER TAB RATHER THAN ONE PLAYER PER SECTION, which is the shape
 * the old flow used. The API returns a board per team with five values per
 * player, so a stat is a column of that board - and "who is predicted to
 * score most" is the question a props page is actually asked. The previous
 * arrangement made the reader pick players one at a time to find out.
 *
 * modelUsed IS SHOWN PER PLAYER, and it is not an implementation detail
 * here. The shipped player-prop model is a hybrid: a player with a complete
 * rolling history is scored by a linear model, everyone else by XGBoost,
 * and the routing is decided per player per request. Which half answered is
 * a real property of the number, and the API already enforces that
 * transparency by returning the field at all.
 */
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
      {/* Away first, matching the header's "away @ home" reading order. */}
      <RosterBoard team={board.awayTeam} stat={stat} teamId={game.awayTeamId} />
      <RosterBoard team={board.homeTeam} stat={stat} teamId={game.homeTeamId} />
    </div>
  );
}

function RosterBoard({ team, stat, teamId }) {
  if (!team) return null;

  // Descending, so the board answers "who is predicted to do most of this"
  // without the reader scanning. A copy, because the response object is
  // shared by all five tabs and sorting it in place would reorder the
  // others as a side effect of rendering this one.
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

      {/*
        THE AVAILABILITY NOTE IS RENDERED VERBATIM, never paraphrased.
        It says the absence of an injury report is not a clean bill of
        health, and softening that inverts its meaning - an unfiltered
        roster would otherwise read as a confirmed lineup. It is false on
        every response until live injury serving is packaged, so this is
        the normal case rather than an edge one.
      */}
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
