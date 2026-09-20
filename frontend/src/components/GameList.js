import GameRow from './GameRow';
import './GameList.css';

/**
 * A flat list of game rows.
 *
 * NO GROUPING BRANCH ANY MORE. It existed for Phase 3's `Upcoming` tab,
 * which grouped by league under its own headings. Leagues are subsections
 * with their own headings now, and each hands this the games it already
 * owns - so a second grouping mechanism here would be a second answer to
 * the same question.
 */
function GameList({ games, league }) {
  return (
    <ul className="game-list">
      {games.map((game) => (
        // `league` travels with the list rather than being read from the
        // URL inside each row: on General the page's route has no league
        // in it at all, and each subsection knows its own.
        <GameRow key={game.key} game={game} league={league} />
      ))}
    </ul>
  );
}

export default GameList;
