import GameRow from './GameRow';
import './GameList.css';

/**
 * The populated middle column.
 *
 * `Upcoming` groups by league with a header per group; `NBA` is flat,
 * because the tab itself already says which league you are looking at.
 *
 * WITH ONE LEAGUE THERE IS EXACTLY ONE GROUP, and it renders as a group
 * anyway rather than being special-cased away. Special-casing "only one"
 * is how a second league turns into a structural change instead of a data
 * change — the same reason SPORTS and LEAGUES are constants.
 */
function GameList({ games, grouped }) {
  if (!grouped) {
    return (
      <ul className="game-list">
        {games.map((game) => (
          <GameRow key={game.key} game={game} />
        ))}
      </ul>
    );
  }

  const groups = games.reduce((acc, game) => {
    (acc[game.league] = acc[game.league] || []).push(game);
    return acc;
  }, {});

  return (
    <div className="game-groups">
      {Object.entries(groups).map(([league, entries]) => (
        <section className="game-group" key={league}>
          <h2 className="game-group-heading">{league}</h2>
          <ul className="game-list">
            {entries.map((game) => (
              <GameRow key={game.key} game={game} />
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}

export default GameList;
