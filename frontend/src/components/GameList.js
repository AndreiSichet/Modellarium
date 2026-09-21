import GameRow from './GameRow';
import './GameList.css';

function GameList({ games, league }) {
  return (
    <ul className="game-list">
      {games.map((game) => (

        <GameRow key={game.key} game={game} league={league} />
      ))}
    </ul>
  );
}

export default GameList;
