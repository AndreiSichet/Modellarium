import { Link, useOutletContext, useParams } from 'react-router-dom';

import Breadcrumb from './Breadcrumb';

import { findLeague } from '../data/leagues';
import GameList from './GameList';
import { byDate } from './GamesPage';
import { findSport } from './SportsRail';
import './LeaguePage.css';

function LeaguePage() {
  const { sport: sportSlug, league: leagueSlug } = useParams();
  const sport = findSport(sportSlug);
  const league = findLeague(sportSlug, leagueSlug);
  const { games } = useOutletContext();

  if (!sport || !league) {
    return <NotFound sportSlug={sportSlug} leagueSlug={leagueSlug} sport={sport} />;
  }

  const listed = byDate(games.filter((game) => game.leagueSlug === league.slug));

  return (
    <div className="league-page">
      <Breadcrumb
        items={[
          { label: 'Predictions', to: '/predictions' },
          { label: sport.label, to: `/predictions/${sport.slug}` },
          { label: league.label },
        ]}
      />

      <h1 className="league-page-title">{league.label} Predictions</h1>

      {listed.length === 0 ? (
        <p className="predictions-message-body">
          Nothing is predictable for {league.label} yet. Predictions become
          available once the season is under way and each team has played
          enough games for its recent form to be computed.
        </p>
      ) : (
        <GameList games={listed} league={league} />
      )}
    </div>
  );
}

function NotFound({ sportSlug, leagueSlug, sport }) {
  return (
    <div className="predictions-message">
      <h1 className="predictions-message-title">No such league</h1>
      <p className="predictions-message-body">
        {sport ? (
          <>
            <strong>{sport.label}</strong> has no league called{' '}
            <strong>{leagueSlug}</strong>.
          </>
        ) : (
          <>
            Nothing here covers <strong>{sportSlug}</strong>.
          </>
        )}
      </p>
      <p className="predictions-message-body">
        <Link className="league-section-more" to="/predictions">
          Back to all predictions
        </Link>
      </p>
    </div>
  );
}

export default LeaguePage;
