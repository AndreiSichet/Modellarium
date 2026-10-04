import { Link, useOutletContext, useParams } from 'react-router-dom';

import Breadcrumb from './Breadcrumb';

import { findLeague } from '../data/leagues';
import { dataAsOfFor } from '../dates';
import GameList from './GameList';
import { byDate } from './GamesPage';
import { findSport } from './SportsRail';
import './LeaguePage.css';

function LeaguePage() {
  const { sport: sportSlug, league: leagueSlug } = useParams();
  const sport = findSport(sportSlug);
  const league = findLeague(sportSlug, leagueSlug);
  const { games, health } = useOutletContext();

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
        <SeasonEmptyState league={league} health={health} />
      ) : (
        <GameList games={listed} league={league} />
      )}
    </div>
  );
}

/**
 * Why this league has nothing, in that league's own words.
 *
 * THE COPY COMES FROM THE LEAGUE, NOT FROM A SHARED SENTENCE. The NBA's
 * explanation turns on a ten-game warm-up, which is true of within-season
 * rolling windows and false of the WNBA's carried ones - so one shared
 * sentence would state the wrong mechanism for one of the two leagues. The
 * cutoff shown is also that league's own, from its entry in /api/health.
 */
export function SeasonEmptyState({ league, health }) {
  const dataAsOf = dataAsOfFor(health, league);

  return (
    <div className="predictions-message">
      {league.seasonNote.map((line) => (
        <p className="predictions-message-body" key={line}>
          {line}
        </p>
      ))}

      {dataAsOf ? (
        <p className="predictions-message-meta">
          {league.label} model data is current to {dataAsOf}.
        </p>
      ) : null}
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
