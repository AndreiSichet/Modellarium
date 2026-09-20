import { Link, useOutletContext, useParams } from 'react-router-dom';

import Breadcrumb from './Breadcrumb';

import { findLeague } from '../data/leagues';
import GameList from './GameList';
import { byDate } from './GamesPage';
import { findSport } from './SportsRail';
import './LeaguePage.css';

/**
 * /predictions/:sport/:league - everything available for one league.
 *
 * NO CAP AND NO SHORTCUT BAR, and both absences are the point of the page.
 * General and Upcoming cap each league at five and offer this as the way to
 * the rest; capping here would leave no way to the rest at all.
 *
 * The shortcut bar is omitted because there is exactly one section, so
 * every chip would scroll to where the reader already is. A control that
 * does nothing is worse than its absence - the same call as not rendering
 * a disabled placeholder league.
 *
 * FLAGGED AS A JUDGEMENT CALL: if the three routes looking identical
 * matters more than the bar doing something, put it back with the current
 * league active. It is one line, and the hook it needs is already exported.
 *
 * "EVERYTHING AVAILABLE" MEANS EVERY GAME WITH A PREDICTION, which is every
 * game the layout could score - not every fixture. Today that set is one
 * date wide, because MAX_DAYS_AHEAD is 1, so this page and the league's
 * subsection on General show the same games. That stops being true the
 * moment the horizon grows, and the structure is right either way.
 */
function LeaguePage() {
  const { sport: sportSlug, league: leagueSlug } = useParams();
  const sport = findSport(sportSlug);
  const league = findLeague(sportSlug, leagueSlug);
  const { games } = useOutletContext();

  // Checked as a pair: `nba` is a real league, but /predictions/tennis/nba
  // is not a real page, and rendering it would put a basketball league
  // under a tennis breadcrumb.
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

/**
 * Inline, with the rail intact - the same behaviour an unknown sport has
 * had since Phase 2. Redirecting would hide which part of the URL was
 * wrong, and this can say so.
 */
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
