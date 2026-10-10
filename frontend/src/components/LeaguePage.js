import { Link, useOutletContext, useParams } from 'react-router-dom';

import Breadcrumb from './Breadcrumb';

import { findLeague, leagueCopy } from '../data/leagues';
import { dataAsOfFor } from '../dates';
import Attribution from './Attribution';
import GameList from './GameList';
import { byDate } from './GamesPage';
import { findSport } from './SportsRail';
import WeekGroups from './WeekGroups';
import './LeaguePage.css';

function LeaguePage() {
  const { sport: sportSlug, league: leagueSlug } = useParams();
  const sport = findSport(sportSlug);
  const league = findLeague(sportSlug, leagueSlug);
  const { games, schedule, health } = useOutletContext();

  if (!sport || !league) {
    return <NotFound sportSlug={sportSlug} leagueSlug={leagueSlug} sport={sport} />;
  }

  const listed = byDate(games.filter((game) => game.leagueSlug === league.slug));

  // THE LICENCE TEXT COMES OFF A PREDICTION BODY, because that is where the
  // API puts it. Any of this league's predictions carries the same `source`,
  // so the first one is enough; with none, Attribution renders nothing
  // rather than a claim this page cannot substantiate.
  const source = listed.find((game) => game.prediction?.source)
    ?.prediction?.source;

  // A WEEKLY LEAGUE GETS WEEK HEADINGS AND ITS UNPREDICTABLE FIXTURES SHOWN.
  // Declared on the league rather than branched on a slug, so basketball's
  // page is the flat date-ordered list it has always been.
  const weekly = league.weeklySlate === true;

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

      {weekly ? (
        <WeeklyLeague
          league={league}
          health={health}
          schedule={schedule}
          games={listed}
        />
      ) : listed.length === 0 ? (
        <SeasonEmptyState
          league={league}
          health={health}
          schedule={schedule}
        />
      ) : (
        <GameList games={listed} league={league} />
      )}

      {league.attributionRequired ? <Attribution source={source} /> : null}
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
export function SeasonEmptyState({ league, health, schedule }) {
  const dataAsOf = dataAsOfFor(health, league);

  // Resolved against the schedule, because one league's copy needs a date
  // only the data has: the G League's regular-season start comes from its
  // first cached fixture rather than from a typed-in guess.
  const note = leagueCopy(league.seasonNote, { schedule });

  return (
    <div className="predictions-message">
      {note.map((line) => (
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

/**
 * A league whose page is organised by week rather than by date.
 *
 * The empty state still applies when nothing at all is scheduled - the
 * offseason - but a week with fixtures and no predictions is NOT empty: it
 * shows the fixtures marked as not yet predictable, which is the whole point
 * of grouping them.
 */
function WeeklyLeague({ league, health, schedule, games }) {
  const scheduled = (schedule || []).filter(
    (game) => game.leagueSlug === league.slug && game.week != null
  );

  if (!scheduled.length) {
    return (
      <SeasonEmptyState league={league} health={health} schedule={schedule} />
    );
  }

  return (
    <>
      <WeekGroups schedule={schedule} games={games} league={league} />

      {/* THIS LEAGUE'S OWN CUTOFF, and it was missing until a test asked for
          it: the meta line lived only in SeasonEmptyState, so a weekly page
          with fixtures showed no data-as-of anywhere. Never another league's
          - dataAsOfFor reads the league's own block and yields null rather
          than falling back, which is the shared-date bug the G League phase
          had to fix. */}
      <LeagueFreshness league={league} health={health} />
    </>
  );
}

function LeagueFreshness({ league, health }) {
  const dataAsOf = dataAsOfFor(health, league);
  if (!dataAsOf) return null;

  return (
    <p className="predictions-message-meta">
      {league.label} model data is current to {dataAsOf}.
    </p>
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
