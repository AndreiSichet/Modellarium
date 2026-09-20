import { useCallback, useEffect, useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';

import { createPrediction, getHealth, getSchedule } from '../api';
import { DEV_FIXTURES_ON, DEV_HEALTH, DEV_SCHEDULE, devPredictionFor } from '../data/devFixtures';
import { latestPredictableDate } from '../dates';
import SportsRail, { findSport } from './SportsRail';
import './PredictionsLayout.css';

/**
 * How far ahead to ask for fixtures. Wider than MAX_DAYS_AHEAD on purpose:
 * the schedule endpoint lists fixtures, and deciding which of them can be
 * scored is this layout's job, not the API's.
 */
const SCHEDULE_DAYS_AHEAD = 14;

/**
 * The sport slug in a /predictions URL, or null on /predictions itself.
 *
 * NOT useParams(), and the reason is structural rather than stylistic. In
 * react-router a layout route sees the params matched by ITS OWN path and
 * those of its ancestors - never its children's. `:sport` belongs to the
 * child, so useParams() here returns an empty object whatever the URL says.
 *
 * Exported so the parsing is testable on its own rather than only through a
 * rendered route.
 */
export function sportSlugFromPath(pathname) {
  const [, section, sport] = pathname.split('/');
  return section === 'predictions' && sport ? sport : null;
}

/**
 * The two-column shell every /predictions route renders inside, and THE ONE
 * PLACE THE DATA IS FETCHED.
 *
 * THE FETCH LIVES HERE RATHER THAN IN THE PAGES, and that is what holds the
 * request count down. Three routes each fetching for themselves would mean
 * a fresh schedule call and a fresh POST per game every time the reader
 * moved between General, a sport and a league - and POST /api/predictions
 * is append-only, so those are rows, not just requests. Mounted once by the
 * route table, it survives navigation between all three.
 *
 * ONE POST PER GAME, AND IT WRITES A ROW EACH TIME. getSchedule returns
 * fixtures, not predictions. Ten games on screen is ten requests, which is
 * fine at this scale and needs no backend change - but it is N rows per
 * visit where the old browse view wrote one per click.
 *
 * DO NOT FIX THAT IN THE FRONTEND. Caching here would hide the rate without
 * changing it, and would put a policy decision in the wrong layer. The real
 * fix is a backend batch endpoint, or a get-if-recent path that returns an
 * existing prediction instead of always writing.
 */
function PredictionsLayout() {
  const { pathname } = useLocation();
  const slug = sportSlugFromPath(pathname);

  // An unrecognised sport must not cost a request, let alone N prediction
  // rows. The child still renders its own inline not-found with the rail
  // intact; this only decides whether to go to the network for it.
  const unknownSport = Boolean(slug) && !findSport(slug);

  const [status, setStatus] = useState('loading');
  const [games, setGames] = useState([]);
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setStatus('loading');
    setError(null);
    try {
      const [schedule, freshness] = DEV_FIXTURES_ON
        ? [DEV_SCHEDULE, DEV_HEALTH]
        : await Promise.all([getSchedule(SCHEDULE_DAYS_AHEAD), getHealth()]);

      setHealth(freshness);

      const cutoff = latestPredictableDate(freshness?.dataAsOf);
      const predictable = cutoff
        ? schedule.filter((game) => game.gameDate <= cutoff)
        : [];

      const predicted = await Promise.all(
        predictable.map(async (game, index) => {
          const summary = DEV_FIXTURES_ON
            ? devPredictionFor(game, index)
            : await createPrediction({
                homeTeamId: game.homeTeamId,
                awayTeamId: game.awayTeamId,
                gameDate: game.gameDate,
              });

          return {
            key: `${game.homeTeamId}-${game.awayTeamId}-${game.gameDate}`,
            gameId: summary.id,
            /*
             * THE ONE-LEAGUE ASSUMPTION, NAMED RATHER THAN HIDDEN. Neither
             * /schedule nor /api/predictions returns a league, so the only
             * honest thing to say about an NBA fixture from the basketball
             * schedule is that it is NBA. When a second league arrives this
             * has to come off the wire; leaving it derived from nothing
             * would silently file every game under the first league.
             */
            leagueSlug: game.leagueSlug || 'nba',
            homeTeamId: game.homeTeamId,
            homeTeamName: game.homeTeamName,
            awayTeamId: game.awayTeamId,
            awayTeamName: game.awayTeamName,
            gameDate: game.gameDate,
            prediction: summary.latestPrediction,
          };
        })
      );

      setGames(predicted);
      setStatus('ready');
    } catch (failure) {
      // ERROR AND EMPTY ARE DIFFERENT FACTS and must not collapse. "The
      // backend is down" and "there are no games" look identical on screen
      // if a failure falls through to the empty state, and the empty state
      // is reassuring - exactly the misleading green this project rejects.
      setError(failure.message);
      setStatus('error');
    }
  }, []);

  useEffect(() => {
    if (!unknownSport) load();
  }, [unknownSport, load]);

  /*
   * Loading and error are handled HERE; empty is handled by each page.
   * Both of the first two are facts about the fetch and are identical
   * whichever route asked for it. "Nothing to show" is not - it means
   * something different on General, on a sport, and on one league.
   */
  const body =
    !unknownSport && status === 'loading' ? (
      // A line, not a spinner. The call returns fast enough that a spinner
      // appears and vanishes as a flash of noise.
      <p className="predictions-muted">Loading…</p>
    ) : !unknownSport && status === 'error' ? (
      <FetchFailed message={error} onRetry={load} />
    ) : (
      <Outlet
        context={{
          games,
          health,
          predictableDate: latestPredictableDate(health?.dataAsOf),
        }}
      />
    );

  return (
    <div className="predictions">
      <div className="predictions-rail">
        <SportsRail />
      </div>

      {/*
        THE CONTENT COLUMN IS THE LANDMARK, and the rail above is a sibling
        nav. A screen reader user can jump straight to the predictions
        without first walking past the sports list.
      */}
      <main className="predictions-content">{body}</main>
    </div>
  );
}

function FetchFailed({ message, onRetry }) {
  return (
    <div className="predictions-message">
      <h1 className="predictions-message-title">Could not load predictions</h1>
      <p className="predictions-message-body">
        The request failed, so this is not the same as there being no games —
        it means the service did not answer.
      </p>
      {message ? <p className="predictions-error-detail">{message}</p> : null}
      <button type="button" className="predictions-retry" onClick={onRetry}>
        Try again
      </button>
    </div>
  );
}

export default PredictionsLayout;
