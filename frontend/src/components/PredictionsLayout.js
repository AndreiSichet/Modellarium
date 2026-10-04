import { useCallback, useEffect, useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';

import { createPredictionFor, getHealth, getSchedule, normalisePredictionBody } from '../api';
import { DEV_FIXTURES_ON, DEV_HEALTH, DEV_SCHEDULE, devPredictionFor } from '../data/devFixtures';
import { leagueBySlug } from '../data/leagues';
import { predictableDateFor } from '../dates';
import SportsRail, { findSport } from './SportsRail';
import './PredictionsLayout.css';

const SCHEDULE_DAYS_AHEAD = 14;

export function sportSlugFromPath(pathname) {
  const [, section, sport] = pathname.split('/');
  return section === 'predictions' && sport ? sport : null;
}

function PredictionsLayout() {
  const { pathname } = useLocation();
  const slug = sportSlugFromPath(pathname);

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

      // Filtered PER GAME against its own league's cutoff, not once against a
      // single one. The two leagues' data ends on different dates, so one
      // cutoff is wrong for whichever league it did not come from.
      const predictable = schedule.filter((game) => {
        const league = leagueBySlug(game.leagueSlug || 'nba');
        const cutoff = predictableDateFor(freshness, league);
        return cutoff ? game.gameDate <= cutoff : false;
      });

      const predicted = await Promise.all(
        predictable.map(async (game, index) => {
          const leagueSlug = game.leagueSlug || 'nba';

          // Both paths end in the same normaliser: the fixtures are written
          // in each backend DTO's own shape, so dev cannot quietly agree with
          // itself while production disagrees.
          const summary = DEV_FIXTURES_ON
            ? normalisePredictionBody(leagueSlug, devPredictionFor(game, index))
            : await createPredictionFor(leagueSlug, {
                homeTeamId: game.homeTeamId,
                awayTeamId: game.awayTeamId,
                gameDate: game.gameDate,
              });

          return {
            key: `${game.homeTeamId}-${game.awayTeamId}-${game.gameDate}`,
            gameId: summary.gameId,

            leagueSlug,
            homeTeamId: game.homeTeamId,
            homeTeamName: game.homeTeamName,
            awayTeamId: game.awayTeamId,
            awayTeamName: game.awayTeamName,
            gameDate: game.gameDate,
            prediction: summary.prediction,
          };
        })
      );

      setGames(predicted);
      setStatus('ready');
    } catch (failure) {
      setError(failure.message);
      setStatus('error');
    }
  }, []);

  useEffect(() => {
    if (!unknownSport) load();
  }, [unknownSport, load]);

  const body =
    !unknownSport && status === 'loading' ? (

      <p className="predictions-muted">Loading…</p>
    ) : !unknownSport && status === 'error' ? (
      <FetchFailed message={error} onRetry={load} />
    ) : (
      <Outlet
        context={{
          games,
          health,
          // A function rather than a date, so a page with two leagues on it
          // cannot accidentally judge both by one cutoff.
          predictableDateFor: (league) => predictableDateFor(health, league),
        }}
      />
    );

  return (
    <div className="predictions">
      <div className="predictions-rail">
        <SportsRail />
      </div>

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
