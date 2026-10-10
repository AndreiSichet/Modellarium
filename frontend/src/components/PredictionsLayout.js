import { useCallback, useEffect, useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';

import {
  createPredictionFor,
  getHealth,
  getSchedule,
  normalisePredictionBody,
  normaliseSchedule,
} from '../api';
import { DEV_FIXTURES_ON, DEV_HEALTH, DEV_SCHEDULE, devPredictionFor } from '../data/devFixtures';
import { leagueBySlug } from '../data/leagues';
import { isPredictable, predictableDateFor } from '../dates';
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
  // THE UNFILTERED LIST, kept beside the predicted one because a league can
  // need a date from a fixture it cannot yet predict: the G League's empty
  // state derives its regular-season start from the first cached fixture,
  // and every one of those is months past its cutoff.
  const [schedule, setSchedule] = useState([]);
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setStatus('loading');
    setError(null);
    try {
      // The dev fixtures go through normaliseSchedule too, because they are
      // written in the wire shape. Same reason the prediction fixtures go
      // through normalisePredictionBody: a mapping only dev skips is a
      // mapping dev cannot catch a mistake in.
      const [schedule, freshness] = DEV_FIXTURES_ON
        ? [normaliseSchedule(DEV_SCHEDULE), DEV_HEALTH]
        : await Promise.all([getSchedule(SCHEDULE_DAYS_AHEAD), getHealth()]);

      setHealth(freshness);

      // Filtered PER GAME against its own league's cutoff, not once against a
      // single one. The three leagues' data ends on different dates, so one
      // cutoff is wrong for whichever league it did not come from.
      //
      // NO `|| 'nba'` HERE ANY MORE. It read as a fallback and was in fact
      // the only path that ever ran, because the wire field is `league` and
      // this read `leagueSlug` - so every fixture, G League included, was
      // judged by the NBA's cutoff and would have been POSTed to the NBA
      // endpoint. `getSchedule` now maps the field and drops anything it
      // cannot route, so a league here is always real.
      //
      // AND THE RULE IS NOW PER LEAGUE, not a date for everyone. Basketball
      // keeps the date rule; the NFL's answer arrives on the fixture itself,
      // because its rule is a dependency the inference service measures and
      // a copy of it here would drift. dates.isPredictable dispatches.
      const predictable = schedule.filter((game) =>
        isPredictable(game, freshness, leagueBySlug(game.leagueSlug))
      );

      const predicted = await Promise.all(
        predictable.map(async (game, index) => {
          const { leagueSlug } = game;

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
            // CARRIED THROUGH FOR THE BADGE. The G League's franchises are
            // not in teams.js, so without the API's abbreviation their
            // badges would all read "?" - see TeamBadge.
            homeTeamAbbr: game.homeTeamAbbr,
            awayTeamId: game.awayTeamId,
            awayTeamName: game.awayTeamName,
            awayTeamAbbr: game.awayTeamAbbr,
            gameDate: game.gameDate,
            // CARRIED THROUGH OR THE KICKOFF NEVER SHOWS. GameRow reads
            // these from the game it is handed, not from the schedule, so
            // omitting them here made formatKickoff fall back to the bare
            // date for every NFL row - the time would simply have been
            // absent, with nothing failing anywhere.
            kickoffUtc: game.kickoffUtc,
            flex: game.flex,
            week: game.week,
            predictable: game.predictable,
            prediction: summary.prediction,
          };
        })
      );

      setGames(predicted);
      setSchedule(schedule);
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
          schedule,
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
