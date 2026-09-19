import { useCallback, useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';

import { createPrediction, getHealth, getSchedule } from '../api';
import { DEV_FIXTURES_ON, DEV_SCHEDULE, devPredictionFor } from '../data/devFixtures';
import { latestPredictableDate } from '../dates';
import GameList from './GameList';
import LeagueTabs, { LEAGUES } from './LeagueTabs';
import SportsRail, { findSport } from './SportsRail';
import './PredictionsPage.css';

/**
 * How far ahead to ask for fixtures. Matches BrowseView's window so the two
 * views cannot disagree about what "upcoming" means while both exist.
 */
const SCHEDULE_DAYS_AHEAD = 14;

/**
 * The three-column Predictions shell, with the middle column populated.
 *
 * TWO API CALLS FOR THE LIST, NOT ONE. getSchedule returns fixtures and
 * getHealth returns freshness (dataAsOf/stale/daysBehind) — no single call
 * returns both, and a fixture list without the cutoff cannot say which
 * entries are reachable.
 *
 * WHAT "EMPTY" MEANS is unchanged from Phase 2: not "no fixtures" but "no
 * PREDICTABLE fixtures". MAX_DAYS_AHEAD is 1 and structural, so the
 * schedule routinely returns games nothing can score.
 */
function PredictionsPage() {
  const { sport: slug } = useParams();
  const sport = findSport(slug);

  const [status, setStatus] = useState('loading');
  const [games, setGames] = useState([]);
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(null);
  const [league, setLeague] = useState(LEAGUES[0].id);

  const load = useCallback(async () => {
    setStatus('loading');
    setError(null);
    try {
      const [schedule, freshness] = DEV_FIXTURES_ON
        ? [DEV_SCHEDULE, { dataAsOf: '2026-04-12', stale: true, daysBehind: 160 }]
        : await Promise.all([getSchedule(SCHEDULE_DAYS_AHEAD), getHealth()]);

      setHealth(freshness);

      const cutoff = latestPredictableDate(freshness?.dataAsOf);
      const predictable = cutoff
        ? schedule.filter((game) => game.gameDate <= cutoff)
        : [];

      /*
       * ONE POST PER GAME, AND IT WRITES A ROW EACH TIME.
       *
       * getSchedule returns fixtures, not predictions — predictions come
       * from POST /api/predictions, one game at a time. Ten games on screen
       * is ten requests, which is fine at this scale and needs no backend
       * change.
       *
       * THE COST IS DATABASE GROWTH, and it is worth naming here rather
       * than discovering it from a row count. That endpoint is append-only
       * BY DESIGN — predictions are an immutable history, not a mutated
       * record — so this writes N rows per page view where the old browse
       * view wrote one per click. Over a season that accumulates.
       *
       * DO NOT FIX THIS IN THE FRONTEND. Caching or de-duplicating here
       * would hide the rate without changing it, and would put a policy
       * decision in the wrong layer. The real fix is backend work: a batch
       * endpoint taking many fixtures, or a get-if-recent path that returns
       * an existing prediction instead of always writing. Either is its own
       * piece of work.
       */
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
            league: 'NBA',
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
      // is reassuring — exactly the misleading-green this project rejects.
      setError(failure.message);
      setStatus('error');
    }
  }, []);

  useEffect(() => {
    if (sport) load();
  }, [sport, load]);

  const grouped = LEAGUES.find((entry) => entry.id === league)?.grouped;

  return (
    <div className="predictions">
      <div className="predictions-rail">
        <SportsRail />
      </div>

      <div className="predictions-content">
        {!sport ? (
          <UnknownSport slug={slug} />
        ) : (
          <>
            <LeagueTabs active={league} onSelect={setLeague} />
            {status === 'loading' ? (
              // A line, not a spinner. The call returns fast enough that a
              // spinner appears and vanishes as a flash of noise.
              <p className="predictions-muted">Loading…</p>
            ) : status === 'error' ? (
              <FetchFailed message={error} onRetry={load} />
            ) : games.length > 0 ? (
              <GameList games={games} grouped={grouped} />
            ) : (
              <NoPredictionsYet dataAsOf={health?.dataAsOf} />
            )}
          </>
        )}
      </div>

      <div className="predictions-overview">
        {/* The inner div is what sticks. See PredictionsPage.css — a
            stretched grid item cannot stick to itself. */}
        <div className="predictions-overview-inner">
          <Placeholder label="Overview — Phase 4" />
        </div>
      </div>
    </div>
  );
}

/**
 * Deliberately unstyled beyond a hairline border, so that nobody mistakes
 * placeholder styling for finished styling in a screenshot.
 */
function Placeholder({ label }) {
  return <div className="predictions-placeholder">{label}</div>;
}

function UnknownSport({ slug }) {
  return (
    <div className="predictions-message">
      <h1 className="predictions-message-title">No such sport</h1>
      <p className="predictions-message-body">
        Nothing here covers <strong>{slug}</strong>. Pick a sport from the
        rail to see its predictions.
      </p>
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

/**
 * The state this page will be in for the next several weeks, so it reads as
 * an explanation rather than a blank column.
 *
 * TWO THINGS HERE ARE DELIBERATE AND SHOULD NOT BE "TIDIED":
 *
 * The second date is DESCRIBED, NOT PINNED. Each team needs roughly ten
 * games of its own before its rolling form exists, and teams play on
 * different schedules — so there is no single date on which predictions
 * switch on. A hardcoded one would be a guess that goes stale silently.
 *
 * dataAsOf is THE REAL VALUE from the API, not a constant. It is the same
 * freshness fact the old stale badge carried.
 */
function NoPredictionsYet({ dataAsOf }) {
  return (
    <div className="predictions-message">
      <h1 className="predictions-message-title">No predictions yet</h1>
      <p className="predictions-message-body">
        The 2026-27 NBA season begins on 20 October 2026.
      </p>
      <p className="predictions-message-body">
        Predictions need recent form to work from, so they become available
        once each team has played enough games for that form to be computed —
        around ten games in, which is roughly three weeks after opening night.
      </p>
      {dataAsOf ? (
        <p className="predictions-message-meta">
          Model data is current to {dataAsOf}.
        </p>
      ) : null}
    </div>
  );
}

export default PredictionsPage;
