import { useCallback, useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';

import { getHealth, getSchedule } from '../api';
import { latestPredictableDate } from '../dates';
import SportsRail, { findSport } from './SportsRail';
import './PredictionsPage.css';

/**
 * How far ahead to ask for fixtures. Matches BrowseView's window so the two
 * views cannot disagree about what "upcoming" means while both exist.
 */
const SCHEDULE_DAYS_AHEAD = 14;

/**
 * The three-column Predictions shell.
 *
 * TWO API CALLS, NOT ONE. getSchedule returns the fixture list and
 * getHealth returns the freshness metadata (dataAsOf, stale, daysBehind) —
 * there is no single call that returns both. They go through Promise.all
 * because a fixture list without the cutoff cannot say which of its entries
 * are reachable, so half of this pair is useless on its own.
 *
 * WHAT "EMPTY" MEANS HERE is worth being precise about: not "no fixtures
 * exist" but "no PREDICTABLE fixtures exist". MAX_DAYS_AHEAD is 1 and
 * structural (see dates.js), so the schedule routinely returns fixtures
 * that cannot be scored. Counting raw fixtures would show a populated state
 * listing games nothing can predict.
 */
function PredictionsPage() {
  const { sport: slug } = useParams();
  const sport = findSport(slug);

  const [status, setStatus] = useState('loading');
  const [games, setGames] = useState([]);
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setStatus('loading');
    setError(null);
    try {
      const [schedule, freshness] = await Promise.all([
        getSchedule(SCHEDULE_DAYS_AHEAD),
        getHealth(),
      ]);
      setGames(schedule);
      setHealth(freshness);
      setStatus('ready');
    } catch (failure) {
      // ERROR AND EMPTY ARE DIFFERENT FACTS and must not collapse into one
      // another. "The backend is down" and "there are no games" look the
      // same on screen if the failure falls through to the empty state, and
      // the empty state is reassuring — exactly the misleading-green this
      // project keeps rejecting.
      setError(failure.message);
      setStatus('error');
    }
  }, []);

  useEffect(() => {
    // Only fetch for a sport that exists. An unknown slug has nothing to
    // load and should not spend a request finding that out.
    if (sport) load();
  }, [sport, load]);

  const cutoff = latestPredictableDate(health?.dataAsOf);
  const predictable = cutoff
    ? games.filter((game) => game.gameDate <= cutoff)
    : [];

  return (
    <div className="predictions">
      <div className="predictions-rail">
        <SportsRail />
      </div>

      <div className="predictions-content">
        {!sport ? (
          <UnknownSport slug={slug} />
        ) : status === 'loading' ? (
          // A line, not a spinner. The call returns fast enough that a
          // spinner appears and vanishes as a flash of noise.
          <p className="predictions-muted">Loading…</p>
        ) : status === 'error' ? (
          <FetchFailed message={error} onRetry={load} />
        ) : predictable.length > 0 ? (
          <Placeholder label={`Sport content — Phase 3 · ${predictable.length} games`} />
        ) : (
          <NoPredictionsYet dataAsOf={health?.dataAsOf} />
        )}
      </div>

      <div className="predictions-overview">
        <Placeholder label="Overview — Phase 4" />
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
 * switch on. A hardcoded one would be a guess that goes stale silently,
 * which is a failure mode already corrected elsewhere in this project.
 *
 * dataAsOf is THE REAL VALUE from the API, not a constant. It is the same
 * freshness fact the old stale badge carried, and losing it in a redesign
 * would quietly drop information the user had before.
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
