/**
 * One refused fixture costs one row, not the page.
 *
 * THE DEFECT. `PredictionsLayout` predicted every predictable fixture with a
 * bare `Promise.all` inside one try/catch, so a single rejected prediction
 * rejected the lot and the whole league page became "could not load". The
 * G League refuses a team's FIRST game of a season - `rest_for` is
 * season-scoped, so a seven-month off-season is not a week off - and its
 * season opens in early November.
 *
 * MEASURED ON A COPY OF THE SERVED HISTORY, 2026-10-30: with zero 2026-27 G
 * League games, every 2026-27 fixture refuses; with ONE game recorded for a
 * team, that team's next fixture SCORES. So the exposure is opening night per
 * team - and on opening night the page would have gone dark entirely.
 *
 * A 4xx IS KEPT, ANYTHING ELSE RE-THROWS. A declined fixture is information.
 * A 5xx or a network failure is a fault and must still take the page down, or
 * a broken backend would look like a thousand unpredictable fixtures.
 */
import { render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

import App from './App';
import { createPredictionFor, getHealth, getSchedule } from './api';
import { REFUSAL_TEXT } from './refusalText';

jest.mock('./api');

const GLEAGUE_DATE = '2026-03-29';

const HEALTH = {
  dataAsOf: '2026-04-12',
  daysBehind: 177,
  stale: true,
  wnba: { dataAsOf: '2026-09-24', daysBehind: 12, stale: true },
  gleague: { dataAsOf: '2026-03-28', daysBehind: 192, stale: true },
};

function fixture(homeId, homeAbbr, homeName, awayId, awayAbbr, awayName) {
  return {
    leagueSlug: 'gleague',
    homeTeamId: homeId,
    homeTeamAbbr: homeAbbr,
    homeTeamName: homeName,
    awayTeamId: awayId,
    awayTeamAbbr: awayAbbr,
    awayTeamName: awayName,
    gameDate: GLEAGUE_DATE,
  };
}

const FIXTURES = [
  fixture(1612709889, 'OKL', 'Oklahoma City Blue',
          1612709890, 'AUS', 'Austin Spurs'),
  fixture(1612709905, 'SBL', 'Salt Lake City Stars',
          1612709919, 'WES', 'Westchester Knicks'),
  fixture(1612709923, 'WCB', 'Wisconsin Herd',
          1612709913, 'RIP', 'Rip City Remix'),
];

const SUMMARY = {
  gameId: 1,
  prediction: {
    homeWinProbability: 0.55,
    homeMargin: 2.5,
    totalPoints: 220.5,
    predictedAt: '2026-10-30T09:00:00Z',
  },
};

/** The error api.js builds from a 400 whose body carries a code. */
function refusal(code, status = 400) {
  const error = new Error('Inference service rejected the request: {…}');
  error.status = status;
  error.code = code;
  error.fields = {};
  error.body = { detail: 'incomplete features for this fixture', code };
  return error;
}

function renderGleague() {
  return render(
    <MemoryRouter initialEntries={['/predictions/basketball/gleague']}>
      <App />
    </MemoryRouter>
  );
}

beforeEach(() => {
  jest.clearAllMocks();
  getHealth.mockResolvedValue(HEALTH);
  getSchedule.mockResolvedValue(FIXTURES);
});

test('one refusing fixture among three: the other two still render', async () => {
  // The SECOND fixture refuses. Deliberately not the first: a catch that only
  // worked on the first element would pass a one-fixture test and fail on a
  // real slate.
  createPredictionFor.mockImplementation((_league, body) =>
    body.homeTeamId === 1612709905
      ? Promise.reject(refusal('season_opener'))
      : Promise.resolve(SUMMARY)
  );

  renderGleague();

  // THE PAGE RENDERS AT ALL, which is the whole point.
  await screen.findByText('Oklahoma City Blue');
  expect(screen.getByText('Wisconsin Herd')).toBeInTheDocument();

  // AND THE REFUSED ONE IS PRESENT, NAMED, AND EXPLAINED - not dropped.
  expect(screen.getByText('Salt Lake City Stars')).toBeInTheDocument();
  expect(screen.getByText(/Not yet predictable/i)).toBeInTheDocument();
  expect(screen.getByText(REFUSAL_TEXT.season_opener)).toBeInTheDocument();

  // and NOT the page-level failure
  expect(screen.queryByText(/could not load/i)).not.toBeInTheDocument();
});

test('every fixture refusing still renders a page, not an error', async () => {
  // Opening night: no team has a game recorded yet, so the whole slate is
  // declined. This is the state that would have produced a blank page.
  createPredictionFor.mockRejectedValue(refusal('season_opener'));

  renderGleague();

  await screen.findByText('Oklahoma City Blue');
  expect(screen.getAllByText(/Not yet predictable/i)).toHaveLength(3);
  expect(screen.queryByText(/could not load/i)).not.toBeInTheDocument();
});

test('a 500 still takes the page down - a fault is not a refusal', async () => {
  // THE CONTROL. Without it, the catch could swallow a genuinely broken
  // backend and present it as a slate of unpredictable fixtures, which is
  // the misleading green this project keeps removing.
  createPredictionFor.mockRejectedValue(refusal(null, 500));

  renderGleague();

  await screen.findByText(/could not load/i);
  expect(screen.queryByText(/Not yet predictable/i)).not.toBeInTheDocument();
});

test('a 4xx with no code still gets a row, with the API prose', async () => {
  const error = refusal(null, 400);
  createPredictionFor.mockImplementation((_league, body) =>
    body.homeTeamId === 1612709905
      ? Promise.reject(error)
      : Promise.resolve(SUMMARY)
  );

  renderGleague();

  await screen.findByText('Oklahoma City Blue');
  expect(screen.getByText(/Not yet predictable/i)).toBeInTheDocument();
  // An unknown code falls back to the API's own sentence rather than a blank.
  expect(
    screen.getByText('incomplete features for this fixture')
  ).toBeInTheDocument();
});

test('PLANT: without the per-fixture catch, one refusal fails the page', async () => {
  // The pre-2026-10-30 shape, reproduced exactly: a bare Promise.all means
  // one rejection rejects the whole array, which is what set status 'error'
  // for the entire league page.
  const settle = (body) =>
    body.homeTeamId === 1612709905
      ? Promise.reject(refusal('season_opener'))
      : Promise.resolve(SUMMARY);

  await expect(
    Promise.all(FIXTURES.map((f) => settle(f)))
  ).rejects.toThrow(/rejected the request/);

  // and with the catch in place the same three settle to three results
  const caught = await Promise.all(
    FIXTURES.map((f) =>
      settle(f).catch((failure) =>
        failure.status >= 400 && failure.status < 500
          ? { refused: true }
          : Promise.reject(failure)
      )
    )
  );
  expect(caught).toHaveLength(3);
  expect(caught.filter((row) => row.refused)).toHaveLength(1);
});
