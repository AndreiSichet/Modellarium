const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:8080/api';

async function readError(response) {
  let message = null;

  try {
    const body = await response.json();
    message = body.message || body.error || null;
  } catch {
  }

  return new Error(message || `Request failed with status ${response.status}`);
}

async function request(path, options) {
  const response = await fetch(`${API_BASE}${path}`, options);

  if (!response.ok) {
    throw await readError(response);
  }

  return response.json();
}

// One entry per league, because the two endpoints disagree about BOTH their
// path and their response field names: GameSummaryDto says
// {id, latestPrediction} and WnbaSummaryDto says {gameId, prediction}. That
// disagreement is deliberate and documented - renaming a field the frontend
// already consumes would be a breaking change to win tidiness - so it is
// absorbed here at the boundary instead of leaking into components.
const PREDICTION_ENDPOINTS = {
  nba: { path: '/predictions', gameIdField: 'id', predictionField: 'latestPrediction' },
  wnba: { path: '/predictions/wnba', gameIdField: 'gameId', predictionField: 'prediction' },
  gleague: {
    path: '/predictions/gleague',
    gameIdField: 'gameId',
    predictionField: 'prediction',
  },
  nfl: {
    path: '/predictions/nfl',
    gameIdField: 'gameId',
    predictionField: 'prediction',
  },
};

/**
 * Cached fixtures, each carrying the league it belongs to.
 *
 * THE WIRE FIELD IS `league` AND COMPONENTS READ `leagueSlug`, which is the
 * whole reason this mapping exists. Before it, nothing translated the two:
 * `game.leagueSlug` was always undefined, so the `|| 'nba'` that used to sit
 * in the layout was not a fallback for a missing league - it was the ONLY
 * path that ever ran, and all 962 cached fixtures were treated as NBA ones.
 * That was invisible because every component test mocks this function and
 * supplies `leagueSlug` directly, so the wire name was pinned nowhere.
 *
 * Nothing broke only because the calendar hid it: G League fixtures start in
 * late December and the NBA's predictable date is in April, so the per-game
 * cutoff filter dropped them before anything tried to POST one. Once the two
 * leagues' dates coincide, a G League fixture would be sent to the NBA
 * endpoint and rejected as an unknown team.
 */
export async function getSchedule(daysAhead = 14) {
  return normaliseSchedule(
    await request(`/games/schedule?daysAhead=${daysAhead}`)
  );
}

/**
 * Wire fixtures in the shape components read, with unroutable ones dropped.
 *
 * A fixture whose league is missing or unknown is SKIPPED AND LOGGED, never
 * filed under a default. There is no safe default: this project has twice
 * had to remove a silent fallback that made it serve one of two things
 * without saying which, and a fixture we cannot name a league for is one we
 * cannot ask for a prediction for either.
 *
 * Validated against the endpoint table rather than against the league
 * constants, because the operative question is not "is this a league we have
 * heard of" but "can we actually request a prediction for it".
 */
export function normaliseSchedule(fixtures) {
  const routable = [];
  const skipped = [];

  (fixtures || []).forEach((game) => {
    const leagueSlug = game?.league;

    if (!leagueSlug || !PREDICTION_ENDPOINTS[leagueSlug]) {
      skipped.push(game);
      return;
    }

    routable.push({ ...game, leagueSlug });
  });

  if (skipped.length) {
    const leagues = Array.from(
      new Set(skipped.map((game) => game?.league ?? 'missing'))
    );
    console.warn(
      `Skipped ${skipped.length} scheduled fixture(s) with no routable `
        + `league: ${leagues.join(', ')}. A fixture is never filed under a `
        + 'default league, because the prediction would go to the wrong '
        + 'endpoint.'
    );
  }

  return routable;
}

export function getHealth() {
  return request('/health');
}

export function createPrediction(payload) {
  return request('/predictions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

/**
 * A prediction for one fixture, in the league's own endpoint and shape.
 *
 * Throws on a league with no endpoint rather than defaulting to the NBA's.
 * Defaulting would send a WNBA fixture to the NBA models, which would answer
 * "unknown team id" - but a league added to the constants and forgotten here
 * would otherwise fail in a way that looks like a data problem.
 */
export async function createPredictionFor(leagueSlug, payload) {
  const endpoint = PREDICTION_ENDPOINTS[leagueSlug];

  if (!endpoint) {
    throw new Error(`No prediction endpoint for league "${leagueSlug}".`);
  }

  const body = await request(endpoint.path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  return normalisePredictionBody(leagueSlug, body);
}

/**
 * One league's response body, in the shape components read.
 *
 * Exported so the dev-fixture path goes through the SAME mapping as a real
 * response. The fixtures are written in each backend DTO's own shape, so if
 * this mapping is wrong the fixtures break too - rather than dev quietly
 * agreeing with itself while production disagrees.
 */
export function normalisePredictionBody(leagueSlug, body) {
  const endpoint = PREDICTION_ENDPOINTS[leagueSlug];

  if (!endpoint) {
    throw new Error(`No prediction endpoint for league "${leagueSlug}".`);
  }

  return {
    gameId: body?.[endpoint.gameIdField],
    prediction: body?.[endpoint.predictionField],
  };
}

export function createQuarterHalfPrediction(payload) {
  return request('/predictions/quarter-half', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export function getPlayerPropPredictions(payload) {
  return request('/predictions/player-props', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}
