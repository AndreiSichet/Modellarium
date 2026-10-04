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

export function getSchedule(daysAhead = 14) {
  return request(`/games/schedule?daysAhead=${daysAhead}`);
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

// One entry per league, because the two endpoints disagree about BOTH their
// path and their response field names: GameSummaryDto says
// {id, latestPrediction} and WnbaSummaryDto says {gameId, prediction}. That
// disagreement is deliberate and documented - renaming a field the frontend
// already consumes would be a breaking change to win tidiness - so it is
// absorbed here at the boundary instead of leaking into components.
const PREDICTION_ENDPOINTS = {
  nba: { path: '/predictions', gameIdField: 'id', predictionField: 'latestPrediction' },
  wnba: { path: '/predictions/wnba', gameIdField: 'gameId', predictionField: 'prediction' },
};

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
