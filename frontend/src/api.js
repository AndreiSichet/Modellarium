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
