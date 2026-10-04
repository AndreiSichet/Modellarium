import { createPredictionFor, normalisePredictionBody } from './api';

// THE ONLY PLACE THE RAW WIRE SHAPES ARE PINNED.
//
// Every other test mocks the api module, which is right for them - they are
// about components - but it leaves the backend's response shapes unasserted
// anywhere. Here fetch is mocked instead, so the real request and the real
// normalisation both run, against bodies written in each Java DTO's own
// shape.
//
// The two DTOs disagree: GameSummaryDto says {id, latestPrediction} and
// WnbaSummaryDto says {gameId, prediction}. That is deliberate and recorded -
// renaming a field the frontend already consumes would be a breaking change
// to win tidiness - so the disagreement has to be absorbed somewhere, and
// this asserts it is absorbed correctly rather than guessed.

const NBA_BODY = {
  id: 4,
  homeTeamAbbreviation: 'ATL',
  awayTeamAbbreviation: 'BOS',
  gameDate: '2026-04-13',
  played: false,
  latestPrediction: {
    homeWinProbability: 0.42541608214378357,
    homeMargin: 1.4149,
    totalPoints: 232.9323,
    dataAsOf: '2026-04-12',
    stale: true,
  },
};

const WNBA_BODY = {
  gameId: 715,
  homeTeamAbbreviation: 'PHX',
  awayTeamAbbreviation: 'LVA',
  gameDate: '2026-09-25',
  prediction: {
    homeWinProbability: 0.18193019489666626,
    homeMargin: -8.403725674288566,
    totalPoints: 178.40689601339275,
    moneylineWindow: 'CARRY5',
    spreadWindow: 'CARRY5',
    totalsWindow: 'CARRY10',
    dataAsOf: '2026-09-24',
    stale: true,
  },
};

function mockFetchOnce(body) {
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    json: () => Promise.resolve(body),
  });
}

const PAYLOAD = { homeTeamId: 1, awayTeamId: 2, gameDate: '2026-04-13' };

afterEach(() => {
  delete global.fetch;
});

describe('normalisePredictionBody', () => {
  test('reads the NBA DTO field names', () => {
    expect(normalisePredictionBody('nba', NBA_BODY)).toEqual({
      gameId: 4,
      prediction: NBA_BODY.latestPrediction,
    });
  });

  test('reads the WNBA DTO field names, which are different', () => {
    expect(normalisePredictionBody('wnba', WNBA_BODY)).toEqual({
      gameId: 715,
      prediction: WNBA_BODY.prediction,
    });
  });

  test('the two mappings are not interchangeable', () => {
    // Applying one league's field names to the other's body yields undefined
    // rather than a wrong-but-plausible value - which is what makes a
    // mismatch loud instead of silent.
    expect(normalisePredictionBody('nba', WNBA_BODY)).toEqual({
      gameId: undefined,
      prediction: undefined,
    });
    expect(normalisePredictionBody('wnba', NBA_BODY)).toEqual({
      gameId: undefined,
      prediction: undefined,
    });
  });

  test('an unknown league throws rather than defaulting to the NBA', () => {
    expect(() => normalisePredictionBody('euroleague', NBA_BODY)).toThrow(
      /No prediction endpoint for league "euroleague"/
    );
  });
});

describe('createPredictionFor', () => {
  test('posts to the NBA path and returns the normalised shape', async () => {
    mockFetchOnce(NBA_BODY);

    const result = await createPredictionFor('nba', PAYLOAD);

    expect(global.fetch).toHaveBeenCalledTimes(1);
    expect(global.fetch.mock.calls[0][0]).toMatch(/\/predictions$/);
    expect(result.gameId).toBe(4);
    expect(result.prediction.homeWinProbability).toBe(0.42541608214378357);
  });

  test('posts to the WNBA path, which is a different endpoint', async () => {
    mockFetchOnce(WNBA_BODY);

    const result = await createPredictionFor('wnba', PAYLOAD);

    expect(global.fetch.mock.calls[0][0]).toMatch(/\/predictions\/wnba$/);
    expect(result.gameId).toBe(715);

    // The reference values phase 4 pinned, so a frontend change cannot
    // quietly start reading a different field and still look plausible.
    expect(result.prediction.homeWinProbability).toBe(0.18193019489666626);
    expect(result.prediction.homeMargin).toBe(-8.403725674288566);
    expect(result.prediction.totalPoints).toBe(178.40689601339275);

    // No caveat on the wire. The windows are what remains of the provenance.
    expect(result.prediction).not.toHaveProperty('moneylineCaveat');
    expect(result.prediction.totalsWindow).toBe('CARRY10');
  });

  test('an unknown league makes no request at all', async () => {
    mockFetchOnce(NBA_BODY);

    await expect(createPredictionFor('ncaa', PAYLOAD)).rejects.toThrow(
      /No prediction endpoint/
    );
    expect(global.fetch).not.toHaveBeenCalled();
  });
});
