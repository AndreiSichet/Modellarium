import {
  createPredictionFor,
  getSchedule,
  normalisePredictionBody,
  normaliseSchedule,
} from './api';

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
    homeWinProbability: 0.4191701412200928,
    homeMargin: 1.3146,
    totalPoints: 230.9997,
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

const GLEAGUE_BODY = {
  gameId: 981,
  homeTeamAbbreviation: 'OKL',
  awayTeamAbbreviation: 'AUS',
  gameDate: '2026-03-29',
  prediction: {
    homeWinProbability: 0.5908580792747549,
    homeMargin: 2.927269925841589,
    totalPoints: 244.46527901729075,
    moneylineWindow: 'CARRY10',
    spreadWindow: 'CARRY10',
    totalsWindow: 'CARRY10',
    dataAsOf: '2026-03-28',
    stale: true,
  },
};

// ScheduledGameDto's OWN field name. It says `league`; components read
// `leagueSlug`. Nothing translated the two until this mapping existed, and
// nothing pinned the wire name either - every component test mocks
// getSchedule and supplies the internal name - so `game.leagueSlug` was
// always undefined and the layout's `|| 'nba'` was the only path that ever
// ran. This file is where that can no longer happen quietly.
const SCHEDULE_BODY = [
  {
    league: 'nba',
    homeTeamId: 1610612737,
    homeTeamAbbr: 'ATL',
    homeTeamName: 'Atlanta Hawks',
    awayTeamId: 1610612738,
    awayTeamAbbr: 'BOS',
    awayTeamName: 'Boston Celtics',
    gameDate: '2026-04-13',
  },
  {
    league: 'gleague',
    homeTeamId: 1612709889,
    homeTeamAbbr: 'OKL',
    homeTeamName: 'Oklahoma City Blue',
    awayTeamId: 1612709890,
    awayTeamAbbr: 'AUS',
    awayTeamName: 'Austin Spurs',
    gameDate: '2026-12-27',
  },
];

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
    expect(result.prediction.homeWinProbability).toBe(0.4191701412200928);
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

describe('the schedule wire shape', () => {
  test("maps ScheduledGameDto's `league` onto the `leagueSlug` components read", async () => {
    mockFetchOnce(SCHEDULE_BODY);
    const fixtures = await getSchedule(14);

    expect(fixtures).toHaveLength(2);
    expect(fixtures.map((game) => game.leagueSlug)).toEqual(['nba', 'gleague']);

    // The wire field survives alongside it rather than being renamed away,
    // so a reader comparing against the DTO can still see where it came
    // from.
    expect(fixtures[0].league).toBe('nba');
  });

  test('reading the wrong field name yields nothing, which is the bug this had', () => {
    // Before the mapping, this was the state every fixture was in - and
    // `|| 'nba'` turned it into "every fixture is an NBA fixture".
    expect(SCHEDULE_BODY.every((game) => game.leagueSlug === undefined)).toBe(true);
  });

  test('a fixture with no league is skipped and logged, never defaulted', () => {
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});

    const fixtures = normaliseSchedule([
      ...SCHEDULE_BODY,
      { homeTeamId: 1, awayTeamId: 2, gameDate: '2026-04-13' },
    ]);

    expect(fixtures).toHaveLength(2);
    expect(fixtures.some((game) => game.leagueSlug === undefined)).toBe(false);
    expect(warn).toHaveBeenCalledWith(expect.stringMatching(/Skipped 1 scheduled fixture/));
    expect(warn).toHaveBeenCalledWith(expect.stringMatching(/missing/));

    warn.mockRestore();
  });

  test('a league with no prediction endpoint is skipped, not passed through', () => {
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});

    const fixtures = normaliseSchedule([
      { league: 'euroleague', homeTeamId: 1, awayTeamId: 2, gameDate: '2026-04-13' },
    ]);

    // Validated against the endpoint table, because the question is not
    // whether we have heard of the league but whether we can ask for a
    // prediction for it.
    expect(fixtures).toEqual([]);
    expect(warn).toHaveBeenCalledWith(expect.stringMatching(/euroleague/));

    warn.mockRestore();
  });

  test('an empty or absent schedule is not an error', () => {
    expect(normaliseSchedule([])).toEqual([]);
    expect(normaliseSchedule(undefined)).toEqual([]);
  });
});

describe('the G League prediction endpoint', () => {
  test('posts to its own path and reads GleagueSummaryDto field names', async () => {
    mockFetchOnce(GLEAGUE_BODY);
    const summary = await createPredictionFor('gleague', PAYLOAD);

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/predictions/gleague'),
      expect.objectContaining({ method: 'POST' })
    );
    expect(summary.gameId).toBe(981);
    expect(summary.prediction.homeWinProbability).toBe(0.5908580792747549);
  });

  test("the NBA mapping would read nothing from a G League body", () => {
    // Same guard the WNBA has: the two shapes are not interchangeable, and
    // a wrong mapping must yield undefined rather than a plausible value.
    expect(normalisePredictionBody('nba', GLEAGUE_BODY).gameId).toBeUndefined();
    expect(normalisePredictionBody('nba', GLEAGUE_BODY).prediction).toBeUndefined();
  });
});
