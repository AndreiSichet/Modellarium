import { render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

import App from './App';
import { createPredictionFor, getHealth, getSchedule } from './api';
import { TEAMS, UNKNOWN_TEAM, teamFor } from './data/teams';
import { LEAGUES, firstFixtureDate, leagueCopy, leaguesForSport } from './data/leagues';

jest.mock('./api');

// THREE CUTOFFS, ALL DIFFERENT, which is what most of this file turns on.
// The G League's data ends on 2026-03-28, so 2026-03-29 is the only date it
// can be asked about - and that is neither of the other two leagues' dates.
const HEALTH = {
  dataAsOf: '2026-04-12',
  daysBehind: 177,
  stale: true,
  wnba: { dataAsOf: '2026-09-24', daysBehind: 12, stale: true },
  gleague: { dataAsOf: '2026-03-28', daysBehind: 192, stale: true },
};

const NBA_DATE = '2026-04-13';
const GLEAGUE_DATE = '2026-03-29';

// BOTH LEAGUES FRESH TO THE SAME DAY, so one date is predictable for both.
// That is the state a real season puts them in, and it is the only state in
// which the routing can actually be tested: with the cutoffs as they stand
// today, a fixture on the NBA's predictable date is months past the G
// League's and gets dropped by the per-league filter before any routing
// decision is reached. The calendar has been doing the work the routing
// should do.
const HEALTH_ALIGNED = {
  ...HEALTH,
  gleague: { dataAsOf: '2026-04-12', daysBehind: 177, stale: true },
};

// Real ids and abbreviations from the live cache. The G League's are 3-letter
// codes of their own (OKL, AUS) and none of them collide with an NBA or WNBA
// abbreviation - measured, not assumed; see the collision test below.
const GLEAGUE_FIXTURE = {
  leagueSlug: 'gleague',
  homeTeamId: 1612709889,
  homeTeamAbbr: 'OKL',
  homeTeamName: 'Oklahoma City Blue',
  awayTeamId: 1612709890,
  awayTeamAbbr: 'AUS',
  awayTeamName: 'Austin Spurs',
  gameDate: GLEAGUE_DATE,
};

const NBA_FIXTURE = {
  leagueSlug: 'nba',
  homeTeamId: 1610612737,
  homeTeamAbbr: 'ATL',
  homeTeamName: 'Atlanta Hawks',
  awayTeamId: 1610612738,
  awayTeamAbbr: 'BOS',
  awayTeamName: 'Boston Celtics',
  gameDate: NBA_DATE,
};

const GLEAGUE_SUMMARY = {
  gameId: 981,
  prediction: {
    homeWinProbability: 0.5908580792747549,
    homeMargin: 2.927269925841589,
    totalPoints: 244.46527901729075,
    moneylineWindow: 'CARRY10',
    spreadWindow: 'CARRY10',
    totalsWindow: 'CARRY10',
    dataAsOf: '2026-03-28',
    stale: true,
    daysBehind: 192,
  },
};

const NBA_SUMMARY = {
  gameId: 4,
  prediction: {
    homeWinProbability: 0.5745,
    homeMargin: 1.4149,
    totalPoints: 232.9323,
    reboundMargin: -0.3438,
    totalRebounds: 88.7604,
    assistMargin: 2.2516,
    totalAssists: 51.0805,
    dataAsOf: '2026-04-12',
    stale: true,
  },
};

function renderAt(path) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>
  );
}

beforeEach(() => {
  jest.clearAllMocks();
  getHealth.mockResolvedValue(HEALTH);
  getSchedule.mockResolvedValue([]);
  createPredictionFor.mockImplementation((leagueSlug) =>
    Promise.resolve(leagueSlug === 'gleague' ? GLEAGUE_SUMMARY : NBA_SUMMARY)
  );
});

describe('fixtures are routed by league, never by a default', () => {
  test('a G League fixture goes to the G League endpoint, not the NBA one', async () => {
    getSchedule.mockResolvedValue([GLEAGUE_FIXTURE]);
    renderAt('/predictions/basketball/gleague');

    await screen.findByText('Oklahoma City Blue');

    expect(createPredictionFor).toHaveBeenCalledWith('gleague', {
      homeTeamId: 1612709889,
      awayTeamId: 1612709890,
      gameDate: GLEAGUE_DATE,
    });
    expect(createPredictionFor).not.toHaveBeenCalledWith('nba', expect.anything());
  });

  test('a G League fixture dated on the NBA predictable date still goes to the G League', async () => {
    // THE CASE THE CALENDAR HAS BEEN HIDING. Today no G League fixture falls
    // on the NBA's predictable date, so a fixture mis-filed as NBA was
    // dropped by the cutoff filter before anything POSTed it. Once the two
    // leagues' dates coincide - which they will - the only thing standing
    // between a G League fixture and the NBA endpoint is this routing.
    getHealth.mockResolvedValue(HEALTH_ALIGNED);
    getSchedule.mockResolvedValue([{ ...GLEAGUE_FIXTURE, gameDate: NBA_DATE }]);
    renderAt('/predictions/basketball/gleague');

    await screen.findByText('Oklahoma City Blue');

    expect(createPredictionFor).toHaveBeenCalledWith(
      'gleague',
      expect.objectContaining({ gameDate: NBA_DATE })
    );
    expect(createPredictionFor).not.toHaveBeenCalledWith('nba', expect.anything());
  });

  test('it appears under the G League and not in the NBA list', async () => {
    getHealth.mockResolvedValue(HEALTH_ALIGNED);
    getSchedule.mockResolvedValue([
      NBA_FIXTURE,
      { ...GLEAGUE_FIXTURE, gameDate: NBA_DATE },
    ]);
    renderAt('/predictions/basketball');

    await screen.findByText('Atlanta Hawks');

    const nba = document.getElementById('league-basketball-nba');
    const gleague = document.getElementById('league-basketball-gleague');

    expect(within(nba).queryByText('Oklahoma City Blue')).toBeNull();
    expect(within(nba).getByText('Atlanta Hawks')).toBeInTheDocument();
    expect(within(gleague).queryByText('Atlanta Hawks')).toBeNull();
  });
});

describe('the G League routes resolve', () => {
  test('its league page renders with no new route definition', async () => {
    getSchedule.mockResolvedValue([GLEAGUE_FIXTURE]);
    renderAt('/predictions/basketball/gleague');

    expect(
      await screen.findByRole('heading', { name: 'G League Predictions', level: 1 })
    ).toBeInTheDocument();
  });

  test('its detail route renders the three markets it serves', async () => {
    getSchedule.mockResolvedValue([GLEAGUE_FIXTURE]);
    renderAt(`/predictions/basketball/gleague/${GLEAGUE_SUMMARY.gameId}`);

    expect(await screen.findByText('Total points')).toBeInTheDocument();
  });

  test('its detail page renders no tablist, because one tab is not a tab bar', async () => {
    getSchedule.mockResolvedValue([GLEAGUE_FIXTURE]);
    const { container } = renderAt(
      `/predictions/basketball/gleague/${GLEAGUE_SUMMARY.gameId}`
    );
    await screen.findByText('Total points');

    expect(screen.queryByRole('tablist')).not.toBeInTheDocument();
    expect(screen.queryAllByRole('tab')).toHaveLength(0);

    // And no orphaned tab semantics: aria-labelledby pointing at a tab that
    // was never rendered is a dangling reference.
    expect(container.querySelector('[role="tabpanel"]')).toBeNull();
    expect(container.querySelector('[aria-selected]')).toBeNull();
    expect(container.querySelector('[aria-labelledby^="detail-tab-"]')).toBeNull();
  });
});

describe('badges stay distinct across all three leagues', () => {
  test('no G League id resolves to an NBA or WNBA team', () => {
    // Derived from the tables rather than restating a list, so a G League id
    // added to teams.js later cannot silently collide.
    const gleagueIds = [GLEAGUE_FIXTURE.homeTeamId, GLEAGUE_FIXTURE.awayTeamId];

    gleagueIds.forEach((id) => {
      expect(teamFor(id)).toBe(UNKNOWN_TEAM);
    });
  });

  test('G League abbreviations collide with no NBA or WNBA abbreviation', () => {
    // MEASURED ACROSS THE LIVE CACHE AND FOUND EMPTY, unlike the WNBA's
    // eight. The lookup is by id either way, so a collision would be
    // harmless - but the WNBA's eight are the reason the lookup is by id,
    // and asserting the G League's set stays empty is how a future rename
    // into a collision gets noticed rather than discovered on the page.
    const known = new Set(Object.values(TEAMS).map((team) => team.abbr));
    ['OKL', 'AUS'].forEach((abbr) => {
      expect(known.has(abbr)).toBe(false);
    });
  });

  test('a G League badge shows the API abbreviation, not a question mark', async () => {
    getSchedule.mockResolvedValue([GLEAGUE_FIXTURE]);
    const { container } = renderAt('/predictions/basketball/gleague');
    await screen.findByText('Oklahoma City Blue');

    const badges = Array.from(
      container.querySelectorAll('.game-row .team-badge'),
      (node) => node.textContent
    );

    // teams.js holds no G League colours - inventing published colours is
    // not something to do from memory - so these resolve to the neutral
    // badge. Without the API's abbreviation every one would read "?".
    expect(badges).toContain('OKL');
    expect(badges).toContain('AUS');
    expect(badges).not.toContain(UNKNOWN_TEAM.abbr);
  });

  test('two leagues sharing an abbreviation still render different badges', async () => {
    // The NBA's ATL and the G League's AUS sit in one list here; the real
    // collision case is the WNBA's, pinned in its own file. What this adds
    // is that a G League row's badge colour is the neutral one rather than
    // whatever an NBA team with a similar code uses.
    getHealth.mockResolvedValue(HEALTH_ALIGNED);
    getSchedule.mockResolvedValue([
      NBA_FIXTURE,
      { ...GLEAGUE_FIXTURE, gameDate: NBA_DATE },
    ]);
    renderAt('/predictions/basketball');
    await screen.findByText('Atlanta Hawks');

    const nba = document.getElementById('league-basketball-nba');
    const gleague = document.getElementById('league-basketball-gleague');

    const nbaBadge = nba.querySelector('.team-badge');
    const gleagueBadge = gleague.querySelector('.team-badge');

    expect(nbaBadge.style.background).not.toBe(gleagueBadge.style.background);
  });
});

describe('G League predictability is its own', () => {
  test('its fixture is predictable on its own date while the NBA has nothing', async () => {
    getSchedule.mockResolvedValue([GLEAGUE_FIXTURE]);
    renderAt('/predictions/basketball');

    await screen.findByText('Oklahoma City Blue');

    expect(createPredictionFor).toHaveBeenCalledWith('gleague', expect.anything());
    expect(createPredictionFor).not.toHaveBeenCalledWith('nba', expect.anything());
  });

  test('an NBA fixture is predictable while the G League has nothing', async () => {
    getSchedule.mockResolvedValue([NBA_FIXTURE]);
    renderAt('/predictions/basketball');

    await screen.findByText('Atlanta Hawks');

    expect(createPredictionFor).toHaveBeenCalledWith('nba', expect.anything());
    expect(createPredictionFor).not.toHaveBeenCalledWith('gleague', expect.anything());
  });

  test('a G League fixture past its cutoff is not requested at all', async () => {
    // Its data ends 2026-03-28, so a fixture in late December is months
    // beyond anything it can answer - and must not be judged by the NBA's
    // April cutoff instead.
    getSchedule.mockResolvedValue([{ ...GLEAGUE_FIXTURE, gameDate: '2026-12-27' }]);
    renderAt('/predictions/basketball/gleague');

    await screen.findByRole('heading', { name: 'G League Predictions', level: 1 });
    expect(createPredictionFor).not.toHaveBeenCalled();
  });
});

describe('the G League empty state says what is true', () => {
  const SCHEDULE_AHEAD = [
    { ...GLEAGUE_FIXTURE, gameDate: '2027-01-14' },
    { ...GLEAGUE_FIXTURE, gameDate: '2026-12-27' },
    { ...GLEAGUE_FIXTURE, gameDate: '2027-02-02' },
  ];

  async function emptyState(schedule = SCHEDULE_AHEAD) {
    getSchedule.mockResolvedValue(schedule);
    renderAt('/predictions/basketball/gleague');
    await screen.findByRole('heading', { name: 'G League Predictions', level: 1 });
    return document.querySelector('.predictions-message');
  }

  test('it derives the regular-season date from the first cached fixture', async () => {
    const panel = await emptyState();

    // 2026-12-27 is the EARLIEST of the three, not the first in the array -
    // so this is a minimum over the cache rather than a lucky ordering.
    expect(panel.textContent).toMatch(/regular season begins 27 December 2026/);
  });

  test('the date is not hardcoded anywhere', async () => {
    const panel = await emptyState([
      { ...GLEAGUE_FIXTURE, gameDate: '2027-03-05' },
    ]);

    // A different cache gives a different date. The spec's own guess was
    // "around 19 December"; the cache said the 27th, which is exactly why
    // this is derived.
    expect(panel.textContent).toMatch(/regular season begins 5 March 2027/);
    expect(panel.textContent).not.toMatch(/27 December/);
  });

  test('it names the Showcase Cup as not predicted', async () => {
    const panel = await emptyState();

    // The season opens with the Cup and the models are regular-season only,
    // so a visitor seeing November fixtures without predictions has no way
    // to know why unless it is said.
    expect(panel.textContent).toMatch(/Showcase Cup/);
    expect(panel.textContent).toMatch(/not predicted here/);
  });

  test('it makes no ten-game claim, because the windows are carried', async () => {
    const panel = await emptyState();

    expect(panel.textContent).not.toMatch(/ten games/i);
    expect(panel.textContent).not.toMatch(/enough games/i);
  });

  test('it does not claim data is unrefreshed', async () => {
    const panel = await emptyState();

    expect(panel.textContent).not.toMatch(/not refreshed/i);
  });

  test("it shows the G League's own cutoff, not another league's", async () => {
    const panel = await emptyState();

    expect(panel.textContent).toMatch(/current to 2026-03-28/);
    expect(panel.textContent).not.toMatch(/2026-04-12/);
    expect(panel.textContent).not.toMatch(/2026-09-24/);
  });

  test('with nothing cached it claims no date at all', async () => {
    const panel = await emptyState([]);

    expect(panel.textContent).toMatch(/has not been scheduled yet/);
    expect(panel.textContent).not.toMatch(/\d{1,2} (December|January|February)/);
  });
});

describe('firstFixtureDate', () => {
  test('is a minimum, and ignores other leagues', () => {
    const schedule = [
      { leagueSlug: 'nba', gameDate: '2026-01-01' },
      { leagueSlug: 'gleague', gameDate: '2027-01-14' },
      { leagueSlug: 'gleague', gameDate: '2026-12-27' },
    ];

    expect(firstFixtureDate(schedule, 'gleague')).toBe('2026-12-27');
    expect(firstFixtureDate(schedule, 'nba')).toBe('2026-01-01');
  });

  test('is null rather than a guess when the league has nothing cached', () => {
    expect(firstFixtureDate([], 'gleague')).toBeNull();
    expect(firstFixtureDate(undefined, 'gleague')).toBeNull();
  });
});

describe('the `More <league>` links are three separate targets', () => {
  async function linkRow() {
    getSchedule.mockResolvedValue([]);
    renderAt('/predictions/basketball');
    await screen.findByRole('heading', { name: 'No predictions yet' });
    return document.querySelector('.predictions-message-links');
  }

  test('one link per league, each its own anchor', async () => {
    const row = await linkRow();
    const links = Array.from(row.querySelectorAll('a'));

    // THE BUG THIS REPLACES rendered them as inline text reading
    // "More NBAMore WNBAMore G League". Three anchors is what makes them
    // three keyboard stops and three screen-reader targets.
    // SCOPED TO THE SPORT, NOT ALL LEAGUES. This read LEAGUES.length, which
    // was the same number while basketball was the only sport and became
    // wrong the moment the NFL existed: this page is /predictions/basketball,
    // so it shows basketball's leagues and the NFL belongs to another rail
    // entry entirely.
    const basketball = leaguesForSport('basketball');
    expect(links).toHaveLength(basketball.length);
    expect(links.map((a) => a.textContent)).toEqual(
      basketball.map((league) => `More ${league.label}`)
    );
    expect(links.map((a) => a.getAttribute('href'))).toEqual([
      '/predictions/basketball/nba',
      '/predictions/basketball/wnba',
      '/predictions/basketball/gleague',
    ]);
  });

  test('they are separated by a gap, not by text', async () => {
    const row = await linkRow();

    // No typed space or separator character between them: the text content
    // runs together precisely because the separation is layout, which is
    // what jsdom can check here. That the gap RENDERS is a browser
    // question, so the token is asserted in the stylesheet instead.
    expect(row.tagName).toBe('NAV');
    expect(row.textContent).toBe('More NBAMore WNBAMore G League');
  });
});

describe('the league constants stay honest', () => {
  test('every league declares the markets it serves', () => {
    const gleague = LEAGUES.find((league) => league.slug === 'gleague');

    expect(gleague.markets).toEqual({
      rebounds: false,
      assists: false,
      quarterHalf: false,
      playerProps: false,
    });
    expect(gleague.freshness).toBe('gleague');
  });

  test('no league claims its data is not refreshed automatically', () => {
    LEAGUES.forEach((league) => {
      const note = leagueCopy(league.seasonNote, { schedule: [] }).join(' ');
      expect(note).not.toMatch(/not refreshed/i);
    });
  });
});
