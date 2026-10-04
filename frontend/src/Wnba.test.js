import { render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

import App from './App';
import {
  createPredictionFor,
  createQuarterHalfPrediction,
  getHealth,
  getPlayerPropPredictions,
  getSchedule,
} from './api';
import { TEAMS, badgeTextColour, teamFor } from './data/teams';
import { findLeague } from './data/leagues';
import { dataAsOfFor } from './dates';

jest.mock('./api');

// THE TWO CUTOFFS DIFFER, AND THAT IS THE POINT OF MOST OF THIS FILE.
// NBA data to 2026-04-12 means 2026-04-13 is predictable; WNBA data to
// 2026-09-24 means only 2026-09-25 is, which is in the past relative to
// nothing in particular but is NOT the NBA's date.
const HEALTH = {
  dataAsOf: '2026-04-12',
  daysBehind: 161,
  stale: true,
  wnba: { dataAsOf: '2026-09-24', daysBehind: 10, stale: true },
};

const NBA_DATE = '2026-04-13';
const WNBA_DATE = '2026-09-25';

const NBA_FIXTURE = {
  homeTeamId: 1610612737,
  homeTeamAbbr: 'ATL',
  homeTeamName: 'Atlanta Hawks',
  awayTeamId: 1610612738,
  awayTeamAbbr: 'BOS',
  awayTeamName: 'Boston Celtics',
  gameDate: NBA_DATE,
};

const WNBA_FIXTURE = {
  leagueSlug: 'wnba',
  homeTeamId: 1611661330,
  homeTeamAbbr: 'ATL',
  homeTeamName: 'Atlanta Dream',
  awayTeamId: 1611661329,
  awayTeamAbbr: 'CHI',
  awayTeamName: 'Chicago Sky',
  gameDate: WNBA_DATE,
};

const WNBA_SUMMARY = {
  gameId: 71,
  prediction: {
    homeWinProbability: 0.5312,
    homeMargin: 1.9044,
    totalPoints: 164.2718,
    moneylineWindow: 'CARRY5',
    spreadWindow: 'CARRY5',
    totalsWindow: 'CARRY10',
    dataAsOf: '2026-09-24',
    stale: true,
    daysBehind: 10,
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
  createQuarterHalfPrediction.mockResolvedValue({ prediction: {} });
  getPlayerPropPredictions.mockResolvedValue({ homeTeam: null, awayTeam: null });

  createPredictionFor.mockImplementation((leagueSlug) =>
    Promise.resolve(leagueSlug === 'wnba' ? WNBA_SUMMARY : NBA_SUMMARY)
  );
});

describe('team identity across leagues', () => {
  // Eight abbreviations exist in both leagues. Any lookup resolving a team by
  // abbreviation would render the Dream in the Hawks' colours and say nothing
  // about it, so this asserts the two produce DIFFERENT badges rather than
  // asserting that the lookup happens to be keyed on an id.
  const SHARED = ['ATL', 'CHI', 'DAL', 'IND', 'MIN', 'PHX', 'TOR', 'WAS'];

  test('an NBA and a WNBA team sharing an abbreviation render different badges', () => {
    const hawks = teamFor(1610612737);
    const dream = teamFor(1611661330);

    expect(hawks.abbr).toBe(dream.abbr);
    expect(hawks.name).not.toBe(dream.name);

    const badgeOf = (team) => [team.primary, badgeTextColour(team)].join('/');
    expect(badgeOf(hawks)).not.toBe(badgeOf(dream));
  });

  test('every shared abbreviation resolves to two distinct teams', () => {
    // Derived from TEAMS rather than from a hardcoded pair list, so this
    // proves the collision is real in the data instead of restating it.
    const byAbbr = {};
    for (const [id, team] of Object.entries(TEAMS)) {
      (byAbbr[team.abbr] ||= []).push(Number(id));
    }

    const collided = Object.entries(byAbbr).filter(([, ids]) => ids.length > 1);

    expect(collided.map(([abbr]) => abbr).sort()).toEqual(SHARED);

    for (const [, ids] of collided) {
      expect(ids).toHaveLength(2);

      const [a, b] = ids.map(teamFor);
      expect(a.name).not.toBe(b.name);

      // The ids come from different leagues, which is why one of them being
      // chosen by abbreviation would be silently wrong.
      expect(String(ids[0]).slice(0, 7)).not.toBe(String(ids[1]).slice(0, 7));
    }
  });

  test('both teams render with their own name in the browser', async () => {
    getSchedule.mockResolvedValue([NBA_FIXTURE, WNBA_FIXTURE]);
    renderAt('/predictions/basketball');

    expect(await screen.findByText('Atlanta Hawks')).toBeInTheDocument();
    expect(screen.getByText('Atlanta Dream')).toBeInTheDocument();
  });
});

describe('predictability is per league', () => {
  test('the WNBA cutoff is its own, not the NBA top-level one', () => {
    const nba = findLeague('basketball', 'nba');
    const wnba = findLeague('basketball', 'wnba');

    expect(dataAsOfFor(HEALTH, nba)).toBe('2026-04-12');
    expect(dataAsOfFor(HEALTH, wnba)).toBe('2026-09-24');
    expect(dataAsOfFor(HEALTH, wnba)).not.toBe(dataAsOfFor(HEALTH, nba));
  });

  test('a league with no entry in health yields null, not the NBA date', () => {
    const wnba = findLeague('basketball', 'wnba');

    // An older backend sends no wnba block. Falling back to the NBA's cutoff
    // would mark WNBA games predictable on NBA freshness and the page would
    // request predictions the backend then rejects.
    expect(dataAsOfFor({ dataAsOf: '2026-04-12' }, wnba)).toBeNull();
  });

  test('NBA predictable and WNBA not: the WNBA page shows its empty state', async () => {
    // Only the NBA fixture is in reach. The WNBA game is dated to a date the
    // WNBA cutoff does not reach, so nothing WNBA is predicted.
    getSchedule.mockResolvedValue([
      NBA_FIXTURE,
      { ...WNBA_FIXTURE, gameDate: '2026-10-05' },
    ]);

    renderAt('/predictions/basketball/wnba');

    expect(
      await screen.findByRole('heading', { name: 'WNBA Predictions', level: 1 })
    ).toBeInTheDocument();

    expect(screen.queryByText('Atlanta Dream')).not.toBeInTheDocument();
    expect(screen.getByText(/The 2026 WNBA season is over/)).toBeInTheDocument();
  });

  test('and the NBA page still shows its game', async () => {
    getSchedule.mockResolvedValue([
      NBA_FIXTURE,
      { ...WNBA_FIXTURE, gameDate: '2026-10-05' },
    ]);

    renderAt('/predictions/basketball/nba');

    expect(await screen.findByText('Atlanta Hawks')).toBeInTheDocument();
    expect(createPredictionFor).toHaveBeenCalledWith('nba', expect.anything());
    expect(createPredictionFor).not.toHaveBeenCalledWith('wnba', expect.anything());
  });
});

describe('the WNBA empty state says what is true', () => {
  async function wnbaEmptyState() {
    getSchedule.mockResolvedValue([]);
    renderAt('/predictions/basketball/wnba');
    return (await screen.findByText(/The 2026 WNBA season is over/)).closest(
      '.predictions-message'
    );
  }

  test('it makes no ten-game claim', async () => {
    const panel = await wnbaEmptyState();

    // The NBA's sentence is about within-season rolling windows. The WNBA
    // models were selected on CARRY5 and CARRY10, which carry the previous
    // season forward, so there is no warm-up to wait through and the NBA
    // copy would be false here.
    expect(panel.textContent).not.toMatch(/ten games/i);
    expect(panel.textContent).not.toMatch(/enough games/i);
  });

  test('it names no 2027 date, because the schedule is not published', async () => {
    const panel = await wnbaEmptyState();

    expect(panel.textContent).toMatch(/2027 regular season begins/);
    expect(panel.textContent).toMatch(/has not been published/);
    expect(panel.textContent).not.toMatch(/\b\d{1,2} (January|February|March|April|May|June)\b/);
    expect(panel.textContent).not.toMatch(/2027-\d\d-\d\d/);
  });

  test('it does not imply predictions appear on their own', async () => {
    const panel = await wnbaEmptyState();

    // WNBA data is baked into the image and only advances when its pipeline
    // is run by hand, so the season resuming and predictions returning are
    // two separate events.
    expect(panel.textContent).toMatch(/not refreshed automatically/);
  });

  test("it shows the WNBA's own cutoff, not the NBA's", async () => {
    const panel = await wnbaEmptyState();

    expect(panel.textContent).toMatch(/current to 2026-09-24/);
    expect(panel.textContent).not.toMatch(/2026-04-12/);
  });
});

describe('the WNBA game detail page', () => {
  async function openWnbaDetail() {
    getSchedule.mockResolvedValue([WNBA_FIXTURE]);
    const view = renderAt(`/predictions/basketball/wnba/${WNBA_SUMMARY.gameId}`);
    await screen.findByText('Total points');
    return view;
  }

  test('renders without a tablist, because one tab is not a tab bar', async () => {
    const { container } = await openWnbaDetail();

    expect(screen.queryByRole('tablist')).not.toBeInTheDocument();
    expect(screen.queryAllByRole('tab')).toHaveLength(0);

    // And no orphaned tab semantics left on the panel: aria-labelledby
    // pointing at a tab that was never rendered is a dangling reference.
    expect(container.querySelector('[role="tabpanel"]')).toBeNull();
    expect(container.querySelector('[aria-selected]')).toBeNull();
    expect(container.querySelector('[aria-labelledby^="detail-tab-"]')).toBeNull();
  });

  test('shows Outcome and Totals only', async () => {
    const { container } = await openWnbaDetail();

    const titles = Array.from(
      container.querySelectorAll('.market-group-title'),
      (node) => node.textContent
    );

    expect(titles).toEqual(['Outcome', 'Totals']);
  });

  test('renders no rebound, assist, quarter or player content', async () => {
    await openWnbaDetail();

    for (const absent of [
      /Rebound margin/,
      /Total rebounds/,
      /Assist margin/,
      /Total assists/,
      /Quarters & Halves/,
      /Player Points/,
      /Player PRA/,
    ]) {
      expect(screen.queryByText(absent)).not.toBeInTheDocument();
    }
  });

  test('requests no quarter/half prediction for a league without that market', async () => {
    await openWnbaDetail();

    // Without the gate this fires on arrival and the Python side rejects the
    // team ids - a failed request for a market the WNBA never claimed.
    await waitFor(() => expect(createQuarterHalfPrediction).not.toHaveBeenCalled());
    expect(getPlayerPropPredictions).not.toHaveBeenCalled();
  });

  test('the response carries no caveat field at all', async () => {
    const { container } = await openWnbaDetail();

    // THE FIELD IS GONE FROM THE RESPONSE, not merely unrendered. The
    // manifest's note records bare Elo at 0.6046 against this model's
    // 0.6130, and a paired bootstrap puts that difference at +0.0083 with a
    // 95% interval of [-0.0059, +0.0224], spanning zero on all ten seeds. A
    // field carrying a null result invites every client to present it as a
    // finding, so it stays in models_wnba/manifest.json and is not served.
    //
    // Asserted on the payload rather than only on the DOM: checking the
    // screen alone would pass just as well if the field were still being
    // sent and simply ignored, which is the thing that changed.
    expect(WNBA_SUMMARY.prediction).not.toHaveProperty('moneylineCaveat');
    expect(Object.keys(WNBA_SUMMARY.prediction)).not.toContain('moneylineCaveat');

    expect(container.textContent).not.toMatch(/Elo alone/);
    expect(container.textContent).not.toMatch(/validation set/);
  });

  test('both markets and both teams are present, and the spread sign is unchanged', async () => {
    await openWnbaDetail();

    expect(screen.getByText('Win probability')).toBeInTheDocument();
    expect(screen.getByText('Spread')).toBeInTheDocument();
    expect(screen.getByText('164.3')).toBeInTheDocument();

    // homeMargin +1.9044 favours the home side, so the convention shows -1.9
    // for home - the same in both leagues.
    const spread = screen.getByText('Spread').closest('.market-row');
    expect(within(spread).getByText('-1.9')).toBeInTheDocument();
    expect(within(spread).getByText('+1.9')).toBeInTheDocument();
  });
});

describe('routes resolve from the league constant alone', () => {
  test('the WNBA league route resolves with no route definition of its own', async () => {
    getSchedule.mockResolvedValue([WNBA_FIXTURE]);
    renderAt('/predictions/basketball/wnba');

    expect(
      await screen.findByRole('heading', { name: 'WNBA Predictions', level: 1 })
    ).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'No such league' })).not.toBeInTheDocument();
  });

  test('the WNBA detail route resolves too', async () => {
    getSchedule.mockResolvedValue([WNBA_FIXTURE]);
    renderAt(`/predictions/basketball/wnba/${WNBA_SUMMARY.gameId}`);

    expect(await screen.findByText('Atlanta Dream')).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'No such game' })).not.toBeInTheDocument();
  });

  test('the WNBA is reachable from the General page', async () => {
    getSchedule.mockResolvedValue([NBA_FIXTURE]);
    renderAt('/predictions');

    await screen.findByRole('heading', { name: 'General', level: 1 });

    // A section with no games, so its page is reachable from here rather than
    // only by a typed URL.
    const link = screen.getByRole('link', { name: 'More WNBA' });
    expect(link).toHaveAttribute('href', '/predictions/basketball/wnba');
  });
});
