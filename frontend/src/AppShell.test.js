import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

import App from './App';
import { createPrediction, getHealth, getSchedule } from './api';

jest.mock('./api');

/**
 * Phase 1 (shell, routing, About) and Phase 2 (three columns, the rail, the
 * three data states).
 *
 * MemoryRouter with initialEntries rather than BrowserRouter, which is why
 * App exports only the route table and index.js mounts the router. Driving
 * jsdom's location instead would make "render /predictions" a global side
 * effect rather than an argument.
 */
const HEALTH = { dataAsOf: '2026-04-12', daysBehind: 131, stale: true };

/** Within the cutoff (dataAsOf + MAX_DAYS_AHEAD), so predictable. */
const PREDICTABLE_GAME = {
  homeTeamId: 1610612737,
  homeTeamAbbr: 'ATL',
  homeTeamName: 'Atlanta Hawks',
  awayTeamId: 1610612738,
  awayTeamAbbr: 'BOS',
  awayTeamName: 'Boston Celtics',
  gameDate: '2026-04-13',
};

/** Shaped as GameSummaryDto, which is what POST /api/predictions returns. */
const SUMMARY = {
  id: 4,
  homeTeamAbbreviation: 'ATL',
  awayTeamAbbreviation: 'BOS',
  gameDate: '2026-04-13',
  played: false,
  latestPrediction: {
    homeWinProbability: 0.5745,
    homeMargin: 1.4149,
    totalPoints: 232.9323,
    reboundMargin: -0.3438,
    totalRebounds: 88.7604,
    assistMargin: 2.2516,
    totalAssists: 51.0805,
    dataAsOf: '2026-04-12',
    stale: true,
    predictedAt: '2026-09-19T10:00:00Z',
  },
};

/** A real fixture, but months past the cutoff — listed, not predictable. */
const FAR_GAME = { ...PREDICTABLE_GAME, gameDate: '2026-10-20' };

function renderAt(path) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>
  );
}

beforeEach(() => {
  jest.clearAllMocks();
  getSchedule.mockResolvedValue([]);
  getHealth.mockResolvedValue(HEALTH);
  createPrediction.mockResolvedValue(SUMMARY);
});

describe('routing', () => {
  test('/ lands on About, not on predictions', () => {
    renderAt('/');

    expect(
      screen.getByRole('heading', {
        name: /Computing sports predictions for the love of the game/,
      })
    ).toBeInTheDocument();
  });

  test('/predictions redirects to the one sport that exists', async () => {
    renderAt('/predictions');

    // The redirect is invisible: the rail shows Basketball active, which is
    // what a landing page would have asked the user to click.
    await waitFor(() =>
      expect(screen.getByRole('link', { name: /Basketball/ })).toHaveClass(
        'active'
      )
    );
  });

  test('an unknown sport renders the layout with a message, and does not redirect', async () => {
    renderAt('/predictions/tennis');

    expect(
      screen.getByRole('heading', { name: 'No such sport', level: 1 })
    ).toBeInTheDocument();
    // Still the real page, not a blanked one.
    expect(screen.getByRole('navigation', { name: 'Sports' })).toBeInTheDocument();
    // And it must not have spent a request finding out.
    expect(getSchedule).not.toHaveBeenCalled();
  });

  test('an unknown path redirects home', () => {
    renderAt('/does-not-exist');

    expect(
      screen.getByRole('heading', {
        name: /Computing sports predictions for the love of the game/,
      })
    ).toBeInTheDocument();
  });
});

describe('app shell', () => {
  test('the logo is hidden from assistive tech so the wordmark is not read twice', () => {
    const { container } = renderAt('/');

    const logo = container.querySelector('.wordmark-logo');
    expect(logo).toHaveAttribute('alt', '');
    expect(logo).toHaveAttribute('aria-hidden', 'true');

    const header = screen.getByRole('banner');
    expect(within(header).getAllByText(/MODELLARIUM/i)).toHaveLength(1);
  });

  /**
   * THE CHECK THAT CATCHES A NavLink WIRED TO THE WRONG PATH. Rendering at
   * a route directly is the test equivalent of a hard reload: there is no
   * click to set the active class, so it can only come from the router
   * matching the path.
   */
  test('the active link is derived from the URL, not from a click', async () => {
    const { unmount } = renderAt('/');
    expect(screen.getByRole('link', { name: 'About' })).toHaveClass('active');
    unmount();

    renderAt('/predictions/basketball');
    await waitFor(() =>
      expect(screen.getByRole('link', { name: 'Predictions' })).toHaveClass(
        'active'
      )
    );
    expect(screen.getByRole('link', { name: 'About' })).not.toHaveClass('active');
  });

  /**
   * `end` on the About NavLink is load-bearing: without it "/" is a prefix
   * of every path and About would stay underlined everywhere.
   */
  test('About does not stay active on a child route', async () => {
    renderAt('/predictions/basketball');

    await waitFor(() =>
      expect(screen.getByRole('link', { name: 'About' })).not.toHaveClass('active')
    );
  });
});

describe('sports rail', () => {
  test('lists only sports that exist — no placeholder entries', () => {
    renderAt('/predictions/basketball');

    const rail = screen.getByRole('navigation', { name: 'Sports' });
    expect(within(rail).getAllByRole('link')).toHaveLength(1);
    expect(within(rail).getByRole('link', { name: /Basketball/ })).toBeInTheDocument();
  });
});

describe('predictions data states', () => {
  test('empty: succeeded with nothing predictable, and the real dataAsOf is shown', async () => {
    getSchedule.mockResolvedValue([]);
    renderAt('/predictions/basketball');

    expect(
      await screen.findByRole('heading', { name: 'No predictions yet' })
    ).toBeInTheDocument();
    // The real value from the API, not a hardcoded string.
    expect(screen.getByText(/Model data is current to 2026-04-12/)).toBeInTheDocument();
  });

  /**
   * "Empty" means no PREDICTABLE games, not no games. The schedule returns
   * fixtures months out that MAX_DAYS_AHEAD will refuse, and counting raw
   * fixtures would show a populated state listing games nothing can score.
   */
  test('empty: fixtures exist but none are within the cutoff', async () => {
    getSchedule.mockResolvedValue([FAR_GAME]);
    renderAt('/predictions/basketball');

    expect(
      await screen.findByRole('heading', { name: 'No predictions yet' })
    ).toBeInTheDocument();
  });

  test('populated: only the fixture within the cutoff is predicted and listed', async () => {
    getSchedule.mockResolvedValue([PREDICTABLE_GAME, FAR_GAME]);
    const { container } = renderAt('/predictions/basketball');

    expect(await screen.findByText('Atlanta Hawks')).toBeInTheDocument();
    // One of the two is predictable; the far one must not reach the list,
    // and must not cost a POST either.
    //
    // Scoped to .game-row: an unscoped getAllByRole('listitem') also
    // matches the sports rail's own <li>, which made this read 2.
    expect(container.querySelectorAll('.game-row')).toHaveLength(1);
    expect(createPrediction).toHaveBeenCalledTimes(1);
  });

  /**
   * ONE POST PER GAME, AND EACH WRITES A ROW. The endpoint is append-only
   * by design, so this is N rows per page view rather than one per click.
   * The rate is asserted here so a future change to it is visible rather
   * than silent — the fix belongs in a backend batch endpoint, not in a
   * frontend cache.
   */
  test('populated: one prediction request per predictable game, no more', async () => {
    getSchedule.mockResolvedValue([
      PREDICTABLE_GAME,
      { ...PREDICTABLE_GAME, homeTeamId: 1610612744, homeTeamName: 'Golden State Warriors' },
    ]);
    renderAt('/predictions/basketball');

    expect(await screen.findByText('Golden State Warriors')).toBeInTheDocument();
    expect(createPrediction).toHaveBeenCalledTimes(2);
  });

  /**
   * THE DISTINCTION MOST LIKELY TO BE WRONG. A failed request falling
   * through to the empty state would report "there are no games" when the
   * truth is "the service did not answer" — reassuring, and false.
   */
  test('error: a failed request is not shown as an empty schedule', async () => {
    getSchedule.mockRejectedValue(new Error('Failed to fetch'));
    renderAt('/predictions/basketball');

    expect(
      await screen.findByRole('heading', { name: 'Could not load predictions' })
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Try again' })).toBeInTheDocument();
    expect(screen.queryByText('No predictions yet')).not.toBeInTheDocument();
    // The backend's own message survives to the screen.
    expect(screen.getByText(/Failed to fetch/)).toBeInTheDocument();
  });

  test('error: retry re-issues the request', async () => {
    getSchedule.mockRejectedValueOnce(new Error('Failed to fetch'));
    renderAt('/predictions/basketball');

    const retry = await screen.findByRole('button', { name: 'Try again' });
    getSchedule.mockResolvedValue([]);
    // fireEvent, not the raw DOM click: the latter dispatches outside
    // React's act() and the resulting state update warns.
    fireEvent.click(retry);

    expect(
      await screen.findByRole('heading', { name: 'No predictions yet' })
    ).toBeInTheDocument();
    expect(getSchedule).toHaveBeenCalledTimes(2);
  });
});

describe('league tabs', () => {
  test('only leagues that exist appear — no disabled placeholders', async () => {
    getSchedule.mockResolvedValue([PREDICTABLE_GAME]);
    renderAt('/predictions/basketball');

    const tabs = await screen.findAllByRole('tab');
    expect(tabs.map((tab) => tab.textContent)).toEqual(['Upcoming', 'NBA']);
  });

  test('Upcoming groups under a league heading; NBA is flat', async () => {
    getSchedule.mockResolvedValue([PREDICTABLE_GAME]);
    renderAt('/predictions/basketball');

    // Upcoming is the default tab and groups, even with one league — the
    // group is rendered rather than special-cased away.
    expect(await screen.findByRole('heading', { name: 'NBA', level: 2 })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('tab', { name: 'NBA' }));
    expect(screen.queryByRole('heading', { name: 'NBA', level: 2 })).not.toBeInTheDocument();
    // The game itself survives the switch.
    expect(screen.getByText('Atlanta Hawks')).toBeInTheDocument();
  });

  /**
   * Tabs are page state, not routes. Were they routes, the browser Back
   * button would step through tab presses instead of leaving the page.
   */
  test('switching tabs does not refetch', async () => {
    getSchedule.mockResolvedValue([PREDICTABLE_GAME]);
    renderAt('/predictions/basketball');

    await screen.findByText('Atlanta Hawks');
    const before = createPrediction.mock.calls.length;

    fireEvent.click(screen.getByRole('tab', { name: 'NBA' }));
    fireEvent.click(screen.getByRole('tab', { name: 'Upcoming' }));

    expect(createPrediction).toHaveBeenCalledTimes(before);
    expect(getSchedule).toHaveBeenCalledTimes(1);
  });

  test('the active tab is marked for assistive tech, not just visually', async () => {
    getSchedule.mockResolvedValue([PREDICTABLE_GAME]);
    renderAt('/predictions/basketball');

    const upcoming = await screen.findByRole('tab', { name: 'Upcoming' });
    expect(upcoming).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('tab', { name: 'NBA' })).toHaveAttribute(
      'aria-selected',
      'false'
    );
  });
});

describe('about page', () => {
  test('renders all four prose blocks', () => {
    renderAt('/');

    for (const heading of [
      'How predictions are made',
      'Choosing a model',
      "What it doesn't claim",
      'Staying current',
    ]) {
      expect(screen.getByRole('heading', { name: heading })).toBeInTheDocument();
    }
  });

  test('the retraining copy describes a regression guard, not an improvement bar', () => {
    renderAt('/');

    // Pinned deliberately. "not worse" is what the promotion gate actually
    // enforces, and an edit to something stronger would make the page claim
    // behaviour the code does not have.
    expect(
      screen.getByText(/only replaces the current one if it is not worse/)
    ).toBeInTheDocument();
  });
});
