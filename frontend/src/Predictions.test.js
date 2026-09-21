import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

import App from './App';
import { createPrediction, getHealth, getSchedule } from './api';
import { byDate, soonestGames } from './components/GamesPage';
import { sportSlugFromPath } from './components/PredictionsLayout';

jest.mock('./api');

const HEALTH = { dataAsOf: '2026-04-12', daysBehind: 161, stale: true };

const TODAY = '2026-04-13';
const YESTERDAY = '2026-04-12';

const SUMMARY = {
  id: 4,
  homeTeamAbbreviation: 'ATL',
  awayTeamAbbreviation: 'BOS',
  gameDate: TODAY,
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
    predictedAt: '2026-09-20T10:00:00Z',
  },
};

const HOME_IDS = [
  1610612738, 1610612744, 1610612752, 1610612743, 1610612749, 1610612757,
  1610612747, 1610612745, 1610612761,
];

function fixture(index, gameDate, extra = {}) {
  return {
    homeTeamId: HOME_IDS[index],
    homeTeamAbbr: 'H',
    homeTeamName: `Home ${index}`,
    awayTeamId: 1610612751,
    awayTeamAbbr: 'BKN',
    awayTeamName: 'Brooklyn Nets',
    gameDate,
    ...extra,
  };
}

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

describe('sportSlugFromPath', () => {
  test('reads the sport segment, and null when there is none', () => {
    expect(sportSlugFromPath('/predictions')).toBeNull();
    expect(sportSlugFromPath('/predictions/')).toBeNull();
    expect(sportSlugFromPath('/predictions/basketball')).toBe('basketball');
    expect(sportSlugFromPath('/predictions/basketball/nba')).toBe('basketball');
    expect(sportSlugFromPath('/')).toBeNull();
  });
});

describe('soonestGames', () => {
  const dated = (gameDate, key) => ({ gameDate, key });

  test('sorts before capping, so the cap drops the latest', () => {
    const picked = soonestGames([
      dated('2026-04-13', 'c'),
      dated('2026-04-11', 'a'),
      dated('2026-04-14', 'd'),
      dated('2026-04-12', 'b'),
      dated('2026-04-16', 'f'),
      dated('2026-04-15', 'e'),
    ]);
    expect(picked.map((g) => g.key)).toEqual(['a', 'b', 'c', 'd', 'e']);
  });

  test('ties keep schedule order rather than shuffling', () => {
    const same = ['x', 'y', 'z'].map((k) => dated('2026-04-12', k));
    expect(soonestGames(same).map((g) => g.key)).toEqual(['x', 'y', 'z']);
  });

  test('fewer than five shows what exists, unpadded', () => {
    expect(soonestGames([dated('2026-04-12', 'x')])).toHaveLength(1);
  });

  test('byDate does not reorder the caller array', () => {
    const games = [dated('2026-04-13', 'c'), dated('2026-04-11', 'a')];
    byDate(games);
    expect(games.map((g) => g.key)).toEqual(['c', 'a']);
  });
});

describe('routing', () => {
  test('/predictions renders General and does not redirect', async () => {
    getSchedule.mockResolvedValue([fixture(0, TODAY)]);
    renderAt('/predictions');

    expect(await screen.findByRole('heading', { name: 'General', level: 1 })).toBeInTheDocument();

    expect(screen.getByRole('link', { name: /Basketball/ })).not.toHaveClass('active');
  });

  test('/predictions/basketball renders Upcoming, scoped to the sport', async () => {
    getSchedule.mockResolvedValue([fixture(0, TODAY)]);
    renderAt('/predictions/basketball');

    expect(await screen.findByRole('heading', { name: 'Upcoming', level: 1 })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Basketball/ })).toHaveClass('active');
    expect(screen.getByText(/Basketball games for 2026-04-13/)).toBeInTheDocument();
  });

  test('General and Upcoming render the same structure, differing only in scope', async () => {
    getSchedule.mockResolvedValue([fixture(0, TODAY), fixture(1, TODAY)]);

    const shapeOf = (root) => ({
      sections: Array.from(root.querySelectorAll('.league-section'), (s) => s.id),
      shortcuts: Array.from(root.querySelectorAll('.league-shortcut'), (c) => c.textContent),
      more: root.querySelector('.league-section-more').getAttribute('href'),
      rows: root.querySelectorAll('.game-row').length,
    });

    const { container: general, unmount } = renderAt('/predictions');
    await screen.findByRole('heading', { name: 'General', level: 1 });
    const generalShape = shapeOf(general);
    unmount();

    const { container: upcoming } = renderAt('/predictions/basketball');
    await screen.findByRole('heading', { name: 'Upcoming', level: 1 });
    const upcomingShape = shapeOf(upcoming);

    expect(generalShape).toEqual(upcomingShape);
    expect(generalShape.sections).toEqual(['league-basketball-nba']);
  });

  test('an unknown sport renders inline with the rail, and costs no request', async () => {
    renderAt('/predictions/tennis');

    expect(screen.getByRole('heading', { name: 'No such sport', level: 1 })).toBeInTheDocument();
    expect(screen.getByRole('navigation', { name: 'Sports' })).toBeInTheDocument();
    expect(getSchedule).not.toHaveBeenCalled();
    expect(createPrediction).not.toHaveBeenCalled();
  });

  test('an unknown league renders inline with the rail', async () => {
    getSchedule.mockResolvedValue([fixture(0, TODAY)]);
    renderAt('/predictions/basketball/nonsense');

    expect(await screen.findByRole('heading', { name: 'No such league', level: 1 })).toBeInTheDocument();
    expect(screen.getByRole('navigation', { name: 'Sports' })).toBeInTheDocument();
  });

  test('a real league under the wrong sport is not found', async () => {
    renderAt('/predictions/tennis/nba');

    expect(await screen.findByRole('heading', { name: 'No such league', level: 1 })).toBeInTheDocument();
  });

  test('an unknown path still goes home', () => {
    renderAt('/does-not-exist');
    expect(
      screen.getByRole('heading', { name: /Computing sports predictions for the love of the game/ })
    ).toBeInTheDocument();
  });
});

describe('today, the cap, and the league page', () => {
  const SPREAD = [
    fixture(0, TODAY),
    fixture(1, YESTERDAY),
    fixture(2, TODAY),
    fixture(3, '2026-04-11'),
    fixture(4, TODAY),
    fixture(5, TODAY),
    fixture(6, YESTERDAY),
    fixture(7, TODAY),
    fixture(8, TODAY),
  ];

  test('General shows today only, capped at five', async () => {
    getSchedule.mockResolvedValue(SPREAD);
    const { container } = renderAt('/predictions');

    await screen.findByRole('heading', { name: 'General', level: 1 });

    const dates = Array.from(container.querySelectorAll('.game-row-when'), (n) => n.textContent);
    expect(dates).toHaveLength(5);
    expect(new Set(dates)).toEqual(new Set([TODAY]));
  });

  test('the league page lists every predictable game, uncapped and sorted', async () => {
    getSchedule.mockResolvedValue(SPREAD);
    const { container } = renderAt('/predictions/basketball/nba');

    await screen.findByRole('heading', { name: 'NBA Predictions', level: 1 });

    const dates = Array.from(container.querySelectorAll('.game-row-when'), (n) => n.textContent);

    expect(dates).toHaveLength(9);
    expect(dates).toEqual([...dates].sort());
    expect(container.querySelector('.league-shortcuts')).toBeNull();
  });

  test('the league page carries a breadcrumb back to both parents', async () => {
    getSchedule.mockResolvedValue([fixture(0, TODAY)]);
    renderAt('/predictions/basketball/nba');

    const crumbs = await screen.findByRole('navigation', { name: 'Breadcrumb' });
    expect(within(crumbs).getByRole('link', { name: 'Predictions' })).toHaveAttribute(
      'href',
      '/predictions'
    );
    expect(within(crumbs).getByRole('link', { name: 'Basketball' })).toHaveAttribute(
      'href',
      '/predictions/basketball'
    );

    expect(within(crumbs).queryByRole('link', { name: 'NBA' })).toBeNull();
  });

  test('More NBA links to the league page', async () => {
    getSchedule.mockResolvedValue([fixture(0, TODAY)]);
    renderAt('/predictions');

    const more = await screen.findByRole('link', { name: 'More NBA' });
    expect(more).toHaveAttribute('href', '/predictions/basketball/nba');

    fireEvent.click(more);
    expect(await screen.findByRole('heading', { name: 'NBA Predictions', level: 1 })).toBeInTheDocument();
  });

  test('navigating between routes does not refetch', async () => {
    getSchedule.mockResolvedValue([fixture(0, TODAY), fixture(1, TODAY)]);
    renderAt('/predictions');

    await screen.findByRole('heading', { name: 'General', level: 1 });
    expect(getSchedule).toHaveBeenCalledTimes(1);
    expect(getHealth).toHaveBeenCalledTimes(1);
    expect(createPrediction).toHaveBeenCalledTimes(2);

    fireEvent.click(screen.getByRole('link', { name: 'More NBA' }));
    await screen.findByRole('heading', { name: 'NBA Predictions', level: 1 });

    const crumbs = screen.getByRole('navigation', { name: 'Breadcrumb' });
    fireEvent.click(within(crumbs).getByRole('link', { name: 'Predictions' }));
    await screen.findByRole('heading', { name: 'General', level: 1 });

    expect(getSchedule).toHaveBeenCalledTimes(1);
    expect(getHealth).toHaveBeenCalledTimes(1);
    expect(createPrediction).toHaveBeenCalledTimes(2);
  });
});

describe('data states', () => {
  test('empty: succeeded with nothing predictable, and the real dataAsOf is shown', async () => {
    getSchedule.mockResolvedValue([]);
    renderAt('/predictions/basketball');

    expect(await screen.findByRole('heading', { name: 'No predictions yet' })).toBeInTheDocument();
    expect(screen.getByText(/Model data is current to 2026-04-12/)).toBeInTheDocument();
  });

  test('empty: fixtures exist but none are within the cutoff', async () => {
    getSchedule.mockResolvedValue([fixture(0, '2026-10-20')]);
    renderAt('/predictions/basketball');

    expect(await screen.findByRole('heading', { name: 'No predictions yet' })).toBeInTheDocument();
    expect(createPrediction).not.toHaveBeenCalled();
  });

  test('error: a failed request is not shown as an empty schedule', async () => {
    getSchedule.mockRejectedValue(new Error('Failed to fetch'));
    renderAt('/predictions');

    expect(await screen.findByRole('heading', { name: 'Could not load predictions' })).toBeInTheDocument();
    expect(screen.queryByText('No predictions yet')).not.toBeInTheDocument();
    expect(screen.getByText(/Failed to fetch/)).toBeInTheDocument();
  });

  test('error: retry re-issues the request', async () => {
    getSchedule.mockRejectedValueOnce(new Error('Failed to fetch'));
    renderAt('/predictions');

    const retry = await screen.findByRole('button', { name: 'Try again' });
    getSchedule.mockResolvedValue([]);
    fireEvent.click(retry);

    expect(await screen.findByRole('heading', { name: 'No predictions yet' })).toBeInTheDocument();
    expect(getSchedule).toHaveBeenCalledTimes(2);
  });
});

describe('scroll-spy', () => {
  let callback;
  let observed;

  beforeEach(() => {
    callback = null;
    observed = [];
    global.IntersectionObserver = class {
      constructor(fn) {
        callback = fn;
      }
      observe(element) {
        observed.push(element);
      }
      disconnect() {}
    };
  });

  afterEach(() => {
    delete global.IntersectionObserver;
  });

  test('observes every section and marks the first shortcut', async () => {
    getSchedule.mockResolvedValue([fixture(0, TODAY)]);
    const { container } = renderAt('/predictions');

    await screen.findByRole('heading', { name: 'General', level: 1 });

    const sections = Array.from(container.querySelectorAll('.league-section'), (s) => s.id);
    expect(observed.map((el) => el.id)).toEqual(sections);

    const chip = screen.getByRole('button', { name: 'NBA' });
    expect(chip).toHaveClass('league-shortcut--active');
    expect(chip).toHaveAttribute('aria-current', 'location');

    act(() => callback([{ target: observed[0], isIntersecting: true }]));
    expect(screen.getByRole('button', { name: 'NBA' })).toHaveClass('league-shortcut--active');
  });

  test('clicking a shortcut scrolls its section into view', async () => {
    getSchedule.mockResolvedValue([fixture(0, TODAY)]);
    const { container } = renderAt('/predictions');

    await screen.findByRole('heading', { name: 'General', level: 1 });

    const section = container.querySelector('.league-section');
    section.scrollIntoView = jest.fn();

    fireEvent.click(screen.getByRole('button', { name: 'NBA' }));

    expect(section.scrollIntoView).toHaveBeenCalledWith({
      behavior: 'smooth',
      block: 'start',
    });
  });
});

test('no IntersectionObserver: the shortcut bar still renders and marks a chip', async () => {
  expect(global.IntersectionObserver).toBeUndefined();

  getSchedule.mockResolvedValue([fixture(0, TODAY)]);
  renderAt('/predictions');

  await screen.findByRole('heading', { name: 'General', level: 1 });
  expect(screen.getByRole('button', { name: 'NBA' })).toHaveClass('league-shortcut--active');
});

test('the rail still lists only sports that exist', async () => {
  renderAt('/predictions');

  const rail = screen.getByRole('navigation', { name: 'Sports' });
  await waitFor(() => expect(within(rail).getAllByRole('link')).toHaveLength(1));
});
