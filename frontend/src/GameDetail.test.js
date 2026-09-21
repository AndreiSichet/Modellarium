import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

import App from './App';
import {
  createPrediction,
  createQuarterHalfPrediction,
  getHealth,
  getPlayerPropPredictions,
  getSchedule,
} from './api';

jest.mock('./api');

const HEALTH = { dataAsOf: '2026-04-12', daysBehind: 161, stale: true };
const TODAY = '2026-04-13';

const FIXTURE = {
  homeTeamId: 1610612738,
  homeTeamAbbr: 'BOS',
  homeTeamName: 'Boston Celtics',
  awayTeamId: 1610612747,
  awayTeamAbbr: 'LAL',
  awayTeamName: 'Los Angeles Lakers',
  gameDate: TODAY,
};

const SUMMARY = {
  id: 4,
  homeTeamAbbreviation: 'BOS',
  awayTeamAbbreviation: 'LAL',
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

const QUARTER_HALF = {
  gameId: 4,
  gameDate: TODAY,
  prediction: {
    q1Spread: -1.1241,
    q1Total: 59.2612,
    q1WinnerProbability: 0.4469873,
    q1WinnerConfidence: 'low',
    q1WinnerInterpretation: 'P(home leads | not tied)',
    half1Spread: -1.1725,
    half1Total: 117.8875,
    half1WinnerProbability: 0.45369307,
    half1WinnerConfidence: 'medium',
    half1WinnerInterpretation: 'P(home leads | not tied)',
    dataAsOf: '2026-04-12',
    stale: true,
    daysBehind: 161,
    predictedAt: '2026-09-20T10:00:00Z',
  },
};

const NOTE_HOME =
  '18 players (18 via linear, 0 via xgb).  AVAILABILITY UNKNOWN - no injury report was available, so nobody has been excluded. This is not a clean bill of health.';
const NOTE_AWAY =
  '16 players (14 via linear, 2 via xgb).  AVAILABILITY UNKNOWN - no injury report was available.';

function line(playerId, playerName, modelUsed, points, rebounds, assists, threes) {
  return {
    playerId,
    playerName,
    predictedPoints: points,
    predictedRebounds: rebounds,
    predictedAssists: assists,
    predictedThreesMade: threes,
    predictedPra: points + rebounds + assists,
    modelUsed,
  };
}

const PLAYER_PROPS = {
  gameId: 4,
  gameDate: TODAY,
  homeTeam: {
    teamId: 1610612738,
    teamAbbreviation: 'BOS',
    availabilityKnown: false,
    availabilityNote: NOTE_HOME,
    players: [

      line(2, 'Second Scorer', 'linear', 22.8, 5.4, 3.3, 2.4),
      line(1, 'Top Scorer', 'linear', 27.4, 8.1, 4.6, 3.2),
      line(3, 'Big Rebounder', 'xgb', 6.1, 12.4, 1.2, 0.0),
    ],
  },
  awayTeam: {
    teamId: 1610612747,
    teamAbbreviation: 'LAL',
    availabilityKnown: false,
    availabilityNote: NOTE_AWAY,
    players: [line(11, 'Away Star', 'linear', 29.1, 8.7, 8.2, 3.6)],
  },
};

const DETAIL_URL = '/predictions/basketball/nba/4';

function renderAt(path) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>
  );
}

beforeEach(() => {
  jest.clearAllMocks();
  getSchedule.mockResolvedValue([FIXTURE]);
  getHealth.mockResolvedValue(HEALTH);
  createPrediction.mockResolvedValue(SUMMARY);
  createQuarterHalfPrediction.mockResolvedValue(QUARTER_HALF);
  getPlayerPropPredictions.mockResolvedValue(PLAYER_PROPS);
});

async function openDetail(path = DETAIL_URL) {
  const view = renderAt(path);
  await screen.findByRole('tab', { name: 'Game' });
  return view;
}

function openTab(name) {
  fireEvent.click(screen.getByRole('tab', { name }));
}

describe('reaching the page', () => {
  test('More on this carries sport and league from all three list pages', async () => {
    for (const from of ['/predictions', '/predictions/basketball', '/predictions/basketball/nba']) {
      const view = renderAt(from);
      const link = await screen.findByRole('link', { name: 'More on this' });
      expect(link).toHaveAttribute('href', DETAIL_URL);
      view.unmount();
    }
  });

  test('the breadcrumb names all three parents plus the matchup', async () => {
    await openDetail();

    const crumbs = screen.getByRole('navigation', { name: 'Breadcrumb' });
    expect(within(crumbs).getByRole('link', { name: 'Predictions' })).toHaveAttribute('href', '/predictions');
    expect(within(crumbs).getByRole('link', { name: 'Basketball' })).toHaveAttribute('href', '/predictions/basketball');
    expect(within(crumbs).getByRole('link', { name: 'NBA' })).toHaveAttribute('href', '/predictions/basketball/nba');

    expect(within(crumbs).getByText('LAL @ BOS')).toBeInTheDocument();
    expect(within(crumbs).queryByRole('link', { name: 'LAL @ BOS' })).toBeNull();
  });

  test('the header shows both teams and the date, and invents no tip-off time', async () => {
    const { container } = await openDetail();

    const header = container.querySelector('.game-header');
    expect(within(header).getByText('Boston Celtics')).toBeInTheDocument();
    expect(within(header).getByText('Los Angeles Lakers')).toBeInTheDocument();
    expect(within(header).getByText(TODAY)).toBeInTheDocument();

    expect(within(header).queryByText(/\d{1,2}:\d{2}/)).toBeNull();
  });

  test('an unknown gameId renders inline not-found with the rail intact', async () => {
    renderAt('/predictions/basketball/nba/99999');

    expect(await screen.findByRole('heading', { name: 'No such game', level: 1 })).toBeInTheDocument();
    expect(screen.getByRole('navigation', { name: 'Sports' })).toBeInTheDocument();
    expect(screen.getByRole('navigation', { name: 'Breadcrumb' })).toBeInTheDocument();

    expect(createQuarterHalfPrediction).not.toHaveBeenCalled();
  });
});

describe('fetching', () => {
  test('quarter/half is fetched on mount; the game tab costs no extra POST', async () => {
    await openDetail();

    expect(createQuarterHalfPrediction).toHaveBeenCalledTimes(1);
    expect(createQuarterHalfPrediction).toHaveBeenCalledWith({
      homeTeamId: 1610612738,
      awayTeamId: 1610612747,
      gameDate: TODAY,
    });

    expect(createPrediction).toHaveBeenCalledTimes(1);

    expect(getPlayerPropPredictions).not.toHaveBeenCalled();
  });

  test('player props are not fetched until a player tab is opened', async () => {
    await openDetail();

    openTab('Quarters & Halves');
    await screen.findByText('low confidence');
    expect(getPlayerPropPredictions).not.toHaveBeenCalled();

    openTab('Player Points');
    await screen.findByText('Top Scorer');
    expect(getPlayerPropPredictions).toHaveBeenCalledTimes(1);
  });

  test('a second player tab, and a return to Game, refetch nothing', async () => {
    await openDetail();

    openTab('Player Points');
    await screen.findByText('Top Scorer');

    openTab('Player Rebounds');
    await screen.findByText('Big Rebounder');
    openTab('Player PRA');
    await screen.findByText('Top Scorer');
    openTab('Game');
    await screen.findByText('57.5%');
    openTab('Quarters & Halves');
    await screen.findByText('low confidence');

    expect(getPlayerPropPredictions).toHaveBeenCalledTimes(1);
    expect(createQuarterHalfPrediction).toHaveBeenCalledTimes(1);
    expect(createPrediction).toHaveBeenCalledTimes(1);
    expect(getSchedule).toHaveBeenCalledTimes(1);
  });
});

describe('Game tab', () => {
  test('shows all seven team-level markets', async () => {
    const { container } = await openDetail();
    const panel = container.querySelector('.detail-panel');

    expect(within(panel).getByText('57.5%')).toBeInTheDocument();
    expect(within(panel).getByText('42.5%')).toBeInTheDocument();
    expect(within(panel).getByText('232.9')).toBeInTheDocument();
    expect(within(panel).getByText('-0.3')).toBeInTheDocument();
    expect(within(panel).getByText('88.8')).toBeInTheDocument();
    expect(within(panel).getByText('2.3')).toBeInTheDocument();
    expect(within(panel).getByText('51.1')).toBeInTheDocument();
    expect(within(panel).getByText('STALE')).toBeInTheDocument();
  });

  test('the spread matches the list row exactly', async () => {
    const view = renderAt('/predictions/basketball/nba');
    const row = await screen.findByText('More on this');
    const rowCells = Array.from(
      row.closest('.game-row').querySelectorAll('.game-row-cell'),
      (n) => n.textContent
    );
    view.unmount();

    const { container } = await openDetail();
    const panel = container.querySelector('.detail-panel');

    expect(rowCells).toContain('-1.4');
    expect(rowCells).toContain('+1.4');
    expect(within(panel).getByText('-1.4')).toBeInTheDocument();
    expect(within(panel).getByText('+1.4')).toBeInTheDocument();
  });

  test('the markets are grouped, not listed flat', async () => {
    const { container } = await openDetail();

    const titles = Array.from(
      container.querySelectorAll('.market-group-title'),
      (n) => n.textContent
    );
    expect(titles).toEqual(['Outcome', 'Totals', 'Rebounds', 'Assists']);
  });
});

describe('Quarters & Halves tab', () => {
  test('confidence is per market, with q1_winner labelled low', async () => {
    await openDetail();
    openTab('Quarters & Halves');

    expect(await screen.findByText('low confidence')).toBeInTheDocument();

    expect(screen.getByText('medium confidence')).toBeInTheDocument();
  });

  test('both winner markets show the conditional interpretation as visible text', async () => {
    const { container } = await openDetail();
    openTab('Quarters & Halves');

    await screen.findByText('low confidence');
    const notes = screen.getAllByText('P(home leads | not tied)');
    expect(notes).toHaveLength(2);

    notes.forEach((note) => expect(note).toBeVisible());
    expect(container.querySelector('[title*="not tied"]')).toBeNull();
  });

  test('a tie is explained near the winner markets', async () => {
    await openDetail();
    openTab('Quarters & Halves');

    expect(await screen.findByText(/a tie is a push rather than a wrong call/i)).toBeInTheDocument();
  });

  test('all six markets render', async () => {
    const { container } = await openDetail();
    openTab('Quarters & Halves');

    await screen.findByText('low confidence');
    const panel = container.querySelector('.detail-panel');
    expect(Array.from(container.querySelectorAll('.market-group-title'), (n) => n.textContent)).toEqual([
      'First quarter',
      'First half',
    ]);
    expect(within(panel).getByText('44.7%')).toBeInTheDocument();
    expect(within(panel).getByText('45.4%')).toBeInTheDocument();
    expect(within(panel).getByText('59.3')).toBeInTheDocument();
    expect(within(panel).getByText('117.9')).toBeInTheDocument();
  });
});

describe('player tabs', () => {
  test('both rosters render, sorted by the tab stat, descending', async () => {
    const { container } = await openDetail();
    openTab('Player Points');
    await screen.findByText('Top Scorer');

    const boards = container.querySelectorAll('.player-board');
    expect(boards).toHaveLength(2);

    const homeNames = Array.from(
      boards[1].querySelectorAll('.player-row-name'),
      (n) => n.textContent
    );

    expect(homeNames).toEqual(['Top Scorer', 'Second Scorer', 'Big Rebounder']);
  });

  test('a different tab sorts by its own stat', async () => {
    const { container } = await openDetail();
    openTab('Player Rebounds');
    await screen.findByText('Big Rebounder');

    const homeNames = Array.from(
      container.querySelectorAll('.player-board')[1].querySelectorAll('.player-row-name'),
      (n) => n.textContent
    );

    expect(homeNames).toEqual(['Big Rebounder', 'Top Scorer', 'Second Scorer']);
  });

  test('modelUsed appears per player, and both values are shown', async () => {
    const { container } = await openDetail();
    openTab('Player Points');
    await screen.findByText('Top Scorer');

    const tags = Array.from(container.querySelectorAll('.model-tag'), (n) => n.textContent);
    expect(tags).toHaveLength(4);
    expect(new Set(tags)).toEqual(new Set(['linear', 'xgb']));
  });

  test('the availability note renders verbatim, per team', async () => {
    const { container } = await openDetail();
    openTab('Player Points');
    await screen.findByText('Top Scorer');

    const notes = Array.from(
      container.querySelectorAll('.player-board-note'),
      (node) => node.textContent
    );
    expect(notes).toEqual([NOTE_AWAY, NOTE_HOME]);
  });
});

describe('per-tab states', () => {
  test('a broken player fetch leaves the Game tab intact and retries alone', async () => {
    getPlayerPropPredictions.mockRejectedValue(new Error('player props exploded'));
    await openDetail();

    openTab('Player Points');
    expect(await screen.findByRole('heading', { name: /Could not load the player predictions/ })).toBeInTheDocument();
    expect(screen.getByText('player props exploded')).toBeInTheDocument();

    openTab('Game');
    expect(await screen.findByText('57.5%')).toBeInTheDocument();
    expect(screen.queryByText('player props exploded')).not.toBeInTheDocument();

    openTab('Player Points');
    getPlayerPropPredictions.mockResolvedValue(PLAYER_PROPS);
    fireEvent.click(await screen.findByRole('button', { name: 'Try again' }));

    await screen.findByText('Top Scorer');
    expect(getPlayerPropPredictions).toHaveBeenCalledTimes(2);
    expect(createQuarterHalfPrediction).toHaveBeenCalledTimes(1);
  });

  test('a broken quarter/half fetch is scoped to its own tab', async () => {
    createQuarterHalfPrediction.mockRejectedValue(new Error('quarters exploded'));
    await openDetail();

    expect(screen.getByText('57.5%')).toBeInTheDocument();

    openTab('Quarters & Halves');
    expect(await screen.findByRole('heading', { name: /Could not load the quarter and half markets/ })).toBeInTheDocument();
    expect(screen.getByText('quarters exploded')).toBeInTheDocument();
  });

  test('the backend message survives to the screen', async () => {
    createQuarterHalfPrediction.mockRejectedValue(
      new Error('game_date 2026-10-20 is more than 1 day past the newest game in the data (2026-04-12).')
    );
    await openDetail();

    openTab('Quarters & Halves');
    expect(await screen.findByText(/more than 1 day past the newest game/)).toBeInTheDocument();
    expect(screen.queryByText(/Request failed with status/)).not.toBeInTheDocument();
  });
});

describe('the tab bar', () => {
  test('seven tabs, in order, with the active one marked for assistive tech', async () => {
    await openDetail();

    const tabs = screen.getAllByRole('tab');
    expect(tabs.map((t) => t.textContent)).toEqual([
      'Game',
      'Quarters & Halves',
      'Player Points',
      'Player Rebounds',
      'Player Assists',
      'Player Threes',
      'Player PRA',
    ]);
    expect(tabs[0]).toHaveAttribute('aria-selected', 'true');

    openTab('Player Assists');
    await waitFor(() =>
      expect(screen.getByRole('tab', { name: 'Player Assists' })).toHaveAttribute('aria-selected', 'true')
    );
    expect(screen.getByRole('tab', { name: 'Game' })).toHaveAttribute('aria-selected', 'false');
  });

  test('switching tabs does not change the URL', async () => {
    await openDetail();

    const crumbs = screen.getByRole('navigation', { name: 'Breadcrumb' });
    expect(within(crumbs).getByText('LAL @ BOS')).toBeInTheDocument();

    openTab('Player Threes');
    await screen.findByText('Top Scorer');

    expect(within(crumbs).getByRole('link', { name: 'NBA' })).toHaveAttribute(
      'href',
      '/predictions/basketball/nba'
    );
  });
});
