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

/**
 * Phase 6: the game detail page.
 *
 * THIS SUITE REPLACES App.test.js, which exercised the old browse/detail
 * flow through PredictionsFlow. It was written and made green BEFORE that
 * file was deleted — the mapping of which old assertions have equivalents
 * here, and which cover behaviour that no longer exists, is in the report
 * rather than assumed.
 *
 * jsdom has no layout engine, so sticky positioning and scroll behaviour
 * are invisible here and are checked in a browser.
 */
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

/** GameSummaryDto. homeMargin is POSITIVE: the home side is favoured. */
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

/** QuarterHalfSummaryDto — markets nested under `prediction`. */
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
      // Deliberately NOT in points order, so the sort has to do something.
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

/** Waits for the detail page to be past its own mount fetch. */
async function openDetail(path = DETAIL_URL) {
  const result = renderAt(path);
  await screen.findByRole('tab', { name: 'Game' });
  return result;
}

function openTab(name) {
  fireEvent.click(screen.getByRole('tab', { name }));
}

/* --------------------------------------------------------- reaching it */

describe('reaching the page', () => {
  /**
   * THE LINK HAS TO KNOW ITS SPORT AND LEAGUE. Phase 3 hardcoded
   * /predictions/basketball/game/<id>; on General a row can belong to any
   * league, so deriving the sport from the current URL would send every row
   * to whichever sport the reader happened to be looking at.
   */
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
    // The last crumb is where the reader is, so it is text, not a link.
    expect(within(crumbs).getByText('LAL @ BOS')).toBeInTheDocument();
    expect(within(crumbs).queryByRole('link', { name: 'LAL @ BOS' })).toBeNull();
  });

  test('the header shows both teams and the date, and invents no tip-off time', async () => {
    const { container } = await openDetail();

    const header = container.querySelector('.game-header');
    expect(within(header).getByText('Boston Celtics')).toBeInTheDocument();
    expect(within(header).getByText('Los Angeles Lakers')).toBeInTheDocument();
    expect(within(header).getByText(TODAY)).toBeInTheDocument();
    // No time exists anywhere in the chain; rendering one would invent it.
    expect(within(header).queryByText(/\d{1,2}:\d{2}/)).toBeNull();
  });

  test('an unknown gameId renders inline not-found with the rail intact', async () => {
    renderAt('/predictions/basketball/nba/99999');

    expect(await screen.findByRole('heading', { name: 'No such game', level: 1 })).toBeInTheDocument();
    expect(screen.getByRole('navigation', { name: 'Sports' })).toBeInTheDocument();
    expect(screen.getByRole('navigation', { name: 'Breadcrumb' })).toBeInTheDocument();
    // Nothing was fetched for a game that does not exist.
    expect(createQuarterHalfPrediction).not.toHaveBeenCalled();
  });
});

/* -------------------------------------------------------------- fetching */

describe('fetching', () => {
  /**
   * THE GAME TAB REUSES THE LAYOUT'S PREDICTION rather than fetching again.
   * The layout already POSTed for every predictable game to build the list,
   * and POST /api/predictions is append-only — a refetch would return the
   * same seven numbers and write a second row.
   */
  test('quarter/half is fetched on mount; the game tab costs no extra POST', async () => {
    await openDetail();

    expect(createQuarterHalfPrediction).toHaveBeenCalledTimes(1);
    expect(createQuarterHalfPrediction).toHaveBeenCalledWith({
      homeTeamId: 1610612738,
      awayTeamId: 1610612747,
      gameDate: TODAY,
    });
    // One per predictable game from the layout, and not one more.
    expect(createPrediction).toHaveBeenCalledTimes(1);
    // The heavy call has not happened.
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

  /**
   * SWITCHING TABS MUST NEVER REFETCH. This page already costs POSTs
   * against an append-only table; a tab that refetches turns browsing into
   * row-writing.
   */
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

/* ------------------------------------------------------------- game tab */

describe('Game tab', () => {
  test('shows all seven team-level markets', async () => {
    const { container } = await openDetail();
    const panel = container.querySelector('.detail-panel');

    expect(within(panel).getByText('57.5%')).toBeInTheDocument(); // home win
    expect(within(panel).getByText('42.5%')).toBeInTheDocument(); // away win
    expect(within(panel).getByText('232.9')).toBeInTheDocument(); // total points
    expect(within(panel).getByText('-0.3')).toBeInTheDocument(); // reb margin
    expect(within(panel).getByText('88.8')).toBeInTheDocument(); // total reb
    expect(within(panel).getByText('2.3')).toBeInTheDocument(); // ast margin
    expect(within(panel).getByText('51.1')).toBeInTheDocument(); // total ast
    expect(within(panel).getByText('STALE')).toBeInTheDocument();
  });

  /**
   * THE SPREAD SIGN, ASSERTED AGAINST THE ROW ON THE PREVIOUS PAGE, same
   * game, same numbers. Both read predictionFormat, so this catches a
   * second implementation appearing rather than just a wrong constant.
   */
  test('the spread matches the list row exactly', async () => {
    const list = renderAt('/predictions/basketball/nba');
    const row = await screen.findByText('More on this');
    const rowCells = Array.from(
      row.closest('.game-row').querySelectorAll('.game-row-cell'),
      (n) => n.textContent
    );
    list.unmount();

    const { container } = await openDetail();
    const panel = container.querySelector('.detail-panel');

    // homeMargin +1.4149: the favoured home side shows NEGATIVE.
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

/* --------------------------------------------------- quarters and halves */

describe('Quarters & Halves tab', () => {
  /**
   * THE THREE QUALIFIER SURFACES, asserted as RENDERED TEXT rather than as
   * fields in a payload. A prediction without its caveat misleads, and a
   * redesign is exactly when a caveat gets dropped for looking cluttered.
   */
  test('confidence is per market, with q1_winner labelled low', async () => {
    await openDetail();
    openTab('Quarters & Halves');

    expect(await screen.findByText('low confidence')).toBeInTheDocument();
    // Per market, not per page — the 1H winner is labelled differently.
    expect(screen.getByText('medium confidence')).toBeInTheDocument();
  });

  test('both winner markets show the conditional interpretation as visible text', async () => {
    const { container } = await openDetail();
    openTab('Quarters & Halves');

    await screen.findByText('low confidence');
    const notes = screen.getAllByText('P(home leads | not tied)');
    expect(notes).toHaveLength(2);
    // Visible text, not a title attribute — a caveat you must hover to
    // find is one most readers never see, and touch screens have no hover.
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

/* ----------------------------------------------------------- player tabs */

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
    // Fixture order is Second, Top, Big — points order is Top, Second, Big.
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
    // 12.4 rebounds puts the big man top, where he was last on points.
    expect(homeNames).toEqual(['Big Rebounder', 'Top Scorer', 'Second Scorer']);
  });

  /**
   * THE HYBRID ROUTES PER PLAYER, so which half answered is a property of
   * the number rather than an implementation detail. The API enforces that
   * transparency by returning the field; the page has to render it.
   */
  test('modelUsed appears per player, and both values are shown', async () => {
    const { container } = await openDetail();
    openTab('Player Points');
    await screen.findByText('Top Scorer');

    const tags = Array.from(container.querySelectorAll('.model-tag'), (n) => n.textContent);
    expect(tags).toHaveLength(4);
    expect(new Set(tags)).toEqual(new Set(['linear', 'xgb']));
  });

  /**
   * VERBATIM, NEVER PARAPHRASED. The note says the absence of an injury
   * report is not a clean bill of health; softening it inverts its meaning,
   * and an unfiltered roster would read as a confirmed lineup.
   */
  test('the availability note renders verbatim, per team', async () => {
    const { container } = await openDetail();
    openTab('Player Points');
    await screen.findByText('Top Scorer');

    /*
     * textContent, NOT getByText, AND THE DIFFERENCE IS THE POINT OF THIS
     * TEST. Testing Library normalizes the DOM text before comparing but
     * leaves the matcher string alone, so the note's double space after
     * "xgb)." collapses on one side only and never matches. That makes
     * getByText the wrong instrument for an assertion whose whole claim is
     * "verbatim" - it would pass on a paraphrase that happened to
     * normalize the same, and fail on an exact copy. A raw comparison says
     * what is meant: these bytes, unaltered.
     */
    const notes = Array.from(
      container.querySelectorAll('.player-board-note'),
      (node) => node.textContent
    );
    expect(notes).toEqual([NOTE_AWAY, NOTE_HOME]);
  });
});

/* ---------------------------------------------------------- tab isolation */

describe('per-tab states', () => {
  /**
   * THE CHECK THAT MATTERS MOST HERE. A failed player-props call must not
   * blank a Game tab that already succeeded — states are per tab, not per
   * page.
   */
  test('a broken player fetch leaves the Game tab intact and retries alone', async () => {
    getPlayerPropPredictions.mockRejectedValue(new Error('player props exploded'));
    await openDetail();

    openTab('Player Points');
    expect(await screen.findByRole('heading', { name: /Could not load the player predictions/ })).toBeInTheDocument();
    expect(screen.getByText('player props exploded')).toBeInTheDocument();

    // The Game tab still works, with its numbers, not an error.
    openTab('Game');
    expect(await screen.findByText('57.5%')).toBeInTheDocument();
    expect(screen.queryByText('player props exploded')).not.toBeInTheDocument();

    // And the retry re-runs only the call that failed.
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

    // The Game tab is the default and is unaffected.
    expect(screen.getByText('57.5%')).toBeInTheDocument();

    openTab('Quarters & Halves');
    expect(await screen.findByRole('heading', { name: /Could not load the quarter and half markets/ })).toBeInTheDocument();
    expect(screen.getByText('quarters exploded')).toBeInTheDocument();
  });

  /**
   * A REFUSED REQUEST STILL SHOWS THE REAL REASON. Spring puts it in
   * `message` only because include-message=always is set, and api.js reads
   * it — without both, this would say "Request failed with status 400".
   */
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

/* ------------------------------------------------------------------ tabs */

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

  /**
   * TABS ARE PAGE STATE, NOT ROUTES, consistent with Phase 3. Were they
   * routes, Back would step through tab presses instead of leaving the
   * game.
   */
  test('switching tabs does not change the URL', async () => {
    await openDetail();

    const crumbs = screen.getByRole('navigation', { name: 'Breadcrumb' });
    expect(within(crumbs).getByText('LAL @ BOS')).toBeInTheDocument();

    openTab('Player Threes');
    await screen.findByText('Top Scorer');

    // Still the same page: the breadcrumb, which is built from the route
    // params, is unchanged.
    expect(within(crumbs).getByRole('link', { name: 'NBA' })).toHaveAttribute(
      'href',
      '/predictions/basketball/nba'
    );
  });
});
