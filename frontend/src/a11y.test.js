import { fireEvent, render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

import App from './App';
import { createPrediction, createQuarterHalfPrediction, getHealth, getPlayerPropPredictions, getSchedule } from './api';

jest.mock('./api');

/**
 * Phase 7: the accessibility work jsdom CAN see.
 *
 * WHAT THIS FILE IS NOT. It is not an accessibility audit. jsdom has no
 * layout engine and no assistive technology, so three of the things that
 * matter most here are invisible to it:
 *
 *   - whether a focus ring is VISIBLE against cream (a rendering question)
 *   - whether the tab ORDER is sensible (a layout question)
 *   - what a screen reader actually announces (an AT question)
 *
 * Those were checked by hand, and the report says so rather than implying
 * this file covered them. What IS here is the structural half: roles,
 * relationships, landmarks, and keyboard handlers that fire real events.
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

const SUMMARY = {
  id: 4,
  gameDate: TODAY,
  latestPrediction: {
    homeWinProbability: 0.5745,
    homeMargin: 1.4149,
    totalPoints: 232.9,
    reboundMargin: -0.3,
    totalRebounds: 88.8,
    assistMargin: 2.3,
    totalAssists: 51.1,
    dataAsOf: '2026-04-12',
    stale: true,
  },
};

beforeEach(() => {
  jest.clearAllMocks();
  getSchedule.mockResolvedValue([FIXTURE]);
  getHealth.mockResolvedValue(HEALTH);
  createPrediction.mockResolvedValue(SUMMARY);
  createQuarterHalfPrediction.mockResolvedValue({ prediction: {} });
  getPlayerPropPredictions.mockResolvedValue({ homeTeam: null, awayTeam: null });
});

function renderAt(path) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>
  );
}

/* ------------------------------------------------------------ landmarks */

describe('landmarks', () => {
  /**
   * EXACTLY ONE <main> PER ROUTE, and on a predictions page it wraps the
   * CONTENT COLUMN rather than the whole layout. The sports rail is a
   * <nav>; nesting it inside main would file page navigation as page
   * content and make a landmark jump land in the wrong place.
   */
  test('the predictions layout has one main, with the rail beside it', async () => {
    renderAt('/predictions');
    await screen.findByRole('heading', { name: 'General', level: 1 });

    const mains = screen.getAllByRole('main');
    expect(mains).toHaveLength(1);

    const rail = screen.getByRole('navigation', { name: 'Sports' });
    expect(mains[0].contains(rail)).toBe(false);
  });

  test('About has its own main, and the header is a banner', () => {
    renderAt('/');

    expect(screen.getAllByRole('main')).toHaveLength(1);
    expect(screen.getByRole('banner')).toBeInTheDocument();
    expect(screen.getByRole('navigation', { name: 'Main' })).toBeInTheDocument();
  });

  test('the rail is reachable as navigation, not as an unlabelled list', async () => {
    renderAt('/predictions/basketball');
    await screen.findByRole('heading', { name: 'Upcoming', level: 1 });

    const rail = screen.getByRole('navigation', { name: 'Sports' });
    expect(within(rail).getAllByRole('link').length).toBeGreaterThan(0);
  });
});

/* ------------------------------------------------------------ tab roles */

describe('roles are accurate, not decorative', () => {
  /**
   * THE SHORTCUT BAR IS NOT A TABLIST, and labelling it as one would tell a
   * screen reader something false. Its buttons scroll the page to a
   * section; they reveal and hide nothing. A tab claims to switch panels.
   */
  test('the league shortcut bar is navigation, not a tablist', async () => {
    const { container } = renderAt('/predictions');
    await screen.findByRole('heading', { name: 'General', level: 1 });

    const bar = container.querySelector('.league-shortcuts');
    expect(bar.getAttribute('role')).toBeNull();
    expect(bar.tagName).toBe('NAV');
    expect(within(bar).queryAllByRole('tab')).toHaveLength(0);
    // aria-current="location" is the value that means "the current position
    // within a container" — not aria-selected, which implies a panel.
    expect(within(bar).getByRole('button', { name: 'NBA' })).toHaveAttribute(
      'aria-current',
      'location'
    );
  });

  /**
   * THE DETAIL TABS ARE A REAL TABLIST — they do switch panels — so the
   * relationship has to be wired both ways: each tab names its panel, and
   * the panel names the tab that labels it.
   */
  test('each detail tab is wired to the panel it controls', async () => {
    renderAt('/predictions/basketball/nba/4');
    const gameTab = await screen.findByRole('tab', { name: 'Game' });

    const panel = screen.getByRole('tabpanel');
    expect(gameTab).toHaveAttribute('aria-controls', panel.id);
    expect(panel).toHaveAttribute('aria-labelledby', gameTab.id);
    expect(gameTab.id).toBeTruthy();
    expect(panel.id).toBeTruthy();
  });
});

/* --------------------------------------------------------- tab keyboard */

describe('tablist keyboard behaviour', () => {
  async function openTabs() {
    renderAt('/predictions/basketball/nba/4');
    await screen.findByRole('tab', { name: 'Game' });
    return screen.getAllByRole('tab');
  }

  /**
   * A ROVING TABINDEX, AND THE REASON IS A KEYBOARD TRAP. Seven focusable
   * buttons in a row means seven presses of Tab to get past the bar. With
   * only the active tab in the tab order, Tab moves INTO the bar once and
   * then straight out of it.
   */
  test('only the active tab is in the tab order', async () => {
    const tabs = await openTabs();

    expect(tabs[0]).toHaveAttribute('tabindex', '0');
    for (const tab of tabs.slice(1)) {
      expect(tab).toHaveAttribute('tabindex', '-1');
    }
  });

  test('ArrowRight and ArrowLeft move between tabs, and wrap', async () => {
    const tabs = await openTabs();

    fireEvent.keyDown(tabs[0], { key: 'ArrowRight' });
    expect(screen.getByRole('tab', { name: 'Quarters & Halves' })).toHaveAttribute('aria-selected', 'true');

    fireEvent.keyDown(screen.getByRole('tab', { name: 'Quarters & Halves' }), { key: 'ArrowLeft' });
    expect(screen.getByRole('tab', { name: 'Game' })).toHaveAttribute('aria-selected', 'true');

    // Wrapping, so the bar has no dead ends at either edge.
    fireEvent.keyDown(screen.getByRole('tab', { name: 'Game' }), { key: 'ArrowLeft' });
    expect(screen.getByRole('tab', { name: 'Player PRA' })).toHaveAttribute('aria-selected', 'true');
  });

  test('Home and End jump to the ends', async () => {
    const tabs = await openTabs();

    fireEvent.keyDown(tabs[0], { key: 'End' });
    expect(screen.getByRole('tab', { name: 'Player PRA' })).toHaveAttribute('aria-selected', 'true');

    fireEvent.keyDown(screen.getByRole('tab', { name: 'Player PRA' }), { key: 'Home' });
    expect(screen.getByRole('tab', { name: 'Game' })).toHaveAttribute('aria-selected', 'true');
  });

  /**
   * THE HANDLER MUST NOT SWALLOW Tab. preventDefault is called only after
   * the key has been recognised as one of ours — calling it first would
   * block Tab and Shift+Tab and build the exact trap the roving tabindex
   * above exists to prevent.
   */
  test('Tab is left alone', async () => {
    const tabs = await openTabs();

    const event = new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true });
    tabs[0].dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
    // And it changed nothing.
    expect(screen.getByRole('tab', { name: 'Game' })).toHaveAttribute('aria-selected', 'true');
  });

  test('focus follows the selection so the next arrow press works', async () => {
    const tabs = await openTabs();

    tabs[0].focus();
    fireEvent.keyDown(tabs[0], { key: 'ArrowRight' });
    expect(document.activeElement).toBe(screen.getByRole('tab', { name: 'Quarters & Halves' }));
  });
});

/* --------------------------------------------------------------- badges */

/**
 * THE BADGE IS DECORATIVE EVERYWHERE IT APPEARS, which is only true because
 * the team is always named in text beside it — the row, the header and the
 * board all do. If a badge ever stands alone it stops being decorative and
 * needs a label, so this asserts the condition rather than the attribute.
 */
test('every team badge sits beside a text name, so aria-hidden is correct', async () => {
  const { container } = renderAt('/predictions/basketball/nba/4');
  await screen.findByRole('tab', { name: 'Game' });

  const badges = container.querySelectorAll('.team-badge');
  expect(badges.length).toBeGreaterThan(0);

  for (const badge of badges) {
    expect(badge).toHaveAttribute('aria-hidden', 'true');
    // The badge's own text is the abbreviation; its parent must carry a
    // name a screen reader can read instead.
    const sibling = badge.parentElement.textContent.replace(badge.textContent, '').trim();
    expect(sibling.length).toBeGreaterThan(0);
  }
});

test('the logo is hidden and the wordmark is not announced twice', () => {
  const { container } = renderAt('/');

  const logo = container.querySelector('.wordmark-logo');
  expect(logo).toHaveAttribute('alt', '');
  expect(logo).toHaveAttribute('aria-hidden', 'true');
  expect(within(screen.getByRole('banner')).getAllByText(/MODELLARIUM/i)).toHaveLength(1);
});
