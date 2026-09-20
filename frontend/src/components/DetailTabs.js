import { useRef } from 'react';

import './DetailTabs.css';

/**
 * THE FIVE PLAYER STATS, declared once. Each tab is one stat across both
 * rosters, rather than one player across all five - the API returns a board
 * per team, so a stat is the grain the data already has.
 *
 * `field` is the DTO's own property name. Naming it here rather than in the
 * tab body means adding a sixth stat is an entry in this list.
 */
export const PLAYER_STATS = [
  { id: 'points', label: 'Player Points', field: 'predictedPoints', short: 'Points' },
  { id: 'rebounds', label: 'Player Rebounds', field: 'predictedRebounds', short: 'Rebounds' },
  { id: 'assists', label: 'Player Assists', field: 'predictedAssists', short: 'Assists' },
  { id: 'threes', label: 'Player Threes', field: 'predictedThreesMade', short: 'Threes' },
  { id: 'pra', label: 'Player PRA', field: 'predictedPra', short: 'Points + rebounds + assists' },
];

export const TABS = [
  { id: 'game', label: 'Game' },
  { id: 'quarters', label: 'Quarters & Halves' },
  ...PLAYER_STATS,
];

/** Ids shared by the bar and the panel, so the aria wiring cannot
 *  drift between the two files that reference it. */
export function tabId(id) {
  return `detail-tab-${id}`;
}

export function panelId(id) {
  return `detail-panel-${id}`;
}

/** The five that share one lazily-fetched response. */
export function isPlayerTab(id) {
  return PLAYER_STATS.some((stat) => stat.id === id);
}

/**
 * Seven tabs over three prediction domains.
 *
 * TABS ARE PAGE STATE, NOT ROUTES, consistent with the decision taken in
 * Phase 3. Were they routes, the browser Back button would step through tab
 * presses instead of leaving the game - which is not what a tab bar means.
 *
 * THIS IS A REAL TABLIST, unlike the league shortcuts one column over. The
 * distinction is not pedantry: these swap which panel is rendered, so
 * role="tab" and aria-selected are accurate here, where the shortcuts only
 * move the reading position and use aria-current="location" instead.
 */
function DetailTabs({ active, onSelect }) {
  const ref = useRef(null);

  /*
   * ARROW KEYS MOVE BETWEEN TABS; Tab MOVES OUT OF THE BAR.
   *
   * That is the expected behaviour for a tablist and it is not what you get
   * for free. Seven focusable buttons in a row means seven presses of Tab
   * to get past them, which is exactly the kind of keyboard trap that makes
   * a page unusable without a mouse. The fix is a roving tabindex: the
   * active tab is the only one in the tab order, and Left/Right move focus
   * between them.
   *
   * Focus is moved explicitly rather than left to follow state, because
   * activating a tab re-renders the bar and the newly-selected button is a
   * different DOM node — without this the focus ring would land back on the
   * body and the next arrow press would do nothing.
   */
  function onKeyDown(event) {
    const index = TABS.findIndex((tab) => tab.id === active);
    let next = null;

    if (event.key === 'ArrowRight') next = (index + 1) % TABS.length;
    else if (event.key === 'ArrowLeft') next = (index - 1 + TABS.length) % TABS.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = TABS.length - 1;
    else return;

    // Only after deciding this is ours: otherwise Tab and Shift+Tab would
    // be swallowed too, which is the trap this exists to avoid.
    event.preventDefault();
    onSelect(TABS[next].id);
    ref.current?.querySelectorAll('[role="tab"]')[next]?.focus();
  }

  return (
    <div
      className="detail-tabs"
      role="tablist"
      aria-label="Prediction markets"
      ref={ref}
      onKeyDown={onKeyDown}
    >
      {TABS.map((tab) => (
        <button
          key={tab.id}
          type="button"
          role="tab"
          id={tabId(tab.id)}
          aria-selected={tab.id === active}
          // The panel this tab controls, so a screen reader can move
          // straight to it rather than hunting for what changed.
          aria-controls={panelId(tab.id)}
          tabIndex={tab.id === active ? 0 : -1}
          className={
            tab.id === active ? 'detail-tab detail-tab--active' : 'detail-tab'
          }
          onClick={() => onSelect(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

export default DetailTabs;
