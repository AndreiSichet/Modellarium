import { useRef } from 'react';

import './DetailTabs.css';

export const PLAYER_STATS = [
  { id: 'points', label: 'Player Points', field: 'predictedPoints', short: 'Points' },
  { id: 'rebounds', label: 'Player Rebounds', field: 'predictedRebounds', short: 'Rebounds' },
  { id: 'assists', label: 'Player Assists', field: 'predictedAssists', short: 'Assists' },
  { id: 'threes', label: 'Player Threes', field: 'predictedThreesMade', short: 'Threes' },
  { id: 'pra', label: 'Player PRA', field: 'predictedPra', short: 'Points + rebounds + assists' },
];

export const GAME_TAB = { id: 'game', label: 'Game' };
export const QUARTERS_TAB = { id: 'quarters', label: 'Quarters & Halves' };

/** Every tab, which is the NBA's set. */
export const TABS = [GAME_TAB, QUARTERS_TAB, ...PLAYER_STATS];

/**
 * The tabs one league actually has.
 *
 * A tab for a market a league does not serve would advertise something that
 * is not there and then render an empty panel or a failed request - the same
 * rule that keeps a shortcut chip off a league with no predictions. The WNBA
 * serves three markets, all of them on the Game tab, so it gets one tab and
 * the caller drops the tablist entirely.
 */
export function tabsFor(league) {
  const markets = league?.markets ?? {};

  return [
    GAME_TAB,
    ...(markets.quarterHalf ? [QUARTERS_TAB] : []),
    ...(markets.playerProps ? PLAYER_STATS : []),
  ];
}

export function tabId(id) {
  return `detail-tab-${id}`;
}

export function panelId(id) {
  return `detail-panel-${id}`;
}

export function isPlayerTab(id) {
  return PLAYER_STATS.some((stat) => stat.id === id);
}

function DetailTabs({ active, onSelect, tabs = TABS }) {
  const ref = useRef(null);

  function onKeyDown(event) {
    const index = tabs.findIndex((tab) => tab.id === active);
    let next = null;

    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    else if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = tabs.length - 1;
    else return;

    // Only after the key is recognised as ours, or Tab is swallowed too.
    event.preventDefault();
    onSelect(tabs[next].id);
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
      {tabs.map((tab) => (
        <button
          key={tab.id}
          type="button"
          role="tab"
          id={tabId(tab.id)}
          aria-selected={tab.id === active}

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
