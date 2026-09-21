import { useRef } from 'react';

import './DetailTabs.css';

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

export function tabId(id) {
  return `detail-tab-${id}`;
}

export function panelId(id) {
  return `detail-panel-${id}`;
}

export function isPlayerTab(id) {
  return PLAYER_STATS.some((stat) => stat.id === id);
}

function DetailTabs({ active, onSelect }) {
  const ref = useRef(null);

  function onKeyDown(event) {
    const index = TABS.findIndex((tab) => tab.id === active);
    let next = null;

    if (event.key === 'ArrowRight') next = (index + 1) % TABS.length;
    else if (event.key === 'ArrowLeft') next = (index - 1 + TABS.length) % TABS.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = TABS.length - 1;
    else return;

    // Only after the key is recognised as ours, or Tab is swallowed too.
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
