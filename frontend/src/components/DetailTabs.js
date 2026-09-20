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
  return (
    <div className="detail-tabs" role="tablist" aria-label="Prediction markets">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          type="button"
          role="tab"
          aria-selected={tab.id === active}
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
