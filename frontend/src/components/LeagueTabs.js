import './LeagueTabs.css';

/**
 * THE ONE PLACE A LEAGUE IS DECLARED, the same arrangement SPORTS uses in
 * the rail: adding one is an entry here rather than a component edit.
 *
 * `Upcoming` is not a league — it is the everything view, and it groups by
 * league. The real leagues follow it. Only leagues Modellarium actually
 * covers appear: no WNBA or NCAAW placeholders, because a tab bar
 * advertising what does not exist is worse than a short one.
 */
export const LEAGUES = [
  { id: 'upcoming', label: 'Upcoming', grouped: true },
  { id: 'nba', label: 'NBA', grouped: false },
];

/**
 * Tabs drive state inside the page, NOT routes.
 *
 * Phase 2 established /predictions/:sport. Adding /:league on top would put
 * every tab change into browser history, so Back would step through tab
 * presses instead of leaving the page — which is not what a tab bar means.
 */
function LeagueTabs({ active, onSelect }) {
  return (
    <div className="league-tabs" role="tablist" aria-label="Leagues">
      {LEAGUES.map((league) => (
        <button
          key={league.id}
          type="button"
          role="tab"
          aria-selected={active === league.id}
          className={
            active === league.id ? 'league-tab league-tab--active' : 'league-tab'
          }
          onClick={() => onSelect(league.id)}
        >
          {league.label}
        </button>
      ))}
    </div>
  );
}

export default LeagueTabs;
