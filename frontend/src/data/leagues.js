/**
 * THE ONE PLACE A LEAGUE IS DECLARED, the same arrangement SPORTS uses in
 * the rail: adding one is an entry here, not a component edit.
 *
 * EACH LEAGUE CARRIES ITS SPORT, and that field is load-bearing rather than
 * descriptive. The General page lists leagues from every sport at once, so
 * a league has to know which sport it belongs to for two things to work:
 * grouping, and building `/predictions/<sport>/<league>` for its More link.
 * Without it, General could only guess - and guessing would be right today
 * only because there is one sport.
 *
 * Exactly one entry, and deliberately no placeholders. A shortcut bar
 * listing leagues with no predictions behind them promises something the
 * app does not have.
 *
 * This replaces Phase 3's LEAGUES in LeagueTabs.js. That list carried an
 * `Upcoming` entry, which was never a league - it was the everything-view
 * tab. Upcoming is a ROUTE now, so the pretend league is gone with it.
 */
export const LEAGUES = [{ slug: 'nba', label: 'NBA', sport: 'basketball' }];

/** The leagues of one sport, in declaration order. */
export function leaguesForSport(sportSlug) {
  return LEAGUES.filter((league) => league.sport === sportSlug);
}

/**
 * A league within a sport. Both halves are checked: `nba` is a real league
 * but `/predictions/tennis/nba` is not a real page, and returning the NBA
 * for it would render a basketball league under a tennis breadcrumb.
 */
export function findLeague(sportSlug, leagueSlug) {
  return (
    LEAGUES.find(
      (league) => league.sport === sportSlug && league.slug === leagueSlug
    ) || null
  );
}
