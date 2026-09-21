export const LEAGUES = [{ slug: 'nba', label: 'NBA', sport: 'basketball' }];

export function leaguesForSport(sportSlug) {
  return LEAGUES.filter((league) => league.sport === sportSlug);
}

export function findLeague(sportSlug, leagueSlug) {
  return (
    LEAGUES.find(
      (league) => league.sport === sportSlug && league.slug === leagueSlug
    ) || null
  );
}
