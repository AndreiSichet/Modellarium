// What differs between leagues lives here, so components read capability
// rather than branching on a slug. Three kinds of difference are real:
//
//   freshness - which cutoff in /api/health applies. The NBA's is top-level;
//               the WNBA's is nested, because the two leagues' data ends on
//               different dates and one cutoff would be wrong for whichever
//               league it did not come from.
//   markets   - which markets the league actually serves. The WNBA has three
//               against the NBA's eighteen, and a tab or a group for a market
//               that does not exist advertises something that is not there.
//   seasonNote - what to say when there is nothing to predict. The mechanism
//               genuinely differs; see the WNBA entry.
export const LEAGUES = [
  {
    slug: 'nba',
    label: 'NBA',
    sport: 'basketball',
    markets: { rebounds: true, assists: true, quarterHalf: true, playerProps: true },
    seasonStatus: 'No games today — the season begins 20 October 2026',
    seasonNote: [
      'The 2026-27 NBA season begins on 20 October 2026.',
      'Predictions need recent form to work from, so they become available once each team has played enough games for that form to be computed — around ten games in, which is roughly three weeks after opening night.',
    ],
  },
  {
    slug: 'wnba',
    label: 'WNBA',
    sport: 'basketball',
    freshness: 'wnba',
    markets: { rebounds: false, assists: false, quarterHalf: false, playerProps: false },
    seasonStatus: 'No games today — the 2026 season is over',
    // THIS COPY IS NOT THE NBA'S, AND THE DIFFERENCE IS MEASURED.
    //
    // No "ten games in": the NBA uses within-season rolling windows, but the
    // WNBA models were selected on CARRY5 and CARRY10, which draw on the
    // previous season's games. There is no warm-up to wait through, so the
    // NBA's sentence would be false here about the exact mechanism phase 3
    // measured.
    //
    // No 2027 date: the schedule is not published, and the 2027 schedule
    // endpoint errors outright. A date invented here is a date that can be
    // wrong.
    //
    // And it must not imply the app refreshes itself. WNBA data is baked into
    // the image and only advances when its pipeline is run by hand, so the
    // season resuming and predictions returning are two separate events.
    seasonNote: [
      'The 2026 WNBA season is over. Predictions resume when the 2027 regular season begins.',
      'No date is shown for that because the 2027 schedule has not been published yet.',
      'WNBA data is not refreshed automatically, so predictions return once the new season has been loaded rather than on opening night.',
    ],
  },
];

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

/** By slug alone, for paths that carry a league without its sport. */
export function leagueBySlug(leagueSlug) {
  return LEAGUES.find((league) => league.slug === leagueSlug) || null;
}
