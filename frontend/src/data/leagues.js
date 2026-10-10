import { formatLongDate } from '../dates';

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
//               genuinely differs; see the WNBA and G League entries.
//
// seasonStatus and seasonNote may each be a FUNCTION of { schedule }, because
// one league's copy needs a date it can only get from the data: the G League's
// regular-season start is derived from the first cached fixture rather than
// typed in. A hardcoded date is a date that can be wrong, and this one was -
// the cache says 27 December where the obvious guess was the 19th.
export const LEAGUES = [
  {
    slug: 'nba',
    label: 'NBA',
    sport: 'basketball',
    rule: 'date',
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
    rule: 'date',
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
    // THE REFRESH CLAUSE WAS REVERSED BY A4, AND THE SENTENCE WAS CORRECTED
    // RATHER THAN DROPPED. It used to say WNBA data is "not refreshed
    // automatically", which was true when the table was baked into the
    // inference image and only advanced when someone ran its pipeline. Since
    // served data moved onto a volume refreshed daily, for all three leagues,
    // that sentence became false - so the season resuming and predictions
    // returning are now the SAME event, and the copy says so. The test that
    // pinned the old phrase pins the new one; a test whose sentence went
    // stale still earns its place, it just needs the right sentence.
    seasonNote: [
      'The 2026 WNBA season is over. Predictions resume when the 2027 regular season begins.',
      'No date is shown for that because the 2027 schedule has not been published yet.',
      'WNBA data is refreshed daily, so predictions return on their own once the 2027 season is under way rather than waiting for anyone to load it.',
    ],
  },
  {
    slug: 'gleague',
    label: 'G League',
    sport: 'basketball',
    freshness: 'gleague',
    rule: 'date',
    // Three markets, like the WNBA: no rebounds, assists, quarter/half or
    // player props, so the detail page renders one tab and therefore no
    // tablist at all.
    markets: { rebounds: false, assists: false, quarterHalf: false, playerProps: false },
    seasonStatus: ({ schedule }) => {
      const opens = firstFixtureDate(schedule, 'gleague');
      return opens
        ? `No games today — the regular season begins ${formatLongDate(opens)}`
        : 'No games today — the regular season has not been scheduled yet';
    },
    // THIS COPY DIFFERS FROM BOTH OTHER LEAGUES, on three counts.
    //
    // The Showcase Cup is named because the G League season OPENS with it and
    // Modellarium predicts none of it - the models are regular-season only,
    // by the type digit the pipeline filters on. Without saying so, a visitor
    // seeing November fixtures with no predictions has no way to know why.
    //
    // The date is DERIVED from the first cached regular-season fixture, not
    // typed. Unlike the WNBA, whose 2027 schedule is unpublished, these
    // fixtures are already in the cache - so there is a real date available
    // and no reason to guess one.
    //
    // And no "ten games in": all three shipped windows are CARRY10, which
    // draw on the previous season, so there is no warm-up to wait through.
    seasonNote: ({ schedule }) => {
      const opens = firstFixtureDate(schedule, 'gleague');

      return [
        opens
          ? `The G League regular season begins ${formatLongDate(opens)}.`
          : 'The G League regular season has not been scheduled yet.',
        'The season opens before that with the Showcase Cup, which is not predicted here — the models are trained on regular-season games only, so Cup fixtures appear nowhere.',
        'Predictions are available from the first regular-season game, because the models carry the previous season forward rather than waiting for form to build up.',
      ];
    },
  },
  {
    slug: 'nfl',
    label: 'NFL',
    sport: 'american-football',
    freshness: 'nfl',
    // THE SERVER DECIDES, AND THIS IS THE ONLY LEAGUE THAT SAYS SO.
    //
    // Basketball predictability is a date - one day after that league's
    // cutoff - which this client computes. The NFL's is a dependency: both
    // teams' previous games must be in history. That was measured in phase 2
    // and implemented in the inference service in phase 4, so the answer
    // travels with each fixture as `predictable` and dates.isPredictable
    // dispatches on this flag. A second copy of the dependency rule in
    // JavaScript would drift from the one that refuses the request.
    rule: 'server',
    // Three markets, so one tab and therefore no tablist - the same shape as
    // the WNBA and G League.
    markets: { rebounds: false, assists: false, quarterHalf: false, playerProps: false },
    // The licence requires attribution wherever this league's data is shown.
    // Taken from the `source` field the API returns rather than written here,
    // so it cannot drift from the backend; this flag only says the line is
    // required for this league.
    attributionRequired: true,
    // The league page groups by week rather than listing by date, and shows
    // fixtures that cannot be predicted yet. A week is the unit an NFL
    // reader thinks in, and the predictable set is a slate across Thursday,
    // Sunday and Monday - so a flat date list would strip out the structure
    // that makes sense of it.
    weeklySlate: true,
    seasonStatus: ({ schedule }) => {
      const opens = firstFixtureDate(schedule, 'nfl');
      return opens
        ? 'No games to predict yet — the week’s slate is not set'
        : 'No games today — no fixtures are scheduled yet';
    },
    // THIS COPY IS THE NFL'S MECHANISM, NOT BASKETBALL'S.
    //
    // No "ten games in" and nothing about rolling-window warm-up: the NFL
    // models carry form across weeks and the gate on a prediction is the
    // dependency rule, not an amount of history. And nothing about injury
    // reports - those are an NBA feature and this league has none.
    //
    // Two states, because they read very differently to a visitor: fixtures
    // are cached but none is predictable yet (typically Monday night to the
    // morning after), or nothing is cached at all (the offseason), where no
    // date is claimed and none is invented - the same discipline the G
    // League's copy follows.
    seasonNote: ({ schedule }) => {
      const opens = firstFixtureDate(schedule, 'nfl');

      if (!opens) {
        return [
          'No NFL fixtures are scheduled here yet, so there is nothing to predict.',
          'No date is shown for the next game because none has been published.',
        ];
      }

      return [
        'Predictions for the next games appear once both teams have played their previous game.',
        'The week’s slate usually arrives the morning after Monday night’s game.',
      ];
    },
  },
];

/**
 * The earliest cached fixture for one league, as an ISO date.
 *
 * The schedule holds regular-season fixtures only - the backend filters on
 * the game-id type digit, which is also why the Showcase Cup is absent from
 * it - so the earliest one IS the regular-season opener.
 */
export function firstFixtureDate(schedule, leagueSlug) {
  const dates = (schedule || [])
    .filter((game) => game?.leagueSlug === leagueSlug && game?.gameDate)
    .map((game) => game.gameDate);

  return dates.length ? dates.reduce((a, b) => (a < b ? a : b)) : null;
}

/**
 * One league's copy, resolved against whatever the page knows.
 *
 * Either a literal or a function, so a league whose copy depends on the data
 * does not force every other league's to become a function too.
 */
export function leagueCopy(value, context) {
  return typeof value === 'function' ? value(context || {}) : value;
}

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
