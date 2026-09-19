/**
 * Development fixtures, off unless REACT_APP_DEV_FIXTURES=1.
 *
 * WHY THEY EXIST. Today is 19 September 2026, the season opens 20 October,
 * `data_as_of` is 12 April and MAX_DAYS_AHEAD is 1 — so nothing in the real
 * schedule is predictable and the list is empty for at least another month.
 * Building a game list nobody can look at is not workable.
 *
 * AN ENV VAR, NOT A QUERY PARAMETER OR A BOOLEAN TO FLIP BACK. A query
 * parameter is reachable in production by anyone who types it; a hardcoded
 * flag is reachable by anyone who forgets. create-react-app inlines
 * REACT_APP_* at build time, so a production bundle built without it cannot
 * take this path at all.
 *
 * THE SHAPES ARE THE BACKEND'S, NOT THE INFERENCE SERVICE'S — and that is
 * worth stating because it is easy to take from the wrong place. The Java
 * fixtures in backend/src/test/resources/fixtures/ are captured
 * inference-service responses and are snake_case: `home_team_id`,
 * `home_win_probability`. The frontend never sees those. It talks to the
 * backend, which returns camelCase records:
 *
 *   getSchedule()       -> ScheduledGameDto
 *                          { homeTeamId, homeTeamAbbr, homeTeamName,
 *                            awayTeamId, awayTeamAbbr, awayTeamName,
 *                            gameDate }
 *   createPrediction()  -> GameSummaryDto
 *                          { id, homeTeamAbbreviation, awayTeamAbbreviation,
 *                            gameDate, played, latestPrediction: {...} }
 *
 * Both shapes below are copied from those record declarations and match the
 * fixture already used by App.test.js, which CLAUDE.md records as having
 * been diffed against real responses rather than trusted.
 */
export const DEV_FIXTURES_ON = process.env.REACT_APP_DEV_FIXTURES === '1';

/** Shaped exactly like ScheduledGameDto. */
export const DEV_SCHEDULE = [
  {
    homeTeamId: 1610612738,
    homeTeamAbbr: 'BOS',
    homeTeamName: 'Boston Celtics',
    awayTeamId: 1610612747,
    awayTeamAbbr: 'LAL',
    awayTeamName: 'Los Angeles Lakers',
    gameDate: '2026-04-13',
  },
  {
    // The long-name case: both sides run past a narrow column.
    homeTeamId: 1610612757,
    homeTeamAbbr: 'POR',
    homeTeamName: 'Portland Trail Blazers',
    awayTeamId: 1610612750,
    awayTeamAbbr: 'MIN',
    awayTeamName: 'Minnesota Timberwolves',
    gameDate: '2026-04-13',
  },
  {
    homeTeamId: 1610612744,
    homeTeamAbbr: 'GSW',
    homeTeamName: 'Golden State Warriors',
    awayTeamId: 1610612759,
    awayTeamAbbr: 'SAS',
    awayTeamName: 'San Antonio Spurs',
    gameDate: '2026-04-13',
  },
  {
    // A deliberately unknown id, so the neutral "?" badge is visible in
    // dev rather than only provable in a test. The map is right today and
    // will be wrong the first time a franchise moves.
    homeTeamId: 1610612999,
    homeTeamAbbr: 'XXX',
    homeTeamName: 'Relocated Franchise',
    awayTeamId: 1610612751,
    awayTeamAbbr: 'BKN',
    awayTeamName: 'Brooklyn Nets',
    gameDate: '2026-04-13',
  },
];

/**
 * Shaped exactly like GameSummaryDto, keyed by home team so a fixture
 * prediction can be matched to its fixture game.
 *
 * The spread of values is deliberate: one near coin-flip, one lopsided, and
 * two in between, so the layout is exercised honestly rather than against
 * four numbers that all render the same width.
 */
const PREDICTIONS = {
  1610612738: { homeWinProbability: 0.5745, homeMargin: 1.4149, totalPoints: 232.9323 },
  1610612757: { homeWinProbability: 0.4985, homeMargin: -0.0412, totalPoints: 218.4471 },
  1610612744: { homeWinProbability: 0.8123, homeMargin: 11.8302, totalPoints: 241.0615 },
  1610612999: { homeWinProbability: 0.2216, homeMargin: -8.5507, totalPoints: 205.7788 },
};

export function devPredictionFor(game, index) {
  const values = PREDICTIONS[game.homeTeamId] || PREDICTIONS[1610612738];
  return {
    id: 900 + index,
    homeTeamAbbreviation: game.homeTeamAbbr,
    awayTeamAbbreviation: game.awayTeamAbbr,
    gameDate: game.gameDate,
    played: false,
    latestPrediction: {
      ...values,
      reboundMargin: -0.3438,
      totalRebounds: 88.7604,
      assistMargin: 2.2516,
      totalAssists: 51.0805,
      dataAsOf: '2026-04-12',
      stale: true,
      predictedAt: '2026-09-19T10:00:00Z',
    },
  };
}
