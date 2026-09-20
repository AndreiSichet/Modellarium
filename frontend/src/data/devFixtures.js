/**
 * Development fixtures, off unless REACT_APP_DEV_FIXTURES=1.
 *
 * WHY THEY EXIST. Today is 20 September 2026, the season opens 20 October,
 * `data_as_of` is 12 April and MAX_DAYS_AHEAD is 1 - so nothing in the real
 * schedule is predictable and the list is empty for at least another month.
 * Building a game list nobody can look at is not workable.
 *
 * AN ENV VAR, NOT A QUERY PARAMETER OR A BOOLEAN TO FLIP BACK. A query
 * parameter is reachable in production by anyone who types it; a hardcoded
 * flag is reachable by anyone who forgets. create-react-app inlines
 * REACT_APP_* at build time, so a production bundle built without it cannot
 * take this path at all.
 *
 * THE SHAPES ARE THE BACKEND'S, NOT THE INFERENCE SERVICE'S - and that is
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

/**
 * Shaped exactly like ScheduledGameDto.
 *
 * NINE GAMES ACROSS THREE DATES, DELIBERATELY OUT OF ORDER, and the spread
 * is what makes two behaviours visible rather than merely present:
 *
 *   SIX on 2026-04-13, which is "today" - data_as_of plus MAX_DAYS_AHEAD.
 *   General and Upcoming cap a league at five, so one of those six has to
 *   be dropped, and it is only the right one if the sort ran before the cap.
 *
 *   THREE on earlier dates. Those never appear on General or Upcoming,
 *   which show today only, but the league page shows everything predictable
 *   - so it lists nine where General lists five. Without them the two pages
 *   would look identical and neither cap nor scope would be observable.
 *
 * EVERY DATE IS AT OR BEFORE 2026-04-13. A later one is filtered out as
 * unpredictable before it reaches any list, so the spread has to fit inside
 * that window.
 */
export const DEV_SCHEDULE = [
  {
    homeTeamId: 1610612749,
    homeTeamAbbr: 'MIL',
    homeTeamName: 'Milwaukee Bucks',
    awayTeamId: 1610612741,
    awayTeamAbbr: 'CHI',
    awayTeamName: 'Chicago Bulls',
    gameDate: '2026-04-12',
  },
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
    homeTeamId: 1610612761,
    homeTeamAbbr: 'TOR',
    homeTeamName: 'Toronto Raptors',
    awayTeamId: 1610612765,
    awayTeamAbbr: 'DET',
    awayTeamName: 'Detroit Pistons',
    gameDate: '2026-04-11',
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
    // A deliberately unknown id, so the neutral '?' badge is visible in
    // dev rather than only provable in a test.
    homeTeamId: 1610612999,
    homeTeamAbbr: 'XXX',
    homeTeamName: 'Relocated Franchise',
    awayTeamId: 1610612751,
    awayTeamAbbr: 'BKN',
    awayTeamName: 'Brooklyn Nets',
    gameDate: '2026-04-13',
  },
  {
    homeTeamId: 1610612745,
    homeTeamAbbr: 'HOU',
    homeTeamName: 'Houston Rockets',
    awayTeamId: 1610612742,
    awayTeamAbbr: 'DAL',
    awayTeamName: 'Dallas Mavericks',
    gameDate: '2026-04-12',
  },
  {
    homeTeamId: 1610612752,
    homeTeamAbbr: 'NYK',
    homeTeamName: 'New York Knicks',
    awayTeamId: 1610612748,
    awayTeamAbbr: 'MIA',
    awayTeamName: 'Miami Heat',
    gameDate: '2026-04-13',
  },
  {
    homeTeamId: 1610612743,
    homeTeamAbbr: 'DEN',
    homeTeamName: 'Denver Nuggets',
    awayTeamId: 1610612760,
    awayTeamAbbr: 'OKC',
    awayTeamName: 'Oklahoma City Thunder',
    gameDate: '2026-04-13',
  },
];

/**
 * The freshness payload, in the shape GET /api/health returns. Held here
 * rather than inline in the layout so that the fixture date and the
 * schedule dates above cannot drift apart - every game's predictability is
 * decided from this one value.
 */
export const DEV_HEALTH = { dataAsOf: '2026-04-12', stale: true, daysBehind: 161 };

/**
 * Shaped exactly like GameSummaryDto, keyed by home team so a fixture
 * prediction can be matched to its fixture game.
 *
 * The spread of values is deliberate: near coin-flips, lopsided games, and
 * several in between, so the layout is exercised honestly rather than
 * against numbers that all render the same width.
 */
const PREDICTIONS = {
  1610612738: { homeWinProbability: 0.5745, homeMargin: 1.4149, totalPoints: 232.9323 },
  1610612757: { homeWinProbability: 0.4985, homeMargin: -0.0412, totalPoints: 218.4471 },
  1610612744: { homeWinProbability: 0.8123, homeMargin: 11.8302, totalPoints: 241.0615 },
  1610612999: { homeWinProbability: 0.2216, homeMargin: -8.5507, totalPoints: 205.7788 },
  1610612752: { homeWinProbability: 0.6391, homeMargin: 4.2077, totalPoints: 224.6104 },
  1610612743: { homeWinProbability: 0.7048, homeMargin: 6.9315, totalPoints: 236.1892 },
  1610612749: { homeWinProbability: 0.3517, homeMargin: -3.8824, totalPoints: 229.4436 },
  1610612745: { homeWinProbability: 0.5508, homeMargin: 0.7913, totalPoints: 227.3159 },
  1610612761: { homeWinProbability: 0.4102, homeMargin: -2.6640, totalPoints: 213.8827 },
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

/**
 * QuarterHalfSummaryDto, as POST /api/predictions/quarter-half returns it.
 *
 * NESTED UNDER `prediction`, where GameSummaryDto uses `latestPrediction`.
 * That asymmetry is real and is in the Java records; a fixture that tidied
 * it away would hide a shape the detail page has to unwrap.
 *
 * The two confidence values differ ON PURPOSE. q1_winner really does ship
 * as "low" - it scores 0.5796 against a 0.5184 always-home baseline - and a
 * fixture giving both markets the same label would let a per-market tag
 * regress into a page-level one without failing anything.
 */
export function devQuarterHalfFor(gameDate) {
  return {
    gameId: 4,
    homeTeamAbbreviation: 'ATL',
    awayTeamAbbreviation: 'BOS',
    gameDate,
    prediction: {
      q1Spread: -1.124112908717173,
      q1Total: 59.26116643514509,
      q1WinnerProbability: 0.4469873035961624,
      q1WinnerConfidence: 'low',
      q1WinnerInterpretation: 'P(home leads | not tied)',
      half1Spread: -1.1725223199629748,
      half1Total: 117.88749888276107,
      half1WinnerProbability: 0.45369307064976155,
      half1WinnerConfidence: 'medium',
      half1WinnerInterpretation: 'P(home leads | not tied)',
      dataAsOf: '2026-04-12',
      stale: true,
      daysBehind: 161,
      predictedAt: '2026-09-20T10:00:00Z',
    },
  };
}

/**
 * The player-props board, both teams, five stats each.
 *
 * availabilityKnown IS FALSE, AND THE NOTE IS THE REAL ONE. That is not a
 * placeholder: it is false on every response the service returns today,
 * because the injury-report path is packaged but the NBA publishes nothing
 * between seasons. The note says the absence of a report is not a clean
 * bill of health, and the page renders it verbatim.
 *
 * Both modelUsed values appear, because the hybrid routes per player and a
 * fixture with one value would let the tag regress into a constant.
 */
function devLine(playerId, playerName, modelUsed, points, rebounds, assists, threes) {
  return {
    playerId,
    playerName,
    predictedPoints: points,
    predictedRebounds: rebounds,
    predictedAssists: assists,
    predictedThreesMade: threes,
    predictedPra: points + rebounds + assists,
    modelUsed,
  };
}

export function devPlayerPropsFor(gameDate) {
  return {
    gameId: 4,
    gameDate,
    homeTeam: {
      teamId: 1610612738,
      teamAbbreviation: 'BOS',
      availabilityKnown: false,
      availabilityNote:
        '18 players (18 via linear, 0 via xgb).  AVAILABILITY UNKNOWN - no injury report was available, so nobody has been excluded. This is not a clean bill of health.',
      players: [
        devLine(1, 'Jayson Tatum', 'linear', 27.4, 8.1, 4.6, 3.2),
        devLine(2, 'Jaylen Brown', 'linear', 22.8, 5.4, 3.3, 2.4),
        devLine(3, 'Derrick White', 'linear', 15.2, 4.0, 5.1, 2.9),
        devLine(4, 'Payton Pritchard', 'xgb', 9.7, 3.2, 3.8, 1.8),
        devLine(5, 'Luke Kornet', 'xgb', 6.1, 6.4, 1.2, 0.0),
      ],
    },
    awayTeam: {
      teamId: 1610612747,
      teamAbbreviation: 'LAL',
      availabilityKnown: false,
      availabilityNote:
        '16 players (14 via linear, 2 via xgb).  AVAILABILITY UNKNOWN - no injury report was available, so nobody has been excluded. This is not a clean bill of health.',
      players: [
        devLine(11, 'Luka Doncic', 'linear', 29.1, 8.7, 8.2, 3.6),
        devLine(12, 'Austin Reaves', 'linear', 18.3, 4.5, 5.9, 2.1),
        devLine(13, 'Rui Hachimura', 'linear', 12.6, 4.9, 1.4, 1.5),
        devLine(14, 'Dalton Knecht', 'xgb', 8.4, 2.6, 1.1, 1.9),
        devLine(15, 'Jaxson Hayes', 'xgb', 5.2, 4.1, 0.8, 0.0),
      ],
    },
  };
}
