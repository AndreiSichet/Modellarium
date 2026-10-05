export const DEV_FIXTURES_ON = process.env.REACT_APP_DEV_FIXTURES === '1';

/**
 * Fixtures in the WIRE shape, so dev exercises the same mapping production
 * does.
 *
 * `league`, not `leagueSlug`, because that is what `ScheduledGameDto`
 * actually sends. These used to say `leagueSlug` and so agreed with the
 * components while disagreeing with the server - which is precisely how the
 * field-name mismatch survived: every component test mocks `getSchedule` and
 * supplied the internal name, so nothing pinned the wire one. The layout
 * runs this through `normaliseSchedule` for the same reason it runs the
 * prediction fixtures through `normalisePredictionBody`.
 */
export const DEV_SCHEDULE = [
  {
    league: 'nba',
    homeTeamId: 1610612749,
    homeTeamAbbr: 'MIL',
    homeTeamName: 'Milwaukee Bucks',
    awayTeamId: 1610612741,
    awayTeamAbbr: 'CHI',
    awayTeamName: 'Chicago Bulls',
    gameDate: '2026-04-12',
  },
  {
    league: 'nba',
    homeTeamId: 1610612738,
    homeTeamAbbr: 'BOS',
    homeTeamName: 'Boston Celtics',
    awayTeamId: 1610612747,
    awayTeamAbbr: 'LAL',
    awayTeamName: 'Los Angeles Lakers',
    gameDate: '2026-04-13',
  },
  {
    league: 'nba',
    homeTeamId: 1610612757,
    homeTeamAbbr: 'POR',
    homeTeamName: 'Portland Trail Blazers',
    awayTeamId: 1610612750,
    awayTeamAbbr: 'MIN',
    awayTeamName: 'Minnesota Timberwolves',
    gameDate: '2026-04-13',
  },
  {
    league: 'nba',
    homeTeamId: 1610612761,
    homeTeamAbbr: 'TOR',
    homeTeamName: 'Toronto Raptors',
    awayTeamId: 1610612765,
    awayTeamAbbr: 'DET',
    awayTeamName: 'Detroit Pistons',
    gameDate: '2026-04-11',
  },
  {
    league: 'nba',
    homeTeamId: 1610612744,
    homeTeamAbbr: 'GSW',
    homeTeamName: 'Golden State Warriors',
    awayTeamId: 1610612759,
    awayTeamAbbr: 'SAS',
    awayTeamName: 'San Antonio Spurs',
    gameDate: '2026-04-13',
  },
  {
    league: 'nba',
    homeTeamId: 1610612999,
    homeTeamAbbr: 'XXX',
    homeTeamName: 'Relocated Franchise',
    awayTeamId: 1610612751,
    awayTeamAbbr: 'BKN',
    awayTeamName: 'Brooklyn Nets',
    gameDate: '2026-04-13',
  },
  {
    league: 'nba',
    homeTeamId: 1610612745,
    homeTeamAbbr: 'HOU',
    homeTeamName: 'Houston Rockets',
    awayTeamId: 1610612742,
    awayTeamAbbr: 'DAL',
    awayTeamName: 'Dallas Mavericks',
    gameDate: '2026-04-12',
  },
  {
    league: 'nba',
    homeTeamId: 1610612752,
    homeTeamAbbr: 'NYK',
    homeTeamName: 'New York Knicks',
    awayTeamId: 1610612748,
    awayTeamAbbr: 'MIA',
    awayTeamName: 'Miami Heat',
    gameDate: '2026-04-13',
  },
  {
    league: 'nba',
    homeTeamId: 1610612743,
    homeTeamAbbr: 'DEN',
    homeTeamName: 'Denver Nuggets',
    awayTeamId: 1610612760,
    awayTeamAbbr: 'OKC',
    awayTeamName: 'Oklahoma City Thunder',
    gameDate: '2026-04-13',
  },

  // WNBA. Dated to the WNBA's own cutoff + 1, not the NBA's - so a bug that
  // judged these by NBA freshness would show them on the wrong date or not
  // at all.
  {
    league: 'wnba',
    homeTeamId: 1611661317,
    homeTeamAbbr: 'PHX',
    homeTeamName: 'Phoenix Mercury',
    awayTeamId: 1611661319,
    awayTeamAbbr: 'LVA',
    awayTeamName: 'Las Vegas Aces',
    gameDate: '2026-09-25',
  },
  {
    // ATL and CHI both exist in the NBA too. Rendered side by side with the
    // Hawks or the Bulls, a lookup that resolved teams by abbreviation would
    // show identical badges here.
    league: 'wnba',
    homeTeamId: 1611661330,
    homeTeamAbbr: 'ATL',
    homeTeamName: 'Atlanta Dream',
    awayTeamId: 1611661329,
    awayTeamAbbr: 'CHI',
    awayTeamName: 'Chicago Sky',
    gameDate: '2026-09-25',
  },
];

// Two cutoffs, because the real payload has two and they differ. A single one
// here would let a bug that judges the WNBA by NBA freshness pass in dev.
export const DEV_HEALTH = {
  dataAsOf: '2026-04-12',
  stale: true,
  daysBehind: 161,
  wnba: { dataAsOf: '2026-09-24', stale: true, daysBehind: 10 },
};

const WNBA_PREDICTIONS = {
  // Phoenix Mercury vs Las Vegas Aces - the reference fixture, carrying the
  // values the serving path actually returns for it.
  1611661317: {
    homeWinProbability: 0.18193019489666626,
    homeMargin: -8.403725674288566,
    totalPoints: 178.40689601339275,
  },
  // Atlanta Dream vs Chicago Sky. BOTH abbreviations also belong to an NBA
  // team, so this fixture makes the collision visible in a browser rather
  // than only provable in a test.
  1611661330: {
    homeWinProbability: 0.5312,
    homeMargin: 1.9044,
    totalPoints: 164.2718,
  },
};

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

/**
 * Each league's response in ITS OWN backend DTO shape.
 *
 * Shaped from the Java records - GameSummaryDto {id, latestPrediction} and
 * WnbaSummaryDto {gameId, prediction} - and NOT from the inference service's
 * snake_case bodies, which never reach a browser. The caller runs these
 * through the same normaliser a real response goes through, so a wrong
 * mapping breaks dev too instead of dev agreeing with itself.
 */
export function devPredictionFor(game, index) {
  if (game.leagueSlug === 'wnba') {
    const values = WNBA_PREDICTIONS[game.homeTeamId] || WNBA_PREDICTIONS[1611661317];

    return {
      gameId: 950 + index,
      homeTeamAbbreviation: game.homeTeamAbbr,
      awayTeamAbbreviation: game.awayTeamAbbr,
      gameDate: game.gameDate,
      prediction: {
        ...values,
        moneylineWindow: 'CARRY5',
        spreadWindow: 'CARRY5',
        totalsWindow: 'CARRY10',
        dataAsOf: '2026-09-24',
        stale: true,
        daysBehind: 10,
        predictedAt: '2026-10-03T19:49:17Z',
      },
    };
  }

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
