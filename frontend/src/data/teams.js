// The only file allowed to hold colour literals: published team colours
// are facts about the world, not design tokens.
export const TEAMS = {
  1610612737: { abbr: 'ATL', name: 'Atlanta Hawks', primary: '#E03A3E', secondary: '#C1D32F' },
  1610612738: { abbr: 'BOS', name: 'Boston Celtics', primary: '#007A33', secondary: '#BA9653' },
  1610612739: { abbr: 'CLE', name: 'Cleveland Cavaliers', primary: '#860038', secondary: '#FDBB30' },
  1610612740: { abbr: 'NOP', name: 'New Orleans Pelicans', primary: '#0C2340', secondary: '#C8102E' },
  1610612741: { abbr: 'CHI', name: 'Chicago Bulls', primary: '#CE1141', secondary: '#000000' },
  1610612742: { abbr: 'DAL', name: 'Dallas Mavericks', primary: '#00538C', secondary: '#002B5E' },
  1610612743: { abbr: 'DEN', name: 'Denver Nuggets', primary: '#0E2240', secondary: '#FEC524' },
  1610612744: { abbr: 'GSW', name: 'Golden State Warriors', primary: '#1D428A', secondary: '#FFC72C' },
  1610612745: { abbr: 'HOU', name: 'Houston Rockets', primary: '#CE1141', secondary: '#000000' },
  1610612746: { abbr: 'LAC', name: 'LA Clippers', primary: '#C8102E', secondary: '#1D42BA' },
  1610612747: { abbr: 'LAL', name: 'Los Angeles Lakers', primary: '#552583', secondary: '#FDB927' },
  1610612748: { abbr: 'MIA', name: 'Miami Heat', primary: '#98002E', secondary: '#F9A01B' },
  1610612749: { abbr: 'MIL', name: 'Milwaukee Bucks', primary: '#00471B', secondary: '#EEE1C6' },
  1610612750: { abbr: 'MIN', name: 'Minnesota Timberwolves', primary: '#0C2340', secondary: '#236192' },
  1610612751: { abbr: 'BKN', name: 'Brooklyn Nets', primary: '#000000', secondary: '#FFFFFF' },
  1610612752: { abbr: 'NYK', name: 'New York Knicks', primary: '#006BB6', secondary: '#F58426' },
  1610612753: { abbr: 'ORL', name: 'Orlando Magic', primary: '#0077C0', secondary: '#C4CED4' },
  1610612754: { abbr: 'IND', name: 'Indiana Pacers', primary: '#002D62', secondary: '#FDBB30' },
  1610612755: { abbr: 'PHI', name: 'Philadelphia 76ers', primary: '#006BB6', secondary: '#ED174C' },
  1610612756: { abbr: 'PHX', name: 'Phoenix Suns', primary: '#1D1160', secondary: '#E56020' },
  1610612757: { abbr: 'POR', name: 'Portland Trail Blazers', primary: '#E03A3E', secondary: '#000000' },
  1610612758: { abbr: 'SAC', name: 'Sacramento Kings', primary: '#5A2D81', secondary: '#63727A' },
  1610612759: { abbr: 'SAS', name: 'San Antonio Spurs', primary: '#C4CED4', secondary: '#000000' },
  1610612760: { abbr: 'OKC', name: 'Oklahoma City Thunder', primary: '#007AC1', secondary: '#EF3B24' },
  1610612761: { abbr: 'TOR', name: 'Toronto Raptors', primary: '#CE1141', secondary: '#000000' },
  1610612762: { abbr: 'UTA', name: 'Utah Jazz', primary: '#002B5C', secondary: '#00471B' },
  1610612763: { abbr: 'MEM', name: 'Memphis Grizzlies', primary: '#5D76A9', secondary: '#12173F' },
  1610612764: { abbr: 'WAS', name: 'Washington Wizards', primary: '#002B5C', secondary: '#E31837' },
  1610612765: { abbr: 'DET', name: 'Detroit Pistons', primary: '#C8102E', secondary: '#1D42BA' },
  1610612766: { abbr: 'CHA', name: 'Charlotte Hornets', primary: '#1D1160', secondary: '#00788C' },

  // WNBA. KEYED BY TEAM_ID LIKE EVERYTHING ELSE, AND THAT IS LOAD-BEARING
  // HERE RATHER THAN TIDY: eight of these abbreviations also belong to an NBA
  // team - ATL, CHI, DAL, IND, MIN, PHX, TOR, WAS - so any lookup that
  // resolved a team by abbreviation would render the Atlanta Dream in the
  // Hawks' colours and say nothing about it. The id ranges do not overlap
  // (1611661313-1611661332 against 1610612737-1610612766).
  1611661313: { abbr: 'NYL', name: 'New York Liberty', primary: '#6ECEB2', secondary: '#000000' },
  1611661317: { abbr: 'PHX', name: 'Phoenix Mercury', primary: '#201747', secondary: '#E56020' },
  1611661319: { abbr: 'LVA', name: 'Las Vegas Aces', primary: '#000000', secondary: '#A7A8AA' },
  1611661320: { abbr: 'LAS', name: 'Los Angeles Sparks', primary: '#552583', secondary: '#FDB927' },
  1611661321: { abbr: 'DAL', name: 'Dallas Wings', primary: '#002B5C', secondary: '#C4D600' },
  1611661322: { abbr: 'WAS', name: 'Washington Mystics', primary: '#002B5C', secondary: '#E31837' },
  1611661323: { abbr: 'CON', name: 'Connecticut Sun', primary: '#0A2240', secondary: '#DC4405' },
  1611661324: { abbr: 'MIN', name: 'Minnesota Lynx', primary: '#0C2340', secondary: '#78BE20' },
  1611661325: { abbr: 'IND', name: 'Indiana Fever', primary: '#002D62', secondary: '#E03A3E' },
  1611661328: { abbr: 'SEA', name: 'Seattle Storm', primary: '#2C5234', secondary: '#FEE11A' },
  1611661329: { abbr: 'CHI', name: 'Chicago Sky', primary: '#418FDE', secondary: '#FDD023' },
  1611661330: { abbr: 'ATL', name: 'Atlanta Dream', primary: '#C8102E', secondary: '#418FDE' },
  1611661331: { abbr: 'GSV', name: 'Golden State Valkyries', primary: '#582C83', secondary: '#FFFFFF' },

  // The two 2026 expansion franchises. Their identities are new and these
  // values are PROVISIONAL rather than confirmed against a published brand
  // guide - the badge is cosmetic, but a guessed colour should be labelled as
  // one rather than sitting beside thirteen verified ones unmarked.
  1611661327: { abbr: 'PDX', name: 'Portland Fire', primary: '#C8102E', secondary: '#000000' },
  1611661332: { abbr: 'TOR', name: 'Toronto Tempo', primary: '#000000', secondary: '#E4002B' },
};

export const UNKNOWN_TEAM = {
  abbr: '?',
  name: 'Unknown team',
  primary: '#6B6A65',
  secondary: '#FFFFFF',
};

export function teamFor(teamId) {
  return TEAMS[teamId] || UNKNOWN_TEAM;
}

function luminance(hex) {
  const channels = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  const [r, g, b] = channels.map((c) =>
    c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  );
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(a, b) {
  const [light, dark] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (light + 0.05) / (dark + 0.05);
}

export function badgeTextColour({ primary, secondary }) {
  if (contrast(primary, secondary) >= 3) return secondary;
  return contrast(primary, '#FFFFFF') >= contrast(primary, '#000000')
    ? '#FFFFFF'
    : '#000000';
}
