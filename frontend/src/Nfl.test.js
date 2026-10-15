import fs from 'fs';
import path from 'path';

import { render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

import App from './App';
import { createPredictionFor, getHealth, getSchedule } from './api';
import { LEAGUES, leagueBySlug, leagueCopy } from './data/leagues';
import { TEAMS, abbrOf, nameOf } from './data/teams';
import { isPredictable } from './dates';
import { weekGroups } from './components/WeekGroups';
import { SPORTS } from './components/SportsRail';

jest.mock('./api');

// FOUR CUTOFFS, ALL DIFFERENT. The NFL's data ends 2026-10-08, which is
// neither of the three basketball dates - and the point of most of this file
// is that the NFL is not judged by any of them.
const HEALTH = {
  dataAsOf: '2026-04-12',
  daysBehind: 181,
  stale: true,
  wnba: { dataAsOf: '2026-09-24', daysBehind: 16, stale: true },
  gleague: { dataAsOf: '2026-03-28', daysBehind: 196, stale: true },
  nfl: { dataAsOf: '2026-10-08', daysBehind: 2, stale: false },
};

const NBA_DATE = '2026-04-13';

// Real ids, abbreviations and kickoffs from the live schedule.
const PREDICTABLE = {
  leagueSlug: 'nfl',
  homeTeamId: 1613000012,
  homeTeamAbbr: 'GB',
  homeTeamName: 'Green Bay Packers',
  awayTeamId: 1613000006,
  awayTeamAbbr: 'CHI',
  awayTeamName: 'Chicago Bears',
  gameDate: '2026-10-11',
  kickoffUtc: '2026-10-11T17:00:00Z',
  flex: false,
  week: 5,
  predictable: true,
};

// Same week, no numbers yet - the state most of a week's fixtures are in.
const NOT_YET = {
  leagueSlug: 'nfl',
  homeTeamId: 1613000017,
  homeTeamAbbr: 'MIA',
  homeTeamName: 'Miami Dolphins',
  awayTeamId: 1613000007,
  awayTeamAbbr: 'CIN',
  awayTeamName: 'Cincinnati Bengals',
  gameDate: '2026-10-11',
  kickoffUtc: '2026-10-11T17:00:00Z',
  flex: false,
  week: 5,
  predictable: false,
};

// Next week, and a TBD kickoff - the NFL publishes 24 of those.
const NEXT_WEEK_TBD = {
  leagueSlug: 'nfl',
  homeTeamId: 1613000022,
  homeTeamAbbr: 'NYJ',
  homeTeamName: 'New York Jets',
  awayTeamId: 1613000008,
  awayTeamAbbr: 'CLE',
  awayTeamName: 'Cleveland Browns',
  gameDate: '2026-10-18',
  kickoffUtc: null,
  flex: false,
  week: 6,
  predictable: false,
};

// THE REAL SUNDAY-NIGHT ROLL-OVER, ids and instants from the live schedule.
// Its `gameDate` is the United States date - the 15th - and it kicks off at
// 00:15Z on the 16th, so the date a reader should see is NOT its gameDate.
const SUNDAY_NIGHT = {
  leagueSlug: 'nfl',
  homeTeamId: 1613000010,
  homeTeamAbbr: 'DEN',
  homeTeamName: 'Denver Broncos',
  awayTeamId: 1613000026,
  awayTeamAbbr: 'SEA',
  awayTeamName: 'Seattle Seahawks',
  gameDate: '2026-10-15',
  kickoffUtc: '2026-10-16T00:15:00Z',
  flex: false,
  week: 6,
  predictable: true,
};

// MIDDAY UTC, WHICH IS THE SAME CALENDAR DAY IN EVERY ZONE ON EARTH - 12:00Z
// on the 17th lands on the 17th from UTC-12 to UTC+12 and never on the 15th
// anywhere. So an assertion that `gameDate` is not what gets rendered holds
// wherever this suite runs, rather than only in a zone east of Greenwich.
const DATE_DISAGREES = {
  ...SUNDAY_NIGHT,
  gameDate: '2026-10-15',
  kickoffUtc: '2026-10-17T12:00:00Z',
};

const NBA_FIXTURE = {
  leagueSlug: 'nba',
  homeTeamId: 1610612737,
  homeTeamAbbr: 'ATL',
  homeTeamName: 'Atlanta Hawks',
  awayTeamId: 1610612738,
  awayTeamAbbr: 'BOS',
  awayTeamName: 'Boston Celtics',
  gameDate: NBA_DATE,
};

const NFL_SUMMARY = {
  gameId: 1028,
  prediction: {
    homeWinProbability: 0.4390723587536829,
    homeMargin: -2.7356462478637695,
    totalPoints: 43.34370323002624,
    homeWinInterpretation: 'P(home wins | not tied)',
    winnerModel: 'linear/mov/elo_context',
    marginModel: 'xgboost/mov/elo_context',
    totalsModel: 'linear/mov/elo_context_carry8',
    // CARRIED ON THE WIRE AND DELIBERATELY NOT RENDERED - see the note test.
    winnerNote: 'TIES Elo alone',
    marginNote: 'TIES Elo alone',
    totalsNote: 'no Elo baseline; judged against naive only',
    imputedFeatures: [],
    season: 2026,
    week: 5,
    dataAsOf: '2026-10-08',
    stale: false,
    daysBehind: 2,
    source: 'English Wikipedia, CC BY-SA 4.0',
  },
};

const NBA_SUMMARY = {
  gameId: 4,
  prediction: {
    homeWinProbability: 0.5745,
    homeMargin: 1.4149,
    totalPoints: 232.9323,
    reboundMargin: -0.3438,
    totalRebounds: 88.7604,
    assistMargin: 2.2516,
    totalAssists: 51.0805,
    dataAsOf: '2026-04-12',
    stale: true,
  },
};

function renderAt(path) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>
  );
}

beforeEach(() => {
  jest.clearAllMocks();
  getHealth.mockResolvedValue(HEALTH);
  getSchedule.mockResolvedValue([]);
  createPredictionFor.mockImplementation((leagueSlug) =>
    Promise.resolve(leagueSlug === 'nfl' ? NFL_SUMMARY : NBA_SUMMARY)
  );
});

describe('the sport and its routes', () => {
  test('American football is in the rail, and is not called football', async () => {
    renderAt('/predictions');

    const rail = screen.getByRole('navigation', { name: 'Sports' });
    const link = await within(rail).findByRole('link', {
      name: /American football/,
    });
    expect(link).toHaveAttribute('href', '/predictions/american-football');

    // NOT "Football". For a European reader that word means soccer, and the
    // slug becomes a URL.
    expect(within(rail).queryByRole('link', { name: /^Football$/ })).toBeNull();
    expect(SPORTS.map((sport) => sport.slug)).toContain('american-football');
  });

  test('all three NFL routes resolve with no new route definitions', async () => {
    getSchedule.mockResolvedValue([PREDICTABLE]);

    // UNMOUNTED BETWEEN ROUTES. Three renders in one test leave three trees
    // in the document, so a getBy* query matches the same team three times
    // and reports "found multiple elements" - which looks like a page bug
    // and is a test bug.
    const { unmount: closeUpcoming } = renderAt('/predictions/american-football');
    expect(
      await screen.findByRole('heading', { name: 'Upcoming', level: 1 })
    ).toBeInTheDocument();
    closeUpcoming();

    const { unmount: closeLeague } = renderAt('/predictions/american-football/nfl');
    expect(
      await screen.findByRole('heading', { name: 'NFL Predictions', level: 1 })
    ).toBeInTheDocument();
    closeLeague();

    renderAt(`/predictions/american-football/nfl/${NFL_SUMMARY.gameId}`);
    expect(await screen.findByText(/Green Bay Packers/)).toBeInTheDocument();
  });
});

describe('the server decides NFL predictability, never a date', () => {
  test('a fixture with predictable: false makes no prediction request', async () => {
    getSchedule.mockResolvedValue([NOT_YET]);
    renderAt('/predictions/american-football');

    await screen.findByRole('heading', { name: 'Upcoming', level: 1 });
    expect(createPredictionFor).not.toHaveBeenCalled();
  });

  test('a fixture with predictable: true is requested, on the NFL endpoint', async () => {
    getSchedule.mockResolvedValue([PREDICTABLE]);
    renderAt('/predictions/american-football');

    await screen.findByRole('heading', { name: 'Upcoming', level: 1 });
    expect(createPredictionFor).toHaveBeenCalledWith(
      'nfl',
      expect.objectContaining({ homeTeamId: PREDICTABLE.homeTeamId })
    );
  });

  test('the NFL is never judged by a date rule', () => {
    const nfl = leagueBySlug('nfl');

    // THE FIXTURE IS ON A DATE NO CUTOFF ALLOWS, and still predictable,
    // because its rule does not read dates at all. Under the date rule this
    // would be false: the NFL's cutoff is 2026-10-08, so the latest
    // predictable date would be the 9th and this fixture is on the 11th.
    expect(isPredictable(PREDICTABLE, HEALTH, nfl)).toBe(true);
    expect(PREDICTABLE.gameDate > '2026-10-09').toBe(true);

    // And a fixture the server refuses stays refused however fresh the data.
    expect(isPredictable(NOT_YET, HEALTH, nfl)).toBe(false);
  });

  test('basketball ignores predictable even when it is present', () => {
    const nba = leagueBySlug('nba');

    // A basketball fixture carrying the field is still judged by its date:
    // predictable: false on the predictable date is shown, and
    // predictable: true outside the window is not.
    expect(
      isPredictable({ ...NBA_FIXTURE, predictable: false }, HEALTH, nba)
    ).toBe(true);
    expect(
      isPredictable(
        { ...NBA_FIXTURE, gameDate: '2027-01-01', predictable: true },
        HEALTH,
        nba
      )
    ).toBe(false);
  });

  test('a league declaring no rule offers nothing, and says so', () => {
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});

    // FAILS CLOSED. §53's `|| 'nba'` read as a fallback and was the only
    // path that ever ran; a league with no rule must not inherit
    // basketball's date rule by accident.
    expect(
      isPredictable(PREDICTABLE, HEALTH, { slug: 'mystery' })
    ).toBe(false);
    expect(warn).toHaveBeenCalledWith(expect.stringMatching(/mystery/));

    warn.mockRestore();
  });

  test('every league declares a rule', () => {
    LEAGUES.forEach((league) => {
      expect(['date', 'server']).toContain(league.rule);
    });
    expect(leagueBySlug('nfl').rule).toBe('server');
  });
});

describe('the weekly slate', () => {
  const schedule = [PREDICTABLE, NOT_YET, NEXT_WEEK_TBD];

  test('Upcoming shows predictable fixtures, not a single day', async () => {
    getSchedule.mockResolvedValue(schedule);
    renderAt('/predictions/american-football');

    await screen.findByRole('heading', { name: 'Upcoming', level: 1 });

    // Only the predictable one is offered; the other two have no numbers.
    expect(createPredictionFor).toHaveBeenCalledTimes(1);
    expect(await screen.findByText(/Green Bay Packers/)).toBeInTheDocument();
  });

  test('the league page groups by week and shows what cannot be predicted', async () => {
    getSchedule.mockResolvedValue(schedule);
    renderAt('/predictions/american-football/nfl');

    await screen.findByRole('heading', { name: 'NFL Predictions', level: 1 });

    expect(
      await screen.findByRole('heading', { name: 'Week 5' })
    ).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Week 6' })).toBeInTheDocument();

    // UNPREDICTABLE FIXTURES ARE VISIBLE AND MARKED, not hidden - a reader on
    // a Tuesday should see what is coming and why it has no numbers.
    expect(await screen.findByText(/Miami Dolphins/)).toBeInTheDocument();
    expect(screen.getByText(/New York Jets/)).toBeInTheDocument();
    expect(
      screen.getAllByText(/Not predictable yet/).length
    ).toBeGreaterThanOrEqual(2);
  });

  test('the current week is the lowest still scheduled, derived not clocked', () => {
    const groups = weekGroups(schedule, [], leagueBySlug('nfl'));

    expect(groups.map((group) => group.week)).toEqual([5, 6]);

    // Drop week 5 and week 6 becomes current, with 7 beside it - no date
    // arithmetic, so it cannot disagree with the data.
    const later = weekGroups(
      [NEXT_WEEK_TBD, { ...NEXT_WEEK_TBD, week: 7, gameDate: '2026-10-25' }],
      [],
      leagueBySlug('nfl')
    );
    expect(later.map((group) => group.week)).toEqual([6, 7]);
  });
});

describe('the game detail page', () => {
  async function detail() {
    getSchedule.mockResolvedValue([PREDICTABLE]);
    renderAt(`/predictions/american-football/nfl/${NFL_SUMMARY.gameId}`);
    return screen.findByText(/Green Bay Packers/);
  }

  test('three markets, with the winner qualifier visible', async () => {
    await detail();

    // ONE WORDING ACROSS LEAGUES. None of basketball's three labels is
    // wrong for American football - "Spread" is if anything more native to
    // it, and an NFL total is points - so there is no per-league label map
    // and the same quantity reads the same everywhere.
    expect(screen.getByText('Win probability')).toBeInTheDocument();
    expect(screen.getByText('Spread')).toBeInTheDocument();
    expect(screen.getByText('Total points')).toBeInTheDocument();

    // THE QUALIFIER IS A RENDERED ELEMENT, not a field in a payload. The
    // winner model is trained on decided games, so a bare percentage is a
    // different quantity from what it means.
    expect(
      screen.getByText('P(home wins | not tied)')
    ).toBeInTheDocument();
  });

  test('the manifest note is NOT rendered', async () => {
    await detail();

    // IT IS ON THE WIRE AND STAYS OFF THE PAGE. "TIES Elo alone" is a
    // model-selection note for engineers - the same decision made for the
    // WNBA's withdrawn caveat - and a reader shown it would read it as a
    // confidence statement about this fixture.
    expect(screen.queryByText(/TIES Elo alone/i)).toBeNull();
    expect(screen.queryByText(/judged against naive only/i)).toBeNull();

    // The model identifiers ARE shown, as for the other leagues.
    expect(screen.getByText('linear/mov/elo_context')).toBeInTheDocument();
  });

  test('one tab, so no tablist and no orphaned tab semantics', async () => {
    await detail();

    expect(screen.queryByRole('tablist')).toBeNull();
    expect(screen.queryAllByRole('tab')).toHaveLength(0);
    expect(
      document.querySelector('[aria-labelledby^="detail-tab-"]')
    ).toBeNull();
    expect(document.querySelector('[role="tabpanel"]')).toBeNull();
  });
});

describe('team labels come from the API, never from teams.js', () => {
  // `teams.js` holds 45 of the 108 teams this app serves, because its real
  // job is COLOURS and published colours for the NFL's 32 franchises and the
  // G League's 31 are not something to invent. `teamFor` therefore answers
  // UNKNOWN_TEAM for all of them, and UNKNOWN_TEAM's abbreviation is the
  // literal "?".

  test('neither NFL team is in teams.js, which is what makes this possible', () => {
    expect(TEAMS[SUNDAY_NIGHT.homeTeamId]).toBeUndefined();
    expect(TEAMS[SUNDAY_NIGHT.awayTeamId]).toBeUndefined();
  });

  test('the detail page shows both abbreviations and no lone "?"', async () => {
    getSchedule.mockResolvedValue([SUNDAY_NIGHT]);
    renderAt(`/predictions/american-football/nfl/${NFL_SUMMARY.gameId}`);
    await screen.findByText(/Denver Broncos/);

    // WHAT SHIPPED: the breadcrumb read "? @ ?" and all four market-row side
    // labels read "?", because both read `teamFor(id).abbr` with no API
    // fallback. The badges were already right, which is why only part of the
    // page was wrong and the screenshot was needed to see it.
    expect(screen.queryAllByText('?')).toHaveLength(0);

    expect(screen.getByText('SEA @ DEN')).toBeInTheDocument();
    expect(screen.getAllByText('DEN').length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText('SEA').length).toBeGreaterThanOrEqual(2);
  });

  test('the league page and General name NFL teams, with no lone "?"', async () => {
    getSchedule.mockResolvedValue([SUNDAY_NIGHT, NOT_YET]);

    const { unmount: closeLeague } = renderAt('/predictions/american-football/nfl');
    await screen.findByRole('heading', { name: 'NFL Predictions', level: 1 });
    expect(screen.getByText(/Denver Broncos/)).toBeInTheDocument();
    // The unpredictable row too - it renders through its own component.
    expect(screen.getByText(/Miami Dolphins/)).toBeInTheDocument();
    expect(screen.queryAllByText('?')).toHaveLength(0);
    closeLeague();

    renderAt('/predictions');
    await screen.findByText(/Denver Broncos/);
    expect(screen.queryAllByText('?')).toHaveLength(0);
  });

  test('only TeamBadge reads the table directly, and only for colours', () => {
    // THE ONLY TEST HERE THAT READS SOURCE RATHER THAN RENDERING IT, and it
    // is here because this defect shipped TWICE in the same shape: the G
    // League silently, then the NFL visibly. The three behavioural tests
    // above cover the three surfaces that exist today; a fourth surface added
    // later would pass all of them while reading `teamFor(id).abbr` again.
    // The same reasoning as the design-token scan - a rule worth stating is
    // worth enforcing where it can be broken.
    const dir = path.join(__dirname, 'components');
    const offenders = fs
      .readdirSync(dir)
      .filter((name) => name.endsWith('.js'))
      .filter((name) => name !== 'TeamBadge.js')
      .filter((name) => /\bteamFor\b/.test(fs.readFileSync(path.join(dir, name), 'utf8')));

    // TeamBadge is exempt because it uses `teamFor` for `primary` and
    // `secondary` - colours, which are the table's actual job and which the
    // API does not send. Its LABEL already goes through abbrOf.
    expect(offenders).toEqual([]);
  });

  test('the rule is API first, table second, and is league-agnostic', () => {
    // A G League id, absent from the table for the same reason and carrying
    // the identical defect - invisible only because no G League fixture is
    // predictable today, so its detail page cannot currently be reached.
    expect(TEAMS[1612709911]).toBeUndefined();
    expect(abbrOf(1612709911, 'IWA')).toBe('IWA');
    expect(nameOf(1612709911, 'Iowa Wolves')).toBe('Iowa Wolves');

    // With nothing from the API the table still answers, so a basketball
    // caller that passes no label is unchanged.
    expect(abbrOf(1610612737, undefined)).toBe('ATL');
    expect(nameOf(1610612737, undefined)).toBe('Atlanta Hawks');

    // An id in neither place degrades to the placeholder rather than
    // rendering nothing at all.
    expect(abbrOf(999, undefined)).toBe('?');
  });
});

describe('the kickoff on the detail page', () => {
  async function headerText(fixture) {
    getSchedule.mockResolvedValue([fixture]);
    renderAt(`/predictions/american-football/nfl/${NFL_SUMMARY.gameId}`);
    await screen.findByText(/Denver Broncos/);
    return document.querySelector('.game-header-date').textContent;
  }

  test('the header uses the same formatter as the rows', async () => {
    getSchedule.mockResolvedValue([SUNDAY_NIGHT]);
    const { unmount: closeLeague } = renderAt('/predictions/american-football/nfl');
    await screen.findByRole('heading', { name: 'NFL Predictions', level: 1 });
    const row = document.querySelector('.game-row-when').textContent;
    closeLeague();

    const header = await headerText(SUNDAY_NIGHT);

    // ONE FORMATTER, SO THE TWO SURFACES CANNOT DISAGREE. Asserted as
    // equality rather than against a literal because `formatKickoff` renders
    // the time through Intl at the DEFAULT locale: this runner resolves
    // en-US and prints "3:15 AM", while a browser set to en-GB or ro-RO
    // prints "03:15". Pinning either string would pin the environment.
    expect(header).toBe(row);

    // THE BUG: the header was `game.gameDate` alone, with no time at all.
    expect(header).not.toBe(SUNDAY_NIGHT.gameDate);
  });

  test('the date comes from the kickoff instant, not from gameDate', async () => {
    const header = await headerText(DATE_DISAGREES);

    // 12:00Z is the 17th in every zone, so this holds wherever it runs.
    expect(header).not.toMatch(/2026-10-15/);
    expect(header).toMatch(/2026-10-1[678]/);
  });

  test('a TBD kickoff shows the date alone rather than a guessed time', async () => {
    const header = await headerText({ ...SUNDAY_NIGHT, kickoffUtc: null });

    // The NFL publishes 24 of these. No kickoff means no time, and the
    // scheduled date is all there is to show.
    expect(header).toBe(SUNDAY_NIGHT.gameDate);
  });

  test('a flexed kickoff is marked on the header too', async () => {
    const header = await headerText({ ...SUNDAY_NIGHT, flex: true });

    // A time that may still move is a caveat about what is shown, and the
    // rows carry it - dropping it here is how a qualifier vanishes from one
    // surface while looking present on another.
    expect(header).toMatch(/flex/i);
  });

  test('the instant really is 03:15 on the 16th in Bucharest', () => {
    // THIS TESTS THE INSTANT, NOT OUR CODE, and it is here because the
    // defect was reported in those terms. Pinned to an explicit zone and
    // locale so it means the same thing on every runner.
    const reading = new Intl.DateTimeFormat('en-GB', {
      timeZone: 'Europe/Bucharest',
      dateStyle: 'short',
      timeStyle: 'short',
    }).format(new Date(SUNDAY_NIGHT.kickoffUtc));

    expect(reading).toBe('16/10/2026, 03:15');
    expect(SUNDAY_NIGHT.gameDate).toBe('2026-10-15');
  });

  test('basketball is unchanged: no kickoff, so the bare date', async () => {
    getSchedule.mockResolvedValue([NBA_FIXTURE]);
    renderAt(`/predictions/basketball/nba/${NBA_SUMMARY.gameId}`);
    await screen.findByText(/Atlanta Hawks/);

    expect(document.querySelector('.game-header-date').textContent).toBe(
      NBA_DATE
    );
  });
});

describe('attribution is required by the licence', () => {
  test('it is on the NFL league page and the NFL detail page', async () => {
    getSchedule.mockResolvedValue([PREDICTABLE]);

    const { unmount } = renderAt('/predictions/american-football/nfl');
    await screen.findByRole('heading', { name: 'NFL Predictions', level: 1 });
    const onLeague = document.querySelector('.attribution');
    expect(onLeague).not.toBeNull();
    expect(onLeague.textContent).toMatch(/Wikipedia/);
    expect(onLeague.textContent).toMatch(/CC BY-SA 4\.0/);
    expect(
      within(onLeague).getByRole('link', { name: /Wikipedia/ })
    ).toHaveAttribute('href', 'https://en.wikipedia.org/');
    expect(
      within(onLeague).getByRole('link', { name: /CC BY-SA 4\.0/ })
    ).toHaveAttribute(
      'href',
      'https://creativecommons.org/licenses/by-sa/4.0/'
    );
    unmount();

    renderAt(`/predictions/american-football/nfl/${NFL_SUMMARY.gameId}`);
    await screen.findByText(/Green Bay Packers/);
    expect(document.querySelector('.attribution')).not.toBeNull();
  });

  test('it is absent from basketball pages, by the flag and not by luck', async () => {
    getSchedule.mockResolvedValue([NBA_FIXTURE]);

    // A SOURCE IS PLANTED ON THE BASKETBALL PREDICTION ON PURPOSE. Two
    // things keep attribution off these pages: league.attributionRequired is
    // absent, and a basketball prediction carries no `source` so Attribution
    // renders null anyway. The first version of this test supplied no source
    // and so passed on the second defence alone - removing the flag entirely
    // left it green, which is the vacuous guard this project keeps catching.
    // With a source present, only the flag can keep the line off the page.
    createPredictionFor.mockResolvedValue({
      ...NBA_SUMMARY,
      prediction: {
        ...NBA_SUMMARY.prediction,
        source: 'Planted, to isolate the flag from the missing source',
      },
    });

    const { unmount } = renderAt('/predictions/basketball/nba');
    await screen.findByRole('heading', { name: 'NBA Predictions', level: 1 });
    expect(document.querySelector('.attribution')).toBeNull();
    unmount();

    renderAt(`/predictions/basketball/nba/${NBA_SUMMARY.gameId}`);
    await screen.findByText(/Atlanta Hawks/);
    expect(document.querySelector('.attribution')).toBeNull();
  });

  test('the words come from the API, so they cannot drift', async () => {
    getSchedule.mockResolvedValue([PREDICTABLE]);
    createPredictionFor.mockResolvedValue({
      ...NFL_SUMMARY,
      prediction: {
        ...NFL_SUMMARY.prediction,
        source: 'Some Other Encyclopaedia, ODbL 1.0',
      },
    });

    renderAt('/predictions/american-football/nfl');
    await screen.findByRole('heading', { name: 'NFL Predictions', level: 1 });

    const line = document.querySelector('.attribution');
    expect(line.textContent).toMatch(/Some Other Encyclopaedia/);
    expect(line.textContent).toMatch(/ODbL 1\.0/);

    // AND NEITHER URL IS ATTACHED to words that do not name them, so a
    // changed source cannot end up pointing at the wrong licence text.
    expect(within(line).queryByRole('link')).toBeNull();
  });
});

describe('copy and badges', () => {
  test('the empty state is the NFL mechanism, not basketball wording', async () => {
    getSchedule.mockResolvedValue([NOT_YET]);
    renderAt('/predictions/american-football/nfl');

    await screen.findByRole('heading', { name: 'NFL Predictions', level: 1 });

    // Fixtures ARE cached, so this is the "nothing predictable yet" state and
    // the week groups render - the offseason copy belongs to the other state.
    expect(await screen.findByRole('heading', { name: 'Week 5' })).toBeInTheDocument();
  });

  test('the offseason copy claims no date and no basketball mechanism', () => {
    const nfl = leagueBySlug('nfl');
    const withNothing = leagueCopy(nfl.seasonNote, { schedule: [] }).join(' ');

    expect(withNothing).toMatch(/no date is shown/i);
    // No "ten games": the NFL has no rolling-window warm-up.
    expect(withNothing).not.toMatch(/ten games/i);
    // No injury-report wording: that is an NBA feature.
    expect(withNothing).not.toMatch(/injury/i);
    // No invented date.
    expect(withNothing).not.toMatch(/\d{4}-\d{2}-\d{2}/);

    const withFixtures = leagueCopy(nfl.seasonNote, {
      schedule: [PREDICTABLE],
    }).join(' ');
    expect(withFixtures).toMatch(/both teams have played their previous game/i);
    expect(withFixtures).not.toMatch(/ten games/i);
  });

  test('the NFL data-as-of is its own, never that of another league', async () => {
    getSchedule.mockResolvedValue([NOT_YET]);
    renderAt('/predictions/american-football/nfl');

    await screen.findByRole('heading', { name: 'NFL Predictions', level: 1 });
    const page = document.body.textContent;

    // Its own cutoff, and neither of the three basketball ones.
    expect(page).toMatch(/2026-10-08/);
    expect(page).not.toMatch(/2026-04-12/);
    expect(page).not.toMatch(/2026-09-24/);
    expect(page).not.toMatch(/2026-03-28/);
  });

  test('NFL abbreviations collide with other leagues, and lookups are by id', () => {
    const byLeague = {};
    Object.values(TEAMS).forEach((team) => {
      byLeague[team.abbr] = (byLeague[team.abbr] || 0) + 1;
    });

    // DERIVED FROM THE DATA, NOT LISTED. teams.js holds no NFL franchises -
    // no colours are invented for them, as for the G League - so the NFL's
    // abbreviations arrive from the API and the collision that matters is
    // between those and the table's. Measured here rather than asserted.
    const nflAbbrs = ['ATL', 'CHI', 'DAL', 'GB', 'MIA', 'NYJ', 'CIN', 'CLE'];
    const collided = nflAbbrs.filter((abbr) => byLeague[abbr]);
    expect(collided.length).toBeGreaterThan(0);

    // Nothing resolves a team by abbreviation anywhere in the app, which is
    // why a collision is harmless. The ids are disjoint by range.
    expect(
      Object.values(TEAMS).some((team) => String(team.id).startsWith('1613'))
    ).toBe(false);
  });
});
