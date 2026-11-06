import fs from 'fs';
import path from 'path';

import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';

import App from './App';
import { createPredictionFor, getHealth, getSchedule } from './api';
import { LEAGUES, leagueBySlug } from './data/leagues';
import { SPORTS } from './components/SportsRail';
import { isPredictable } from './dates';
import { scheduleReason, GENERIC_NOT_PREDICTABLE } from './refusalText';

jest.mock('./api');

// THE CAVEAT LIVES ON /health AND THE SHORT ONE ON THE PREDICTION. Both are
// the real strings the manifest ships, because the page must render what the
// API says and not a copy - so a test that invented them would pass against
// a frontend that had hardcoded its own.
const SHORT_CAVEAT =
  'Weak signal - team ratings only, and barely better than guessing across '
  + 'our test seasons.';
const READER_CAVEAT =
  'These NHL win probabilities come from team ratings alone. Across the two '
  + 'seasons we held back for testing they beat a simple base-rate guess '
  + 'only slightly, and in one of those seasons not at all - so treat them '
  + 'as a weak signal rather than a forecast.';

const HEALTH = {
  dataAsOf: '2026-04-12',
  daysBehind: 208,
  stale: true,
  wnba: { dataAsOf: '2026-09-24', daysBehind: 43, stale: true },
  gleague: { dataAsOf: '2026-03-28', daysBehind: 223, stale: true },
  nfl: { dataAsOf: '2026-11-02', daysBehind: 4, stale: false },
  nhl: {
    served: true,
    dataAsOf: '2026-11-05',
    daysBehind: 1,
    stale: false,
    // DELIBERATELY DIFFERENT FROM WHAT THE ROWS BELOW SAY. /health is a live
    // pass-through to inference while the rows come from the backend's
    // six-hourly cache, and on 6 November they really did disagree: 2
    // against 0. The page must take its count from the ROWS, so this number
    // must never appear.
    predictableFixtures: 97,
    fixtures: 1083,
    predictionRule: "both teams' previous games in history",
    source: 'English Wikipedia, CC BY-SA 4.0',
    winnerCaveat: {
      shortText: SHORT_CAVEAT,
      reader: READER_CAVEAT,
      confirmedOnTest: false,
    },
  },
};

const PREDICTABLE = {
  league: 'nhl',
  leagueSlug: 'nhl',
  homeTeamId: 1614000014,
  homeTeamAbbr: 'MIN',
  homeTeamName: 'Minnesota Wild',
  awayTeamId: 1614000023,
  awayTeamAbbr: 'SJS',
  awayTeamName: 'San Jose Sharks',
  gameDate: '2026-11-06',
  predictable: true,
  reasonCode: null,
  blockedBy: [],
};

// BOTH SIDES WAITING, which is the common case: 1,074 of 1,083 on
// 6 November. A fixture that only ever blocked on one side would not notice
// a page that renders the first blocker and drops the rest.
const BOTH_PENDING = {
  league: 'nhl',
  leagueSlug: 'nhl',
  homeTeamId: 1614000005,
  homeTeamAbbr: 'CAR',
  homeTeamName: 'Carolina Hurricanes',
  awayTeamId: 1614000006,
  awayTeamAbbr: 'CHI',
  awayTeamName: 'Chicago Blackhawks',
  gameDate: '2026-11-07',
  predictable: false,
  reasonCode: 'previous_game_pending',
  reason: 'the home side has 1 earlier fixture(s) with no result yet, the '
    + 'latest on 2026-11-05; its previous game is not in history',
  blockedBy: [
    {
      side: 'home',
      franchiseId: 1614000005,
      code: 'previous_game_pending',
      pending: ['2026-11-05'],
    },
    {
      side: 'away',
      franchiseId: 1614000006,
      code: 'previous_game_pending',
      pending: ['2026-11-05'],
    },
  ],
};

const ONE_PENDING = {
  ...BOTH_PENDING,
  homeTeamId: 1614000030,
  homeTeamAbbr: 'VGK',
  homeTeamName: 'Vegas Golden Knights',
  awayTeamId: 1614000017,
  awayTeamAbbr: 'NJD',
  awayTeamName: 'New Jersey Devils',
  gameDate: '2026-11-08',
  blockedBy: [
    {
      side: 'away',
      franchiseId: 1614000017,
      code: 'previous_game_pending',
      pending: ['2026-11-06'],
    },
  ],
};

const OPENER = {
  league: 'nhl',
  leagueSlug: 'nhl',
  homeTeamId: 1614000019,
  homeTeamAbbr: 'MTL',
  homeTeamName: 'Montreal Canadiens',
  awayTeamId: 1614000033,
  awayTeamAbbr: 'UTA',
  awayTeamName: 'Utah Mammoth',
  gameDate: '2026-11-12',
  predictable: false,
  reasonCode: 'season_opener',
  blockedBy: [
    {
      side: 'away',
      franchiseId: 1614000033,
      code: 'season_opener',
      pending: [],
    },
  ],
};

const UNKNOWN_CODE = {
  ...ONE_PENDING,
  gameDate: '2026-11-09',
  reasonCode: 'something_nobody_has_words_for',
  reason: 'a server sentence that must never reach the page',
  blockedBy: [],
};

const SUMMARY = {
  gameId: 1552,
  homeTeamAbbreviation: 'MIN',
  awayTeamAbbreviation: 'SJS',
  gameDate: '2026-11-06',
  prediction: {
    homeWinProbability: 0.6961913660612236,
    homeWinsByTwoOrMore: 0.453166316786782,
    winnerModel: 'elo',
    winnerCaveat: SHORT_CAVEAT,
    puckLineDefinition: 'HOME_MARGIN >= 2',
    totalsServed: false,
    totalsReasonCode: 'no_skill_over_constant',
    totalsDetail:
      'Totals are not served for the NHL: a constant predictor beats the '
      + 'fitted model on test.',
    season: '2026–27',
    predictionRule: "both teams' previous games in history",
    dataAsOf: '2026-11-05',
    stale: false,
    daysBehind: 1,
    source: 'English Wikipedia, CC BY-SA 4.0',
    predictedAt: '2026-11-06T14:39:39.018618806Z',
  },
};

function renderAt(route) {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <App />
    </MemoryRouter>
  );
}

beforeEach(() => {
  jest.clearAllMocks();
  getHealth.mockResolvedValue(HEALTH);
  getSchedule.mockResolvedValue([]);
  createPredictionFor.mockResolvedValue(SUMMARY);
});

describe('the sport and the league are in the navigation', () => {
  test('Ice hockey is in the rail, and is not called hockey', async () => {
    renderAt('/predictions');

    const rail = screen.getByRole('navigation', { name: 'Sports' });
    const link = await within(rail).findByRole('link', { name: /Ice hockey/ });
    expect(link).toHaveAttribute('href', '/predictions/ice-hockey');

    // NOT "Hockey". Outside North America that word means field hockey, and
    // the slug becomes a URL - the same reasoning as "American football".
    expect(within(rail).queryByRole('link', { name: /^Hockey$/ })).toBeNull();
    expect(SPORTS.map((sport) => sport.slug)).toContain('ice-hockey');
  });

  test('the NHL is declared through the same registry as every league', () => {
    const nhl = leagueBySlug('nhl');
    expect(nhl).toBeTruthy();
    expect(nhl.sport).toBe('ice-hockey');
    // THE SERVER DECIDES, like the NFL. A second copy of the dependency rule
    // in JavaScript would drift from the one that refuses the request.
    expect(nhl.rule).toBe('server');
    expect(nhl.attributionRequired).toBe(true);
    // One entry in one list, not a second mechanism.
    expect(LEAGUES.filter((league) => league.slug === 'nhl')).toHaveLength(1);
  });

  test('predictability is read off the flag, never computed from a date', () => {
    const nhl = leagueBySlug('nhl');
    expect(isPredictable(PREDICTABLE, HEALTH, nhl)).toBe(true);
    expect(isPredictable(BOTH_PENDING, HEALTH, nhl)).toBe(false);
    // The NHL's cutoff is 2026-11-05 and this fixture is the 7th, so a date
    // rule would have offered it. The flag says no.
    expect(BOTH_PENDING.gameDate > HEALTH.nhl.dataAsOf).toBe(true);
  });
});

describe('the sentence on a not-yet-predictable row', () => {
  test('both teams are named when both are waiting', () => {
    const sentence = scheduleReason(BOTH_PENDING);
    expect(sentence).toContain('Carolina Hurricanes');
    expect(sentence).toContain('Chicago Blackhawks');
    expect(sentence).toContain('5 Nov');
    // NOT THE API'S PROSE. The row carries `reason` as well and it must not
    // be what the page shows.
    expect(sentence).not.toContain('earlier fixture(s)');
  });

  test('one team is named when only one is waiting', () => {
    const sentence = scheduleReason(ONE_PENDING);
    expect(sentence).toBe('Waiting for New Jersey Devils’s result from 6 Nov.');
    expect(sentence).not.toContain('Vegas');
  });

  test('a season opener says there is no history, and names no date', () => {
    const sentence = scheduleReason(OPENER);
    expect(sentence).toContain('First game of the season');
    expect(sentence).toContain('Utah Mammoth');
    // ITS `pending` IS EMPTY, so there is nothing to wait for and the
    // sentence must not invent a date.
    expect(sentence).not.toMatch(/\d+ \w{3}/);
  });

  test('an unknown code gets the generic sentence and never the prose', () => {
    const sentence = scheduleReason(UNKNOWN_CODE);
    expect(sentence).toBe(GENERIC_NOT_PREDICTABLE);
    expect(sentence).not.toContain('a server sentence');
  });

  test('a predictable row has no sentence at all', () => {
    expect(scheduleReason(PREDICTABLE)).toBeNull();
  });
});

describe('the schedule page', () => {
  test('the count comes from the rows, not from /health', async () => {
    getSchedule.mockResolvedValue([PREDICTABLE, BOTH_PENDING, OPENER]);
    renderAt('/predictions/ice-hockey/nhl');

    expect(
      await screen.findByText(/upcoming games can be predicted today/)
    ).toBeInTheDocument();
    // ONE PREDICTABLE OF THREE ROWS. /health says 97 of 1,083, which is the
    // live inference figure against a six-hourly cache - and the page must
    // never show a number its own list contradicts.
    const count = screen.getByText(/can be predicted today/);
    expect(count).toHaveTextContent('1 of 3');
    expect(count).not.toHaveTextContent('97');
    expect(count).not.toHaveTextContent('1,083');
  });

  test('the explainer states the rule once, above the list', async () => {
    getSchedule.mockResolvedValue([PREDICTABLE, BOTH_PENDING]);
    renderAt('/predictions/ice-hockey/nhl');

    const rule = await screen.findAllByText(
      /can be predicted once both teams/i
    );
    expect(rule).toHaveLength(1);
  });

  test('a predictable fixture links to its game page', async () => {
    getSchedule.mockResolvedValue([PREDICTABLE, BOTH_PENDING]);
    renderAt('/predictions/ice-hockey/nhl');

    // ONE LINK FOR TWO ROWS: the predictable fixture offers "More on this"
    // and the blocked one offers nothing, which is the whole distinction.
    const links = await screen.findAllByRole('link', { name: 'More on this' });
    expect(links).toHaveLength(1);
    expect(links[0]).toHaveAttribute(
      'href', '/predictions/ice-hockey/nhl/1552'
    );
  });

  test('a not-yet-predictable fixture is shown, with its reason and no link',
    async () => {
      getSchedule.mockResolvedValue([PREDICTABLE, BOTH_PENDING]);
      renderAt('/predictions/ice-hockey/nhl');

      expect(await screen.findByText(/Not yet predictable/)).toBeInTheDocument();
      expect(
        screen.getByText(/Waiting for Carolina Hurricanes and Chicago/)
      ).toBeInTheDocument();
      // THE BLOCKED ROW OFFERS NO LINK AT ALL, which is how a reader knows
      // it is not a dead end but a "not yet".
      const links = screen.getAllByRole('link', { name: 'More on this' });
      expect(links).toHaveLength(1);
    });

  test('zero predictable is its own message, not an empty list', async () => {
    getSchedule.mockResolvedValue([BOTH_PENDING, ONE_PENDING, OPENER]);
    renderAt('/predictions/ice-hockey/nhl');

    expect(
      await screen.findByText(/None of these games can be predicted yet/)
    ).toBeInTheDocument();
    // THE ROWS ARE STILL THERE. A reader looking at a night with a full
    // slate must see the slate, not an apparently broken page. Matched on
    // the team-name cell specifically: the name also appears inside the
    // waiting sentence, and a bare text query finds both - which is the
    // sentence working, not two rows.
    expect(
      screen.getAllByText('Carolina Hurricanes', {
        selector: '.game-row-team-name',
      })
    ).toHaveLength(1);
    const count = screen.getByText(/can be predicted today/);
    expect(count).toHaveTextContent('0 of 3');
  });

  test('the attribution is on the schedule page, with the licence link',
    async () => {
      getSchedule.mockResolvedValue([PREDICTABLE, BOTH_PENDING]);
      renderAt('/predictions/ice-hockey/nhl');

      const licence = await screen.findByRole('link', {
        name: /CC BY-SA 4\.0/,
      });
      expect(licence).toHaveAttribute(
        'href',
        'https://creativecommons.org/licenses/by-sa/4.0/'
      );
      expect(
        screen.getByRole('link', { name: /Wikipedia/ })
      ).toHaveAttribute('href', 'https://en.wikipedia.org/');
    });
});

describe('the game page', () => {
  async function openGamePage() {
    getSchedule.mockResolvedValue([PREDICTABLE]);
    renderAt('/predictions/ice-hockey/nhl');
    const link = await screen.findByRole('link', { name: 'More on this' });
    await userEvent.click(link);
  }

  test('the winner shows both sides with the short caveat beside it',
    async () => {
      await openGamePage();

      expect(await screen.findByText(/MIN 69\.6%/)).toBeInTheDocument();
      expect(screen.getByText(/SJS 30\.4%/)).toBeInTheDocument();
      // IMMEDIATELY BESIDE THE NUMBER, not in a footer. A reader who sees
      // 70% and scrolls past a footnote has been told the wrong thing.
      expect(screen.getByText(SHORT_CAVEAT)).toBeInTheDocument();
    });

  test('the reader caveat comes from the API and is one click away',
    async () => {
      await openGamePage();

      // NOT ON THE PAGE UNTIL ASKED FOR, and then the API's exact words.
      expect(screen.queryByText(READER_CAVEAT)).toBeNull();
      const toggle = await screen.findByRole('button', {
        name: /What this means/,
      });
      await userEvent.click(toggle);
      expect(screen.getByText(READER_CAVEAT)).toBeInTheDocument();
    });

  test('the puck line is labelled from its definition, with no away side',
    async () => {
      await openGamePage();

      expect(
        await screen.findByText(/Minnesota Wild wins by 2 or more goals/)
      ).toBeInTheDocument();
      expect(screen.getByText(/45\.3%/)).toBeInTheDocument();
      // NO +/-1.5 WORDING ANYWHERE, and no away-side line: the model does
      // not predict one, and 1 - p would assert a market that is not served.
      expect(screen.queryByText(/1\.5/)).toBeNull();
      expect(screen.queryByText(/San Jose Sharks wins by/)).toBeNull();
    });

  test('totals say they are not offered, and why', async () => {
    await openGamePage();

    expect(await screen.findByText(/Not offered/)).toBeInTheDocument();
    expect(
      screen.getByText(/a constant predictor beats the fitted model on test/)
    ).toBeInTheDocument();
  });

  test('the footnote carries the cutoff, the season and the rule',
    async () => {
      await openGamePage();

      const footnote = await screen.findByText(/Computed from data as of/);
      expect(footnote).toHaveTextContent('2026-11-05');
      // THE EN DASH SURVIVES TO THE BROWSER: 2026-27 with U+2013, not a
      // hyphen and not a mojibake.
      expect(footnote).toHaveTextContent('2026–27');
      expect(footnote).not.toHaveTextContent('2026-27 season');
      expect(footnote).toHaveTextContent(
        /both teams. previous games in history/
      );
    });

  test('the attribution is on the game page too', async () => {
    await openGamePage();

    const licence = await screen.findByRole('link', { name: /CC BY-SA 4\.0/ });
    expect(licence).toHaveAttribute(
      'href',
      'https://creativecommons.org/licenses/by-sa/4.0/'
    );
  });
});

describe('every other league is untouched', () => {
  test('the four existing leagues keep their sports and rules', () => {
    expect(leagueBySlug('nba').sport).toBe('basketball');
    expect(leagueBySlug('wnba').sport).toBe('basketball');
    expect(leagueBySlug('gleague').sport).toBe('basketball');
    expect(leagueBySlug('nfl').sport).toBe('american-football');

    expect(leagueBySlug('nba').rule).toBe('date');
    expect(leagueBySlug('nfl').rule).toBe('server');

    // THE NHL IS THE ONLY LEAGUE SHOWING A DATE SLATE OF UNPREDICTABLE
    // FIXTURES, and the NFL the only weekly one - so neither flag leaked.
    expect(
      LEAGUES.filter((league) => league.showsUnpredictable).map((l) => l.slug)
    ).toEqual(['nhl']);
    expect(
      LEAGUES.filter((league) => league.weeklySlate).map((l) => l.slug)
    ).toEqual(['nfl']);
  });

  test('only the NHL requires the winner caveat', () => {
    expect(
      LEAGUES.filter((league) => league.winnerCaveatRequired)
        .map((league) => league.slug)
    ).toEqual(['nhl']);
  });
});

describe('the styling has something to style', () => {
  // THE ONE CSS CHECK jsdom CAN HONESTLY MAKE. It applies no layout, so
  // "does this look right at 390 px" is not answerable here - that is the
  // owner's browser check. What IS answerable, and is the real risk when a
  // component and its stylesheet are written together, is whether the class
  // names agree: a selector in the CSS that matches nothing renders an
  // unstyled element and nothing fails anywhere.
  function classesIn(cssPath) {
    // COMMENTS STRIPPED FIRST. These files explain themselves at length and
    // name other stylesheets by filename, so a bare class-selector regex
    // picked up "css" from "GameDetailPage.css" and reported it as a
    // selector matching nothing - the harness finding its own prose.
    const css = fs
      .readFileSync(path.join(__dirname, cssPath), 'utf8')
      .replace(/\/\*[\s\S]*?\*\//g, ' ');
    // Class selectors only, and not the ones inside a media query's
    // condition. Deduplicated, because a class styled twice is one class.
    return Array.from(new Set(css.match(/\.[a-z][a-z0-9-]*/gi) || []))
      .map((name) => name.slice(1))
      // `.market-*` belongs to GameDetailPage.css and is shared with the
      // other leagues' tabs; this file only reuses it.
      .filter((name) => !name.startsWith('market-'));
  }

  test('every class in NhlGameTab.css is rendered by the tab', async () => {
    getSchedule.mockResolvedValue([PREDICTABLE]);
    const { container } = renderAt('/predictions/ice-hockey/nhl');
    const link = await screen.findByRole('link', { name: 'More on this' });
    await userEvent.click(link);
    // The reader caveat is behind a toggle, so its class is not in the DOM
    // until it is opened - open it, or the check would call a live selector
    // dead.
    await userEvent.click(
      await screen.findByRole('button', { name: /What this means/ })
    );

    const styled = classesIn('components/NhlGameTab.css');
    expect(styled.length).toBeGreaterThan(4);
    const missing = styled.filter(
      (name) => container.getElementsByClassName(name).length === 0
    );
    expect(missing).toEqual([]);
  });

  test('every class in DateSlate.css is rendered by the slate', async () => {
    // BOTH BRANCHES IN ONE PASS: a slate with nothing predictable renders
    // `date-slate-none-yet`, and one with a fixture renders the rest. A
    // single render would leave one selector looking dead.
    getSchedule.mockResolvedValue([BOTH_PENDING]);
    const none = renderAt('/predictions/ice-hockey/nhl');
    await screen.findByText(/None of these games can be predicted yet/);

    getSchedule.mockResolvedValue([]);
    const empty = renderAt('/predictions/ice-hockey/nhl');
    await screen.findByText(/No upcoming fixtures are cached/);

    const styled = classesIn('components/DateSlate.css');
    expect(styled.length).toBeGreaterThan(3);
    const missing = styled.filter(
      (name) => none.container.getElementsByClassName(name).length === 0
        && empty.container.getElementsByClassName(name).length === 0
    );
    expect(missing).toEqual([]);
  });

  test('both widths render the same elements, so neither is a dead branch',
    async () => {
      // THE BREAKPOINT IS 480 px AND BOTH SIDES OF IT MUST HAVE CONTENT.
      // jsdom will not compute the media query, but a rule written for a
      // width whose markup does not exist is still a rule styling nothing -
      // and this catches the case where a phone-only element was forgotten.
      for (const width of [1440, 390]) {
        window.innerWidth = width;
        getSchedule.mockResolvedValue([PREDICTABLE, BOTH_PENDING]);
        const { container, unmount } = renderAt('/predictions/ice-hockey/nhl');
        await screen.findByText(/can be predicted today/);

        expect(
          container.getElementsByClassName('date-slate-explainer')
        ).toHaveLength(1);
        expect(
          container.getElementsByClassName('game-row').length
        ).toBeGreaterThan(0);
        unmount();
      }
    });
});
