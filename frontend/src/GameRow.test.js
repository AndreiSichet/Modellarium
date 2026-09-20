import { render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

import GameRow from './components/GameRow';
import { formatSpread, formatWin } from './predictionFormat';
import TeamBadge from './components/TeamBadge';
import { badgeTextColour, teamFor, UNKNOWN_TEAM } from './data/teams';
import { LEAGUES } from './data/leagues';

const GAME = {
  key: 'k',
  gameId: 4,
  league: 'NBA',
  homeTeamId: 1610612738,
  homeTeamName: 'Boston Celtics',
  awayTeamId: 1610612747,
  awayTeamName: 'Los Angeles Lakers',
  gameDate: '2026-04-13',
  // Home favoured by 1.41 points.
  prediction: {
    homeWinProbability: 0.5745,
    homeMargin: 1.4149,
    totalPoints: 232.9323,
  },
};

/*
 * The league is REQUIRED, not optional, and the row throws without it
 * rather than falling back. Phase 6 made the detail URL
 * /predictions/:sport/:league/:gameId, and a row on the General page can
 * belong to any league — a default would silently send every row to
 * whichever sport happened to be first.
 */
function renderRow(game = GAME, league = LEAGUES[0]) {
  return render(
    <MemoryRouter>
      <ul>
        <GameRow game={game} league={league} />
      </ul>
    </MemoryRouter>
  );
}

/**
 * THE SPREAD SIGN IS THE EASIEST THING IN THIS PHASE TO GET BACKWARDS, and
 * it fails silently because both versions look like plausible numbers.
 *
 * homeMargin is the predicted HOME MARGIN — positive means home wins by
 * that much. Betting convention gives the FAVOURITE a NEGATIVE spread, so
 * the sign flips on the way to the screen.
 */
describe('spread sign', () => {
  test('a favoured home team shows negative, the away team positive', () => {
    expect(formatSpread(1.4149, true)).toBe('-1.4');
    expect(formatSpread(1.4149, false)).toBe('+1.4');
  });

  test('and it reverses when the away team is favoured', () => {
    expect(formatSpread(-8.5507, true)).toBe('+8.6');
    expect(formatSpread(-8.5507, false)).toBe('-8.6');
  });

  test('a margin rounding to zero shows 0.0 on both sides, never -0.0', () => {
    // A 0.04-point edge rounds to nothing at one decimal. Printing "-0.0"
    // reads as a typo, and the sign carries no information at that
    // magnitude, so both sides show a plain 0.0 - a pick'em.
    //
    // An earlier version of this test asserted "-0.0", contradicting the
    // normalisation it was written to cover.
    expect(formatSpread(0.0412, true)).toBe('0.0');
    expect(formatSpread(0.0412, false)).toBe('0.0');
    expect(formatSpread(-0.0412, true)).toBe('0.0');
  });

  test('the rendered row carries both signs, not one', () => {
    renderRow();
    expect(screen.getByText('-1.4')).toBeInTheDocument();
    expect(screen.getByText('+1.4')).toBeInTheDocument();
  });
});

describe('win percentages', () => {
  test('away is the complement of home', () => {
    expect(formatWin(0.5745, true)).toBe('57.5%');
    expect(formatWin(0.5745, false)).toBe('42.5%');
  });

  test('they sum to 100% on the rendered row', () => {
    renderRow();
    const shown = screen
      .getAllByText(/^\d+\.\d%$/)
      .map((node) => parseFloat(node.textContent));
    expect(shown).toHaveLength(2);
    expect(shown[0] + shown[1]).toBeCloseTo(100, 5);
  });
});

/**
 * The total is one number ABOUT THE GAME, not a per-team value. A
 * sportsbook prints O/U twice because it sells two sides of a bet;
 * Modellarium does not, so printing it twice would invent a symmetry.
 */
test('the total renders exactly once per row', () => {
  const { container } = renderRow();
  expect(screen.getAllByText('232.9')).toHaveLength(1);
  expect(container.querySelectorAll('.game-row-total')).toHaveLength(1);
});

test('a long team name is present in full in the DOM', () => {
  renderRow();
  // Truncation is visual (text-overflow), so the accessible text stays whole.
  expect(screen.getByText('Los Angeles Lakers')).toBeInTheDocument();
});

describe('team badges', () => {
  test('a known team renders its abbreviation', () => {
    render(<TeamBadge teamId={1610612738} />);
    expect(screen.getByText('BOS')).toBeInTheDocument();
  });

  /**
   * The 30-team map is right today and wrong the first time a franchise
   * relocates. A missing key must be a visible oddity, not a blank page.
   */
  test('an unknown id degrades to a neutral badge rather than throwing', () => {
    render(<TeamBadge teamId={1610612999} />);
    expect(screen.getByText('?')).toBeInTheDocument();
    expect(teamFor(1610612999)).toBe(UNKNOWN_TEAM);
  });

  test('a row with an unknown team still renders its predictions', () => {
    renderRow({ ...GAME, homeTeamId: 1610612999, homeTeamName: 'Relocated Franchise' });
    expect(screen.getByText('?')).toBeInTheDocument();
    expect(screen.getByText('-1.4')).toBeInTheDocument();
  });

  /**
   * COMPUTED, NOT HAND-PICKED: 14 of the 30 teams have a secondary colour
   * that fails 3:1 against their own primary, so choosing by eye would have
   * been wrong 14 times and wrong again on every future edit.
   */
  test('an illegible secondary falls back, a legible one is kept', () => {
    // Boston: gold on green, 1.98:1 -> falls back.
    expect(badgeTextColour(teamFor(1610612738))).toBe('#FFFFFF');
    // Denver: gold on navy, comfortably legible -> kept.
    expect(badgeTextColour(teamFor(1610612743))).toBe('#FEC524');
  });
});

test('More on this links at the game detail route', () => {
  renderRow();
  // Built from the league it was handed, not from the current URL: a
  // /game/ segment would also shadow a league slug called "game".
  expect(screen.getByRole('link', { name: 'More on this' })).toHaveAttribute(
    'href',
    '/predictions/basketball/nba/4'
  );
});

/**
 * ScheduledGameDto carries gameDate as a LocalDate, and the inference
 * service's /schedule returns game_date alone — there is no tip-off time
 * anywhere in the chain, so the footer shows the date rather than inventing
 * one.
 */
test('the footer shows the date, since no time exists in the API', () => {
  const { container } = renderRow();
  const footer = container.querySelector('.game-row-footer');
  expect(within(footer).getByText('2026-04-13')).toBeInTheDocument();
});
