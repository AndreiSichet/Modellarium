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

  prediction: {
    homeWinProbability: 0.5745,
    homeMargin: 1.4149,
    totalPoints: 232.9323,
  },
};

function renderRow(game = GAME, league = LEAGUES[0]) {
  return render(
    <MemoryRouter>
      <ul>
        <GameRow game={game} league={league} />
      </ul>
    </MemoryRouter>
  );
}

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

test('the total renders exactly once per row', () => {
  const { container } = renderRow();
  expect(screen.getAllByText('232.9')).toHaveLength(1);
  expect(container.querySelectorAll('.game-row-total')).toHaveLength(1);
});

test('a long team name is present in full in the DOM', () => {
  renderRow();

  expect(screen.getByText('Los Angeles Lakers')).toBeInTheDocument();
});

describe('team badges', () => {
  test('a known team renders its abbreviation', () => {
    render(<TeamBadge teamId={1610612738} />);
    expect(screen.getByText('BOS')).toBeInTheDocument();
  });

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

  test('an illegible secondary falls back, a legible one is kept', () => {
    expect(badgeTextColour(teamFor(1610612738))).toBe('#FFFFFF');

    expect(badgeTextColour(teamFor(1610612743))).toBe('#FEC524');
  });
});

test('More on this links at the game detail route', () => {
  renderRow();

  expect(screen.getByRole('link', { name: 'More on this' })).toHaveAttribute(
    'href',
    '/predictions/basketball/nba/4'
  );
});

test('the footer shows the date, since no time exists in the API', () => {
  const { container } = renderRow();
  const footer = container.querySelector('.game-row-footer');
  expect(within(footer).getByText('2026-04-13')).toBeInTheDocument();
});
