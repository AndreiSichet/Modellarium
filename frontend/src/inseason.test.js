import { render, screen } from '@testing-library/react';

import PlayerStatTab from './components/PlayerStatTab';
import { PLAYER_STATS } from './components/DetailTabs';
import {
  DEV_HEALTH,
  DEV_HEALTH_IN_SEASON,
  devPlayerPropsFor,
  devPlayerPropsInSeasonFor,
} from './data/devFixtures';

/**
 * The in-season shapes, captured 2026-10-21 - the day after the NBA season
 * opened. They sit beside the offseason suites rather than replacing them.
 *
 * Every frontend fixture before these was taken with a stale cutoff and no
 * injury report, so two things had never been through any frontend suite: a
 * cutoff that is NOT stale, and a player board that arrives EMPTY because no
 * player has usable history yet.
 */

const GAME = { homeTeamId: 1610612753, awayTeamId: 1610612737 };
const STAT = PLAYER_STATS[0];

function renderBoard(board) {
  return render(
    <PlayerStatTab
      stat={STAT}
      state={{ status: 'ready', data: board }}
      onRetry={() => {}}
      game={GAME}
    />
  );
}

test('the in-season health fixture is the not-stale branch the offseason one cannot reach', () => {
  expect(DEV_HEALTH_IN_SEASON.stale).toBe(false);
  expect(DEV_HEALTH_IN_SEASON.daysBehind).toBeGreaterThan(0);

  // Not a pinned number: the point is that it is nothing like the offseason
  // fixture's, which is in the hundreds. Pinning 1 would pin the capture day.
  expect(DEV_HEALTH_IN_SEASON.daysBehind).toBeLessThan(DEV_HEALTH.daysBehind);

  // The offseason fixture is still there and still stale. These coexist.
  expect(DEV_HEALTH.stale).toBe(true);

  // The WNBA is judged on ITS OWN cutoff, which is the reason the block
  // exists. In season the two leagues disagree, and this is the first
  // fixture where they do: the NBA is current while the WNBA is not.
  expect(DEV_HEALTH_IN_SEASON.wnba.stale).toBe(true);
  expect(DEV_HEALTH_IN_SEASON.wnba.dataAsOf).not.toBe(DEV_HEALTH_IN_SEASON.dataAsOf);
});

test('the in-season player board is empty on both sides and says why', () => {
  const props = devPlayerPropsInSeasonFor('2026-10-21');

  for (const side of ['homeTeam', 'awayTeam']) {
    expect(props[side].players).toHaveLength(0);
    expect(props[side].availabilityKnown).toBe(false);
    expect(props[side].availabilityNote).toBe('no players with usable history');

    // A DIFFERENT unknown from the offseason one, which reads
    // "AVAILABILITY UNKNOWN - no injury report was available". Neither is a
    // zero, and the suite had only ever seen the first.
    expect(props[side].availabilityNote).not.toMatch(/AVAILABILITY UNKNOWN/);
  }

  // And the offseason fixture still carries the other sentence.
  const offseason = devPlayerPropsFor('2026-04-13');
  expect(offseason.homeTeam.availabilityNote).toMatch(/AVAILABILITY UNKNOWN/);
  expect(offseason.homeTeam.players.length).toBeGreaterThan(0);
});

test('an empty in-season board renders its note and no player rows', () => {
  renderBoard(devPlayerPropsInSeasonFor('2026-10-21'));

  // Both boards still render - the teams are present, only their rosters are
  // empty - so the user gets two headed panels with an explanation and no
  // rows, which is what the service actually supports saying today.
  // By heading rather than by text: the abbreviation also appears inside the
  // team badge, so a plain text query matches twice per board and says
  // nothing about which element carried it.
  const titles = screen.getAllByRole('heading', { level: 2 }).map((h) => h.textContent);
  expect(titles).toEqual(['ATL', 'ORL']);

  expect(screen.getAllByText('no players with usable history')).toHaveLength(2);
  expect(screen.queryAllByRole('listitem')).toHaveLength(0);
});

test('a populated board still renders its rows, so the empty case is not the only one', () => {
  // The control. Without it, an empty render above would also pass if the
  // component had stopped rendering player rows altogether.
  renderBoard(devPlayerPropsFor('2026-04-13'));

  expect(screen.queryAllByRole('listitem').length).toBeGreaterThan(0);
  expect(screen.getByText('Jayson Tatum')).toBeInTheDocument();
});
