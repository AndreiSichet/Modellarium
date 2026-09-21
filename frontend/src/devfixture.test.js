import { devPlayerPropsFor, devPredictionFor, devQuarterHalfFor, DEV_HEALTH, DEV_SCHEDULE } from './data/devFixtures';
import { PLAYER_STATS } from './components/DetailTabs';

test('dev fixtures match the fields the components read', () => {
  const qh = devQuarterHalfFor('2026-04-13').prediction;
  for (const f of ['q1Spread','q1Total','q1WinnerProbability','q1WinnerConfidence','q1WinnerInterpretation',
                   'half1Spread','half1Total','half1WinnerProbability','half1WinnerConfidence','half1WinnerInterpretation']) {
    expect(qh[f]).toBeDefined();
  }
  expect(qh.q1WinnerConfidence).toBe('low');
  expect(qh.q1WinnerInterpretation).toBe('P(home leads | not tied)');

  const props = devPlayerPropsFor('2026-04-13');
  for (const side of ['homeTeam','awayTeam']) {
    expect(props[side].availabilityKnown).toBe(false);
    expect(props[side].availabilityNote).toMatch(/AVAILABILITY UNKNOWN/);
    expect(props[side].players.length).toBeGreaterThan(0);
    for (const p of props[side].players) {
      expect(['linear','xgb']).toContain(p.modelUsed);
      for (const stat of PLAYER_STATS) expect(typeof p[stat.field]).toBe('number');
    }
  }

  const models = new Set(props.homeTeam.players.concat(props.awayTeam.players).map((p) => p.modelUsed));
  expect(models).toEqual(new Set(['linear','xgb']));

  const summary = devPredictionFor(DEV_SCHEDULE[0], 0);
  expect(summary.latestPrediction.dataAsOf).toBe(DEV_HEALTH.dataAsOf);
  expect(typeof summary.id).toBe('number');
});
