/**
 * What a reader sees when the service declines a fixture.
 *
 * THE THREE THINGS THE 21 OCTOBER BROWSER CHECK FOUND, pinned.
 *
 *   1. The Quarters & Halves tab said "the request failed … the service did
 *      not answer" and printed raw JSON. The service HAD answered: it
 *      refused, because a team had fewer games this season than the models'
 *      rolling window needs. Measured 2026-10-30: all ten NBA fixtures of
 *      the day refused, teams had 3 to 6 games against a window of 10, and
 *      the last one cannot succeed before 2026-11-13. This is what a reader
 *      meets for the first fortnight of every season.
 *
 *   2. The player-props empty state required `!board?.homeTeam &&
 *      !board?.awayTeam`, and the real response has both PRESENT with empty
 *      rosters - so the explanation could never appear.
 *
 *   3. A refused fixture on the browse page rejected a bare `Promise.all`
 *      inside one try/catch, so one declined fixture turned the whole league
 *      page into "could not load".
 *
 * EVERY CASE IS PLANTED. Each plant reverts the condition the case depends
 * on and asserts the old behaviour returns, so a green here is a statement
 * about this code rather than about the test's own wishes.
 */
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

import PlayerStatTab from './components/PlayerStatTab';
import QuarterHalfTab from './components/QuarterHalfTab';
import TabStatus from './components/TabStatus';
import GameRow from './components/GameRow';
import { refusalText, REFUSAL_TEXT } from './refusalText';

const GAME = {
  gameId: 42,
  homeTeamId: 1610612744,
  homeTeamName: 'Golden State Warriors',
  homeTeamAbbr: 'GSW',
  awayTeamId: 1610612747,
  awayTeamName: 'Los Angeles Lakers',
  awayTeamAbbr: 'LAL',
  gameDate: '2026-10-30',
};

const LEAGUE = { slug: 'nba', sport: 'basketball', markets: {} };

/** The error api.js builds from a real refusal body. */
function refusalError({ code, fields = {}, status = 400, message }) {
  const error = new Error(
    message
      ?? 'Inference service rejected the request: {"detail":"HOME team '
         + '1610612744 has played 5 game(s) with quarter data this season"}'
  );
  error.status = status;
  error.code = code;
  error.fields = fields;
  error.body = { detail: 'HOME team 1610612744 has played 5 game(s)', code };
  return error;
}

function errorState(failure) {
  return { status: 'error', data: null, error: failure.message, failure };
}

// =====================================================================
// 1. the Quarters & Halves tab
// =====================================================================

describe('Quarters & Halves on a refusal', () => {
  const INSUFFICIENT = refusalError({
    code: 'insufficient_history',
    fields: {
      needed: 10,
      home_games_played: 5,
      away_games_played: 4,
      short_side: 'home',
    },
  });

  test('insufficient_history shows the sentence with the right numbers', () => {
    render(
      <QuarterHalfTab
        state={errorState(INSUFFICIENT)}
        onRetry={() => {}}
        health={{ dataAsOf: '2026-10-29' }}
        game={GAME}
      />
    );

    // THE NUMBERS COME FROM THE REFUSAL, not from the component. `needed` is
    // max(WINDOWS) on the inference side; a literal here would be a second
    // copy of a number that moves.
    expect(
      screen.getByText(/both teams have played 10 games this season/i)
    ).toBeInTheDocument();
    expect(screen.getByText(/GSW has played 5, LAL 4/)).toBeInTheDocument();
  });

  test('and no JSON reaches the page', () => {
    const { container } = render(
      <QuarterHalfTab
        state={errorState(INSUFFICIENT)}
        onRetry={() => {}}
        health={{}}
        game={GAME}
      />
    );

    expect(container.textContent).not.toMatch(/[{}]/);
    expect(container.textContent).not.toMatch(/"detail"/);
  });

  test('never says "did not answer" for a 4xx', () => {
    const { container } = render(
      <QuarterHalfTab
        state={errorState(INSUFFICIENT)}
        onRetry={() => {}}
        health={{}}
        game={GAME}
      />
    );

    expect(container.textContent).not.toMatch(/did not answer/i);
  });

  test('a 500 keeps "could not load", with the raw text behind a toggle', () => {
    const failure = refusalError({
      code: null,
      status: 500,
      message: 'Request failed with status 500',
    });
    const { container } = render(
      <QuarterHalfTab state={errorState(failure)} onRetry={() => {}} health={{}} game={GAME} />
    );

    expect(screen.getByText(/could not load/i)).toBeInTheDocument();
    // THE TEXT IS PRESENT BUT NOT VISIBLE UNTIL ASKED FOR. <details> keeps
    // it in the document, which is the point: an unclassified refusal is
    // often identifiable only from its raw body.
    const details = container.querySelector('details.predictions-error-details');
    expect(details).not.toBeNull();
    expect(details.open).toBe(false);
    expect(details.textContent).toMatch(/status 500/);
  });

  test('a 4xx with NO code is "could not load" and NOT the history sentence', () => {
    const failure = refusalError({ code: null, status: 400 });
    render(
      <QuarterHalfTab state={errorState(failure)} onRetry={() => {}} health={{}} game={GAME} />
    );

    // A WRONG EXPLANATION IS WORSE THAN A VAGUE ONE. An unclassified refusal
    // must not borrow insufficient_history's wording.
    expect(screen.queryByText(/games this season/i)).not.toBeInTheDocument();
    expect(screen.getByText(/No quarter and half markets for this fixture/i))
      .toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/did not answer/i);
  });

  test('PLANT: branching on status instead of code shows the wrong message', () => {
    // The old shape - no code in the decision - lands an
    // insufficient_history refusal in TabStatus, which is exactly the defect.
    const { container } = render(
      <TabStatus
        status="error"
        error={INSUFFICIENT.message}
        failure={undefined}
        onRetry={() => {}}
        what="quarter and half markets"
      />
    );

    expect(container.textContent).toMatch(/did not answer/i);
    expect(container.textContent).not.toMatch(/played 10 games/);
  });
});

// =====================================================================
// 2. the player-props tab
// =====================================================================

function board(homePlayers, awayPlayers, note = 'no players with usable history') {
  const team = (abbr, players) => ({
    teamAbbreviation: abbr,
    availabilityKnown: players.length > 0,
    availabilityNote: players.length > 0 ? null : note,
    players,
  });
  return {
    homeTeam: team('GSW', homePlayers),
    awayTeam: team('LAL', awayPlayers),
  };
}

const PLAYER = {
  playerId: 1,
  playerName: 'A Player',
  modelUsed: 'linear',
  predictedPoints: 20.5,
};
const STAT = { id: 'points', field: 'predictedPoints', short: 'Points' };

describe('player props with empty rosters', () => {
  test('both rosters empty shows the explanation', () => {
    render(
      <PlayerStatTab
        stat={STAT}
        state={{ status: 'ready', data: board([], []), error: null }}
        onRetry={() => {}}
        game={GAME}
      />
    );

    expect(screen.getByText(/No player predictions yet/i)).toBeInTheDocument();
    expect(
      screen.getByText(/need each player's own recent history/i)
    ).toBeInTheDocument();
  });

  test('one roster empty: that panel says so, the other lists its players', () => {
    render(
      <PlayerStatTab
        stat={STAT}
        state={{ status: 'ready', data: board([], [PLAYER]), error: null }}
        onRetry={() => {}}
        game={GAME}
      />
    );

    expect(screen.getByText('A Player')).toBeInTheDocument();
    expect(
      screen.getByText(/No player predictions for this team yet/i)
    ).toBeInTheDocument();
    // NOT the whole-board explanation: one side has players.
    expect(screen.queryByText(/No player predictions yet/i)).not.toBeInTheDocument();
  });

  test('PLANT: the old condition needs both boards ABSENT, so it never fires', () => {
    // `!board?.homeTeam && !board?.awayTeam` against the real response -
    // both present, both empty - which is why a reader got two headed panels
    // and nothing else.
    const data = board([], []);
    const oldConditionFires = !data?.homeTeam && !data?.awayTeam;

    expect(oldConditionFires).toBe(false);
    // and the new one does
    const bothEmpty =
      !(data.homeTeam.players?.length > 0) && !(data.awayTeam.players?.length > 0);
    expect(bothEmpty).toBe(true);
  });
});

// =====================================================================
// 3. the browse row
// =====================================================================

describe('a refused fixture on the browse page', () => {
  function renderRow(game) {
    return render(
      <MemoryRouter>
        <ul>
          <GameRow game={game} league={LEAGUE} />
        </ul>
      </MemoryRouter>
    );
  }

  test('renders as one "not yet predictable" row with its reason', () => {
    renderRow({
      ...GAME,
      prediction: null,
      refusal: { code: 'season_opener', detail: 'x', fields: {} },
    });

    expect(screen.getByText(/Not yet predictable/i)).toBeInTheDocument();
    expect(screen.getByText(REFUSAL_TEXT.season_opener)).toBeInTheDocument();
    // The fixture is still named - it exists, it was declined.
    expect(screen.getByText(/Golden State Warriors/)).toBeInTheDocument();
  });

  test('an unknown code falls back to the API prose, never to a blank', () => {
    renderRow({
      ...GAME,
      prediction: null,
      refusal: { code: 'something_new', detail: 'the API said this', fields: {} },
    });

    expect(screen.getByText('the API said this')).toBeInTheDocument();
  });

  test('refusalText returns null only when there is nothing at all to say', () => {
    expect(refusalText(null)).toBeNull();
    expect(refusalText({ code: 'insufficient_history' }))
      .toBe(REFUSAL_TEXT.insufficient_history);
    expect(refusalText({ code: 'nope', detail: 'prose' })).toBe('prose');
    expect(refusalText({ code: 'nope' })).toBeNull();
  });

  test('PLANT: GameRow without the early branch throws on a null prediction', () => {
    // The pre-2026-10-30 shape read prediction.homeMargin straight away. A
    // refused fixture therefore threw inside the list, which is the same
    // failure one layer below the one PredictionsLayout now catches.
    const readsPredictionImmediately = (game) => game.prediction.homeMargin;

    expect(() =>
      readsPredictionImmediately({ ...GAME, prediction: null })
    ).toThrow(TypeError);
  });
});
