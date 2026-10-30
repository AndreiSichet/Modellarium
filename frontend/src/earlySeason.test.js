/**
 * The early-season note appears while it is true, and then stops.
 *
 * WHAT IT IS FOR, MEASURED 2026-10-30. The team models answer every fixture,
 * but `drop_incomplete` removed every early-season row from their training
 * set, so October is a regime they were never fitted on. Of 34 feature
 * columns, 28 were NaN in both fixtures of a tied pair; the totals model put
 * them in the SAME LEAF OF ALL 108 TREES and returned 219.0228271484375
 * twice, while spread and moneyline separated the same pair in 27 and 23
 * trees. Mid-season: 0.0% of totals non-unique across three seasons.
 *
 * SELF-CLEARING, AND THAT IS THE PART WORTH TESTING. No date is written in
 * the component or in the backend. `needed` is max(WINDOWS) from the feature
 * definitions, the counts come from the served tables, and the date comes
 * from the real remaining schedule - so the note removes itself without
 * anybody editing anything.
 */
import { render, screen } from '@testing-library/react';

import EarlySeasonNote from './components/EarlySeasonNote';
import { normalisePredictionBody } from './api';

function renderNote(earlySeason) {
  return render(<EarlySeasonNote earlySeason={earlySeason} />);
}

const AT_FIVE = {
  needed: 10,
  homeGamesPlayed: 5,
  awayGamesPlayed: 5,
  fullWindowDate: '2026-11-07',
};

describe('the note while the season is young', () => {
  test('shows at 5 of 10 games, with the window and the date', () => {
    renderNote(AT_FIVE);

    // THE EXACT WORDING, because it is the thing a reader sees and the one
    // part of this change that is not derivable from anything.
    expect(
      screen.getByText(
        /Early in the season, teams have played fewer than 10 games, so these predictions lean mostly on last season’s ratings and rest days\. Totals can come out close or identical across games\. Both teams reach a full window on 2026-11-07\./
      )
    ).toBeInTheDocument();
  });

  test('the 10 is the API’s `needed`, not a literal in the component', () => {
    // A changed window must move the sentence. If `needed` were written in
    // the component this would still say 10.
    renderNote({ ...AT_FIVE, needed: 17 });

    expect(screen.getByText(/fewer than 17 games/)).toBeInTheDocument();
    expect(screen.queryByText(/fewer than 10 games/)).not.toBeInTheDocument();
  });

  test('the date is whatever the schedule said, not computed here', () => {
    renderNote({ ...AT_FIVE, fullWindowDate: '2026-12-02' });

    expect(
      screen.getByText(/reach a full window on 2026-12-02/)
    ).toBeInTheDocument();
  });

  test('one team short is still early season - a window needs both', () => {
    renderNote({ ...AT_FIVE, homeGamesPlayed: 12, awayGamesPlayed: 6 });

    expect(screen.getByText(/Early in the season/)).toBeInTheDocument();
  });
});

describe('the note disappears by itself', () => {
  test('gone at 10 of 10', () => {
    const { container } = renderNote({
      ...AT_FIVE,
      homeGamesPlayed: 10,
      awayGamesPlayed: 10,
    });

    expect(container).toBeEmptyDOMElement();
  });

  test('gone past the window too', () => {
    const { container } = renderNote({
      ...AT_FIVE,
      homeGamesPlayed: 41,
      awayGamesPlayed: 38,
    });

    expect(container).toBeEmptyDOMElement();
  });

  test('exactly at the window counts as reached, not as short', () => {
    // `>= needed`, not `> needed`. One game either side of the boundary is
    // the whole question, and an off-by-one here would leave the note up for
    // an extra game-day every season.
    const { container } = renderNote({
      ...AT_FIVE,
      homeGamesPlayed: 10,
      awayGamesPlayed: 11,
    });

    expect(container).toBeEmptyDOMElement();
  });
});

describe('nothing to say is said as nothing', () => {
  test('no block at all renders nothing', () => {
    const { container } = renderNote(undefined);
    expect(container).toBeEmptyDOMElement();
  });

  test('a block with no `needed` renders nothing, never a blank sentence', () => {
    const { container } = renderNote({
      homeGamesPlayed: 5,
      awayGamesPlayed: 5,
    });
    expect(container).toBeEmptyDOMElement();
  });

  test('no date: the first two sentences still show, the third is omitted', () => {
    // THE DATE IS THE PART THAT CAN FAIL. Inference returns null for it
    // whenever the schedule cannot be read, and a note reading "reach a full
    // window on null" would be worse than one that stops after two
    // sentences.
    renderNote({ ...AT_FIVE, fullWindowDate: null });

    expect(screen.getByText(/fewer than 10 games/)).toBeInTheDocument();
    expect(screen.queryByText(/reach a full window/)).not.toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/null|undefined/);
  });
});

describe('the block survives the only mapping that matters', () => {
  test('normalisePredictionBody carries earlySeason through', () => {
    const body = {
      id: 7,
      latestPrediction: { homeMargin: 1 },
      earlySeason: AT_FIVE,
    };

    expect(normalisePredictionBody('nba', body).earlySeason).toEqual(AT_FIVE);
  });

  test('and is undefined for a league whose backend sends none', () => {
    const body = { gameId: 9, prediction: { homeMargin: 1 } };

    expect(normalisePredictionBody('wnba', body).earlySeason).toBeUndefined();
  });

  test('PLANT: a normaliser that drops the key means the note never shows', () => {
    // The mapping is the only path from a backend body to the shape
    // components read, so a field it drops does not exist as far as the page
    // is concerned - which is how the NFL's kickoff came to be absent with
    // nothing failing anywhere.
    const dropped = (body) => ({
      gameId: body?.id,
      prediction: body?.latestPrediction,
    });
    const mapped = dropped({ id: 7, latestPrediction: {}, earlySeason: AT_FIVE });

    expect(mapped.earlySeason).toBeUndefined();
    const { container } = renderNote(mapped.earlySeason);
    expect(container).toBeEmptyDOMElement();
  });

  test('PLANT: keying on max() instead of min() hides a half-short fixture', () => {
    // One team at 12 and one at 6 is early season. `max()` would read 12,
    // clear the note, and leave a reader with a prediction built on six
    // games and no indication of it.
    const home = 12;
    const away = 6;
    const needed = 10;

    expect(Math.max(home, away) >= needed).toBe(true);
    expect(Math.min(home, away) >= needed).toBe(false);
  });
});
