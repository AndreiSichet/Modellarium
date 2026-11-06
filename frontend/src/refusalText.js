import { shortDate } from './dates';
import { nameOf } from './data/teams';

/**
 * One sentence per refusal code, for a fixture that will not be predicted.
 *
 * ONE PLACE, BECAUSE THE SAME CODES REACH TWO SURFACES. The browse row and
 * the detail tabs both have to say why a fixture was declined, and two copies
 * of that wording is two things to update when a seventh code appears.
 *
 * AN UNKNOWN CODE IS NOT A BLANK. `null` from here means "nothing has words
 * for this", and the caller falls back to the API's own prose - which is why
 * `detail` is still sent and still carried. A missing entry must never render
 * as an empty explanation.
 */
export const REFUSAL_TEXT = {
  insufficient_history: 'Not enough of this season played yet.',
  season_opener:
    "This is a team's first game of the season, so there is no rest or " +
    'recent form to measure from.',
  incomplete_features:
    'Some of the inputs these models need are missing for this fixture.',
  no_usable_history:
    'No player here has enough of this season recorded yet.',
  unknown_team: 'One of these teams has no completed games in the data.',
  date_out_of_range: 'This fixture is further ahead than the data allows.',
  same_team: 'A fixture needs two different teams.',
  synthetic_data_refused:
    'The service is refusing to serve predictions from placeholder data.',
  unclassified: 'The service declined this fixture without saying why.',
  // ADDED FOR THE NHL, WHERE IT IS ALMOST EVERY FIXTURE: 1,081 of 1,083 on
  // 6 November. This is the generic form; `scheduleReason` below writes a
  // better one when the row carries `blockedBy`, because naming the team
  // and the date is the difference between "wait" and "wait for what".
  previous_game_pending:
    'Waiting on an earlier result for one of these teams.',
};

/** The words for a refusal, or the API's own prose, or null. */
export function refusalText(refusal) {
  if (!refusal) return null;
  return REFUSAL_TEXT[refusal.code] ?? refusal.detail ?? refusal.message ?? null;
}

/**
 * The generic sentence, for a code nothing here has words for.
 *
 * USED WHERE PROSE MUST NOT REACH THE PAGE. `refusalText` falls back to the
 * API's own `detail`, which is right on the detail tabs - the reader asked
 * about one fixture and the server's sentence is the most specific thing
 * available. It is wrong on a schedule row: an unrecognised code there would
 * put a server message in a list of a thousand, where nobody asked for it
 * and nothing has read it for shape.
 */
export const GENERIC_NOT_PREDICTABLE = 'Not predictable yet.';

function teamName(blocker, game) {
  // The blocker names a SIDE and a FRANCHISE. The side tells us which of
  // the row's two teams it is, which is where the display name lives; the
  // franchise id is the fallback when a row has no names on it.
  if (blocker?.side === 'home') {
    return nameOf(game?.homeTeamId, game?.homeTeamName);
  }
  if (blocker?.side === 'away') {
    return nameOf(game?.awayTeamId, game?.awayTeamName);
  }
  return nameOf(blocker?.franchiseId, null);
}

function joinNames(names) {
  if (names.length <= 1) return names[0] || '';
  return `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
}

/**
 * Why a SCHEDULE row is not predictable, in one plain sentence.
 *
 * BUILT FROM THE CODE AND ITS FIELDS, NEVER FROM `reason`. The API sends an
 * English sentence too; this does not read it. The sentence belongs to the
 * reader and gets reworded, and a page that parses it breaks silently when
 * it is - which is why the server sends `blockedBy` as data at all.
 *
 * BOTH TEAMS WHEN BOTH ARE WAITING. Measured 2026-11-06: 1,074 of 1,083 NHL
 * fixtures were blocked on both sides. Naming one would send the reader to
 * watch the wrong game half the time.
 *
 * AN UNKNOWN CODE GETS THE GENERIC SENTENCE AND NEVER RAW TEXT, which is the
 * one place this differs from `refusalText` - see GENERIC_NOT_PREDICTABLE.
 */
export function scheduleReason(game) {
  const code = game?.reasonCode;
  if (!code) return null;

  const blockers = Array.isArray(game.blockedBy) ? game.blockedBy : [];

  const waiting = blockers.filter(
    (blocker) => blocker?.code === 'previous_game_pending'
      && Array.isArray(blocker.pending) && blocker.pending.length
  );
  if (waiting.length) {
    const names = waiting.map((blocker) => teamName(blocker, game))
      .filter(Boolean);
    // The LATEST pending date is the one the reader is actually waiting
    // for: an earlier one cannot resolve this fixture on its own.
    const latest = waiting
      .flatMap((blocker) => blocker.pending)
      .sort()
      .pop();
    const when = shortDate(latest);
    if (names.length && when) {
      return names.length > 1
        ? `Waiting for ${joinNames(names)} to have their results from ${when}.`
        : `Waiting for ${names[0]}’s result from ${when}.`;
    }
  }

  const openers = blockers.filter(
    (blocker) => blocker?.code === 'season_opener'
  );
  if (openers.length) {
    const names = openers.map((blocker) => teamName(blocker, game))
      .filter(Boolean);
    return names.length
      ? `First game of the season for ${joinNames(names)} — no history yet.`
      : 'First game of the season — no history yet.';
  }

  return REFUSAL_TEXT[code] ?? GENERIC_NOT_PREDICTABLE;
}
