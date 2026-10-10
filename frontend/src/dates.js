export function addDays(isoDate, days) {
  const parsed = new Date(`${isoDate}T00:00:00`);
  parsed.setDate(parsed.getDate() + days);

  const month = String(parsed.getMonth() + 1).padStart(2, '0');
  const day = String(parsed.getDate()).padStart(2, '0');
  return `${parsed.getFullYear()}-${month}-${day}`;
}

export const MAX_DAYS_AHEAD = 1;

export function latestPredictableDate(dataAsOf) {
  return dataAsOf ? addDays(dataAsOf, MAX_DAYS_AHEAD) : null;
}

/**
 * The cutoff for one league's data, from /api/health.
 *
 * PREDICTABILITY IS PER LEAGUE AND MUST NOT FALL BACK TO THE NBA'S. The two
 * leagues' data ends on different dates, so using the top-level cutoff for a
 * WNBA game would mark it predictable whenever NBA data is fresh - which
 * in season is every day - and the page would then request a prediction the
 * backend rejects. A league with no cutoff yields null, which reads as
 * "nothing is predictable" rather than as "today".
 */
export function dataAsOfFor(health, league) {
  if (!health) return null;
  const source = league?.freshness ? health[league.freshness] : health;
  return source?.dataAsOf ?? null;
}

export function predictableDateFor(health, league) {
  return latestPredictableDate(dataAsOfFor(health, league));
}

/**
 * Whether one fixture can be predicted, by whichever rule its league uses.
 *
 * TWO RULES, AND THE LEAGUE DECLARES WHICH. Basketball's is a date: one day
 * after that league's cutoff, which a client can compute. The NFL's is a
 * dependency - both teams' previous games must be in history - which is
 * measured in the inference service and is not derivable from any date. So
 * the NFL's answer travels with the fixture as `predictable`, and this
 * function dispatches rather than reimplementing it. A second copy of that
 * rule here would drift from the one that refuses the request.
 *
 * NO DEFAULT RULE. §53's `|| 'nba'` read as a fallback and was in fact the
 * only path that ever ran, so a league that has not declared a rule is a bug
 * rather than a basketball league: it is reported and treated as not
 * predictable, which fails closed.
 *
 * `exactDate` is the one place the two call sites genuinely differ. The
 * General and Upcoming pages show the predictable DAY's games, so they want
 * an exact match; the layout decides what to request a prediction for and
 * wants everything up to the cutoff. The server rule is unaffected either
 * way - a fixture is predictable or it is not.
 */
export function isPredictable(game, health, league, options) {
  if (!game || !league) return false;

  if (league.rule === 'server') return game.predictable === true;

  if (league.rule === 'date') {
    const date = predictableDateFor(health, league);
    if (!date) return false;
    return options?.exactDate ? game.gameDate === date : game.gameDate <= date;
  }

  // eslint-disable-next-line no-console
  console.warn(
    `league ${league.slug} declares no predictability rule, so none of its `
      + 'fixtures will be offered. Add rule: "date" or rule: "server".'
  );
  return false;
}

const MONTHS = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
];

/**
 * An ISO date as "27 December 2026", matching the hand-written copy.
 *
 * Parsed by hand rather than through Date, because `new Date('2026-12-27')`
 * is treated as UTC midnight and then rendered in local time, which moves
 * the day backwards for anyone west of Greenwich. A date shown in copy must
 * not depend on where the reader is.
 */
export function formatLongDate(isoDate) {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(isoDate || '');
  if (!match) return null;

  const [, year, month, day] = match;
  const name = MONTHS[Number(month) - 1];
  return name ? `${Number(day)} ${name} ${year}` : null;
}

/**
 * When a fixture starts, in the viewer's own zone.
 *
 * BASKETBALL IS UNCHANGED BY CONSTRUCTION. Those fixtures carry no kickoff,
 * so this returns the bare ISO date their rows already showed and no empty
 * time slot appears anywhere.
 *
 * ONE DATE FORMAT FOR EVERY LEAGUE. The date is rendered the way basketball
 * rows already render it - ISO, `2026-10-12` - with the time appended, rather
 * than switching to a prose style for one sport. It is built from the local
 * calendar components rather than through a locale formatter, because the
 * DATE must not change shape with the reader's locale; only the TIME should.
 *
 * THE DATE COMES FROM THE KICKOFF INSTANT, NOT FROM `gameDate`, WHENEVER ONE
 * EXISTS. An NFL Sunday-night game kicks off at 00:20Z the following day; its
 * `gameDate` is the United States date, so pairing that date with a local
 * time would show a reader in Bucharest "2026-10-11, 03:20" for a game that
 * is on the 12th where they are. Both halves have to come from the same
 * instant or they contradict each other.
 *
 * THE TIME GOES THROUGH Intl AT THE DEFAULT LOCALE, so a 24-hour reader sees
 * 20:00 and a 12-hour one sees 8:00 PM. Nothing here forces a clock
 * convention; `timeStyle: 'short'` lets the locale decide, which is the whole
 * point of showing a local time at all.
 *
 * A null kickoff is a fixture whose time is not set - the NFL publishes 24 of
 * those - and it shows the scheduled date alone rather than a guessed time.
 * `flex` means the time may still move, which is a fact about the fixture
 * rather than about the rendering, so it is returned separately for the
 * caller to mark.
 */
export function formatKickoff(game) {
  const kickoff = game?.kickoffUtc;
  const flex = game?.flex === true;
  const fallback = { text: game?.gameDate ?? null, flex, timed: false };

  if (!kickoff) return fallback;

  const at = new Date(kickoff);
  if (Number.isNaN(at.getTime())) return fallback;

  // Local calendar components, the same construction addDays uses above, so
  // the date is the viewer's own day without passing through a locale.
  const month = String(at.getMonth() + 1).padStart(2, '0');
  const day = String(at.getDate()).padStart(2, '0');
  const date = `${at.getFullYear()}-${month}-${day}`;

  const time = new Intl.DateTimeFormat(undefined, {
    timeStyle: 'short',
  }).format(at);

  return { text: `${date}, ${time}`, flex, timed: true };
}
