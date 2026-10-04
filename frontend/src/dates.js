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
