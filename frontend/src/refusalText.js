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
};

/** The words for a refusal, or the API's own prose, or null. */
export function refusalText(refusal) {
  if (!refusal) return null;
  return REFUSAL_TEXT[refusal.code] ?? refusal.detail ?? refusal.message ?? null;
}
