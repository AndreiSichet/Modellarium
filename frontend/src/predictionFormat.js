/**
 * How a prediction becomes a string. Shared, because these rules are now
 * read in two places and a second copy of a SIGN CONVENTION is the kind of
 * duplication that inverts silently.
 *
 * Extracted from GameRow in Phase 6, when the detail page needed the same
 * spread on the same game. Both surfaces now derive from this, so "the row
 * says -1.4 and the detail page says +1.4" cannot happen.
 */

/**
 * THE SINGLE EASIEST THING HERE TO GET BACKWARDS, and it fails silently
 * because both versions look like plausible numbers.
 *
 * `homeMargin` is the predicted home margin: positive means the home team
 * is expected to win by that much. Betting convention gives the FAVOURITE a
 * NEGATIVE spread, so the sign flips on the way to the screen. A homeMargin
 * of +1.41 displays as -1.4 for home and +1.4 for away.
 */
export function formatSpread(homeMargin, isHome) {
  const value = isHome ? -homeMargin : homeMargin;
  // toFixed(1) on -0.04 gives "-0.0"; normalising through +0 avoids
  // printing a negative zero, which reads as a typo at a magnitude where
  // the sign carries no information anyway.
  const rounded = Number((value + 0).toFixed(1)) + 0;
  return rounded > 0 ? `+${rounded.toFixed(1)}` : rounded.toFixed(1);
}

/** `homeWinProbability` is the home figure; away is its complement. */
export function formatWin(homeWinProbability, isHome) {
  const value = isHome ? homeWinProbability : 1 - homeWinProbability;
  return `${(value * 100).toFixed(1)}%`;
}

/** One decimal, the convention every model output on screen follows. */
export function formatValue(value) {
  return typeof value === 'number' ? value.toFixed(1) : '—';
}
