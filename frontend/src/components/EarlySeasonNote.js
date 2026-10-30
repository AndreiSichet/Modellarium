/**
 * Why October's numbers look the way they do.
 *
 * THE MEASUREMENT THIS EXISTS FOR, 2026-10-30. The team models answer every
 * fixture, but `drop_incomplete` removed every early-season row from their
 * training set - so October is a regime they were never fitted on. Of 34
 * feature columns, 28 were NaN in both fixtures of a tied pair; the totals
 * model put them in the SAME LEAF OF ALL 108 TREES and returned
 * 219.0228271484375 twice, while spread and moneyline separated the same
 * pair in 27 and 23 trees. Mid-season the effect is absent: 0.0% of totals
 * are non-unique across three seasons' mid-season samples.
 *
 * So the predictions are not wrong and are not broken. They are running on
 * missing-value defaults, and a reader had no way to tell.
 *
 * IT DISAPPEARS BY ITSELF. No date is written here and none is written in
 * the backend: inference sends `needed` - max(WINDOWS) from the feature
 * definitions - and each side's games played, and this renders while the
 * lower of the two is short. The date comes from the real remaining
 * schedule, because the spread between the first and last team to reach the
 * window was five days on the morning this was built and no average would
 * have produced it.
 */
function EarlySeasonNote({ earlySeason }) {
  if (!earlySeason) return null;

  const { needed, homeGamesPlayed, awayGamesPlayed, fullWindowDate } = earlySeason;
  if (typeof needed !== 'number') return null;
  if (typeof homeGamesPlayed !== 'number' || typeof awayGamesPlayed !== 'number') {
    return null;
  }

  // THE LOWER OF THE TWO, because a window needs BOTH sides. One team at
  // twelve games does not make a fixture mid-season.
  if (Math.min(homeGamesPlayed, awayGamesPlayed) >= needed) return null;

  return (
    <p className="early-season-note" role="note">
      Early in the season, teams have played fewer than {needed} games, so these
      predictions lean mostly on last season&rsquo;s ratings and rest days.
      Totals can come out close or identical across games.
      {fullWindowDate
        ? ` Both teams reach a full window on ${fullWindowDate}.`
        : ''}
    </p>
  );
}

export default EarlySeasonNote;
