import GameList from './GameList';
import './DateSlate.css';

/**
 * A flat date-ordered slate that SHOWS the fixtures it cannot predict yet.
 *
 * THE THIRD PAGE SHAPE, and it exists because `weeklySlate` bundled two
 * decisions that come apart. That flag meant both "group by week" and "show
 * unpredictable fixtures"; the NFL wants both, and a league that plays most
 * nights wants only the second. "Week 10" is not a heading a hockey reader
 * uses, and hiding 1,081 of 1,083 fixtures would leave the page apparently
 * empty on a night with a full slate.
 *
 * THE COUNT AND THE LIST COME FROM THE SAME ROWS, which is the whole point.
 * `/api/health` reports a live `predictableFixtures` from the inference
 * service, while the schedule rows come from the backend's cache, synced
 * every six hours - so on 6 November health said 2 and the cached rows said
 * 0. A page that took its headline from one and its list from the other
 * would contradict itself in the reader's own view. So the number below is
 * counted here, from the rows being rendered.
 */
function DateSlate({ league, schedule, games }) {
  const scheduled = (schedule || []).filter(
    (game) => game.leagueSlug === league.slug
  );

  // PREDICTED ROWS REPLACE THEIR SCHEDULED TWIN. `games` carries the ones a
  // prediction was fetched for (and the ones that were refused); `schedule`
  // carries every fixture. Keyed the way PredictionsLayout keys them, so the
  // two sides match without a second convention.
  const predictedByKey = new Map(
    (games || [])
      .filter((game) => game.leagueSlug === league.slug)
      .map((game) => [game.key, game])
  );

  const rows = scheduled
    .map((game) => {
      const key = `${game.homeTeamId}-${game.awayTeamId}-${game.gameDate}`;
      return predictedByKey.get(key) ?? { ...game, key, prediction: null };
    })
    .sort((a, b) => (a.gameDate < b.gameDate ? -1
      : a.gameDate > b.gameDate ? 1 : 0));

  const predictable = rows.filter(
    (row) => row.predictable === true
  ).length;

  return (
    <>
      <section className="date-slate-explainer">
        <p className="date-slate-rule">
          {league.predictabilityExplainer
            || 'A game can be predicted once both teams’ previous results are confirmed.'}
        </p>
        {/* ONE SENTENCE, ONE SOURCE. Both numbers are `rows`, so they cannot
            disagree with the list underneath them. */}
        <p className="date-slate-count">
          <strong>{predictable.toLocaleString()}</strong>
          {' of '}
          {rows.length.toLocaleString()}
          {' upcoming games can be predicted today.'}
        </p>
      </section>

      {/* ZERO IS A STATE WITH ITS OWN MESSAGE, not an empty list. A reader
          looking at a thousand rows none of which is clickable needs to be
          told that is expected - otherwise the app looks broken, which is
          exactly what this page has to avoid. */}
      {rows.length === 0 ? (
        <p className="date-slate-empty">
          No upcoming fixtures are cached for this league yet.
        </p>
      ) : predictable === 0 ? (
        <>
          <p className="date-slate-none-yet">
            None of these games can be predicted yet — each one is waiting on
            an earlier result. They become available as those results are
            confirmed, usually the day after both teams last played.
          </p>
          <GameList games={rows} league={league} />
        </>
      ) : (
        <GameList games={rows} league={league} />
      )}
    </>
  );
}

export default DateSlate;
