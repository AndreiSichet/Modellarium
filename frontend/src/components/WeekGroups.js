import { formatKickoff } from '../dates';
import { teamFor } from '../data/teams';
import GameRow from './GameRow';
import TeamBadge from './TeamBadge';
import './WeekGroups.css';

/**
 * The current week and the next, grouped under Week N headings.
 *
 * WHY THIS EXISTS AT ALL, rather than the flat date-ordered list every other
 * league's page uses: an NFL week is the unit a reader thinks in, and the
 * predictable set is a slate spread across Thursday, Sunday and Monday. A
 * flat list of dates would show the same fixtures with the structure that
 * makes sense of them removed.
 *
 * UNPREDICTABLE FIXTURES ARE SHOWN AND MARKED, NOT HIDDEN. A reader on a
 * Tuesday should see what is coming and why it has no numbers yet; hiding
 * them would make the page look empty for most of the week and give no
 * account of itself.
 *
 * THE SCHEDULE IS THE SOURCE HERE, NOT THE PREDICTED GAMES. Only predictable
 * fixtures get a prediction, so the unpredictable ones exist nowhere else -
 * they are in the cached schedule alone, and a prediction is attached where
 * one happens to exist.
 */
export function weekGroups(schedule, games, league) {
  const fixtures = (schedule || []).filter(
    (game) => game?.leagueSlug === league.slug && game?.week != null
  );

  if (!fixtures.length) return [];

  // THE CURRENT WEEK IS THE LOWEST ONE STILL SCHEDULED, derived rather than
  // taken from a clock. The backend returns unplayed fixtures only, so a week
  // disappears from this list as it is played - which makes the minimum the
  // current week by construction and needs no date arithmetic that could
  // disagree with the data.
  const weeks = fixtures.map((game) => game.week);
  const current = Math.min(...weeks);

  const predictionFor = new Map(
    (games || [])
      .filter((game) => game.leagueSlug === league.slug)
      .map((game) => [keyOf(game), game])
  );

  return [current, current + 1]
    .map((week) => ({
      week,
      fixtures: fixtures
        .filter((game) => game.week === week)
        .sort(byKickoff)
        .map((game) => predictionFor.get(keyOf(game)) ?? game),
    }))
    .filter((group) => group.fixtures.length > 0);
}

function keyOf(game) {
  return `${game.homeTeamId}-${game.awayTeamId}-${game.gameDate}`;
}

/**
 * Soonest first, by the instant where there is one.
 *
 * A fixture with no kickoff sorts after the timed ones of the same date
 * rather than being dropped or floated to the top: TBD means "later than
 * anything already scheduled that day is known to be", which is the least
 * misleading place to put it.
 */
function byKickoff(a, b) {
  if (a.gameDate !== b.gameDate) return a.gameDate < b.gameDate ? -1 : 1;
  if (a.kickoffUtc && b.kickoffUtc) {
    return a.kickoffUtc < b.kickoffUtc ? -1 : a.kickoffUtc > b.kickoffUtc ? 1 : 0;
  }
  if (a.kickoffUtc) return -1;
  if (b.kickoffUtc) return 1;
  return 0;
}

function WeekGroups({ schedule, games, league }) {
  const groups = weekGroups(schedule, games, league);

  return (
    <div className="week-groups">
      {groups.map(({ week, fixtures }) => (
        <section className="week-group" key={week}>
          <h2 className="week-group-title">Week {week}</h2>
          <ul className="game-list">
            {fixtures.map((game) =>
              game.prediction ? (
                <GameRow key={keyOf(game)} game={game} league={league} />
              ) : (
                <NotYetRow key={keyOf(game)} game={game} />
              )
            )}
          </ul>
        </section>
      ))}
    </div>
  );
}

/**
 * A fixture that cannot be predicted yet, with the reason in place of numbers.
 *
 * Deliberately not a GameRow with blanks: that component reads a prediction's
 * fields directly, and a row of empty cells under Spread/Win/Total reads as a
 * failure rather than as "not yet".
 */
function NotYetRow({ game }) {
  const when = formatKickoff(game);

  return (
    <li className="game-row week-not-yet">
      <div className="week-not-yet-teams">
        <Side teamId={game.homeTeamId} name={game.homeTeamName} abbr={game.homeTeamAbbr} />
        <span className="week-not-yet-against">v</span>
        <Side teamId={game.awayTeamId} name={game.awayTeamName} abbr={game.awayTeamAbbr} />
      </div>

      <div className="game-row-footer">
        <span className="game-row-when">
          {when.text}
          {when.flex ? (
            <span className="game-row-flex" title="The kickoff time may move">
              flex
            </span>
          ) : null}
        </span>
        <span className="week-not-yet-reason">
          Not predictable yet — both teams need to have played their previous
          game
        </span>
      </div>
    </li>
  );
}

function Side({ teamId, name, abbr }) {
  return (
    <span className="week-not-yet-side">
      <TeamBadge teamId={teamId} abbr={abbr} />
      <span className="game-row-team-name">{name || teamFor(teamId).name}</span>
    </span>
  );
}

export default WeekGroups;
