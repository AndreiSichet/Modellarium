import { Link, useOutletContext, useParams } from 'react-router-dom';

import { LEAGUES, leaguesForSport } from '../data/leagues';
import GameList from './GameList';
import LeagueShortcuts, { useActiveSection } from './LeagueShortcuts';
import { findSport } from './SportsRail';
import './GamesPage.css';

/** A summary; the league page is where the rest of a league's games are. */
const MAX_GAMES_PER_SECTION = 5;

/**
 * THE FIVE SOONEST, NOT THE FIRST FIVE RETURNED.
 *
 * The schedule arrives in whatever order the API gives it, so capping
 * without sorting would show five arbitrary games and call them upcoming.
 * Exported because the sort and the cap are the two things here that can be
 * wrong while still looking entirely plausible.
 */
export function soonestGames(games, limit = MAX_GAMES_PER_SECTION) {
  return byDate(games).slice(0, limit);
}

/**
 * slice() first: sort mutates in place, and the array handed in is the
 * layout's own state. Sorting it directly would reorder another page as a
 * side effect of rendering this one.
 *
 * Dates are plain YYYY-MM-DD strings, which compare correctly
 * lexicographically - the same property dates.js relies on. Array#sort is
 * stable (ES2019), so games on the same date keep schedule order rather
 * than shuffling between renders.
 */
export function byDate(games) {
  return games
    .slice()
    .sort((a, b) => (a.gameDate < b.gameDate ? -1 : a.gameDate > b.gameDate ? 1 : 0));
}

function sectionId(league) {
  return `league-${league.sport}-${league.slug}`;
}

/**
 * GENERAL AND UPCOMING ARE THIS ONE COMPONENT, differing only in scope.
 *
 *   /predictions          -> sport is undefined -> every league
 *   /predictions/:sport   -> one sport's leagues
 *
 * Same subsections, same shortcuts, same More links, same cap. Two
 * components would be two copies of all of that, and the first edit to
 * either is where they start telling the user different things.
 *
 * WHAT "TODAY" MEANS HERE, because it is not `new Date()`. MAX_DAYS_AHEAD
 * is 1, so the only date the service will score is data_as_of + 1. In
 * normal in-season operation the pipeline retrains overnight and that is
 * the real calendar today. When the data is stale - as now - they diverge,
 * and a literal today would render an empty page while predictions for a
 * real date sat one route away. dates.js already owns that rule; this reads
 * it rather than adding a second one beside it.
 */
function GamesPage() {
  const { sport: slug } = useParams();
  const sport = slug ? findSport(slug) : null;
  const { games, predictableDate } = useOutletContext();

  // An unrecognised sport renders inline with the rail intact rather than
  // redirecting: a wrong URL silently becoming a different page hides the
  // mistake instead of reporting it.
  const unknownSport = Boolean(slug) && !sport;

  const leagues = unknownSport ? [] : slug ? leaguesForSport(slug) : LEAGUES;

  const sections = leagues
    .map((league) => ({
      id: sectionId(league),
      league,
      games: soonestGames(
        games.filter(
          (game) =>
            game.leagueSlug === league.slug && game.gameDate === predictableDate
        )
      ),
    }))
    // A league with nothing today gets no section and no shortcut. Not the
    // same as hiding a league: it still exists, and its own page still
    // lists everything it has.
    .filter((section) => section.games.length > 0);

  const [active, setActive] = useActiveSection(sections.map((s) => s.id));

  if (unknownSport) return <UnknownSport slug={slug} />;

  return (
    <div className="games-page">
      <header className="games-page-head">
        <h1 className="games-page-title">{sport ? 'Upcoming' : 'General'}</h1>
        {/*
          THE DATE IS STATED, NOT IMPLIED. "Today's games" over a stale
          dataset means a date months in the past, and a page that shows it
          without saying so is the same misleading-green the stale badge
          exists to prevent.
        */}
        <p className="games-page-sub">
          {sport ? `${sport.label} games` : 'Games across every sport'} for{' '}
          {predictableDate || 'an unknown date'}
        </p>
      </header>

      {sections.length === 0 ? (
        <NoPredictionsYet sport={sport} />
      ) : (
        <>
          <LeagueShortcuts
            sections={sections}
            active={active}
            onActivate={setActive}
          />

          {sections.map((section) => (
            <LeagueSection key={section.id} section={section} />
          ))}
        </>
      )}
    </div>
  );
}

/**
 * One league's subsection: a heading, a link to the league's own page, and
 * up to five games as full GameRows.
 *
 * THE FULL ROW, NOT A COMPACT ONE. Phase 4's compact row existed only
 * because a 420px column could not hold a real one. The column is ~950px
 * now, so the row that already exists fits and the second one is deleted
 * rather than kept "for summaries" - two row components is how two places
 * come to show different numbers for the same game.
 */
function LeagueSection({ section }) {
  const { league } = section;

  return (
    <section className="league-section" id={section.id}>
      <div className="league-section-head">
        <h2 className="league-section-title">{league.label}</h2>
        {/*
          A LINK NOW, NOT A BUTTON. In Phase 4 this switched a tab, so it
          had to be a button - an anchor that does not navigate breaks
          middle-click and lies to a screen reader. The league is a route
          now, so it is genuinely a link, and the tab state lifted for the
          old arrangement is gone with it.

          The URL is built from the league's own sport rather than from the
          current route, which is what makes it correct on General where
          the two can differ.
        */}
        <Link
          className="league-section-more"
          to={`/predictions/${league.sport}/${league.slug}`}
        >
          More {league.label}
        </Link>
      </div>

      <GameList games={section.games} league={league} />
    </section>
  );
}

function UnknownSport({ slug }) {
  return (
    <div className="predictions-message">
      <h1 className="predictions-message-title">No such sport</h1>
      <p className="predictions-message-body">
        Nothing here covers <strong>{slug}</strong>. Pick a sport from the
        rail to see its predictions.
      </p>
    </div>
  );
}

/**
 * The state this page will be in until the season starts, so it reads as an
 * explanation rather than a blank column.
 *
 * The second date is DESCRIBED, NOT PINNED. Each team needs roughly ten
 * games of its own before its rolling form exists, and teams play on
 * different schedules - so there is no single date on which predictions
 * switch on. A hardcoded one would be a guess that goes stale silently.
 */
function NoPredictionsYet({ sport }) {
  const { health } = useOutletContext();

  return (
    <div className="predictions-message">
      <h1 className="predictions-message-title">No predictions yet</h1>
      <p className="predictions-message-body">
        The 2026-27 NBA season begins on 20 October 2026.
      </p>
      <p className="predictions-message-body">
        Predictions need recent form to work from, so they become available
        once each team has played enough games for that form to be computed —
        around ten games in, which is roughly three weeks after opening night.
      </p>
      {health?.dataAsOf ? (
        <p className="predictions-message-meta">
          Model data is current to {health.dataAsOf}
          {sport ? `, and nothing is scheduled for ${sport.label} today` : ''}.
        </p>
      ) : null}
    </div>
  );
}

export default GamesPage;
