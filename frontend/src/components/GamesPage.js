import { Link, useOutletContext, useParams } from 'react-router-dom';

import { LEAGUES, leagueCopy, leaguesForSport } from '../data/leagues';
import { dataAsOfFor, isPredictable } from '../dates';
import GameList from './GameList';
import LeagueShortcuts, { useActiveSection } from './LeagueShortcuts';
import { findSport } from './SportsRail';
import './GamesPage.css';

const MAX_GAMES_PER_SECTION = 5;

export function soonestGames(games, limit = MAX_GAMES_PER_SECTION) {
  return byDate(games).slice(0, limit);
}

export function byDate(games) {
  return games
    .slice()
    .sort((a, b) => (a.gameDate < b.gameDate ? -1 : a.gameDate > b.gameDate ? 1 : 0));
}

function sectionId(league) {
  return `league-${league.sport}-${league.slug}`;
}

function GamesPage() {
  const { sport: slug } = useParams();
  const sport = slug ? findSport(slug) : null;
  // `health` as well as the ready-made predictableDateFor, because the
  // per-league dispatch needs the raw freshness to answer the date rule and
  // the NFL's rule does not use a date at all.
  const { games, schedule, health, predictableDateFor } = useOutletContext();

  const unknownSport = Boolean(slug) && !sport;

  const leagues = unknownSport ? [] : slug ? leaguesForSport(slug) : LEAGUES;

  // One section per league, each judged by ITS OWN cutoff. A single shared
  // date would mark one league's games predictable on the other league's
  // freshness.
  const sections = leagues.map((league) => {
    const date = predictableDateFor(league);

    return {
      id: sectionId(league),
      league,
      date,
      // THE SAME DISPATCH AS THE LAYOUT, with exactDate because General and
      // Upcoming show the predictable DAY rather than everything up to the
      // cutoff. For the NFL that distinction does not apply: its predictable
      // set is a slate across Thursday, Sunday and Monday, so the flag on the
      // fixture is the whole answer and a date would be the wrong question.
      games: soonestGames(
        games.filter(
          (game) =>
            game.leagueSlug === league.slug
            && isPredictable(game, health, league, { exactDate: true })
        )
      ),
    };
  });

  const withGames = sections.filter((section) => section.games.length > 0);

  const [active, setActive] = useActiveSection(sections.map((s) => s.id));

  if (unknownSport) return <UnknownSport slug={slug} />;

  return (
    <div className="games-page">
      <header className="games-page-head">
        <h1 className="games-page-title">{sport ? 'Upcoming' : 'General'}</h1>

        <p className="games-page-sub">
          {sport ? `${sport.label} games` : 'Games across every sport'}, by the
          newest date each league can be predicted for
        </p>
      </header>

      {withGames.length === 0 ? (
        <NothingPredictable sport={sport} leagues={leagues} />
      ) : (
        <>
          <LeagueShortcuts
            sections={sections}
            active={active}
            onActivate={setActive}
          />

          {sections.map((section) => (
            <LeagueSection
              key={section.id}
              section={section}
              schedule={schedule}
            />
          ))}
        </>
      )}
    </div>
  );
}

function LeagueSection({ section, schedule }) {
  const { league, games, date } = section;

  return (
    <section className="league-section" id={section.id}>
      <div className="league-section-head">
        <h2 className="league-section-title">{league.label}</h2>

        <Link
          className="league-section-more"
          to={`/predictions/${league.sport}/${league.slug}`}
        >
          More {league.label}
        </Link>
      </div>

      {games.length > 0 ? (
        <>
          {date ? <p className="league-section-date">{date}</p> : null}
          <GameList games={games} league={league} />
        </>
      ) : (
        // ONE QUIET LINE RATHER THAN NO SECTION AT ALL. Dropping an empty
        // league would leave its page reachable only by a typed URL, which
        // defeats having built it; a line is cheaper than an undiscoverable
        // page. It is not a placeholder for a league with no capability -
        // this league serves predictions, it just has no games today.
        <p className="league-section-empty">
          {leagueCopy(league.seasonStatus, { schedule })}
        </p>
      )}
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

function NothingPredictable({ sport, leagues }) {
  const { health, schedule } = useOutletContext();

  return (
    <div className="predictions-message">
      <h1 className="predictions-message-title">No predictions yet</h1>

      {/* EACH LEAGUE STATES ITS OWN DATA DATE, beside its own note. This
          used to print one shared `health.dataAsOf` under all of them, which
          is the NBA's cutoff - so the WNBA's line was followed by a date ten
          weeks wrong, and the G League's would be wrong again differently.
          Same principle as the per-league predictability filter, applied to
          what the page SAYS rather than to what it requests. */}
      {leagues.map((league) => {
        const dataAsOf = dataAsOfFor(health, league);
        const note = leagueCopy(league.seasonNote, { schedule });

        return (
          <div className="predictions-message-league" key={league.slug}>
            <p className="predictions-message-body">
              <strong>{league.label}:</strong> {note[0]}
            </p>

            {dataAsOf ? (
              <p className="predictions-message-meta">
                {league.label} model data is current to {dataAsOf}.
              </p>
            ) : null}
          </div>
        );
      })}

      {sport ? (
        <p className="predictions-message-meta">
          Nothing is scheduled for {sport.label} today.
        </p>
      ) : null}

      {/* Links, so no league page is reachable only by a typed URL while
          every league happens to be between seasons.

          A NAV WITH A TOKEN GAP, NOT INLINE TEXT. These rendered as
          "More NBAMore WNBA" - three links running together once the G
          League joined - because anchors are inline and nothing separated
          them. Spaced by flex gap rather than by a typed space or a
          separator character, so each stays one distinct target for a
          keyboard and a screen reader. */}
      <nav className="predictions-message-links" aria-label="Leagues">
        {leagues.map((league) => (
          <Link
            className="league-section-more"
            key={league.slug}
            to={`/predictions/${league.sport}/${league.slug}`}
          >
            More {league.label}
          </Link>
        ))}
      </nav>
    </div>
  );
}

export default GamesPage;
