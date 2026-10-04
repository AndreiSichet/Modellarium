import { Link, useOutletContext, useParams } from 'react-router-dom';

import { LEAGUES, leaguesForSport } from '../data/leagues';
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
  const { games, predictableDateFor } = useOutletContext();

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
      games: soonestGames(
        games.filter(
          (game) => game.leagueSlug === league.slug && game.gameDate === date
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
            <LeagueSection key={section.id} section={section} />
          ))}
        </>
      )}
    </div>
  );
}

function LeagueSection({ section }) {
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
        <p className="league-section-empty">{league.seasonStatus}</p>
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
  const { health } = useOutletContext();

  return (
    <div className="predictions-message">
      <h1 className="predictions-message-title">No predictions yet</h1>

      {leagues.map((league) => (
        <p className="predictions-message-body" key={league.slug}>
          <strong>{league.label}:</strong> {league.seasonNote[0]}
        </p>
      ))}

      {health?.dataAsOf ? (
        <p className="predictions-message-meta">
          Model data is current to {health.dataAsOf}
          {sport ? `, and nothing is scheduled for ${sport.label} today` : ''}.
        </p>
      ) : null}

      {/* Links, so no league page is reachable only by a typed URL while
          every league happens to be between seasons. */}
      <p className="predictions-message-body">
        {leagues.map((league) => (
          <Link
            className="league-section-more"
            key={league.slug}
            to={`/predictions/${league.sport}/${league.slug}`}
          >
            More {league.label}
          </Link>
        ))}
      </p>
    </div>
  );
}

export default GamesPage;
