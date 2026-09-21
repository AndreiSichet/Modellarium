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
  const { games, predictableDate } = useOutletContext();

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

    .filter((section) => section.games.length > 0);

  const [active, setActive] = useActiveSection(sections.map((s) => s.id));

  if (unknownSport) return <UnknownSport slug={slug} />;

  return (
    <div className="games-page">
      <header className="games-page-head">
        <h1 className="games-page-title">{sport ? 'Upcoming' : 'General'}</h1>

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

function LeagueSection({ section }) {
  const { league } = section;

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
