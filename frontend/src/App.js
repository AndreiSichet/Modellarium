import { Navigate, Route, Routes } from 'react-router-dom';

import AboutPage from './components/AboutPage';
import AppShell from './components/AppShell';
import GameDetailPage from './components/GameDetailPage';
import GamesPage from './components/GamesPage';
import LeaguePage from './components/LeaguePage';
import PredictionsLayout from './components/PredictionsLayout';

/**
 * Route table.
 *
 * `/` IS ABOUT, not predictions, and that is deliberate rather than a
 * placeholder ordering.
 *
 * The router itself is mounted in index.js rather than here, so that a test
 * can wrap this component in a MemoryRouter and drive the route directly.
 * With BrowserRouter inside App there would be no way to render a route
 * other than whatever jsdom's location happened to be.
 */
function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<AboutPage />} />

        {/*
          A LAYOUT ROUTE, and that is what keeps the request count flat.
          PredictionsLayout owns the rail and the fetch; the three children
          below render into its Outlet. Because the layout is matched by all
          three, moving between them does not remount it and does not
          refetch - which matters more than it sounds, since every game on
          screen costs a POST that writes a row.
        */}
        <Route path="/predictions" element={<PredictionsLayout />}>
          {/*
            /predictions NO LONGER REDIRECTS. Phase 2 sent it to
            /predictions/basketball on the grounds that a landing page
            asking the user to choose from a list of one decides nothing.
            That was right about the rail and wrong about the page: General
            is a real destination with its own content - today's games
            across every sport - not a menu.
          */}
          <Route index element={<GamesPage />} />
          {/*
            Same component as the index route, scoped to one sport. Two
            components would be two copies of the subsections, the
            shortcuts, the cap and the More links, and the first edit to
            either is where they start disagreeing.

            An unrecognised sport renders an inline message with the rail
            intact rather than redirecting: a wrong URL silently becoming a
            different page hides the mistake instead of reporting it.
          */}
          <Route path=":sport" element={<GamesPage />} />
          <Route path=":sport/:league" element={<LeaguePage />} />
          {/*
            THE GAME ID IS A FOURTH SEGMENT, not a /game/ sub-path. It reads
            naturally, it gives the page its breadcrumb for free, and it
            avoids a literal `game` segment that would shadow a league slug
            the first time a league were called that.
          */}
          <Route path=":sport/:league/:gameId" element={<GameDetailPage />} />
        </Route>

        {/* Anything unknown goes home. `replace` so the bad URL does not
            sit in history waiting for the back button. */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  );
}

export default App;
