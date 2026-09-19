import { Navigate, Route, Routes } from 'react-router-dom';

import AboutPage from './components/AboutPage';
import AppShell from './components/AppShell';
import PredictionsPage from './components/PredictionsPage';

/**
 * Route table. Two screens did not justify a router; five will, and the
 * sport parameter has to live somewhere the URL can carry it.
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
        {/* /predictions REDIRECTS rather than landing, and this is a
            decision that expires. With exactly one sport available, a
            landing state whose job is to ask the user to choose from a
            list of one is a click that decides nothing. The rail still
            shows Basketball as active, so the redirect is invisible.

            WHEN A SECOND SPORT IS ADDED, this should become a real landing
            route instead — at that point the choice is genuine. */}
        <Route
          path="/predictions"
          element={<Navigate to="/predictions/basketball" replace />}
        />
        {/* An unrecognised sport renders the layout with an inline message
            rather than redirecting. A wrong URL silently becoming a
            different page hides the mistake instead of reporting it. */}
        <Route path="/predictions/:sport" element={<PredictionsPage />} />
        {/* Anything unknown goes home. `replace` so the bad URL does not
            sit in history waiting for the back button. */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  );
}

export default App;
