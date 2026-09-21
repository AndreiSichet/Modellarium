import { Navigate, Route, Routes } from 'react-router-dom';

import AboutPage from './components/AboutPage';
import AppShell from './components/AppShell';
import GameDetailPage from './components/GameDetailPage';
import GamesPage from './components/GamesPage';
import LeaguePage from './components/LeaguePage';
import PredictionsLayout from './components/PredictionsLayout';

function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<AboutPage />} />

        <Route path="/predictions" element={<PredictionsLayout />}>

          <Route index element={<GamesPage />} />

          <Route path=":sport" element={<GamesPage />} />
          <Route path=":sport/:league" element={<LeaguePage />} />

          <Route path=":sport/:league/:gameId" element={<GameDetailPage />} />
        </Route>

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  );
}

export default App;
