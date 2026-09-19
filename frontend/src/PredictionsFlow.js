import { useState } from 'react';
import './App.css';
import BrowseView from './BrowseView';
import PredictionView from './PredictionView';

/**
 * The browse/detail view controller, lifted VERBATIM out of App.js when
 * App became a route table in Phase 1. Not one line of its behaviour
 * changed in the move.
 *
 * WHY IT EXISTS AS A SEPARATE FILE RATHER THAN BEING DELETED OR INLINED.
 * Phase 1 makes PredictionsPage a placeholder and leaves the prediction
 * components unimported by the app. But fifteen tests exercise this flow
 * through App, and they cannot pass against a route table. Deleting them
 * would trade a green suite for a green suite with no coverage; skipping
 * them would leave the project's largest test file inert for a phase.
 *
 * So the controller keeps existing, the tests keep exercising it, and
 * Phase 2 wires this into PredictionsPage instead of rebuilding it.
 *
 * NOTHING IN THE APP IMPORTS THIS. Only the test file does, which is what
 * keeps BrowseView and PredictionView out of the production bundle while
 * still compiling them.
 *
 * scheduleGames lives here rather than in BrowseView so that going Back
 * from a prediction returns to the list already fetched, instead of
 * hitting the NBA schedule API again.
 */
function PredictionsFlow() {
  const [view, setView] = useState('browse');
  const [scheduleGames, setScheduleGames] = useState([]);
  const [selectedResult, setSelectedResult] = useState(null);

  function handleSelect(result) {
    setSelectedResult(result);
    setView('detail');
  }

  return (
    <div className="app">
      {view === 'browse' ? (
        <BrowseView
          games={scheduleGames}
          onGamesLoaded={setScheduleGames}
          onSelect={handleSelect}
        />
      ) : (
        <PredictionView
          result={selectedResult}
          onBack={() => setView('browse')}
        />
      )}
    </div>
  );
}

export default PredictionsFlow;
