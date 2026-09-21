import { useCallback, useEffect, useState } from 'react';
import { useOutletContext, useParams } from 'react-router-dom';

import { createQuarterHalfPrediction, getPlayerPropPredictions } from '../api';
import { DEV_FIXTURES_ON, devPlayerPropsFor, devQuarterHalfFor } from '../data/devFixtures';
import { findLeague } from '../data/leagues';
import { teamFor } from '../data/teams';
import Breadcrumb from './Breadcrumb';
import DetailTabs, { PLAYER_STATS, TABS, isPlayerTab, panelId, tabId } from './DetailTabs';
import GameTab from './GameTab';
import PlayerStatTab from './PlayerStatTab';
import QuarterHalfTab from './QuarterHalfTab';
import { findSport } from './SportsRail';
import TeamBadge from './TeamBadge';
import './GameDetailPage.css';

function GameDetailPage() {
  const { sport: sportSlug, league: leagueSlug, gameId } = useParams();
  const { games, health } = useOutletContext();

  const sport = findSport(sportSlug);
  const league = findLeague(sportSlug, leagueSlug);

  const game = games.find((entry) => String(entry.gameId) === String(gameId));

  const [tab, setTab] = useState(TABS[0].id);
  const [quarterHalf, setQuarterHalf] = useState({ status: 'loading', data: null, error: null });
  const [playerProps, setPlayerProps] = useState({ status: 'idle', data: null, error: null });

  const homeTeamId = game?.homeTeamId;
  const awayTeamId = game?.awayTeamId;
  const gameDate = game?.gameDate;
  const hasGame = Boolean(game);

  const loadQuarterHalf = useCallback(async () => {
    if (!gameDate) return;
    setQuarterHalf({ status: 'loading', data: null, error: null });
    try {
      const result = DEV_FIXTURES_ON
        ? devQuarterHalfFor(gameDate)
        : await createQuarterHalfPrediction({ homeTeamId, awayTeamId, gameDate });

      setQuarterHalf({ status: 'ready', data: result?.prediction ?? null, error: null });
    } catch (failure) {
      setQuarterHalf({ status: 'error', data: null, error: failure.message });
    }
  }, [homeTeamId, awayTeamId, gameDate]);

  const loadPlayerProps = useCallback(async () => {
    if (!gameDate) return;
    setPlayerProps({ status: 'loading', data: null, error: null });
    try {
      const result = DEV_FIXTURES_ON
        ? devPlayerPropsFor(gameDate)
        : await getPlayerPropPredictions({ homeTeamId, awayTeamId, gameDate });
      setPlayerProps({ status: 'ready', data: result, error: null });
    } catch (failure) {
      setPlayerProps({ status: 'error', data: null, error: failure.message });
    }
  }, [homeTeamId, awayTeamId, gameDate]);

  useEffect(() => {
    if (hasGame) loadQuarterHalf();
  }, [hasGame, loadQuarterHalf]);

  useEffect(() => {
    if (hasGame && isPlayerTab(tab) && playerProps.status === 'idle') {
      loadPlayerProps();
    }
  }, [hasGame, tab, playerProps.status, loadPlayerProps]);

  if (!sport || !league || !game) {
    return <NotFound sport={sport} league={league} gameId={gameId} />;
  }

  const stat = PLAYER_STATS.find((entry) => entry.id === tab);

  return (
    <div className="game-detail">
      <Breadcrumb
        items={[
          { label: 'Predictions', to: '/predictions' },
          { label: sport.label, to: `/predictions/${sport.slug}` },
          { label: league.label, to: `/predictions/${sport.slug}/${league.slug}` },
          { label: `${teamFor(game.awayTeamId).abbr} @ ${teamFor(game.homeTeamId).abbr}` },
        ]}
      />

      <GameHeader game={game} />

      <DetailTabs active={tab} onSelect={setTab} />

      <div
        className="detail-panel"
        role="tabpanel"
        id={panelId(tab)}

        aria-labelledby={tabId(tab)}

        tabIndex={0}
      >
        {tab === 'game' ? (
          <GameTab game={game} health={health} />
        ) : tab === 'quarters' ? (
          <QuarterHalfTab state={quarterHalf} onRetry={loadQuarterHalf} health={health} />
        ) : (
          <PlayerStatTab
            stat={stat}
            state={playerProps}
            onRetry={loadPlayerProps}
            game={game}
          />
        )}
      </div>
    </div>
  );
}

function GameHeader({ game }) {
  return (
    <header className="game-header">
      <div className="game-header-side">
        <TeamBadge teamId={game.awayTeamId} />
        <span className="game-header-team">{game.awayTeamName}</span>
      </div>

      <div className="game-header-middle">
        <span className="game-header-at">@</span>
        <span className="game-header-date">{game.gameDate}</span>
      </div>

      <div className="game-header-side game-header-side--home">
        <span className="game-header-team">{game.homeTeamName}</span>
        <TeamBadge teamId={game.homeTeamId} />
      </div>
    </header>
  );
}

function NotFound({ sport, league, gameId }) {
  return (
    <div className="game-detail">
      <Breadcrumb
        items={[
          { label: 'Predictions', to: '/predictions' },
          ...(sport ? [{ label: sport.label, to: `/predictions/${sport.slug}` }] : []),
          ...(sport && league
            ? [{ label: league.label, to: `/predictions/${sport.slug}/${league.slug}` }]
            : []),
          { label: 'Not found' },
        ]}
      />

      <div className="predictions-message">
        <h1 className="predictions-message-title">No such game</h1>
        <p className="predictions-message-body">
          {sport && league ? (
            <>
              Nothing predictable in <strong>{league.label}</strong> has the id{' '}
              <strong>{gameId}</strong>. A game only has a page while it is
              within reach of the models — one day past the newest data.
            </>
          ) : (
            <>That address does not name a league this app covers.</>
          )}
        </p>
      </div>
    </div>
  );
}

export default GameDetailPage;
