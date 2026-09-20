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

/**
 * /predictions/:sport/:league/:gameId - everything known about one game.
 *
 * THREE DOMAINS, TWO FETCH TIMES, AND ONE REUSE.
 *
 *   Game              reused from the layout - see below
 *   Quarters & Halves fetched on mount
 *   Player tabs       fetched on the first visit to any of them, then cached
 *
 * THE GAME TAB DOES NOT FETCH, AND THAT IS A DELIBERATE DEPARTURE from the
 * obvious arrangement. The layout already called POST /api/predictions for
 * every predictable game in order to build the list this page was reached
 * from, so its seven markets are already in hand. Calling it again would
 * return the same seven numbers AND write a second row - the endpoint is
 * append-only by design, so a refetch is not free the way a GET would be.
 *
 * The page is still addressable directly: the layout fetches on mount
 * whatever the route, so a pasted URL populates `games` before this
 * renders, and a gameId that is not in it is genuinely not found.
 *
 * SWITCHING TABS MUST NEVER REFETCH. Each domain's status is held here, at
 * the page, rather than inside the tab that shows it - a tab that owned its
 * own fetch would restart it every time it was unmounted and remounted, and
 * browsing would turn into row-writing.
 */
function GameDetailPage() {
  const { sport: sportSlug, league: leagueSlug, gameId } = useParams();
  const { games, health } = useOutletContext();

  const sport = findSport(sportSlug);
  const league = findLeague(sportSlug, leagueSlug);
  // String compare: the id is a path segment, so it arrives as text.
  const game = games.find((entry) => String(entry.gameId) === String(gameId));

  const [tab, setTab] = useState(TABS[0].id);
  const [quarterHalf, setQuarterHalf] = useState({ status: 'loading', data: null, error: null });
  const [playerProps, setPlayerProps] = useState({ status: 'idle', data: null, error: null });

  /*
   * THREE PRIMITIVES, NOT ONE OBJECT. A `payload` object is a fresh value
   * every render, so depending on it re-runs the effects forever. Packing
   * the three into one string would have worked too, except that gameDate
   * is itself hyphenated - "...-2026-04-13".split('-') does not give three
   * parts back, and the unpacked date would have been "2026".
   */
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
      // QuarterHalfSummaryDto nests its markets under `prediction`, where
      // GameSummaryDto uses `latestPrediction`. Unwrapped once, here, so
      // the tab below never has to know which shape it came from.
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

  /*
   * LAZY, AND THE CONDITION IS `idle` RATHER THAN `!data`. Twenty players
   * by five stats is heavier than the other two calls combined, and a
   * visitor who only wanted the spread should not pay for it. Keying off
   * the absence of data would re-run the call after a failure every time
   * the reader touched a player tab; `idle` fires exactly once, and the
   * error state carries its own retry.
   */
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

      {/*
        One panel, swapped by tab, rather than seven kept mounted and
        hidden. The data all lives above, so nothing is lost by unmounting a
        panel - and seven mounted panels would put twenty player rows in the
        accessibility tree five times over for tabs nobody is looking at.
      */}
      <div
        className="detail-panel"
        role="tabpanel"
        id={panelId(tab)}
        // Named by its own tab rather than by a duplicate aria-label, so
        // the two cannot come to say different things.
        aria-labelledby={tabId(tab)}
        // 0, not -1: the panel holds no focusable element of its own on
        // most tabs, so without this a keyboard user tabbing out of the bar
        // skips the content entirely and lands in the footer.
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

/**
 * NO TIP-OFF TIME, and its absence is a fact about the data rather than an
 * omission. ScheduledGameDto carries gameDate as a LocalDate and the
 * inference service's /schedule returns game_date alone - there is no time
 * anywhere in the chain, so rendering one would mean inventing it.
 */
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

/**
 * Inline, with the rail and a breadcrumb intact - the same behaviour an
 * unknown sport has had since Phase 2 and an unknown league since Phase 5.
 * Redirecting would hide which part of the URL was wrong; this can say.
 */
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
