import { useState } from 'react';

import { MarketGroup, Metric } from './GameTab';
import { formatWin } from '../predictionFormat';
import { abbrOf } from '../data/teams';
import { refusalText, scheduleReason } from '../refusalText';

import './NhlGameTab.css';

/**
 * Why a market is not offered, in words a reader can act on.
 *
 * KEYED ON THE CODE, NEVER ON THE PROSE. `totalsReasonCode` is the stable
 * half of the pair the manifest ships - the gate's false REGRESSED is the
 * recorded reason the code exists - so the sentence is chosen by the code
 * and the API's own `totalsDetail` is never what the page keys on. An
 * unknown code gets the generic line rather than the API's wording, for the
 * same reason the schedule rows do: prose written for a maintainer, shown to
 * a reader, is how a page ends up saying "no_skill_over_constant".
 */
const TOTALS_REASON = {
  no_skill_over_constant:
    'Our total-goals model did no better than always predicting the league '
    + 'average, so we don’t show it.',
};

const TOTALS_REASON_GENERIC =
  'Total goals are not shown for this league.';

/**
 * Which model produced a number, written out.
 *
 * THE WIRE SENDS A FAMILY NAME, not a sentence: the NHL's winner is `elo`,
 * which rendered as the bare word "elo" in the corner of the row and told a
 * reader nothing. Mapped here rather than on the wire because it is a
 * display decision, and an unmapped family still says something true.
 */
const MODEL_NAME = {
  elo: 'team Elo ratings',
  logistic_elo: 'a logistic model over team Elo ratings',
};

function modelLabel(family) {
  if (!family) return null;
  return `Model: ${MODEL_NAME[family] || family}`;
}

/**
 * The NHL's own markets, because they are not the shape `GameTab` renders.
 *
 * A SEPARATE TAB RATHER THAN FLAGS ON GameTab, following QuarterHalfTab and
 * PlayerStatTab. `GameTab` reads `homeMargin` and `totalPoints`; the NHL has
 * neither. It has a winner, a puck line defined as "home wins by 2 or more",
 * and totals it deliberately does not serve. Three capability flags on a tab
 * that then renders none of its own rows is a worse seam than a second tab.
 *
 * THREE THINGS THIS TAB HAS TO GET RIGHT, all of them about honesty:
 *
 *   - the winner is a WEAK SIGNAL - Elo alone, and its test interval
 *     crosses zero - so the short caveat sits beside the number and the
 *     reader caveat is one click away. Both come from the API.
 *   - the puck line is "home wins by 2 or more", NOT a +/-1.5 line, and
 *     there is no away-side line because the model does not predict one.
 *   - totals are NOT OFFERED, and that is stated rather than hidden, so a
 *     reader knows it is a decision and not a gap.
 */
function NhlGameTab({ game, health, league, healthCaveat }) {
  const [showReader, setShowReader] = useState(false);
  const [showTotalsDetail, setShowTotalsDetail] = useState(false);
  const prediction = game.prediction;

  if (!prediction) {
    return <NhlNoPrediction game={game} health={health} league={league} />;
  }

  const home = abbrOf(game.homeTeamId, game.homeTeamAbbr);
  const away = abbrOf(game.awayTeamId, game.awayTeamAbbr);
  const homeName = game.homeTeamName || home;

  // THE API'S WORDS, NEVER A COPY HERE. The short caveat travels on the
  // prediction; the reader paragraph is on /health, because the schedule
  // page needs it before any prediction has been fetched. Either may be
  // absent from an older backend, and then nothing is claimed.
  const shortCaveat = prediction.winnerCaveat;
  const readerCaveat = healthCaveat?.reader;

  return (
    <div className="market-groups">
      <MarketGroup title="Outcome">
        <div className="market-row">
          <dt className="market-label">Win probability</dt>
          <dd className="market-value">
            <span className="nhl-win-pair">
              <span>{home} {formatWin(prediction.homeWinProbability, true)}</span>
              <span>{away} {formatWin(prediction.homeWinProbability, false)}</span>
            </span>
            {/* IMMEDIATELY BESIDE THE NUMBER, not in a footer. A reader who
                sees 70% and scrolls past a footnote has been told the wrong
                thing. */}
            {shortCaveat ? (
              <span className="market-note nhl-caveat-short">
                {shortCaveat}
              </span>
            ) : null}
            {readerCaveat ? (
              <>
                <button
                  type="button"
                  className="nhl-caveat-toggle"
                  aria-expanded={showReader}
                  onClick={() => setShowReader((open) => !open)}
                >
                  What this means
                </button>
                {showReader ? (
                  <p className="nhl-caveat-reader">{readerCaveat}</p>
                ) : null}
              </>
            ) : null}
            {prediction.winnerModel ? (
              <span className="market-model nhl-model">
                {modelLabel(prediction.winnerModel)}
              </span>
            ) : null}
          </dd>
        </div>

        {/* LABELLED FROM ITS DEFINITION, so the row cannot say one thing
            while the model predicts another. `HOME_MARGIN >= 2` is what the
            manifest ships; the label below is that, in English, with the
            home team named. No away-side line: the model does not predict
            one, and inventing 1 - p would assert a market that is not
            served.

            AND NO SECOND COPY OF IT UNDERNEATH. This carried a note reading
            "Defined as HOME_MARGIN >= 2." - the same fact twice, once in
            English and once in a column name from a file the reader cannot
            see. The label is the definition; a note repeating it in the
            pipeline's vocabulary adds nothing a reader can use. */}
        <Metric
          // "Carolina Hurricanes wins" - a team name is plural
          // here almost always, so the verb agrees with it. Same
          // family as the possessive in refusalText, and found the
          // same way: by reading the rendered page.
          label={`${homeName} ${homeName.endsWith('s') ? 'win' : 'wins'} by 2 or more goals`}
          value={formatWin(prediction.homeWinsByTwoOrMore, true)}
        />
      </MarketGroup>

      {/* NOT HIDDEN. A missing Totals group would read as an oversight; this
          says it is a decision and gives the reason. `totalsServed` is a
          boolean on the wire, so the false case is a state rather than an
          absence.

          THE SENTENCE IS CHOSEN BY THE CODE, NOT COPIED FROM THE API. It
          used to render `totalsDetail`, which is written for a maintainer:
          "a constant predictor beats the fitted model on test" is true and
          is not a sentence a reader can do anything with. The API's wording
          is still available, one click away, so nothing is lost. */}
      <MarketGroup title="Totals">
        <div className="market-row market-row--unserved">
          <dt className="market-label">Total goals</dt>
          <dd className="market-value">
            <span className="nhl-not-offered">Not offered</span>
            <span className="market-note nhl-totals-why">
              {TOTALS_REASON[prediction.totalsReasonCode]
                || TOTALS_REASON_GENERIC}
            </span>
            {prediction.totalsDetail ? (
              <>
                <button
                  type="button"
                  className="nhl-caveat-toggle"
                  aria-expanded={showTotalsDetail}
                  onClick={() => setShowTotalsDetail((open) => !open)}
                >
                  The measurement
                </button>
                {showTotalsDetail ? (
                  <p className="nhl-caveat-reader">
                    {prediction.totalsDetail}
                  </p>
                ) : null}
              </>
            ) : null}
          </dd>
        </div>
      </MarketGroup>

      <p className="market-freshness nhl-footnote">
        {prediction.stale ? <span className="stale-badge">STALE</span> : null}
        Computed from data as of {prediction.dataAsOf}
        {prediction.season ? ` · ${prediction.season} season` : null}
        {prediction.predictionRule
          ? ` · predictable once ${prediction.predictionRule}`
          : null}
      </p>
    </div>
  );
}

/**
 * A direct URL to a fixture this league will not predict.
 *
 * THE SCHEDULE'S OWN SENTENCE, not raw JSON and never "the service did not
 * answer". The service answered and refused, which is a different fact - the
 * early-season lesson. `scheduleReason` is tried first because a refused
 * fixture reached here from a schedule row that already carried the code and
 * the blockers; `refusalText` covers a refusal that arrived from the
 * prediction call itself.
 */
function NhlNoPrediction({ game, health, league }) {
  const why = scheduleReason(game) ?? refusalText(game.refusal);

  return (
    <div className="no-prediction">
      <p className="no-prediction-lead">
        This game cannot be predicted yet.
      </p>
      {why ? <p className="no-prediction-why">{why}</p> : null}
      <p className="no-prediction-rule">
        {league?.label || 'This league'} predictions appear once both teams’
        previous results are confirmed.
      </p>
    </div>
  );
}

export default NhlGameTab;
