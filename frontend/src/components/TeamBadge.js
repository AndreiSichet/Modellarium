import { badgeTextColour, teamFor } from '../data/teams';
import './TeamBadge.css';

/**
 * A team's colours, with its abbreviation as the label.
 *
 * COLOURS COME FROM `teams.js` KEYED BY ID; THE LABEL MAY COME FROM THE API.
 * The G League's 31 franchises are not in `teams.js` - published colours for
 * them are not something to invent, and this project already labels the
 * WNBA's two expansion colours provisional for the same reason - so they
 * resolve to the neutral badge. Without an abbreviation from the caller that
 * badge would read "?" for every G League team; `ScheduledGameDto` carries
 * `homeTeamAbbr`, so it reads "OKL" and still identifies the team.
 *
 * The lookup stays by id and never by abbreviation: eight abbreviations
 * belong to both an NBA and a WNBA team, so resolving by abbreviation would
 * paint the Atlanta Dream in the Hawks' colours.
 */
function TeamBadge({ teamId, abbr }) {
  const team = teamFor(teamId);

  return (
    <span
      className="team-badge"
      style={{ background: team.primary, color: badgeTextColour(team) }}
      aria-hidden="true"
    >
      {abbr || team.abbr}
    </span>
  );
}

export default TeamBadge;
