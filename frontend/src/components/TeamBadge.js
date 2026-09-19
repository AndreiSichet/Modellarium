import { badgeTextColour, teamFor } from '../data/teams';
import './TeamBadge.css';

/**
 * A monogram in the team's colours, standing in for a logo.
 *
 * The two colours come from data/teams.js — the one file allowed to hold
 * hex values, because published brand colours are facts about the world
 * rather than design tokens. Everything else here uses tokens.
 *
 * aria-hidden: the badge repeats the abbreviation, and the team's full name
 * sits beside it in the row. Announcing "BOS Boston Celtics" is the same
 * double-read the header logo avoids.
 */
function TeamBadge({ teamId }) {
  const team = teamFor(teamId);

  return (
    <span
      className="team-badge"
      style={{ background: team.primary, color: badgeTextColour(team) }}
      aria-hidden="true"
    >
      {team.abbr}
    </span>
  );
}

export default TeamBadge;
