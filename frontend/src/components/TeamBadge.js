import { badgeTextColour, teamFor } from '../data/teams';
import './TeamBadge.css';

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
