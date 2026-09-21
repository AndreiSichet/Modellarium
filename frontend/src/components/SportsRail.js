import { NavLink } from 'react-router-dom';

import './SportsRail.css';

function BasketballIcon() {
  return (
    <svg
      className="sport-icon"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      aria-hidden="true"
      focusable="false"
    >
      <circle cx="12" cy="12" r="9" />
      <path d="M12 3v18" />
      <path d="M3 12h18" />
      <path d="M5.64 5.64a9 9 0 0 0 0 12.72" />
      <path d="M18.36 5.64a9 9 0 0 1 0 12.72" />
    </svg>
  );
}

export const SPORTS = [
  { slug: 'basketball', label: 'Basketball', icon: BasketballIcon },
];

export function findSport(slug) {
  return SPORTS.find((sport) => sport.slug === slug) || null;
}

function SportsRail() {
  return (
    <nav className="sports-rail" aria-label="Sports">
      <h2 className="sports-rail-heading">Sports</h2>

      <ul className="sports-rail-list">
        {SPORTS.map(({ slug, label, icon: Icon }) => (
          <li key={slug}>
            <NavLink to={`/predictions/${slug}`} className="sports-rail-link">
              <Icon />
              <span>{label}</span>
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  );
}

export default SportsRail;
