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

function AmericanFootballIcon() {
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
      <path d="M4.5 19.5c-1.5-5 .5-10.5 4.5-13.5s9.5-3 13.5-1.5c1.5 5-.5 10.5-4.5 13.5s-9.5 3-13.5 1.5Z" />
      <path d="M9 15l6-6" />
      <path d="M10.5 11.25l1.5 1.5" />
      <path d="M12.75 9l1.5 1.5" />
    </svg>
  );
}

export const SPORTS = [
  { slug: 'basketball', label: 'Basketball', icon: BasketballIcon },
  // "American football", never "football": for a European reader that word
  // means soccer, and the slug becomes a URL.
  {
    slug: 'american-football',
    label: 'American football',
    icon: AmericanFootballIcon,
  },
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
