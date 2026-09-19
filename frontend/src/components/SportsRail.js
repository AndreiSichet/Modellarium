import { NavLink } from 'react-router-dom';

import './SportsRail.css';

/**
 * An inline SVG rather than an image file: it inherits currentColor, so the
 * active/inactive colour change needs no second asset, it scales without a
 * second resolution, and there is nothing to keep in sync on disk.
 *
 * Line art rather than a filled or photographic mark — the design language
 * here is hairline rules and flat cream, and a solid icon reads as a
 * different system immediately.
 *
 * The two side arcs are drawn between points that genuinely sit on the
 * circle (distance 9 from centre at 135° and 225°), so the seams meet the
 * ball's edge rather than floating near it.
 */
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

/**
 * THE ONE PLACE A SPORT IS DECLARED. Adding one is an entry here, not a
 * component edit — which is the whole reason this is a constant rather
 * than JSX written out in the rail.
 *
 * Exactly one entry today, and deliberately no placeholders for sports
 * that do not exist. A navigation rail listing things that cannot be
 * clicked is worse than a short rail: it promises something the app does
 * not have.
 */
export const SPORTS = [
  { slug: 'basketball', label: 'Basketball', icon: BasketballIcon },
];

export function findSport(slug) {
  return SPORTS.find((sport) => sport.slug === slug) || null;
}

/**
 * The left rail.
 *
 * SCROLLS INDEPENDENTLY OF THE PAGE, and the CSS that achieves it is
 * easier to get subtly wrong than it looks — see SportsRail.css, where
 * `height` rather than `max-height` is the load-bearing choice.
 */
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
