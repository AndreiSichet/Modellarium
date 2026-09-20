import { Link } from 'react-router-dom';

import './Breadcrumb.css';

/**
 * A trail of links ending in plain text.
 *
 * THE LAST CRUMB IS NEVER A LINK. It is where the reader already is, and a
 * link that goes nowhere is the same dead control the league page omits its
 * shortcut bar to avoid. aria-current="page" says so to assistive tech.
 *
 * Extracted in Phase 6 at the second call site rather than the third: the
 * detail page's trail is the league page's plus one, so two copies would
 * have been two places to get the separator, the markup and the
 * aria-current rule right.
 */
function Breadcrumb({ items }) {
  return (
    <nav className="breadcrumb" aria-label="Breadcrumb">
      {items.map((item, index) => (
        <span className="breadcrumb-item" key={item.to || item.label}>
          {index > 0 ? (
            <span className="breadcrumb-sep" aria-hidden="true">
              /
            </span>
          ) : null}
          {item.to ? (
            <Link className="breadcrumb-link" to={item.to}>
              {item.label}
            </Link>
          ) : (
            <span className="breadcrumb-current" aria-current="page">
              {item.label}
            </span>
          )}
        </span>
      ))}
    </nav>
  );
}

export default Breadcrumb;
