import { NavLink } from 'react-router-dom';

import logo from '../assets/modellarium-logo.png';
import './AppShell.css';

/**
 * The frame every route renders inside: sticky header, centred main.
 *
 * NavLink rather than Link plus a pathname comparison. The router already
 * knows which route is active, and hand-rolling that check is how an
 * underline ends up correct on click and wrong after a hard reload — the
 * two paths through the same state.
 */
function AppShell({ children }) {
  return (
    <div className="shell">
      <header className="shell-header">
        <div className="shell-header-inner">
          <NavLink to="/" className="wordmark">
            {/*
              alt="" and aria-hidden on purpose. The mark already contains
              the word "Modellarium", and the text beside it carries the
              accessible name — without this a screen reader announces the
              same word twice in a row.

              The asset is white-on-black with the background baked in, so
              it renders as a black badge on cream. That is intended: it is
              not inverted, recoloured or cut out.
            */}
            <img src={logo} alt="" aria-hidden="true" className="wordmark-logo" />
            <span className="wordmark-text">MODELLARIUM</span>
          </NavLink>

          <nav className="shell-nav" aria-label="Main">
            <NavLink to="/" end className="shell-nav-link">
              About
            </NavLink>
            <NavLink to="/predictions" className="shell-nav-link">
              Predictions
            </NavLink>
          </nav>
        </div>
      </header>

      {/*
        A DIV, NOT <main>, and the routed page supplies its own.
        <main> has to wrap the PRIMARY content, and the predictions layout
        puts a <nav> (the sports rail) beside its content column — nesting
        that nav inside main would file page navigation as page content.
        Each route now renders exactly one <main> around the column that
        actually holds its content.
      */}
      <div className="shell-body">{children}</div>
    </div>
  );
}

export default AppShell;
