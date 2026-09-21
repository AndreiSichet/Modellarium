import { NavLink } from 'react-router-dom';

import logo from '../assets/modellarium-logo.png';
import './AppShell.css';

function AppShell({ children }) {
  return (
    <div className="shell">
      <header className="shell-header">
        <div className="shell-header-inner">
          <NavLink to="/" className="wordmark">

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

      <div className="shell-body">{children}</div>
    </div>
  );
}

export default AppShell;
