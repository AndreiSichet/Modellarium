import { Link } from 'react-router-dom';

import './Breadcrumb.css';

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
