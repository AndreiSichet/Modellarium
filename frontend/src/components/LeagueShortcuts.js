import { useCallback, useEffect, useRef, useState } from 'react';

import './LeagueShortcuts.css';

function stickyOffset(chipRow) {
  const root = getComputedStyle(document.documentElement);

  const header = parseInt(root.getPropertyValue('--header-height'), 10) || 0;
  return header + (chipRow ? chipRow.offsetHeight : 0);
}

function LeagueShortcuts({ sections, active, onActivate }) {
  const chipsRef = useRef(null);

  const scrollTo = useCallback(
    (id) => {
      const element = document.getElementById(id);

      element?.scrollIntoView?.({ behavior: 'smooth', block: 'start' });

      onActivate(id);
    },
    [onActivate]
  );

  if (sections.length === 0) return null;

  return (
    <nav
      className="league-shortcuts"
      ref={chipsRef}
      aria-label="Leagues on this page"
    >
      {sections.map((section) => (
        <button
          key={section.id}
          type="button"
          className={
            section.id === active
              ? 'league-shortcut league-shortcut--active'
              : 'league-shortcut'
          }
          aria-current={section.id === active ? 'location' : undefined}
          onClick={() => scrollTo(section.id)}
        >
          {section.league.label}
        </button>
      ))}
    </nav>
  );
}

export function useActiveSection(ids) {
  const idsKey = ids.join('|');
  const [activeId, setActiveId] = useState(null);

  useEffect(() => {
    if (typeof IntersectionObserver === 'undefined') return undefined;

    const order = idsKey ? idsKey.split('|') : [];
    if (order.length === 0) return undefined;

    const elements = order
      .map((id) => document.getElementById(id))
      .filter(Boolean);
    if (elements.length === 0) return undefined;

    const visible = new Set();
    const offset = stickyOffset(document.querySelector('.league-shortcuts'));

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) visible.add(entry.target.id);
          else visible.delete(entry.target.id);
        }

        const topmost = order.find((id) => visible.has(id));
        if (topmost) setActiveId(topmost);
      },
      {
        root: null,

        rootMargin: `-${offset}px 0px -55% 0px`,
        threshold: 0,
      }
    );

    for (const element of elements) observer.observe(element);
    return () => observer.disconnect();
  }, [idsKey]);

  const active = ids.includes(activeId) ? activeId : ids[0];
  return [active, setActiveId];
}

export default LeagueShortcuts;
