import { useCallback, useEffect, useRef, useState } from 'react';

import './LeagueShortcuts.css';

/**
 * THE STICKY OFFSET IS THE HEADER PLUS THIS ROW, and that is a conclusion
 * about the ancestor chain rather than a value carried over from Phase 4.
 *
 * In the overview column this row lived inside a panel with its own
 * `overflow-y`, so it stuck at `top: 0` of that panel and the observer
 * watched the panel. Here the chain is
 *
 *   .shell-body > .predictions > .predictions-content > this
 *
 * and not one of those scrolls - checked, not assumed. So the scrolling
 * ancestor is the document, the row sticks below the app header at
 * `top: var(--header-height)`, and the observer's root is the viewport
 * (null). Copying either of the old values would have been wrong.
 */
function stickyOffset(chipRow) {
  const root = getComputedStyle(document.documentElement);
  // The header height is a token; the row's height is whatever it rendered
  // at. Both are read rather than restated, so neither can drift from the
  // stylesheet. A missing token degrades to 0, which makes the active chip
  // flip slightly early - wrong by a little, rather than wrong by a lot.
  const header = parseInt(root.getPropertyValue('--header-height'), 10) || 0;
  return header + (chipRow ? chipRow.offsetHeight : 0);
}

/**
 * The shortcut bar above the league subsections.
 *
 * SHORTCUTS, NOT TABS. Clicking scrolls the page to a subsection; it does
 * not change what is rendered. That is why these are buttons with
 * aria-current="location" rather than a tablist - a tab claims to show and
 * hide a panel, and these show everything at once.
 *
 * Phase 3's LeagueTabs really were tabs and are deleted. This is Phase 4's
 * overview chip row, moved into the main column: the scroll-spy, the
 * IntersectionObserver guard and the shared --chips-height all came with
 * it, because all three were right and only their container changed.
 */
function LeagueShortcuts({ sections, active, onActivate }) {
  const chipsRef = useRef(null);

  const scrollTo = useCallback(
    (id) => {
      const element = document.getElementById(id);
      // Optional-called: jsdom does not implement scrollIntoView, and a
      // click in a test must not throw. WHERE it lands is handled by
      // scroll-margin-top in the stylesheet, not by an offset here.
      element?.scrollIntoView?.({ behavior: 'smooth', block: 'start' });
      // Set it now rather than waiting for the observer: a smooth scroll
      // takes a few hundred milliseconds, and a chip that does not light up
      // until it finishes reads as an unresponsive button.
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

/**
 * Tracks which subsection is in view and returns its id.
 *
 * A hook rather than state inside the bar, because the sections it watches
 * are the bar's SIBLINGS, not its children - the page owns both, so the
 * page owns the answer.
 *
 * jsdom implements neither layout nor IntersectionObserver, so the guard
 * below is taken in every test and in any browser old enough to matter. The
 * fallback is the first section: the bar still renders, the first chip is
 * still underlined, the clicks still scroll, and nothing throws.
 */
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
        // The topmost section still in view, in document order. If the
        // negative margin has pushed every section out, keep the last
        // answer rather than blanking the bar.
        const topmost = order.find((id) => visible.has(id));
        if (topmost) setActiveId(topmost);
      },
      {
        // null: the viewport. See stickyOffset above for why this is not
        // an element here even though it was in Phase 4.
        root: null,
        /*
         * Shrink the observed box down from the top by the header plus the
         * shortcut bar, so a section sitting UNDERNEATH that chrome does
         * not count as visible. Without it the next section activates while
         * its heading is still hidden behind the bar.
         *
         * The bottom -55% stops a section that has barely appeared from the
         * bottom edge taking the underline off the one being read.
         */
        rootMargin: `-${offset}px 0px -55% 0px`,
        threshold: 0,
      }
    );

    for (const element of elements) observer.observe(element);
    return () => observer.disconnect();
  }, [idsKey]);

  /*
   * DERIVED, NOT RESET IN AN EFFECT. Navigating between General and a sport
   * page changes the section list, and a stored id can name a section that
   * no longer exists. Falling back at render time means there is never a
   * frame with nothing underlined, and no second render to correct one.
   */
  const active = ids.includes(activeId) ? activeId : ids[0];
  return [active, setActiveId];
}

export default LeagueShortcuts;
