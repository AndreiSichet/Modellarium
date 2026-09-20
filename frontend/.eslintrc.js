/**
 * ESLint config.
 *
 * MOVED OUT OF package.json, and not copied. ESLint uses the
 * highest-priority config in a directory and ignores the rest, and
 * `.eslintrc.js` outranks package.json's `eslintConfig` — so leaving that
 * key behind would have left a dead config that still looked authoritative.
 * The move is also what makes this comment possible: JSON cannot hold one,
 * and the override below needs a reason attached to it.
 *
 * `react-app/jest` brings in eslint-plugin-testing-library. Note that
 * nothing in the normal workflow runs these rules over test files:
 * `react-scripts test` does not lint at all, and `react-scripts build`
 * only lints modules reachable from the entry point, which test files are
 * not. They surface in the editor and nowhere else — which is exactly why
 * they are worth settling rather than living with as permanent red.
 */
module.exports = {
  extends: ['react-app', 'react-app/jest'],

  overrides: [
    {
      files: ['**/*.test.js'],
      rules: {
        /*
         * TWO RULES OFF FOR TEST FILES, DELIBERATELY, AND SCOPED TO THEM.
         *
         * Both push toward querying the way a user does — getByRole,
         * getByText — and that is the right default. Every test here that
         * asserts BEHAVIOUR already does it: tabs, links, headings and
         * buttons are all reached by role.
         *
         * What these rules cannot express is an assertion about STRUCTURE,
         * and this suite has several that are deliberate:
         *
         *   - COUNTING ROWS. `.game-row`, `.overview-row`, `.player-board`
         *     have no distinguishing role. An unscoped
         *     getAllByRole('listitem') also matches the sports rail's own
         *     <li>, which is a real bug this suite hit in Phase 3 and
         *     fixed by scoping to the class.
         *
         *   - DOM ORDER. "the five soonest, in date order", "the market
         *     groups are Outcome/Totals/Rebounds/Assists in that order" —
         *     the claim is about sequence, and reading textContent off a
         *     NodeList says so directly.
         *
         *   - VERBATIM TEXT. Testing Library normalizes whitespace in the
         *     DOM text but not in the matcher string, so getByText cannot
         *     assert that the availability note is reproduced exactly. It
         *     would pass a paraphrase that normalized the same and fail an
         *     exact copy. Phase 6 switched that assertion TO node access
         *     for precisely this reason; turning it back would weaken the
         *     one test whose whole point is fidelity.
         *
         *   - ELEMENTS WITH NO ROLE AT ALL. `.wordmark-logo`,
         *     `.game-row-footer`, `.game-header` — there is no accessible
         *     query for a decorative image or a layout wrapper, and some
         *     of these tests exist to check that the element is hidden
         *     from assistive tech, which rules it out by construction.
         *
         * Scoped to test files by the `files` key above — which cannot be
         * repeated here, because the glob contains the two characters that
         * end a block comment. Application code never reaches into the DOM
         * this way, and if it ever starts to, these rules should still say
         * so.
         */
        'testing-library/no-node-access': 'off',
        'testing-library/no-container': 'off',
      },
    },
  ],
};
