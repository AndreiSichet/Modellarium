import './Attribution.css';

/**
 * The licence line the NFL's data requires.
 *
 * REQUIRED, NOT DECORATIVE. The NFL history is CC BY-SA 4.0 from English
 * Wikipedia, and that licence asks for attribution wherever the data is
 * shown - so this appears on the NFL league page and on every NFL game
 * detail page, as a visible line rather than a tooltip.
 *
 * THE WORDS COME FROM THE API, NOT FROM HERE. `source` travels on every NFL
 * prediction body as "English Wikipedia, CC BY-SA 4.0", and this renders
 * THAT - publisher and licence both - so the line cannot drift from what the
 * backend says it is serving. The first version of this component hardcoded
 * the whole sentence and merely checked `source` was present, which would
 * have gone on claiming Wikipedia and CC BY-SA 4.0 after a backend change to
 * either.
 *
 * The two URLs are the only hardcoded part, because a URL is not something
 * the API sends. They are attached by matching the API's own words, so a
 * source naming a different publisher or licence renders unlinked rather
 * than pointing at the wrong licence text.
 *
 * Nothing renders at all without a source, rather than a licence claim this
 * page cannot substantiate.
 */
const LINKS = [
  [/wikipedia/i, 'https://en.wikipedia.org/'],
  [/cc by-sa 4\.0/i, 'https://creativecommons.org/licenses/by-sa/4.0/'],
];

function Attribution({ source }) {
  if (!source) return null;

  // "English Wikipedia, CC BY-SA 4.0" -> publisher, licence. An unexpected
  // shape is rendered whole rather than forced into the sentence.
  const parts = String(source).split(',');
  const publisher = parts[0]?.trim();
  const licence = parts.slice(1).join(',').trim();

  return (
    <p className="attribution">
      {publisher && licence ? (
        <>
          NFL game data from {linked(publisher)}, licensed {linked(licence)}.
        </>
      ) : (
        <>NFL game data from {linked(String(source))}.</>
      )}
    </p>
  );
}

/** The API's own words, linked where they name something with a URL. */
function linked(text) {
  const match = LINKS.find(([pattern]) => pattern.test(text));
  if (!match) return text;

  return (
    <a href={match[1]} rel="noreferrer" target="_blank">
      {text}
    </a>
  );
}

export default Attribution;
