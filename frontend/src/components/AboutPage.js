import './AboutPage.css';

/**
 * The landing route.
 *
 * THE COPY IS SPORT-AGNOSTIC ON PURPOSE. An earlier version named eleven
 * seasons, quarter-by-quarter scoring and gradient-boosted trees — all true
 * of today's basketball implementation and all of it wrong the moment a
 * second sport or a different model family arrives. It describes the idea,
 * not the current instance, so it does not need editing to stay accurate.
 *
 * TWO PHRASES ARE DELIBERATE AND SHOULD NOT BE SMOOTHED:
 *
 * "is not worse" in the last block describes the promotion gate's real
 * behaviour — it is a regression guard, not an improvement bar. "only if
 * it's better" reads more confident and would be a stronger claim than the
 * system actually makes.
 *
 * The third block stays because the UI depends on it: the low-confidence
 * tags and the availability-unknown notices exist precisely for the reason
 * it gives, and the copy and the interface should agree.
 */
const BLOCKS = [
  {
    heading: 'How predictions are made',
    body:
      'Every prediction starts with what already happened. Modellarium ' +
      'ingests completed games and turns them into a picture of form going ' +
      'into a fixture — recent performance, rest, the strength of the ' +
      'opponent, who is unavailable. Models trained on that history then ' +
      'predict what comes next.',
  },
  {
    heading: 'Choosing a model',
    body:
      'Different questions call for different models. Some outcomes are best ' +
      'served by complex models, others by simple ones, and a more ' +
      'sophisticated approach is not automatically a better one. Every choice ' +
      'here was made by measuring the alternatives against real results ' +
      'rather than by preference.',
  },
  {
    heading: "What it doesn't claim",
    body:
      'A prediction is a best estimate, not a certainty, and some are far ' +
      'stronger than others. Where a prediction is weak, Modellarium says so ' +
      'rather than presenting every number with the same confidence. Where ' +
      'information is missing, it reports that too, instead of quietly ' +
      'assuming the best case.',
  },
  {
    heading: 'Staying current',
    body:
      'The models are not fixed. As new results come in, the data refreshes, ' +
      'the models retrain, and each new version is compared against the one ' +
      'currently deployed. A new model only replaces the current one if it is ' +
      'not worse. Nothing ships on the assumption that more recent is better.',
  },
];

function AboutPage() {
  return (
    <div className="about">
      <section className="about-hero">
        <h1 className="about-hero-title">
          Computing sports predictions for the love of the game
        </h1>
        <p className="about-hero-lede">
          Modellarium is a sportsbook built as a personal project, computing
          predictions for various sports and rubrics using advanced ML
          techniques.
        </p>
      </section>

      <hr className="about-rule" />

      <section className="about-prose">
        <h2 className="about-prose-title">How Modellarium works</h2>

        <div className="about-blocks">
          {BLOCKS.map((block) => (
            <article className="about-block" key={block.heading}>
              <h3 className="about-block-heading">{block.heading}</h3>
              <p className="about-block-body">{block.body}</p>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

export default AboutPage;
