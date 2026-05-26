import { InsightGallery } from '@/components/insight/InsightGallery';
import { SAMPLE_SIGNALS } from '@/lib/mocks';

/**
 * Landing demo — the meat of the page. Renders the bullish XAU M15 mock
 * with the History section opened by default so the PF + IC95% pépite is
 * the first visual a visitor sees. The two other mocks (EUR bear, XAU
 * neutral) are rendered as secondary examples below, all collapsed, to
 * give a sense of variability without flooding the landing.
 */
export function DemoSection() {
  const defaultOpenByIndex = new Map<
    number,
    ReadonlyArray<'history'>
  >([[0, ['history']]]);

  return (
    <section
      id="demo"
      aria-labelledby="demo-title"
      className="container-prose space-y-6 py-12 sm:py-16"
    >
      <header className="space-y-2">
        <h2
          id="demo-title"
          className="text-balance text-2xl font-semibold tracking-tight sm:text-3xl"
        >
          Voici à quoi ressemble une lecture.
        </h2>
        <p className="max-w-2xl text-muted-foreground">
          Un verdict en cinq secondes — puis tout le détail si vous voulez le
          voir. Cliquez sur « Demander à Sentinel » pour ouvrir le chatbot
          contextualisé sur la lecture.
        </p>
      </header>

      <InsightGallery
        signals={SAMPLE_SIGNALS}
        defaultOpenByIndex={defaultOpenByIndex}
      />
    </section>
  );
}
