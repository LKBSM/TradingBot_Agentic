'use client';

import { type ReactNode } from 'react';
import { useTranslations } from 'next-intl';
import styles from './lp1.module.css';
import { MiaPane } from './MiaPane';
import type { Layers } from './structure-chart';

/**
 * LP-3 — the refusal block.
 *
 * What it replaces: a section that introduced M.I.A in prose, proved her with a
 * SCRIPTED chat, and then re-explained the same four capabilities in four cards
 * with four stock icons — a book, a magnifying glass, some bars, a warning
 * triangle. Three tellings of one thing, the middle one a fake.
 *
 * What it does instead: states the one claim a competitor cannot copy by
 * shipping a feature — she refuses to guess — and then hands the visitor the
 * REAL agent to try to break. There is no scripted chat left on this page.
 *
 * The four capability sentences survive verbatim (LP-2A cut each to a single
 * factual sentence and locked them with lp2a-copy.test.ts across nine locales;
 * their content is the part the demo does NOT carry — the scope list, the
 * grounding mechanism, the command examples, the three refusals). LP-3 changes
 * only how they are rendered: running text under the exchange, no icons, no
 * card grid.
 */
export function RefusalBlock({
  layers,
  setLayers,
  jumpToReading,
}: {
  layers: Layers;
  setLayers: (l: Layers) => void;
  jumpToReading: () => void;
}) {
  const t = useTranslations('home');
  const rich = (k: string): ReactNode => t.rich(k, { b: (c) => <b>{c}</b> });

  return (
    <section id="mia" className={styles.refuse}>
      <div className={styles.wrap}>
        <h2 className={styles.secH}>{t('refuse.title')}</h2>
        {/* LP-2A ruled this paragraph explicitly out of scope for trimming: it
            carries the concrete things a visitor might ask (« why is this candle
            an Order Block », « what does mitigation mean ») that no demo shows.
            LP-3 moves it here verbatim — the section around it changed, the
            sentence did not. Its guard still runs. */}
        <p className={styles.secP}>{rich('miaSection.p')}</p>
        <p className={styles.secP}>{t('refuse.lead')}</p>

        <MiaPane layers={layers} setLayers={setLayers} jumpToReading={jumpToReading} />

        <ul className={styles.caps}>
          {(['c1', 'c2', 'c3', 'c4'] as const).map((c) => (
            <li key={c} className={styles.cap}>
              <b className={styles.capH}>{t(`miaSection.caps.${c}.h`)}</b>{' '}
              <span className={styles.capP}>{rich(`miaSection.caps.${c}.p`)}</span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
