'use client';

import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { useTranslations } from 'next-intl';
import styles from './lp1.module.css';
import {
  ALL_ON,
  LayerChips,
  StructureChart,
  type Layers,
} from './structure-chart';
import type { LayerKey } from './data';

/**
 * LP-3 — THE single orchestrated moment of the page: the reading writes itself
 * as you scroll.
 *
 * The claim this section makes is the product's central one, and the page used
 * to merely ASSERT it in prose ("recomposée à chaque clôture, jamais un texte
 * figé"). Here the reader watches it happen: each layer lands on the chart at
 * the moment its sentence arrives, in the order a structural reading is
 * actually built — break, then zone, then imbalance, then liquidity.
 *
 * Three rules this component keeps:
 *
 *  1. It reuses the REAL demo chart (see structure-chart.tsx), never a
 *     simplified copy. What you watch assemble is what you then control.
 *  2. Scroll only ever ADDS. Nothing is hidden behind the animation — reach the
 *     end and every layer is on, exactly as if you had never scrolled.
 *  3. It degrades to a plain stack. No IntersectionObserver, reduced motion, or
 *     a narrow screen → every sentence is rendered at once, all layers on, and
 *     the chips are live from the first paint. That is also the server-rendered
 *     markup, so a visitor without JS loses nothing but the choreography.
 *
 * The section ends by handing over: the chips below the chart are the product's
 * REAL control (LP-2B §1 — a promise of interactivity must point at the real
 * control, never at a shortcut button). The first click switches the section to
 * manual for good; scroll never steals the chart back.
 */

const STEPS: readonly LayerKey[] = ['str', 'ob', 'fvg', 'liq'];

/** Layers on at step `i`, cumulative: the reading only ever gains detail. */
function layersUpTo(i: number): Layers {
  const on: Layers = { ob: false, fvg: false, liq: false, str: false };
  STEPS.slice(0, i + 1).forEach((k) => { on[k] = true; });
  return on;
}

export function ReadingUnfolds({
  manual,
  setManual,
}: {
  /** Page-level manual override — set by a chip here, or by a M.I.A view action
   *  further down the page. Non-null means the reader (or the agent) is driving
   *  and the scroll position no longer decides. */
  manual: Layers | null;
  setManual: (l: Layers) => void;
}) {
  const t = useTranslations('home');
  const rich = (k: string): ReactNode => t.rich(k, { b: (c) => <b>{c}</b> });

  // Server render + every degraded case: the whole reading, all layers on.
  const [scrolly, setScrolly] = useState(false);
  const [active, setActive] = useState(STEPS.length - 1);
  const stepRefs = useRef<Array<HTMLLIElement | null>>([]);

  useEffect(() => {
    if (typeof IntersectionObserver === 'undefined') return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    // Below the sticky breakpoint the two columns stack; pinning there fights
    // the reader's thumb for no gain.
    if (!window.matchMedia('(min-width: 900px)').matches) return;
    setActive(0);
    setScrolly(true);
  }, []);

  useEffect(() => {
    if (!scrolly) return;
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (!e.isIntersecting) continue;
          const i = Number((e.target as HTMLElement).dataset.step);
          if (Number.isFinite(i)) setActive(i);
        }
      },
      // A band across the middle of the viewport: a sentence becomes "the
      // current one" when it reaches reading height, not when it peeks in.
      { rootMargin: '-45% 0px -45% 0px', threshold: 0 },
    );
    stepRefs.current.forEach((el) => el && io.observe(el));
    return () => io.disconnect();
  }, [scrolly]);

  // Manual control, once taken, wins over scroll position — permanently.
  const layers: Layers = manual ?? (scrolly ? layersUpTo(active) : ALL_ON);
  const setLayers = useCallback((l: Layers) => setManual(l), [setManual]);
  /** Sentences still legitimate once the reader drives: one per visible layer. */
  const shown = STEPS.filter((k) => layers[k]);

  return (
    <section id="lecture" className={styles.unfold}>
      <div className={styles.wrap}>
        <h2 className={styles.unfoldH}>{t('unfold.title')}</h2>
        <p className={styles.unfoldP}>{rich('unfold.lead')}</p>

        <div className={`${styles.unfoldGrid} ${scrolly ? styles.unfoldPinned : ''}`}>
          <div className={styles.unfoldStage}>
            <div className={styles.unfoldStageIn}>
              <StructureChart layers={layers} />
              <LayerChips layers={layers} setLayers={setLayers} />
              <p className={styles.unfoldHand}>
                {manual ? t('unfold.handTaken') : rich('unfold.hand')}
              </p>
              <div className={styles.illus}>{t('demo.illus')}</div>
            </div>
          </div>

          {/* Once the reader takes the chips, the promise made two lines above
              has to be kept: the reading describes ONLY what is still drawn. So
              the sentence of a layer that has been switched off is removed, not
              dimmed — and switching everything off yields the honest empty
              state rather than a paragraph about nothing. */}
          {manual ? (
            <div className={styles.unfoldSteps}>
              <p className={styles.unfoldPrefix}>{t('demo.structure.prefix')}</p>
              {shown.length === 0 ? (
                <p className={styles.unfoldEmpty}>{t('demo.structure.empty')}</p>
              ) : (
                shown.map((k) => (
                  <div key={k} className={styles.unfoldStep}>
                    <span className={styles.unfoldStepK}>{t(`unfold.step.${k}`)}</span>
                    <p className={styles.unfoldStepP}>{rich(`demo.structure.frag.${k}`)}</p>
                  </div>
                ))
              )}
            </div>
          ) : (
            <ol className={styles.unfoldSteps}>
              {STEPS.map((k, i) => (
                <li
                  key={k}
                  data-step={i}
                  ref={(el) => { stepRefs.current[i] = el; }}
                  className={`${styles.unfoldStep} ${scrolly && i === active ? styles.unfoldStepOn : ''}`}
                  aria-current={scrolly && i === active ? 'step' : undefined}
                >
                  <span className={styles.unfoldStepK}>{t(`unfold.step.${k}`)}</span>
                  <p className={styles.unfoldStepP}>{rich(`demo.structure.frag.${k}`)}</p>
                </li>
              ))}
            </ol>
          )}
        </div>
      </div>
    </section>
  );
}
