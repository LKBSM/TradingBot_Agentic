'use client';

import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { useTranslations } from 'next-intl';
import styles from './lp1.module.css';
import {
  ALL_ON,
  StructureChart,
  StructureNarration,
  type Layers,
} from './StructureVisual';
import type { LayerKey } from './data';

/**
 * LP-3 — the product window that plays by itself in the hero.
 *
 * It is the REAL structure visual (`StructureVisual`, shared with the demo
 * section below), not a lookalike: the same candles, the same four layer
 * families, the same narration recomposed from what is on. The arrival is a
 * presentation layer on top, and it obeys three rules:
 *
 *  1. **Reduced motion gets the finished state, immediately.** Every reveal is a
 *     CSS animation declared inside `@media (prefers-reduced-motion: no-preference)`
 *     (see `.stage*` in lp1.module.css), so a visitor who asked for less motion is
 *     never animated at all — not slowed down, not queued: the base rules already
 *     paint the finished window. The one JS-driven part (the typewriter) is
 *     skipped by the same check.
 *  2. **It never holds the interaction hostage.** The layer chips are live from
 *     the first paint. The first pointer, key or focus event anywhere in the
 *     window ends the arrival (`data-seq="done"` snaps every animation to its
 *     end) and the visitor's own action goes through on that same event.
 *  3. **No animated counters.** Every figure — the price tag, the levels — is
 *     rendered at its final value by the server. The typewriter *reveals* a
 *     sentence that is already composed; it never counts anything up.
 *
 * The server renders `data-seq="idle"`, which IS the finished state for anyone
 * without JS: only the CSS reveals (and a 2.4 s failsafe on the narration) hide
 * anything, and only while motion is allowed.
 */

/** The reading starts writing once the four layer families have landed. */
const TYPE_START_MS = 2500;
/**
 * The reading is typed in a FIXED duration rather than at a fixed speed per
 * character: the same sentence is 321 characters in Arabic and 449 in German,
 * and a per-character speed would make the arrival a second longer in some
 * languages than others — and drift out of step with the CSS delays that drive
 * the status label. One duration keeps the nine locales identical.
 */
const TYPE_DURATION_MS = 2800;

type Seq = 'idle' | 'typing' | 'done';

const NARRATION_ORDER: readonly LayerKey[] = ['str', 'ob', 'fvg', 'liq'];

export function HeroStage() {
  const t = useTranslations('home');
  const [layers, setLayers] = useState<Layers>({ ...ALL_ON });
  const [seq, setSeq] = useState<Seq>('idle');
  const [typed, setTyped] = useState(0);
  // Guards every scheduled step: once the visitor takes over, a timer still in
  // flight must not resurrect the sequence (nor re-hide the narration).
  const stopped = useRef(false);
  const timers = useRef<number[]>([]);

  /** The narration as PLAIN text, for the typewriter only. The finished state
   * renders the real rich component — this is a reveal, not a second source. */
  const plain = (key: string) => String(t.raw(key)).replace(/<[^>]+>/g, '');
  const fullText = [
    plain('demo.structure.prefix'),
    ...NARRATION_ORDER.filter((k) => layers[k]).map((k) => plain(`demo.structure.frag.${k}`)),
  ].join(' ');

  const stop = useCallback(() => {
    if (stopped.current) return;
    stopped.current = true;
    timers.current.forEach((id) => { window.clearTimeout(id); window.clearInterval(id); });
    timers.current = [];
    setSeq('done');
  }, []);

  useEffect(() => {
    // Rule 1 — asked for less motion: nothing is scheduled at all.
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      stopped.current = true;
      setSeq('done');
      return undefined;
    }
    const total = fullText.length;
    let tick = 0;
    timers.current.push(window.setTimeout(() => {
      if (stopped.current) return;
      setSeq('typing');
      const startedAt = Date.now();
      tick = window.setInterval(() => {
        if (stopped.current) { window.clearInterval(tick); return; }
        const ratio = (Date.now() - startedAt) / TYPE_DURATION_MS;
        if (ratio >= 1) {
          window.clearInterval(tick);
          setSeq('done');   // hands the paragraph back to the real rich component
          return;
        }
        setTyped(Math.ceil(ratio * total));
      }, 16);
      timers.current.push(tick);
    }, TYPE_START_MS));
    return () => {
      window.clearInterval(tick);
      timers.current.forEach((id) => window.clearTimeout(id));
      timers.current = [];
    };
    // Deliberately mount-only: the sequence is an arrival, it never replays.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** Rule 2 — the visitor's first move ends the arrival; the event itself is not
   * swallowed, it keeps travelling to the chip / link it was aimed at. */
  const yieldToVisitor = () => stop();

  const setLayer = (k: LayerKey) => {
    stop();
    setLayers((prev) => ({ ...prev, [k]: !prev[k] }));
  };

  const chip = (k: LayerKey, color: string) => (
    <button
      type="button"
      className={`${styles.dchip} ${layers[k] ? styles.dchipOn : styles.dchipOff}`}
      aria-pressed={layers[k]}
      onClick={() => setLayer(k)}
    >
      <span className={styles.sq} style={{ background: color }} />
      {t(`demo.structure.chips.${k}`)}
    </button>
  );

  const rich = (k: string): ReactNode => t.rich(k, { b: (c) => <b>{c}</b> });

  return (
    <div
      className={styles.stage}
      data-seq={seq}
      onPointerDownCapture={yieldToVisitor}
      onKeyDownCapture={yieldToVisitor}
      onFocusCapture={yieldToVisitor}
    >
      <div className={styles.stageTop}>
        {/* Three stacked labels crossfaded by CSS rather than one label swapped
            by JS: the server can then paint the FINAL one, so there is no flash
            of "lecture à jour" before the window rewinds to "se charge". The two
            transient ones are decorative; the final one is the readable truth. */}
        <span className={styles.stageStatus}>
          <span className={`${styles.stageSt} ${styles.stSt1}`} aria-hidden="true">{t('hero.stage.loading')}</span>
          <span className={`${styles.stageSt} ${styles.stSt2}`} aria-hidden="true">{t('hero.stage.reading')}</span>
          <span className={`${styles.stageSt} ${styles.stSt3}`}>{t('hero.stage.done')}</span>
        </span>
        <span className={styles.stageMkt}>{t('tools.app.visTitle')}</span>
      </div>

      <div className={styles.stageBody}>
        <div className={styles.stageChips}>
          {chip('ob', 'var(--bear)')}
          {chip('fvg', 'var(--fvg-l)')}
          {chip('liq', 'var(--liq)')}
          {chip('str', 'var(--dim)')}
        </div>
        <StructureChart layers={layers} stagger />
        <div className={styles.stageRead}>
          {seq === 'typing'
            ? <div className={styles.dnarr}>{fullText.slice(0, typed)}</div>
            : <StructureNarration layers={layers} />}
        </div>
      </div>

      {/* The exchange that says what the product refuses to do. Same strings as
          the M.I.A section below — one wording, one place to change it. */}
      <div className={styles.stageChat}>
        <div className={`${styles.miaBub} ${styles.mbU} ${styles.stageBubQ}`}>{t('miaSection.chat.u3')}</div>
        <div className={`${styles.miaBub} ${styles.mbNo} ${styles.stageBubA}`}>{rich('miaSection.chat.a3')}</div>
      </div>

      <div className={styles.stageIllus}>{t('demo.illus')}</div>
    </div>
  );
}
