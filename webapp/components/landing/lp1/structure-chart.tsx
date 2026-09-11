'use client';

import { useTranslations } from 'next-intl';
import styles from './lp1.module.css';
import { CandleSvg } from './CandleSvg';
import { ZoneRect } from './ZoneRect';
import { buildCandles, priceBounds, yPct } from './chart';
import { DEMO_CLOSES, DEMO_LEVELS, type LayerKey } from './data';

/**
 * LP-3 — the layered structure chart, extracted from DemoTabs so the scroll
 * section and the "Lire une structure" demo share ONE drawing.
 *
 * This extraction is deliberate and load-bearing: the mission forbids building a
 * simplified second version of a demo that already exists. Whatever the reader
 * watches assemble itself while scrolling is, pixel for pixel, the chart they
 * then take control of — same candles, same bounds, same <ZoneRect>, same
 * scenario file (config/demo_illustration.json, guarded by
 * __tests__/demo-illustration-parity.test.ts).
 */

export type Layers = Record<LayerKey, boolean>;

export const ALL_ON: Layers = { ob: true, fvg: true, liq: true, str: true };
export const ALL_OFF: Layers = { ob: false, fvg: false, liq: false, str: false };

export const LEVEL_KEYS: readonly (keyof typeof DEMO_LEVELS)[] = [
  'obLow', 'obHigh', 'fvgLow', 'fvgHigh', 'liqIntact', 'liqSwept', 'chochLevel', 'bosLevel', 'currentPrice',
];

export function useBounds() {
  const candles = buildCandles(DEMO_CLOSES);
  return priceBounds(candles, LEVEL_KEYS.map((k) => DEMO_LEVELS[k]));
}

/** Shared structure chart — rendered full in the Structure pane, in the scroll
 * section, and compact in the MIA action preview, all bound to the same
 * `layers`. */
export function StructureChart({ layers }: { layers: Layers }) {
  const t = useTranslations('home');
  const b = useBounds();
  const L = DEMO_LEVELS;
  const lay = (on: boolean) => `${styles.lay} ${on ? styles.layOn : ''}`;
  return (
    <div className={styles.dchart}>
      <CandleSvg width={560} height={250} extra={LEVEL_KEYS.map((k) => L[k])} />

      <div className={lay(layers.ob)} aria-hidden={!layers.ob}>
        <ZoneRect
          kind="ob"
          style={{
            left: '60%', right: '64px', top: `${yPct(L.obHigh, b)}%`,
            height: `${yPct(L.obLow, b) - yPct(L.obHigh, b)}%`,
          }}
          labelStyle={{ left: '60.5%', top: `${yPct(L.obHigh, b) - 8}%` }}
          label={t('demo.structure.labels.ob')}
        />
      </div>

      <div className={lay(layers.fvg)} aria-hidden={!layers.fvg}>
        <ZoneRect
          kind="fvg"
          style={{
            left: '36%', right: '64px', top: `${yPct(L.fvgHigh, b)}%`,
            height: `${yPct(L.fvgLow, b) - yPct(L.fvgHigh, b)}%`,
          }}
          labelStyle={{ left: '36.5%', top: `${yPct(L.fvgHigh, b) - 8}%` }}
          label={t('demo.structure.labels.fvg')}
        />
      </div>

      <div className={lay(layers.liq)} aria-hidden={!layers.liq}>
        <div style={{ position: 'absolute', left: '10px', right: '64px', top: `${yPct(L.liqIntact, b)}%`, height: '1.5px', background: 'var(--liq)', opacity: 0.85 }} />
        <div className={styles.dl} style={{ left: '12px', top: `${yPct(L.liqIntact, b) - 8}%`, background: 'rgba(214,162,74,.16)', color: 'var(--liq)' }}>
          {t('demo.structure.labels.liqIntact')}
        </div>
        <div style={{ position: 'absolute', left: '10px', right: '64px', top: `${yPct(L.liqSwept, b)}%`, height: '1px', background: 'repeating-linear-gradient(90deg,var(--liq) 0 4px,transparent 4px 8px)', opacity: 0.5 }} />
        <div className={styles.dl} style={{ left: '12px', top: `${yPct(L.liqSwept, b) + 2}%`, background: 'rgba(214,162,74,.10)', color: 'var(--faint)' }}>
          {t('demo.structure.labels.liqSwept')}
        </div>
      </div>

      <div className={lay(layers.str)} aria-hidden={!layers.str}>
        <div className={styles.dl} style={{ left: '34%', top: `${yPct(L.chochLevel, b) - 4}%`, background: 'rgba(55,185,140,.16)', color: 'var(--bull)' }}>
          {t('demo.structure.labels.choch')}
        </div>
        <div className={styles.dl} style={{ left: '58%', top: `${yPct(L.bosLevel, b) - 10}%`, background: 'rgba(55,185,140,.16)', color: 'var(--bull)' }}>
          {t('demo.structure.labels.bos')}
        </div>
      </div>

      {/* the live price tag rides the price axis, so its `top` is computed */}
      <div className={styles.price} style={{ top: `${yPct(L.currentPrice, b) - 4}%` }}>
        4&nbsp;026,77
      </div>
    </div>
  );
}

/** The four layer chips — the REAL control (LP-2B §1: a promise of
 * interactivity must point at the real control, never at a shortcut). */
export function LayerChips({ layers, setLayers }: { layers: Layers; setLayers: (l: Layers) => void }) {
  const t = useTranslations('home');
  const chip = (k: LayerKey, color: string) => (
    <button
      key={k}
      type="button"
      className={`${styles.dchip} ${layers[k] ? styles.dchipOn : styles.dchipOff}`}
      aria-pressed={layers[k]}
      onClick={() => setLayers({ ...layers, [k]: !layers[k] })}
    >
      <span className={styles.sq} style={{ background: color }} />
      {t(`demo.structure.chips.${k}`)}
    </button>
  );
  return (
    <div className={styles.dchips}>
      {chip('str', 'var(--dim)')}
      {chip('ob', 'var(--bear)')}
      {chip('fvg', 'var(--fvg-l)')}
      {chip('liq', 'var(--liq)')}
    </div>
  );
}
