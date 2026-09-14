'use client';

import { useTranslations } from 'next-intl';
import { type ReactNode } from 'react';
import styles from './lp1.module.css';
import { CandleSvg } from './CandleSvg';
import { ZoneRect } from './ZoneRect';
import { buildCandles, priceBounds, yPct } from './chart';
import { DEMO_CLOSES, DEMO_LEVELS, type LayerKey } from './data';

/**
 * The structure visual, shared by the two surfaces that draw it: the interactive
 * demo (DemoTabs) and the hero stage (HeroStage). It used to live inside
 * DemoTabs; LP-3 lifted it out so the hero could reuse the REAL component rather
 * than re-implement a chart that looks like it.
 *
 * Nothing here knows about the hero's arrival sequence: the chart draws whatever
 * `layers` says, the narration recomposes from the same object. The sequence is
 * pure CSS on top (see `.stage*` in lp1.module.css) plus one typewriter effect —
 * so a visitor who unticks a layer mid-arrival is served by the same code path
 * as one who unticks it a minute later.
 */

export type Layers = Record<LayerKey, boolean>;
export const ALL_ON: Layers = { ob: true, fvg: true, liq: true, str: true };

export const LEVEL_KEYS: readonly (keyof typeof DEMO_LEVELS)[] = [
  'obLow', 'obHigh', 'fvgLow', 'fvgHigh', 'liqIntact', 'liqSwept', 'chochLevel', 'bosLevel', 'currentPrice',
];

export function useBounds() {
  const candles = buildCandles(DEMO_CLOSES);
  return priceBounds(candles, LEVEL_KEYS.map((k) => DEMO_LEVELS[k]));
}

/** Shared structure chart — rendered full in the Structure pane and compact in
 * the MIA action preview, both bound to the same `layers`. */
export function StructureChart({ layers, stagger = false }: { layers: Layers; stagger?: boolean }) {
  const t = useTranslations('home');
  const b = useBounds();
  const L = DEMO_LEVELS;
  const lay = (on: boolean) => `${styles.lay} ${on ? styles.layOn : ''}`;
  return (
    <div className={styles.dchart}>
      <CandleSvg width={560} height={250} extra={LEVEL_KEYS.map((k) => L[k])} stagger={stagger} />

      <div className={`${lay(layers.ob)} ${styles.layOb}`} aria-hidden={!layers.ob}>
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

      <div className={`${lay(layers.fvg)} ${styles.layFvg}`} aria-hidden={!layers.fvg}>
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

      <div className={`${lay(layers.liq)} ${styles.layLiq}`} aria-hidden={!layers.liq}>
        <div style={{ position: 'absolute', left: '10px', right: '64px', top: `${yPct(L.liqIntact, b)}%`, height: '1.5px', background: 'var(--liq)', opacity: 0.85 }} />
        <div className={styles.dl} style={{ left: '12px', top: `${yPct(L.liqIntact, b) - 8}%`, background: 'rgba(214,162,74,.16)', color: 'var(--liq)' }}>
          {t('demo.structure.labels.liqIntact')}
        </div>
        <div style={{ position: 'absolute', left: '10px', right: '64px', top: `${yPct(L.liqSwept, b)}%`, height: '1px', background: 'repeating-linear-gradient(90deg,var(--liq) 0 4px,transparent 4px 8px)', opacity: 0.5 }} />
        <div className={styles.dl} style={{ left: '12px', top: `${yPct(L.liqSwept, b) + 2}%`, background: 'rgba(214,162,74,.10)', color: 'var(--faint)' }}>
          {t('demo.structure.labels.liqSwept')}
        </div>
      </div>

      <div className={`${lay(layers.str)} ${styles.layStr}`} aria-hidden={!layers.str}>
        <div className={styles.dl} style={{ left: '34%', top: `${yPct(L.chochLevel, b) - 4}%`, background: 'rgba(55,185,140,.16)', color: 'var(--bull)' }}>
          {t('demo.structure.labels.choch')}
        </div>
        <div className={styles.dl} style={{ left: '58%', top: `${yPct(L.bosLevel, b) - 10}%`, background: 'rgba(55,185,140,.16)', color: 'var(--bull)' }}>
          {t('demo.structure.labels.bos')}
        </div>
      </div>

      <div style={{ position: 'absolute', right: '8px', top: `${yPct(L.currentPrice, b) - 4}%`, fontFamily: 'var(--font-mono)', fontSize: '10px', background: 'var(--acc)', color: 'var(--acc-txt)', padding: '2px 7px', borderRadius: '4px', fontWeight: 700 }}>
        4&nbsp;026,77
      </div>
    </div>
  );
}

export function StructureNarration({ layers }: { layers: Layers }) {
  const t = useTranslations('home');
  const order: LayerKey[] = ['str', 'ob', 'fvg', 'liq'];
  const active = order.filter((k) => layers[k]);
  const rich = (key: string): ReactNode => t.rich(key, { b: (c) => <b>{c}</b> });
  if (active.length === 0) {
    return <div className={styles.dnarr}><span style={{ color: 'var(--faint)' }}>{t('demo.structure.empty')}</span></div>;
  }
  return (
    <div className={styles.dnarr}>
      <b>{t('demo.structure.prefix')}</b>{' '}
      {active.map((k, i) => (
        <span key={k}>{rich(`demo.structure.frag.${k}`)}{i < active.length - 1 ? ' ' : ''}</span>
      ))}
    </div>
  );
}
