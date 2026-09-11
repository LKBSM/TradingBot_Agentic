'use client';

import { useEffect, useState, type ReactNode } from 'react';
import { useTranslations } from 'next-intl';
import styles from './lp1.module.css';
import { CandleSvg } from './CandleSvg';
import { ZoneRect } from './ZoneRect';
import { yPct } from './chart';
import { LEVEL_KEYS, useBounds } from './structure-chart';
import {
  DEMO_LEVELS,
  DEMO_MARKETS,
  DEMO_ZONES,
  DEMO_REGIME,
  type DemoZone,
} from './data';

function ScannerPane() {
  const t = useTranslations('home');
  const [on, setOn] = useState<boolean[]>([true, true, false, false, false]);
  const condKeys = ['trend_bullish', 'higher_tf_agrees', 'price_in_ob', 'zone_untested', 'liquidity_swept'] as const;
  const active = on.map((v, i) => (v ? i : -1)).filter((i) => i >= 0);
  const matched = DEMO_MARKETS.filter((m) => active.every((i) => m.c[i] === 1));
  const almost = DEMO_MARKETS.filter((m) => !active.every((i) => m.c[i] === 1) && active.some((i) => m.c[i] === 1)).slice(0, 2);
  const rich = (key: string): ReactNode => t.rich(key, { b: (c) => <b>{c}</b> });

  return (
    <div className={styles.dgrid}>
      <div>
        <div className={styles.try}>{t('demo.scanner.try')}</div>
        {condKeys.map((k, i) => (
          <button
            key={k}
            type="button"
            className={`${styles.cond} ${on[i] ? styles.condOn : ''}`}
            aria-pressed={on[i]}
            onClick={() => setOn((prev) => prev.map((v, j) => (j === i ? !v : v)))}
          >
            <span className={styles.bx}>{on[i] ? '✓' : ''}</span>
            {t(`demo.scanner.conditions.${k}`)}
          </button>
        ))}
      </div>
      <div className={styles.dside}>
        <div className={styles.res}>
          <div className={styles.resh}>
            <span className={styles.rn}>{active.length ? matched.length : '—'}</span>
            <span className={styles.rl}>{t('demo.scanner.count')}</span>
          </div>
          {active.length === 0 ? (
            <div className={styles.emptyMsg}>{rich('demo.scanner.emptyNoCond')}</div>
          ) : matched.length === 0 ? (
            <div className={styles.emptyMsg}>{rich('demo.scanner.emptyNoMatch')}</div>
          ) : (
            <>
              {matched.map((m) => (
                <div key={m.key} className={styles.rrow}>
                  <span className={styles.sym2}>{m.sym}</span>
                  <span>{t(`demo.scanner.markets.${m.key}`)}</span>
                  <span className={`${styles.sc2} ${styles.scOk}`} style={{ color: 'var(--bull)' }}>{active.length} / {active.length}</span>
                </div>
              ))}
              {almost.map((m) => (
                <div key={m.key} className={`${styles.rrow} ${styles.rrowMiss}`}>
                  <span className={styles.sym2}>{m.sym}</span>
                  <span>{t(`demo.scanner.markets.${m.key}`)}</span>
                  <span className={styles.sc2}>{active.filter((i) => m.c[i] === 1).length} / {active.length}</span>
                </div>
              ))}
            </>
          )}
        </div>
        {active.length > 0 && matched.length > 0 && (
          <div className={styles.against}>
            <b>{t('demo.scanner.againstLabel')}</b> — {rich('demo.scanner.against')}
          </div>
        )}
        <p style={{ marginTop: '13px' }}>{rich('demo.scanner.side')}</p>
        <div className={styles.illus}>{t('demo.illus')}</div>
      </div>
    </div>
  );
}

/** The selected zone drawn on the SAME candle series and the SAME bounds as the
 * Structure tab, with the SAME <ZoneRect>. A zone is a rectangle on a chart
 * before it is a paragraph — the card below tells its story, this shows it.
 * `hidden` really removes the drawing (the chart stays), which is exactly what
 * the "Masquer du graphique" button claims to do. */
function ZoneMiniChart({ zone, hidden }: { zone: DemoZone; hidden: boolean }) {
  const t = useTranslations('home');
  const b = useBounds();
  const top = yPct(zone.high, b);
  // A spent zone has been traversed end to end: 100 % eaten.
  const fill = zone.state === 'filled' ? 100 : zone.fill;
  const label = `${t(`demo.zones.kind.${zone.kind}`)} ${zone.dir === 'up' ? '↑' : '↓'} · ${t(`demo.zones.z.${zone.key}.state`)}`;
  return (
    <div className={`${styles.dchart} ${styles.dchartSm}`}>
      <CandleSvg width={560} height={170} extra={LEVEL_KEYS.map((k) => DEMO_LEVELS[k])} />
      {!hidden && (
        <ZoneRect
          kind={zone.kind}
          style={{ left: '10px', right: '64px', top: `${top}%`, height: `${yPct(zone.low, b) - top}%` }}
          labelStyle={{ left: '12px', top: `${top - 9}%` }}
          label={label}
          {...(fill != null ? { fillPct: fill } : null)}
          {...(zone.state === 'filled' ? { spent: true } : null)}
        />
      )}
    </div>
  );
}

function ZonesPane() {
  const t = useTranslations('home');
  const [sel, setSel] = useState(0);
  const [hidden, setHidden] = useState(false);
  const [fillW, setFillW] = useState(0);
  const zone = DEMO_ZONES[sel] ?? DEMO_ZONES[0]!;
  const rich = (key: string): ReactNode => t.rich(key, { b: (c) => <b>{c}</b> });

  useEffect(() => {
    setHidden(false);
    setFillW(0);
    const id = window.setTimeout(() => setFillW(zone.fill ?? 0), 60);
    return () => window.clearTimeout(id);
  }, [sel, zone.fill]);

  const badgeColor =
    zone.state === 'untested'
      ? { bg: 'rgba(55,185,140,.14)', bd: 'var(--bull)', fg: 'var(--bull)' }
      : zone.state === 'tested'
        ? { bg: 'rgba(214,162,74,.14)', bd: 'var(--liq)', fg: 'var(--liq)' }
        : { bg: 'var(--panel-3)', bd: 'var(--line-2)', fg: 'var(--faint)' };

  return (
    <div className={styles.dgrid}>
      <div>
        <div className={styles.zsel} role="tablist" aria-label={t('demo.zones.side.title')}>
          {DEMO_ZONES.map((z, i) => (
            <button
              key={z.key}
              type="button"
              role="tab"
              aria-selected={sel === i}
              className={`${styles.zb} ${sel === i ? styles.zbOn : ''}`}
              onClick={() => setSel(i)}
            >
              {t(`demo.zones.tabs.${z.key}`)}
            </button>
          ))}
        </div>
        <ZoneMiniChart zone={zone} hidden={hidden} />
        {/* The card is NOT dimmed when the zone is hidden: what disappears is the
         * drawing, not the facts — exactly what `zones.hiddenNote` says. */}
        <div className={styles.zcard} style={{ marginTop: '12px' }}>
          <div className={styles.zhead}>
            <span className={styles.chip} style={{ background: badgeColor.bg, borderColor: 'transparent', color: badgeColor.fg }}>
              {t(`demo.zones.kind.${zone.kind}`)} {zone.dir === 'up' ? '↑' : '↓'}
            </span>
            <span className="mono" style={{ fontSize: '13px', fontWeight: 600 }}>{zone.band}</span>
            <span className={styles.chip} style={{ marginLeft: 'auto', background: badgeColor.bg, borderColor: badgeColor.bd, color: badgeColor.fg }}>
              {t(`demo.zones.z.${zone.key}.state`)}
            </span>
          </div>
          <div className={styles.tl}>
            {zone.state === 'untested' && (<><i style={{ background: 'var(--acc)' }} /><s style={{ background: 'var(--line-2)' }} /><i style={{ border: '1.5px solid var(--line-2)' }} /></>)}
            {zone.state === 'tested' && (<><i style={{ background: 'var(--acc)' }} /><s style={{ background: 'var(--acc)', opacity: 0.6 }} /><i style={{ background: 'var(--liq)' }} /><s style={{ background: 'var(--acc)', opacity: 0.6 }} /><i style={{ background: 'var(--liq)' }} /><s style={{ background: 'var(--line-2)' }} /><i style={{ border: '1.5px solid var(--line-2)' }} /></>)}
            {zone.state === 'filled' && (<><i style={{ background: 'var(--faint)' }} /><s style={{ background: 'var(--line-2)' }} /><i style={{ background: 'var(--faint)' }} /><s style={{ background: 'var(--line-2)' }} /><i style={{ background: 'var(--bear)' }} /></>)}
          </div>
          <div className={styles.tlab}>
            {zone.state === 'untested' && (<><span>{t('demo.zones.tl.formed')}</span><span>{t('demo.zones.tl.now')}</span></>)}
            {zone.state === 'tested' && (<><span>{t('demo.zones.tl.formed')}</span><span>{t('demo.zones.tl.touch1')}</span><span>{t('demo.zones.tl.touch2')}</span><span>{t('demo.zones.tl.now')}</span></>)}
            {zone.state === 'filled' && (<><span>{t('demo.zones.tl.formed')}</span><span>{t('demo.zones.tl.touch1')}</span><span>{t('demo.zones.tl.filled')}</span></>)}
          </div>
          {zone.fill != null && (
            <>
              <div className={styles.fill}><i style={{ width: `${fillW}%` }} /></div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--faint)', marginTop: '6px' }}>
                {t('demo.zones.fillLabel', { pct: zone.fill })}
              </div>
            </>
          )}
          <div className={styles.zprose}>{rich(`demo.zones.z.${zone.key}.prose`)}</div>
          <div style={{ marginTop: '14px' }}>
            <button type="button" className={styles.opt} style={{ marginBottom: 0 }} aria-pressed={hidden} onClick={() => setHidden((h) => !h)}>
              {hidden ? t('demo.zones.show') : t('demo.zones.hide')}
            </button>
            {hidden && <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--faint)', marginTop: '8px' }}>{t('demo.zones.hiddenNote')}</div>}
          </div>
        </div>
        <div className={styles.illus}>{t('demo.illus')}</div>
      </div>
      <div className={styles.dside}>
        <h4>{t('demo.zones.side.title')}</h4>
        <p>{t.rich('demo.zones.side.desc', { b: (c) => <b>{c}</b> })}</p>
        <div className={styles.try}>{t('demo.zones.side.compare')}</div>
        <p style={{ fontSize: '13px', color: 'var(--faint)' }}>{t('demo.zones.side.hint')}</p>
      </div>
    </div>
  );
}

function CalculPane() {
  const t = useTranslations('home');
  const [open, setOpen] = useState(false);
  const R = DEMO_REGIME;
  const row = (label: string, value: string) => (
    <div className={styles.calcRow}><span>{label}</span><b>{value}</b></div>
  );
  return (
    <div className={styles.dgrid}>
      <div>
        <div className={styles.tile}>
          <div className={styles.tileTop}>
            <span className={styles.tileLabel}>{t('demo.calcul.tileLabel')}</span>
          </div>
          <div className={styles.tileVerdict}>{t('demo.calcul.verdict')}</div>
          <div className={styles.tileSub}>{t('demo.calcul.sub')}</div>
          {open && (
            <div className={styles.calc}>
              {row(t('demo.calcul.rows.recent'), `${R.recentAtr}`)}
              {row(t('demo.calcul.rows.baseline'), `${R.baselineAtr}`)}
              {row(t('demo.calcul.rows.ratio'), `${R.ratio}`)}
              {row(t('demo.calcul.rows.thresholds'), `${R.lowThreshold} · ${R.highThreshold}`)}
              <div className={styles.calcNote}>{t.rich('demo.calcul.concl', { b: (c) => <b>{c}</b> })}</div>
              <div className={styles.calcNote}><b style={{ color: 'var(--txt)' }}>{t('demo.calcul.notSayLabel')}</b> {t('demo.calcul.notSay')}</div>
            </div>
          )}
        </div>
        <div className={styles.illus}>{t('demo.illus')}</div>
      </div>
      <div className={styles.dside}>
        <h4>{t('demo.calcul.side.title')}</h4>
        <p>{t.rich('demo.calcul.side.desc', { b: (c) => <b>{c}</b> })}</p>
        <div className={styles.try}>{t('demo.calcul.side.try')}</div>
        <button type="button" className={styles.opt} aria-pressed={open} onClick={() => setOpen((o) => !o)}>
          {open ? t('demo.calcul.side.hide') : t('demo.calcul.side.toggle')}
        </button>
      </div>
    </div>
  );
}

/**
 * LP-3 — the three hands-on demos that are still their own thing.
 *
 * They used to sit behind a five-tab strip, each tab wearing a stock icon in a
 * tinted rounded square. Two of the five have moved: "Lire une structure" is now
 * the scroll section at the top of the page, and "Parler à M.I.A" has its own
 * block. What is left is three named panels, stacked and open — no tab strip, no
 * icons. A visitor scrolling past sees all three exist, instead of discovering
 * two of them only by clicking.
 */
const PANELS = ['scanner', 'zones', 'calcul'] as const;

export function DemoTabs() {
  const t = useTranslations('home');
  return (
    <div className={styles.hands}>
      {PANELS.map((k) => (
        <section key={k} className={styles.hand}>
          <h3 className={styles.handH}>{t(`demo.tabs.${k}`)}</h3>
          {k === 'scanner' && <ScannerPane />}
          {k === 'zones' && <ZonesPane />}
          {k === 'calcul' && <CalculPane />}
        </section>
      ))}
    </div>
  );
}
