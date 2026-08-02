'use client';

import { useEffect, useRef, useState, type ReactNode } from 'react';
import { useTranslations } from 'next-intl';
import styles from './lp1.module.css';
import { CandleSvg } from './CandleSvg';
import { buildCandles, priceBounds, yPct } from './chart';
import {
  DEMO_CLOSES,
  DEMO_LEVELS,
  DEMO_MARKETS,
  DEMO_ZONES,
  DEMO_MIA,
  DEMO_REGIME,
  type LayerKey,
} from './data';

type Layers = Record<LayerKey, boolean>;
const ALL_ON: Layers = { ob: true, fvg: true, liq: true, str: true };

const LEVEL_KEYS: readonly (keyof typeof DEMO_LEVELS)[] = [
  'obLow', 'obHigh', 'fvgLow', 'fvgHigh', 'liqIntact', 'liqSwept', 'chochLevel', 'bosLevel', 'currentPrice',
];

function useBounds() {
  const candles = buildCandles(DEMO_CLOSES);
  return priceBounds(candles, LEVEL_KEYS.map((k) => DEMO_LEVELS[k]));
}

/** Shared structure chart — rendered full in the Structure pane and compact in
 * the MIA action preview, both bound to the same `layers`. */
function StructureChart({ layers }: { layers: Layers }) {
  const t = useTranslations('home');
  const b = useBounds();
  const L = DEMO_LEVELS;
  const lay = (on: boolean) => `${styles.lay} ${on ? styles.layOn : ''}`;
  return (
    <div className={styles.dchart}>
      <CandleSvg width={560} height={250} extra={LEVEL_KEYS.map((k) => L[k])} />

      <div className={lay(layers.ob)} aria-hidden={!layers.ob}>
        <div
          className={styles.dz}
          style={{
            left: '60%', right: '64px', top: `${yPct(L.obHigh, b)}%`,
            height: `${yPct(L.obLow, b) - yPct(L.obHigh, b)}%`,
            background: 'var(--ob)', border: '1px dashed var(--ob-l)',
          }}
        />
        <div className={styles.dl} style={{ left: '60.5%', top: `${yPct(L.obHigh, b) - 8}%`, background: 'var(--ob)', color: 'var(--bear)' }}>
          {t('demo.structure.labels.ob')}
        </div>
      </div>

      <div className={lay(layers.fvg)} aria-hidden={!layers.fvg}>
        <div
          className={styles.dz}
          style={{
            left: '36%', right: '64px', top: `${yPct(L.fvgHigh, b)}%`,
            height: `${yPct(L.fvgLow, b) - yPct(L.fvgHigh, b)}%`,
            background: 'var(--fvg)', border: '1px dashed var(--fvg-l)',
          }}
        />
        <div className={styles.dl} style={{ left: '36.5%', top: `${yPct(L.fvgHigh, b) - 8}%`, background: 'var(--fvg)', color: 'var(--fvg-l)' }}>
          {t('demo.structure.labels.fvg')}
        </div>
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

      <div style={{ position: 'absolute', right: '8px', top: `${yPct(L.currentPrice, b) - 4}%`, fontFamily: 'var(--font-mono)', fontSize: '10px', background: 'var(--acc)', color: 'var(--acc-txt)', padding: '2px 7px', borderRadius: '4px', fontWeight: 700 }}>
        4&nbsp;026,77
      </div>
    </div>
  );
}

function StructureNarration({ layers }: { layers: Layers }) {
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

function StructurePane({ layers, setLayers }: { layers: Layers; setLayers: (l: Layers) => void }) {
  const t = useTranslations('home');
  const chip = (k: LayerKey, color: string) => (
    <button
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
    <div className={styles.dgrid}>
      <div>
        <div className={styles.dchips}>
          {chip('ob', 'var(--bear)')}
          {chip('fvg', 'var(--fvg-l)')}
          {chip('liq', 'var(--liq)')}
          {chip('str', 'var(--dim)')}
        </div>
        <StructureChart layers={layers} />
        <StructureNarration layers={layers} />
        <div className={styles.illus}>{t('demo.illus')}</div>
      </div>
      <div className={styles.dside}>
        <h4>{t('demo.structure.side.title')}</h4>
        <p>{t.rich('demo.structure.side.desc', { b: (c) => <b>{c}</b> })}</p>
        <div className={styles.try}>{t('demo.structure.side.try')}</div>
        <button type="button" className={styles.opt} onClick={() => setLayers({ ob: true, fvg: false, liq: false, str: false })}>{t('demo.structure.side.onlyOb')}</button>
        <button type="button" className={styles.opt} onClick={() => setLayers({ ob: false, fvg: false, liq: true, str: false })}>{t('demo.structure.side.onlyLiq')}</button>
        <button type="button" className={styles.opt} onClick={() => setLayers({ ...ALL_ON })}>{t('demo.structure.side.all')}</button>
      </div>
    </div>
  );
}

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
        <div className={styles.zcard} style={{ opacity: hidden ? 0.4 : 1 }}>
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

function MiaPane({ layers, setLayers, jumpToStructure }: { layers: Layers; setLayers: (l: Layers) => void; jumpToStructure: () => void }) {
  const t = useTranslations('home');
  const [thread, setThread] = useState<number[]>([]);
  const [showPreview, setShowPreview] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ block: 'nearest' });
  }, [thread]);

  const ask = (i: number) => {
    setThread((prev) => [...prev, i]);
    const ex = DEMO_MIA[i]!;
    if (ex.action) {
      const next: Layers = { ob: false, fvg: false, liq: false, str: false };
      ex.action.only.forEach((k) => { next[k] = true; });
      setLayers(next);
      setShowPreview(true);
    }
  };

  const rich = (key: string): ReactNode => t.rich(key, { b: (c) => <b>{c}</b> });

  return (
    <div className={styles.dgrid}>
      <div>
        <div className={styles.mchat}>
          {thread.length === 0 && <div className={styles.mbA} style={{ opacity: 0.7 }}>{t('demo.mia.greeting')}</div>}
          {thread.map((qi, idx) => {
            const ex = DEMO_MIA[qi]!;
            const bubbleClass = ex.kind === 'refusal' ? styles.mbNo : styles.mbA;
            return (
              <div key={`${qi}-${idx}`} style={{ display: 'contents' }}>
                <div className={styles.mbU} style={{ alignSelf: 'flex-end' }}>{t(`demo.mia.q.${ex.key}.q`)}</div>
                <div className={bubbleClass} style={{ alignSelf: 'flex-start' }}>{rich(`demo.mia.q.${ex.key}.a`)}</div>
                {ex.action && showPreview && (
                  <div style={{ alignSelf: 'stretch' }}>
                    <div className={styles.mbAction}>{t('demo.mia.actionNote')}</div>
                    <StructureChart layers={layers} />
                    <button type="button" className={styles.opt} style={{ marginTop: '8px', marginBottom: 0 }} onClick={jumpToStructure}>
                      {t('demo.mia.actionCta')}
                    </button>
                  </div>
                )}
              </div>
            );
          })}
          <div ref={endRef} />
        </div>
        <div className={styles.illus}>{t('demo.illus')}</div>
      </div>
      <div className={styles.dside}>
        <h4>{t('demo.mia.side.title')}</h4>
        <p>{t.rich('demo.mia.side.desc', { b: (c) => <b>{c}</b> })}</p>
        <div className={styles.try}>{t('demo.mia.side.try')}</div>
        {DEMO_MIA.map((ex, i) => (
          <button key={ex.key} type="button" className={styles.opt} onClick={() => ask(i)}>
            {t(`demo.mia.q.${ex.key}.q`)}
          </button>
        ))}
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
            <span className={styles.eyebrow} style={{ margin: 0 }}>{t('demo.calcul.tileLabel')}</span>
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

const TAB_KEYS = ['structure', 'scanner', 'zones', 'mia', 'calcul'] as const;

function TabIcon({ which }: { which: (typeof TAB_KEYS)[number] }) {
  const paths: Record<string, ReactNode> = {
    structure: <><path d="M4 20V10M9 20V4M14 20v-7M19 20V8" /></>,
    scanner: <><circle cx="12" cy="12" r="9" /><circle cx="12" cy="12" r="4" /></>,
    zones: <><path d="M12 3l9 5-9 5-9-5z" /><path d="M3 13l9 5 9-5" /></>,
    mia: <><path d="M12 3l1.9 4.6L18.5 9l-4.6 1.4L12 15l-1.9-4.6L5.5 9l4.6-1.4z" /></>,
    calcul: <><path d="M4 6h16M4 12h16M4 18h10" /></>,
  };
  return <svg viewBox="0 0 24 24">{paths[which]}</svg>;
}

export function DemoTabs() {
  const t = useTranslations('home');
  const [tab, setTab] = useState(0);
  const [layers, setLayers] = useState<Layers>({ ...ALL_ON });

  return (
    <div className={styles.demo}>
      <div className={styles.dtabs} role="tablist" aria-label={t('demoSection.title')}>
        {TAB_KEYS.map((k, i) => (
          <button
            key={k}
            type="button"
            role="tab"
            aria-selected={tab === i}
            className={`${styles.dtab} ${tab === i ? styles.dtabOn : ''}`}
            onClick={() => setTab(i)}
          >
            <span className={styles.dtabIc} style={{ background: 'var(--acc-dim)', color: 'var(--acc)' }}><TabIcon which={k} /></span>
            {t(`demo.tabs.${k}`)}
          </button>
        ))}
      </div>
      <div className={styles.dpane} role="tabpanel">
        {tab === 0 && <StructurePane layers={layers} setLayers={setLayers} />}
        {tab === 1 && <ScannerPane />}
        {tab === 2 && <ZonesPane />}
        {tab === 3 && <MiaPane layers={layers} setLayers={setLayers} jumpToStructure={() => setTab(0)} />}
        {tab === 4 && <CalculPane />}
      </div>
    </div>
  );
}
