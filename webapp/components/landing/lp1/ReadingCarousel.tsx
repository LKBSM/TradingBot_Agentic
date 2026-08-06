'use client';

import { useCallback, useRef, useState, type CSSProperties, type ReactNode } from 'react';
import { useTranslations } from 'next-intl';
import styles from './lp1.module.css';
import { CandleSvg } from './CandleSvg';

const SLIDE_COUNT = 5;

const LAYER_SWATCH = ['var(--bear)', 'var(--lp-vio)', 'var(--liq)', 'var(--dim)', 'var(--line-2)'];

const JOURNAL_TONE: Record<string, { bg: string; fg: string }> = {
  bos: { bg: 'rgba(55,185,140,.16)', fg: 'var(--bull)' },
  pocket: { bg: 'rgba(214,162,74,.16)', fg: 'var(--liq)' },
  choch: { bg: 'rgba(55,185,140,.16)', fg: 'var(--bull)' },
  ob: { bg: 'rgba(255,107,129,.16)', fg: 'var(--bear)' },
  fvg: { bg: 'rgba(167,139,250,.16)', fg: 'var(--lp-vio)' },
};

type Tile = { k: string; v: string; n: string };
type JournalRow = { kind: string; badge: string; text: string; time: string };
type TfRow = { tf: string; text: string; note: string };

/**
 * §5 — The reading-space showcase (LP-2). A five-panel carousel proving the
 * reading space is FOUR panels + a chat, not "an annotated chart": chart with
 * layers · full narrated reading · the real regime tiles · the event journal ·
 * the per-timeframe reading. Static illustration data; the manipulable proof
 * lives in the Demo section. Arrows · dots · keyboard · swipe · named counter.
 */
export function ReadingCarousel() {
  const t = useTranslations('home');
  const rich = (k: string): ReactNode => t.rich(k, { b: (c) => <b>{c}</b> });
  const [i, setI] = useState(0);
  const touchX = useRef<number | null>(null);

  const go = useCallback((delta: number) => {
    setI((prev) => (prev + delta + SLIDE_COUNT) % SLIDE_COUNT);
  }, []);

  // next-intl messages hold no arrays (they're index-keyed objects); read the
  // values back into arrays for rendering.
  const vals = <T,>(k: string): T[] => Object.values(t.raw(k) as Record<string, T>);
  const names = vals<string>('carousel.names');
  const layers = vals<string>('carousel.chart.layers');
  const tiles = vals<Tile>('carousel.regime.tiles');
  const rows = vals<JournalRow>('carousel.journal.rows');
  const tfChips = vals<string>('carousel.timeframes.chips');
  const tfRows = vals<TfRow>('carousel.timeframes.rows');
  const activeChip = t('carousel.timeframes.activeChip');

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft') { e.preventDefault(); go(-1); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); go(1); }
  };

  const frame = (title: string, children: ReactNode, foot?: string) => (
    <div className={styles.vis}>
      <div className={styles.vish}>
        <span className={styles.dots}><i /><i /><i /></span>
        <span className={styles.vt}>{title}</span>
      </div>
      <div className={styles.visb} style={{ minHeight: 300, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
        {children}
        {foot && <div className={styles.illus}>{foot}</div>}
      </div>
    </div>
  );

  const lbl = (style: CSSProperties, text: string) => (
    <div className={styles.ml} style={style}>{text}</div>
  );

  return (
    <div
      className={styles.car}
      role="region"
      aria-roledescription="carousel"
      aria-label={t('carousel.regionAria')}
      tabIndex={0}
      onKeyDown={onKeyDown}
    >
      <div
        className={styles.carViewport}
        onTouchStart={(e) => { touchX.current = e.touches[0]!.clientX; }}
        onTouchEnd={(e) => {
          if (touchX.current == null) return;
          const dx = e.changedTouches[0]!.clientX - touchX.current;
          if (Math.abs(dx) > 44) go(dx < 0 ? 1 : -1);
          touchX.current = null;
        }}
      >
        <div
          className={styles.carTrack}
          style={{ width: `${SLIDE_COUNT * 100}%`, transform: `translateX(-${(i * 100) / SLIDE_COUNT}%)` }}
        >
          {/* 1 — chart */}
          <div className={styles.carSlide} aria-hidden={i !== 0}>
            {frame(
              t('carousel.chart.title'),
              <>
                <div className={styles.lyr}>
                  {layers.map((l, idx) => (
                    <span key={l} className={`${styles.lyb} ${idx === 4 ? styles.lybOff : ''}`}>
                      <span className={styles.sq} style={{ background: LAYER_SWATCH[idx] }} />{l}
                    </span>
                  ))}
                </div>
                <div className={styles.mc} style={{ height: 200 }}>
                  <CandleSvg width={560} height={200} extra={[4034.2, 4011.4, 4025.1, 4021.45]} />
                  <div className={styles.mz} style={{ left: '74%', right: '56px', top: '47px', height: '14px', background: 'rgba(255,107,129,.11)', border: '1px dashed rgba(255,107,129,.55)' }} />
                  {lbl({ left: '74.5%', top: '30px', background: 'rgba(255,107,129,.17)', color: 'var(--bear)' }, t('carousel.chart.obLabel'))}
                  <div style={{ position: 'absolute', left: '10px', right: '56px', top: '30px', height: '1.4px', background: 'var(--liq)', opacity: 0.8 }} />
                  {lbl({ left: '12px', top: '15px', background: 'rgba(214,162,74,.15)', color: 'var(--liq)' }, t('carousel.chart.liqLabel'))}
                  {lbl({ left: '41%', top: '143px', background: 'rgba(55,185,140,.16)', color: 'var(--bull)' }, t('carousel.chart.chochLabel'))}
                  {lbl({ left: '66%', top: '47px', background: 'rgba(55,185,140,.16)', color: 'var(--bull)' }, t('carousel.chart.bosLabel'))}
                  <div style={{ position: 'absolute', left: '12px', bottom: '8px', fontFamily: 'var(--font-mono)', fontSize: '8.5px', color: 'var(--faint)' }}>{t('carousel.chart.localTime')}</div>
                </div>
              </>,
            )}
          </div>

          {/* 2 — narrated reading */}
          <div className={styles.carSlide} aria-hidden={i !== 1}>
            {frame(
              t('carousel.narrated.title'),
              <>
                <div className="mono" style={{ fontSize: '9.5px', letterSpacing: '.11em', textTransform: 'uppercase', color: 'var(--faint)', marginBottom: '12px', fontFamily: 'var(--font-mono)' }}>{t('carousel.narrated.freshness')}</div>
                <div className={styles.narr} style={{ fontSize: '13px', marginTop: 0 }}>{rich('carousel.narrated.body')}</div>
              </>,
              t('carousel.narrated.foot'),
            )}
          </div>

          {/* 3 — regime */}
          <div className={styles.carSlide} aria-hidden={i !== 2}>
            {frame(
              t('carousel.regime.title'),
              <div className={styles.rg}>
                {tiles.map((tile, idx) => (
                  <div key={idx} className={styles.rgc}>
                    <div className={styles.rgk}>{tile.k}</div>
                    <div className={styles.rgv}>{tile.v}</div>
                    <div className={styles.rgn}>{tile.n}</div>
                  </div>
                ))}
              </div>,
              t('carousel.regime.foot'),
            )}
          </div>

          {/* 4 — event journal */}
          <div className={styles.carSlide} aria-hidden={i !== 3}>
            {frame(
              t('carousel.journal.title'),
              <div style={{ display: 'flex', flexDirection: 'column', gap: '7px' }}>
                {rows.map((r, idx) => {
                  const tone = JOURNAL_TONE[r.kind] ?? JOURNAL_TONE.ob!;
                  return (
                    <div key={idx} className={styles.jr}>
                      <span className={styles.jb} style={{ background: tone.bg, color: tone.fg }}>{r.badge}</span>
                      {r.text}
                      <span className={styles.tm}>{r.time}</span>
                    </div>
                  );
                })}
              </div>,
              t('carousel.journal.foot'),
            )}
          </div>

          {/* 5 — timeframes */}
          <div className={styles.carSlide} aria-hidden={i !== 4}>
            {frame(
              t('carousel.timeframes.title'),
              <>
                <div className={styles.tfs}>
                  {tfChips.map((c) => (
                    <span key={c} className={`${styles.tfb} ${c === activeChip ? styles.tfbOn : ''}`}>{c}</span>
                  ))}
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '7px' }}>
                  {tfRows.map((r, idx) => (
                    <div key={idx} className={styles.jr} style={idx === 0 ? { borderColor: 'var(--lp-acc)', background: 'var(--acc-dim)' } : undefined}>
                      <span className={styles.jb} style={{ background: idx === 0 ? 'var(--acc-dim)' : 'var(--panel-3)', color: idx === 0 ? 'var(--lp-acc)' : 'var(--faint)' }}>{r.tf}</span>
                      {r.text}
                      <span className={styles.tm}>{r.note}</span>
                    </div>
                  ))}
                </div>
              </>,
              t('carousel.timeframes.foot'),
            )}
          </div>
        </div>
      </div>

      <div className={styles.carNav}>
        <div className={styles.carDots}>
          {names.map((n, idx) => (
            <button
              key={idx}
              type="button"
              className={`${styles.carDot} ${idx === i ? styles.carDotOn : ''}`}
              aria-label={t('carousel.dotAria', { n: idx + 1 })}
              aria-current={idx === i}
              onClick={() => setI(idx)}
            />
          ))}
        </div>
        <span className={styles.carLabel}>
          <b>{i + 1}</b> / {SLIDE_COUNT} · {names[i]}
        </span>
        <span className={styles.carBtns}>
          <button type="button" className={styles.carBtn} aria-label={t('carousel.prevAria')} onClick={() => go(-1)}>
            <svg viewBox="0 0 24 24" width={18} height={18} fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M15 6l-6 6 6 6" /></svg>
          </button>
          <button type="button" className={styles.carBtn} aria-label={t('carousel.nextAria')} onClick={() => go(1)}>
            <svg viewBox="0 0 24 24" width={18} height={18} fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M9 6l6 6-6 6" /></svg>
          </button>
        </span>
      </div>
      <div className={styles.illus}>{t('carousel.illus')}</div>
    </div>
  );
}
