'use client';

import { type ReactNode } from 'react';
import { useTranslations } from 'next-intl';
import styles from './lp1.module.css';

/** Star/sparkle mark used for the M.I.A avatar. */
function Spark() {
  return (
    <svg viewBox="0 0 24 24" width={17} height={17} fill="currentColor" aria-hidden>
      <path d="M12 3l1.9 4.6L18.5 9l-4.6 1.4L12 15l-1.9-4.6L5.5 9l4.6-1.4z" />
      <path d="M18.5 15.5l.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7z" />
    </svg>
  );
}

const CAP_ICONS: Record<string, ReactNode> = {
  c1: <path d="M4 19V5a2 2 0 012-2h12a2 2 0 012 2v14M8 8h8M8 12h8M8 16h5" />,
  c2: <><circle cx="11" cy="11" r="7" /><path d="M20 20l-4-4" /></>,
  c3: <path d="M4 20V10M9 20V4M14 20v-7M19 20V8" />,
  c4: <path d="M12 9v4M12 17h.01M10.3 3.9L2 18a2 2 0 001.7 3h16.6a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z" />,
};

const CAP_TONE: Record<string, string> = {
  c1: 'var(--acc)',
  c2: 'var(--bull)',
  c3: 'var(--lp-vio)',
  c4: 'var(--liq)',
};

/**
 * §3 — M.I.A section (LP-2). The product's rarest differentiator, raised near
 * the top and given room: what she can do, where, and why. A grounded example
 * exchange (ending on an explicit refusal) plus four capability cards. Static —
 * no network, the interactive proof lives in the Demo section below.
 */
export function MiaSection() {
  const t = useTranslations('home');
  const rich = (k: string): ReactNode => t.rich(k, { b: (c) => <b>{c}</b> });
  const questions = Object.values(t.raw('miaSection.questions') as Record<string, string>);

  return (
    <section id="mia">
      <div className={styles.wrap}>
        <div className={styles.sechead}>
          <div className={styles.eyebrow}>{t('miaSection.eyebrow')}</div>
          <h2>{t('miaSection.title')}</h2>
          <p>{t('miaSection.subtitle')}</p>
        </div>

        <div className={styles.miaHero}>
          <div className={styles.miaGrid}>
            <div>
              <div className={styles.miaBadge}>
                <span className={styles.miaAvatar}>
                  <Spark />
                  <span className={styles.miaOnline} />
                </span>
                <span>
                  <span className={styles.miaName}>{t('miaSection.badgeName')}</span>
                  <br />
                  <span className={styles.miaSub}>{t('miaSection.badgeSub')}</span>
                </span>
              </div>
              <h3 className={styles.miaH3}>{t('miaSection.h3')}</h3>
              <p className={styles.miaP}>{rich('miaSection.p')}</p>
              <div className={styles.qlist}>
                {questions.map((q, i) => (
                  <span key={i} className={styles.ql}>
                    {q}
                    <span aria-hidden className={styles.qlArrow}>↗</span>
                  </span>
                ))}
              </div>
            </div>

            <div className={styles.vis}>
              <div className={styles.vish}>
                <span className={styles.dots}><i /><i /><i /></span>
                <span className={styles.vt}>{t('miaSection.chatHeader')}</span>
              </div>
              <div className={styles.visb}>
                <div className={styles.mchat} style={{ minHeight: 250, border: 'none', background: 'transparent', padding: 0 }}>
                  <div className={styles.mbU} style={{ alignSelf: 'flex-end' }}>{t('miaSection.chat.u1')}</div>
                  <div className={styles.mbA} style={{ alignSelf: 'flex-start' }}>{rich('miaSection.chat.a1')}</div>
                  <div className={styles.mbU} style={{ alignSelf: 'flex-end' }}>{t('miaSection.chat.u2')}</div>
                  <div className={styles.mbA} style={{ alignSelf: 'flex-start' }}>{rich('miaSection.chat.a2')}</div>
                  <div className={styles.mbU} style={{ alignSelf: 'flex-end' }}>{t('miaSection.chat.u3')}</div>
                  <div className={styles.mbNo} style={{ alignSelf: 'flex-start' }}>{rich('miaSection.chat.a3')}</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className={styles.capgrid}>
          {(['c1', 'c2', 'c3', 'c4'] as const).map((c) => (
            <div key={c} className={styles.cap}>
              <span className={styles.capIc} style={{ color: CAP_TONE[c], background: 'color-mix(in srgb, currentColor 14%, transparent)' }}>
                <svg viewBox="0 0 24 24" width={18} height={18} fill="none" stroke="currentColor" strokeWidth={1.9} strokeLinecap="round" strokeLinejoin="round">
                  {CAP_ICONS[c]}
                </svg>
              </span>
              <h4>{t(`miaSection.caps.${c}.h`)}</h4>
              <p>{rich(`miaSection.caps.${c}.p`)}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
