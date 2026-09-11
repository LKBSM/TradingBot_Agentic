'use client';

import { useCallback, useState, type ReactNode } from 'react';
import { useTranslations } from 'next-intl';
import { useLocalizedHref } from '@/lib/i18n/href';
import styles from './lp1.module.css';
import { Opening } from './Opening';
import { ReadingUnfolds } from './ReadingUnfolds';
import { RefusalBlock } from './RefusalBlock';
import { DemoTabs } from './DemoTabs';
import { ALL_ON, type Layers } from './structure-chart';
import { PRICING } from '@/lib/pricing.generated';

function Check() {
  return <span className={styles.ck} aria-hidden="true">✓</span>;
}

/**
 * LP-3 — the home page.
 *
 * Twelve sections became seven and 3 583 words became roughly 2 000, because
 * the parts that were describing the product in prose now ARE the product.
 * What went, and why:
 *
 *   · the four alternating feature rows (869 words of "tag · heading ·
 *     paragraph · five ticked bullets · fake browser window", four times in
 *     mirror) — they narrated what the demos below already did;
 *   · the five-panel carousel — absorbed by the scroll section;
 *   · "How it works" as three 01/02/03 cards and "Who it's for" as three
 *     symmetric persona cards — the two most template-shaped blocks on the web;
 *   · the four M.I.A capability cards and their stock icons (the sentences
 *     survive verbatim in RefusalBlock — LP-2A's guard still holds);
 *   · every uppercase letter-spaced eyebrow (there were FIVE separate systems
 *     of them), the decorative hero gradient, and the nine fake macOS window
 *     frames.
 *
 * What arrived: an opening built on the real product surface, ONE orchestrated
 * scroll moment, and the refusal given the room it earns.
 *
 * `manual` is the page-level layer override, deliberately held here rather than
 * inside a section: the reader can take the chart with the chips in the scroll
 * section, and M.I.A can take it with a view action several screens below. Both
 * write the same state, so her "I have hidden the Fair Value Gaps" moves the
 * very chart the reader watched assemble itself.
 */
export function HomeLanding() {
  const t = useTranslations('home');
  const lh = useLocalizedHref();
  const rich = (k: string): ReactNode => t.rich(k, { b: (c) => <b>{c}</b> });
  const [openFaq, setOpenFaq] = useState<number>(-1);
  const [manual, setManual] = useState<Layers | null>(null);

  const jumpToReading = useCallback(() => {
    document.getElementById('lecture')?.scrollIntoView({ block: 'start' });
  }, []);

  return (
    <div className={styles.page}>
      <Opening />

      <ReadingUnfolds manual={manual} setManual={setManual} />

      <RefusalBlock
        layers={manual ?? ALL_ON}
        setLayers={setManual}
        jumpToReading={jumpToReading}
      />

      {/* The three demos that are still their own thing — no tab strip. */}
      <section id="demo"><div className={styles.wrap}>
        <h2 className={styles.secH}>{t('hands.title')}</h2>
        <p className={styles.secP}>{t('hands.lead')}</p>
        <DemoTabs />
      </div></section>

      {/* The argument, in the two paragraphs that already said it best. */}
      <section id="honnetete"><div className={styles.wrap}><div className={styles.narrow}>
        <h2 className={styles.secH}>{t('distinguish.h2')}</h2>
        <p className={styles.argP}>{rich('distinguish.p1')}</p>
        <p className={styles.argP}>{rich('distinguish.p2')}</p>
      </div></div></section>

      {/* PRICING */}
      <section id="tarifs"><div className={styles.wrap}>
        <h2 className={styles.secH}>{t('pricing.title')}</h2>
        <p className={styles.secP}>{t('pricing.subtitle')}</p>
        <div className={styles.prices}>
          <div className={`${styles.pc} ${styles.pcHi}`}>
            <span className={styles.pcLbl}>{t('pricing.paid.label')}</span>
            <h3 className={styles.pcName}>{t('pricing.paid.name')}</h3>
            <div className={styles.pcSub}>{t('pricing.paid.sub')}</div>
            <div className={styles.amt}>
              <span className={styles.amtV}>{PRICING.monthly} $</span>
              <span className={styles.amtU}>US / {t('pricing.paid.perMonth')}</span>
            </div>
            <div className={styles.bill}>
              {t('pricing.paid.bill', { total: String(PRICING.annualPerYear), perMonth: String(PRICING.annualPerMonth) })}
            </div>
            <div className={styles.pfeat}>
              {['1', '2', '3', '4', '5', '6'].map((i) => (
                <div key={i} className={styles.pf}><Check />{t.rich(`pricing.paid.f${i}`, { b: (c) => <b>{c}</b> })}</div>
              ))}
            </div>
            <a className={`${styles.btn} ${styles.btnPri} ${styles.btnMd} ${styles.btnFull}`} href={lh('/inscription')}>
              {t('pricing.paid.cta')}
            </a>
            <div className={styles.pcNote}>{t('pricing.paid.note')}</div>
          </div>
        </div>
        <p className={styles.plegal}>{t('pricing.legal')}</p>
      </div></section>

      {/* FAQ — every answer closed by default: eight open answers read as a
          wall, and a visitor who has a question knows which one is theirs. */}
      <section id="faq"><div className={styles.wrap}>
        <h2 className={styles.secH}>{t('faq.title')}</h2>
        <div className={styles.faq}>
          {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => {
            const open = openFaq === i;
            return (
              <div key={i} className={`${styles.fq} ${open ? styles.fqOpen : ''}`}>
                <button type="button" className={styles.fqh} aria-expanded={open} onClick={() => setOpenFaq(open ? -1 : i)}>
                  <span className={styles.fqq}>{t(`faq.q${i}`)}</span>
                  <span className={styles.fqc} aria-hidden="true">+</span>
                </button>
                {open && <div className={styles.fqb}>{rich(`faq.a${i}`)}</div>}
              </div>
            );
          })}
        </div>
      </div></section>

      {/* FINAL — the sharpest sentence on the page, which used to be the last
          line of a paragraph inside a card at the very bottom. */}
      <section><div className={styles.wrap}><div className={styles.final}>
        <h2 className={styles.finalH}>{t('final.h2')}</h2>
        <p className={styles.finalP}>{t('final.p')}</p>
        <a className={`${styles.btn} ${styles.btnPri} ${styles.btnLg}`} href={lh('/inscription')}>{t('final.cta1')}</a>
      </div></div></section>
    </div>
  );
}
