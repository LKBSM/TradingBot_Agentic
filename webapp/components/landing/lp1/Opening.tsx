'use client';

import { type ReactNode } from 'react';
import { useLocale, useTranslations } from 'next-intl';
import { useLocalizedHref } from '@/lib/i18n/href';
import { MarketReadingCard } from '@/components/market-reading/MarketReadingCard';
import { SAMPLE_READING_XAU_H4 } from '@/lib/ds-samples';
import styles from './lp1.module.css';

/**
 * LP-3 — the opening.
 *
 * Two changes of intent from what it replaces:
 *
 *  1. The screen shows the PRODUCT, not a picture of it. The right-hand column
 *     is `<MarketReadingCard>` — the same component /app renders — fed a real
 *     archived reading from lib/ds-samples (DS-1: extracted from
 *     market_readings.db, not written by hand). It is deliberately clipped by
 *     the right edge: a product continues past the frame, a screenshot does not.
 *     No fake browser chrome, no three little dots.
 *
 *  2. The promise is stated by what the product REFUSES. Three sentences, each
 *     of which the page then has to make good on further down. It is the only
 *     claim on this page a competitor cannot copy by shipping a feature.
 *
 * Honesty of the sample: the card carries its own freshness badge, so the
 * archived candle openly reads "closed six weeks ago" rather than pretending to
 * be live — which is precisely what faq.a5 promises the product always does.
 * The archive line above states the instrument, the timeframe and the close
 * date, read from the sample itself so it can never drift from the data.
 */
export function Opening() {
  const t = useTranslations('home');
  const locale = useLocale();
  const lh = useLocalizedHref();
  const rich = (k: string): ReactNode => t.rich(k, { b: (c) => <b>{c}</b> });

  const header = SAMPLE_READING_XAU_H4.header;
  // Fixed timestamp + explicit UTC ⇒ server and client format the same string
  // (no hydration mismatch), and the date can never drift from the sample.
  const closedOn = new Intl.DateTimeFormat(locale, {
    day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC',
  }).format(new Date(header.candle_close_ts));

  return (
    <section className={styles.open}>
      <div className={styles.openGrid}>
        <div className={styles.openText}>
          <h1 className={styles.openH1}>{t('opening.h1')}</h1>
          <p className={styles.openLead}>{t('opening.lead')}</p>

          <ul className={styles.openRefus}>
            {(['r1', 'r2', 'r3'] as const).map((r) => (
              <li key={r} className={styles.openRefu}>
                <span className={styles.openRefuNo} aria-hidden="true" />
                <span>
                  <b className={styles.openRefuH}>{t(`opening.refusals.${r}.h`)}</b>{' '}
                  <span className={styles.openRefuP}>{t(`opening.refusals.${r}.p`)}</span>
                </span>
              </li>
            ))}
          </ul>

          <div className={styles.openCtas}>
            <a className={`${styles.btn} ${styles.btnPri} ${styles.btnLg}`} href={lh('/inscription')}>
              {t('hero.ctaPrimary')}
            </a>
            <a className={`${styles.btn} ${styles.btnLg}`} href="#lecture">
              {t('opening.ctaSecondary')}
            </a>
          </div>

          {/* LP-2S — the product's real perimeter, kept verbatim (its own guard
              in home.test.tsx checks the wording across the nine locales). */}
          <p className={styles.roadmap}>{rich('hero.roadmap')}</p>
        </div>

        <div className={styles.openStage}>
          <p className={styles.openArchive}>
            {t('opening.archive', {
              instrument: header.instrument,
              timeframe: header.timeframe,
              when: closedOn,
            })}
          </p>
          <div className={styles.openCard}>
            {/* The card's own "Ask M.I.A" button is disabled without a handler,
                and a dead control in the first screen is exactly the kind of
                prop this redesign exists to remove. Here it does the honest
                thing available on a public page: it takes you to the real agent
                further down, rather than opening a chat that needs an account. */}
            <MarketReadingCard
              reading={SAMPLE_READING_XAU_H4}
              className={styles.openCardSurface}
              onAskChatbot={() => document.getElementById('mia')?.scrollIntoView({ block: 'start' })}
            />
          </div>
        </div>
      </div>
    </section>
  );
}
