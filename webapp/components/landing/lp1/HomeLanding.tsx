'use client';

import { useState, type ReactNode } from 'react';
import { useTranslations } from 'next-intl';
import { useLocalizedHref } from '@/lib/i18n/href';
import styles from './lp1.module.css';
import { DemoTabs } from './DemoTabs';
import { MiaSection } from './MiaSection';
import { ReadingCarousel } from './ReadingCarousel';
import { CandleSvg } from './CandleSvg';
import { LANDING_STATS, LANDING_STAT_ORDER } from '@/lib/landing/stats';
import { PRICING } from '@/lib/pricing.generated';

function Check() {
  return <span className={styles.ck} aria-hidden="true">✓</span>;
}

/** Feature bullet list — `count` items pulled from `${base}.b{n}`. */
function Bullets({ t, base, count = 4 }: { t: ReturnType<typeof useTranslations>; base: string; count?: number }) {
  return (
    <div className={styles.bul}>
      {Array.from({ length: count }, (_, k) => k + 1).map((i) => (
        <div key={i} className={styles.bu}>
          <Check />
          <span>{t.rich(`${base}.b${i}`, { b: (c) => <b>{c}</b> })}</span>
        </div>
      ))}
    </div>
  );
}

function VisFrame({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className={styles.vis}>
      <div className={styles.vish}>
        <span className={styles.dots}><i /><i /><i /></span>
        <span className={styles.vt}>{title}</span>
      </div>
      <div className={styles.visb}>{children}</div>
    </div>
  );
}

export function HomeLanding() {
  const t = useTranslations('home');
  const lh = useLocalizedHref();
  const rich = (k: string): ReactNode => t.rich(k, { b: (c) => <b>{c}</b> });
  const [openFaq, setOpenFaq] = useState<number>(0);

  return (
    <div className={styles.page}>
      {/* HERO + STATS */}
      <div className={styles.hero}>
        <div className={styles.aura} aria-hidden="true" />
        <div className={styles.wrap}>
          <div className={styles.heroIn}>
            <div className={styles.pill}><span className={styles.dot} />{t('hero.pill')}</div>
            <h1 className={styles.h1}>{t('hero.h1a')} <em>{t('hero.h1b')}</em></h1>
            <p className={styles.lead}>{t('hero.lead')}</p>
            <div className={styles.ctas}>
              <a className={`${styles.btn} ${styles.btnPri} ${styles.btnLg}`} href={lh('/inscription')}>{t('hero.ctaPrimary')}</a>
              <a className={`${styles.btn} ${styles.btnLg}`} href="#demo">{t('hero.ctaSecondary')}</a>
            </div>
            <div className={styles.micro}>{t('hero.micro')}</div>

            <div className={styles.stats}>
              {LANDING_STAT_ORDER.map((k) => (
                <div key={k} className={styles.stat}>
                  <div className={styles.statN}>{LANDING_STATS[k]}</div>
                  <div className={styles.statL}>{t(`stats.${k}`)}</div>
                </div>
              ))}
            </div>
            <div className={styles.roadmap}>{rich('hero.roadmap')}</div>
          </div>
        </div>
      </div>

      {/* §3 — M.I.A */}
      <MiaSection />

      {/* §4 — INTERACTIVE DEMO */}
      <section id="demo" style={{ paddingTop: 0 }}><div className={styles.wrap}>
        <div className={styles.sechead}>
          <div className={styles.eyebrow}>{t('demoSection.eyebrow')}</div>
          <h2>{t('demoSection.title')}</h2>
          <p>{t('demoSection.subtitle')}</p>
        </div>
        <DemoTabs />
        <div style={{ textAlign: 'center', marginTop: 24 }}>
          <a className={`${styles.btn} ${styles.btnPri} ${styles.btnMd}`} href={lh('/inscription')}>{t('demoSection.cta')}</a>
          <div className={styles.micro} style={{ marginTop: 11 }}>{t('demoSection.illus')}</div>
        </div>
      </div></section>

      {/* §5-8 — TOOLS */}
      <section id="outils"><div className={styles.wrap}>
        <div className={styles.sechead}>
          <div className={styles.eyebrow}>{t('tools.eyebrow')}</div>
          <h2>{t('tools.title')}</h2>
          <p>{t('tools.subtitle')}</p>
        </div>

        {/* READING SPACE + CAROUSEL */}
        <div className={styles.feat}>
          <div className="txt">
            <span className={`${styles.tag} ${styles.tagB}`}>{t('tools.app.tag')}</span>
            <h3>{t('tools.app.h3')}</h3>
            <p className={styles.featP}>{rich('tools.app.p')}</p>
            <Bullets t={t} base="tools.app" count={5} />
            <div className={styles.carHint}>{t('carousel.hint')}</div>
          </div>
          <ReadingCarousel />
        </div>

        {/* SCANNER */}
        <div className={`${styles.feat} ${styles.featRev}`}>
          <div className="txt">
            <span className={`${styles.tag} ${styles.tagG}`}>{t('tools.scanner.tag')}</span>
            <h3>{t('tools.scanner.h3')}</h3>
            <p className={styles.featP}>{rich('tools.scanner.p')}</p>
            <Bullets t={t} base="tools.scanner" />
          </div>
          <VisFrame title={t('tools.scanner.visTitle')}>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 11 }}>
              {['c1', 'c2', 'c3', 'c4'].map((c) => (
                <span key={c} className={styles.chip} style={{ background: 'var(--acc-dim)', borderColor: 'var(--acc)', color: 'var(--acc)' }}>{t(`tools.scanner.chip.${c}`)}</span>
              ))}
            </div>
            <div className={`${styles.rw} ${styles.rwHit}`}><span className={styles.sy}>Au</span><span><span className={styles.nm}>{t('tools.scanner.row.gold15')}</span><br /><span className={styles.mt}>4 027,36</span></span><span className={`${styles.sc} ${styles.scOk}`}>4 / 4</span></div>
            <div className={`${styles.rw} ${styles.rwHit}`}><span className={styles.sy}>€</span><span><span className={styles.nm}>{t('tools.scanner.row.euro15')}</span><br /><span className={styles.mt}>1,0842</span></span><span className={`${styles.sc} ${styles.scOk}`}>4 / 4</span></div>
            <div className={styles.rw}><span className={styles.sy}>Au</span><span><span className={styles.nm}>{t('tools.scanner.row.goldH4')}</span><br /><span className={styles.mt}>4 026,90</span></span><span className={styles.sc} style={{ color: 'var(--faint)' }}>2 / 4</span></div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--faint)', marginTop: 10, textAlign: 'center' }}>{t('tools.scanner.foot')}</div>
            <div className={styles.illus}>{t('demo.illus')}</div>
          </VisFrame>
        </div>

        {/* ZONES */}
        <div className={styles.feat}>
          <div className="txt">
            <span className={`${styles.tag} ${styles.tagV}`}>{t('tools.zones.tag')}</span>
            <h3>{t('tools.zones.h3')}</h3>
            <p className={styles.featP}>{rich('tools.zones.p')}</p>
            <Bullets t={t} base="tools.zones" />
          </div>
          <VisFrame title={t('tools.zones.visTitle')}>
            <div style={{ border: '1px solid var(--line)', borderRadius: 10, padding: '12px 13px', background: 'var(--panel-2)', marginBottom: 8 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 9, marginBottom: 4, flexWrap: 'wrap' }}>
                <span className={styles.chip} style={{ background: 'rgba(55,185,140,.14)', borderColor: 'transparent', color: 'var(--bull)' }}>{t('tools.zones.card1.tag')}</span>
                <span className="mono" style={{ fontSize: 11.5, fontWeight: 600 }}>4 026,80 – 4 028,90</span>
                <span className={styles.chip} style={{ marginLeft: 'auto', background: 'rgba(55,185,140,.14)', borderColor: 'var(--bull)', color: 'var(--bull)' }}>{t('tools.zones.card1.state')}</span>
              </div>
              <div className={styles.tl}><i style={{ background: 'var(--acc)' }} /><s style={{ background: 'var(--line-2)' }} /><i style={{ border: '1.5px solid var(--line-2)' }} /></div>
              <div className={styles.tlab}><span>{t('tools.zones.card1.formed')}</span><span>{t('tools.zones.now')}</span></div>
            </div>
            <div style={{ border: '1px solid var(--line)', borderRadius: 10, padding: '12px 13px', background: 'var(--panel-2)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 9, marginBottom: 4, flexWrap: 'wrap' }}>
                <span className={styles.chip} style={{ background: 'rgba(167,139,250,.14)', borderColor: 'transparent', color: 'var(--fvg-l)' }}>{t('tools.zones.card2.tag')}</span>
                <span className="mono" style={{ fontSize: 11.5, fontWeight: 600 }}>4 020,10 – 4 021,85</span>
                <span className={styles.chip} style={{ marginLeft: 'auto', background: 'rgba(214,162,74,.14)', borderColor: 'var(--liq)', color: 'var(--liq)' }}>{t('tools.zones.card2.state')}</span>
              </div>
              <div className={styles.fill}><i style={{ width: '60%' }} /></div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9.5, color: 'var(--faint)', marginTop: 5 }}>{t('tools.zones.card2.fill')}</div>
            </div>
            <div className={styles.illus}>{t('demo.illus')}</div>
          </VisFrame>
        </div>

        {/* NEWS */}
        <div className={`${styles.feat} ${styles.featRev}`}>
          <div className="txt">
            <span className={`${styles.tag} ${styles.tagA}`}>{t('tools.news.tag')}</span>
            <h3>{t('tools.news.h3')}</h3>
            <p className={styles.featP}>{rich('tools.news.p')}</p>
            <Bullets t={t} base="tools.news" />
          </div>
          <VisFrame title={t('tools.news.visTitle')}>
            <div className={styles.qgrid}>
              {['q1', 'q2', 'q3', 'q4'].map((q) => (
                <div key={q} className={styles.qc}>
                  <div className={styles.qq}>{t(`tools.news.${q}.q`)}</div>
                  <div className={styles.qa}>{rich(`tools.news.${q}.a`)}</div>
                </div>
              ))}
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9.5, color: 'var(--faint)', marginTop: 11, lineHeight: 1.6 }}>{t('tools.news.foot')}</div>
            <div className={styles.illus}>{t('demo.illus')}</div>
          </VisFrame>
        </div>
      </div></section>

      {/* HOW IT WORKS */}
      <section id="comment"><div className={styles.wrap}>
        <div className={styles.sechead}>
          <div className={styles.eyebrow}>{t('how.eyebrow')}</div>
          <h2>{t('how.title')}</h2>
          <p>{t('how.subtitle')}</p>
        </div>
        <div className={styles.steps}>
          {['1', '2', '3'].map((n) => (
            <div key={n} className={styles.step}>
              <div className={styles.stepN}>0{n}</div>
              <h4>{t(`how.step${n}.h`)}</h4>
              <p>{t(`how.step${n}.p`)}</p>
            </div>
          ))}
        </div>
      </div></section>

      {/* WHO */}
      <section><div className={styles.wrap}>
        <div className={styles.sechead}>
          <div className={styles.eyebrow}>{t('who.eyebrow')}</div>
          <h2>{t('who.title')}</h2>
          <p>{t('who.subtitle')}</p>
        </div>
        <div className={styles.who}>
          {['1', '2', '3'].map((n) => {
            const items = Object.values(t.raw(`who.w${n}.items`) as Record<string, string>);
            return (
              <div key={n} className={styles.wc}>
                <span className={styles.wcLb}>{t(`who.w${n}.lb`)}</span>
                <h4>{t(`who.w${n}.h`)}</h4>
                <p>{t(`who.w${n}.p`)}</p>
                <ul className={styles.wcList}>
                  {items.map((it, idx) => <li key={idx}>{it}</li>)}
                </ul>
              </div>
            );
          })}
        </div>
      </div></section>

      {/* DISTINGUISH */}
      <section id="honnetete" style={{ paddingTop: 0 }}><div className={styles.wrap}><div className={styles.narrow}>
        <div className={styles.distinguish}>
          <div className={styles.eyebrow}>{t('distinguish.eyebrow')}</div>
          <h2>{t('distinguish.h2')}</h2>
          <p>{rich('distinguish.p1')}</p>
          <p>{rich('distinguish.p2')}</p>
        </div>
      </div></div></section>

      {/* PRICING */}
      <section id="tarifs"><div className={styles.wrap}>
        <div className={styles.sechead}>
          <div className={styles.eyebrow}>{t('pricing.eyebrow')}</div>
          <h2>{t('pricing.title')}</h2>
          <p>{t('pricing.subtitle')}</p>
        </div>
        <div className={styles.prices}>
          <div className={`${styles.pc} ${styles.pcHi}`}>
            <span className={styles.pcLbl}>{t('pricing.paid.label')}</span>
            <h4>{t('pricing.paid.name')}</h4>
            <div className={styles.pcSub}>{t('pricing.paid.sub')}</div>
            <div className={styles.amt}><span className={styles.amtV}>{PRICING.monthly} $</span><span className={styles.amtU}>US / {t('pricing.paid.perMonth')}</span></div>
            <div className={styles.bill}>{t('pricing.paid.bill', { total: String(PRICING.annualPerYear), perMonth: String(PRICING.annualPerMonth) })}</div>
            <div className={styles.pfeat}>
              {['1', '2', '3', '4', '5', '6'].map((i) => (<div key={i} className={styles.pf}><Check />{t.rich(`pricing.paid.f${i}`, { b: (c) => <b>{c}</b> })}</div>))}
            </div>
            <a className={`${styles.btn} ${styles.btnPri} ${styles.btnMd}`} style={{ width: '100%' }} href={lh('/inscription')}>{t('pricing.paid.cta')}</a>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--faint)', marginTop: 11, textAlign: 'center' }}>{t('pricing.paid.note')}</div>
          </div>
        </div>
        <div className={styles.plegal}>{t('pricing.legal')}</div>
      </div></section>

      {/* FAQ */}
      <section id="faq"><div className={styles.wrap}>
        <div className={styles.sechead}>
          <div className={styles.eyebrow}>{t('faq.eyebrow')}</div>
          <h2>{t('faq.title')}</h2>
        </div>
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

      {/* FINAL */}
      <section><div className={styles.wrap}><div className={styles.final}>
        <h2>{t('final.h2')}</h2>
        <p>{t('final.p')}</p>
        <div className={styles.ctas}>
          <a className={`${styles.btn} ${styles.btnPri} ${styles.btnLg}`} href={lh('/inscription')}>{t('final.cta1')}</a>
          <a className={`${styles.btn} ${styles.btnLg}`} href="#outils">{t('final.cta2')}</a>
        </div>
      </div></div></section>
    </div>
  );
}
