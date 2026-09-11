'use client';

import { useEffect, useRef, useState, type ReactNode } from 'react';
import { useLocale, useTranslations } from 'next-intl';
import styles from './lp1.module.css';
import { DemoQuotaError, applyDemoViewActions, askDemoMia } from '@/lib/landing/demo-chat';
import { StructureChart, type Layers } from './structure-chart';
import { DEMO_MIA } from './data';

/** One exchange in the M.I.A tab. `scripted` marks a recorded fallback answer,
 * which is labelled as such rather than passed off as a live reply. */
interface MiaExchange {
  question: string;
  answer: ReactNode;
  refusal: boolean;
  scripted: boolean;
  showsChart: boolean;
}

/**
 * MIA-4S — the M.I.A pane talks to the REAL agent (same orchestrator, same four
 * defence layers) over the public showcase endpoint, on the frozen illustration
 * scenario and nothing else.
 *
 * The prompts on the right are STARTERS, not a menu: any question can be typed.
 * When the endpoint is unavailable (no backend, quota reached, network blocked)
 * the pane degrades to the recorded exchanges — clearly labelled as recorded,
 * never presented as a live answer.
 *
 * LP-3 moved it out of the tab strip and gave it its own block, because the
 * refusal is the product's sharpest claim and it was buried behind a tab. Its
 * view actions now drive the SAME chart the reader watched assemble itself
 * further up the page: `setLayers` writes the page-level manual override, and
 * the CTA scrolls back to it instead of switching tabs.
 */
export function MiaPane({
  layers,
  setLayers,
  jumpToReading,
}: {
  layers: Layers;
  setLayers: (l: Layers) => void;
  jumpToReading: () => void;
}) {
  const t = useTranslations('home');
  const locale = useLocale();
  const [thread, setThread] = useState<MiaExchange[]>([]);
  const [draft, setDraft] = useState('');
  const [pending, setPending] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [closed, setClosed] = useState(false);
  const [left, setLeft] = useState<number | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Only once there is something to scroll TO. This pane used to live inside
    // a tab and mounted on click, so an unconditional scrollIntoView was
    // harmless; on LP-3 it mounts with the page and yanked a visitor who had
    // not touched anything from the opening down to the chat.
    if (thread.length === 0 && !pending) return;
    endRef.current?.scrollIntoView?.({ block: 'nearest' });
  }, [thread, pending]);

  const rich = (key: string): ReactNode => t.rich(key, { b: (c) => <b>{c}</b> });

  /** The recorded exchange for a starter — the offline fallback, and the only
   * place the scripted answers survive. */
  const scriptedFor = (question: string): MiaExchange | null => {
    const ex = DEMO_MIA.find((d) => t(`demo.mia.q.${d.key}.q`) === question);
    if (!ex) return null;
    if (ex.action) {
      const next: Layers = { ob: false, fvg: false, liq: false, str: false };
      ex.action.only.forEach((k) => { next[k] = true; });
      setLayers(next);
    }
    return {
      question,
      answer: rich(`demo.mia.q.${ex.key}.a`),
      refusal: ex.kind === 'refusal',
      scripted: true,
      showsChart: Boolean(ex.action),
    };
  };

  const ask = async (question: string) => {
    const trimmed = question.trim();
    if (!trimmed || pending || closed) return;
    setDraft('');
    setPending(true);
    setNotice(null);
    const history = thread.flatMap((x) => [
      { role: 'user' as const, content: x.question },
      { role: 'assistant' as const, content: typeof x.answer === 'string' ? x.answer : '' },
    ]).filter((m) => m.content);

    try {
      const answer = await askDemoMia({ question: trimmed, history, locale });
      setLeft(answer.messagesLeft);
      const next = applyDemoViewActions(layers, answer.viewActions);
      const changed = answer.viewActions.length > 0;
      if (changed) setLayers(next);
      setThread((prev) => [...prev, {
        question: trimmed,
        answer: answer.text,
        refusal: Boolean(answer.blockedReason),
        scripted: false,
        showsChart: changed,
      }]);
    } catch (err) {
      if (err instanceof DemoQuotaError) {
        // A quota is a product rule, not a failure: say it plainly and stop.
        setThread((prev) => [...prev, {
          question: trimmed,
          answer: err.message,
          refusal: false,
          scripted: false,
          showsChart: false,
        }]);
        setClosed(true);
        setLeft(0);
      } else {
        // No backend here (static host, offline, blocked): fall back to the
        // recorded exchange when the question is one of the starters, and say
        // that it is recorded.
        const fallback = scriptedFor(trimmed);
        setNotice(t('demo.mia.live.offline'));
        if (fallback) setThread((prev) => [...prev, fallback]);
      }
    } finally {
      setPending(false);
    }
  };

  return (
    <div className={styles.dgrid}>
      <div>
        <div className={styles.mchat}>
          {thread.length === 0 && !pending && (
            <div className={styles.mbA} style={{ opacity: 0.7 }}>{t('demo.mia.greeting')}</div>
          )}
          {thread.map((x, idx) => (
            <div key={`${idx}-${x.question}`} style={{ display: 'contents' }}>
              <div className={styles.mbU} style={{ alignSelf: 'flex-end' }}>{x.question}</div>
              <div className={x.refusal ? styles.mbNo : styles.mbA} style={{ alignSelf: 'flex-start' }}>
                {x.answer}
              </div>
              {x.showsChart && (
                <div style={{ alignSelf: 'stretch' }}>
                  <div className={styles.mbAction}>{t('demo.mia.actionNote')}</div>
                  <StructureChart layers={layers} />
                  <button type="button" className={styles.opt} style={{ marginTop: '8px', marginBottom: 0 }} onClick={jumpToReading}>
                    {t('demo.mia.actionCta')}
                  </button>
                </div>
              )}
            </div>
          ))}
          {pending && (
            <div className={styles.mbA} style={{ alignSelf: 'flex-start', opacity: 0.7 }} aria-live="polite">
              {t('demo.mia.live.thinking')}
            </div>
          )}
          <div ref={endRef} />
        </div>
        <form
          className={styles.mform}
          onSubmit={(e) => { e.preventDefault(); void ask(draft); }}
        >
          <input
            className={styles.minput}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder={t('demo.mia.live.placeholder')}
            aria-label={t('demo.mia.live.placeholder')}
            maxLength={600}
            disabled={closed}
          />
          <button type="submit" className={styles.msend} disabled={pending || closed || !draft.trim()}>
            {t('demo.mia.live.send')}
          </button>
        </form>
        {notice && <div className={styles.illus}>{notice}</div>}
        {left != null && !notice && (
          <div className={styles.illus}>{t('demo.mia.live.left', { n: left })}</div>
        )}
        <div className={styles.illus}>{t('demo.illus')}</div>
      </div>
      <div className={styles.dside}>
        {/* `demo.mia.live.note` used to say "this is the real M.I.A, on
            illustration data" right here. LP-3's `refuse.lead` says it in the
            block heading, two paragraphs above — so this column drops it rather
            than repeating it, which is the redundancy LP-2A went after. */}
        <div className={styles.try}>{t('demo.mia.side.try')}</div>
        {DEMO_MIA.map((ex) => (
          <button
            key={ex.key}
            type="button"
            className={styles.opt}
            disabled={pending || closed}
            onClick={() => void ask(t(`demo.mia.q.${ex.key}.q`))}
          >
            {t(`demo.mia.q.${ex.key}.q`)}
          </button>
        ))}
      </div>
    </div>
  );
}
