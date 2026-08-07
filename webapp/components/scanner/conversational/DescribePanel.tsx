'use client';

import * as React from 'react';
import { useTranslations } from 'next-intl';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { useSpeechDictation, type DictationError } from '@/lib/scanner-chat/use-speech-dictation';

const MAX_TEXT = 500;

/**
 * State 1 — "Décrire". A large text field, a voice-dictation button (hidden when
 * the browser cannot dictate — no dead button), six clickable examples, an
 * access to saved strategies, and a plain statement of what M.I.A does: she
 * translates toward verifiable conditions, she chooses nothing for you.
 */
export function DescribePanel({
  text,
  onTextChange,
  onTranslate,
  onOpenStrategies,
  isTranslating,
  inlineError,
  locale,
}: {
  text: string;
  onTextChange(value: string): void;
  onTranslate(): void;
  onOpenStrategies(): void;
  isTranslating: boolean;
  inlineError: string | null;
  locale: string;
}) {
  const t = useTranslations('scannerChat');
  const speechLang = locale.toLowerCase().startsWith('en') ? 'en-US' : 'fr-FR';

  const appendFinal = React.useCallback(
    (chunk: string) => {
      onTextChange(joinTranscript(text, chunk).slice(0, MAX_TEXT));
    },
    // text is read at call time via closure refresh below
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [text],
  );

  const dictation = useSpeechDictation({ lang: speechLang, onFinal: appendFinal });

  const examples = [
    t('describe.examples.0'),
    t('describe.examples.1'),
    t('describe.examples.2'),
    t('describe.examples.3'),
    t('describe.examples.4'),
    t('describe.examples.5'),
  ];

  const canTranslate = text.trim().length > 0 && !isTranslating;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-foreground">{t('describe.title')}</h1>
        <p className="mt-1 text-muted-foreground">{t('describe.subtitle')}</p>
      </div>

      <div className="rounded-2xl border border-border/70 bg-card/50 p-4 sm:p-5">
        <div className="mb-3 flex items-center gap-2">
          <span
            aria-hidden
            className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-primary to-primary/60 text-xs font-bold text-primary-foreground"
          >
            M
          </span>
          <div className="leading-tight">
            <div className="text-sm font-semibold text-foreground">{t('describe.miaTitle')}</div>
            <div className="font-mono text-[10px] text-muted-foreground">{t('describe.miaSub')}</div>
          </div>
        </div>

        <div
          className={cn(
            'flex items-start gap-2 rounded-xl border bg-background/70 p-3 transition',
            dictation.listening ? 'border-primary ring-2 ring-primary/20' : 'border-border/70',
          )}
        >
          <textarea
            data-testid="describe-input"
            value={text}
            maxLength={MAX_TEXT}
            onChange={(e) => onTextChange(e.target.value)}
            placeholder={t('describe.placeholder')}
            aria-label={t('describe.title')}
            className="min-h-[64px] flex-1 resize-none bg-transparent text-[15px] leading-relaxed text-foreground outline-none placeholder:text-muted-foreground/70"
          />
          {dictation.supported && (
            <button
              type="button"
              data-testid="mic-button"
              aria-pressed={dictation.listening}
              aria-label={dictation.listening ? t('dictation.stop') : t('dictation.start')}
              title={dictation.listening ? t('dictation.stop') : t('dictation.start')}
              onClick={() => (dictation.listening ? dictation.stop() : dictation.start())}
              className={cn(
                'grid h-9 w-9 shrink-0 place-items-center rounded-lg border transition',
                dictation.listening
                  ? 'border-primary bg-primary/15 text-primary'
                  : 'border-border/70 text-muted-foreground hover:border-primary hover:text-primary',
              )}
            >
              <MicIcon />
            </button>
          )}
        </div>

        {/* Live listening + transcript feedback — NEVER hidden. */}
        {dictation.listening && (
          <p data-testid="dictation-listening" className="mt-2 font-mono text-[11px] text-primary">
            {t('dictation.listening')}
            {dictation.interim ? <span className="text-muted-foreground"> — “{dictation.interim}”</span> : null}
          </p>
        )}
        {dictation.error && (
          <p data-testid="dictation-error" role="alert" className="mt-2 text-[12px] text-amber-600">
            {dictationErrorMessage(dictation.error, t)}
          </p>
        )}
        {dictation.supported && (
          <p className="mt-2 font-mono text-[10px] leading-relaxed text-muted-foreground/80">
            {t('dictation.privacy')}
          </p>
        )}

        <div className="mt-3 flex flex-wrap items-center gap-2">
          <Button data-testid="translate-button" disabled={!canTranslate} onClick={onTranslate}>
            {isTranslating ? t('describe.translating') : t('describe.translate')}
          </Button>
          <Button variant="outline" onClick={onOpenStrategies}>
            {t('describe.myStrategies')}
          </Button>
          <span className="ml-auto font-mono text-[10.5px] text-muted-foreground">{t('describe.scope')}</span>
        </div>

        {inlineError && (
          <p data-testid="translate-inline-error" role="alert" className="mt-2 text-[12px] text-destructive">
            {inlineError}
          </p>
        )}
      </div>

      <div>
        <div className="mb-2 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          {t('describe.examplesLabel')}
        </div>
        <div className="grid gap-2 sm:grid-cols-2">
          {examples.map((example, i) => (
            <button
              key={i}
              type="button"
              data-testid="example-chip"
              onClick={() => onTextChange(example)}
              className="rounded-lg border border-border/60 bg-background/50 p-2.5 text-left text-[13px] text-muted-foreground transition hover:border-primary/50 hover:text-foreground"
            >
              {example}
            </button>
          ))}
        </div>
      </div>

      <p className="text-[12.5px] leading-relaxed text-muted-foreground">{t('describe.disclaimer')}</p>
    </div>
  );
}

function dictationErrorMessage(error: DictationError, t: ReturnType<typeof useTranslations>): string {
  const key = `dictation.errors.${error}`;
  return t.has(key) ? t(key) : t('dictation.errors.unknown');
}

/** Append a dictated chunk with a single separating space, trimming doubles. */
function joinTranscript(current: string, chunk: string): string {
  const base = current.replace(/\s+$/, '');
  if (!base) return chunk;
  return `${base} ${chunk}`.replace(/\s{2,}/g, ' ');
}

function MicIcon() {
  return (
    <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.9" aria-hidden>
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5 11a7 7 0 0014 0M12 18v3" strokeLinecap="round" />
    </svg>
  );
}
