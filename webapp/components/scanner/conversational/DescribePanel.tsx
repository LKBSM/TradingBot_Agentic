'use client';

import { useTranslations } from 'next-intl';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { MicButton } from '@/components/dictation/MicButton';
import { useVoiceInput } from '@/lib/scanner-chat/use-voice-input';
import { useDictationCopy } from '@/lib/scanner-chat/use-dictation-copy';

const MAX_TEXT = 500;

/**
 * State 1 — "Décrire". A large command surface (mono prompt marker, dictation
 * button hidden when the browser cannot dictate — no dead button), six clickable
 * numbered examples, an access to saved strategies, and a plain statement of what
 * M.I.A does: she translates toward verifiable conditions, she chooses nothing.
 *
 * Visual direction V3 ("Signature"): a mono eyebrow, a two-line title whose
 * second line is the accent, and a framed console. Everything stays on the six
 * `--fs-*` steps and the shadcn role tokens, so it themes across all four skins
 * and honours the UI-2 typographic guard (see tests/e2e/ui2-audit.spec.ts).
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

  // Shared voice-input adapter — the exact same hook the /app, /zones and
  // /actualites M.I.A chats use. It only appends to the field the user controls.
  const voice = useVoiceInput({
    locale,
    value: text,
    onValueChange: onTextChange,
    maxLength: MAX_TEXT,
  });
  const dictationCopy = useDictationCopy();

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
    <div className="space-y-4">
      {/* Eyebrow + two-line title (second line = accent). All on the fs-* scale. */}
      <div>
        <p className="flex items-center gap-2.5 fs-label font-mono uppercase tracking-[0.2em] text-primary">
          <span aria-hidden className="h-px w-5 bg-primary/70" />
          {t('describe.eyebrow')}
        </p>
        <h1 className="mt-2.5 fs-title font-bold leading-tight tracking-tight text-foreground">
          {t('describe.title')}
          <br />
          <span className="text-primary">{t('describe.titleAccent')}</span>
        </h1>
        <p className="mt-2 fs-secondary text-muted-foreground">{t('describe.subtitle')}</p>
      </div>

      {/* Command console. */}
      <div className="overflow-hidden rounded-2xl border border-border bg-card shadow-sm">
        <div className="flex items-center gap-2.5 border-b border-border px-4 py-3">
          <span
            aria-hidden
            className="grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-gradient-to-br from-primary to-primary/60 fs-legal font-bold text-primary-foreground"
          >
            M
          </span>
          <div className="min-w-0 flex-1 leading-tight">
            <div className="fs-secondary font-semibold text-foreground">{t('describe.miaTitle')}</div>
            <div data-testid="mia-sub" className="fs-legal text-muted-foreground">{t('describe.miaSub')}</div>
          </div>
        </div>

        <div className="p-4 sm:p-5">
          <div
            className={cn(
              'flex items-start gap-2.5 rounded-xl border bg-muted p-3 transition',
              voice.listening ? 'border-primary ring-2 ring-primary/20' : 'border-border',
            )}
          >
            <span aria-hidden className="mt-0.5 select-none font-mono fs-section leading-none text-primary">
              ›
            </span>
            <textarea
              data-testid="describe-input"
              value={text}
              maxLength={MAX_TEXT}
              onChange={(e) => onTextChange(e.target.value)}
              placeholder={t('describe.placeholder')}
              aria-label={t('describe.title')}
              className="min-h-[64px] flex-1 resize-none bg-transparent font-mono fs-secondary leading-relaxed text-foreground outline-none placeholder:text-muted-foreground/70"
            />
            {voice.supported && (
              <MicButton
                listening={voice.listening}
                denied={voice.denied}
                onToggle={voice.toggle}
                startLabel={dictationCopy.startLabel}
                stopLabel={dictationCopy.stopLabel}
              />
            )}
          </div>

          {/* Live listening + transcript feedback — NEVER hidden. */}
          {voice.listening && (
            <p data-testid="dictation-listening" className="mt-2 fs-legal text-primary">
              {dictationCopy.listeningLabel}
              {voice.interim ? <span className="text-muted-foreground"> — “{voice.interim}”</span> : null}
            </p>
          )}
          {voice.error && (
            <p data-testid="dictation-error" role="alert" className="mt-2 fs-label text-amber-600">
              {dictationCopy.errorText(voice.error)}
            </p>
          )}

          {/* The action is the point of this page — buttons first, then the field's
              small caption. */}
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <Button data-testid="translate-button" disabled={!canTranslate} onClick={onTranslate}>
              {isTranslating ? t('describe.translating') : t('describe.translate')}
            </Button>
            <Button variant="outline" onClick={onOpenStrategies}>
              {t('describe.myStrategies')}
            </Button>
          </div>

          {/* What M.I.A can turn your words into — a plain sentence, proportional. */}
          <p data-testid="scope-note" className="mt-3 fs-legal leading-relaxed text-muted-foreground">
            {t('describe.scope')}
          </p>

          {/* Browser-transcription notice — legal, readable, no longer dominant. */}
          {voice.supported && (
            <p data-testid="transcription-note" className="mt-1 fs-legal leading-relaxed text-muted-foreground/80">
              {dictationCopy.privacy}
            </p>
          )}

          {inlineError && (
            <p data-testid="translate-inline-error" role="alert" className="mt-2 fs-label text-destructive">
              {inlineError}
            </p>
          )}
        </div>
      </div>

      {/* Examples — kept in a two-column grid so the whole page still fits the
          first fold (UI-1b density guard). The chip's text is EXACTLY the example
          phrase (no decorative prefix) so a click composes the field verbatim. */}
      <div>
        <div
          data-testid="examples-label"
          className="mb-2 flex items-center gap-2.5 fs-label uppercase tracking-widest text-muted-foreground"
        >
          <span aria-hidden className="h-px w-5 bg-border" />
          {t('describe.examplesLabel')}
        </div>
        <div className="grid gap-2 sm:grid-cols-2">
          {examples.map((example, i) => {
            const active = text.includes(example);
            return (
              <button
                key={i}
                type="button"
                data-testid="example-chip"
                aria-pressed={active}
                onClick={() => onTextChange(toggleExample(text, example))}
                className={cn(
                  'rounded-lg border p-2.5 text-left fs-label leading-snug transition',
                  active
                    ? 'border-primary bg-primary/10 text-foreground ring-1 ring-primary/40'
                    : 'border-border bg-muted/50 text-muted-foreground hover:border-primary/50 hover:text-foreground',
                )}
              >
                {example}
              </button>
            );
          })}
        </div>
      </div>

      <p className="fs-legal leading-relaxed text-muted-foreground">{t('describe.disclaimer')}</p>
    </div>
  );
}

const EXAMPLE_SEPARATOR = ', ';

/**
 * Additive composition of the example chips. Clicking an example APPENDS its
 * text (comma-joined); clicking it again REMOVES exactly that phrase and its
 * separator. Manual input is never overwritten — examples are added after it.
 *
 * The active state is derived from the text itself (`text.includes(example)`),
 * so the textarea stays the single source of truth: what is sent to « Traduire
 * ma stratégie » is exactly the visible concatenation, with no hidden transform.
 * The example sentences are mutually non-substring (verified fr+en), so no
 * example can falsely read as active because another contains it.
 */
function toggleExample(text: string, phrase: string): string {
  if (text.includes(phrase)) {
    let next = text;
    if (next.includes(`${EXAMPLE_SEPARATOR}${phrase}`)) {
      next = next.replace(`${EXAMPLE_SEPARATOR}${phrase}`, '');
    } else if (next.includes(`${phrase}${EXAMPLE_SEPARATOR}`)) {
      next = next.replace(`${phrase}${EXAMPLE_SEPARATOR}`, '');
    } else {
      next = next.replace(phrase, '');
    }
    return next.trim();
  }
  const base = text.replace(/\s+$/, '');
  return base ? `${base}${EXAMPLE_SEPARATOR}${phrase}` : phrase;
}
