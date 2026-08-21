import { describe, expect, it } from 'vitest';
import fr from '@/messages/fr.json';
import en from '@/messages/en.json';
import es from '@/messages/es.json';
import de from '@/messages/de.json';
import itMsg from '@/messages/it.json';
import nl from '@/messages/nl.json';
import pl from '@/messages/pl.json';
import pt from '@/messages/pt.json';
import ar from '@/messages/ar.json';

/**
 * Voice-dictation copy-honesty guard (mission "Dictée vocale").
 *
 * The dictation block (scannerChat.dictation) is reused verbatim by EVERY M.I.A
 * chat surface — /app, /zones, /actualites and the scanner. Two guarantees must
 * hold in all 9 locales:
 *
 *  1. The privacy note must describe the REAL mechanism. The Web Speech API is
 *     not guaranteed on-device (Chrome/Edge send audio to the browser vendor's
 *     servers), so the note must name the browser AND acknowledge server transit
 *     — and must never claim the transcription is purely local / on-device.
 *
 *  2. Because the copy is shared, the error strings must be surface-neutral:
 *     they invite the user back to the keyboard, never to "type your strategy"
 *     (which only made sense on the scanner).
 */

const LOCALES = { fr, en, es, de, it: itMsg, nl, pl, pt, ar } as Record<string, Record<string, any>>;

// Per-locale proof-words: the privacy note must mention the browser AND servers.
const MECHANISM_WORDS: Record<string, [browser: string, server: string]> = {
  fr: ['navigateur', 'serveurs'],
  en: ['browser', 'servers'],
  es: ['navegador', 'servidores'],
  de: ['Browser', 'Server'],
  it: ['browser', 'server'],
  nl: ['browser', 'servers'],
  pl: ['przeglądarkę', 'serwery'],
  pt: ['navegador', 'servidores'],
  ar: ['متصفحك', 'خوادمه'],
};

// The "strategy" word per locale — must NOT appear in the shared error copy.
const STRATEGY_WORD: Record<string, string> = {
  fr: 'stratégie',
  en: 'strategy',
  es: 'estrategia',
  de: 'Strategie',
  it: 'strategia',
  nl: 'strategie',
  pl: 'strategię',
  pt: 'estratégia',
  ar: 'استراتيجيتك',
};

// Phrases that would falsely claim on-device transcription (checked where we can
// reason about the language — the two we author natively).
const PURELY_LOCAL_LIES: Record<string, string[]> = {
  fr: ['entièrement local', 'sur ton appareil', 'aucune donnée ne quitte'],
  en: ['purely local', 'on-device', 'on your device', 'never leaves your device'],
};

function dictation(locale: string) {
  const d = LOCALES[locale]?.scannerChat?.dictation;
  if (!d) throw new Error(`missing scannerChat.dictation for ${locale}`);
  return d as {
    start: string;
    stop: string;
    listening: string;
    privacy: string;
    errors: Record<string, string>;
  };
}

describe('dictation copy — present and complete in all 9 locales', () => {
  for (const locale of Object.keys(LOCALES)) {
    it(`${locale} has start/stop/listening/privacy + 6 error codes`, () => {
      const d = dictation(locale);
      for (const k of ['start', 'stop', 'listening', 'privacy'] as const) {
        expect(d[k], `${locale}.${k}`).toBeTypeOf('string');
        expect(d[k].length, `${locale}.${k}`).toBeGreaterThan(0);
      }
      for (const code of ['not-allowed', 'no-speech', 'audio-capture', 'network', 'timeout', 'unknown']) {
        expect(d.errors[code], `${locale}.errors.${code}`).toBeTypeOf('string');
        expect(d.errors[code].length, `${locale}.errors.${code}`).toBeGreaterThan(0);
      }
    });
  }
});

describe('dictation privacy note — describes the REAL browser mechanism', () => {
  for (const locale of Object.keys(LOCALES)) {
    it(`${locale} names the browser and acknowledges server transit`, () => {
      const note = dictation(locale).privacy;
      const [browserWord, serverWord] = MECHANISM_WORDS[locale];
      expect(note, `${locale} privacy must mention the browser`).toContain(browserWord);
      expect(note, `${locale} privacy must acknowledge server transit`).toContain(serverWord);
    });
  }

  for (const [locale, lies] of Object.entries(PURELY_LOCAL_LIES)) {
    it(`${locale} privacy never claims purely-local / on-device transcription`, () => {
      const note = dictation(locale).privacy.toLowerCase();
      for (const lie of lies) {
        expect(note.includes(lie.toLowerCase()), `${locale} must not claim « ${lie} »`).toBe(false);
      }
    });
  }
});

describe('dictation errors — surface-neutral (no "type your strategy")', () => {
  for (const locale of Object.keys(LOCALES)) {
    it(`${locale} error copy invites the keyboard, not "the strategy"`, () => {
      const { errors } = dictation(locale);
      const strategyWord = STRATEGY_WORD[locale].toLowerCase();
      for (const [code, text] of Object.entries(errors)) {
        expect(
          text.toLowerCase().includes(strategyWord),
          `${locale}.errors.${code} must be surface-neutral (no « ${STRATEGY_WORD[locale]} »)`,
        ).toBe(false);
      }
    });
  }
});
