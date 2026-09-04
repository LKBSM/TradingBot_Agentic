import { describe, expect, it } from 'vitest';
import { SUPPORTED_LOCALES } from '@/i18n';
import fr from '@/messages/fr.json';
import en from '@/messages/en.json';
import es from '@/messages/es.json';

/**
 * I18N-1 — the WITNESS-WORD guard: prove that each shipped locale renders ONLY
 * its own language. Key parity (locale-parity.test.ts) proves every slot is
 * FILLED; this proves each slot is filled in the RIGHT language — that no English
 * string leaked into the Spanish bundle, no French into the English one, etc.
 *
 * A silent fallback to another language is an interface lie (mission §0). The
 * repo had ~220 English strings sitting in the Spanish bundle (the auth / billing
 * / zones surfaces were authored after the last es translation pass); this guard
 * would have caught them and fails the build if any reappear.
 *
 * Method: a curated list of witness PHRASES unique to each language. A phrase
 * from language X must never appear in the bundle of language Y. Phrases (not
 * bare words) keep cognates — « Accumulation », « Distribution », « Concept »,
 * « Scanner », identical in fr/en — from tripping the guard.
 */

const BUNDLES: Record<string, Record<string, unknown>> = { fr, en, es };

// Developer-only comment keys are never rendered — excluded from the scan.
function flattenValues(obj: unknown, prefix = '', out: string[] = []): string[] {
  if (obj && typeof obj === 'object' && !Array.isArray(obj)) {
    for (const [k, v] of Object.entries(obj)) {
      const key = prefix ? `${prefix}.${k}` : k;
      if (k === '_comment' || key.endsWith('._comment')) continue;
      if (v && typeof v === 'object' && !Array.isArray(v)) flattenValues(v, key, out);
      else if (typeof v === 'string') out.push(v);
    }
  }
  return out;
}

// SMC jargon + brand + provider names + instrument codes are the SAME in every
// language by policy (see docs/product/SMC-JARGON-POLICY.md), so a line carrying
// one of these is exempt from the witness scan.
const EXEMPT =
  /Order Block|Fair Value Gap|\bBOS\b|CHOCH|\bFVG\b|\bOB\b|\bBSL\b|\bSSL\b|M\.I\.A|XAU|EUR\/USD|Google|Stripe|FOMC|JOLTS|Bureau|Census/;

// Witness phrases UNIQUE to each language. Curated to be zero-false-positive
// against the current clean bundles.
const WITNESS: Record<string, RegExp> = {
  en: /\b(your account|you can|with your|sign in|we sent|choose your|back to home|turns on|the assistant|the zone)\b/i,
  fr: /(votre compte|vous pouvez|avec votre|ton adresse|ta formule|retour à l|s'est mal|choisis ta)/i,
  es: /(tu cuenta|puedes|con tu|tu correo|volver al inicio|elige tu|iniciar sesión|se activa)/i,
};

describe('i18n witness-word guard (no cross-language leak)', () => {
  it('ships exactly the three launch locales (fr/en/es)', () => {
    expect([...SUPPORTED_LOCALES].sort()).toEqual(['en', 'es', 'fr']);
  });

  for (const loc of Object.keys(BUNDLES)) {
    const values = flattenValues(BUNDLES[loc]).filter((v) => !EXEMPT.test(v));
    for (const foreign of Object.keys(WITNESS)) {
      if (foreign === loc) continue;
      it(`${loc} bundle contains no ${foreign} witness phrase`, () => {
        const own = WITNESS[loc]!;
        const foreignRe = WITNESS[foreign]!;
        const offenders = values.filter((v) => foreignRe.test(v) && !own.test(v));
        expect(offenders, `${loc} leaks ${foreign}: ${offenders.slice(0, 5).join(' | ')}`).toEqual([]);
      });
    }
  }

  it('keeps SMC jargon in English across ALL locales (never translated)', () => {
    // The jargon must survive verbatim in every bundle where it is explained.
    for (const loc of Object.keys(BUNDLES)) {
      const blob = flattenValues(BUNDLES[loc]).join(' ');
      expect(blob, `${loc} must keep "Order Block"`).toContain('Order Block');
      expect(blob, `${loc} must keep "Fair Value Gap"`).toContain('Fair Value Gap');
    }
  });
});
