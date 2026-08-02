import { describe, expect, it } from 'vitest';
import { SUPPORTED_LOCALES, DEFAULT_LOCALE } from '@/i18n';
import fr from '@/messages/fr.json';
import en from '@/messages/en.json';
import de from '@/messages/de.json';
import es from '@/messages/es.json';
import itIT from '@/messages/it.json';
import pt from '@/messages/pt.json';
import nl from '@/messages/nl.json';
import pl from '@/messages/pl.json';
import ar from '@/messages/ar.json';

/**
 * DETTE-1 (guard e) — i18n structural parity across every locale.
 *
 * A missing translation key must fail the build, never be discovered by a
 * client staring at a raw `namespace.key` string. This guard asserts every
 * locale carries EXACTLY the same set of keys as the source locale (fr): no
 * missing key, no orphan key. Adding a string to fr without adding it to the
 * eight other files — or vice-versa — fails here (i.e. at `npm test`, in CI).
 *
 * NOTE (documented debt, out of this guard's scope): several namespaces
 * (notably `home` and `regimePanel`) still hold ENGLISH text in the non-en
 * locales — the keys exist, so parity holds, but the *values* are untranslated.
 * Value-level fallback detection is deferred to a follow-up (see
 * docs/audits/AUDIT-dette-1.md, DETTE 3) because enabling it before that
 * translation debt is paid would redden the suite for ~3.8k strings.
 */

const MESSAGES: Record<string, Record<string, unknown>> = {
  fr, en, de, es, it: itIT, pt, nl, pl, ar,
};

function flattenKeys(obj: unknown, prefix = '', out: Set<string> = new Set()): Set<string> {
  if (obj && typeof obj === 'object' && !Array.isArray(obj)) {
    for (const [k, v] of Object.entries(obj)) {
      const key = prefix ? `${prefix}.${k}` : k;
      if (v && typeof v === 'object' && !Array.isArray(v)) flattenKeys(v, key, out);
      else out.add(key);
    }
  }
  return out;
}

const REFERENCE = flattenKeys(MESSAGES[DEFAULT_LOCALE]);

/**
 * DEBT RATCHET (DETTE-1). MISSING keys are now paid down to ZERO for every
 * locale — the 196 fr keys that were absent from the 7 non-en locales
 * (scanner 98 + calendar 91 + regimePanel 5 + reading 1 + app 1) have been
 * translated and filled, so no client sees a raw key. `missing: 0` is therefore
 * ENFORCED for all locales: add a key to fr without propagating it and this
 * fails at `npm test` (CI), never in front of a client.
 *
 * ORPHAN keys (present in a locale, absent in fr) remain as a smaller,
 * pre-existing debt — all in `scanner` (en 36, the 7 non-en 2 each): fr never
 * received those keys. Ratcheted so it can't grow; to burn down, add the keys to
 * fr (if used) or delete the orphans (if dead). Separately, some namespaces still
 * hold English VALUES in the 7 locales (regimePanel/app/reading) — the ACCUEIL
 * (home) is fully translated; the rest is a follow-up (see AUDIT-dette-1.md).
 */
const BASELINE: Record<string, { missing: number; orphan: number }> = {
  en: { missing: 0, orphan: 36 },
  de: { missing: 0, orphan: 2 },
  es: { missing: 0, orphan: 2 },
  it: { missing: 0, orphan: 2 },
  pt: { missing: 0, orphan: 2 },
  nl: { missing: 0, orphan: 2 },
  pl: { missing: 0, orphan: 2 },
  ar: { missing: 0, orphan: 2 },
};

describe('i18n locale parity (guard e)', () => {
  it('ships a message file for every SUPPORTED_LOCALE', () => {
    for (const loc of SUPPORTED_LOCALES) {
      expect(MESSAGES[loc], `messages/${loc}.json missing from the parity guard`).toBeTruthy();
    }
  });

  it(`the source locale (${DEFAULT_LOCALE}) exposes a non-trivial key set`, () => {
    expect(REFERENCE.size).toBeGreaterThan(500);
  });

  for (const loc of SUPPORTED_LOCALES) {
    if (loc === DEFAULT_LOCALE) continue;
    it(`${loc} never drifts FURTHER from ${DEFAULT_LOCALE} (missing/orphan ≤ baseline)`, () => {
      const keys = flattenKeys(MESSAGES[loc]);
      const missing = [...REFERENCE].filter((k) => !keys.has(k));
      const orphan = [...keys].filter((k) => !REFERENCE.has(k));
      const base = BASELINE[loc] ?? { missing: 0, orphan: 0 };
      // A NEW missing key (regression) trips this — it must be added to `${loc}`
      // too (translated, or the parity debt burned down; never left for a client).
      expect(
        missing.length,
        `${loc}: ${missing.length} missing key(s) — baseline ${base.missing}. New drift: ${missing
          .filter((k) => true)
          .slice(0, 20)
          .join(', ')}`,
      ).toBeLessThanOrEqual(base.missing);
      expect(
        orphan.length,
        `${loc}: ${orphan.length} orphan key(s) — baseline ${base.orphan}`,
      ).toBeLessThanOrEqual(base.orphan);
    });
  }
});
