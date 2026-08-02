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
 * STRICT PARITY (DETTE-1). Every non-fr locale now carries EXACTLY the same key
 * set as fr — the 196 absent keys were filled, the scanner opt.* divergence was
 * healed (added to fr), and the dead `mtf_aligned` orphans were removed. So this
 * guard enforces zero missing AND zero orphan keys: add a key to fr without
 * propagating it (or leave an orphan) and it fails at `npm test` (in CI), never
 * in front of a client staring at a raw `namespace.key`.
 *
 * (This is KEY parity. A stricter "no English VALUE" check is separate — the
 * ACCUEIL is fully translated and most other namespaces now are too; a few
 * legitimately-same-as-English tokens remain, see AUDIT-dette-1.md.)
 */
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
    it(`${loc} has exactly the same keys as ${DEFAULT_LOCALE} (no missing, no orphan)`, () => {
      const keys = flattenKeys(MESSAGES[loc]);
      const missing = [...REFERENCE].filter((k) => !keys.has(k));
      const orphan = [...keys].filter((k) => !REFERENCE.has(k));
      expect(
        { missing: missing.slice(0, 25), orphan: orphan.slice(0, 25) },
        `${loc}: ${missing.length} missing, ${orphan.length} orphan key(s)`,
      ).toEqual({ missing: [], orphan: [] });
    });
  }
});
