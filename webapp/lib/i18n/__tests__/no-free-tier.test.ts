import { describe, expect, it } from 'vitest';
import { SUPPORTED_LOCALES } from '@/i18n';
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
 * PAY-2 (mission §3.A) — NO free tier, NO free trial, in any visible string.
 *
 * The product is paid-only: paying is the condition of entry. There is no free
 * plan, no trial, no "try for free" offer. This guard scans the MARKETING /
 * PRODUCT-OFFER namespaces (the landing `home`, the top-nav `nav`, and the
 * legacy top-level `pricing` block) of EVERY locale for free-tier / free-trial
 * language and fails the build if any resurfaces.
 *
 * Scope note: we scan only the offer-bearing namespaces on purpose. Words like
 * "trial" legitimately appear elsewhere as a Stripe *status* label ("trialing"),
 * "Essaie" as a demo imperative ("try the demo"), and "Free-form input" in the
 * chat — none of those offer free ACCESS to the product, so they are out of
 * scope. Add a free-tier phrase to the landing and this test goes red at
 * `npm test` (in CI), never in front of a customer.
 */

const MESSAGES: Record<string, Record<string, unknown>> = {
  fr, en, de, es, it: itIT, pt, nl, pl, ar,
};

// Namespaces that speak to the visitor about what they get and what it costs.
const OFFER_NAMESPACES = ['home', 'nav', 'pricing'] as const;

// Free-tier / free-trial language across the nine shipped locales. Each pattern
// targets an OFFER of free access — not incidental words (see scope note).
const FORBIDDEN: { re: RegExp; label: string }[] = [
  { re: /gratuit/i, label: 'gratuit* (fr/es/it/pt)' },
  { re: /gr[aá]tis/i, label: 'gratis / grátis (es/it/pt/de/nl)' },
  { re: /\bfree\b/i, label: 'free (en)' },
  { re: /free\s*(trial|tier|account|plan|forever)/i, label: 'free trial/tier/account (en)' },
  { re: /\btrial\b/i, label: 'trial (en)' },
  { re: /try\s+(it\s+)?for\s+free/i, label: 'try for free (en)' },
  { re: /kostenlos/i, label: 'kostenlos (de)' },
  { re: /(darmow|za\s+darmo)/i, label: 'darmowy / za darmo (pl)' },
  { re: /prueba\s+grat/i, label: 'prueba gratis (es)' },
  { re: /no\s+credit\s+card/i, label: 'no credit card (en)' },
  { re: /sans\s+carte\s+de\s+cr[eé]dit/i, label: 'sans carte de crédit (fr)' },
  { re: /(sin\s+tarjeta|senza\s+carta|ohne\s+kreditkarte|geen\s+creditcard|bez\s+karty|sem\s+cart[aã]o)/i, label: 'no-card offer (es/it/de/nl/pl/pt)' },
  { re: /مجاني|مجانا/, label: 'free (ar)' },
];

function collectStrings(obj: unknown, path: string, out: { path: string; value: string }[]): void {
  if (typeof obj === 'string') {
    out.push({ path, value: obj });
  } else if (obj && typeof obj === 'object' && !Array.isArray(obj)) {
    for (const [k, v] of Object.entries(obj)) {
      collectStrings(v, path ? `${path}.${k}` : k, out);
    }
  }
}

describe('PAY-2 — no free tier / free trial in visible offer strings', () => {
  it('ships a message file for every supported locale', () => {
    for (const loc of SUPPORTED_LOCALES) {
      expect(MESSAGES[loc], `messages/${loc}.json missing`).toBeTruthy();
    }
  });

  it('the free plan block was removed from the landing (home.pricing.free)', () => {
    for (const [loc, msg] of Object.entries(MESSAGES)) {
      const pricing = (msg as any)?.home?.pricing;
      expect(pricing, `${loc}: home.pricing missing`).toBeTruthy();
      expect(
        pricing.free,
        `${loc}: home.pricing.free must be gone (no free plan)`,
      ).toBeUndefined();
    }
  });

  for (const [loc, msg] of Object.entries(MESSAGES)) {
    it(`${loc}: no free-tier / free-trial language in offer namespaces`, () => {
      const strings: { path: string; value: string }[] = [];
      for (const ns of OFFER_NAMESPACES) {
        collectStrings((msg as any)[ns], ns, strings);
      }
      const hits: string[] = [];
      for (const { path, value } of strings) {
        for (const { re, label } of FORBIDDEN) {
          if (re.test(value)) {
            hits.push(`${path}: "${value}"  ⟵ ${label}`);
          }
        }
      }
      expect(hits, `${loc} free-tier language found:\n${hits.join('\n')}`).toEqual([]);
    });
  }
});
