/**
 * LEG-1 — guards on the legal texts themselves.
 *
 * The documents live in `docs/legal/*.{fr,en,es}.md` and are served verbatim.
 * Nothing in the build reads them, so nothing would catch a divergence between
 * the three languages, or a promise hardened in one of them only. This file is
 * that catch.
 *
 * Rule of the mission: product speech may say what the tool does NOT do; it may
 * never say what a market is GOING to do, and it may never harden the refund
 * promise ("final sale", "non-refundable", …) in any language.
 */
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

import fr from '@/messages/fr.json';
import en from '@/messages/en.json';
import de from '@/messages/de.json';
import es from '@/messages/es.json';
// `it` would collide with vitest's own `it`.
import itIT from '@/messages/it.json';
import pt from '@/messages/pt.json';
import nl from '@/messages/nl.json';
import pl from '@/messages/pl.json';
import ar from '@/messages/ar.json';

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const LEGAL_DIR = path.join(REPO_ROOT, 'docs', 'legal');

/** The locales we publish a legal text in (NOT the nine UI locales). */
const LEGAL_LOCALES = ['fr', 'en', 'es'] as const;
type LegalLocale = (typeof LEGAL_LOCALES)[number];

const DOCUMENTS = {
  terms: 'conditions-utilisation',
  privacy: 'politique-confidentialite',
} as const;
type DocKind = keyof typeof DOCUMENTS;

function docPath(doc: DocKind, locale: LegalLocale): string {
  return path.join(LEGAL_DIR, `${DOCUMENTS[doc]}.${locale}.md`);
}

function readDoc(doc: DocKind, locale: LegalLocale): string {
  return readFileSync(docPath(doc, locale), 'utf-8');
}

/**
 * The documents are hard-wrapped for review, so a sentence routinely straddles
 * a newline. Assertions about WORDING must therefore run on a whitespace-flat
 * copy — otherwise they pass or fail on where the wrap happens to fall.
 */
function flat(doc: DocKind, locale: LegalLocale): string {
  return readDoc(doc, locale).replace(/\s+/g, ' ');
}

/** Leading numbers of the `## n. Title` sections, in order. */
function sectionNumbers(markdown: string): number[] {
  return markdown
    .split('\n')
    .map((line) => /^##\s+(\d+)\./.exec(line))
    .filter((m): m is RegExpExecArray => m !== null)
    .map((m) => Number(m[1]));
}

const ALL_BUNDLES: Record<string, unknown> = {
  fr, en, de, es, it: itIT, pt, nl, pl, ar,
};

/** Every string value in a message bundle, flattened. */
function bundleStrings(obj: unknown, acc: string[] = []): string[] {
  if (typeof obj === 'string') acc.push(obj);
  else if (obj && typeof obj === 'object')
    for (const v of Object.values(obj as Record<string, unknown>))
      bundleStrings(v, acc);
  return acc;
}

// =============================================================================
// The documents exist, in all three published languages
// =============================================================================

describe('LEG-1 — the published legal documents', () => {
  for (const doc of Object.keys(DOCUMENTS) as DocKind[]) {
    for (const locale of LEGAL_LOCALES) {
      it(`${doc} exists in ${locale} and is a real document`, () => {
        expect(existsSync(docPath(doc, locale))).toBe(true);
        expect(readDoc(doc, locale).length).toBeGreaterThan(1500);
      });
    }

    it(`${doc}: the three languages carry the SAME sections, in the same order`, () => {
      // A missing clause in one language is the failure mode that matters: a
      // customer reading Spanish must be accepting the same contract.
      const reference = sectionNumbers(readDoc(doc, 'fr'));
      expect(reference.length).toBeGreaterThanOrEqual(9);
      expect(reference).toEqual([...reference].sort((a, b) => a - b));
      for (const locale of LEGAL_LOCALES) {
        expect(sectionNumbers(readDoc(doc, locale)), `${doc}.${locale}`).toEqual(
          reference,
        );
      }
    });

    it(`${doc}: every language states its version and last-updated date`, () => {
      for (const locale of LEGAL_LOCALES) {
        const head = readDoc(doc, locale).slice(0, 400);
        expect(head, `${doc}.${locale}`).toMatch(/\d{4}-\d{2}-\d{2}/);
      }
    });
  }

  it('the terms carry the twelve clauses the mission requires', () => {
    expect(sectionNumbers(readDoc('terms', 'fr'))).toEqual([
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    ]);
  });
});

// =============================================================================
// Forbidden wording — documents AND message bundles
// =============================================================================

/**
 * Hardened refund wording. Forbidden outright: the annual plan carries a 14-day
 * guarantee, and Quebec's Consumer Protection Act prevails regardless — a "final
 * sale" line would contradict both.
 */
const FORBIDDEN_REFUND = [
  'vente finale',
  'non remboursable',
  'non-remboursable',
  'final sale',
  'no refund',
  'non-refundable',
  'nonrefundable',
  'venta final',
  'sin reembolso',
  'no reembolsable',
] as const;

describe('LEG-1 — no hardened refund wording', () => {
  for (const doc of Object.keys(DOCUMENTS) as DocKind[]) {
    for (const locale of LEGAL_LOCALES) {
      it(`${doc}.${locale} is clean`, () => {
        const body = readDoc(doc, locale).toLowerCase();
        for (const banned of FORBIDDEN_REFUND) {
          expect(body.includes(banned), `${doc}.${locale} contient « ${banned} »`).toBe(
            false,
          );
        }
      });
    }
  }

  for (const [locale, bundle] of Object.entries(ALL_BUNDLES)) {
    it(`messages/${locale}.json is clean`, () => {
      const body = bundleStrings(bundle).join('\n').toLowerCase();
      for (const banned of FORBIDDEN_REFUND) {
        expect(body.includes(banned), `${locale}.json contient « ${banned} »`).toBe(
          false,
        );
      }
    });
  }
});

// =============================================================================
// No predictive future about markets
// =============================================================================

describe('LEG-1 — the legal texts never predict a market', () => {
  // Saying what the tool does NOT do is allowed and wanted. Saying where a price
  // is headed is not, in any tense that asserts it.
  const PREDICTIVE: RegExp[] = [
    /\b(?:va|vont) (?:monter|baisser|augmenter|chuter|progresser)\b/i,
    /\ble (?:prix|marché) (?:montera|baissera|augmentera)\b/i,
    /\bprices? will (?:rise|fall|drop|increase|decrease)\b/i,
    /\bthe market will\b/i,
    /\bwill (?:outperform|beat the market)\b/i,
    /\b(?:el precio|el mercado) (?:subirá|bajará|caerá)\b/i,
  ];

  for (const doc of Object.keys(DOCUMENTS) as DocKind[]) {
    for (const locale of LEGAL_LOCALES) {
      it(`${doc}.${locale} carries no predictive claim`, () => {
        const body = flat(doc, locale);
        for (const pattern of PREDICTIVE) {
          expect(pattern.test(body), `${doc}.${locale} : ${pattern}`).toBe(false);
        }
      });
    }
  }

  it('the terms state the opposite: past measures do not carry over', () => {
    expect(flat('terms', 'fr')).toMatch(/ne s'appliquent pas aux situations à venir/);
    expect(flat('terms', 'en')).toMatch(/do not apply to situations still to come/);
    expect(flat('terms', 'es')).toMatch(/No se aplican a las situaciones por venir/);
  });
});

// =============================================================================
// What the terms must say, in all three languages
// =============================================================================

describe('LEG-1 — clauses that must never quietly disappear', () => {
  it('the market-data redistribution bar is present in all three', () => {
    // Our data licence requires us to pass this on — losing it is a breach.
    expect(flat('terms', 'fr')).toMatch(/redistribuer, revendre, republier/);
    expect(flat('terms', 'en')).toMatch(/redistribute, resell, republish/);
    expect(flat('terms', 'es')).toMatch(/redistribuir, revender, republicar/);
  });

  it('the cancellation clause reproduces the imposed wording VERBATIM', () => {
    // This clause was dictated word for word and must not be hardened, softened
    // or reworded — so it is compared in full, not by fragments.
    const IMPOSED =
      "Tu peux résilier à tout moment, aussi simplement que tu t'es abonné, depuis " +
      "ton espace client. Au mensuel, l'accès reste ouvert jusqu'à la fin de la " +
      "période payée ; il n'y a pas de remboursement au prorata. À l'annuel, nous " +
      'offrons une garantie de 14 jours à compter du paiement. Si tu résides au ' +
      "Québec, les droits que t'accorde la Loi sur la protection du consommateur " +
      "s'appliquent intégralement et priment sur ce qui précède.";
    const section = readDoc('terms', 'fr').split('## 8.')[1]?.split('## 9.')[0] ?? '';
    const body = section.replace(/\s+/g, ' ').trim();
    expect(body.slice(body.indexOf('Tu peux'))).toBe(IMPOSED);
  });

  it('the price and the currency are stated, and match the shipped pricing', async () => {
    const { PRICING } = await import('@/lib/pricing.generated');
    for (const locale of LEGAL_LOCALES) {
      const body = readDoc('terms', locale);
      expect(body, `terms.${locale}`).toContain(String(PRICING.monthly));
      expect(body, `terms.${locale}`).toContain(String(PRICING.annualPerYear));
      expect(body, `terms.${locale}`).toContain('USD');
    }
  });

  it('no processor we do not use is named in the privacy policy', () => {
    // Clerk was explicitly rejected; Telegram is not part of the subscription
    // product. Naming either would be a false declaration of sub-processors.
    for (const locale of LEGAL_LOCALES) {
      const body = flat('privacy', locale);
      expect(body, `privacy.${locale}`).not.toContain('Clerk');
      expect(body, `privacy.${locale}`).not.toContain('Telegram');
    }
  });

  it('the privacy policy never claims a completed privacy assessment', () => {
    // None has been carried out. "Under way" is true; "was carried out" is not.
    const claims = [
      /évaluation[^.]{0,80}a été réalisée/i,
      /assessment[^.]{0,80}has been (?:carried out|completed)/i,
      /evaluación[^.]{0,80}(?:ha sido realizada|se ha realizado)/i,
    ];
    for (const locale of LEGAL_LOCALES) {
      const body = flat('privacy', locale);
      for (const claim of claims) {
        expect(claim.test(body), `privacy.${locale} : ${claim}`).toBe(false);
      }
    }
  });
});

// =============================================================================
// Consent screen copy
// =============================================================================

describe('LEG-1 — consent screen copy', () => {
  for (const [locale, bundle] of Object.entries(ALL_BUNDLES)) {
    it(`${locale}: the consent label carries BOTH document links`, () => {
      const billing = (
        bundle as {
          billing: { consent: { label: string; blocked: string; error: string } };
        }
      ).billing.consent;
      expect(billing.label).toContain('<terms>');
      expect(billing.label).toContain('</terms>');
      expect(billing.label).toContain('<privacy>');
      expect(billing.label).toContain('</privacy>');
      expect(billing.blocked.length).toBeGreaterThan(10);
      expect(billing.error.length).toBeGreaterThan(10);
    });
  }
});
