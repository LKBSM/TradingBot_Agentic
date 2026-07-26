import { render, screen } from '@/components/test-utils';
import { describe, expect, it } from 'vitest';
import fr from '@/messages/fr.json';
import { EarlyAccessBadge } from '@/components/shared/DisclaimerStub';
import { MarketReadingCard } from '@/components/market-reading/MarketReadingCard';
import { FIXTURE_XAU_M15 } from '@/lib/market-reading/fixtures';

/**
 * UI-2b i18n guard — no unresolved translation key must ever reach the screen.
 * next-intl renders a missing key as its dotted path (e.g. `legal.earlyAccessBadge`
 * — the exact bug this mission fixed). We scan the rendered App reading surface
 * for any token shaped like a dotted i18n key and fail if one appears.
 */

// A token that is a lowercase-initial dotted identifier (namespace.key[.key…]),
// with no space/@/slash/colon — the signature of a raw i18n key, not real copy
// ("XAU/USD", "M.I.A", "2 398,40", emails/URLs are all excluded).
const RAW_KEY = /^[a-z][a-zA-Z0-9]*(\.[a-zA-Z][a-zA-Z0-9]*)+$/;

function unresolvedKeysIn(el: HTMLElement): string[] {
  // Walk TEXT NODES individually: a missing key renders as a standalone node
  // whose entire content is the dotted path. Scanning `textContent` instead would
  // glue adjacent elements ("…investissement." + "Phase…") into false positives.
  const out: string[] = [];
  const walker = el.ownerDocument.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  let node = walker.nextNode();
  while (node) {
    const t = (node.textContent ?? '').trim();
    if (t && !t.includes('@') && !t.includes('/') && !t.includes(':') && RAW_KEY.test(t)) {
      out.push(t);
    }
    node = walker.nextNode();
  }
  return out;
}

function get(path: string): unknown {
  return path.split('.').reduce<unknown>((o, k) => (o as Record<string, unknown>)?.[k], fr);
}

describe('UI-2b — no raw i18n keys on the App surface', () => {
  it('EarlyAccessBadge resolves to real copy (not legal.earlyAccessBadge)', () => {
    render(<EarlyAccessBadge />);
    expect(screen.getByText('Accès anticipé')).toBeInTheDocument();
  });

  it('the App reading card renders no raw i18n key', () => {
    const { container } = render(
      <MarketReadingCard reading={FIXTURE_XAU_M15} chartSlot={<div data-testid="chart" />} />,
    );
    expect(unresolvedKeysIn(container)).toEqual([]);
  });

  it('the keys this mission touched all resolve in the fr bundle', () => {
    for (const key of [
      'legal.earlyAccessBadge',
      'footer.earlyAccessBadge',
      'nav.account',
      'app.chat.openPanel',
    ]) {
      expect(get(key), key).toBeTypeOf('string');
      expect((get(key) as string).length, key).toBeGreaterThan(0);
      // The resolved value must NOT itself look like a key path.
      expect(RAW_KEY.test(get(key) as string), `${key} resolved to a key-like value`).toBe(false);
    }
    // The nav entry is now "Réglages", never "Compte".
    expect(get('nav.account')).toBe('Réglages');
    // The early-access badge carries no seat-count scarcity claim (a digit run
    // like the old "· NN seats" leak in the non-FR locales).
    expect(get('footer.earlyAccessBadge')).not.toMatch(/\d/);
    expect(get('legal.earlyAccessBadge')).not.toMatch(/\d/);
  });
});
