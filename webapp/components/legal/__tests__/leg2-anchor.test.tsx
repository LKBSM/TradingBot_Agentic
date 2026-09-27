import { render, screen, waitFor } from '@/components/test-utils';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { LegalDocument } from '../LegalDocument';

/**
 * LEG-2 — `/conditions#remboursement` has to LAND on the clause.
 *
 * Two things stood between the link and the clause, and both are covered here:
 *
 *   1. the heading had no `id` at all (the renderer emitted none), so the anchor
 *      pointed at nothing;
 *   2. the document is fetched in an effect AFTER hydration. By the time the
 *      clause exists, the browser has long finished with the hash and will not
 *      scroll on its own — the component has to do it.
 *
 * Without (2) the footer link and the one next to the payment button would look
 * like they work (right URL, right page) and quietly drop the reader at the top
 * of a ten-clause contract.
 */

const TERMS_MD = [
  '# M.I.A Markets — Conditions d\'utilisation',
  '',
  '## 7. Prix et facturation',
  '',
  'Du texte.',
  '',
  '## 8. Résiliation et remboursement {#remboursement}',
  '',
  "Tu peux résilier à tout moment, aussi simplement que tu t'es abonné.",
  '',
  '### 8.1 Résilier ton abonnement',
  '',
  'Depuis ton espace client.',
  '',
].join('\n');

function mockDocument(markdown = TERMS_MD) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    headers: { get: () => '2026-09-27' },
    text: async () => markdown,
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

beforeEach(() => {
  window.location.hash = '';
  // jsdom implements no scrolling at all; the spy is how we observe the jump.
  Element.prototype.scrollIntoView = vi.fn();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('LEG-2 — the refund clause is reachable by its anchor', () => {
  it('gives the clause the id the public links point at', async () => {
    mockDocument();
    render(<LegalDocument doc="terms" />);

    const heading = await screen.findByText('8. Résiliation et remboursement');
    expect(heading.id).toBe('remboursement');
    // The declaration itself never reaches the reader.
    expect(heading.textContent).not.toContain('{#');
  });

  it('scrolls to the clause when the URL arrives with the hash', async () => {
    window.location.hash = '#remboursement';
    mockDocument();
    render(<LegalDocument doc="terms" />);

    await waitFor(() =>
      expect(Element.prototype.scrollIntoView).toHaveBeenCalledTimes(1),
    );
    // It is the clause that was scrolled to, not just anything.
    const call = (Element.prototype.scrollIntoView as ReturnType<typeof vi.fn>).mock
      .instances[0] as HTMLElement;
    expect(call.id).toBe('remboursement');
  });

  it('scrolls nowhere when the URL carries no hash', async () => {
    mockDocument();
    render(<LegalDocument doc="terms" />);

    await screen.findByText('8. Résiliation et remboursement');
    expect(Element.prototype.scrollIntoView).not.toHaveBeenCalled();
  });

  it('survives a hash pointing at a clause that does not exist', async () => {
    window.location.hash = '#clause-inventee';
    mockDocument();
    render(<LegalDocument doc="terms" />);

    // The document still renders; nothing throws, nothing scrolls.
    await screen.findByText('8. Résiliation et remboursement');
    expect(Element.prototype.scrollIntoView).not.toHaveBeenCalled();
  });
});
