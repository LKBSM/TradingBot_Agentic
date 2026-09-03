import { describe, expect, it } from 'vitest';
import fr from '@/messages/fr.json';
import en from '@/messages/en.json';

/**
 * CLN-1 (2e passe) — the trading/advice disclaimer clause was removed from the
 * two widget microcopy lines (the /app docked chat and the /scanner/decrire M.I.A
 * note), because the SINGLE page disclaimer (rail footer / mobile footer) already
 * carries it. Each line keeps its own NON-disclaimer content (what M.I.A does /
 * the scanner-specific « ne classe rien, ne devine aucune condition » honesty).
 *
 * This guard fails if the removed clause creeps back, or if the surviving
 * content is emptied.
 */

type Dict = Record<string, unknown>;
function get(root: Dict, path: string): string {
  const v = path.split('.').reduce<unknown>((o, k) => (o as Dict)?.[k], root);
  if (typeof v !== 'string') throw new Error(`Missing/invalid key: ${path}`);
  return v;
}

describe('CLN-1 widget microcopy — no duplicated trading/advice disclaimer', () => {
  it('the /app chat line keeps its description, drops « ni signal / recommandation »', () => {
    const line = { fr: get(fr as Dict, 'app.chat.complianceLine'), en: get(en as Dict, 'app.chat.complianceLine') };
    expect(line.fr.length).toBeGreaterThan(0);
    expect(line.en.length).toBeGreaterThan(0);
    // Descriptive content stays.
    expect(line.fr.toLowerCase()).toContain('lecture algorithmique');
    expect(line.en.toLowerCase()).toContain('algorithmic reading');
    // The disclaimer clause is gone (the footer carries it).
    expect(line.fr.toLowerCase()).not.toContain('signal');
    expect(line.fr.toLowerCase()).not.toContain('recommand');
    expect(line.en.toLowerCase()).not.toContain('signal');
    expect(line.en.toLowerCase()).not.toContain('recommend');
  });

  it('the /scanner note keeps « ne classe rien / ne devine », drops the advice clause', () => {
    const line = { fr: get(fr as Dict, 'scannerChat.describe.disclaimer'), en: get(en as Dict, 'scannerChat.describe.disclaimer') };
    expect(line.fr.length).toBeGreaterThan(0);
    expect(line.en.length).toBeGreaterThan(0);
    // Scanner-specific honesty (NOT covered by the footer) stays.
    expect(line.fr.toLowerCase()).toContain('ne classe rien');
    expect(line.fr.toLowerCase()).toContain('ne devine');
    expect(line.en.toLowerCase()).toContain('orders nothing');
    // The advice clause (« ne conseille rien » / « advises nothing ») is gone.
    expect(line.fr.toLowerCase()).not.toContain('conseille');
    expect(line.en.toLowerCase()).not.toContain('advises');
  });
});
