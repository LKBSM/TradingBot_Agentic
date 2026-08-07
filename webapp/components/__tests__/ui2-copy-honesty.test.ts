import { describe, expect, it } from 'vitest';
import fr from '@/messages/fr.json';

/**
 * UI-2 copy-honesty guard (mission UI-2). The reference reproduction must stay
 * descriptive: no predictive promise, no "Trader" CTA, "Analyser" for the analyse
 * action. This test scans the microcopy keys UI-2 introduced and fails if any of
 * them ever start promising direction / outcome.
 *
 * Nuance: the Scanner note deliberately QUOTES the forbidden phrases to REFUSE
 * them ("Toute condition prédictive est refusée — pas de « va rebondir », pas de
 * cible."). That single key is whitelisted for the phrase scan and separately
 * asserted to remain a refusal — its presence is a feature, not a leak.
 */

// The exact set of keys UI-2 added (kept explicit so this guard scans only the
// new surface, never pre-existing copy it doesn't own).
const NEW_UI2_KEYS = [
  'app.layers.ob', 'app.layers.fvg', 'app.layers.liquidity', 'app.layers.breaks', 'app.layers.mitigated',
  'app.desktop.live', 'app.desktop.earlyAccess', 'app.desktop.legalInline',
  'app.desktop.narratedTitle', 'app.desktop.narratedBadge', 'app.desktop.narratedFooter',
  'app.desktop.structureFooter',
  'app.desktop.reg.trend', 'app.desktop.reg.volatility', 'app.desktop.reg.maturity',
  'app.desktop.reg.alignment', 'app.desktop.reg.lastEvent', 'app.desktop.reg.density',
  'app.desktop.reg.disagreement',
  'app.desktop.liq.badge', 'app.desktop.liq.empty', 'app.desktop.liq.intactDesc',
  'app.desktop.liq.sweptDesc', 'app.desktop.liq.brokenDesc',
  'app.desktop.news.title', 'app.desktop.news.badge', 'app.desktop.news.empty', 'app.desktop.news.scheduledOn',
  'scanner.page.title', 'scanner.page.subtitle', 'scanner.livebadge', 'scanner.activeStrategy',
  'scanner.note', 'scanner.resultsMeta', 'scanner.unavailable',
  'zones.badge.count', 'zones.header.untouched', 'zones.header.consumed', 'zones.header.neverFilled',
  'zones.timeline.now', 'zones.groups.inside',
  'app.account.title', 'app.account.subtitle', 'app.account.sectionAccount', 'app.account.emailRow',
  'app.account.edit', 'app.account.editCancel', 'app.account.passwordRow', 'app.account.passwordMasked',
  'app.account.comingSoon', 'app.account.sessionRow', 'app.account.sessionValue',
  'app.account.sectionAppearance', 'app.account.appearanceBadge', 'app.account.sectionSubscription',
  'app.account.planRow', 'app.account.planValue', 'app.account.earlyAccessBadge',
  'app.account.sectionLegal', 'app.account.docsRow', 'app.account.docsValue', 'app.account.consult',
  'app.account.exportRow', 'app.account.exportValue', 'app.account.export',
  'app.account.deleteRow', 'app.account.deleteValue', 'app.account.delete',
  'connexion.title', 'connexion.subtitle', 'connexion.or', 'connexion.trust',
] as const;

// Phrases that would turn a descriptive reading into a prediction / trade call.
const FORBIDDEN = [
  'va rebondir',
  'setup gagnant',
  'cible',
  'biais',
  "signal d'achat",
  'signal de vente',
];

// The one key that is ALLOWED to quote the forbidden phrases, because it refuses
// them rather than promising them.
const REFUSAL_KEYS = new Set(['scanner.note']);

function get(path: string): string {
  const v = path.split('.').reduce<unknown>((o, k) => (o as Record<string, unknown>)?.[k], fr);
  if (typeof v !== 'string') throw new Error(`Missing/invalid fr key: ${path}`);
  return v;
}

describe('UI-2 copy honesty', () => {
  it('every new UI-2 key resolves to a French string', () => {
    for (const key of NEW_UI2_KEYS) {
      expect(get(key), key).toBeTypeOf('string');
      expect(get(key).length, key).toBeGreaterThan(0);
    }
  });

  it('no new UI-2 copy makes a predictive/directional promise', () => {
    for (const key of NEW_UI2_KEYS) {
      if (REFUSAL_KEYS.has(key)) continue;
      const value = get(key).toLowerCase();
      for (const phrase of FORBIDDEN) {
        expect(value.includes(phrase.toLowerCase()), `${key} must not contain « ${phrase} »`).toBe(false);
      }
    }
  });

  it('the Scanner note keeps quoting the forbidden phrases only to REFUSE them', () => {
    const note = get('scanner.note').toLowerCase();
    expect(note).toContain('refusée');
    expect(note).toContain('va rebondir');
    expect(note).toContain('cible');
  });

  it('never labels an action « Trader » — the analyse CTA is « Analyser »', () => {
    for (const key of NEW_UI2_KEYS) {
      expect(get(key)).not.toMatch(/\bTrader\b/);
    }
    // The analyse action (Scanner combo) is worded "Analyser". VZ-1 removed the
    // Zones-card « Analyser » CTA — selecting the card is the gesture now.
    const analyseCtas = ['scanner.combo.analyse'];
    const found = analyseCtas
      .map((k) => {
        try {
          return get(k);
        } catch {
          return '';
        }
      })
      .filter(Boolean);
    expect(found.length, 'expected at least one "Analyser" CTA key present').toBeGreaterThan(0);
    for (const label of found) expect(label).toMatch(/Analyser/);
  });
});
