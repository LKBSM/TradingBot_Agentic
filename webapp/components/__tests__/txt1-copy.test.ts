import { describe, expect, it } from 'vitest';
import fr from '@/messages/fr.json';
import en from '@/messages/en.json';

/**
 * TXT-1 copy guard — the textual-load reduction must never turn into a loss of a
 * fact or of a protected statement.
 *
 * The mission trims words that carry NO market information (positioning copy) on
 * the scanner entries, and leaves every fact / denominator / regulatory notice
 * untouched. This guard fails if:
 *   1. a protected statement disappears from either language;
 *   2. a trimmed key stops being SHORTER than its pre-TXT-1 wording (regression);
 *   3. a forbidden (predictive / judgemental) word is introduced while condensing;
 *   4. a trimmed key leaks the other language (fr === en fallback);
 *   5. the proximity line loses a placeholder (pts / pct / edge — three facts);
 *   6. the scope note stops naming what a description can turn into.
 */

type Dict = Record<string, unknown>;
function get(root: Dict, path: string): string {
  const v = path.split('.').reduce<unknown>((o, k) => (o as Dict)?.[k], root);
  if (typeof v !== 'string') throw new Error(`Missing/invalid key: ${path}`);
  return v;
}

// Strip rich-text tags, then count whitespace-separated tokens.
function words(s: string): number {
  return s
    .replace(/<[^>]+>/g, '')
    .split(/\s+/)
    .filter(Boolean).length;
}

// Statements that must stay present, verbatim in spirit, in BOTH languages.
// Their EXACT wording can be compacted; their PRESENCE cannot vanish.
const PROTECTED = [
  'scanner.combo.againstBlock', // « Ce qui va à l'encontre » — never hidden
  'scanner.combo.disclaimer', // regulatory notice on results
  'scanner.builder.zeroNote', // zero condition ≠ all markets
  'scanner.strategyPanel.subtitle', // « non synchronisé »
  'scannerChat.describe.disclaimer', // regulatory notice on /scanner/decrire
  'zones.mia.disclaimer', // regulatory notice on /zones
  'zones.proximity.distanceLine', // distance fact (kept; only visually receded)
  'app.desktop.legalInline', // regulatory notice on /app
] as const;

// The keys TXT-1 trimmed, with their PRE-TXT-1 wording. The current value must be
// strictly shorter (fewer words) — the "before/after" count the report cites.
const TRIMMED: Record<string, { fr: string; en: string }> = {
  'scannerChat.describe.subtitle': {
    fr: "Pas de formulaire à remplir. Écris ce que tu cherches comme tu le dirais à quelqu'un.",
    en: "No form to fill in. Write what you're looking for as you'd say it to someone.",
  },
  'scannerChat.describe.scope': {
    fr: 'Order Blocks, Fair Value Gaps, liquidité, structure, momentum : décris-les avec tes mots, elle en fait des conditions exactes.',
    en: 'Order Blocks, Fair Value Gaps, liquidity, structure, momentum: describe them in your own words, she turns them into exact conditions.',
  },
  'scanner.builder.intro': {
    fr: 'Choisis les faits structurels <b>présents</b> qui composent ta stratégie. Le scanner te montre sur quels marchés et timeframes ils sont réunis <b>en ce moment</b>.',
    en: 'Choose the structural facts <b>present</b> that make up your strategy. The scanner shows you on which markets and timeframes they come together <b>right now</b>.',
  },
};

// Predictive / judgemental vocabulary that condensing must NEVER introduce.
// Multi-char, substring-safe on the trimmed surface. « signal »/« va » live in
// protected keys we do not touch, so they are not scanned here.
const FORBIDDEN_FR = ['cible', 'biais', 'setup', 'opportunité', 'meilleur', 'recommandé', 'probabilité', 'va rebondir'];
const FORBIDDEN_EN = ['target', 'bias', 'setup', 'opportunity', 'best', 'recommended', 'probability'];

describe('TXT-1 copy honesty', () => {
  it('every protected statement resolves in both languages', () => {
    for (const key of PROTECTED) {
      expect(get(fr as Dict, key).length, `fr ${key}`).toBeGreaterThan(0);
      expect(get(en as Dict, key).length, `en ${key}`).toBeGreaterThan(0);
    }
  });

  it('the proximity distance line keeps its three facts (pts, pct, edge)', () => {
    for (const [loc, root] of [['fr', fr], ['en', en]] as const) {
      const line = get(root as Dict, 'zones.proximity.distanceLine');
      for (const ph of ['{pts}', '{pct}', '{side}', '{edge}']) {
        expect(line.includes(ph), `${loc} distanceLine must keep ${ph}`).toBe(true);
      }
    }
  });

  it('trimmed keys are strictly shorter than their pre-TXT-1 wording', () => {
    for (const [key, before] of Object.entries(TRIMMED)) {
      const nowFr = get(fr as Dict, key);
      const nowEn = get(en as Dict, key);
      expect(words(nowFr), `fr ${key}: "${nowFr}"`).toBeLessThan(words(before.fr));
      expect(words(nowEn), `en ${key}: "${nowEn}"`).toBeLessThan(words(before.en));
    }
  });

  it('trimmed keys introduce no forbidden predictive/judgemental word', () => {
    for (const key of Object.keys(TRIMMED)) {
      const vFr = get(fr as Dict, key).toLowerCase();
      for (const w of FORBIDDEN_FR) expect(vFr.includes(w), `fr ${key} « ${w} »`).toBe(false);
      const vEn = get(en as Dict, key).toLowerCase();
      for (const w of FORBIDDEN_EN) expect(vEn.includes(w), `en ${key} « ${w} »`).toBe(false);
    }
  });

  it('trimmed keys do not leak the other language (no fr/en fallback)', () => {
    for (const key of Object.keys(TRIMMED)) {
      expect(get(fr as Dict, key), `${key} fr===en`).not.toBe(get(en as Dict, key));
    }
  });

  it('the scope note still names what a description turns into (Order Blocks)', () => {
    expect(get(fr as Dict, 'scannerChat.describe.scope')).toContain('Order Blocks');
    expect(get(en as Dict, 'scannerChat.describe.scope')).toContain('Order Blocks');
  });
});
