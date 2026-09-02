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
  'scanner.builder.zeroNote', // zero condition ≠ all markets
  'scanner.strategyPanel.subtitle', // « non synchronisé »
  'scannerChat.describe.disclaimer', // regulatory notice on /scanner/decrire
  'zones.proximity.distanceLine', // distance fact (kept; only visually receded)
  // CLN-1 §5 — the single per-page educational/legal disclaimer. The former
  // per-surface duplicates (scanner.combo.disclaimer, zones.mia.disclaimer,
  // app.desktop.legalInline) were retired: one notice per page, carried by the
  // rail footer (desktop) and the mobile footer (< 768px). Its PRESENCE is what
  // this guard protects, now at its single home.
  'legal.disclaimer.chart', // the one educational/legal disclaimer per page
  // /actualites/[eventId] — 2nd pass
  'calendar.detail.nono.title', // regulatory notice on the publication page
  'calendar.detail.nono.items.0',
  'calendar.detail.nono.items.1',
  'calendar.detail.nono.items.2',
  'calendar.detail.actualPending', // 3 distinct absence states
  'calendar.detail.actualUnfetched',
  'calendar.detail.actualUnavailable',
  'calendar.pub.mia.capability', // M.I.A non-advice honesty (kept)
  'calendar.pub.questions.readGuide.body', // « décompte, pas une probabilité » honesty
  'calendar.pub.curve.note', // unit / « MIA ne l'estime pas » honesty
  'calendar.pub.source.intro', // scoping line (always shown)
  'calendar.pub.source.onlyNote', // « aucun site de commentaire ni de prévision » honesty
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
  // /actualites/[eventId] — the source note dropped its duplicate lead sentence
  // (« Ces liens mènent à l'organisme… et à lui seul », already carried by
  // pub.source.intro) and kept the unique « no forecast site » honesty.
  'calendar.pub.source.onlyNote': {
    fr: "Ces liens mènent à l'organisme officiel et à lui seul. MIA ne renvoie vers aucun site de commentaire ni de prévision : choisir un tel lien, ce serait le recommander.",
    en: 'These links lead to the official organism and to it alone. MIA points to no commentary or forecast site: picking such a link would be recommending it.',
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

  it('the three publication absence states stay distinct in both languages', () => {
    for (const root of [fr, en] as const) {
      const s = [
        get(root as Dict, 'calendar.detail.actualPending'),
        get(root as Dict, 'calendar.detail.actualUnfetched'),
        get(root as Dict, 'calendar.detail.actualUnavailable'),
      ];
      expect(new Set(s).size, `absence states must differ: ${s.join(' | ')}`).toBe(3);
    }
  });

  it('the source note keeps its « no forecast site » honesty after the trim', () => {
    expect(get(fr as Dict, 'calendar.pub.source.onlyNote')).toContain('site de commentaire ni de prévision');
    expect(get(en as Dict, 'calendar.pub.source.onlyNote')).toContain('commentary or forecast site');
  });
});
