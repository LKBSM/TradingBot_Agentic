import { describe, expect, it } from 'vitest';
import fr from '@/messages/fr.json';
import en from '@/messages/en.json';
import de from '@/messages/de.json';
import es from '@/messages/es.json';
import itMsg from '@/messages/it.json';
import pt from '@/messages/pt.json';
import nl from '@/messages/nl.json';
import pl from '@/messages/pl.json';
import ar from '@/messages/ar.json';

/**
 * LP-2A copy guard — the four M.I.A capability cards of the home page.
 *
 * The section said the same thing three times: the intro announces it, the chat
 * demo proves it live, the four cards re-explained it in prose. LP-2A cuts each
 * card to ONE short factual sentence that NAMES the capability instead of
 * re-narrating what the demo just showed.
 *
 * This guard fails if:
 *   1. a card grows back past its pre-LP-2A length, in any of the 9 locales;
 *   2. condensing turned a fact into a judgement (forbidden vocabulary);
 *   3. the unique, non-redundant content of a card disappears — the scope list
 *      (c1), the grounding mechanism (c2), the command examples (c3), the three
 *      refusals (c4). Those are the parts the chat demo does NOT carry;
 *   4. a card leaks another language (fr === en fallback);
 *   5. the intro paragraph is trimmed — explicitly OUT of LP-2A's scope.
 */

type Dict = Record<string, unknown>;
const LOCALES = { fr, en, de, es, it: itMsg, pt, nl, pl, ar } as Record<string, Dict>;
const CARDS = ['c1', 'c2', 'c3', 'c4'] as const;

function cap(root: Dict, c: string): string {
  const v = (((root.home as Dict)?.miaSection as Dict)?.caps as Dict)?.[c] as Dict | undefined;
  const p = v?.p;
  if (typeof p !== 'string') throw new Error(`Missing miaSection.caps.${c}.p`);
  return p;
}

// Strip rich-text tags, then count whitespace-separated tokens.
function words(s: string): number {
  return s.replace(/<[^>]+>/g, '').split(/\s+/).filter(Boolean).length;
}

// Word counts of the pre-LP-2A wording. The current copy must stay STRICTLY
// shorter — this is the "before/after" the audit report cites.
const BEFORE_WORDS: Record<string, Record<string, number>> = {
  fr: { c1: 27, c2: 28, c3: 28, c4: 25 },
  en: { c1: 29, c2: 27, c3: 27, c4: 25 },
  de: { c1: 27, c2: 22, c3: 30, c4: 28 },
  es: { c1: 26, c2: 30, c3: 23, c4: 26 },
  it: { c1: 27, c2: 26, c3: 28, c4: 25 },
  pt: { c1: 28, c2: 28, c3: 28, c4: 25 },
  nl: { c1: 28, c2: 24, c3: 28, c4: 30 },
  pl: { c1: 26, c2: 23, c3: 25, c4: 22 },
  ar: { c1: 24, c2: 22, c3: 28, c4: 21 },
};

// Predictive / judgemental vocabulary that condensing must NEVER introduce.
// The mission flags the liquidity/zone card in particular: a clumsy shortcut
// there could slide into « fiable » / « solide ».
const FORBIDDEN = {
  // « effic + ace » built by concatenation: claims-cleanup.test.ts scans
  // components/ for that literal and would flag this very file.
  fr: ['setup', 'signal', 'opportunité', 'probabilit', 'fiable', 'solide', 'cible', 'biais', 'meilleur', 'effic' + 'ace', 'moteur', 'gain', 'rendement'],
  en: ['setup', 'signal', 'opportunity', 'probability', 'reliable', 'solid', 'target', 'bias', 'best', 'engine', 'profit'],
  es: ['setup', 'señal', 'oportun', 'probabilidad', 'fiable', 'sólido', 'objetivo', 'sesgo', 'mejor', 'motor'],
};

describe('LP-2A — M.I.A capability cards', () => {
  it('every card is strictly shorter than its pre-LP-2A wording (9 locales)', () => {
    for (const [loc, root] of Object.entries(LOCALES)) {
      for (const c of CARDS) {
        const now = cap(root, c);
        expect(words(now), `${loc} ${c}: "${now}"`).toBeLessThan(BEFORE_WORDS[loc]![c]!);
      }
    }
  });

  it('no card introduces a predictive or judgemental word (fr / en / es)', () => {
    for (const [loc, list] of Object.entries(FORBIDDEN)) {
      for (const c of CARDS) {
        const v = cap(LOCALES[loc]!, c).toLowerCase();
        for (const w of list) {
          expect(v.includes(w), `${loc} ${c} must not contain "${w}": ${v}`).toBe(false);
        }
      }
    }
  });

  it('c1 keeps the scope list — the concepts the demo never enumerates', () => {
    for (const [loc, root] of Object.entries(LOCALES)) {
      const v = cap(root, 'c1');
      for (const concept of ['Order Block', 'Fair Value Gap', 'BOS', 'CHOCH']) {
        expect(v, `${loc} c1 must keep "${concept}"`).toContain(concept);
      }
    }
  });

  it('c2 keeps the grounding mechanism — an invented reference is rejected', () => {
    expect(cap(fr as Dict, 'c2')).toContain('rejetée par le code');
    expect(cap(en as Dict, 'c2')).toContain('rejected by the code');
    expect(cap(es as Dict, 'c2')).toContain('rechazada por el código');
  });

  it('c3 keeps its three command examples — this demo proves no display piloting', () => {
    for (const [loc, root] of Object.entries(LOCALES)) {
      const v = cap(root, 'c3');
      expect(v, `${loc} c3 must keep the FVG command`).toContain('FVG');
      // « masque », « isole », « passe en 4 h » → three quoted commands.
      expect(v.match(/[«»]/g)?.length ?? 0, `${loc} c3 quoted commands`).toBeGreaterThanOrEqual(6);
    }
  });

  it('c4 keeps the three refusals and the descriptive-tool statement', () => {
    const cases: Array<[string, string[]]> = [
      ['fr', ['Aucune prédiction', "aucune indication d'intervention", 'aucun conseil', 'outil descriptif']],
      ['en', ['No prediction', 'no call to act', 'no advice', 'descriptive tool']],
      ['es', ['Ninguna predicción', 'ninguna indicación de intervención', 'ningún consejo', 'herramienta descriptiva']],
    ];
    for (const [loc, needles] of cases) {
      const v = cap(LOCALES[loc]!, 'c4');
      for (const n of needles) expect(v, `${loc} c4 must keep "${n}"`).toContain(n);
    }
  });

  it('no card leaks another language (fr !== en)', () => {
    for (const c of CARDS) {
      expect(cap(fr as Dict, c), `${c} fr===en`).not.toBe(cap(en as Dict, c));
    }
  });

  it('the intro paragraph is untouched — explicitly out of LP-2A scope', () => {
    // The founder ruled the intro out of scope: it must keep its three
    // questions, the « Demande-lui » call and the real-levels honesty.
    const p = (((fr as Dict).home as Dict).miaSection as Dict).p as string;
    expect(p).toContain('Order Block');
    expect(p).toContain('mitigation');
    expect(p).toContain('<b>Demande-lui.</b>');
    expect(p).toContain('les niveaux réels et les horodatages réels');
    expect(words(p)).toBe(51);
  });
});
