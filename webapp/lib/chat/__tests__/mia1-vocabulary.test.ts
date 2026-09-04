import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

/**
 * MIA-1 — the suggestions and the waiting/activity messages must never carry
 * predictive or prescriptive vocabulary, in ANY locale. M.I.A describes; it
 * never tells the user to act, judges a moment, or projects a move. This guards
 * the exact strings shipped in messages/*.json (fr + en required by the mission,
 * plus the other seven for good measure).
 */

// Forbidden stems (fr + en), matched case-insensitively as substrings. Kept to
// unambiguous directive/predictive vocabulary so descriptive words stay allowed.
const FORBIDDEN = [
  // action de trading
  'achète', 'achete', 'achetez', 'vends', 'vendez', 'buy', 'sell',
  // recommandation / prescription
  'recommande', 'conseille', 'tu devrais', 'vous devriez', 'you should',
  'il faut', 'évite', 'evite',
  // jugement de moment / prédiction
  'bon moment', 'mauvais moment', 'opportunité', 'opportunite', 'opportunity',
  'setup parfait', 'va chercher', 'will reach', 'will go', 'va monter', 'va baisser',
  // jugement de risque / garantie
  'risqué', 'risque de', 'dangereux', 'garanti', 'guaranteed', 'safe bet',
];

const LOCALES = ['fr', 'en', 'es'] as const;

/** Keys under the `chat` and `app.chat` namespaces that are shown as
 * suggestions or waiting/activity messages — the surfaces the mission pins. */
function collectStrings(msg: Record<string, any>): { key: string; value: string }[] {
  const out: { key: string; value: string }[] = [];
  const chat = msg.chat ?? {};
  const appChat = msg.app?.chat ?? {};
  const push = (ns: string, obj: Record<string, any>, keys: string[]) => {
    for (const k of keys) {
      if (typeof obj[k] === 'string') out.push({ key: `${ns}.${k}`, value: obj[k] });
    }
  };
  // Waiting / activity messages (chat namespace).
  push('chat', chat, [
    'thinking',
    'activityReadingMarket',
    'activityReading',
    'activityDiagnostic',
  ]);
  // Starter suggestions (both the docked /app panel and the slide-over).
  push('app.chat', appChat, ['starter_structure', 'starter_choch', 'starter_order-blocks']);
  // Any `suggested*` / `starter*` keys under chat, defensively.
  for (const [k, v] of Object.entries(chat)) {
    if (typeof v === 'string' && (k.startsWith('starter') || k.startsWith('suggest'))) {
      out.push({ key: `chat.${k}`, value: v });
    }
  }
  return out;
}

describe('MIA-1 — suggestions & waiting messages vocabulary', () => {
  for (const loc of LOCALES) {
    it(`${loc}: no predictive/prescriptive word in suggestions or waiting messages`, () => {
      const msg = JSON.parse(readFileSync(`messages/${loc}.json`, 'utf8'));
      const strings = collectStrings(msg);
      // Sanity: we actually found the activity strings we just added.
      expect(strings.some((s) => s.key === 'chat.activityReadingMarket')).toBe(true);
      for (const { key, value } of strings) {
        const hay = value.toLowerCase();
        for (const bad of FORBIDDEN) {
          expect(hay.includes(bad), `${loc} ${key} contains "${bad}": ${value}`).toBe(false);
        }
      }
    });
  }
});
