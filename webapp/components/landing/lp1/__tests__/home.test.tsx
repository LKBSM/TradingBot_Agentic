import { render as rtlRender, screen, fireEvent, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { NextIntlClientProvider } from 'next-intl';
import fr from '@/messages/fr.json';
import en from '@/messages/en.json';
import de from '@/messages/de.json';
import es from '@/messages/es.json';
import itMsg from '@/messages/it.json';
import pt from '@/messages/pt.json';
import nl from '@/messages/nl.json';
import pl from '@/messages/pl.json';
import ar from '@/messages/ar.json';
import { HomeLanding } from '../HomeLanding';
import { DemoTabs } from '../DemoTabs';
import { ALL_MARKET_IDS } from '@/lib/markets';

function render(ui: React.ReactElement, locale: 'fr' | 'en' = 'fr') {
  const messages = locale === 'fr' ? fr : en;
  return rtlRender(
    <NextIntlClientProvider locale={locale} messages={messages}>
      {ui}
    </NextIntlClientProvider>,
  );
}

// Recursively collect every string leaf under an object.
function collectStrings(obj: unknown, out: string[] = []): string[] {
  if (typeof obj === 'string') out.push(obj);
  else if (Array.isArray(obj)) obj.forEach((v) => collectStrings(v, out));
  else if (obj && typeof obj === 'object') Object.values(obj).forEach((v) => collectStrings(v, out));
  return out;
}

const FORBIDDEN_FR = [
  'setup', 'signal', 'opportunité', 'gagnant', 'probabilit',
  'meilleur moment', 'ne rate plus', 'gain', 'rendement', 'réussite',
];
const FORBIDDEN_EN = [
  'setup', 'signal', 'opportunity', 'winner', 'winning',
  'probability', 'best time', "don't miss", 'guaranteed', 'profit',
];

describe('LP-1 home — forbidden vocabulary', () => {
  it('the fr home namespace uses no forbidden marketing word', () => {
    const strings = collectStrings((fr as Record<string, unknown>).home).join(' ').toLowerCase();
    for (const word of FORBIDDEN_FR) {
      expect(strings, `fr copy must not contain "${word}"`).not.toContain(word);
    }
  });

  it('the en home namespace uses no forbidden marketing word', () => {
    const strings = collectStrings((en as Record<string, unknown>).home).join(' ').toLowerCase();
    for (const word of FORBIDDEN_EN) {
      expect(strings, `en copy must not contain "${word}"`).not.toContain(word);
    }
  });

  // LP-2: the word "moteur" (fr) / "engine" (en) must never surface in visible
  // home copy — it leaks the internal machinery. Replacements: "MIA détecte",
  // "l'analyse", "la détection automatique", "le produit".
  it('never says "moteur" (fr) or "engine" (en) in the home namespace', () => {
    const frStrings = collectStrings((fr as Record<string, unknown>).home).join(' ').toLowerCase();
    const enStrings = collectStrings((en as Record<string, unknown>).home).join(' ').toLowerCase();
    expect(frStrings, 'fr home must not contain "moteur"').not.toContain('moteur');
    expect(enStrings, 'en home must not contain "engine"').not.toContain('engine');
  });

  // DETTE-1: the 7 non-fr/en locales were natively translated. Guard every locale
  // against the forbidden concepts in their own language (loanwords + the local
  // words for signal / opportunity / probability). This would have caught the
  // Arabic « إشارة » (signal) that slipped into the first translation pass.
  const ALL_LOCALES: Record<string, unknown> = { fr, en, de, es, it: itMsg, pt, nl, pl, ar };
  const FORBIDDEN_XLANG = [
    'setup',
    'signal', 'signaal', 'señal', 'segnale', 'sinal', 'sygnał', 'إشارة',
    'opportun', 'oportun', 'okazja',
    'probabilit', 'probabilidad', 'probabilidade', 'waarschijn', 'wahrscheinlich', 'prawdopodobie', 'احتمال',
  ];
  it('no locale home namespace leaks a signal/opportunity/probability/setup word (9 locales)', () => {
    for (const [loc, msg] of Object.entries(ALL_LOCALES)) {
      const strings = collectStrings((msg as Record<string, unknown>).home).join(' ').toLowerCase();
      for (const word of FORBIDDEN_XLANG) {
        expect(strings, `${loc} home must not contain "${word}"`).not.toContain(word.toLowerCase());
      }
    }
  });
});

/**
 * LP-2S — the four-stat banner (markets / timeframes / conditions / structures)
 * is gone: it told a first-time visitor nothing. One line replaces it, and that
 * line carries a claim that MUST stay honest — 80 is an AMBITION, the perimeter
 * actually in production is XAUUSD + EURUSD.
 *
 * The guard below is deliberately stricter than the old `not.toMatch(/80\s*march/)`:
 * that one banned the figure outright, which cannot express « qualified is fine,
 * bare is a lie ». Here a market count is legal ONLY when a scope word (planned /
 * au programme / previstos / geplant …) sits in the SAME sentence segment. A
 * translator who trims « 80 marchés au programme » down to « 80 marchés » turns
 * a roadmap into a false claim about the live product — and turns this test red.
 */
type Loc = 'fr' | 'en' | 'de' | 'es' | 'it' | 'pt' | 'nl' | 'pl' | 'ar';
const LOCALES: Record<Loc, unknown> = { fr, en, de, es, it: itMsg, pt, nl, pl, ar };

/** The word for « market » in each locale — what makes a number a MARKET COUNT. */
const MARKET_WORD: Record<Loc, RegExp> = {
  fr: /marchés?/i,
  en: /markets?/i,
  de: /märkten?|markt/i,
  es: /mercados?/i,
  it: /mercati|mercato/i,
  pt: /mercados?/i,
  nl: /markten|markt/i,
  pl: /rynk\w*|rynek/i,
  ar: /سوق|أسواق/,
};

/** Scope words — ONLY planning/ambition terms. « up to » alone does not qualify. */
const SCOPE_WORD: Record<Loc, RegExp> = {
  fr: /programme|prévue?s?|visée?s?|objectif|à terme/i,
  en: /planned|targeted|roadmap|goal/i,
  de: /geplant|vorgesehen|ziel/i,
  es: /previstos?|planeados?|objetivo/i,
  it: /previsti|pianificati|obiettivo/i,
  pt: /previstos?|planeados?|objetivo/i,
  nl: /gepland|voorzien|doel/i,
  pl: /plan\w*|docelowo|cel/i,
  ar: /مخطط|مستهدف/,
};

// Arabic scope/market words carry harakat in the copy (مخطَّطة); strip the marks
// so the guard matches the word, not one particular vocalisation of it.
const stripHarakat = (s: string) => s.replace(/[ً-ْٰ]/g, '');

/** Split a string into sentence segments — a scope word only counts inside the
 *  SAME segment as the number, so a qualifier three clauses away cannot launder
 *  a bare count. */
const segments = (s: string) => s.split(/[—–·.!?\n]|<\/?b>/);

/**
 * Every segment that states a market count the copy is not allowed to state.
 *
 * A market count is legitimate in exactly two cases:
 *   1. it equals the perimeter REALLY in production (ALL_MARKET_IDS.length) —
 *      then it is a checkable fact and needs no hedge (« Les 2 marchés »);
 *   2. it is any other number AND a scope word sits in the same segment —
 *      then it reads as an ambition, which is what it is (« 80 au programme »).
 * Anything else is a future number worn as a present one. That is the lie this
 * guard exists to catch, and the reason it survives a copy rewrite: it compares
 * against the registry, not against a hard-coded 2.
 */
function bareMarketCounts(locale: Loc, text: string): string[] {
  const mw = MARKET_WORD[locale].source;
  const scope = SCOPE_WORD[locale];
  // A number adjacent to the market word, in either reading order (Arabic puts
  // the count before the noun too, but keep both so no word order slips through).
  // The `(?![\w-])` guard keeps the word a NOUN: « 12 market-and-timeframe
  // combinations » counts combinations, not markets, and must not trip this.
  // NOTE: no `g` flag — a global regex carries `lastIndex` across `.test()` calls
  // and would skip every other segment inside the filter below.
  const count = new RegExp(
    `(?:(\\d[\\d\\s]*)\\s*(?:${mw})(?![\\w-])|(?:${mw})\\s*(\\d[\\d\\s]*))`,
    'i',
  );
  return segments(stripHarakat(text))
    .filter((seg) => {
      const m = count.exec(seg);
      if (!m) return false;
      const n = Number((m[1] ?? m[2] ?? '').replace(/\s/g, ''));
      if (n === ALL_MARKET_IDS.length) return false; // the live perimeter, stated as fact
      return !scope.test(seg); // any other figure needs its scope word
    })
    .map((seg) => seg.trim());
}

describe('LP-2S home — the markets line is never a bare number', () => {
  it('the hero renders the honest markets line, and no stat banner (fr)', () => {
    render(<HomeLanding />);
    const txt = document.body.textContent ?? '';
    expect(txt).toContain('80 marchés au programme');
    expect(txt).toContain('XAUUSD et EURUSD');
    // the four removed tiles must not come back
    expect(screen.queryByText('marchés suivis')).not.toBeInTheDocument();
    expect(screen.queryByText('unités de temps')).not.toBeInTheDocument();
    expect(screen.queryByText('conditions de recherche')).not.toBeInTheDocument();
    expect(screen.queryByText('structures détectées')).not.toBeInTheDocument();
  });

  it('names exactly the markets really in production, from the registry (9 locales)', () => {
    // MKT-1 registry is the single source of truth for the LIVE perimeter. Adding
    // a third market turns this red — the sentence must then be updated, not the test.
    for (const [loc, msg] of Object.entries(LOCALES)) {
      const line = ((msg as Record<string, any>).home.hero.roadmap ?? '') as string;
      for (const id of ALL_MARKET_IDS) {
        expect(line, `${loc} must name the live market ${id}`).toContain(id);
      }
      const tickers = line.match(/\b[A-Z]{6}\b/g) ?? [];
      expect(tickers.length, `${loc} names a market that is not in the registry`)
        .toBe(ALL_MARKET_IDS.length);
    }
  });

  it('never states a market count without a scope word in the same segment (9 locales)', () => {
    for (const loc of Object.keys(LOCALES) as Loc[]) {
      const line = ((LOCALES[loc] as Record<string, any>).home.hero.roadmap ?? '') as string;
      expect(line, `${loc} roadmap line is missing`).toBeTruthy();
      // the ambition figure is present…
      expect(line, `${loc} must still carry the 80 ambition`).toMatch(/80/);
      // …and never bare
      expect(bareMarketCounts(loc, line), `${loc}: bare market count`).toEqual([]);
    }
  });

  it('no string anywhere in the home namespace states a bare market count (9 locales)', () => {
    for (const loc of Object.keys(LOCALES) as Loc[]) {
      const strings = collectStrings((LOCALES[loc] as Record<string, unknown>).home);
      const offenders = strings.flatMap((s) => bareMarketCounts(loc, s));
      expect(offenders, `${loc}: bare market count in home copy`).toEqual([]);
    }
  });

  // A guard that cannot fail protects nothing. These are the exact sentences a
  // translator (human or model) produces when they "simplify" the line — the
  // scope word dropped, the ambition left wearing the present tense. Each one
  // MUST be caught, in its own language.
  it('the guard catches a translation stripped of its scope word (9 locales)', () => {
    const STRIPPED: Record<Loc, string> = {
      fr: '80 marchés — XAUUSD et EURUSD disponibles dès aujourd’hui.',
      en: '80 markets — XAUUSD and EURUSD available today.',
      de: '80 Märkte — XAUUSD und EURUSD ab heute verfügbar.',
      es: '80 mercados — XAUUSD y EURUSD disponibles desde hoy.',
      it: '80 mercati — XAUUSD ed EURUSD disponibili da oggi.',
      pt: '80 mercados — XAUUSD e EURUSD disponíveis desde hoje.',
      nl: '80 markten — XAUUSD en EURUSD vandaag al beschikbaar.',
      pl: '80 rynków — XAUUSD i EURUSD dostępne już dziś.',
      ar: '80 سوقًا — XAUUSD وEURUSD متاحان اليوم.',
    };
    for (const loc of Object.keys(STRIPPED) as Loc[]) {
      expect(
        bareMarketCounts(loc, STRIPPED[loc]),
        `${loc}: the guard let a bare "80 markets" through`,
      ).not.toEqual([]);
    }
  });
});

describe('LP-1 home — honest figures', () => {
  it('does not advertise the maquette fictions (480 combinations)', () => {
    render(<HomeLanding />);
    const txt = document.body.textContent ?? '';
    expect(txt).not.toMatch(/480/);
  });
});

describe('LP-1 home — mandatory mentions', () => {
  it('shows the illustration-data mention', () => {
    render(<HomeLanding />);
    expect(screen.getAllByText(/Données d'illustration/i).length).toBeGreaterThan(0);
  });

  it('never shows a price without its currency', () => {
    render(<HomeLanding />);
    const txt = document.body.textContent ?? '';
    expect(txt).toContain('39 $');
    expect(txt).toContain('348 $ US');
    expect(txt).toContain('29 $ US');
  });

  it('carries the required legal mentions (fr)', () => {
    render(<HomeLanding />);
    const txt = document.body.textContent ?? '';
    expect(txt).toMatch(/dollars américains/i);
    expect(txt).toMatch(/risque de perte/i);
    expect(txt).toMatch(/18 ans/i);
    expect(txt).toMatch(/ni conseil/i);
    expect(txt).toMatch(/annulable à tout moment/i);
  });

  it('carries the required legal mentions (en)', () => {
    render(<HomeLanding />, 'en');
    const txt = document.body.textContent ?? '';
    expect(txt).toMatch(/US dollars/i);
    expect(txt).toMatch(/risk of loss/i);
    expect(txt).toMatch(/18 and over/i);
  });
});

/**
 * The offline invariant, re-ported (MIA-4S). It used to cover all five demos;
 * the M.I.A tab now talks to the real agent, so the rule is narrowed to where
 * it still holds — and enforced there just as hard. The M.I.A tab has its own
 * describe block below, which asserts what it does with and without a backend.
 */
describe('LP-1 home — the non-M.I.A demos run offline', () => {
  beforeEach(() => {
    vi.spyOn(global, 'fetch').mockImplementation(() => {
      throw new Error('these demos must never hit the network');
    });
  });
  afterEach(() => vi.restoreAllMocks());

  it('the structure demo rewrites its narration and never calls the network', () => {
    render(<DemoTabs />);
    // default: all layers on → narration mentions the CHOCH
    expect(screen.getByText(/CHOCH haussier/i)).toBeInTheDocument();
    // Unticking the LAYER CHIPS is the promise the side paragraph makes
    // ("Décoche une couche : le paragraphe se réécrit") — assert it on the
    // chips themselves, not on a shortcut button.
    fireEvent.click(screen.getByRole('button', { name: 'BOS / CHOCH' }));
    expect(screen.queryByText(/CHOCH haussier/i)).not.toBeInTheDocument();
    // down to liquidity alone → only the liquidity fragment is left
    fireEvent.click(screen.getByRole('button', { name: 'Order Blocks' }));
    fireEvent.click(screen.getByRole('button', { name: 'Fair Value Gaps' }));
    expect(screen.queryByText(/Order Block haussier/i)).not.toBeInTheDocument();
    expect(screen.getByText(/liquidité achat reste intacte/i)).toBeInTheDocument();
    // and the honest empty state when nothing is left to describe
    fireEvent.click(screen.getByRole('button', { name: 'Liquidité' }));
    expect(screen.getByText(/n'invente rien pour remplir le vide/i)).toBeInTheDocument();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it('the scanner demo shows the two honest empty states', () => {
    render(<DemoTabs />);
    fireEvent.click(screen.getByRole('tab', { name: /Définir une stratégie/i }));
    // untick both default conditions → "no condition" empty state (not all markets)
    fireEvent.click(screen.getByText('La tendance structurelle est haussière'));
    fireEvent.click(screen.getByText("L'unité supérieure va dans le même sens"));
    expect(screen.getByText(/et surtout pas tous les marchés/i)).toBeInTheDocument();
    // tick a very restrictive combo that no market meets → "not an error"
    fireEvent.click(screen.getByText('Le prix est dans un Order Block'));
    fireEvent.click(screen.getByText("La zone n'a jamais été testée"));
    fireEvent.click(screen.getByText('Une poche a été prise récemment'));
    expect(screen.getByText(/Ce n'est pas une erreur/i)).toBeInTheDocument();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it('the zones demo draws each zone in the structure tab visual language', () => {
    render(<DemoTabs />);
    fireEvent.click(screen.getByRole('tab', { name: /Suivre une zone/i }));
    // zone 1 — an untested Order Block, drawn untouched
    expect(screen.getByText(/ORDER BLOCK ↑ · active · jamais testée/)).toBeInTheDocument();
    // zone 2 — a Fair Value Gap eaten at 60 %
    fireEvent.click(screen.getByRole('tab', { name: 'Fair Value Gap · comblé à 60 %' }));
    expect(screen.getByText(/FAIR VALUE GAP ↓ · pénétrée/)).toBeInTheDocument();
    // zone 3 — a spent Order Block
    fireEvent.click(screen.getByRole('tab', { name: 'Order Block · comblé' }));
    expect(screen.getByText(/ORDER BLOCK ↓ · comblée/)).toBeInTheDocument();
    // "hide from the chart" removes the DRAWING; the facts stay readable
    fireEvent.click(screen.getByRole('button', { name: 'Masquer du graphique' }));
    expect(screen.queryByText(/ORDER BLOCK ↓ · comblée/)).not.toBeInTheDocument();
    expect(screen.getByText(/Elle n'est plus active/i)).toBeInTheDocument();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it('the régime demo reveals the raw calculation on demand', () => {
    render(<DemoTabs />);
    fireEvent.click(screen.getByRole('tab', { name: /Ouvrir le calcul/i }));
    expect(screen.queryByText('Parcours moyen récent')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Ouvre le calcul' }));
    const panel = screen.getByText('Parcours moyen récent');
    expect(panel).toBeInTheDocument();
    expect(screen.getByText(/Ce que ça ne dit pas/i)).toBeInTheDocument();
  });
});

/**
 * MIA-4S — the M.I.A tab is the one demo that DOES call the network, because it
 * is the real agent. These tests pin both halves of that: what it shows when the
 * backend answers, and what it shows when there is none.
 */
describe('LP-1 home — the M.I.A tab talks to the real agent', () => {
  const sse = (payload: Record<string, unknown>) =>
    `data: ${JSON.stringify({ event: 'activity' })}\n\ndata: ${JSON.stringify({ event: 'answer', ...payload })}\n\n`;

  const mockAnswer = (payload: Record<string, unknown>) =>
    vi.spyOn(global, 'fetch').mockResolvedValue({
      ok: true,
      status: 200,
      body: undefined,
      text: async () => sse(payload),
    } as unknown as Response);

  afterEach(() => vi.restoreAllMocks());

  const openMia = () => {
    render(<DemoTabs />);
    fireEvent.click(screen.getByRole('tab', { name: /Parler à M\.I\.A/i }));
  };

  it('a starter is sent to the agent and its real answer is shown', async () => {
    mockAnswer({ content: 'Le scénario montre un Order Block jamais testé.', messages_left: 5 });
    openMia();
    fireEvent.click(screen.getByRole('button', { name: 'Montre-moi seulement les OB non testés' }));
    expect(
      await screen.findByText('Le scénario montre un Order Block jamais testé.'),
    ).toBeInTheDocument();
    // the starter is a PROMPT, not a canned reply: it was posted to the agent
    const call = (global.fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0]!;
    expect(call[0]).toBe('/api/demo/chat/stream');
    expect(JSON.parse(String((call[1] as RequestInit).body)).user_message).toBe(
      'Montre-moi seulement les OB non testés',
    );
    expect(await screen.findByText(/5 questions restantes/)).toBeInTheDocument();
  });

  it('a freely typed question reaches the agent (the starters are not a menu)', async () => {
    mockAnswer({ content: 'L’abonnement est à 39 USD par mois.', messages_left: 4 });
    openMia();
    fireEvent.change(screen.getByLabelText('Pose ta question…'), {
      target: { value: 'Combien coûte l’abonnement ?' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Envoyer' }));
    expect(await screen.findByText('L’abonnement est à 39 USD par mois.')).toBeInTheDocument();
  });

  it('a validated view action really moves the chart layers', async () => {
    mockAnswer({
      content: 'J’ai masqué les Fair Value Gaps.',
      messages_left: 4,
      view_actions: [{ action: 'set_layer_visibility', params: { layer: 'fvg', visible: false } }],
    });
    openMia();
    fireEvent.click(screen.getByRole('button', { name: 'Montre-moi seulement les OB non testés' }));
    expect(await screen.findByText('J’ai masqué les Fair Value Gaps.')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('tab', { name: /Lire une structure/i }));
    // the FVG fragment is gone from the narration, the OB one stays
    expect(screen.queryByText(/Fair Value Gap baissier comblé/i)).not.toBeInTheDocument();
    expect(screen.getByText(/Order Block haussier/i)).toBeInTheDocument();
  });

  it('a reached quota is stated plainly and closes the composer', async () => {
    vi.spyOn(global, 'fetch').mockResolvedValue({
      ok: false,
      status: 429,
      json: async () => ({ detail: { reason: 'session_limit', message: 'On s’arrête ici.' } }),
    } as unknown as Response);
    openMia();
    // NB: the i18n copy uses a STRAIGHT apostrophe here — matching a curly one fails.
    fireEvent.click(screen.getByRole('button', { name: "C'est quoi un Order Block ?" }));
    expect(await screen.findByText('On s’arrête ici.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Envoyer' })).toBeDisabled();
  });

  it('with no backend it degrades to the recorded exchanges, and says so', async () => {
    vi.spyOn(global, 'fetch').mockRejectedValue(new Error('offline'));
    openMia();
    fireEvent.click(screen.getByRole('button', { name: 'Montre-moi seulement les OB non testés' }));
    // the recorded answer is shown — and labelled as recorded, not passed off
    expect(await screen.findByText(/Seuls les Order Blocks/i)).toBeInTheDocument();
    expect(
      screen.getByText(/La démonstration en direct n'est pas disponible ici/i),
    ).toBeInTheDocument();
  });
});
