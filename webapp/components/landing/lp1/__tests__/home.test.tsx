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
import { LANDING_STATS, STRUCTURE_TYPES } from '@/lib/landing/stats';

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

describe('LP-1 home — honest figures', () => {
  it('the stats banner renders the real numbers from the single source', () => {
    render(<HomeLanding />);
    // LP-2 banner: markets · timeframes · conditions · structures
    expect(screen.getByText('marchés suivis')).toBeInTheDocument();
    expect(screen.getByText('unités de temps')).toBeInTheDocument();
    expect(screen.getByText('conditions de recherche')).toBeInTheDocument();
    expect(screen.getByText('structures détectées')).toBeInTheDocument();
    // the distinctive figures come straight from LANDING_STATS
    expect(String(LANDING_STATS.conditions)).toBe('22');
    expect(String(LANDING_STATS.structures)).toBe('7');
    expect(screen.getByText('22')).toBeInTheDocument();
  });

  it('the 4th banner figure is sourced from STRUCTURE_TYPES, never a literal', () => {
    // §C4: the "structures détectées" tile must equal the single source length.
    expect(LANDING_STATS.structures).toBe(STRUCTURE_TYPES.length);
    expect(STRUCTURE_TYPES.length).toBe(7);
  });

  it('does not advertise the maquette fictions (80 markets / 480 combinations)', () => {
    render(<HomeLanding />);
    const txt = document.body.textContent ?? '';
    expect(txt).not.toMatch(/480/);
    expect(txt).not.toMatch(/80\s*\+?\s*march/i);
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
    // free tier is explicit dollars too
    expect(txt).toContain('0 $');
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

describe('LP-1 home — demos run offline', () => {
  beforeEach(() => {
    vi.spyOn(global, 'fetch').mockImplementation(() => {
      throw new Error('the landing must never hit the network');
    });
  });
  afterEach(() => vi.restoreAllMocks());

  it('the structure demo rewrites its narration and never calls the network', () => {
    render(<DemoTabs />);
    // default: all layers on → narration mentions the CHOCH
    expect(screen.getByText(/CHOCH haussier/i)).toBeInTheDocument();
    // "keep only the liquidity" → narration drops the CHOCH, keeps liquidity
    fireEvent.click(screen.getByText('Ne garder que la liquidité'));
    expect(screen.queryByText(/CHOCH haussier/i)).not.toBeInTheDocument();
    expect(screen.getByText(/liquidité achat reste intacte/i)).toBeInTheDocument();
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

  it('a MIA question changes the structure demo layers (grounded action)', () => {
    render(<DemoTabs />);
    fireEvent.click(screen.getByRole('tab', { name: /Parler à M\.I\.A/i }));
    fireEvent.click(screen.getByText('Montre-moi seulement les OB non testés'));
    // the answer confirms only OBs remain, and the action note appears
    expect(screen.getByText(/Seuls les Order Blocks/i)).toBeInTheDocument();
    expect(screen.getByText(/Les couches du graphique ont changé/i)).toBeInTheDocument();
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
