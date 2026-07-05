/**
 * Non-régression du nettoyage des claims légaux/marketing (2026-07-04).
 *
 * 1. Les claims faux ou invérifiables supprimés ne doivent JAMAIS réapparaître
 *    dans les sources rendues du frontend (composants, pages, chaînes i18n).
 * 2. Aucun lien mort dans le footer : chaque route interne pointée doit
 *    exister dans `app/[locale]/`, chaque ancre doit exister dans une section.
 *
 * Contexte : la marque EST l'honnêteté anti-scam — un claim faux est une
 * faille critique. En cas de doute sur un claim → le supprimer, pas le
 * reformuler (règle de mission, cf. docs/audits/FOOTER_CLAIMS_CLEANUP_2026_07_04.md).
 */
import { readdirSync, readFileSync, existsSync, statSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

import { LEGAL_LINKS, PRODUCT_LINKS } from '@/components/Footer';

const WEBAPP_ROOT = path.resolve(__dirname, '..');

// Répertoires source rendus côté client (les tests eux-mêmes sont exclus).
const SCANNED_DIRS = ['components', 'app', 'messages', 'lib'] as const;
const SCANNED_EXTENSIONS = new Set(['.ts', '.tsx', '.json', '.md']);

// Construits par concaténation pour que ce fichier ne se matche pas lui-même
// si jamais il entrait dans le périmètre du scan.
const FORBIDDEN_CLAIMS = [
  '2024' + '/2811', // référence réglementaire UE introuvable
  'CM' + '2C', // médiateur jamais adhéré
  '50' + ' places', // compteur de places jamais décidé
  'hors' + ' Québec', // périmètre géographique jamais décidé côté produit
] as const;

function walk(dir: string, acc: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    if (entry === 'node_modules' || entry === '.next') continue;
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) {
      walk(full, acc);
    } else if (SCANNED_EXTENSIONS.has(path.extname(entry))) {
      acc.push(full);
    }
  }
  return acc;
}

describe('claims cleanup — chaînes interdites', () => {
  const files = SCANNED_DIRS.flatMap((d) => {
    const dir = path.join(WEBAPP_ROOT, d);
    return existsSync(dir) ? walk(dir) : [];
  });

  it('scanne un périmètre non vide', () => {
    expect(files.length).toBeGreaterThan(50);
  });

  it.each(FORBIDDEN_CLAIMS)('aucune occurrence de « %s »', (claim) => {
    const offenders = files.filter((f) =>
      readFileSync(f, 'utf-8').includes(claim),
    );
    expect(
      offenders.map((f) => path.relative(WEBAPP_ROOT, f)),
    ).toEqual([]);
  });
});

describe('claims cleanup — liens du footer', () => {
  const LOCALE_APP_DIR = path.join(WEBAPP_ROOT, 'app', '[locale]');

  /** id="…" déclarés dans les sections de la landing et les pages. */
  function collectAnchorIds(): Set<string> {
    const ids = new Set<string>();
    const sources = [
      path.join(WEBAPP_ROOT, 'components'),
      path.join(WEBAPP_ROOT, 'app'),
    ].flatMap((d) => (existsSync(d) ? walk(d) : []));
    for (const file of sources) {
      for (const m of readFileSync(file, 'utf-8').matchAll(/id="([\w-]+)"/g)) {
        if (m[1]) ids.add(m[1]);
      }
    }
    return ids;
  }

  const anchorIds = collectAnchorIds();
  const allLinks = [...LEGAL_LINKS, ...PRODUCT_LINKS];

  it.each(allLinks.map((l) => [l.href, l.label] as const))(
    'la cible de « %s » (%s) existe',
    (href) => {
      if (href.startsWith('mailto:')) return; // pas une route interne

      expect(href.startsWith('/')).toBe(true); // ancre nue = cassée hors landing

      const [routePart = '', anchor] = href.split('#');
      const route = routePart.replace(/\/$/, ''); // '/' → ''
      const pageFile =
        route === ''
          ? path.join(LOCALE_APP_DIR, 'page.tsx')
          : path.join(LOCALE_APP_DIR, ...route.slice(1).split('/'), 'page.tsx');
      expect(existsSync(pageFile), `page manquante pour ${href}`).toBe(true);

      if (anchor) {
        expect(anchorIds.has(anchor), `ancre #${anchor} introuvable`).toBe(
          true,
        );
      }
    },
  );
});
