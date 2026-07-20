# Chart — attribution TradingView + repositionnement du disclaimer

Date : 2026-07-19
Branche : `feat/chart-attribution-disclaimer` (worktree dédié, base `origin/main` @ 16b8e1e)
Périmètre : zone du graphique + attribution + i18n. **Moteur / détection / IA inchangés.**

---

## 1. Diagnostic — bibliothèque de charting

**Détectée : Lightweight Charts™ v5.2.0 (open-source, Apache-2.0).**
PAS l'« Advanced Charts / charting_library / Trading Platform ».

Preuves :
- `webapp/package.json` → `"lightweight-charts": "5.2.0"`
- `webapp/components/app/ReadingChart.tsx` → `import { … createChart … } from 'lightweight-charts'`
- Aucune trace de `charting_library` / `Advanced Charts` dans le dépôt.
- Un fichier `webapp/NOTICE` documentait déjà l'usage (Apache-2.0).

Le même composant `ReadingChart` alimente le `/app` **et** la landing
(`LandingReadingChart.tsx` le charge en dynamique) → un seul point de réglage
couvre les deux surfaces.

## 2. Logo TradingView — cas LIGHTWEIGHT (décision fondateur : masquer + attribuer)

Justification légale : sous Apache-2.0, Lightweight Charts autorise à masquer le
logo on-chart (`layout.attributionLogo = false`) **à condition** de conserver
l'attribution TradingView et un **lien vers `https://www.tradingview.com/`**
accessible aux utilisateurs.

État initial : logo AFFICHÉ (`attributionLogo: true`) ; l'attribution existait sur
`/methodology#attributions` mais **le seul lien pointait vers apache.org**, pas
vers tradingview.com → non conforme si l'on masque le logo.

Action appliquée :
- `ReadingChart.tsx` → `attributionLogo: false` (+ commentaire renvoyant au NOTICE
  et à /methodology).
- `webapp/NOTICE` → mis à jour (logo masqué + attribution conservée + lien publié).
- `/methodology` (section Attributions) → ajout d'un lien `<tv>` vers
  `https://www.tradingview.com/` autour de « TradingView, Inc. » dans la chaîne
  `attributions.license`, rendu via `t.rich(..., { tv })`.
- i18n : chaîne `methodology.attributions.license` mise à jour dans **les 9 locales**.

> Note : ce chart est une **bibliothèque d'affichage** uniquement ; MIA Markets
> n'utilise aucune API/flux de données de marché TradingView (déjà stipulé au NOTICE
> et dans `attributions.note`). Le point « paywall » de l'Advanced Charts ne
> s'applique donc pas ici (lib open-source Apache-2.0, pas la lib commerciale).

## 3. Disclaimer — Option 1 (badge en tête du chart + mention légale sous le chart)

Séparation nette des deux natures :
- **« Accès anticipé »** = statut **temporaire** → badge discret (`EarlyAccessBadge`,
  `ShieldCheck`), placé en tête du chart (ligne dédiée, aucun chevauchement avec le
  badge live / les contrôles).
- **Mention légale persistante** = « Lecture algorithmique éducative — ne constitue
  ni un signal de trading, ni un conseil en investissement. » → placée
  **immédiatement sous le chart**, sobre, pleine largeur, taille 11px (lisible),
  jamais masquée.

Implémentation :
- `DisclaimerStub` : nouvelle prop `variant: 'hero' | 'chart'`. `'chart'` rend la clé
  `legal.disclaimer.chart` (mention légale seule, sans la phrase « accès anticipé »
  désormais portée par le badge). `'hero'` (défaut) inchangé pour les surfaces
  texte-seul (échantillons landing).
- Nouveau composant exporté `EarlyAccessBadge` (namespace `legal.earlyAccessBadge`).
- `MarketReadingCard` : quand un `chartSlot` est présent (/app), le chart est encadré
  par le badge (en tête) et la mention légale (dessous) ; la rangée basse ne porte
  plus que le CTA. Sans chart (landing texte-seul), la ligne hero complète reste sur
  la rangée basse (comportement préservé, disclaimer toujours visible).
- i18n : clé `legal.disclaimer.chart` ajoutée dans **les 9 locales** (parité stricte).

## 4. Discipline & validation

- Diff limité : `NOTICE`, `/methodology/page.tsx`, `ReadingChart.tsx` (zone chart),
  `MarketReadingCard.tsx` (zone chart), `DisclaimerStub.tsx`, 9 `messages/*.json`.
- **Aucun** fichier moteur / détection / IA / backend touché.
- `tsc --noEmit` : 0 erreur.
- `vitest run` : 512/512 (55 fichiers).
- `next build` : vert.
- Parité i18n 9 locales vérifiée (JSON valides, `disclaimer.chart` + `<tv>` présents partout).

## 5. Reste / à confirmer

- Validation **live** de l'emplacement du disclaimer et de la disparition du logo
  avant merge sur `main`.
- Wording toujours `LEGAL-PENDING` (marqueur conservé) — à figer par le terminal légal.
