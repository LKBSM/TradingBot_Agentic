# Purge lexicale — libellés de zones hors ligne éditoriale (2026-07-17)

**Branche** : `feat/zones-label-line-purge` (rebasée sur `origin/main` = `851e56d`, PR #50).
**Nature** : **présentation uniquement**. Aucune modification du moteur de détection,
des seuils, ni du calcul de cycle de vie des zones.

## Contexte

La ligne éditoriale du produit est **descriptive, au présent, jamais prédictive ni
un score de conviction**. Trois libellés d'interface la franchissaient :

1. **« encore efficace » / « plus efficace »** (page Zones) — « efficace » est
   prédictif : il laisse entendre que la zone *produira* un effet.
2. **« importance {faible/moyenne/élevée} »** (Order Blocks, page Zones + section
   Structure) — un score de conviction déguisé. La règle produit = aucun score /
   classement visible.
3. **« MMIA Markets »** (nom accessible du logo) — double M rapporté.

## Diagnostic (ce que représentait chaque valeur)

### (a) « encore efficace » / « plus efficace »
Source unique : `webapp/components/zones/ZoneLifecycleCard.tsx`, fonction
`effectiveness()`. Valeur sous-jacente = **la zone n'est PAS consommée** :

- OB : `status !== 'invalidated'` → non consommé (le prix n'a **pas clôturé au travers**).
- FVG : `status !== 'filled'` → non entièrement comblé.

C'est **indépendant du test** (un OB mitigé/testé reste non invalidé). Ce n'est donc
pas « pas encore testée » mais bien « pas invalidée / pas comblée ».

### (b) « importance faible/moyenne/élevée »
Champ moteur `ob.importance` (`low/medium/high`), fixé dans
`src/intelligence/market_reading_mappers.py` par seuillage de `OB_STRENGTH_NORM`,
lui-même calculé dans `src/environment/strategy_features.py` :

```
base_strength   = taille_bougie_OB / ATR        (hauteur de zone / volatilité)
OB_STRENGTH_NORM = base_strength + bonus_FVG     (+ bonus si FVG adjacent)
seuils : ≥ 0,75 → high · ≥ 0,4 → medium · sinon low
```

→ ce n'est pas un « déplacement fort » mais **la taille de la zone rapportée à
l'ATR**, reconditionnée en échelle de conviction à 3 niveaux. Classement déguisé.

### (c) « MMIA Markets »
**Déjà corrigé** sur `origin/main`. Les deux seuls logos à glyphe « M »
(`Nav.tsx`, `AppHeader.tsx`) ont le glyphe en `aria-hidden` **et** un `aria-label`
explicite (« MIA Markets — … »). Nom accessible = « MIA Markets ». `git blame` :
`aria-hidden` depuis 2026-05-26, `aria-label` depuis 2026-05-27. Aucun composant
ne produit « MMIA » aujourd'hui — le double-M observé venait d'un build déployé
antérieur. **Aucune modification apportée** (rien à corriger). Signalé pour trace.

## Décision retenue (point b) : **Option 1 — retrait total du libellé**

L'utilisateur a choisi de **supprimer** le libellé d'importance (plutôt que de le
remplacer par un fait structurel nommé). La largeur réelle de la zone reste lisible
via la fourchette de prix déjà affichée. Choix le plus conforme à « aucun score /
classement ».

## Libellés — avant / après

| Emplacement | Avant | Après |
|---|---|---|
| Zone OB vivante | encore efficace | **non invalidée** |
| Zone OB consommée | plus efficace | **invalidée** |
| Zone FVG vivante | encore efficace | **non comblée** |
| Zone FVG consommée | plus efficace | **comblée** |
| Infobulle vivante | « Zone toujours valable… » | fait présent, par type (voir code) |
| OB (carte Zones) | importance élevée | *(libellé retiré)* |
| OB (section Structure) | `{band} · importance élevée · actif` | `{band} · actif` |
| Méthodologie | « Importance d'un Order Block » (faible/moyenne/élevée) | « Order Block » (fourchette + état, sans note) |

## Fichiers touchés (présentation uniquement)

- `webapp/components/zones/ZoneLifecycleCard.tsx` — `effectiveness()` reformulée
  (libellés + infobulles par type), retrait du libellé importance + de l'import
  `formatObImportance`.
- `webapp/components/market-reading/sections/StructureSection.tsx` — retrait du
  segment `· importance …` du libellé OB + de l'import `formatObImportance`.
  (Le tri interne `OB_IMPORTANCE_RANK` — ordre d'affichage non visible — est **conservé**.)
- `webapp/lib/methodology/content.ts` — entrée `order-block-importance` remplacée
  par `order-block` (description factuelle sans échelle de qualité).
- `webapp/components/market-reading/__tests__/zone-click-to-chart.test.tsx` —
  sélecteurs de test alignés (`/importance moyenne · actif/` → `/· actif/`).
- `webapp/tests/claims-cleanup.test.ts` — garde anti-régression étendue :
  interdit `efficace` et `importance élevée` dans `components/app/messages/lib`.

## Détection : STRICTEMENT inchangée

Aucun fichier moteur touché. `market_reading_mappers.py`, `strategy_features.py`,
le calcul de `OB_STRENGTH_NORM`, l'attribution d'`importance` et les statuts de
cycle de vie sont **intacts**. Le champ `ob.importance` reste émis par le moteur et
utilisé en interne (ordre d'affichage, clé de déduplication) — seul son **affichage**
est retiré.

## Vérifications

- `npm run typecheck` : ✅ 0 erreur.
- `npm test` (Vitest) : ✅ voir suite (les 2 tests jadis rouges — palette 14 vs 10
  et Nav matcher — sont corrigés par PR #50, désormais dans la base).
- `npm run build` : ✅ (Next build).
- Garde anti-régression : `efficace` et `importance élevée` interdits — 0 occurrence.
