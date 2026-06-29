# Enrichissement de la section « Régime de marché » — Rapport

**Date :** 2026-06-29
**Branche :** `feat/market-regime-enrich` (worktree dédié `C:\MyPythonProjects\TBOT_regime_enrich`)
**Périmètre :** webapp (affichage / lecture seule). Moteur & détection **inchangés**.

---

## 0. Ligne inviolable respectée

Affichage **lecture seule**, ancré sur les **faits du moteur**, **au présent**, **descriptif**.
Aucune prédiction, aucune probabilité, aucun score de « force de signal », aucun « niveau à
surveiller », aucune conséquence anticipée. Chaque info affichée correspond à une donnée
**réelle** du moteur ; si elle est absente pour un combo → **« non disponible »**, jamais
d'invention ni d'estimation.

Disclaimer de section ajouté :
> *« Ces faits décrivent l'état observé du marché en ce moment. Ils ne constituent pas une
> instruction adressée au trader. »*

---

## 1. Les 5 informations ajoutées

Toutes sont déjà produites par le moteur ; elles sont **dérivées à l'affichage** à partir de
champs réels du contrat `MarketReading` v2.0.0 (aucune nouvelle donnée inventée).

| # | Info | Source moteur | Rendu (exemple réel capturé) |
|---|------|---------------|------------------------------|
| (a) | **Phase de marché** | `regime.market_phase` (même source que la palette du scanner) | Badge `Phase : Tendance` / `Phase : Range` |
| (b) | **Maturité de tendance** | `structure.choch.{direction, broken_at}` + `header.candle_close_ts/timeframe` | *« Structure orientée haussière depuis le CHOCH du 26/05 à 09:30 (≈ 9 bougies M15). »* |
| (c) | **Dernier événement structurel** | dernier de `structure.bos` / `structure.choch` par `broken_at` | *« BOS haussier confirmé (M15) »* |
| (d) | **Densité de zones actives** | `structure.order_blocks` / `fair_value_gaps` filtrés `status==='active'` | *« 1 OB · 1 FVG actifs »* |
| (e) | **Désaccord multi-timeframe** | `regime.mtf_confluence` (M15/H1/H4) | Callout warn *« Désaccord multi-timeframe — Les TF divergent : H4 baissier, H1 haussier et M15 haussier. »* |

### Détails de dérivation (descriptive, pas de calcul prédictif)

- **(b) Maturité — « Bougies + date ».** Le nombre de bougies = `floor((candle_close_ts −
  broken_at) / intervalle_TF)`, gardé `>= 0` et fini ; sinon le compte est omis et seule la date
  est affichée. La date est rendue depuis l'horodatage moteur tel quel (`JJ/MM à HH:MM`, wall-clock
  du moteur, aucun calcul de fuseau) → déterministe. Absence de `choch` → **« non disponible »**.

- **(c) Dernier événement.** On choisit le plus récent (`broken_at`) entre BOS et CHOCH ; le
  statut de validation est rendu en clair (`confirmé` / `en attente de confirmation` / `invalidé`).
  Aucun des deux → **« non disponible »**.

- **(d) Densité.** Toujours disponible : `0 OB · 0 FVG actifs` est un fait réel, pas un trou.

- **(e) Désaccord — « Callout warn in-section ».** Le drapeau `disagreement` ne se déclenche **que
  sur une vraie contradiction** (`up` ET `down` présents simultanément). Une direction face à du
  plat (`partial`) n'est **pas** un désaccord → ligne descriptive normale, sans callout. Le callout
  warn est au **même niveau de visibilité** que le badge d'alignement (icône `AlertTriangle`,
  bordure/fond `sentinel-warn`, titre « Désaccord multi-timeframe »).

---

## 2. Fichiers touchés (affichage uniquement)

**Créés :**
- `webapp/lib/market-reading/regime-facts.ts` — helpers purs au présent (maturité, dernier
  événement, densité, formatage horodatage).
- `webapp/lib/market-reading/__tests__/regime-facts.test.ts`

**Modifiés :**
- `webapp/lib/market-reading/mtf-trend.ts` — ajout `MtfRelation` + `classifyMtfAlignment()`
  (drapeau `disagreement` sur vraie contradiction up/down). `describeMtfAlignment` devient un
  wrapper du `.text` (phrasé existant préservé).
- `webapp/lib/market-reading/formatters.ts` — `formatMarketPhaseShort()` (label compact distinct
  du « Phase de tendance » du hero).
- `webapp/components/market-reading/sections/RegimeSection.tsx` — rendu des 5 infos + callout warn
  + disclaimer ; valeurs nulles → `non disponible` (italique muté).
- `webapp/components/market-reading/MarketReadingSections.tsx` — passe `structure` & `header` à
  `RegimeSection`.
- Tests : `mtf-trend.test.ts`, `RegimeSection.test.tsx`.

**Aucun fichier moteur / détection / schéma modifié.**

---

## 3. Vérifications

| Vérification | Résultat |
|--------------|----------|
| `tsc --noEmit` | **exit 0** |
| `vitest run` (market-reading) | **122/122 ✅** |
| Chaque info = donnée moteur réelle | ✅ (aucune valeur inventée) |
| Donnée absente → « non disponible » | ✅ (structure vide → 2× « non disponible ») |
| Cas désaccord affiché distinctement | ✅ (callout warn vs ligne simple) |
| Aucune sortie prédictive/probabiliste | ✅ (test de vocabulaire interdit) |
| Vérif visuelle | ✅ captures ci-dessous |

**Captures :**
- `market_regime_enrich_disagreement.png` — XAU/USD M15, **vraie contradiction** (H4 baissier vs
  H1/M15 haussiers) → callout warn « Désaccord multi-timeframe ».
- `market_regime_enrich_partial.png` — EUR/USD H1, divergence **partielle** (plat vs baissier) →
  ligne descriptive simple, **sans** callout (rendu distinct correct).

> Note : la capture a été produite via une bascule temporaire `READING_DATA_SOURCE='mock'` +
> un fixture forcé pour exposer le cas contradiction. **Ces deux modifications temporaires ont été
> intégralement annulées** (`git checkout`) ; le spec Playwright de capture a été supprimé. Le diff
> committé ne contient que l'implémentation d'affichage.
