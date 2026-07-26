# AUDIT — UI-2c · Structure de marché · Liquidité externe · tri par carte · aide pédagogique

**Branche** `feat/ui-2c-structure-liquidite-aide` (worktree dédié, depuis `origin/main` 86a0d2d). **Zéro diff backend.**
**Cible** `docs/design/reference-desktop.html` (v5, mergée sur main PR #80).
**Statut** `tsc` 0 · **538 tests** verts (+13 UI-2c ; flake connu `claims-cleanup` sous parallélisme, passe en isolation) · build Next vert · e2e structurel vert · vérif visuelle mock-data 1280 conforme.
**Ligne** merge sur `main` seulement après confirmation live.

---

## 1. Définitions VÉRIFIÉES du Régime de marché (depuis le code moteur)

Source : `src/intelligence/market_reading_mappers.py` (`_derive_volatility`, `candles_to_regime`) + `webapp/lib/market-reading/regime-facts.ts`. **Aucun backend modifié** — ces définitions déterminent quelle sous-ligne de source afficher.

| Mesure | Ce que le moteur compte | Point de départ / TF | Sous-ligne affichée |
|---|---|---|---|
| **Tendance** | `_derive_trend(closes)` — séquence des clôtures | **La TF affichée** (combo actif) | « mesurée sur {TF} » |
| **Volatilité** | `_derive_volatility` : TR=(haut−bas) ; ratio = **moyenne des 7 dernières bougies / moyenne des précédentes** de la fenêtre. <0,7 basse · >1,3 élevée | 7 récentes vs le reste de la fenêtre | « 7 récentes vs précédentes » — **PAS « 20 bougies »** (la v5 était illustrative/fausse) |
| **Maturité** | Nb de bougies clôturées **depuis le CHOCH le plus récent** de `choch_events` (historique) ; **CHOCH only, jamais BOS** | Horodatage de ce CHOCH, sur la TF affichée | « {n} bougies » + « depuis le CHOCH de {HH:MM} » |
| **Alignement** | Direction de structure de la TF affichée + TF supérieures (`mtf_candles_above`), chacune via le trend de ses clôtures | Clés MTF (H4/H1/M15…) | valeur « {n}/{m} TF {flèche} » + sous-ligne « H4 ↓ · H1 ↓ · M15 ↑ » |
| **Densité** | `countActiveZones` : zones **actives uniquement** (`status==='active'`), OB et FVG séparés | Sur la TF affichée | « actives sur {TF} » |
| **Dernier événement** | Le plus récent de **`bos_events ∪ choch_events`** (historique émis par le mapper) | — | « {HH:MM} » |

**« not available » du Dernier événement = bug FRONT corrigé.** L'ancien code lisait `structure.bos`/`choch` (point-in-time, seulement si sur la dernière bougie) ; il lit désormais l'historique `*_events`, qui existe.

**Incohérence « Bullish + Désaccord + H4↓H1↓M15↑ » = PAS un bug de calcul.** `regime.trend` est la tendance de **la TF affichée** ; le panneau Alignement montre chaque TF. Le correctif est la sous-ligne « mesurée sur {TF} » qui lève l'ambiguïté (diagnostiqué au STOP, corrigé ici).

**Maturité — réponse à la question fondatrice :** le compteur part **du CHOCH confirmé le plus récent**, compté en bougies clôturées sur la TF affichée. L'origine est désormais **nommée sous la valeur** (« depuis le CHOCH de HH:MM ») — fini le « ≈ 154 M15 » sans origine.

---

## 2. Éléments SUPPRIMÉS faute de donnée réelle (règle « pas de donnée, pas d'élément »)

| Élément v5 | Décision | Raison (moteur) |
|---|---|---|
| **Badge confluence « 2 TF / 3 TF »** par zone | **SUPPRIMÉ** | Aucun champ par-zone. `mtf_confluence` est **global** (régime), pas par zone — impossible de dire qu'une zone existe sur une autre TF. |
| **Chip TF (H1/M15/H4) par ligne** (zones + poches) | **SUPPRIMÉ** | `OrderBlock`/`FairValueGap`/`LiquidityPool` n'ont **pas de `timeframe`** — toutes les zones/poches d'une lecture sont sur la TF affichée. La TF (constante) est déjà dans le contexte de la carte. |
| **« Testée ×N »** dans les notes | Dégradé en **« Testée »** | `tested` est un **booléen** ; aucun compte ni historique de tests. |
| **Volatilité « vs 20 dernières bougies »** | Corrigé en **« 7 récentes vs précédentes »** | Le moteur calcule 7-vs-reste (cf. §1) — le « 20 » est faux. Idem dans le **texte d'aide `vol`** (seule retouche verbatim, signalée). |

Aucune valeur de remplissage, aucun élément deviné.

---

## 3. Ce qui est livré (frontend only)

**Carte « Structure de marché »** (`components/app/StructureCard.tsx`) : en-tête (compteur réel + `?` aide + bouton tri, boutons accessibles `aria-expanded`), bandeau tri/filtre **repliable** (Type · État {Actives/Testées/Mitigées} · Trier par {Proche du prix/Récente/Large} — que des faits, aucun tri par importance/score), 2 lignes CHOCH/BOS, **liste COMPLÈTE scrollable ~210px** (le « 4 » était `slice(0,2)+slice(0,2)` côté FRONT — retiré). Chaque ligne : tag type+dir · bornes mono · badge état // distance (3 formulations, recalculée au tick **sans re-tri**) · fait honnête (via `lib/zones/lifecycle`) · « En savoir plus ». Vide → message honnête.

**Deux gestes** : clic ligne → surbrillance via le **verrou d'id existant** (`focus_zone`+`highlight_zone`, un id inventé est rejeté) ; « En savoir plus » (stopPropagation) → `/zones?zone=<id>` (nouveau param front dans `ZonesWorkspace` : scroll+highlight de la fiche ; id inconnu → « Cette zone n'est plus détectée dans la lecture courante. », jamais reconstruite).

**Carte « Liquidité externe »** (`LiquidityCard.tsx`) : en-tête (compteur + `?` + tri), filtres Côté/État, liste scrollable (repère d'état · côté · niveau mono · origine EQH/EQL/Sommet/Creux depuis `kind` · badge · distance · fait). Clic → surbrillance du **niveau** (le primitive canvas `zoneOverlayPrimitive` réagit désormais à `highlightId` sur les segments de liquidité — affichage seul).

**Carte « Régime »** (`RegimeCard.tsx`) : chaque mesure porte sa **sous-ligne de source réelle** (§1) + un `?` ; un `?` global ; un seul encart ouvert à la fois **dans toute la page** (état `openHelp` hissé dans `DesktopReading`).

**Aide pédagogique** (`HelpContent.tsx` + `reading.help.*`) : 8 textes **statiques i18n** (fr **verbatim** v5 + en fidèle), **jamais générés à l'exécution**. Chaque texte par mesure contient son bloc « ce que ça ne dit pas » (formulations variées : « ne dit pas », « n'affirmera jamais », « ne veut pas dire »…). CSS : `components/app/ui2c.css` (port verbatim des classes v5, scopé `.app-shell`).

**i18n** : fr + en complets ; les 7 autres locales reçoivent le **fallback anglais** pour les nouvelles chaînes (aucune clé brute affichée) — traduction native à planifier (mission scopée fr+en).

---

## 4. Tests (`components/app/__tests__/ui2c.test.tsx`, 13)
- honnêteté copy : aucune chaîne UI Structure/Liquidité/Régime ne contient un mot interdit ni un bouton « Trader » ;
- chaque texte d'aide par mesure contient un bloc de déni (marqueur de liste blanche) ;
- **verrou d'id** : un id inventé est rejeté par `coerceViewActions` ;
- filtre sans résultat → message honnête, **0 ligne** ;
- **tick de prix** → distances mises à jour **sans réordonner** la liste ;
- Régime : chaque mesure sourcée a une sous-ligne non vide.
- e2e `tests/e2e/ui2c-cards.spec.ts` : 1280 sans scroll horizontal + aucune clé i18n brute (les interactions carte, qui exigent une lecture live, sont couvertes par les 13 tests unitaires + une passe e2e mock manuelle : tri repliable, aide un-à-la-fois, clic→surbrillance, « En savoir plus » → `/zones?zone=<id>` vérifié).

## 5. Écarts restants / à surveiller
- **Troncature moteur** : le front affiche maintenant TOUTES les zones renvoyées par l'API (le « 4 » était front). Si le moteur SMC plafonne bas en interne, c'est **backend** (mission séparée) ; le front reste honnête.
- **7 locales en fallback anglais** pour les chaînes UI-2c (fr+en réels) — relecture/traduction native à planifier.
- Cartes conservées à leur taille/ordre/position ; bandeaux et aides **repliés par défaut** (tableau de bord aussi dense qu'avant au chargement).
