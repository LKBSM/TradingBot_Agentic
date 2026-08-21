# AUDIT — Lecture narrée composée par gabarit déterministe (retrait de l'IA)

**Branche :** `feat/narrated-reading-template-engine` (depuis `origin/main` @ acc72d8)
**Date :** 2026-08-20
**Surface cible :** bloc « Lecture narrée · Ancrée au moteur » de `/app` (colonne centrale)

---

## 1. Objectif

Composer la lecture narrée **uniquement** par un gabarit déterministe côté produit,
à partir des champs structurés réels du moteur — **plus aucun appel à un modèle
IA (Claude Haiku)** pour ce bloc. Même règle que celle déjà appliquée aux
actualités : « résumé composé par gabarit à partir de champs structurés, jamais
généré par un modèle ». Le chatbot M.I.A **reste** sur IA (non touché).

Motifs : économie de tokens/coût client **et** renforcement de l'ancrage (l'IA
n'est jamais juge de la structure). Le libellé « Ancrée au moteur » devient
**littéralement vrai** (100 % moteur).

## 2. Vérification de dépendance (phase 0bis) — RÉSOLUE

Deux constats décisifs, établis en lecture seule :

1. **Le gabarit déterministe existait déjà.** `narrated_reading.render_template()`
   composait déjà la narration à partir des seuls faits moteur. Le Haiku était si
   contraint (mêmes faits, mêmes validateurs *forbidden-token* + *level-anchoring*,
   repli sur le template) que sa sortie **recopiait** le gabarit. L'« exemple IA »
   cité dans la mission est **mot pour mot** ce que produit `render_template`.
   → qualité **identique par construction**, pas seulement « équivalente ».

2. **La relation « s'oppose à la tendance » était déjà un calcul moteur
   déterministe**, PAS une inférence IA. Elle vient de `_contrary_reason()`
   (règle pure : zone **active** de direction opposée au `regime.trend`, ou
   pullback multi-TF). Aucune heuristique floue à ré-inventer — seule la
   **formulation** causale a été neutralisée.

**Site de l'appel IA retiré :** `MarketReadingAssembler._resolve_description()` →
`HaikuDescriptionEngine.generate()`, câblé dans `bootstrap.py`. Le chatbot M.I.A
(`bootstrap.build_chatbot`) et le traducteur scanner
(`bootstrap.build_scanner_translator`) construisent **chacun leur propre** client
Anthropic → non affectés.

## 3. Le gabarit (segments → champ moteur)

| # | Segment | Champ(s) moteur | Présence |
|---|---------|-----------------|----------|
| 1 | `Tendance {x}, volatilité {y}, phase {z}.` | `regime.trend` / `volatility_observed` / `market_phase` | **toujours** (socle) |
| 2 | Alignement multi-TF (1 phrase) | `_mtf_relation(trend, mtf_confluence)` | si `mtf_confluence` |
| 3 | `Le prix évolue près d'…` / `À proximité : … ; …` | `_collect_zones()` (OB/FVG proches, actives d'abord) | si zones |
| 4 | `Structure : CHOCH …, BOS ….` | `_collect_breaks()` (dernier CHOCH + dernier BOS) | si cassures |
| 5 | `Un retest de … est en cours (…).` | `structure.retest_in_progress` | si retest |
| 6 | `À noter : … est présent malgré la tendance …` | `_contrary_reason()` | si co-présence |

**Reformulation de la co-présence (segment 6) — le seul changement de fond :**

- **Forme zone** : `« un {OB/FVG} {dir} actif s'oppose à la tendance {x} »`
  → `« un {OB/FVG} {dir} actif est présent malgré la tendance {x} »`
  (constat de co-présence, jamais de causalité).
- **Forme pullback** : la branche « à contre-courant » a été **supprimée** de
  `_contrary_reason` : quand `mtf_relation == "pullback"`, le segment 2 le dit
  déjà neutralement (« se replie face aux timeframes supérieurs »). On évite ainsi
  la duplication **et** la réintroduction d'un « contre ».

Aucune durée en bougies ; le jargon SMC (OB, FVG, CHOCH, BOS) est conservé tel
quel (public cible), sans explication inline.

## 4. Avant / après (échantillons réels)

**XAUUSD — scénario de l'exemple mission** (tendance haussière, OB baissier actif
proche) :

- *Avant (IA / template legacy)* : « … **À noter : un Order Block baissier actif
  s'oppose à la tendance haussière.** »
- *Après (gabarit)* :
  > Tendance haussière, volatilité normale, phase de tendance. À proximité : une
  > FVG haussier actif (non testé) sous le prix (4 514,95–4 515,66) ; un Order
  > Block baissier actif (non testé) au-dessus du prix (4 517,12–4 519,37).
  > Structure : CHOCH haussier confirmé (4 491,48), BOS baissier confirmé
  > (4 484,10). **À noter : un Order Block baissier actif est présent malgré la
  > tendance haussière.**

**EURUSD H1 — zones réelles (corpus golden), pullback multi-TF** :
  > Tendance baissière, volatilité élevée, phase d'expansion. Le timeframe courant
  > se replie face aux timeframes supérieurs. À proximité : un Order Block haussier
  > actif (non testé) autour du prix (1,07630–1,07698) ; un Order Block baissier
  > actif (déjà testé) autour du prix (1,07574–1,07783). Structure : BOS haussier
  > provisoire (en attente) (1,07556). À noter : un Order Block haussier actif est
  > présent malgré la tendance baissière.

**Cas limite (rien à dire)** : tendance indéterminée, aucune zone/cassure/retest :
  > Tendance indéterminée, volatilité normale, phase de range.

→ concis, fluide, non télégraphique, pas de remplissage « aucune donnée ».

## 5. Tests d'honnêteté (`tests/test_narrated_reading_honesty.py`)

Corpus : **686 order blocks réels** détectés par le moteur
(`tests/fixtures/ob_golden/golden_obs.json` — XAUUSD + EURUSD, M15/H1/H4).

- **Vocabulaire interdit** : sur chaque OB réel rendu sous toutes les combinaisons
  statut × position × testé (≈ 12 000 phrases) **et** sur une matrice exhaustive
  du domaine d'énumération (tendance × volatilité × phase × relation MTF × présence
  zone/cassure/retest), **aucune** sortie ne contient un terme banni
  (`setup, signal, opportunité, gagnant, meilleur, plus sûr, recommandé,
  probabilité, cible, biais, classement, pourcentage`) ni un verbe de causalité
  (`affecte, impacte, influence, oppose, contre`). Le garde moteur
  `contains_forbidden_tokens` passe aussi (défense en profondeur).
- **Non-invention** : chaque token de prix de chaque sortie est un niveau émis par
  le moteur (`references_only_known_levels`) ; un champ null ne produit aucun texte.
- **Cas limites** : zéro zone / cassure / retest / MTF → socle seul, jamais de
  placeholder (`non disponible`, `—`, `aucune`, `null`).
- **Co-présence** : sur tous les OB opposés à la tendance, la clause est
  « présent malgré la tendance », jamais « oppose » / « contre ».

## 6. Gain

- **0 appel Haiku** pour ce bloc (auparavant 1 appel par lecture sur changement
  structurel : ~250 tokens système + 100-200 faits en entrée, ~350 en sortie).
- L'assembler market-reading **n'a plus besoin de `ANTHROPIC_API_KEY` au boot**.
- **−924 lignes** nettes (moteur Haiku + prompt + cache-store + leurs tests supprimés).

## 7. Fichiers modifiés / supprimés

**Supprimés** : `src/intelligence/haiku_description_engine.py`,
`src/storage/haiku_description_cache_store.py`,
`tests/test_haiku_description_engine.py`.

**Cœur** : `narrated_reading.py` (co-présence + retrait prompt Haiku),
`market_reading_assembler.py` (`_resolve_description` = `render_template`,
source `engine_template`, param `description_engine` retiré),
`bootstrap.py` (câblage Haiku retiré), `market_reading_schema.py`
(`DescriptionSource = Literal["engine_template"]`), `storage/__init__.py`.

**Front** : `types/market-reading.ts` (`DescriptionSource`), `ConditionsSection.tsx`
(label source unique), 9 locales (`reading.conditions.source` = « Composée par le
moteur », remplace `sourceGenerated`/`sourceFallback`), fixtures/mocks TS.

**Tests** : honnêteté (nouveau), narrated (co-présence), assembler/schema/endpoint/
chantier3/chatbot/conditions (valeur source), Playwright `narrated-reading.spec.ts`
(1280×800 + 390×844).

## 8. Résultats (à compléter au fil des vérifs)

- Python — modules cœur : `test_narrated_reading` + `test_narrated_reading_honesty`
  + assembler/schema/endpoint/mappers/chantier3 + bootstrap/chatbot/conditions/
  production : **VERTS**.
- Front — `tsc --noEmit` : **0** ; vitest market-reading components **23/23** ;
  locale-parity + copy-honesty + i18n-keys **17/17**.
- `next build` : **exit 0** (BUILD_ID généré).
- Playwright `narrated-reading.spec.ts` (1280×800 + 390×844) : **2/2** — desktop
  NarratedPanel (titre + badge « Ancrée au moteur » + narration) et onglet mobile
  « Lecture » (narration + source « Composée par le moteur »). Captures :
  `docs/audits/narrated-reading-shots/`.

## 9. Reste

- Confirmation visuelle live du fondateur avant merge sur `main`.
