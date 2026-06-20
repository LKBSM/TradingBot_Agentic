# Lecture narrée — remplacement de la « synthèse » faible (2026-06-20)

Branche : `feat/narrated-reading-synthese` (depuis `institutional-overhaul`)
Mission : remplacer la « synthèse » faible par une LECTURE NARRÉE ancrée au moteur.

## 1. Diagnostic (rappel)

La section « 🧭 Synthèse des conditions » (`webapp/.../sections/ConditionsSection.tsx`)
affichait `reading.conditions.description` : une phrase générée côté **backend** par
`HaikuDescriptionEngine`, qui ne recevait **que `tags` + `regime`**. Elle ne voyait
jamais la structure → impossible de parler des zones OB/FVG près du prix, des
niveaux réels, des cassures BOS/CHOCH ou du retest. C'était la faiblesse.

Décision (validée) : **implémentation backend** — c'est là que `conditions.description`
est produit ; un seul chemin LLM, déjà caché + on-demand + niveau-1.5, le front
reste un pur rendu.

> Note de base de branche : `institutional-overhaul` est en retard sur le terminal
> `fix/mia-agent-view-control` ; il n'a ni `mtf-trend.ts`, ni l'historique multi-events
> (`collect_zones`/`*_events`), ni `viewActions.ts`. La narration est donc ancrée sur
> les faits **présents sur cette base** : `regime.mtf_confluence`, les vraies zones
> `order_blocks`/`fair_value_gaps` (fix F3), `bos`/`choch` (dernière cassure),
> `retest_in_progress`, `close_price`.

## 2. Principe inviolable respecté

- **Le moteur = source de vérité.** L'IA ne compose qu'à partir de faits structurés.
- **Validation contre les faits** = le verrou view-control transposé aux prix :
  toute narration qui cite un niveau **absent** des faits est **rejetée** →
  régénérée une fois → sinon repli déterministe. (`references_only_known_levels`,
  analogue de `coerceViewActions` sur les ids.)
- **Présent, descriptif** : zéro prédiction / causalité / conseil / score
  (filtre `contains_forbidden_tokens` conservé).
- **Équilibré** : le contexte contraire (TF supérieur opposé, zone proche opposée
  à la tendance) est explicitement dit.
- **Provisoire vs confirmé** : `pending` → « provisoire (en attente) », `confirmed`
  → « confirmé ».
- **Read-only sur la détection** : aucune mutation ; on ne fait que LIRE
  `structure`/`regime`/`price`.

## 3. Architecture livrée

```
_build_fresh (assembler)
  └─ structure + regime + current_price  ─────────────┐
                                                       ▼
                         narrated_reading.build_reading_facts(...)   ← FAITS bornés
                                                       │   (tendance, vol, phase,
                                                       │    relation MTF, zones
                                                       │    près du prix + statut,
                                                       │    cassures, retest)
                                  ┌────────────────────┴────────────────────┐
                                  ▼                                          ▼
        HaikuDescriptionEngine.generate                        render_template (repli
        ├─ build_user_prompt (faits → Haiku)                   déterministe, toujours
        ├─ VALIDATION : forbidden tokens ET                    factuel, toujours présent)
        │  references_only_known_levels
        ├─ échec → 1 régénération → sinon repli
        └─ cache (empreinte STRUCTURELLE, prix exclu)
                                  │
                                  ▼
              conditions.description  →  ConditionsSection (« Lecture narrée »)
```

### Fichiers
| Fichier | Rôle |
|---|---|
| `src/intelligence/narrated_reading.py` *(nouveau)* | Source unique : FAITS, template déterministe, prompt, validateur d'ancrage, décimales. |
| `src/intelligence/haiku_description_engine.py` | Signature enrichie `generate(tags, regime, structure, price, instrument)` ; double validation + 1 retry + repli ; empreinte de cache structurelle. |
| `src/intelligence/market_reading_assembler.py` | `_resolve_description` passe structure/prix/instrument ; repli via `render_template`. |
| `src/intelligence/market_reading_schema.py` | `DESCRIPTION_MAX_LENGTH` 280 → 500 (paragraphe). Non consommé par Telegram. |
| `webapp/.../sections/ConditionsSection.tsx` | Rend le paragraphe (multi-lignes), label « Lecture narrée / Narration générée / Lecture modèle (repli) ». Remplacement, pas ajout. |
| `scripts/generate_validation_dataset.py` | Appel mis à jour à la nouvelle signature. |

### Maîtrise du coût IA
- Génération **on-demand** (assembler lazy), pas à chaque tick.
- **Cache** par empreinte **structurelle** (regime + zones + cassures + retest) ;
  le **prix brut est exclu de la clé** → un tick calme ne régénère pas ; régénération
  sur **changement notable** (structure/regime).

### Choix d'ingénierie — AFFICHAGE vs VALIDATION (découplés)
- **Affichage fr-FR** : virgule décimale + séparateur de milliers (espace fine
  insécable U+202F, comme `toLocaleString('fr-FR')`) → cohérent avec l'en-tête
  (« 4 320,12 »). C'est ce que voit le lecteur et ce que recopie le modèle.
- **Validation canonique** : point décimal, sans séparateur (`4320.12`). Le
  validateur `references_only_known_levels` **normalise** chaque nombre du texte
  (retrait des espaces, virgule→point) avant de vérifier l'appartenance au set de
  niveaux canoniques. Donc on **affiche** `4 320,12` mais on **valide** sur
  `4320.12`, sans ambiguïté de parsing.
- Les entiers nus (`M15`, « les 3 TF ») n'ont pas de séparateur décimal → ignorés
  par construction. Tout niveau inventé (display ou canonique) → rejet.
- `ZoneFact`/`BreakFact`/`ReadingFacts` portent les **deux** formes (`*` display,
  `*_canon` validation) ; `allowed_levels` ne collecte que le canonique.

## 4. Tests (build vert)

`tests/test_narrated_reading.py` *(nouveau)* couvre les 4 exigences :
- **(a)** ancrage : niveau réel accepté, niveau étranger (`2222.22`) **rejeté** ;
  entiers nus non traités comme niveaux ; `allowed_levels` complet.
- **(b)** repli déterministe **factuel** et auto-ancré ; distingue provisoire/confirmé.
- **(c)** template **sans token interdit**, sans marqueur de futur (`va`, `sera`,
  `devrait`, `pourrait`, `probab`).
- **(d)** **contexte contraire** inclus (pullback contre TF supérieurs ; zone active
  opposée près du prix) ; **absent** quand le tableau est unidirectionnel.
- Sélection des zones près du prix (fenêtre `PROXIMITY_PCT`) + position above/below.

`tests/test_haiku_description_engine.py` réécrit : ancrage au niveau moteur
(rejet → repli), retry-once, cache **n'intègre pas le prix brut**, contamination →
repli sans écriture cache.

Résultats :
- **Backend** : 405 tests sur la surface impactée (market_reading / narrated / haiku /
  chatbot / qa / signal_summary / bootstrap) **verts**, 0 régression.
- **Webapp** : `tsc --noEmit` ✅ · vitest 86/86 ✅ · `next build` ✅.

Tests/copie mis à jour pour le renommage « Synthèse » → « Lecture narrée » :
`market-reading-components.test.tsx`, `e2e/sections.spec.ts`, `FaqSection.tsx`,
`app/[locale]/app/page.tsx`, `test_market_reading_schema.py`,
`test_market_reading_assembler.py`, `test_chantier3_smoke_e2e.py`.

## 5. Suites / non-fait (honnête)
- La narration s'appuie sur la **dernière** cassure `bos`/`choch` (pas d'historique
  multi-events sur cette base). **Ne pas enrichir `build_reading_facts` sur cette
  base périmée** : consolider d'abord les branches dans `institutional-overhaul`,
  puis enrichir (trivial, même contrat). La narration reste ancrée sur les faits
  actuels.
- Affichage/validation découplés (fait) : fr-FR à l'affichage, canonique à la
  validation. Cf. « Choix d'ingénierie » ci-dessus.
