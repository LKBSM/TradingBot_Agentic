# Scanner de conditions — livraison (2026-06-20)

> Branche `feat/conditions-scanner-page` (depuis `institutional-overhaul`),
> worktree isolé `C:/MyPythonProjects/wt-conditions-scanner`.

## 1. Objet

Nouvelle page **« Scanner »** : le client compose ses **conditions structurelles**
(sa « stratégie ») et l'outil affiche les marchés/timeframes où ces conditions
sont **présentes en ce moment**. Gain de temps : il ne feuillette plus 6 graphes,
il garde le jugement.

Outil **descriptif**, pas un service de signaux. Ligne inviolable respectée :
- Conditions = **faits structurels au présent uniquement**. Aucune condition
  prédictive/d'issue n'existe — elle n'est même pas représentable (palette fermée
  + `Literal` Pydantic côté requête).
- Matching **transparent** : chaque combo affiche conditions remplies **et** non
  remplies (« 2 de tes 3 »). Aucun score de similarité, aucun classement qualité,
  aucune conviction.
- Chaque carte montre le **contexte complet**, y compris ce qui va contre.
- Vocabulaire : « Scanner » / « Conditions présentes » ; bouton **« Analyser »**
  (jamais « Trader »). Zéro langage prédictif/prescriptif.
- **Moteur de détection INCHANGÉ.** Le scan est en **lecture seule** par-dessus les
  reads déjà produits.

## 2. Architecture livrée

### Backend (lecture seule)
- `src/intelligence/conditions_scanner.py` — **évaluateur pur** (aucune dépendance
  FastAPI, aucun I/O). `PALETTE` (source de vérité, 5 types présents),
  `evaluate_condition`, `evaluate_reading` (logique AND/OR + contexte neutre).
- `src/api/routes/conditions_scan.py` :
  - `POST /api/conditions-scan` — boucle les 6 combos `XAU/EUR × M15/H1/H4` via
    `assembler.readings_store.get_latest_reading()` (**pur `SELECT`**, jamais
    `get_or_generate`, jamais d'écriture). Renvoie `matches[]`
    (`conditions_met`/`conditions_unmet`/`context` + `matched`/`met_count`) +
    `unavailable[]`. Ordre **fixe** (jamais trié par nombre de conditions →
    aucun classement implicite).
  - `GET /api/conditions-scan/palette` — palette publique pour le builder.
  - Types de condition en `Literal` → une condition inconnue/prédictive est
    rejetée en **422** avant toute évaluation. `assert` de cohérence
    Literal ↔ `PALETTE` au chargement du module.
- `src/api/app.py` — router enregistré (2 lignes : import + `include_router`).

**Garantie read-only** : le double de test `_RecordingStore` lève une
`AssertionError` si `save_reading`/`mark_combination_active` est appelé, et
`_RecordingAssembler.get_or_generate` lève également. Le test passe → le scan ne
touche que `get_latest_reading`.

### Palette (présent uniquement)
| type | fait vérifié au présent |
|---|---|
| `mtf_aligned` | les 3 TF (h4/h1/m15) pointent même direction maintenant |
| `price_in_ob` | prix courant dans un Order Block actif |
| `price_in_fvg` | prix courant dans un FVG non comblé |
| `ob_fvg_confluence` | prix dans OB **et** FVG simultanément |
| `bos_recent_confirmed` | BOS `confirmed` daté des N dernières bougies |

Chaque condition `*_in_*`/`mtf`/`bos` accepte un filtre `direction` (any/bullish/
bearish). `bos_recent_confirmed` accepte `max_bars` (défaut 5). **Aucune** option
d'issue future.

### Frontend (nouvelle page)
- `webapp/lib/conditions/` : `types.ts`, `palette.ts` (miroir backend),
  `config-store.ts` (`useConditionsConfig` — localStorage `mia.conditionsConfig.v1`,
  pattern CookieBanner), `api-client.ts` (POST scan), `app-link.ts`
  (`buildAppHref` deep-link + `resolveComboFromQuery`).
- `webapp/components/scanner/` : `ConditionsBuilder` (onboarding + édition),
  `ComboCard` (met/unmet + contexte complet + « Analyser »), `ScanResults`
  (matchs complets · accordéon « Correspondances partielles » · footer de
  couverture honnête), `ScannerWorkspace` (orchestrateur), `labels.ts`.
- `webapp/app/[locale]/scanner/page.tsx` — la page.
- Deep-link « Analyser » : `webapp/app/[locale]/app/page.tsx` lit
  `?instrument=&timeframe=` (validé), passé à `AppWorkspace(initialCombo)`
  → `ActiveComboProvider(initial)`. **Additif, défaut préservé** (sans query →
  comportement actuel inchangé).
- `webapp/components/Nav.tsx` — lien « Scanner » ajouté (discoverabilité).

### Affichage des résultats (décisions validées)
- **Backend endpoint** pour l'évaluation (vs front seul).
- **Section « partiels » séparée** : matchs complets en tête, accordéon repliable
  pour les combos remplissant ≥1 condition (transparence), footer listant les
  combos sans condition présente et les lectures non encore générées (aucune
  troncature silencieuse).
- **Lecture pure + horodatage** : on lit le dernier reading dispo et on affiche
  son âge ; combos sans reading → `unavailable`. Jamais de génération à la demande.

## 3. Tests

### Backend (`pytest`) — 31 nouveaux, verts
- `tests/test_conditions_scanner.py` (24) : met/unmet sur données connues pour
  chaque condition (mtf alignés/divergents, prix dans/hors OB/FVG, OB mitigé,
  filtres direction, confluence, BOS confirmé/pending/trop ancien/absent), logique
  AND/OR, contexte complet. **Palette : 5 types, tous `present`, aucun vocabulaire
  prédictif.**
- `tests/test_conditions_scan_endpoint.py` (7) : match complet, **partiel
  transparent**, **read-only (seul `get_latest_reading` touché, writes/détection
  lèvent)**, **type prédictif rejeté en 422**, ≥1 condition requise, 503 si non
  câblé, palette présent-only.
- Régression : `tests/test_market_reading_endpoint.py` toujours vert (40 au total
  avec les nouveaux).

### Frontend (`vitest`) — 13 nouveaux, verts (99 au total, 0 régression)
- `palette.test.ts` : 5 types présents, **aucun mot prédictif**.
- `app-link.test.ts` : **« Analyser » pointe vers /app avec le bon
  instrument+timeframe** ; préfixe locale `as-needed` ; validation de périmètre.
- `ComboCard.test.tsx` : affiche met **et** unmet, lien Analyser correct, **aucun
  « Trader »**.
- `config-store.test.ts` : 1re visite = null, persistance, reset.

### Build
- `tsc --noEmit` : **0 erreur**.
- `next build` : **vert**. Route `/[locale]/scanner` = 7.39 kB (126 kB first load).

## 4. Conformité à la ligne inviolable — checklist

| Exigence | Statut |
|---|---|
| Conditions présentes uniquement, aucune prédictive proposable | ✅ palette fermée + `Literal` 422 + tests back & front |
| Matching transparent (remplies/non remplies), pas de score/classement | ✅ `ComboCard` met+unmet, ordre fixe |
| Contexte complet (y compris ce qui va contre) | ✅ `build_context` + bloc contexte |
| Vocabulaire « Scanner »/« Analyser », jamais « Trader »/prédictif | ✅ + test anti-« Trader » |
| Détection inchangée, scan read-only | ✅ `get_latest_reading` only + test anti-write |
| Comportement existant préservé | ✅ deep-link additif, défaut inchangé, 0 régression test |

## 5. Limites / notes

- Le scan lit la **dernière** lecture produite par combo ; la fraîcheur dépend du
  `MarketReadingScheduler` (tâche de fond existante). L'âge est affiché par carte ;
  les combos sans lecture sont listés en `unavailable` (jamais inventés).
- Palette dupliquée back (Python) / front (TS) volontairement, chacune testée
  pour rester présent-only ; un `assert` lie le `Literal` de requête à la palette
  backend pour empêcher toute dérive d'un seul côté.
- i18n dormant dans la webapp (FR en dur) : la page suit cette convention.
