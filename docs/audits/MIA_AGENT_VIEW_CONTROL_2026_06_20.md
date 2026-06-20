# M.I.A Agent — Contrôle de la VUE du graphique (affichage uniquement)

**Date** : 2026-06-20
**Branche** : `fix/mia-agent-view-control` (depuis `feat/proto-live-tick-zones`)
**Périmètre** : front + couche d'actions chatbot. **Moteur / détection INCHANGÉS.**

---

## 1. Objectif

Permettre au chatbot M.I.A Agent de piloter, par la conversation, l'**AFFICHAGE**
du graphique — et **uniquement** l'affichage. Le chatbot agit sur l'état de
rendu (visibilité des couches, filtres d'affichage, cadrage / zoom, instrument /
timeframe, mise en évidence), **jamais** sur les données de détection ni la
géométrie des structures.

### Ligne inviolable

| Autorisé (liste blanche, vue seule) | Interdit (rejeté proprement) |
|---|---|
| masquer/afficher FVG · OB · BOS/CHOCH (ou « toutes ») | créer/placer/déplacer/inventer une structure |
| filtrer les zones DÉTECTÉES (actives / taille min / proximité) | modifier les bornes/géométrie d'une zone détectée |
| se centrer/zoomer (zone détectée · prix courant · ajuster) | toute action hors liste |
| changer instrument/timeframe (combos couverts) | fournir un prix/niveau/coordonnée |
| mettre en évidence une zone DÉTECTÉE | |

**Message de refus on-brand** : « Je n'invente pas de structure — je n'affiche
que ce que le marché montre. Je peux masquer, filtrer, ou me centrer sur les
zones détectées. »

---

## 2. Contrat d'action

Le chatbot émet une action **structurée** via un 3ᵉ tool `apply_chart_view`,
validée contre une **liste blanche fermée**. Le front la **re-valide** puis
l'applique au **rendu uniquement**.

```jsonc
{ "action": "set_layer_visibility",    "params": { "layer": "fvg|ob|breaks|all", "visible": true } }
{ "action": "filter_zones",            "params": { "active_only": true, "proximity_only": true, "proximity_pct": 0.5, "min_size_pct": 0.2 } }
{ "action": "focus_zone",              "params": { "zone_id": "<id DÉTECTÉ>" } }
{ "action": "highlight_zone",          "params": { "zone_id": "<id DÉTECTÉ>" } }
{ "action": "focus_price" }            // se centrer sur le prix courant
{ "action": "fit_chart" }              // ajuster à toutes les bougies
{ "action": "reset_view" }             // réinitialiser l'affichage
{ "action": "set_instrument_timeframe","params": { "instrument": "XAUUSD|EURUSD", "timeframe": "M15|H1|H4" } }
```

### Pourquoi la détection est structurellement inatteignable

1. **Vocabulaire fermé** : aucun verbe create/place/move/resize → toute tentative
   (« mets un OB à 2000 », « agrandis ce FVG ») n'est pas représentable → rejet.
2. **Aucun champ de géométrie** : un garde-fou rejette toute action portant une
   clé `price/level/level_high/level_low/high/low/...` (`GEOMETRY_KEYS`).
3. **Ids validés** : `focus_zone`/`highlight_zone` n'acceptent qu'un `zone_id`
   **réellement émis par le moteur** ce tour-ci. Un id inventé est rejeté.
4. **Lecture seule des deux côtés** : le backend ne fait que *lire* les ids
   depuis les readings (`get_market_reading`) ; le front filtre/cadre/style des
   boîtes — il n'écrit jamais une bande.

---

## 3. Architecture de sécurité — Couche 4 ajoutée

Les 3 couches existantes sont **préservées** et complétées par une 4ᵉ dédiée aux
actions de vue :

| Couche | Fichier | Rôle |
|---|---|---|
| 1 — input adversarial | `adversarial_filter.py` | inchangée |
| 2 — Haiku + tools | `chatbot.py` | +tool `apply_chart_view`, +`view_actions`, harvest ids |
| 3 — output tokens | `output_filter.py` | inchangée |
| **4 — view whitelist** | **`view_action_filter.py`** *(nouveau)* | **valide chaque action vue ; rejette tout le reste** |

Défense en profondeur côté front : `coerceViewActions` (`lib/chart/viewActions.ts`)
re-valide les mêmes règles **et** contre les ids de zones réellement à l'écran
(gère le cas où le combo a changé entre la requête et l'application).

### Honnêteté

Le system prompt impose : décrire l'action comme un changement d'**AFFICHAGE**,
au présent (« j'ai masqué les FVG », « je me centre sur l'OB actif »), jamais
d'implication d'avoir modifié le marché ou créé une structure. Sortie reste
descriptive (les couches 1/3 niveau-1.5 restent actives).

---

## 4. Fichiers

**Backend**
- `src/intelligence/chatbot/view_action_filter.py` *(nouveau)* — Couche 4.
- `src/intelligence/chatbot/chatbot.py` — tool `apply_chart_view`, `ChatResponse.view_actions`, validation + harvest ids, prompt.
- `src/intelligence/chatbot/constants.py` — `VIEW_ACTION_REFUSAL_TEMPLATE`.
- `src/api/routes/chatbot.py` — `view_actions` dans la réponse.

**Frontend**
- `webapp/lib/chart/viewActions.ts` *(nouveau)* — types + validateur + reducer.
- `webapp/lib/chart/viewState.tsx` *(nouveau)* — `ChartViewProvider` / `useChartView(Optional)`.
- `webapp/lib/chart/zoneLayout.ts` — `filterZoneModels` (filtre d'affichage pur).
- `webapp/lib/chat/api-client.ts` — `viewActions` dans `AskResult`.
- `webapp/components/chat/ChatProvider.tsx` — `viewActionSignal`.
- `webapp/components/app/AppWorkspace.tsx` — provider + dispatcher (coerce → apply).
- `webapp/components/app/ReadingColumn.tsx` — lit la vue, la passe au chart.
- `webapp/components/app/ReadingChart.tsx` — applique couches/filtre/focus/highlight.

---

## 5. Tests

- **Backend** : `tests/test_chatbot_view_actions.py` *(nouveau, 18 tests)* —
  accept/reject de chaque action, clamp des seuils, rejet géométrie, rejet
  id inventé avec message on-brand, **détection jamais mutée**. Suite chatbot
  complète : **247 passed**.
- **Frontend** : `webapp/lib/chart/__tests__/viewActions.test.ts` *(nouveau)* —
  coerce (off-list / géométrie / id inventé / enums), reducer (toutes actions +
  reset + nonce), `filterZoneModels` (active/size/proximity, pas de mutation).
  Suite chart/chat/app : **118 passed**.
- **Build** : `npm run build` ✅ vert (`/[locale]/app` 13.5 kB).
- **Typecheck** : `tsc --noEmit` ✅.

### Couverture des exigences de la mission
- (a) chaque action de la liste blanche applique le bon changement de vue ✅
- (b) tentative hors liste (« OB à 2000 », « agrandir ce FVG ») REJETÉE +
  message on-brand ✅
- (c) la détection n'est jamais mutée ✅
- Comportement par défaut préservé (vue par défaut = tout visible, sans filtre) ✅
