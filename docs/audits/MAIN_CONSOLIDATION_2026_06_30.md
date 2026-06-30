# Consolidation `main` — 2026-06-30

**Branche d'intégration** : `integration/consolidate-main-2026-06-30` (depuis `main` `ec22d13`)
**Objet** : regrouper le redesign chat + ma feature clic-zone→graphe et les features UI
récentes dans `main`, en réglant tous les conflits.

> Travail fait sur une branche d'intégration isolée, build+tests verts à chaque étape.
> `main` n'est basculé qu'après validation. Pas de force push.

---

## Périmètre retenu (choix founder : redesign + features récentes)

| # | Branche | Résultat | Note |
|---|---------|----------|------|
| 1 | `feat/zone-list-click-to-chart` (= `feat/chat-ui-redesign` + ma feature) | ✅ mergé | 3 conflits chat résolus |
| 2 | `feat/reading-panel-polish` | ⏭️ no-op | déjà absorbé par le redesign |
| 3 | `feat/agent-multihide-chat-scroll` | ✅ mergé | masquage multi-zones (inclut hide-specific-zone) |
| 4 | `feat/hide-specific-zone` | ⏭️ no-op | absorbé par #3 |
| 5 | `feat/zone-label-styling` | ✅ mergé | libellés de zone (auto-merge ReadingChart) |
| 6 | `feat/scanner-important-news` | ✅ mergé | compte uniquement les actus high-impact |
| 7 | `feat/conditions-scanner-page` | ⛔ **sauté** | **déjà dans `main`** en version corrigée (`40498d8`, port + correct daté postérieurement) — le merger serait régressif |
| 8 | `fix/fvg-session-gap-awareness` | ✅ mergé | ⚠️ modifie la détection FVG (saut de session) |
| 9 | `diagnostic/ohlcdev-quality-eval` | ✅ mergé | doc d'audit seul |

**7 merges effectifs · 2 no-op · 1 sauté (redondant/régressif).**

## Décision clé — `feat/conditions-scanner-page` sauté

Le merge a produit 13 conflits **add/add**. Investigation :
- `main` contient déjà le Conditions Scanner via `40498d8` « Conditions Scanner **ported +
  corrected** onto the chart product » (2026-06-20 16:21).
- Dernier commit de la branche : `93063f6` (2026-06-20 13:49) — **antérieur** au port.
- `git diff --name-status main…feat/conditions-scanner-page` ne liste **aucun fichier
  ajouté** par la branche : tout son contenu est déjà dans `main`.

→ La branche est la version **pré-port** ; la merger réintroduirait du code plus ancien et
annulerait les corrections. **Sautée volontairement.** (À confirmer / objecter par le founder.)

## Résolutions de conflits (merge #1 — redesign vs main)

`main` avait fait évoluer le chat en parallèle (`c8bc8d7` ancrage anti scroll-to-bottom)
pendant que le redesign le refondait. Principe de résolution : **garder la nouvelle UI du
redesign + l'UX d'ancrage de `main`**.

- **`ChatPanel.tsx`** — conserve `useChatAnchorScroll` (import + usage), retire `MiaAgentLogo`
  inutilisé (le header redesign utilise `AgentAvatar`).
- **`ChatMessage.tsx`** — nouvelle bulle redesign **+ réinjection de `data-chat-role={role}`**
  sur les deux wrappers (user/assistant), dont dépend `useChatAnchorScroll` pour ancrer la
  dernière question.
- **`AppChatSidebar.tsx`** — UI redesign (icônes, `askFreeForm`, `ChatWelcome`) + remplacement
  du scroll-to-bottom naïf par `useChatAnchorScroll` ; import `React` retiré (devenu inutile,
  JSX runtime automatique).

Merges #3, #5, #6, #8, #9 : auto-merge sans conflit.

## Vérifications (branche d'intégration)

| Contrôle | Résultat |
|---|---|
| `tsc --noEmit` | ✅ 0 erreur |
| Vitest (suite complète) | ✅ **323 / 323** (36 fichiers) |
| `next build` | ✅ vert |
| pytest (modules touchés) | ✅ 116 / 116 |
| pytest (suite backend complète) | _voir ci-dessous_ |

## Bascule `main`

Une fois tout vert et le rendu live validé par le founder :
`main` fast-forward / merge depuis `integration/consolidate-main-2026-06-30`, puis push.
**Pas avant confirmation, pas de force push.**
