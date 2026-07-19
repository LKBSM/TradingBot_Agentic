# Panneau M.I.A Agent — calage au début de la réponse + en-tête réorganisé

**Date** : 2026-07-19
**Branche** : `feat/chat-panel-scroll-header` (worktree dédié `wt-chat-panel-scroll`, depuis `origin/main` = b2516f9)
**Périmètre** : UX du panneau de chat docké (colonne droite de `/app`). Aucun changement moteur, logique IA, ni autre écran.

## Objectif

Deux correctifs UX ciblés sur le panneau `AppChatSidebar` :

1. **Scroll** — à l'envoi d'un message, caler la vue sur le **début de la réponse de M.I.A Agent** (premier mot visible en haut), au lieu de laisser la vue sauter en bas d'une longue réponse.
2. **En-tête** — « Discussions » et « Réinitialiser » deviennent deux **icon-buttons** en haut à droite (libellé au survol via tooltip), le titre + contexte (`instrument · TF`) tiennent sur **une ligne** à gauche, et le disclaimer d'honnêteté passe sur **une seule ligne** avec icône.

Wording du disclaimer **inchangé** (clé i18n `chat.pedagogicalNote`).

## Fichiers touchés (diff strictement limité au panneau chat)

| Fichier | Nature |
|---|---|
| `webapp/components/chat/useChatAnchorScroll.ts` | Ajout d'une option `{ anchor: 'user' \| 'assistant' }` (défaut `'user'`). |
| `webapp/components/app/AppChatSidebar.tsx` | Passe `anchor: 'assistant'` + refonte markup/style de l'en-tête. |

Le hook est partagé avec le slide-over de la landing (`ChatPanel.tsx`). Le **défaut `'user'` préserve à l'identique** le comportement de la landing — seule la sidebar dockée opte pour l'ancrage réponse. `ChatPanel.tsx` n'est pas modifié.

## (a) Comportement de scroll

`useChatAnchorScroll` ancre déjà un message près du haut (anti scroll-to-bottom, désengagement au 1er geste). Ajout d'une cible d'ancrage :

- `anchor: 'user'` (défaut, landing) : ancre la **question** — question + début de réponse visibles.
- `anchor: 'assistant'` (sidebar) : ancre la **réponse**. Tant que la réponse n'a pas monté (phase « réflexion »), fallback sur la question ; dès que le bloc assistant apparaît, l'ancre bascule sur **son premier mot**. Le streaming re-épingle au même point (no-op une fois atteint) → **ne resaute jamais en bas** à chaque token.

L'ancrage s'appuie sur `[data-chat-role="assistant"]`, déjà posé par `<ChatMessage />` (aucune modification de `ChatMessage`). `scrollTo` clampe automatiquement : sur une réponse courte, la vue s'arrête au maximum scrollable sans « trou » et la question peut rester visible — dégradation gracieuse.

## (b) En-tête réorganisé

- **Deux icon-buttons** (`size="icon"`, `h-8 w-8`, `variant="ghost"`) à droite : `History` (Discussions) et `RotateCcw` (Réinitialiser). `aria-label` conservé + `Tooltip` (`@/components/ui/tooltip`) affichant le libellé au survol. Plus de texte inline (`sr-only sm:not-sr-only` retiré).
- **Titre + contexte sur une ligne** : `M.I.A Agent ● · <instrument> · <TF>` avec `truncate` sur la portion contexte (ne peut plus wrapper).
- **Disclaimer sur une ligne** avec icône `GraduationCap` discrète + `truncate`. Texte identique (`chat.pedagogicalNote`).
- Conditions et actions **inchangées** : Discussions visible si `recentThreads.length > 0` (toggle `showRecents`), Réinitialiser visible si `!empty` (`resetTurns`). La liste « Discussions récentes » sous l'en-tête est intacte.

## Vérifications

- `npx tsc --noEmit` : **0 erreur**.
- `npm run build` : **succès**.
- `npx vitest run` : voir résultat ci-dessous.
- `git diff --stat` : **2 fichiers**, panneau chat uniquement — preuve du périmètre strict.

## Discipline

- Worktree dédié, staging explicite (pas de `git add -A`), pas de force push.
- Merge sur `main` **uniquement après confirmation live** du fondateur.
