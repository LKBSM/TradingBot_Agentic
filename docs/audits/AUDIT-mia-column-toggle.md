# AUDIT — M.I.A en 3e colonne /app + bouton masquer/afficher

**Branche :** `feat/app-mia-column-toggle` (worktree dédié, depuis `origin/main` `ae6904c`)
**Date :** 2026-08-20
**Statut :** implémenté, tests verts — **en attente de confirmation visuelle live du fondateur avant merge**

---

## 1. Contexte & diagnostic

### Le composant colonne existait déjà — rien à réinventer
`AppChatSidebar.tsx` (`webapp/components/app/`) est la colonne M.I.A complète (en-tête
« M.I.A Agent » + statut, note pédagogique, transcript, saisie, ligne de conformité). Elle est
montée par `ShellChat.tsx` → `<aside class="chatcol">` dans la grille du `ProductShell`. **Aucune
suppression, aucune réécriture de chatbot.**

### La « bulle » n'était pas une régression du composant
Traçage git :
- `88316e8 feat(ui-1)` a posé la grille 3 colonnes `232px · 1fr · 338px` avec `.chatcol` dockée en
  permanence sur /app (fidèle à `docs/design/reference-desktop.html`, pur 3-colonnes).
- `f20f4d1 feat(ui-2b)` a **ajouté** le tiroir off-canvas + le bouton rond `.chat-fab`
  **uniquement pour 768–1279px** (tablette), pour préserver la largeur du centre.

À **≥1280px la colonne était donc déjà dockée et visible par défaut**. La bulle ronde en bas à
droite (`.chat-fab`) n'apparaît qu'entre 768 et 1279px — le viewport où le fondateur observait le
problème.

### Le vrai manque
À ≥1280px, **aucun moyen de masquer** la colonne pour élargir le graphique. C'était le livrable
net-new : colonne par défaut **+ bouton masquer/afficher**, sans maintenir de double mode
« colonne vs bulle » sur desktop.

## 2. Décisions produit (validées par le fondateur au diagnostic)
- **Responsive** : ≥1280 = colonne + toggle ; 768–1279 = tiroir actuel **inchangé** ; <768 =
  onglet Chat de `MobileWorkspace` **inchangé**. Le « double mode » n'est retiré que sur desktop.
- **Persistance** : l'état masqué/affiché est **mémorisé en localStorage** (comme les lectures
  enregistrées) et se réapplique aux visites suivantes. Défaut si rien de stocké = **colonne visible**.

## 3. Implémentation

| Fichier | Changement |
|---|---|
| `components/shell/ChatColumnContext.tsx` *(nouveau)* | Contexte `open/hide/show/toggle`, défaut `true`, persistance localStorage (`mia.app.chat-column-open`), hydraté post-montage. `useChatColumn()` renvoie un fallback inerte hors provider (chat monté par le mobile). |
| `components/shell/ProductShell.tsx` | Enveloppe le châssis dans `ChatColumnProvider` (+ `ShellFrame` consommateur). Sur /app, si `!open` → classe `chat-collapsed`. |
| `components/shell/ShellChat.tsx` | Passe `onHide={hideColumn}` à `AppChatSidebar`. |
| `components/app/AppChatSidebar.tsx` | Prop optionnelle `onHide` → bouton « replier » (`PanelRightClose`) dans l'en-tête, `hidden xl:inline-flex` (desktop ≥1280 seulement). |
| `components/app/DesktopReading.tsx` | Bouton de réouverture (`PanelRightOpen` + « M.I.A ») dans l'`apphead`, visible **uniquement** quand la colonne est masquée. **Pas de bulle flottante sur desktop.** |
| `components/shell/shell.css` | `@media (min-width:1280px) .app-shell.chat-collapsed` → grille `232px 1fr` + `.chatcol { display:none }`. Style compact `.chat-reopen`. |
| `messages/*.json` (9 locales) | `app.chat.hidePanel` / `showPanel` / `showPanelShort`. |
| `tests/e2e/mia-column-toggle.spec.ts` *(nouveau)* | Vérifie les 3 états + persistance + non-fuite mobile. |

### Point d'attention traité (§5 du diagnostic)
Les media queries `≤1279`/`≤767` sont scoppées `:not(.no-chat)` pour piloter le tiroir tablette.
J'ai donc **délibérément utilisé une classe dédiée `chat-collapsed`** (et non `no-chat`) et scoppé
ses règles à `min-width:1280px`, pour ne **pas** casser le tiroir/onglet sous 1280.

### Comportement du chatbot inchangé
La colonne monte le même `AppChatSidebar` (ancrage moteur, refus d'ordre, non-prédictif). Seuls
l'emplacement/visibilité changent. Confirmé par `chatbot-backend-integration.spec.ts` Test 2
(demande d'action → refus) qui reste vert.

## 4. Tests d'honnêteté

Captures dans `docs/audits/mia-column-shots/` :
- **a-column-default.png** — vue 3 colonnes, colonne M.I.A visible **par défaut** au chargement.
- **b-column-hidden.png** — colonne masquée, graphique **élargi** pleine largeur (aucun espace
  vide), bouton « M.I.A » de réouverture apparu dans l'apphead.
- **c-column-reopened.png** — colonne rétablie depuis l'état masqué, bouton de réouverture disparu.
- **d-mobile.png** (390×844) — `MobileWorkspace` (onglets Marchés · Lecture · Chat) inchangé,
  aucun toggle desktop ne fuit.

## 5. Vérifications

- `tsc --noEmit` : **0 erreur**
- `next build` (CI=1) : **compilé**
- `vitest run` : **959/959** (104 fichiers)
- Playwright (chromium-desktop) :
  - `mia-column-toggle` **3/3** (défaut/masqué/réouvert + persistance + mobile)
  - `shell` **2/2** (non-régression)
  - `ui2b-centre` **5/5** — dont *« 1024px : tiroir via FAB »* et *« ≥1280px : docké sans FAB »* → **tiroir tablette et dock desktop intacts**
  - `pub-mia-chat` **3/3**, `chatbot-backend-integration` **3/3** (refus d'ordre intact)

## 6. Reste à faire
- **Confirmation visuelle live du fondateur** (les captures sont en données mockées).
- Merge sur `main` seulement après cette confirmation.
