# AUDIT — Bascule interactive M.I.A : mode colonne ↔ mode bulle (/app)

**Branche** : `feat/app-mia-column-bubble-toggle` (depuis `origin/main` @ `acc72d8`, PR #182)
**Worktree dédié** : `C:/MyPythonProjects/wt-mia-bubble`
**Date** : 2026-08-20
**Cible** : page `/app`, desktop ≥1280.

---

## 1. Contexte & motif

Donner au client (et au fondateur pour ses propres tests) un contrôle **dans la page**
`/app` pour basculer à volonté entre deux dispositions de M.I.A :

- **Mode colonne** — M.I.A dockée en 3ᵉ colonne à droite (spec §6), graphique au centre.
- **Mode bulle** — M.I.A réduite à la bulle flottante en bas à droite, le centre récupère
  toute la largeur ; la bulle rouvre le chat en tiroir.

Le comportement du chatbot (ancrage moteur, refus d'ordres, aucune formulation prédictive)
est **identique** dans les deux modes — seule la disposition change.

## 2. Diagnostic de dépendance (§0bis) — le point critique

Vérifié sur `origin/main` à jour (worktree neuf depuis `acc72d8`).

**Les deux rendus existaient-ils ?** Oui, mais asymétriquement :

| Rendu | Où | État avant |
|---|---|---|
| Colonne dockée | `≥1280`, `.chatcol` = 3ᵉ piste grille (338px) | présent |
| Bulle `.chat-fab` + tiroir off-canvas | `768–1279` uniquement (`@media(max-width:1279px)`) | présent, **inactif en desktop** |

La mission précédente (PR #182) avait **explicitement remplacé** l'idée de bulle desktop par
« colonne visible ↔ colonne masquée » + un bouton « rouvrir » dans l'apphead (`.chat-reopen`).
→ Le rendu bulle **n'a donc pas été réinventé** : il existait déjà (tablet), il suffisait de
l'étendre au desktop.

**État concurrent ?** OUI — `ChatColumnContext` fournissait déjà un booléen `open`
(colonne visible/masquée), persisté en localStorage. C'était **déjà** un état d'affichage à 2
valeurs. Empiler un second mécanisme aurait créé deux systèmes concurrents.

## 3. Décision — FUSION (validée par le fondateur)

Trois décisions remontées, toutes tranchées vers la recommandation :

1. **Fusionner** : réutiliser le booléen `ChatColumnContext.open` comme **unique** état de
   disposition — `open=true` → colonne, `open=false` → bulle. `show()`/`hide()` deviennent
   « aller en colonne » / « aller en bulle ». Zéro migration (clé localStorage conservée).
2. **Persistance** : localStorage conservé (défaut = colonne).
3. **Mobile** : bascule visible ≥1280 seulement ; le responsive existant (tiroir 768–1279,
   onglet <768) reste **inchangé**.

## 4. Implémentation

- **`ChatColumnContext.tsx`** — doc réécrite (colonne↔bulle) ; API booléenne inchangée.
- **`ShellChat.tsx`** — expose `.chat-fab` + tiroir en mode bulle ; ajoute `setDisplayMode`
  partagé (passer en colonne ferme aussi le tiroir). Passe `displayMode` + `onSetDisplayMode`
  à `AppChatSidebar`.
- **`AppChatSidebar.tsx`** — le bouton d'entête devient un **toggle bidirectionnel** :
  « Réduire en bulle » (`PanelRightClose`) en colonne, « Afficher en colonne »
  (`PanelLeftClose`) en bulle. Toujours `xl:inline-flex` (≥1280). Prop `onHide` retirée.
- **`DesktopReading.tsx`** — bouton concurrent `.chat-reopen` de l'apphead **supprimé**
  (imports `useChatColumn`/`PanelRightOpen` nettoyés). La bulle porte désormais la
  réouverture + le retour en colonne.
- **`shell.css`** — les règles tiroir/fab/backdrop étendues à `@media(min-width:1280px)` sous
  `.chat-collapsed` (même rendu que le tablet) ; CSS orphelin `.chat-reopen` supprimé.
- **i18n 9 locales** — 3 clés orphelines (`hidePanel`/`showPanel`/`showPanelShort`) remplacées
  par `collapseToBubble` / `dockToColumn` (script Node chirurgical, CRLF/UTF-8 préservés,
  parité stricte maintenue).

**Conversation préservée** : `AppChatSidebar` (et son `ChatProvider`) n'est **jamais
démonté** — basculer ne change que des classes CSS (`display`/`transform`). Prouvé par un
marqueur DOM posé impérativement sur le sous-arbre du chat, qui survit à un aller-retour
colonne→bulle→colonne (test e).

## 5. Tests

- **tsc** : 0 erreur.
- **Build** prod : OK.
- **vitest** : 958/959 (l'unique échec = `AccountMenu` logout, **flake de timeout** sous
  contention IO/Defender, repassé vert isolé — sans rapport avec la modification). Parité i18n
  + garde clés : 13/13.
- **Playwright** (`mia-column-toggle.spec.ts`, prod build, port 3217) : **6/6** (3 tests × 2
  projets).
  - 1280×800 : colonne par défaut → bulle → tiroir → retour colonne (largeur du centre
    +338px puis restaurée à ±2px) ; note pédagogique identique dans les deux modes ; marqueur
    de persistance intact.
  - Persistance : le mode bulle survit à un reload.
  - 390×844 : ni toggle, ni bulle desktop ne fuient (onglets MobileWorkspace).

Captures : `docs/audits/mia-bubble-shots/` (a-column-default, b-bubble, c-bubble-open,
d-column-back, e-mobile).

## 6. Vérification anti-régression

- `grep` confirme **zéro** référence résiduelle à `onHide` / `hidePanel` / `showPanel` /
  `chat-reopen` / `showPanelShort` dans le code et les tests.
- **Un seul** état d'affichage (`ChatColumnContext.open`) : plus de mécanisme « masquer »
  concurrent.

## 7. Reste à faire

- **Confirmation visuelle live du fondateur** avant merge sur `main` (règle de la mission).
