# Persistance du chat M.I.A par combo+TF — client-only (V1)

**Date** : 2026-07-04
**Branche** : `feat/chat-persistence-client` (worktree dédié, base `main` = 1526049)
**Statut** : livré, en attente de confirmation live avant merge.

## Problème

Changer de timeframe (ou d'instrument) puis revenir effaçait la conversation du
chat M.I.A. Attendu : revenir sur un combo+TF (ex. XAU H1) restaure SA
discussion, plus une liste des N dernières discussions.

## Diagnostic (cause racine)

Le reset n'était **pas** un démontage de composant : `ChatProvider` est monté
dans `webapp/app/[locale]/layout.tsx`, au-dessus du switch de combo, et survit
au changement de TF. Le reset était **explicite** : le provider ne tenait qu'un
seul tableau `turns` plat, et `openForCombo` / `openFor` faisaient
`setTurns([])` dès que l'id du contexte changeait (« Resets the conversation
when the combo changes »). `AppWorkspace` appelle `openForCombo(active)` à
chaque changement de combo → conversation écrasée.

Bug latent corrigé au passage : sur la landing, `HeroChatPreview` appelait
`appendExchange` **avant** `openFor` ; sans fil actif, le tout premier échange
scripté était perdu (l'ancien code le wipait via le reset d'`openFor`).
L'ordre est inversé et les updaters fonctionnels composent dans le même batch.

## Solution

### 1. Scoping par fil (règle le bug à lui seul, sans stockage)

`ChatProvider` tient désormais un état unique
`{ active, threads: Record<threadId, StoredThread> }` :

- Clé de fil = id du signal : `app:{instrument}:{timeframe}` pour les combos
  /app (ex. `app:XAUUSD:H1`), id réel du signal pour les lectures landing.
- `openForCombo` / `openFor` **basculent** le fil actif (création paresseuse),
  ne suppriment plus rien.
- `turns` exposé = fil actif → **API du contexte inchangée**, aucun consommateur
  (`AppChatSidebar`, `ChatPanel`, `MobileWorkspace`) à adapter.
- `askFreeForm` épingle le fil de destination au moment de l'envoi : une
  réponse qui arrive après un changement de combo atterrit dans le bon fil.
- `resetTurns` ne vide que le fil actif (mémoire + copie stockée).

### 2. Survie au rafraîchissement — localStorage UNIQUEMENT

Nouveau module `webapp/lib/chat/thread-store.ts` (pattern établi de
`lib/market-reading/pins.ts`) :

- Clé `mia.chatThreads.v1`, hydratation une fois côté client, écriture à chaque
  changement de fil (gardée par un flag `hydrated` pour ne jamais écraser le
  stockage avec l'état initial vide).
- **Plafonds / purge** : 40 tours max par fil (rognage sans démarrer
  mi-échange), 12 fils max (les plus récents), budget sérialisé 200 000
  caractères avec boucle drop-oldest (puis moitié du dernier fil si un seul
  fil déborde).
- **Sanitisation défensive en lecture** : rôles/champs validés, périmètre
  instruments/TF vérifié (`SUPPORTED_INSTRUMENTS × SUPPORTED_TIMEFRAMES`),
  cohérence `id === app:{instrument}:{tf}` exigée, dédoublonnage. Un storage
  corrompu rend `[]`.
- try/catch sur `setItem` : quota/privacy mode → dégradation en mémoire seule.
- Seuls les fils `app:*` sont persistés ; les fils signaux de la landing
  restent en mémoire de session.

### 3. Liste des dernières discussions

- `ChatProvider` expose `recentThreads` (≤ 6 fils `app:*` non vides, triés par
  dernière activité) : instrument, TF, horodatage, extrait du dernier message.
- `AppChatSidebar` : bouton « Discussions » (icône historique) dans l'en-tête →
  panneau listant les fils récents ; cliquer bascule le workspace sur ce combo
  via le `onSelect` existant (prop `onSelectCombo`, câblée desktop + mobile ;
  sur mobile on reste sur l'onglet Chat).

## Frontière légale (Loi 25) — CONFIRMÉE

- **Aucun** nouvel endpoint, **aucune** table, **aucune** écriture dans
  `market_readings.db`, **aucun** appel serveur ajouté. Grep du diff : la seule
  occurrence réseau dans les lignes ajoutées est un commentaire.
- Toute la persistance est dans le localStorage du navigateur de l'utilisateur
  — l'entreprise ne détient rien.
- Flux serveur inchangé : l'historique éphémère (6 derniers tours) envoyé par
  requête à `POST /api/chatbot/message` existait déjà et n'est pas stocké.

## Fichiers

| Fichier | Changement |
|---|---|
| `webapp/lib/chat/thread-store.ts` | **nouveau** — persistance localStorage (caps, purge, sanitisation) |
| `webapp/components/chat/ChatProvider.tsx` | état par fil, hydratation/persistance, `recentThreads` |
| `webapp/components/app/AppChatSidebar.tsx` | liste « Discussions récentes » + prop `onSelectCombo` |
| `webapp/components/app/AppWorkspace.tsx` | câblage `onSelectCombo` (desktop) |
| `webapp/components/app/MobileWorkspace.tsx` | câblage `onSelectCombo` (mobile, reste sur l'onglet Chat) |
| `webapp/components/landing/HeroChatPreview.tsx` | fix ordre `openFor` → `appendExchange` |
| `webapp/lib/chat/__tests__/thread-store.test.ts` | **nouveau** — 9 tests |
| `webapp/components/chat/__tests__/ChatProvider.test.tsx` | +4 tests scoping/persistance |

## Vérifications

- `npx tsc --noEmit` : **0 erreur**.
- `npx vitest run` : **405/405 verts** (44 fichiers) — dont les nouveaux :
  - changer de TF puis revenir sur XAU H1 restaure sa conversation ;
  - les fils ne se mélangent pas entre combos ;
  - un provider fraîchement monté (simule un refresh) réhydrate depuis
    localStorage sans serveur ;
  - `resetTurns` ne purge que le fil actif (mémoire + stockage) ;
  - les fils signaux (non-`app:*`) ne sont jamais persistés ;
  - thread-store : round-trip, storage corrompu, périmètre, caps tours/fils,
    budget de taille (drop-oldest et halving).
- `npm run build` : OK.

## Vérification live à faire (avant merge)

1. Sur /app : discuter sur XAU H1 → passer sur XAU H4 → revenir sur H1 → la
   conversation revient, H4 a la sienne.
2. F5 → la conversation du combo actif est restaurée.
3. Bouton « Discussions » → liste des fils récents, clic = bascule de combo.
4. « Réinitialiser » ne touche que le fil affiché.
