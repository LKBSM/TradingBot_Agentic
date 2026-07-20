# Fiabilité du login au 1er essai — cause racine & correctif

**Date :** 2026-07-20
**Branche :** `fix/auth-login-reliability` (worktree dédié, depuis `origin/main` @ `11df165`)
**Périmètre :** auth/session uniquement — moteur, IA, autres écrans inchangés.
**Sécurité :** aucune vérification affaiblie. Bibliothèques d'auth éprouvées intactes
(argon2-cffi, itsdangerous), mot de passe haché, secret via env, cookie `HttpOnly`.

---

## 1. Symptôme

Sur un appareil **sans cookie préexistant**, la connexion échoue au **premier**
essai avec de bons identifiants ; en recliquant et/ou en rafraîchissant, elle
finit par passer. Intermittent → suspicion timing / état / cookie, pas identifiants.

## 2. Architecture réelle (pré-requis au diagnostic)

Le navigateur ne parle **jamais** au backend FastAPI en cross-origin. Tout transite
par le rewrite **same-origin** de Next (`webapp/next.config.js:107-117`) :

```
navigateur → (Next, même origine) /api/*  ──rewrite serveur──▶  FastAPI :8000
```

Le cookie `mia_session` est donc posé pour l'origine **Next**. Conséquence : le
CORS backend n'entre **jamais** en jeu côté navigateur. L'hypothèse « `allow_credentials=True`
manquant sur le CORS FastAPI » (envisagée) est donc **écartée** — c'est du proxy
serveur-à-serveur, pas un fetch cross-origin du navigateur.

## 3. Ce qui est SAIN (vérifié dans le code)

| Point | Preuve | Verdict |
|---|---|---|
| Écriture de session avant réponse | `accounts.py:206-207` → `create_session` (SQLite autocommit + WAL) puis `set_session_cookie` | ✅ durable avant `Set-Cookie` |
| Attributs cookie | `session_auth.py:137-146` : `HttpOnly`, `SameSite=Lax`, `Secure` (défaut ON), `Path=/`, host-only, `Max-Age=30j` | ✅ corrects |
| Validation cookie | `session_auth.py:110-120` (itsdangerous `max_age=30j`) + `account_store.py:550-569` (`expires_at > now`) | ✅ pas de fenêtre serrée (mono-instance) |
| Envoi du cookie (client auth) | `lib/auth/api-client.ts:41` `credentials: 'same-origin'` | ✅ explicite |
| Commit `Set-Cookie` avant résolution du fetch | comportement navigateur standard | ✅ pas de race « cookie pas encore posé » |

→ **Backend et cookies corrects.** Le bug est côté **navigation client**.

## 4. Cause racine (la plus probable)

**Redirection post-login optimiste servie depuis un Router Cache Next amorcé en
état « déconnecté ».**

Séquence fautive, lockdown **ON** (`NEXT_PUBLIC_BETA_LOCKDOWN=1`, confirmé) :

1. Visite sans cookie d'une route protégée → l'**edge middleware**
   (`webapp/middleware.ts:65-73`) renvoie un **307 → /connexion**. Ce résultat
   (redirection) est **mémorisé dans le Router Cache client** de Next.
2. L'utilisateur se connecte : `login()` réussit, le cookie `mia_session` est posé.
3. `LoginForm` appelait alors **immédiatement** `router.push(dest)` (nav *soft*
   `LoginForm.tsx:52`). Cette navigation peut être **satisfaite depuis l'entrée
   cachée « → /connexion »**, renvoyant l'utilisateur fraîchement authentifié
   **au login** — au 1er essai.
4. Un **refresh** = navigation *dure* = **vide le Router Cache** → la route est
   ré-évaluée à neuf, cookie présent → ça passe. **C'est exactement le symptôme.**

Double source de décision aggravante : l'edge (présence du cookie) **et** le client
`SubscriptionGate` (`/api/access/me`, `SubscriptionGate.tsx:55`) peuvent diverger
juste après login, sans invalidation de cache entre les deux.

## 5. Correctif

### 5.1 Invalider le Router Cache AVANT de naviguer (le fix central)
`webapp/components/auth/LoginForm.tsx` — après `login()` :

```ts
const dest = resolveDestination();
router.refresh();      // vide le Router Cache (dont la redirection pré-login)
router.replace(dest);  // navigue vers une entrée forcément re-fetchée, cookie présent
```

- `router.refresh()` **avant** la navigation : un cache de *redirection* périmé ne
  se corrige QUE par une invalidation préalable. L'idiome courant `push(); refresh()`
  ne corrige que des *données* périmées sur une route déjà correcte, **pas une
  redirection périmée** — d'où l'ordre inversé ici.
- `router.replace` (au lieu de `push`) : `/connexion` ne reste pas dans l'historique.

Même correctif appliqué à `RegisterForm.tsx` (l'inscription ouvre aussi une session).

### 5.1bis Symétrie au logout (renforcement 2026-07-20)
Le **problème inverse** existait au logout : `AccountMenu.tsx` et `AccountPanel.tsx`
faisaient `await logout(); router.push('/')` **sans** invalider le Router Cache. Une
fois le cookie effacé, des entrées RSC **authentifiées** en cache (routes protégées
`/app`, `/compte`, `/abonnement`, `/zones`, `/scanner`) pouvaient encore être servies
à un utilisateur désormais déconnecté. Correctif symétrique :

```ts
await logout();
router.refresh();   // vide les entrées authentifiées cachées
router.push(lh('/'));
```

### 5.2 Durcissement de la sonde d'accès
`webapp/lib/access/api-client.ts` — `fetchAccess` pose désormais
`credentials: 'same-origin'` explicitement (parité avec le client auth ; latent
si l'API est un jour proxifiée). Inoffensif aujourd'hui (défaut same-origin).

### Ce qui n'a PAS changé
- Aucune vérification d'auth contournée. Mauvais identifiants → toujours 401 propre.
- Attributs cookie inchangés, moteur/IA inchangés, backend inchangé.

## 6. Tests

**Nouveaux** — `webapp/components/auth/__tests__/LoginForm.test.tsx` (4) :
- ✅ succès → `router.refresh()` appelé **avant** `router.replace('/app')`, `push`
  jamais appelé, identifiant *trimé* ;
- ✅ `?next=` interne sûr honoré (`/scanner`) ;
- ✅ `?next=//evil.com` hors-site ignoré → fallback `/app` (garde open-redirect AUTH-06) ;
- ✅ mauvais identifiants → message d'erreur, **aucune** navigation ni refresh.

**Nouveaux** — `webapp/lib/access/__tests__/api-client.test.ts` (2) :
- ✅ `fetchAccess` envoie `credentials: 'same-origin'` ;
- ✅ réponse non-OK → throw.

**Nouveaux (renforcement logout)** :
- `webapp/components/app/__tests__/AccountMenu.test.tsx` (1) : ✅ logout →
  `router.refresh()` appelé **avant** `router.push('/')` ;
- `webapp/components/auth/__tests__/AccountPanel.test.tsx` (1) : ✅ idem +
  aucun redirect logged-out parasite sur un panneau authentifié.

**Résultats :**
- `tsc --noEmit` : **0 erreur**.
- `vitest run` : **514/514** (512 initiaux + 2 tests logout). Les lignes stderr
  `useActiveCombo must be used inside…` proviennent d'un test d'error-boundary qui
  **passe** (assertion de throw), pas d'un échec.
- `npm run build` : **vert**.
- Backend : **aucun fichier Python modifié** → suite pytest inchangée.

Couverture des scénarios demandés :
| Scénario | Couvert par |
|---|---|
| Bons identifiants, sans cookie → succès 1er essai | fix 5.1 + test refresh-avant-replace + **validation live requise** |
| Mauvais identifiants → échec propre, pas de session | test « bad credentials » + backend `accounts.py:200-204` (inchangé) |
| Nav directe route protégée sans session → login | `middleware.ts` + `SubscriptionGate` (inchangés) |
| Accès immédiat post-login sans rejet transitoire | fix 5.1 (invalidation cache) |
| Persistance au refresh / logout invalide la session | cookie 30j + `clear_session_cookie` (inchangés) + fix 5.1bis (cache authentifié purgé au logout) + tests AccountMenu/AccountPanel |

## 7. Réserves pour la validation live (appareil neuf, sans cookie)

- **`Secure` cookie** (`SESSION_COOKIE_SECURE=1` par défaut) : accepté sur
  `localhost` (contexte sécurisé) et en HTTPS prod, mais **rejeté sur HTTP en clair
  via IP LAN** (ex. `http://192.168.x.x:3001`). Ce cas donnerait un échec
  *permanent* (pas intermittent) — donc ce n'est pas la cause racine, mais tester
  via `localhost`/HTTPS, ou poser `SESSION_COOKIE_SECURE=0` pour un test HTTP LAN.
- Si une flakiness résiduelle subsistait au 1er essai malgré `replace + refresh`,
  le repli garanti est une navigation dure (`window.location.assign(dest)`) —
  écarté ici par choix (préserver l'expérience SPA).

**Merge sur `main` uniquement après validation live sur appareil neuf sans cookie.**
