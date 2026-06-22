# Squelette compte — auth + comptes + navigation + pages légales + propriétaire

_Mission ① · 2026-06-22 · branche `feat/account-auth-legal-shell` (worktree dédié,
depuis `main` consolidé `08c3e5e`)._

> **Périmètre strict** : authentification, comptes, rôles, navigation, pages
> légales, compte propriétaire. **PAS de paiements** — seul un point d'accroche
> propre est posé pour le futur gate d'abonnement.

---

## 1. Décisions structurantes (validées par l'utilisateur)

| Sujet | Décision | Pourquoi |
|---|---|---|
| Où vit l'auth | **FastAPI** (pas NextAuth/DB Node) | Les comptes y résident déjà (tier_manager, KeyStore) ; le front consomme via le rewrite same-origin `/api/*`. Source de vérité unique. |
| Brique crypto | **argon2-cffi** (mots de passe) + **itsdangerous** (cookies signés) | 100 % du crypto délégué à des libs éprouvées ; aucun hachage maison. Plumbing SQL calqué sur `KeyStore`. |
| Texte CGU | Généré depuis le FR de `routes/legal.py` → `docs/legal/conditions-utilisation.md` | Source canonique versionnée, rendue **tel quel**. |

---

## 2. Livré

### Backend (`src/api/`)
- **`account_store.py`** — `AccountStore` (raw sqlite3 + WAL + RLock + migrations,
  comme `KeyStore`). Tables : `accounts`, `account_consents`, `sessions`,
  `password_resets`. Argon2id (`check_needs_rehash` → upgrade transparent).
  Tokens de session/reset opaques (`secrets.token_urlsafe`), **seul leur SHA-256
  est stocké**. Login par **identifiant OU courriel**. Seed owner idempotent.
- **`session_auth.py`** — cookie `mia_session` `HttpOnly` + `SameSite=Lax` +
  `Secure` (désactivable via `SESSION_COOKIE_SECURE=0` en dev http) ; valeur
  **signée itsdangerous**. Dépendances `optional_account` / `require_account` /
  `require_owner`.
- **`subscription_gate.py`** — **point d'accroche unique** du futur paywall :
  `require_active_subscription` (aujourd'hui pass-through, **owner toujours
  autorisé**) + `account_has_access` avec le seam Stripe documenté. 402 câblé
  mais inactif. **Aucun Stripe.**
- **`routes/accounts.py`** — `POST /api/auth/register|login|logout`,
  `GET /api/auth/me`, `PATCH /api/auth/profile`,
  `POST /api/auth/password-reset/request|confirm`, `GET /api/auth/access`
  (sonde du gate), `GET /api/auth/admin/overview` (**owner-only**, seam futur
  tableau admin). Register exige 18+ **et** les 2 consentements ; version +
  horodatage enregistrés côté serveur (non falsifiables par le client).
- **`routes/legal.py`** — `GET /api/v1/legal/conditions` sert le `.md` **verbatim**
  (+ `X-Document-Version`), `…/conditions/meta`, et `conditions_version` ajouté à
  `…/legal/version`.
- **Câblage** : `AppState.account_store` + instanciation par défaut dans
  `create_app` + `_maybe_seed_owner` au lifespan (idempotent, no-op sans
  `OWNER_*`) + `include_router(accounts.router)`.
- **`requirements.txt`** : `argon2-cffi>=23.1.0`, `itsdangerous>=2.1.0`.

### Frontend (`webapp/`)
- **`lib/auth/`** : `types.ts`, `api-client.ts` (same-origin `/api/auth/*`),
  `store.tsx` (`AuthProvider` + `useAuth`, sonde `/me` au montage).
- **`lib/legal/render-markdown.tsx`** : rendu markdown titré, **sans
  `dangerouslySetInnerHTML`**.
- **Pages** : `/inscription`, `/connexion`, `/mot-de-passe-oublie`, `/compte`
  (profil + consentements + déconnexion), `/conditions` (rend le `.md` tel quel),
  `/confidentialite` (placeholder structuré — mission ④).
- **Composants** : `components/auth/*` (formulaires), `components/legal/ConditionsDocument`.
- **Nav** : `Nav` et `AccountMenu` rendus **conscients de la session**
  (connecté ↔ déconnecté). **Footer** : `/conditions` + `/confidentialite`
  désormais des liens réels (marqueurs LEGAL-PENDING retirés sur ces deux).
- Layout : `AuthProvider` monté autour de l'arbre.

### Compte propriétaire
- Amorcé au 1er démarrage depuis `OWNER_USERNAME` / `OWNER_EMAIL` /
  `OWNER_PASSWORD` (env). **Mot de passe haché à la création**, jamais stocké en
  clair ni en dur. Rôle `owner` = accès complet + bypass du futur gate. Re-seed
  idempotent ; rotation de `OWNER_PASSWORD` **n'écrase pas** un mot de passe
  changé en application.

---

## 3. Sécurité

- Aucun secret commité. `.env` non modifié (gitignoré l.71) ; `.env.example`
  fournit `SESSION_SECRET`, `SESSION_COOKIE_SECURE`, `ACCOUNTS_DB_PATH`,
  `OWNER_USERNAME/EMAIL/PASSWORD` avec **valeurs factices**.
- Mots de passe Argon2id ; tokens hachés au repos ; cookies `HttpOnly` +
  `SameSite=Lax` (mitige le CSRF sur les POST) + signés.
- Messages d'erreur neutres (login générique, reset anti-énumération).
- Reset : token usage unique + révocation de **toutes** les sessions au
  changement de mot de passe.
- `data/*.db` gitignoré — aucune base committée.

## 4. Tests (verts)

- **Backend** : `tests/test_account_auth.py` — **25 tests** (register/login/logout,
  consentement version+horodatage, refus <18 / sans consentement, hachage,
  login id-ou-email, sessions, reset usage-unique, **owner env → accès complet,
  compte normal → 403** sur la surface owner, rendu légal). `pytest -k
  "legal or account or app …"` → **50 passed, 0 régression**.
- **Frontend** : `lib/auth/__tests__/api-client.test.ts` (11) + `Nav.test.tsx`
  mis à jour. **Suite complète : 257 passed (33 fichiers)**.
- **`tsc --noEmit`** : vert. **`next build`** : vert (6 nouvelles routes).

### Hors de mon fait (pré-existant)
- 2 échecs `test_smoke_e2e.py` (`/api/v1/scanner/status` → 503 sans scanner
  injecté) — confirmés en remisant tous mes changements (`git stash`). Sans lien
  avec l'auth.

## 5. Accroche paiements (rappel)

Un point unique : `subscription_gate.account_has_access()`. Mission ② y branchera
Stripe en remplaçant `return True`, sans toucher aux routes. **Owner toujours
autorisé.**

## 6. Suites possibles
- Mission ④ : texte définitif `/confidentialite` + `/mentions-legales` +
  `/cookies` (toujours LEGAL-PENDING).
- Livraison e-mail du token de reset (`_dispatch_reset_token`, aujourd'hui log
  sans le secret).
