# VÉRIFICATION — Système de connexion (auth) + durcissement

Branche : `fix/auth-hardening` (depuis `origin/main`). Worktree `wt-perf-2`.
Date : 2026-08-07.

Méthode : cartographie backend/front/tests → suites exécutées → **flux live réel**
(register → login → /me → reset → logout) + inspections base/code.

---

## 1. Verdict : le cœur est sain

Vérifié **live + tests** :

| Aspect | Constat |
|---|---|
| Hachage mdp | **Argon2id** `m=65536,t=3,p=4` (params OWASP), jamais en clair/loggé/renvoyé — vérifié en base (`$argon2id$v=19$…`) |
| Session | Jeton opaque `token_urlsafe(32)`, **révocable**, stocké en SHA-256 ; cookie `HttpOnly` + `SameSite=lax` + **`Secure` par défaut** (`SESSION_COOKIE_SECURE` défaut "1"), TTL 30 j |
| Login | Faux mdp **et** utilisateur inexistant → **même 401** + **timing quasi identique (0,109 vs 0,106 s)** = anti-énumération/timing OK |
| Reset | Token **usage unique, 1 h** ; requête **enumeration-safe** (même message connu/inconnu) |
| /me, logout | 200 avec cookie / 401 sans ; logout **efface le cookie** (`Max-Age=0`) |
| Secrets/SQL | `SESSION_SECRET` **fail-fast en prod** ; requêtes **paramétrées** |
| Front | Aucun token en localStorage, garde anti-double-submit, protection open-redirect `?next=`, gate d'accès timeout 8 s (REC-1) |
| `TESTING_MODE` | **OFF par défaut**, ne bypasse que l'auth par clé API |
| Tests | backend 134 + frontend 33, tous verts |

## 2. Trois défauts trouvés (vérifiés) → corrigés

### A. [Robustesse] Le fetch d'auth n'avait AUCUN timeout
`webapp/lib/auth/api-client.ts` faisait `fetch()` sans `AbortController` → un
`/api/auth/login` qui pend laissait le bouton figé sur « Connexion… »
indéfiniment, sans message (le bug REC-1, jamais corrigé côté formulaire).

**Fix** : `request()` → `attempt()` borné à **8 s** (comme la gate) + **retry une
fois** sur transitoire réseau (jamais sur timeout / HTTP déterministe) ;
`AuthError.reason` (`timeout`/`network`/`http`) → messages **distincts** (« trop de
temps » vs « injoignable »). Le `finally` réactive le bouton. Tests : timeout →
`reason:'timeout'` sans retry ; retry réseau → succès en 2 appels ; 429 → pas de
retry.

### B. [Sécurité] Aucune limitation de débit / verrouillage
**Prouvé live** : 20 logins ratés d'affilée → 20× 401, zéro throttle. Le middleware
per-IP existe mais est un **no-op** dans `asgi:app` (`create_app(rate_limiter=None)`).

**Fix** : `src/api/auth_throttle.py` — throttle glissant en mémoire, par app
(`app.state.auth_throttle`), appliqué à login / register / reset :
- **login** : par IP **et** par identifiant ; **seuls les échecs comptent**, un
  **succès efface** le compteur → un utilisateur légitime qui se trompe n'est
  jamais bloqué (happy path intact).
- **register / reset** : par IP, chaque tentative compte (vecteurs spam/probe).
- Dépassement → **429 + `Retry-After`**. Config `AUTH_THROTTLE_MAX_ATTEMPTS`
  (défaut 10) / `AUTH_THROTTLE_WINDOW_S` (défaut 300 s) ; `max<=0` désactive.

**Vérifié live** (cap=5) : 5 ratés → 401, 6e → **429 `Retry-After: 271`** ; le bon
mot de passe est aussi bloqué (attaque stoppée) ; après un **succès**, le compteur
est **remis à zéro** (démontré : 3 ratés → succès → 3 ratés encore 401, pas 429).

**Honnêteté / limite** : throttle **par processus** — en multi-worker le seuil
effectif = workers × cap ; un store partagé (Redis) serait la couche suivante.
Documenté dans le module.

### C. [Sécurité] Énumération de comptes à l'inscription
Email déjà pris → **409 « Cette adresse e-mail est déjà utilisée. »** révélait
qu'un compte existe.

**Fix** : message de conflit **générique** (« Ces informations ne peuvent pas être
utilisées… ») qui ne dit plus **lequel** de l'email/username est pris. Les erreurs
de validation (mdp faible, email invalide) restent spécifiques (elles ne fuient
rien). **Vérifié live**. **Limite** : l'élimination *complète* de l'énumération
d'email nécessite une inscription **avec vérification par e-mail** (chantier à
part) ; combiné au throttle per-IP (B), le sondage de masse est déjà bloqué.

## 3. Résultats de tests

- Backend : `test_auth_throttle.py` (nouveau, 13) + suites auth (auth/account/
  reliability/beta/tier/subscription/rate_limit) **verts** ; sweep large 488
  passants (2 échecs `test_smoke_e2e` **pré-existants** — scanner v1 503/env,
  confirmés en base stashée).
- Frontend : auth vitest **22** (dont 3 timeout/retry) ; suite complète 867
  passants (3 gardes repo-scan **flaky sous charge parallèle**, verts en isolé).
- `tsc` 0.

## 4. Ce qui reste (hors périmètre de ce lot)
- Inscription **vérifiée par e-mail** (élimination totale de l'énumération).
- Throttle **partagé** multi-instances (Redis) si passage multi-worker/instance.
- 2FA, journal d'événements d'auth dédié.
