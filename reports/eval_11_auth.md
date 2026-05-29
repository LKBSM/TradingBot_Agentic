# Eval 11 — Auth & Tier Manager (SaaS gating)

**Date** : 2026-04-25
**Périmètre** : `src/api/auth.py` (344 l), `src/api/tier_manager.py` (338 l), `src/api/dependencies.py` (32 l).
**Verdict global** : **4.5/10** — la mécanique technique (clés hashées, HMAC admin) est correcte, mais **trois failles business-critiques** : (1) `TESTING_MODE=1` est le défaut, (2) le rate-limit par tier n'est jamais enforcé, (3) aucune intégration paiement → pas de SaaS opérationnel.

---

## 1. Cartographie : flux de gating

```
Request → [require_api_key dependency]
            │
            ├── if TESTING_MODE=1 → return INSTITUTIONAL fake subscriber (BYPASS TOTAL)
            │
            ├── KeyStore.verify_key(raw_key)  ─→ SHA-256 hash + SQL lookup
            │     │
            │     └── KeyStore.check_rate_limit(key_id)   100/min hardcoded
            │
            ├── KeyStore.record_usage(key_id, endpoint)
            │
            └── UserTierManager.get_user_by_api_key(key_id) → enrich {tier, telegram_chat_id}
                  │
                  └── UserTierManager.check_rate_limit(user_id)  ❌ JAMAIS APPELÉ
```

**Problème structurel** : il existe **3 couches de rate-limit** indépendantes :
1. Middleware IP-based (`security.RateLimiter`, 100/min/IP) — appliqué dans `app.py`
2. KeyStore (`api_usage` table, 100/min/clé hardcoded) — appliqué dans `require_api_key`
3. UserTierManager (`usage_log` table, daily limits par tier) — **dead code**, jamais invoqué

Conséquence : un FREE-tier ($0, censément 10 calls/day) peut faire 100/min × 60 × 24 = 144 000 calls/jour s'il n'a pas IP rate-limit (cf. eval 10 § 5).

---

## 2. KeyStore — audit ligne par ligne

| Aspect | Statut | Ligne | Note |
|---|---|---|---|
| Génération clé | ✅ | `auth.py:108` | `secrets.token_hex(32)` = 256 bits, préfixe `sk_` |
| Stockage | ✅ | `auth.py:104,118` | SHA-256 hex avant insert ; raw key montrée 1× |
| Lookup | ⚠️ | `auth.py:142-149` | `WHERE key_hash = ?` — le storage SHA-256 + lookup unique masque les attaques timing pratiques, mais voir §2.1 |
| Soft delete | ✅ | `auth.py:162-173` | `is_active=0` ; pas de hard delete = audit conservé |
| Listing | ✅ | `auth.py:175-194` | Métadonnées seules, pas de hash exposé |
| Usage tracking | ⚠️ | `auth.py:196-207` | Insert SQL **par requête** = contention WAL writer sous charge |
| Rate limit | ⚠️ | `auth.py:231-246` | 100/min hardcoded ; pas paramétrable per-key |
| Schema migration | ✅ | `auth.py:60-99` | versioned, additive, idempotent (`CREATE TABLE IF NOT EXISTS`) |
| Index | ✅ partiel | `auth.py:92-93` | `idx_usage_key_ts` ; mais `api_keys.key_hash` UNIQUE (déjà indexé via UNIQUE constraint) |

### 2.1 Timing-attack analysis

`hashlib.sha256(raw).hexdigest()` est constant-time. La comparaison se fait dans SQLite `WHERE key_hash = ?` — l'engine peut court-circuiter sur miss (early-out par bucket). Comme l'espace `sk_` + 64 hex est 2^256, pratiquement non-bruteforçable, et la lookup leak (existe / n'existe pas) ne suffit pas à inverser. **Risque effectif : faible**, mais formaliser avec `hmac.compare_digest` après fetch reste best-practice.

`HMACManager` (admin) utilise `hmac.compare_digest` ✅ (`src/security/hmac_manager.py:329,341`).

---

## 3. UserTierManager — audit

| Aspect | Statut | Ligne | Note |
|---|---|---|---|
| TIER_CONFIG | ✅ | `tier_manager.py:35-68` | 4 tiers, prix, quota, narrative_depth, telegram, webhooks, chat |
| `users` table | ⚠️ | `tier_manager.py:124-135` | `api_key_id INTEGER` **pas UNIQUE** → même clé liable à plusieurs users (account hijack vector) |
| `subscription_expires` colonne | ⚠️ | `tier_manager.py:132` | Existe mais **jamais lue** : pas de check expiration dans `require_api_key` ; un abonné qui ne paye plus reste en STRATEGIST |
| `link_telegram` | ✅ | `tier_manager.py:236-248` | OK |
| `update_tier` | ✅ | `tier_manager.py:208-220` | OK, mais déclenché manuellement (pas Stripe webhook) |
| `check_rate_limit(user_id)` | 🔴 | `tier_manager.py:254-280` | Bien implémenté, mais **non câblé** — `auth.py:296` ne l'appelle jamais |
| `record_usage` | ⚠️ | `tier_manager.py:282-293` | Encore une 2e table d'usage (`usage_log`) — duplication avec `api_usage` |
| `get_user_by_api_key` | ⚠️ | `tier_manager.py:195-206` | OK ; mais s'il retourne None → fallback `tier="FREE"` silencieux dans `auth.py:302` |

### 3.1 Tarification déclarée vs enforcement

| Tier | Prix | Quota déclaré | Enforcement effectif |
|---|---|---|---|
| FREE | $0 | 10 calls/day | **❌ aucun** (KeyStore 100/min override l'effet) |
| ANALYST | $49 | 100 calls/day | **❌ aucun** (KeyStore 100/min plafond, mais c'est `/min` pas `/jour`) |
| STRATEGIST | $99 | 500 calls/day | **❌ aucun** |
| INSTITUTIONAL | $149 | 2000 calls/day | **❌ aucun** |

**Diagnostic business** : la grille de pricing **n'a aucun lien** avec le code de gating en place. Un FREE peut consommer 144 000 calls/jour (100/min × 1440 min) — **14 400× le quota déclaré**. Si ça part en prod, marge brute négative immédiate.

---

## 4. TESTING_MODE — analyse de risque

```python
# src/api/auth.py:22
TESTING_MODE = os.environ.get("SENTINEL_TESTING_MODE", "1") == "1"
```

**Problèmes** :
1. **Default = "1"** → tout déploiement sans config explicite est ouvert.
2. Pas de **garde CI/CD** : aucun test ne vérifie que `SENTINEL_TESTING_MODE=0` en prod.
3. Bypass donne `tier=INSTITUTIONAL` directement → toutes features unlocked, y compris `/narratives/chat` ($$ LLM).
4. Le seul indicateur en prod : `/health` expose `testing_mode: bool` mais aucune alerte.

**Mitigation manquante** :
- Aucun log d'avertissement de niveau ERROR si TESTING_MODE actif.
- Aucun startup banner type `⚠️ TESTING MODE — DO NOT USE IN PRODUCTION`.
- Aucun check d'env `ENVIRONMENT=prod` pour forcer désactivation.

---

## 5. Admin HMAC

```python
# auth.py:309-344
async def require_admin(...):
    # Replay protection 5 min ✅
    if abs(time.time() - ts) > 300: ...
    # HMAC verify ✅ (constant-time via hmac_manager.verify)
    is_valid = hmac_manager.verify(data, x_admin_signature)
```

**Forces** :
- Replay protection 5 min ✅
- `hmac.compare_digest` dans `HMACManager` ✅
- Headers `X-Admin-Signature` + `X-Admin-Timestamp` séparés (pas de bundling format ad-hoc).

**Faiblesses** :
- Le HMAC signe **uniquement** `x_admin_timestamp` (`auth.py:338`), **pas le body** ni le path. Un signature valide pour `/admin/keys POST {"label":"x"}` peut être rejouée pour `/admin/keys POST {"label":"prod-stripe-secret"}` dans la même fenêtre 5 min. **Privilege escalation possible** si attaquant intercepte une signature.
- Aucune **rotation** automatique de la clé HMAC ; pas de versioning (`kid`) dans la signature.
- Aucun **audit log** des appels admin (cf. eval 10 § 3).

---

## 6. Stockage & sécurité

| Élément | Statut | Détail |
|---|---|---|
| Hash mot de passe | n/a | API-key only (pas de password) ✅ design propre |
| Clés en clair en DB | ✅ | jamais stockées ; SHA-256 hash |
| Clés en clair en logs | ⚠️ | `_call_api`/auth ne log pas la clé, mais aucun masking explicite. Un `logger.debug("headers=%s", request.headers)` ailleurs leakerait. |
| Rotation clés | ⚠️ partiel | `revoke_key` + `create_key` permettent rotation manuelle ; **pas de TTL automatique** ni de "last 5 keys grace period" |
| Vault | ❌ | Aucune intégration Doppler / AWS Secrets Manager ; tout via `.env` |
| RBAC | ❌ | Pas de rôles intermédiaires (admin, support, billing). Binaire `subscriber / admin`. |
| OAuth / SSO | ❌ | Pas d'OAuth2 / SAML pour INSTITUTIONAL (souvent prerequisite enterprise) |
| Audit log | ❌ | Cf. § 5 |

---

## 7. Stripe / Paddle integration

**État** : aucun. La table `users` a `stripe_customer_id` (column NULL by default), mais :
- Aucun webhook handler `/webhooks/stripe`.
- `update_tier()` doit être appelé manuellement via SQL ou script.
- Pas de gestion `subscription_expires` (champ existe, jamais lu).
- Pas de proration / grace period.
- Pas de tracking `last_payment_at`, `next_billing_at`.

**Conséquence business** : pour vendre 1 abonnement, il faut :
1. Stripe Checkout manuel (hors-app).
2. Récupérer `stripe_customer_id` à la main.
3. Lancer `tier_manager.update_tier(user_id, UserTier.ANALYST)`.
4. Créer une clé via `KeyStore.create_key()`.
5. Linker la clé via `tier_manager.link_api_key(user_id, key_id)`.
6. Envoyer la clé à l'utilisateur par email manuel.

**Effort onboarding** : ~10 minutes par client. Non scalable au-delà de 20 abonnés.

---

## 8. Tests existants

```bash
$ ls tests/test_auth*.py
tests/test_auth.py
```

À auditer en eval_17 : couverture, mais d'après mémoire `MEMORY.md` : "Auth tests patch TESTING_MODE=False for proper auth verification" → suggère que les tests ne valident pas le comportement par défaut (TESTING_MODE=1) qui est exactement le mode déployé.

---

## 9. Top 5 améliorations priorisées

| # | Amélioration | Effort | Impact |
|---|---|---|---|
| **R1** | **Désactiver TESTING_MODE par défaut** + bannière ERROR au boot si actif + assert `ENVIRONMENT != prod` | 30 min | 🔴 critique : empêche déploiement accidentel ouvert |
| **R2** | **Câbler tier rate-limit** : appeler `tier_manager.check_rate_limit(user_id)` dans `require_api_key` après lookup tier ; remplacer le 100/min hardcoded par `TIER_CONFIG[tier]["api_calls_per_day"]` (rolling 24h) | 1 jour | 🔴 critique : protège la marge brute (LLM cost) |
| **R3** | **Stripe integration** : webhook `/webhooks/stripe` (event `customer.subscription.{created,updated,deleted}`) + auto-create user + auto-create key + email key | 5 jours | 🟠 haute : débloque growth (auto-onboarding) |
| **R4** | **Subscription expiry enforcement** : check `subscription_expires` dans `require_api_key` ; downgrade auto vers FREE si expiré | 0.5 jour | 🟠 haute : empêche service offert gratuitement |
| **R5** | **HMAC sign full request** : signer `method + path + body + timestamp` (pas juste timestamp) ; rotation clé HMAC versionnée (`kid`) | 1 jour | 🟡 moyenne : empêche replay sur autre route admin |

**Matrice** :

```
Impact ↑
  5 |  R1   R2
  4 |        R3
  3 |  R4   R5
  2 |
    +-------------------→ Effort
       1   2   3   4   5
```

---

## 10. Plan d'exécution

### Quick wins (< 1 jour)
- **QW1** Bannière ERROR + `logger.critical` startup si TESTING_MODE=1 (15 min)
- **QW2** Default TESTING_MODE → "0" ; doc mise à jour (15 min)
- **QW3** Câbler `tier_manager.check_rate_limit` dans `require_api_key` (45 min)
- **QW4** Check `subscription_expires` dans `require_api_key` (30 min)
- **QW5** Forcer `api_key_id UNIQUE` sur `users` table + migration script (30 min)
- **QW6** Audit log table `admin_audit(actor, action, target_id, ip, ts, payload_hash)` + insert dans chaque route admin (1 h)

### Moyen terme (< 1 semaine)
- **MT1** Stripe Checkout + webhook handler (3 jours)
- **MT2** HMAC sign full request (canonical: `METHOD\nPATH\nBODY_SHA256\nTS`) (1 jour)
- **MT3** Email transactionnel (Resend/SES) pour delivery clé (0.5 jour)
- **MT4** Trial 14 j FREE → ANALYST automatique avec downgrade après expiry (0.5 jour)
- **MT5** Test suite : pytest paramétré sur TESTING_MODE ∈ {0,1} (0.5 jour)
- **MT6** Logs structurés masquage `X-API-Key` automatique dans middleware (0.5 jour)

### Long terme (> 1 semaine)
- **LT1** OAuth2 / SSO (Auth0 ou Keycloak self-hosted) pour INSTITUTIONAL
- **LT2** RBAC avec rôles (admin, support, billing, viewer) sur dashboard interne
- **LT3** Vault integration (Doppler ou AWS SM) avec rotation auto 90 j
- **LT4** Quota burst allowance (token-bucket) en plus du daily limit
- **LT5** Self-service dashboard utilisateur (régénérer clé, voir usage temps-réel, billing portal Stripe)

---

## 11. KPIs mesurables post-amélioration

| KPI | Baseline | 30 j | 90 j |
|---|---|---|---|
| % déploiements avec TESTING_MODE=1 | 100 % (default) | 0 % | 0 % |
| Tier rate-limit enforced | non | oui | oui |
| Onboarding manuel par abonné | ~10 min | < 1 min | 0 (self-service) |
| Délai upgrade tier après paiement | manual (heures à jours) | < 30 s (webhook) | < 30 s |
| Subscription expiry check | non | oui | oui |
| Audit log admin actions | 0 % | 100 % | 100 % |
| HMAC signe body | non | oui | oui |
| Stripe MRR tracking | non | oui | oui (avec churn analytics) |
| OAuth/SSO INSTITUTIONAL | non | non | oui |
| Test coverage auth.py | ? | ≥ 90 % | ≥ 95 % |

---

## 12. Trade-offs assumés

- **R1** désactiver TESTING_MODE par défaut casse le développement local sans config explicite → mitiger via `.env.example` clair + doc `make dev`.
- **R2** tier rate-limit ajoute 1 SQL call par request authentifiée → mitigation : compteur en RAM (Redis ou local cache) avec flush async.
- **R3** Stripe ajoute 1 dépendance majeure + complexité webhook signature → ROI évident (sans paiement automatisé, pas de SaaS).
- Daily quota = rolling 24h vs calendar day : rolling plus juste mais query plus coûteux ; calendar = reset minuit UTC, simple.

---

## 13. Note finale par dimension

| Dimension | Note /10 | Justification |
|---|---|---|
| Sécurité crypto | 7 | SHA-256 + token_hex(32) ✅ ; HMAC compare_digest ✅ ; mais HMAC signe que TS, pas body |
| Default-secure | 2 | TESTING_MODE=1 par défaut = catastrophe potentielle |
| Tier enforcement | 1 | Code existe, jamais branché ; quotas affichés sur landing 100 % menteurs si on lance |
| Billing automation | 0 | Aucune intégration Stripe/Paddle |
| Audit / non-repudiation | 2 | Pas de log admin actions, pas de versioning HMAC |
| Code quality | 8 | Patterns SQLite WAL propres, schema versioning ✅, threading correct |
| Testabilité | 6 | Tests existent mais couvrent surtout TESTING_MODE=0, pas le default |
| Observabilité | 4 | usage_log existe ; pas de dashboard, pas de Stripe MRR |
| **Global** | **4.5/10** | **Crypto OK ; business model non opérationnel** |

---

## 14. Verdict

**Garder** : la séparation KeyStore / UserTierManager (clean), HMAC admin (corriger §5), schéma SQLite versionné.

**Refondre** : `require_api_key` pour câbler tier check + expiry check.

**Ajouter** (du jour 1 si on lance commercialement) :
- Stripe webhook handler.
- Audit log admin.
- TESTING_MODE désactivé par défaut + alerte.
- Subscription expiry enforcement.

**Sans R1+R2+R3+R4** : le SaaS ne peut pas générer de revenu durable — c'est uniquement un outil interne.

---

## Annexe — fichiers critiques

- `src/api/auth.py:22` — TESTING_MODE default
- `src/api/auth.py:288` — KeyStore rate limit (100/min hardcoded)
- `src/api/auth.py:294-302` — tier enrichment ; **insertion R2 ici**
- `src/api/auth.py:338` — HMAC signe seulement timestamp
- `src/api/tier_manager.py:35-68` — TIER_CONFIG
- `src/api/tier_manager.py:254-280` — check_rate_limit (dead code)
- `src/api/tier_manager.py:128-129` — `api_key_id` non UNIQUE
- `src/api/tier_manager.py:132` — `subscription_expires` jamais lu
