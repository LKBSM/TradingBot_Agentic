# Gate d'accès par abonnement — freemium (gratuit / payant / owner)

_Mission ③ · 2026-06-22 · branche `feat/subscription-gate` (worktree dédié, depuis
`main` consolidé `501e397`, après merge des missions ① auth/légal et ② Stripe)._

> **Périmètre strict** : décider **qui accède à quoi** selon l'état d'abonnement,
> branché sur l'accroche unique laissée en ①. **Contrôle d'accès et affichage
> uniquement — la détection est INCHANGÉE** (on filtre l'accès, jamais le calcul).

---

## 1. Décision produit (validée par l'utilisateur)

**Modèle freemium** — un palier gratuit limité qui donne un vrai avant-goût,
payant pour le reste :

| Fonction / Marché | Visiteur | **Gratuit (Découverte)** | Abonné payant | Owner |
|---|---|---|---|---|
| Landing + samples statiques · pages légales/compte | ✅ | ✅ | ✅ | ✅ |
| `/app` — lecture live | ❌ → login | ✅ **XAU/USD M15 uniquement** | ✅ XAU+EUR · M15/H1/H4 | ✅ tout |
| Lecture narrée complète (structure/régime/événements) | ❌ | ✅ **complète sur XAU M15** | ✅ partout | ✅ |
| Chart / candles · live-price | ❌ | ✅ (XAU M15) | ✅ partout | ✅ |
| Chat M.I.A | ❌ | ⚠️ **5 msg/jour** | ✅ illimité | ✅ |
| Scanner multi-marchés (conditions-scan) | ❌ | ❌ → upsell | ✅ complet | ✅ |
| Autres marchés / timeframes | ❌ | ❌ → upsell | ✅ | ✅ |

Paliers ordonnés : `VISITOR < FREE < SUBSCRIBER < OWNER`. **Owner = tout,
toujours** (l'opérateur ne peut jamais se verrouiller dehors).

### Posture par défaut : câblé mais OFF
Cohérent avec `SENTINEL_TESTING_MODE` et le seam ① : tant que
`SUBSCRIPTION_GATE_ENFORCED=0` (défaut, phase de test perso) **les routes-features
restent totalement ouvertes** (visiteur inclus) et tous les garde-fous sont des
no-op → rien ne casse avant le lancement et les tests existants ne bougent pas.
Passer `SUBSCRIPTION_GATE_ENFORCED=1` fait mordre le freemium.

### Tout est env-configurable (politique, pas code en dur)
| Var | Défaut | Effet |
|---|---|---|
| `SUBSCRIPTION_GATE_ENFORCED` | `0` | `1` active le mur (visiteur 401, free limité, payant complet) |
| `FREE_INSTRUMENTS` | `XAUUSD` | marchés du palier gratuit |
| `FREE_TIMEFRAMES` | `M15` | timeframes du palier gratuit |
| `FREE_CHAT_DAILY_LIMIT` | `5` | messages chat/jour gratuits (0 = chat payant) |
| `FREE_SCANNER_ENABLED` | `0` | `1` ouvrirait le scanner au palier gratuit |

---

## 2. Architecture — où vit la décision

```
Requête feature  ──►  route /api/*  ──►  entitlements.enforce_*()
                                          │   (no-op si gate OFF)
                                          ├─ pas authentifié  → 401
                                          ├─ resolve_tier(account, store)
                                          │     owner→OWNER · sub active→SUBSCRIBER · sinon→FREE
                                          └─ périmètre dépassé → 402 (upsell propre)
                                                 ▲
                       subscription_gate (①/②) ──┘  has_active_subscription() lit
                                                    AccountStore.get_subscription()
```

* **`src/api/entitlements.py`** (NOUVEAU) — la politique freemium : enum `Tier`,
  périmètre gratuit env-driven, `resolve_tier`, et les gardes serveur
  `enforce_combo_access` / `enforce_instrument_access` / `enforce_scanner_access`
  / `enforce_chat_access`. **Construit sur** `subscription_gate` (① — seam Stripe)
  sans le dupliquer.
* **`src/api/account_store.py`** — schéma **v2 → v3** : table `chat_usage`
  (`account_id, day, count`, PK composite) + `get_chat_usage` /
  `increment_chat_usage` (upsert atomique sous `RLock`) pour le quota gratuit.
* **`src/api/routes/access.py`** (NOUVEAU) — `GET /api/access/me` : résumé
  tier + entitlements (instruments/timeframes autorisés, scanner, quota chat)
  qui **pilote l'affichage** côté front. Ne lève jamais 401 (renvoie
  `authenticated:false`).

### Routes gated (contrôle **serveur**, non contournable via l'UI)
| Route | Garde |
|---|---|
| `GET /api/market-reading` | `enforce_combo_access` (instrument+TF) |
| `GET /api/candles` | `enforce_combo_access` |
| `GET /api/live-price` | `enforce_instrument_access` |
| `POST /api/conditions-scan` | `enforce_scanner_access` |
| `POST /api/chatbot/message` | `enforce_chat_access` (+ compteur quotidien) |

Le monde B2B legacy `/api/v1/*` (signals, narratives, dashboard) reste sur
`require_api_key` — **hors périmètre** de cette mission.

---

## 3. Frontend — paywall propre + dégradation, jamais d'erreur brute

* **`webapp/lib/access/api-client.ts`** — `fetchAccess()` + `comboAllowed()`.
* **`webapp/lib/access/errors.ts`** — `AccessError` (401/402) + mapping depuis la
  réponse : les clients features (market-reading, candles, conditions, chat)
  surfacent désormais le **détail FR d'invitation à s'abonner** au lieu d'un
  « erreur interne » générique.
* **`webapp/components/access/SubscriptionGate.tsx`** — garde de route pour
  `/app` et `/scanner` :
  * gate OFF → rend les enfants (ouvert) ;
  * gate ON + non authentifié → redirection `/connexion?next=…` ;
  * gate ON + `requireFullAccess` + free → **`<Paywall>`** (upsell) ;
  * échec réseau du résumé → **fail-open** (le mur serveur reste la vérité).
* **`webapp/components/access/Paywall.tsx`** — surface d'upsell propre (CTA
  « Voir les abonnements » → `/abonnement`, lien connexion si visiteur).
* `/app` : garde sans `requireFullAccess` (le free entre, périmètre partiel) ;
  `/scanner` : `requireFullAccess` (fonction payante → paywall pour le free).

---

## 4. Tests — chaque palier voit exactement son périmètre

### Backend — `tests/test_subscription_gate_freemium.py` (16 tests, verts)
* **Gate OFF** : visiteur anonyme passe toutes les routes (503 service non câblé,
  jamais 401/402) ; `/api/access/me` = accès complet.
* **Visiteur (gate ON)** : 401 sur les 5 routes ; access résumé `authenticated:false`.
* **Gratuit (gate ON)** : XAU M15 passe ; autre TF / autre marché / live EUR → 402 ;
  scanner → 402 ; chat 3 tours OK puis 4ᵉ → 402 (quota, `FREE_CHAT_DAILY_LIMIT=3`) ;
  `/api/access/me` = `tier=free`, instruments `["XAUUSD"]`, TF `["M15"]`,
  scanner `false`, chat limit `3`.
* **Abonné (gate ON)** : tous marchés/TF + scanner passent ; chat illimité ;
  résumé `tier=subscriber`, instruments `null`, scanner `true`, chat limit `null`.
* **Owner (gate ON, SANS abonnement)** : bypass total ; résumé `is_owner:true`.
* **Expiration** : `current_period_end` passé → rétrograde **proprement** en FREE
  (XAU M15 ok, EUR → 402, jamais d'erreur brute) ; résumé `tier=free`.

### Frontend — `components/access/__tests__/SubscriptionGate.test.tsx` (10 tests, verts)
Garde : rend les enfants en accès complet · redirige le visiteur vers `/connexion`
· paywall pour un free sur surface payante · laisse entrer le free en surface
partielle · fail-open sur erreur. `comboAllowed` et `accessErrorFromResponse`
(401↔login / 402↔abonnement / autres↔null) couverts.

### Verts
* `pytest tests/test_subscription_gate_freemium.py` → **16 passed**.
* Suites existantes potentiellement impactées
  (`market_reading|candles|live_price|conditions_scan|chatbot|account_billing|bootstrap`)
  → **76 passed, 0 régression** (gate OFF par défaut).
* `npx tsc --noEmit` → **0 erreur**.
* `npx vitest run` (access + lib market-reading/conditions/chat) → **104 passed**.
* `npm run build` → **succès** (`/app` 164 kB, `/scanner` 128 kB).

> **Note pré-existante (non liée)** : 14 tests échouent déjà sur `main` SANS ces
> changements (vérifié par `git stash`) — `test_smoke_e2e` (dépend de
> `SENTINEL_TESTING_MODE` non positionné dans ce shell) et `test_webapp_preview`
> (mockup HTML). Hors périmètre de cette mission.

---

## 5. Garanties de sécurité

* **Non contournable via l'UI** : instrument/TF vérifiés **serveur** (query
  params), quota chat compté **en base** sous verrou — masquer un bouton ne
  débloque rien.
* **Owner inviolable** : `role == 'owner'` → `OWNER` avant toute autre règle.
* **Dégradation propre** : un abonnement expiré tombe en FREE (402 + paywall),
  jamais d'erreur brute ; un détail serveur n'est jamais fuité (messages FR
  d'upsell explicites).
* **Aucune carte** : l'état d'abonnement vient de `AccountStore.get_subscription`
  (écrit par les webhooks Stripe ②), jamais de données de paiement.

---

## 6. Activation (le jour J)

1. `SUBSCRIPTION_GATE_ENFORCED=1`
2. (optionnel) ajuster `FREE_*` selon la grille retenue
3. S'assurer que `STRIPE_PRICE_*` est configuré (mission ②) pour que le paywall
   mène à un Checkout réel.

Tant que l'étape 1 n'est pas faite, le produit reste **entièrement ouvert** pour
la phase de test personnel (owner inclus).
