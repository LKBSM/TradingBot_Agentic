# Sprint Plan — Dual B2C + B2B Architecture

**Période** : 4 semaines, ~8-9h/semaine, total **34h**.
**Owner** : `solo` (Loukmane Bessam) sauf indication.
**Convention DoD** : (1) tests unitaires verts, (2) critère mesurable observable, (3) doc inline minimum, (4) zéro régression sur la suite existante (1366+ tests).
**Référence** : `reports/architecture/dual_b2c_b2b_design.md`.

---

## Sprint 1 — Contrat canonique + builder (semaine 1, 8h)

> Objectif : produire un `InsightSignal` validé Pydantic v2 à chaque tick scanner, sans rien casser de l'existant.

### S1-T1 — Schéma `InsightSignal` Pydantic v2
- **Effort** : 2h
- **Owner** : solo
- **Description** : Implémenter `src/models/insight_signal.py` (cf. arch doc §3). Tous les enums, sub-schemas (`KeyLevels`, `VolatilityForecast`, `Scenario`, `ComponentBreakdown`, `ComplianceContext`), et le BaseModel `InsightSignal` avec ses 3 validators (`_validate_score_label`, `_validate_expiry`, `_validate_invalidation`).
- **DoD** :
  - 15+ tests unitaires `tests/test_insight_signal_schema.py` couvrant : validators (cas pass + fail), sérialisation JSON round-trip, `version_schema` regex, enums coverage.
  - `model_dump_json()` produit du JSON stable (champs nullables présents avec valeur `null`, pas omis).
  - Coverage > 95% sur le module.
- **Dépendances** : aucune.

### S1-T2 — `InsightSignalBuilder`
- **Effort** : 2h
- **Owner** : solo
- **Description** : Implémenter `src/distribution/insight_builder.py`. Classe `InsightSignalBuilder` avec méthode `build(signal: ConfluenceSignal, narrative: SignalNarrative | dict, state_snapshot, lang: str) -> InsightSignal`. Mappe les champs des dataclasses existantes vers le contrat canonique. Calcule `expires_at` selon la table TTL (arch doc §3). Injecte `compliance` via `disclaimers.get_disclaimer(lang)`.
- **DoD** :
  - 8+ tests unitaires couvrant : mapping LONG/SHORT → `bullish/bearish`, score → label thresholds (39, 40, 54, 55, 69, 70), TTL par tier, narrative manquant → fallback template, lang inconnu → fallback `en`.
  - Test E2E avec un `ConfluenceSignal` réel d'un fixture XAU M15.
- **Dépendances** : S1-T1.

### S1-T3 — Intégration au scanner
- **Effort** : 2h
- **Owner** : solo
- **Description** : Modifier `src/intelligence/sentinel_scanner.py:661` (`_publish_signal`). Appeler `InsightSignalBuilder.build()` après `_generate_narrative_safe`. Stocker le `InsightSignal` dans une nouvelle table `insights` (SQLite) en parallèle du `signal_store` existant. **Aucun changement de l'API publique du scanner**.
- **DoD** :
  - Schema `insights` table avec WAL, indices `(asset, generated_at)` et `(insight_id UNIQUE)`.
  - 5+ tests d'intégration : un tick produit 1 `ConfluenceSignal` + 1 `InsightSignal` ; les deux sont persistés ; le legacy `signal_store` reste alimenté ; aucune régression sur `test_sentinel_scanner.py`.
- **Dépendances** : S1-T2.

### S1-T4 — Migration SQL `account_type`
- **Effort** : 1h
- **Owner** : solo
- **Description** : Ajouter colonne `account_type ENUM('B2C','B2B') NOT NULL DEFAULT 'B2C'` à la table `users` (`src/api/tier_manager.py`). Bump `SCHEMA_VERSION` à 2. Migration idempotente.
- **DoD** :
  - Migration fonctionne sur DB neuve ET DB existante avec des users.
  - 3 tests : migration upgrade, downgrade no-op, default value.
- **Dépendances** : aucune (parallèle).

### S1-T5 — Tests E2E Sprint 1
- **Effort** : 1h
- **Owner** : solo
- **Description** : `tests/test_sprint1_insight_canonical.py` — un scénario complet "tick → ConfluenceSignal → InsightSignal → persisté → recharge depuis DB → validate → JSON dump".
- **DoD** : 1 test E2E vert, exécution < 2s.
- **Dépendances** : S1-T1, S1-T2, S1-T3.

**Total Sprint 1 : 8h. Critère go Sprint 2 : `InsightSignal` produit à chaque tick, persisté, validable depuis l'API admin.**

---

## Sprint 2 — Porte B2C (semaine 2, 10h)

> Objectif : Telegram + webapp + email digest consomment l'`InsightSignal` canonique. Vocabulaire UE 2024/2811 strict.

### S2-T1 — `B2CFormatter` Telegram
- **Effort** : 2h
- **Owner** : solo
- **Description** : Implémenter `src/distribution/b2c_formatter.py`, méthode `format_telegram(insight: InsightSignal, tier: UserTier) -> str`. Adapter logique de `telegram_notifier.format_signal_message()` au nouveau contrat. Score arrondi en label (`Strong (55-69)`), JAMAIS le chiffre brut. `narrative_short` au lieu de `full_narrative` pour FREE/ANALYST. Disclaimer footer obligatoire (déjà en place).
- **DoD** :
  - 12+ tests : tier FREE/ANALYST/STRATEGIST × bias bullish/bearish × lang FR/EN.
  - Tous les messages < 800 chars.
  - Aucune occurrence des chaînes interdites : `BUY`, `SELL`, `R-multiple`, `position size`, `pips earned`.
  - Test snapshot pour FR + EN.
- **Dépendances** : S1-T2.

### S2-T2 — Adapter `telegram_notifier.py`
- **Effort** : 1.5h
- **Owner** : solo
- **Description** : Ajouter méthode `send_insight(insight: InsightSignal, chat_id, tier)` dans `TelegramNotifier`. Déléguer formatage à `B2CFormatter`. Conserver l'ancien `send_signal()` pour compat descendante (deprecated marker).
- **DoD** :
  - 5 tests d'intégration : send_insight FREE/ANALYST/STRATEGIST + lookup lang via `lang_store` + circuit breaker open path.
  - Aucune régression sur `test_telegram_notifier.py` existant.
- **Dépendances** : S2-T1.

### S2-T3 — Webapp B2C dynamique
- **Effort** : 2.5h
- **Owner** : solo
- **Description** : Nouvelle route `GET /webapp/insight/{asset}` retournant le HTML reprenant le mockup `mockups/webapp_b2c.html`. Polling `/api/v1/insights/latest?audience=b2c` côté JS. Tier-gating : ring score visible pour tous, `narrative_full` pour ANALYST+, scenarios pour STRATEGIST+, composants_score_breakdown agrégés (pas détaillés) pour STRATEGIST+.
- **DoD** :
  - Page rendable standalone (test manuel browser local).
  - 4 tests : 200 OK, content-type HTML, disclaimer footer présent, no-cache header.
  - Lighthouse score > 90 (manuel).
- **Dépendances** : S1-T2, mockup webapp_b2c.html.

### S2-T4 — Email digest hebdomadaire
- **Effort** : 2h
- **Owner** : solo
- **Description** : Implémenter `src/distribution/email_digest.py`. Sélectionne top 5 insights de la semaine triés par `confluence_score` desc. Template HTML responsive simple. Cron lundi 08:00 UTC (à brancher hors sprint si pas de scheduler). MVP : commande CLI `python -m src.distribution.email_digest --week 2026-W18`.
- **DoD** :
  - 6 tests : sélection top 5, déduplication par insight_id, lang resolution, template render FR + EN, disclaimer présent, no-script HTML.
  - Sortie HTML validable via `html5validator`.
- **Dépendances** : S1-T2.

### S2-T5 — Disclaimer renforcé multilingue webapp
- **Effort** : 0.5h
- **Owner** : solo
- **Description** : Étendre `src/api/disclaimers.py` avec `get_long_disclaimer(lang)` (>= 500 chars, non-dismissible footer webapp). Texte juridique renforcé.
- **DoD** : 4 langues × test contenu, regex check sur mots-clés AMF/BaFin/CNMV/ESMA.
- **Dépendances** : aucune (parallèle).

### S2-T6 — Tests E2E Sprint 2
- **Effort** : 1.5h
- **Owner** : solo
- **Description** : `tests/test_sprint2_b2c_e2e.py` — 1 insight → 3 sorties (Telegram FR, webapp HTML, email digest entry).
- **DoD** : 3 tests E2E verts.
- **Dépendances** : S2-T1, S2-T2, S2-T3, S2-T4.

**Total Sprint 2 : 10h. Critère go Sprint 3 : un user FREE/ANALYST/STRATEGIST reçoit Telegram + webapp + email cohérents avec le même `InsightSignal`.**

---

## Sprint 3 — Porte B2B (semaine 3, 10h)

> Objectif : API REST authentifiée Bearer, OpenAPI auto, rate-limit per-tier, webhooks signés HMAC.

### S3-T1 — Enum `B2BTier` + extension `tier_manager.py`
- **Effort** : 1h
- **Owner** : solo
- **Description** : Ajouter `class B2BTier(str, Enum): PILOT / STANDARD / ENTERPRISE` dans `tier_manager.py`. Étendre `TIER_CONFIG` avec les 3 entrées B2B (cf. arch doc §6.2). Routing par `account_type='B2B'`.
- **DoD** : 6 tests : create user B2B, rate-limit per tier (100/5000/illim), webhook quota par tier.
- **Dépendances** : S1-T4.

### S3-T2 — Endpoints REST `/api/v1/insights/*`
- **Effort** : 3h
- **Owner** : solo
- **Description** : Implémenter `src/distribution/b2b_api.py` (FastAPI router). 4 endpoints :
  - `GET /api/v1/insights/latest?asset=&timeframe=` (200 InsightSignal | 404)
  - `GET /api/v1/insights/{insight_id}` (200 | 404)
  - `GET /api/v1/insights/historical?asset=&from=&to=&page=&page_size=` (200 paginé)
  - `GET /api/v1/health` (public, status moteur)
- **DoD** :
  - 14+ tests : 200/403/404/429 sur chaque endpoint, pagination edge cases, query param validation.
  - OpenAPI 3.0 auto-générée accessible à `/api/v1/docs` ; export `openapi.json` validé schémas Pydantic.
- **Dépendances** : S1-T3, S3-T1.

### S3-T3 — Bearer auth + rate-limit per-tier
- **Effort** : 1.5h
- **Owner** : solo
- **Description** : Implémenter `src/distribution/b2b_auth.py`. Réutilise `KeyStore` existant. Dependency FastAPI `require_b2b_tier(min_tier)` qui vérifie `account_type='B2B'` ET `tier >= min`. Rate-limit per-tier (counter SQLite avec window glissante 24h, leverage `tier_manager.UsageTracker` existant si présent).
- **DoD** :
  - 8 tests : valid/invalid/expired/revoked key, B2C key sur endpoint B2B → 403, rate-limit hit → 429 avec header `X-RateLimit-Reset`.
- **Dépendances** : S3-T1.

### S3-T4 — `WebhookPublisher` + endpoints souscription
- **Effort** : 3h
- **Owner** : solo
- **Description** : Implémenter `src/distribution/webhook_publisher.py`. Table `webhooks(webhook_id, key_id, callback_url, secret_hash, white_label, created_at, last_success_at, failure_count)`. Endpoints `POST /api/v1/webhooks/subscribe` + `DELETE /api/v1/webhooks/{id}`. Push HMAC SHA256 (header `X-Sentinel-Signature`, `X-Sentinel-Timestamp`). Retry exponentiel 3 tentatives (1s/4s/16s). Disable webhook après 10 échecs consécutifs.
- **DoD** :
  - 10 tests : signature HMAC valide, replay-attack guard (timestamp > 5 min), retry exp.backoff mocked, disable après 10 fails, white_label flag pris en compte.
  - Test E2E : insight → webhook poussé → callback mock reçoit payload signé.
- **Dépendances** : S1-T3, S3-T1.

### S3-T5 — Branchement `WebhookPublisher` dans le scanner
- **Effort** : 0.5h
- **Owner** : solo
- **Description** : Dans `_publish_signal`, après `b2c_formatter.dispatch`, appeler `webhook_publisher.publish(insight)`. Async/threaded pour ne pas bloquer le scanner.
- **DoD** : 3 tests : webhook poussé en arrière-plan, scanner pas bloqué, 0 régression `test_sentinel_scanner.py`.
- **Dépendances** : S3-T4.

### S3-T6 — Audit log B2B
- **Effort** : 0.5h
- **Owner** : solo
- **Description** : Étendre table `api_usage` existante avec `(key_id, endpoint, insight_id, ts, response_code, latency_ms)`. Middleware FastAPI logue chaque appel.
- **DoD** : 4 tests : log présent après request, no-log sur 401/403, latency_ms cohérent.
- **Dépendances** : S3-T2.

### S3-T7 — Tests E2E Sprint 3
- **Effort** : 0.5h
- **Owner** : solo
- **Description** : `tests/test_sprint3_b2b_e2e.py` — souscription webhook → insight publié → callback HTTP reçoit payload HMAC valide.
- **DoD** : 1 test E2E vert avec serveur HTTP mock.
- **Dépendances** : S3-T4, S3-T5.

**Total Sprint 3 : 10h. Critère go Sprint 4 : un broker fictif peut consommer `/insights/latest`, souscrire webhook, recevoir push signé HMAC.**

---

## Sprint 4 — Polish, doc, GTM (semaine 4, 6h)

> Objectif : tests bout-en-bout dual, doc API publique, pages tarifs, kit outbound 5 brokers.

### S4-T1 — Tests E2E parallèles B2C + B2B
- **Effort** : 1.5h
- **Owner** : solo
- **Description** : `tests/test_sprint4_dual_e2e.py` — un même `InsightSignal` est servi simultanément à 1 user FREE Telegram + 1 user STRATEGIST webapp + 1 broker B2B Standard via webhook. Vérification cohérence (même `insight_id`, même `confluence_score`, même `narrative_full` côté STRATEGIST/B2B).
- **DoD** : 3 tests E2E verts, exécution < 5s.
- **Dépendances** : Sprint 1-3 complets.

### S4-T2 — Doc API publique
- **Effort** : 1h
- **Owner** : solo
- **Description** : `docs/api/b2b_api.md` — guide d'intégration broker (auth, endpoints, webhooks, codes erreur, SLA). Capture OpenAPI export. Exemples curl.
- **DoD** : doc relue, exemples curl exécutables localement.
- **Dépendances** : Sprint 3 complet.

### S4-T3 — Pages tarifs landing page
- **Effort** : 1h
- **Owner** : solo
- **Description** : `mockups/pricing_b2c.html` (3 tiers FREE/$14/$39) et `mockups/pricing_b2b.html` (3 tiers PILOT/STANDARD/ENTERPRISE). Reuse tokens CSS `tradingview_dashboard_mockup.html`. Pas de paiement live (post-MVP), juste CTA "Get started" / "Talk to sales".
- **DoD** : 2 pages HTML standalone, mobile responsive (DevTools test), CTA pointent vers `/contact`.
- **Dépendances** : aucune.

### S4-T4 — Outbound starter pack 5 brokers
- **Effort** : 1.5h
- **Owner** : solo
- **Description** : `gtm/broker_outbound/` :
  - `email_template_FR.md` + `email_template_EN.md` (cold outreach intégration API)
  - `deck.md` (pitch 6 slides : problème → solution → InsightSignal demo → SLA/pricing → roadmap → next steps)
  - `target_list.csv` : 5 brokers shortlist (IC Markets, Exness, Pepperstone, FxPro, ActivTrades) — colonnes `broker, contact_role, email, status`.
- **DoD** : kit complet, email template < 200 mots, deck convertible PDF.
- **Dépendances** : aucune.

### S4-T5 — Mise à jour docs internes
- **Effort** : 0.5h
- **Owner** : solo
- **Description** : Update `COMPLETE_PROJECT_DOCUMENTATION.md` avec les nouvelles sections Dual-architecture. Update `MEMORY.md` avec les pointeurs vers les artefacts Sprint 1-4.
- **DoD** : doc relue, pas de lien mort.
- **Dépendances** : Sprint 1-3 complets.

### S4-T6 — Smoke retro & no-regression
- **Effort** : 0.5h
- **Owner** : solo
- **Description** : Run complet `pytest -q` ; vérifier 0 régression vs baseline 1366+ tests existants. Run `python -m src.intelligence.main` en TESTING_MODE 5 min ; vérifier dispatch B2C ET B2B sur 1 tick réel.
- **DoD** : suite verte, smoke logs propres.
- **Dépendances** : tous tickets Sprint 1-4.

**Total Sprint 4 : 6h. Critère release : 0 régression, kit outbound prêt à envoyer aux 5 brokers.**

---

## Récapitulatif effort

| Sprint | Charge | Tickets | Cible           |
| ------ | ------ | ------- | --------------- |
| 1      | 8h     | 5       | Contrat canonique vivant en prod |
| 2      | 10h    | 6       | B2C dispatch dual (TG + webapp + email) |
| 3      | 10h    | 7       | B2B API + webhooks broker-ready |
| 4      | 6h     | 6       | Polish + GTM kit |
| **Total** | **34h** | **24** | **Solo founder, 8-9h/sem, 4 semaines** |

---

## Graphe de dépendances (résumé)

```
S1-T1 ──┬─→ S1-T2 ──┬─→ S1-T3 ──┐
        │           │           │
        │           ├─→ S2-T1 ──→ S2-T2 ──┐
        │           │                     │
        │           ├─→ S2-T3 ────────────┤
        │           │                     │
        │           └─→ S2-T4 ────────────┴─→ S2-T6
        │
S1-T4 ──→ S3-T1 ──┬─→ S3-T2 ──→ S3-T6
                  │
                  ├─→ S3-T3
                  │
                  └─→ S3-T4 ──→ S3-T5 ──→ S3-T7

S2-T5 (parallèle)
S4-T3, S4-T4 (parallèle)

(S1-T1..S3-T7) ──→ S4-T1 ──→ S4-T2, S4-T5, S4-T6
```

---

## Risques connus + mitigations

| Risque                                                              | Impact     | Mitigation                                                                 |
| ------------------------------------------------------------------- | ---------- | -------------------------------------------------------------------------- |
| Migration `account_type` casse une instance prod                    | High       | Migration idempotente + test sur copie DB avant déploiement                |
| Rate-limit B2B trop strict bloque un broker pilote                  | Medium     | Soft-cap initial (2× limit) avec log warning ; hard-cap après 1 mois data  |
| Webhook callback broker indisponible pendant maintenance            | Medium     | Retry 3× exp.backoff + disable après 10 fails + email notification owner   |
| Vocabulaire UE 2024/2811 cassé par narrative LLM                    | Medium     | Linter regex sur `narrative_short/full` avant publish (forbidden words)    |
| Pivot pricing B2C $49→$14 angers existing INSTITUTIONAL subscribers | Low (n=0?) | Grandfathering jusqu'à résiliation volontaire ; check si users actifs       |
| Brokers refusent l'intégration ($500 PILOT trop bas, perçu amateur) | Medium     | Pricing test avec 1 broker friendly avant outbound large ; ajuster +50% si rejet systématique |

---

**Fin du sprint plan. À valider avant Sprint 1.**
