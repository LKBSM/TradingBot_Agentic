# Plan de Commercialisation — Catégorie 18 : Compliance & Légal

> **Périmètre.** Geo-blocking, MiFID II / AMF / ESMA, RGPD, CCPA, ePrivacy, CGU/CGV,
> Privacy Policy, data licensing (Dukascopy / ForexFactory / MT5), KYC/PSD2,
> finfluencer 2024/2811, certifications (SOC2 / ISO27001), e-discovery.
>
> **Auteur.** Agent compliance (catégorie 18 du sprint commercialisation).
> **Date.** 2026-05-21 · **Branch.** `institutional-overhaul`
>
> **DISCLAIMER OBLIGATOIRE — CECI N'EST PAS UN AVIS JURIDIQUE.** Ce plan est un
> document de cadrage opérationnel à soumettre à un avocat fintech qualifié pour
> validation. Les références (articles, numéros de directive, dates de mise en
> vigueur) sont citées pour vérification mais peuvent évoluer ; tout livrable
> contractuel publié doit être relu par un cabinet enregistré au Barreau de
> Paris (ou local) avant mise en ligne.

---

## 1. État actuel (Audit) — 7/10 post-W3, W4 manquant, data licensing risqué

### 1.1 Note d'exposition par dimension

| Dimension | Note /10 | Justification | Référence code |
|-----------|----------|---------------|----------------|
| Geo-blocking | **8** | US + UK + OFAC (CU/IR/KP/RU/SY/BY) + CA-QC bloqués. Multi-couche : header CDN > MaxMind > resolver test. Fail-open documenté. | `src/api/middleware/geo_block.py:65-83` |
| Disclaimers multi-langue | **8** | 4 langues (FR/EN/DE/ES) sur deux registres (long + footer Telegram ≤280 ch.). Footer ESMA-compliant 74-89 % loss. | `src/api/disclaimers.py:23-56` |
| Endpoints `/terms` + `/privacy` | **7** | 2 endpoints publics, multilingue, cache 1 h, version stamp. Contenu rédigé en interne, **non revu par avocat**. | `src/api/routes/legal.py:417-455` |
| Telegram language store | **7** | `chat_id → lang` SQLite + cache mémoire, alimenté par `/start`. | `src/delivery/telegram_lang_store.py:42-60` |
| Narratives reformulés UE 2024/2811 | **8** | LLM system prompt interdit explicitement "BUY/SELL", "ACHETER/VENDRE", "KAUFEN/VERKAUFEN", "COMPRAR/VENDER". Force "long setup", "bullish bias". | `src/intelligence/llm_narrative_engine.py:81,172` |
| W4 — Relecture juridique CGU | **0** | **MANQUANT.** Aucun cabinet n'a relu `_TERMS` ni `_PRIVACY` en multilingue. | `src/api/routes/legal.py:36-406` |
| Data licensing (Dukascopy / FF / MT5) | **3** | Dukascopy CGU = non-commercial only, FF scraping = zone grise, MT5 broker feed = clauses broker. Usage commercial déguisé. | `scripts/download_dukascopy_xau.py`, `scripts/fetch_forexfactory_live.py`, `scripts/export_mt5_history.py` |
| Endpoints DSAR (art. 15-22 RGPD) | **2** | Pas d'endpoint `GET /me/data` ni `DELETE /me`. Traitement manuel email-only sous 30 j. | _absent — à créer_ |
| Cookie banner CNIL | **0** | Pas de banner — pas encore de landing publique mais bloquant dès `https://[domain]/`. | _absent_ |
| Payment compliance (Stripe + PSD2 SCA) | **6** | Stripe checkout délégué + webhook signé. **MAIS** : pas de gate "accept terms" avant checkout, pas de retention politique pour `stripe_customer_id`. | `src/api/routes/billing.py:40-70` |
| RGPD — registre Art. 30 | **2** | Tableau dans la Privacy Policy mais pas de registre interne formel. | _absent_ |
| RC Pro + Cyber assurance | **0** | Aucune souscription. | _hors code_ |
| Médiation conso (art. L.612-1) | **0** | Médiateur non désigné. | _absent_ |
| MiFID II finfluencers (directive 2024/2811) | **7** | Reformulation prompt LLM faite. SL/TP encore chiffrés dans payload `InsightSignalV2` (non bloquant si réservé B2B + opt-in user). | `src/api/insight_signal_v2.py` |
| Backtest legal guardrails | **8** | Document normatif `BACKTEST_LEGAL_GUARDRAILS.md` interdit toute publication chiffrée hors walk-forward + IC95. | `BACKTEST_LEGAL_GUARDRAILS.md:1-239` |
| **GLOBAL EXPOSITION** | **6.5 / 10** | Acceptable pour bêta privée FR + EN <200 abonnés, **bloquant** pour public listing payant tant que W4 + data licensing + DSAR endpoints + RC Pro non livrés. | — |

### 1.2 Sprint W1+W2+W3 — déjà livré (rappel)

- **W1.** Geo-block `US/CA-QC/GB/CU/IR/KP/RU/SY/BY` (`src/api/middleware/geo_block.py`).
- **W2.** Disclaimers FR/EN/DE/ES + endpoints `/terms` + `/privacy` + version stamp (`src/api/routes/legal.py`).
- **W3.** Reformulation prompt LLM UE 2024/2811 + TelegramLangStore (`src/delivery/telegram_lang_store.py`).
- **Tests.** `tests/test_geo_block.py`, `tests/test_legal_endpoints.py`, `tests/test_telegram_lang.py` → 263 verts cumulés.

### 1.3 Verdict — 4 bloqueurs absolus pour facturation payante

1. **W4 — Relecture juridique CGU/Privacy** par avocat fintech FR (le contenu actuel est rédigé par IA, non opposable en cas de contentieux).
2. **Data licensing remediation** — migrer Dukascopy/FF vers providers commerciaux licenciés (Polygon, Trading Economics) **OU** restreindre Dukascopy au backtest interne avec note de cadrage.
3. **DSAR endpoints** — `GET /api/v1/me/data` + `DELETE /api/v1/me` (RGPD art. 15 + 17 — obligation, pas optionnel).
4. **Cookie banner CNIL-compliant** + bandeau consentement avant tout cookie analytics non-essentiel sur landing.

Sans ces 4 livrables, **Stripe peut suspendre** (T&C non opposable + DSAR manquant), **CNIL peut sanctionner** (jusqu'à 4 % CA ou 20 M€), **Dukascopy/FF peuvent envoyer cease-and-desist** (faible probabilité mais impact opérationnel élevé : downtime forcé).

---

## 2. Vision cible (W4 done, data 100 % licensed, MiFID 2026 compliant, GDPR full)

### 2.1 Cible 90 jours

- **Compliance** 8.5/10 (vs 6.5/10 actuel).
- **W4** relecture avocat fintech FR → CGU/Privacy v2 signées + tampon date.
- **Data 100 % licensed** : Polygon.io ($199/mois) + Trading Economics ($79/mois) OU TradingEconomics direct API + restriction Dukascopy "internal backtest only" documentée.
- **MiFID 2024/2811** (entrée mars 2026) compliant : `/insights/v2` propose `disclosure_mode=numeric|qualitative` paramètre, défaut `qualitative` (pas de SL/TP chiffrés sauf user opt-in explicite).
- **GDPR full** : registre Art. 30, DPA Anthropic signé, DSAR endpoints opérationnels avec 30 j SLA.
- **Assurance** RC Pro + Cyber bundle Stoïk ou Hiscox (~3-5 k€/an).
- **Cookie consent** Tarteaucitron auto-hébergé (gratuit, CNIL-compliant, MIT) ou Cookiebot (€7/mois jusqu'à 100 pages).
- **CCPA** (Californie) bypassée par geo-block US — pas d'effort additionnel tant que US bloqué.

### 2.2 Cible 12 mois

- **SOC2 Type 1** prep terminée (audit prévu mois 12-15) — accélérateur ventes B2B brokers.
- **ISO 27001** scoping commencé (audit mois 18+).
- **Médiation conso** : adhésion plateforme médiation FR (CM2C, MEDICYS — coût ~150 €/an).
- **DPA template B2B** signé avec 3-5 premiers clients institutionnels.
- **Multi-juridictions** : ouverture UK via Section 21 approval (firm tierce) **OU** maintien blocage UK (recommandé jusqu'à 1 k MRR cumulés).

---

## 3. Gap analysis

| Composant cible | État actuel | Gap | Effort | Catégorie |
|-----------------|-------------|-----|--------|-----------|
| W4 CGU/Privacy revue avocat | Rédigé IA, non revu | Critique — non opposable | 8 h interne + 3-5 k€ externe | P0 |
| Data licensing | Dukascopy + FF + MT5 zone grise | Migrer 2 sources + note "internal only" sur 3e | 24 h | P0 |
| DSAR `GET /me/data` | Absent | Endpoint + export JSON + tests | 16 h | P0 |
| DSAR `DELETE /me` (avec anonymisation signaux) | Absent | Endpoint + purge SQLite + anonymisation `signal_store` | 24 h | P0 |
| Cookie banner | Absent | Tarteaucitron auto-hébergé sur landing | 8 h | P0 |
| MiFID 2024/2811 `disclosure_mode` | Partiel (prompt only) | Param API + UI toggle "show numeric SL/TP" | 12 h | P0 |
| Stripe `terms_accepted_at` | Pas de gate | Checkbox + horodatage + persistance | 6 h | P0 |
| RC Pro + Cyber | Aucune souscription | Devis 3 assureurs + souscription | 4 h + 3-5 k€/an | P0 |
| DPA Anthropic signé | Mention dans Privacy, pas de PDF signé | Email support + signature | 2 h | P1 |
| Registre Art. 30 RGPD | Absent (tableau Privacy seulement) | Document interne 1 page | 4 h | P1 |
| DPA template B2B | Absent | Rédigé par avocat | 4 h + 1-2 k€ externe | P1 |
| Médiation conso (art. L.612-1) | Absent | Adhésion CM2C ou MEDICYS | 2 h + 150 €/an | P1 |
| DPO désignation | Non requis tant que <250 employés et hors cat. spéciales | Évaluer si dépasse seuils Art. 9 | 1 h | P2 |
| SOC2 Type 1 | Absent | Vanta + auditeur (~10-20 k€) | 80 h + 15 k€ externe | P2 |
| ISO 27001 prep | Absent | Scoping + ISMS framework | 40 h + 8 k€ externe | P3 |

---

## 4. Plan d'exécution

### P0 — W4 relecture juridique CGU/Privacy (cabinet avocat, ~3-5 k€)

**Objectif.** Rendre les documents `_TERMS` (`src/api/routes/legal.py:36-236`) et `_PRIVACY` (`src/api/routes/legal.py:239-406`) opposables en cas de contentieux. Le contenu actuel est rédigé par IA — défense fragile devant un juge ("on s'est basé sur un agent IA" ne tient pas si la clause de limitation de responsabilité est attaquée).

**Tâches.**

1. **Préparer brief avocat** (`compliance/avocat_brief.md` à créer, ~5 pages) :
   - Pitch produit (1 page) — "analyses algorithmiques de marché, pas conseil personnalisé".
   - Tableau juridictions cibles vs bloquées.
   - Tableau données traitées + sous-traitants (réutiliser tableau Privacy Policy lignes 250-257).
   - Liste questions ouvertes : qualification CIF, exception Lowe US (étant US bloqué), DPF UE-US.
2. **Sélectionner cabinet** — 3 RDV ronds gratuits ou ~200 € chacun :
   - Reed Smith Paris (gros budget, 500-800 €/h) — réserver pour Series A.
   - De Gaulle Fleurance & Associés (régulation financière, 350-500 €/h) — **recommandé**.
   - Aramis Avocats (SaaS B2C/RGPD, 250-400 €/h) — alternative low-cost.
3. **Mission d'audit** — 1 RDV cadrage (1 h, gratuit) + livrable écrit (5-10 j ouvrés) :
   - Validation qualification "service d'information" vs CIF.
   - Revue CGU v1 (4 langues) — corrections en mode track-changes.
   - Revue Privacy Policy v1 (4 langues).
   - Note séparée sur exposition MiFID II 2024/2811 (durcissement mars 2026).
   - Note séparée sur transferts UE→US (Anthropic, Stripe, hébergeur).
4. **Intégration corrections** dans `src/api/routes/legal.py` :
   - Mettre à jour `LAST_UPDATED` à la date de signature avocat.
   - Versionner via Git tag `legal-v2-YYYY-MM-DD`.
   - Update `compliance/cgu_signed_YYYY-MM-DD.pdf` (PDF horodaté).
5. **Alternative low-cost si budget <3 k€** : Captain Contrat (https://www.captaincontrat.com) — template CGU SaaS 300 € + 1-2 h consult avocat junior (~500 €) = ~800 € total, **acceptable seulement <100 abonnés payants**.

**Livrables.**

- `compliance/avocat_brief.md` (5 pages, à créer).
- 3 devis cabinets comparés.
- 1 mission signée (~3-5 k€).
- `compliance/cgu_signed_v2.pdf` + `compliance/privacy_signed_v2.pdf` horodaté.
- Git tag `legal-v2-YYYY-MM-DD` après merge des corrections.
- Mise à jour `LAST_UPDATED` dans `src/api/routes/legal.py:31`.

**Heures internes.** 12 h (4 brief + 2 RDV + 4 intégration + 2 tests/Git).
**Coût externe.** 3-5 k€ (cabinet) **ou** 800 € (Captain Contrat — bridge low-cost).
**Acceptance.**
- CGU + Privacy 4 langues signées par avocat enregistré au Barreau (FR).
- Note écrite avocat sur qualification CIF (defendable, non-CIF tant que <500 abonnés payants ou pas de chat Q&A personnalisé).
- Aucune modification critique pendante après merge.
**Dépendances.** Aucune (peut démarrer J1).

### P0 — Data licensing remediation (Dukascopy/FF → providers licenciés)

**Contexte.** `BACKTEST_LEGAL_GUARDRAILS.md` traite le risque représentation matérielle mais pas le risque licensing source. CGU Dukascopy = "free for personal use, commercial use requires written agreement" ; FF = scraping interdit ; MT5 broker feed = clauses broker variables.

**Tâches.**

1. **Audit usage interne actuel** :
   - `scripts/download_dukascopy_xau.py` → utilisé pour bars historiques XAU 2019-2025 (`XAU_15MIN_*.csv`).
   - `scripts/fetch_forexfactory_live.py` → calendrier économique HIGH-impact (NFP/FOMC/CPI) en runtime.
   - `scripts/export_mt5_history.py` → bars complémentaires multi-broker en dev only.
2. **Décision par source** :
   - **Dukascopy** → **restreindre au backtest interne** uniquement (jamais servi en live à un user payant). Note dans `BACKTEST_LEGAL_GUARDRAILS.md:7` mise à jour : "Données Dukascopy utilisées sous CGU 'personal use' pour validation interne walk-forward. Aucune donnée Dukascopy redistribuée à des abonnés payants." Migrer **bars live** vers Polygon.io ou Tiingo si jamais besoin.
   - **ForexFactory** → migrer vers **Trading Economics API** ($79-499/mois) ou **FRED + ECB direct feeds** (gratuits, plus de friction tech mais légalement clean). Option intermédiaire : **Investing.com API** via partenariat.
   - **MT5** → broker feed reste OK pour exécution / validation interne, **ne pas redistribuer** les bars comme service. Citer "market price as observed" dans tout payload public.
3. **Migration FF → Trading Economics** :
   - Souscrire compte TE ($79/mois Starter) → API key.
   - Refactor `scripts/fetch_forexfactory_live.py` → `scripts/fetch_trading_economics.py` (~6 h).
   - Update `src/agents/news/economic_calendar.py` pour pointer sur nouvelle source.
   - Run regression : comparer 30 j d'événements FF vs TE pour valider couverture HIGH-impact.
4. **Documentation licences** :
   - Créer `compliance/data_licensing_register.md` (1 page) :
     - Polygon : commercial license ✓ ($199/mois).
     - Trading Economics : commercial API ✓ ($79/mois).
     - Anthropic : DPA standard signé ✓.
     - MaxMind GeoLite2 : CC BY-SA 4.0 (free with attribution) ✓.
     - Dukascopy : "internal backtest only" ✓.
     - MT5 : "internal validation only" ✓.
5. **Mention sources dans Privacy / CGU** — Ajouter section "Sources de données" dans CGU v2 (cabinet avocat à briefer).

**Livrables.**

- `scripts/fetch_trading_economics.py` (NEW, remplace FF).
- `src/agents/news/economic_calendar.py` refactor pour Trading Economics.
- `compliance/data_licensing_register.md` (NEW, 1 page).
- Update `BACKTEST_LEGAL_GUARDRAILS.md` section "Dukascopy internal only".
- Update `.env.template` avec `TRADING_ECONOMICS_API_KEY`.
- `tests/test_trading_economics_provider.py` (NEW, ~20 tests).

**Heures internes.** 24 h (8 audit + 12 dev + 4 docs).
**Coût externe.** $79/mois (TE Starter) = ~80 €/mois récurrent + 0 € one-shot.
**Acceptance.**
- Aucune ligne de code en prod ne lit `forexfactory.com` directement.
- `compliance/data_licensing_register.md` recense 100 % des sources avec leur statut.
- Test e2e `tests/test_trading_economics_provider.py` couvre fetch + parse + compatibilité downstream `NewsAnalysisAgent`.
**Dépendances.** Coordonner avec **catégorie 5/15 (data)** (mêmes sources) ; ne pas casser pipeline en migrant.

### P0 — MiFID 2026 finfluencer audit (wording narratives, SL/TP non chiffrés sauf opt-in)

**Contexte.** Directive **2024/2811** entrée en vigueur mars 2026 — élargit `MiFID II Art. 4(1)(4)` "investment advice" : tout output mentionnant **SL/TP chiffrés** peut être interprété comme "actionnable" → glissement vers conseil personnalisé. Le prompt LLM (`src/intelligence/llm_narrative_engine.py:81`) interdit déjà l'impératif "BUY/SELL", mais le payload `InsightSignalV2` continue d'émettre `stop_loss`, `take_profit`, `entry_price` chiffrés.

**Tâches.**

1. **Ajouter `disclosure_mode` dans `InsightSignalV2`** :
   - `numeric` (défaut **B2B** uniquement) : SL/TP/entry chiffrés.
   - `qualitative` (défaut **B2C public**) : `"stop_loss_zone": "below recent swing low"`, pas de chiffre.
   - `numeric_optin` : numeric **après** acceptation opt-in user (`POST /api/v1/me/preferences {disclosure_numeric: true}`).
2. **Refactor `src/api/insight_signal_v2.py`** :
   - Champ `disclosure_mode: Literal["numeric", "qualitative", "numeric_optin"]` (défaut `qualitative`).
   - Sérialiseur conditionnel : si `qualitative`, masquer `stop_loss`, `take_profit`, `entry_price` derrière `stop_loss_zone`, `take_profit_zone` strings.
3. **Telegram / Discord notifiers** :
   - Par défaut footer + narrative qualitative (cf. `disclosure_mode=qualitative`).
   - Pour les B2B tier INSTITUTIONAL, lire `disclosure_mode=numeric` depuis user profile.
4. **Endpoint user opt-in** :
   - `POST /api/v1/me/preferences` body `{disclosure_numeric: bool}`.
   - Persistance dans SQLite (table `user_preferences`).
   - Disclaimer additionnel affiché lors de l'opt-in : "Vous acceptez de recevoir des niveaux chiffrés à titre purement informatif. Smart Sentinel AI n'est pas un CIF."
5. **Tests** :
   - `tests/test_disclosure_mode.py` — asserte `qualitative` masque chiffres, `numeric` les expose.
   - `tests/test_mifid_2024_2811_wording.py` — asserte narratives ne contiennent jamais "BUY GOLD at 2050.00" ni équivalent.

**Livrables.**

- Refactor `src/api/insight_signal_v2.py` avec `disclosure_mode`.
- Refactor `src/delivery/telegram_notifier.py` + `discord_notifier.py` (lecture user pref).
- Endpoint `src/api/routes/user_preferences.py` (NEW).
- Tests + 100 % coverage `disclosure_mode` branches.
- Mention dans CGU v2 section "modes de divulgation" (cabinet avocat à briefer).

**Heures internes.** 12 h.
**Coût externe.** 0 € (mais inclure dans brief avocat P0 #1).
**Acceptance.**
- En `disclosure_mode=qualitative`, aucun chiffre SL/TP/entry n'est rendu.
- Opt-in numeric persisté et exposé via `GET /api/v1/me/preferences`.
- Pas de régression existante sur tests `test_insight_signal_v2*`.
**Dépendances.** Coordonner avec **catégorie 4 (delivery)** (Telegram / Discord wording) et **catégorie 6 (narratives)** (prompt LLM cohérent).

### P0 — Payment compliance Stripe (no charge before /terms accept, PSD2 SCA)

**Contexte.** `src/api/routes/billing.py:40-70` crée un checkout Stripe sans gate "accept terms". Stripe lui-même refuse onboarding marchand si T&C URL absente, mais une fois listé, **aucun horodatage** ne prouve que l'utilisateur a accepté à la date de souscription.

**Tâches.**

1. **Frontend checkbox** (à coordonner avec catégorie 10 — UI) :
   - Avant `POST /api/v1/billing/checkout`, exiger acceptation explicite "J'ai lu les CGU et la Politique de confidentialité" (non pré-coché).
   - Submit avec `terms_accepted_at: ISO8601` dans body.
2. **Backend** :
   - Ajouter `terms_accepted_at: datetime` + `terms_version: str` dans `CheckoutBody` (`src/api/routes/billing.py:31`).
   - Rejet 400 si `terms_accepted_at` absent ou `terms_version` ≠ `LAST_UPDATED` actuel.
   - Persister dans `user_subscriptions` SQLite (lien `stripe_customer_id` ↔ `terms_accepted_at` ↔ `terms_version`).
3. **PSD2 SCA** — Stripe Checkout gère le SCA out-of-the-box pour cartes EU. **Vérifier** :
   - `stripe.create_checkout_session` utilise mode "payment_method_types=['card']" + `payment_intent_data={'setup_future_usage':'off_session'}` pour récurrence SCA-conforme.
   - Tester avec carte test 3DS Stripe `4000 0025 0000 3155`.
4. **Webhook signature verification** — déjà fait (`src/billing/stripe_client.py::parse_webhook_event`). Asserter via `tests/test_stripe_webhook_signature.py`.
5. **Retention `stripe_customer_id`** — documenter 10 ans (obligation comptable FR, code de commerce art. L123-22). Déjà dans Privacy `legal.py:307`. **Vérifier** que purge `DELETE /me` n'efface pas `stripe_customer_id` mais l'anonymise (cf. P0 DSAR).

**Livrables.**

- Refactor `src/api/routes/billing.py::CheckoutBody` avec `terms_accepted_at` + `terms_version`.
- Migration SQLite `user_subscriptions` ajoute colonnes `terms_accepted_at`, `terms_version`.
- `tests/test_billing_terms_gate.py` (NEW) — asserte 400 si terms manquant.
- `tests/test_stripe_sca_3ds.py` (NEW) — asserte 3DS challenge flow.

**Heures internes.** 6 h.
**Coût externe.** 0 € (Stripe inclus).
**Acceptance.**
- Aucun checkout ne peut être créé sans `terms_accepted_at`.
- 100 % des charges récurrentes passent par SCA (vérifié sur 10 transactions test).
- `terms_version` stocké correspond à `LAST_UPDATED` au moment du checkout.
**Dépendances.** P0 W4 (terms revus avocat) **avant** activation gate — sinon on horodate du contenu non opposable.

### P0 — Cookie banner CNIL-compliant + consent management (Cookiebot / Tarteaucitron)

**Contexte.** `src/api/routes/legal.py:287-289` mentionne "cookies via banner before non-essential cookies" mais aucune implémentation n'existe (et pas encore de landing page publique). Bloquant **dès la mise en ligne** d'une landing si Google Analytics, Hotjar, Meta Pixel, Stripe.js (cookies third-party) sont chargés.

**Tâches.**

1. **Audit cookies utilisés** sur landing future + webapp :
   - Stripe.js (cookies session — exemption art. 82 LIL "strictement nécessaire").
   - Plausible Analytics (cookieless — RECOMMANDÉ, pas de banner requis).
   - Si Google Analytics 4 ou Meta Pixel → consent requis.
2. **Choisir solution** :
   - **Tarteaucitron** (https://tarteaucitron.io/fr/) — open-source, auto-hébergé, MIT, CNIL-compliant, granularité par finalité. **RECOMMANDÉ** (0 €).
   - **Cookiebot** (https://www.cookiebot.com/) — SaaS, scan auto, jusqu'à 100 pages free, ensuite €7/mois. **Alternative low-effort**.
   - **Axeptio** (https://www.axeptio.eu/fr/home) — alternative FR €15/mois.
3. **Implémentation Tarteaucitron** :
   - Servir `tarteaucitron.js` depuis CDN ou bundle landing.
   - Config : `bodyPosition: 'bottomLeft'`, `cookieslist: false`, `services: {plausible, stripe, gtag(opt)}`.
   - Granularité : "Mesure d'audience" / "Réseaux sociaux" / "Vidéos" / "Marketing" séparées.
   - Refus = aussi facile que accepter (bouton "Tout refuser" au même niveau que "Tout accepter").
4. **Documenter** dans `src/api/routes/legal.py::_PRIVACY` section cookies (`legal.py:286-289`) :
   - Lister chaque cookie : finalité, durée, sous-traitant.
   - Lien vers Tarteaucitron settings : "Modifier mes préférences".
5. **Tests E2E Playwright** (catégorie 10 UI) :
   - Asserter banner visible 1er visit.
   - Asserter `document.cookie` ne contient que cookies essentiels avant consent.
   - Asserter `_ga` / `fbp` apparaissent uniquement après consent "Marketing".

**Livrables.**

- Bundle `landing/static/tarteaucitron/` (à créer dans repo landing — hors scope si landing séparée).
- Config `landing/cookie-config.js` (NEW).
- Update `src/api/routes/legal.py::_PRIVACY` sections cookies multilingue (4 langues).
- Tests Playwright `tests/e2e/test_cookie_banner.py` (NEW).

**Heures internes.** 8 h.
**Coût externe.** 0 € (Tarteaucitron) ou 7 €/mois (Cookiebot).
**Acceptance.**
- Aucun cookie non-essentiel set avant consent (vérifié manuellement Chrome DevTools + Playwright).
- Bouton "Tout refuser" présent et fonctionnel.
- CGU section "Cookies" liste tous les cookies utilisés avec finalité.
**Dépendances.** Landing page existante (catégorie 10) — bloquant si landing pas encore live mais peut démarrer en parallèle.

### P1 — DPA template pour clients B2B (Data Processing Agreement)

**Contexte.** Quand Smart Sentinel vend en B2B (brokers, prop firms, néobanques), le client devient **controller** et SSAI devient **processor**. Un DPA mutuel signé est requis par RGPD Art. 28.

**Tâches.**

1. **Rédiger template DPA B2B** (avocat, ~1 k€ external) :
   - Définitions controller / processor.
   - Finalité du traitement (livraison signaux algorithmiques).
   - Durée + suppression / restitution fin de contrat.
   - Sous-processeurs autorisés (Anthropic, Stripe, hébergeur) + clause notification 30 j.
   - Mesures techniques et organisationnelles (TOMs).
   - Audit droit par client (préavis 30 j, 1×/an max).
   - Transferts UE→US via DPF + CCT 2021/914.
2. **Workflow signature** :
   - Template Word/PDF dans `compliance/templates/DPA_b2b_template_v1.pdf`.
   - Envoi DocuSign ou Yousign à chaque nouveau client institutionnel.
   - Archivage signé dans `compliance/signed/DPA_{client_id}_{date}.pdf` (gitignored ou Vault).
3. **Listing sous-processeurs publique** :
   - `GET /api/v1/legal/subprocessors` — JSON listant Anthropic, Stripe, Telegram, Polygon, Trading Economics, hébergeur.
   - Update auto quand nouveau sous-processeur ajouté (notification clients existants 30 j).

**Livrables.**

- `compliance/templates/DPA_b2b_template_v1.pdf` (avocat).
- Endpoint `GET /api/v1/legal/subprocessors` (NEW).
- Process documenté dans `compliance/b2b_onboarding_checklist.md`.

**Heures internes.** 4 h.
**Coût externe.** 1-2 k€ (cabinet — peut être inclus dans mission P0 W4 négociée).
**Acceptance.**
- Template signable en l'état par 3 personas clients (broker mid-market, prop firm, néobanque).
- Endpoint subprocessors retourne JSON valide.
**Dépendances.** P0 W4 (cohérence wording avec CGU principale).

### P1 — Endpoints DSAR (RGPD Art. 15-22) — accès, effacement, portabilité

**Contexte.** Privacy Policy promet "réponse sous 30 jours" (`legal.py:278`) mais aucun endpoint n'existe. Traitement manuel email-only = risque procédure CNIL si user mécontent + non-scalable.

**Tâches.**

1. **`GET /api/v1/me/data`** (Art. 15 + 20 RGPD — droit d'accès + portabilité) :
   - Auth requis (`require_api_key`).
   - Retourne JSON :
     ```json
     {
       "user_id": "...",
       "email": "...",
       "api_key_created_at": "...",
       "tier": "PRO",
       "stripe_customer_id": "cus_...",
       "telegram_chat_id": "12345",
       "preferences": {...},
       "signals_received_last_12_months": [...]
     }
     ```
   - Headers : `Content-Disposition: attachment; filename="my_data_export.json"`.
2. **`DELETE /api/v1/me`** (Art. 17 RGPD — droit à l'oubli) :
   - Auth requis.
   - Workflow :
     1. Confirmation 2-step : `POST /api/v1/me/delete-request` → email confirmation token → `DELETE /api/v1/me?token=...`.
     2. Anonymisation `signal_store` : `user_id → hash_sha256(user_id + salt)` (préserve métriques agrégées, art. 17 §3.b).
     3. Suppression hard : `users` table, `user_preferences`, `telegram_lang_store`, API keys actives.
     4. Conservation `stripe_customer_id` 10 ans (obligation comptable art. L123-22 code commerce) — anonymiser email associé.
     5. Notification email "Compte supprimé. Conservation comptable 10 ans (anonymisée) en application de l'art. L123-22."
3. **`POST /api/v1/me/objection`** (Art. 21 RGPD) — désactivation marketing emails sans suppression compte.
4. **`GET /api/v1/me/preferences` + `POST /api/v1/me/preferences`** (Art. 16 + 21) :
   - Rectification + opt-in/opt-out (disclosure_numeric, marketing_emails, telegram_lang).
5. **Logs DSAR** :
   - Table `dsar_requests` SQLite : `user_id, request_type, requested_at, fulfilled_at, sla_remaining_days`.
   - Endpoint admin `GET /api/v1/admin/dsar/pending` pour monitoring.

**Livrables.**

- `src/api/routes/me.py` (NEW) — 5 endpoints `GET /me/data`, `DELETE /me`, `POST /me/objection`, `GET/POST /me/preferences`.
- Migration SQLite : table `dsar_requests`.
- Anonymisation helper `src/api/anonymize.py` (NEW) — hash determinist SHA256+salt.
- Tests `tests/test_dsar_endpoints.py` (NEW, ~30 tests).
- Documentation `docs/dsar_workflow.md` (NEW).

**Heures internes.** 24 h (16 dev + 4 tests + 4 doc).
**Coût externe.** 0 €.
**Acceptance.**
- Export JSON contient 100 % des données utilisateur listées dans Privacy `legal.py:251-256`.
- DELETE workflow 2-step robuste (token expire 24 h, ré-émission possible).
- Anonymisation préserve métriques agrégées sur signaux historiques.
- SLA 30 j monitoring opérationnel.
**Dépendances.** Coordonner avec **catégorie 12 (auth)** (require_api_key) et **catégorie 7 (signal_store)** (anonymisation).

### P1 — Records of processing Art. 30 RGPD

**Contexte.** Solo founder <250 employés exempté de DPO mais **pas** du registre Art. 30 dès qu'il y a transferts hors-UE (Anthropic US, Stripe US/IE). CNIL contrôle aléatoire → présenter registre dans les 7 j.

**Tâches.**

1. Rédiger `compliance/registre_traitements_2026.md` (1-2 pages) :
   - Identité responsable.
   - 9 traitements (cf. Privacy `legal.py:251-256` × Telegram + IP + tracker).
   - Pour chaque traitement : finalité, base légale, catégories de données, destinataires, transferts hors-UE, durée, mesures de sécurité.
2. Garder en interne (gitignored ou vault — contient potentiellement adresses sous-traitants).
3. Update annuel obligatoire.

**Livrables.**

- `compliance/registre_traitements_2026.md` (gitignored).
- Process annuel documenté dans `compliance/yearly_review_checklist.md`.

**Heures internes.** 4 h.
**Coût externe.** 0 €.
**Acceptance.** Registre couvre 100 % traitements actuels + transferts UE→US documentés.
**Dépendances.** Aucune.

### P1 — Médiation conso (art. L.612-1 Code Consommation)

**Contexte.** Obligation pour tout B2C FR : adhérer à un médiateur de la consommation référencé CECMC (Commission d'évaluation et de contrôle de la médiation de la consommation).

**Tâches.**

1. Choisir plateforme :
   - **CM2C** (Centre de la Médiation de la Consommation de Conciliateurs de Justice) — ~150 €/an, fintech-friendly.
   - **MEDICYS** — ~80-200 €/an.
2. Adhérer, recevoir numéro d'agrément.
3. Mentionner dans CGU v2 article "Médiation" :
   > _En cas de litige non résolu après réclamation écrite, le consommateur peut saisir gratuitement le médiateur [nom] (www.medicys.fr) dans le délai d'un an à compter de la réclamation initiale._
4. Lien dans footer landing.

**Livrables.** Adhésion + mention CGU + footer landing.
**Heures internes.** 2 h.
**Coût externe.** 80-200 €/an.
**Acceptance.** Mention CGU + lien fonctionnel.
**Dépendances.** P0 W4.

### P1 — DPO désignation (si seuils Art. 9 RGPD dépassés)

**Contexte.** DPO obligatoire si :
- Traitement à grande échelle de données sensibles (art. 9 — santé, opinions, biométrie). **Non applicable** à SSAI.
- Suivi régulier et systématique à grande échelle. **Marginal** (Telegram chat IDs).
- Organisme public. **Non applicable**.

**Tâches.**

1. Analyse impact (PIA) légère pour confirmer non-obligation.
2. Designation volontaire **DPO externe mutualisé** (50-150 €/mois) si scale > 5 k abonnés.
3. Si pas DPO formel : nommer "responsable RGPD" interne (founder = Loukmane).

**Livrables.** Note interne `compliance/dpo_assessment.md` (1 page).
**Heures internes.** 1 h.
**Coût externe.** 0 € maintenant, 50-150 €/mois post-PMF si nécessaire.
**Acceptance.** Note d'analyse + nom responsable RGPD désigné.
**Dépendances.** Aucune.

### P2 — Certification SOC2 Type 1 → 2 (B2B sales accelerator)

**Contexte.** Pas obligatoire mais **bloquant** pour ventes broker / institutionnel >$10 k MRR. SOC2 Type 1 = "design controls in place at point in time". Type 2 = "controls effective over 6-12 months".

**Tâches.**

1. **Pre-audit gap analysis** (1 mois) :
   - Choisir auditeur : Drata, Vanta, Tugboat Logic, AssuranceLab.
   - Vanta auto-scan : ~10 k$/an (subscription) + auditeur Type 1 ~8-10 k$.
2. **Contrôles à implémenter** :
   - Access control : MFA admin, RBAC.
   - Encryption at rest + in transit.
   - Backup + DR plan.
   - Incident response runbook.
   - Vendor management (Anthropic, Stripe, etc.) — DPA + risk assessments signés.
   - Change management : Git-based, code review obligatoire, CI tests.
3. **Audit Type 1** (mois 6) → certificat.
4. **Window 6-12 mois** (Type 2) → second audit (mois 12-18) → certificat Type 2.

**Livrables.** Certificat SOC2 Type 1 (mois 12) puis Type 2 (mois 18).
**Heures internes.** 80 h (étalées 6-12 mois).
**Coût externe.** 10-20 k$ (Vanta + auditeur) Type 1 ; +8-12 k$ Type 2.
**Acceptance.** Certificat sur lettrage Vanta dashboard publique.
**Dépendances.** RC Pro + Cyber + DPA Anthropic signés.

### P2 — ISO 27001 prep

**Contexte.** ISO 27001 plus exigeant que SOC2, **standard EU** pour ventes B2B institutionnelles (banques, fonds). Scope ISMS (Information Security Management System).

**Tâches.**

1. Choisir auditeur accrédité COFRAC (FR) — Bureau Veritas, AFNOR, LRQA.
2. Scoping ISMS : périmètre + politique sécurité + risk assessment.
3. 6-12 mois implementation contrôles Annexe A (114 contrôles).
4. Audit blanc puis audit certification.

**Livrables.** Certificat ISO 27001 (mois 18-24).
**Heures internes.** 40 h scoping + 80 h implementation.
**Coût externe.** 8-15 k€.
**Acceptance.** Certificat valide 3 ans + audits de suivi annuels.
**Dépendances.** SOC2 Type 1 acquis (recouvrement contrôles ~70 %).

---

## 5. Tests & validation

| Test | Périmètre | Fichier |
|------|-----------|---------|
| Geo-block US | header `cf-ipcountry=US` → HTTP 451 | `tests/test_geo_block.py` (existe) |
| Geo-block CA-QC | headers `cf-ipcountry=CA` + `cf-region-code=QC` → 451 | `tests/test_geo_block.py` |
| Geo-block bypass `/terms`, `/privacy`, `/health` | paths allowlist toujours servis | `tests/test_geo_block.py` |
| Disclaimer présent Telegram payload | footer multi-langue présent | `tests/test_disclaimer_present.py` (à créer) |
| Disclaimer présent Discord payload | footer présent | `tests/test_disclaimer_present.py` |
| Disclaimer présent API `SignalResponse` | champ `disclaimer` obligatoire | `tests/test_disclaimer_present.py` |
| Langue auto-détectée `Accept-Language` | FR/EN/DE/ES + fallback `en` | `tests/test_legal_endpoints.py` (existe) |
| `/terms` versions stamp | endpoint `/legal/version` retourne version actuelle | `tests/test_legal_endpoints.py` |
| MiFID 2024/2811 — narratives sans "BUY/SELL" | regex assert no imperatives | `tests/test_mifid_2024_2811_wording.py` (à créer) |
| `disclosure_mode=qualitative` masque chiffres | SL/TP/entry absents du payload | `tests/test_disclosure_mode.py` (à créer) |
| `disclosure_mode=numeric_optin` exige opt-in | sans opt-in → 403 | `tests/test_disclosure_mode.py` |
| DSAR `GET /me/data` complet | toutes les données listées dans Privacy retournées | `tests/test_dsar_endpoints.py` (à créer) |
| DSAR `DELETE /me` 2-step | token confirmation requis, expire 24 h | `tests/test_dsar_endpoints.py` |
| DSAR anonymisation signal_store | `user_id` hashé après delete, métriques préservées | `tests/test_dsar_endpoints.py` |
| Stripe `terms_accepted_at` gate | checkout 400 si manquant | `tests/test_billing_terms_gate.py` (à créer) |
| Stripe SCA 3DS | challenge déclenché pour cartes EU | `tests/test_stripe_sca_3ds.py` (à créer) |
| Cookie banner Playwright | aucun cookie non-essentiel avant consent | `tests/e2e/test_cookie_banner.py` (à créer) |

**Couverture cible** : 95 % branches sur `geo_block.py`, `legal.py`, `disclaimers.py`, `insight_signal_v2.py`, `me.py`, `billing.py`.

---

## 6. Sécurité (data subject rights endpoints : export, delete ; PII inventory)

### 6.1 PII inventory

| Surface | PII | Stockage | Chiffrement |
|---------|-----|----------|-------------|
| Signup | Email | SQLite `users` | At-rest TBD (catégorie 12) |
| API auth | API key hash (bcrypt) | SQLite `users` | hash, pas chiffrement |
| Telegram | chat_id + langue | SQLite `telegram_lang_store` | En clair |
| Discord | webhook URL | SQLite `user_settings` | En clair (secret faible) |
| Stripe | customer_id | SQLite `user_subscriptions` | En clair |
| Geo | IP + country code | logs runtime + DB 30 j | En clair (logs rolling) |
| Webhooks delivery | timestamps + status | SQLite 14 j | En clair |

**Gap** : chiffrement at-rest manquant sur SQLite. **Recommandation** : SQLCipher (10 % perf overhead) ou migration Postgres + pgcrypto post-PMF.

### 6.2 Anonymisation pattern (DSAR Delete)

```python
# src/api/anonymize.py (à créer)
import hashlib
import os
SALT = os.environ["ANONYMIZE_SALT"]  # 32 bytes random, persistant

def anonymize_user_id(user_id: str) -> str:
    return "anon_" + hashlib.sha256((user_id + SALT).encode()).hexdigest()[:16]
```

- Déterministe (même input → même output) → préserve jointures métriques agrégées.
- Irréversible (sans `SALT` et fonction inverse).
- Conforme art. 17 §3.b RGPD (anonymisation = pas suppression mais bien dépersonnalisation).

### 6.3 PII en LLM payload

**Audit prompt LLM** (`src/intelligence/llm_narrative_engine.py`) : confirmer que **aucune** PII (email, chat_id, IP) n'est jamais envoyée à Anthropic. Seuls OHLCV + scores + état machine.

**Action** : test régression `tests/test_llm_no_pii.py` (à créer) — mock Anthropic, asserter zéro mention email/IP dans messages envoyés.

---

## 7. Métriques (geo-block coverage %, consent rate, complaints, DSR response time)

| KPI | Mesure | Baseline | Cible 30 j | Cible 90 j |
|-----|--------|----------|------------|------------|
| Geo-block coverage | Pays bloqués / pays cibles déclarés | 9/12 | 12/12 | 12/12 |
| Geo-block fail-open rate | Requests sans pays résolu / total | TBD | <2 % | <0.5 % |
| Disclaimer consent rate | Users acceptant CGU au signup | n/a (no gate) | 100 % | 100 % |
| Cookie consent acceptation rate | Users acceptant ≥1 catégorie analytics | n/a | 40-60 % | 50-70 % |
| DSAR SLA respect | Demandes résolues <30 j / total | n/a | 100 % | 100 % |
| DSAR volume mensuel | Requests DELETE + GET data | 0 | 1-5 | 5-20 |
| Plaintes CNIL reçues | Notifications officielles | 0 | 0 | 0 |
| Cease-and-desist reçus | Dukascopy/FF/autres | 0 | 0 | 0 |
| Incidents data breach | Notifiables CNIL (72 h) | 0 | 0 | 0 |
| Coût compliance / mois | Assurance + Cookiebot + TE + médiation | 0 | ~350 € | ~600 € |
| Coût compliance one-shot | Avocat + setup audit | 0 € | 4-6 k€ | 6-10 k€ |
| MTBF (mean time between failures) compliance | Jours entre incidents compliance | n/a | >180 j | >365 j |
| Tests compliance verts | Tests `test_geo_block`, `test_legal`, `test_dsar`, `test_disclaimer_present` | 35 | 80 | 120 |

---

## 8. Risques & mitigations

| # | Risque | Probabilité (12 mois) | Impact | Mitigation | Owner |
|---|--------|-----------------------|--------|------------|-------|
| 1 | **MiFID 2024/2811 application mars 2026** (durcissement finfluencer) | Haute | €€ (mise en demeure AMF, retrait obligatoire) | `disclosure_mode=qualitative` default + opt-in numeric + reformulation narratives. **P0**. | Founder |
| 2 | **Cease-and-desist Dukascopy / ForexFactory** | Moyenne | €€ (downtime forcé, frais avocat) | Migrer FF → Trading Economics, restreindre Dukascopy internal-only. **P0**. | Founder |
| 3 | **Plainte CNIL absence DSAR endpoint** | Moyenne (1 user fâché suffit) | €€€ (sanction jusqu'à 4 % CA ou 20 M€, souvent rappel à l'ordre d'abord) | DSAR endpoints + registre Art. 30 + DPA Anthropic. **P0-P1**. | Founder |
| 4 | **Réclamation client perte trading** | Faible mais possible | €€ (frais défense + dommages directs) | RC Pro Hiscox 1.5-3.5 k€/an + plafond responsabilité CGU. **P0**. | Founder |
| 5 | **Stripe gel compte (CGU non opposable)** | Faible | €€€ (revenue à 0, 60-90 j négociation) | W4 avocat + `terms_accepted_at` gate + version stamp. **P0**. | Founder |
| 6 | **AMF requalification CIF** | Faible <500 abonnés, croît avec marketing | €€€ (interdiction commercialiser 6 mois) | Note cadrage avocat fintech + ne pas marketer "signaux" → "analyses". **P0 W4**. | Founder |
| 7 | **Schrems III invalide DPF UE-US** | Faible 2026-2027, monitoring | €€ (migration Anthropic ou alternative Mistral) | Veille jurisprudentielle + plan B Mistral Large hébergé UE. **P3**. | Founder |
| 8 | **UE AI Act classification "high-risk"** | Très faible (SaaS financier = "limited risk" actuellement) | €€ (transparence renforcée, audit conformité) | Veille + classification revue annuelle. **P3**. | Founder |
| 9 | **GDPR data breach** | Faible | €€€ (notification CNIL 72 h, sanction si négligence) | Cyber assurance Stoïk + incident response runbook. **P0**. | Founder |
| 10 | **Litige PI (LuxAlgo, TradingView, brevets)** | Très faible | €€ (frais défense) | Audit code propriétaire + pas de reprise littérale techniques publiées. **P3**. | Founder |
| 11 | **Réforme statut auto-entrepreneur (TVA intra-comm 2026)** | Certaine | € (comptable + revue facturation) | Briefer expert-comptable. **P1**. | Founder |
| 12 | **MiCA UE crypto signaux** | Si extension vers crypto | €€ (nouvelle licence) | Reporter extension crypto post-PMF. **P3**. | Founder |

---

## 9. Dépendances

### 9.1 Dépendances internes (autres catégories sprint)

| Catégorie cible | Dépendance | Direction |
|----|------------|-----------|
| **Cat. 5 — Data sources** | Migration FF → Trading Economics + Dukascopy internal-only | Cat. 18 livre note licensing → Cat. 5 implémente refactor |
| **Cat. 6 — LLM Narratives** | Cohérence prompt MiFID 2024/2811 + disclosure_mode | Cat. 18 spec wording → Cat. 6 implémente |
| **Cat. 4 — Delivery** | Footer multi-langue Telegram + Discord + `disclosure_mode` | Cat. 18 livre disclaimers → Cat. 4 wire |
| **Cat. 7 — Auth / Account** | DSAR endpoints `/me/*` + Stripe `terms_accepted_at` | Cat. 18 spec → Cat. 7 implémente |
| **Cat. 10 — UI / Landing** | Cookie banner + checkbox terms | Cat. 18 spec → Cat. 10 implémente |
| **Cat. 12 — Security / Auth** | Chiffrement at-rest SQLite + anonymisation hash | Cat. 18 spec → Cat. 12 implémente |
| **Cat. 15 — Data licensing** | Recensement sources + statut commercial | Cat. 18 livre `compliance/data_licensing_register.md` |

### 9.2 Dépendances externes

| Item | Fournisseur | Lead time | Coût |
|------|-------------|-----------|------|
| Mission audit CGU | Cabinet avocat fintech FR | 2-4 semaines | 3-5 k€ |
| RC Pro + Cyber | Hiscox / Stoïk / Wakam | 1-2 semaines (devis 30 min) | 3-5 k€/an |
| DPA Anthropic | Anthropic Support | 2-4 semaines | 0 € |
| Trading Economics | TE Sales | 1-2 j | $79-499/mois |
| MaxMind GeoLite2 | MaxMind (signup gratuit) | 1 j | 0 € |
| Médiateur conso (CM2C) | CM2C | 1 semaine | 80-200 €/an |
| Vanta (SOC2) | Vanta | 1-3 mois (audit) | ~10 k$/an |
| Tarteaucitron | Self-hosted | 1 j | 0 € |

---

## 10. Estimation totale & timeline

### 10.1 Heures internes + budget externe (toutes priorités cumulées)

| Priorité | Heures internes | Budget externe one-shot | Budget récurrent |
|----------|------------------|--------------------------|-------------------|
| **P0** (J1-J30) | **74 h** | **5-8 k€** | **~80 €/mois** (TE) |
| **P1** (J30-J90) | **34 h** | **1-3 k€** | **~250-450 €/mois** (RC Pro + Cyber + médiation + Cookiebot) |
| **P2** (M3-M12) | **80-120 h** | **15-25 k$** (Vanta + SOC2 auditeur Type 1) | **+ ~830 $/mois** (Vanta) |
| **P3** (M12-M24) | **80-120 h** | **8-15 k€** (ISO 27001) | — |
| **TOTAL Go-Live (P0 + P1)** | **108 h** | **6-11 k€ one-shot** | **~330-530 €/mois récurrent** |

### 10.2 Timeline détaillée (Gantt simplifié)

```
SEMAINE 1 (J1-J7)  — P0 Quick wins
  ├─ J1-J2: Brief avocat (compliance/avocat_brief.md)
  ├─ J1-J2: Devis 3 cabinets fintech FR
  ├─ J3-J4: Souscription RC Pro Hiscox + Cyber Stoïk
  ├─ J5-J6: DPA Anthropic email support
  └─ J6-J7: Registre Art. 30 RGPD v1

SEMAINE 2 (J8-J14) — P0 Code MiFID + DSAR setup
  ├─ J8-J10: Refactor InsightSignalV2 + disclosure_mode
  ├─ J11-J12: DSAR endpoints squelette (GET /me/data)
  └─ J13-J14: Stripe terms_accepted_at gate

SEMAINE 3 (J15-J21) — P0 Data licensing + Cookie banner
  ├─ J15-J17: Migration FF → Trading Economics
  ├─ J18-J19: Note Dukascopy internal-only + register
  └─ J20-J21: Tarteaucitron landing

SEMAINE 4 (J22-J30) — P0 Avocat mission + DSAR finish
  ├─ J22-J24: RDV cabinet sélectionné + brief
  ├─ J25-J27: DSAR DELETE /me + anonymisation
  └─ J28-J30: Tests E2E + intégration corrections avocat

MOIS 2 (J31-J60) — P1 Polish
  ├─ S5: DPA template B2B + endpoint subprocessors
  ├─ S6: Médiation conso adhésion (CM2C)
  ├─ S7: DPO assessment + chiffrement at-rest SQLCipher
  └─ S8: Tests régression compliance + monitoring KPIs

MOIS 3 (J61-J90) — P1 Consolidation
  ├─ S9-S10: Audit interne KPIs compliance + roadmap M4+
  ├─ S11: Veille MiFID 2024/2811 application réelle
  └─ S12: Décision SOC2 / pas SOC2 (selon MRR B2B)

MOIS 4-12 — P2 SOC2 Type 1 (si MRR B2B >$10k)
MOIS 12+  — P3 SOC2 Type 2 + ISO 27001 prep
```

### 10.3 Critères go-live commercial (gate final)

- [ ] CGU + Privacy signées avocat FR (P0 W4 done).
- [ ] `compliance/data_licensing_register.md` listant 100 % sources avec statut commercial.
- [ ] `disclosure_mode=qualitative` default + opt-in numeric fonctionnel.
- [ ] DSAR endpoints `GET /me/data` + `DELETE /me` opérationnels.
- [ ] Cookie banner Tarteaucitron sur landing.
- [ ] Stripe `terms_accepted_at` gate actif + version stamp.
- [ ] RC Pro 1M€ + Cyber 250k€ souscrites.
- [ ] DPA Anthropic + Stripe + Trading Economics signés.
- [ ] 80+ tests compliance verts.
- [ ] Note avocat "ok pour commercialisation B2C FR/EN/DE/ES, hors US/UK/QC/OFAC".

Sans **8/10 cases cochées**, ne pas activer la facturation Stripe public.

---

## Résumé exécutif

**Chemin du livrable.** `C:\MyPythonProjects\TradingBOT_Agentic\reports\commercialization_sprint\18_compliance_legal.md`

**Top 3 P0.**
1. **W4 relecture juridique CGU/Privacy** par cabinet fintech FR (12 h interne + 3-5 k€ externe) — bloquant Stripe + AMF.
2. **DSAR endpoints `GET /me/data` + `DELETE /me`** avec anonymisation signal_store (24 h interne) — obligation RGPD non-négociable.
3. **Data licensing remediation** : migrer ForexFactory → Trading Economics ($79/mois) + restreindre Dukascopy internal-only (24 h interne + 80 €/mois récurrent).

**Heures internes totales P0+P1 (J1-J90).** ~108 h (74 P0 + 34 P1).

**Budget externe estimé.** **6-11 k€ one-shot** (avocat + RC Pro + Cyber initial) + **~330-530 €/mois récurrent** (TE + assurances + médiation + Cookiebot optionnel). **SOC2 Type 1 + ISO 27001 différés P2-P3** : +15-25 k$ supplémentaires sur 12-24 mois si ventes B2B institutionnelles confirmées.

