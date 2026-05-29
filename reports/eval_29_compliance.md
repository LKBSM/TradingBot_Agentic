# Eval 29 — Compliance, Disclaimers, Données & Régulation

> **DISCLAIMER OBLIGATOIRE — CECI N'EST PAS UN AVIS JURIDIQUE.**
> Ce rapport est produit par un agent IA non-avocat (Claude Opus 4.7) et constitue **une checklist de pré-cadrage** destinée à être soumise à un avocat fintech qualifié dans chaque juridiction de distribution. Il ne remplace ni un audit RGPD, ni une consultation auprès du barreau de Paris (CIF), ni un avis "no-action letter" SEC, ni une opinion FCA / ASIC / AMF QC. Les références de textes (articles AMF, ESMA Q&A, SEC IA Act, etc.) sont citées avec n° et date pour vérification mais peuvent évoluer ; toute publication commerciale, paiement, ou incorporation en dépendant **exige validation par un cabinet enregistré** au préalable.
>
> **Périmètre audité** : exposition légale et compliance d'un SaaS de signaux trading IA (Smart Sentinel AI), fondateur solo Loukmane Bessam (loukmanebessam@gmail.com), résidence présumée **France métropolitaine** (à confirmer auprès de l'auteur — détermine la qualification AMF-CIF, la compétence URSSAF/SIRET, et la nationalité du DPO si requis).
>
> **Date** : 2026-04-26 · **Branch** : `main` · **Snapshot** : 632e9dd + Sprint 3 narrative.

---

## 0. TL;DR — Note d'exposition

| Dimension | Note /10 | Justification |
|-----------|----------|---------------|
| Qualification juridique (signal éducatif vs conseil) | **3** | Description actuelle "AI-powered narratives" + recommandations chiffrées (entry/SL/TP) + `signal_type=BUY/SELL` glissent vers "personalised investment advice" au sens AMF & MiFID II — risque de requalification CIF. |
| Disclaimer multi-juridiction | 4 | Phrase _"Smart Sentinel AI — Not financial advice"_ présente (Telegram L142, Discord L154, LLM system prompt L80) — minimaliste, mono-langue (EN), sans renvoi juridiction. Insuffisant FR/EU/US/UK/CA/AU. |
| RGPD (DPA Anthropic, transferts UE→US) | 4 | Aucune CCT 2021/914/UE signée localement, aucun registre traitements visible, dépendance Anthropic US (Schrems II + DPF), pas de politique de rétention sur `signal_store.py`. |
| Licences data (Dukascopy / MT5 / FF) | 5 | Dukascopy gratuit non-commercial only ; MT5 broker feed CGU dépend du broker ; ForexFactory scraping = zone grise. Risque cease-and-desist faible mais réel. |
| CGU / CGV SaaS B2C | 2 | Aucune CGU/CGV identifiée dans le repo. `GET /api/v1/terms` mentionné dans `SPRINT_ROADMAP.md:1265` mais pas implémenté. Bloquant pour go-live payant. |
| PSD2 / KYC | 7 | Choix recommandé déléguer Stripe/Paddle (qui portent la conformité DSP2/PSP) ⇒ exposition limitée. À confirmer si abonnement direct envisagé. |
| Assurances (RC Pro + Cyber) | 3 | Aucune souscription RC Pro fintech / Cyber identifiée. Coût estimé 1.2-3.5 k€/an France (Hiscox / AIG / Stoïk). |
| Restrictions géographiques | 1 | Aucun IP geo-blocking implémenté. Accessible **mondialement** dont US/QC/sanctionnés OFAC ⇒ risque maximal. |
| **GLOBAL EXPOSITION** | **3.5 / 10** | « POC qui peut tourner en TESTING_MODE entre amis, mais ne peut PAS être facturé ni publicly listed avant livrables P1-P5 (ci-dessous). » |

---

## 1. Cartographie des risques juridiques

```
┌──────────────────────────────────────────────────────────────────────┐
│  Smart Sentinel AI — Surface d'exposition régulatoire (2026-04-26)   │
└──────────────────────────────────────────────────────────────────────┘

  ┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
  │ Qualification   │     │ Diffusion        │     │ Données          │
  │ produit         │     │ commerciale      │     │ personnelles     │
  ├─────────────────┤     ├──────────────────┤     ├──────────────────┤
  │ • CIF (AMF FR)  │     │ • CGU/CGV B2C    │     │ • RGPD UE        │
  │ • IA (MiFID II) │     │ • Disclaimer×lang│     │ • DPA Anthropic  │
  │ • RIA (SEC US)  │     │ • Geo-blocking   │     │ • CCT 2021/914   │
  │ • CIF (FCA UK)  │     │ • OFAC/FATF      │     │ • DPF UE-US      │
  │ • AMF Québec    │     │ • Email marketing│     │ • Droit oubli    │
  │ • AFS (ASIC AU) │     │ (LCEN, CAN-SPAM, │     │ • Registre Art.30│
  │                 │     │  PECR UK)        │     │                  │
  └─────────────────┘     └──────────────────┘     └──────────────────┘
            │                       │                        │
            └───────────────────────┴────────────────────────┘
                                    │
                  ┌─────────────────┴──────────────────┐
                  │ Données externes & contrats        │
                  ├────────────────────────────────────┤
                  │ • Dukascopy historical (CGU)       │
                  │ • MT5 broker feed (broker-spec)    │
                  │ • ForexFactory calendar (scraping) │
                  │ • Anthropic API US (DPA)           │
                  │ • Telegram / Discord (PII relay)   │
                  │ • Stripe / Paddle (PSD2 délégué)   │
                  └────────────────────────────────────┘
                                    │
                  ┌─────────────────┴──────────────────┐
                  │ Couverture risque résiduel         │
                  │ • RC Pro fintech                   │
                  │ • Cyber (rançongiciel, fuite PII)  │
                  │ • Pertes financières clients (E&O) │
                  └────────────────────────────────────┘
```

**Lecture** : 4 axes parallèles, chacun avec son propre cycle de remédiation (jours pour disclaimer, semaines pour CGU avocat, mois pour DPA Anthropic enterprise).

---

## 2. Qualification juridique — signal éducatif vs conseil personnalisé

### 2.1 Critères AMF (France)

**Texte de référence** : Code monétaire et financier, **art. L.541-1** (définition CIF) ; **règlement général AMF, livre III**, instruction DOC-2013-10 ; position-recommandation **DOC-2014-04** "Commercialisation des instruments financiers complexes" ; communication AMF **2018 sur la publication de signaux de trading sur les réseaux sociaux** (visa "finfluencers").

Un service tombe sous **CIF (Conseiller en Investissements Financiers)** quand 3 critères sont **cumulativement** remplis :
1. **Recommandation personnalisée** (vs information générale) — adressée à une personne en sa qualité d'investisseur, présentée comme adaptée à sa situation.
2. Porte sur un **instrument financier au sens MiFID II** (actions, ETF, FX OTC sous MiFID 2018, dérivés). XAU/USD spot **n'est pas** instrument financier MiFID — mais un CFD ou future XAU/USD l'est.
3. **Résultat d'une analyse personnelle** par le prestataire.

**Position défendable Smart Sentinel** :
- Le signal est généré **algorithmiquement et systématiquement** sur des bars OHLCV publiques, identique pour tous les abonnés du même tier.
- Aucune connaissance du **profil investisseur** (KYC, situation patrimoniale, objectifs) n'est collectée → critère "personnalisé" non rempli.
- Le produit est positionné **éducatif / market intelligence** (cf. `BUSINESS_PLAN_SMART_SENTINEL.md:117`).

⇒ **Position défendable mais fragile**. La requalification AMF est probable si :
- Le signal contient une **directive d'action** ("BUY now at 2050.00, SL 2045.00, TP 2065.00") — c'est exactement le format actuel.
- Le marketing parle de **"recommandations"**, **"trades"**, **"positions"** plutôt que **"analyse"**, **"observation"**, **"setup"**.
- Le tier INSTITUTIONAL inclut un **chat Q&A** où le LLM répond à des questions individualisées (présent dans `src/api/routes/narratives.py`).

### 2.2 Critères ESMA / MiFID II (Europe)

**Texte** : Directive **2014/65/UE (MiFID II)**, **art. 4(1)(4)** définition "investment advice" ; **ESMA Q&A on investor protection** (ESMA35-43-349) ; **règlement délégué 2017/565**, art. 9 critères de "personnal recommendation".

ESMA Q&A 2024 (Section 4 Q&A 3) précise : **un signal trading distribué publiquement** (canal Telegram broadcast, API publique non-personnalisée) **n'est pas** un conseil en investissement *si* :
- Aucune segmentation par profil client.
- Aucune mention "this is suitable for you".
- Disclaimer clair "general information / not personalised".

**Risque résiduel** : la **directive 2024/2811 (révisant MiFID II sur les "finfluencers")** entre en vigueur **mars 2026** — durcit le critère "présenté comme adapté" dès que l'output mentionne **stop-loss et take-profit chiffrés** (interprété comme actionnable).

⇒ Smart Sentinel actuellement **dans le périmètre du durcissement 2026**. Mitigation : présenter SL/TP comme **"levels observed by algorithm"**, pas **"recommended exits"**.

### 2.3 Critères SEC (États-Unis)

**Texte** : **Investment Advisers Act of 1940, §202(a)(11)** — définition "investment adviser" ; **Lowe v. SEC (1985, 472 U.S. 181)** — exception "publisher" pour publications **bona fide, regular, of general circulation** ; **Persons Providing Investment Advice About Securities (Release No. IA-1092, 1987)** — guidance interprétation ; **Rule 203A** seuils enregistrement ($110M AUM ou state-level <$100M).

**Trois éléments cumulatifs** définissant un Investment Adviser :
1. Fournit des conseils sur **securities** (XAU spot et FX spot **ne sont pas** securities ; CFDs sur stocks le sont ; futures sont CFTC pas SEC).
2. **Engaged in the business** (récurrent, contre rémunération).
3. **For compensation**.

**Lowe v. SEC exception "publisher"** : un newsletter/SaaS échappe à l'enregistrement RIA si :
- **Bona fide publication** (pas vehicle déguisé).
- **Of general and regular circulation** (broadcast, pas one-on-one).
- **Not custom-tailored** au client individuel.
- **Pas de "hot tips"** liés à événements de marché spécifiques.

**Position Smart Sentinel** :
- ✅ Broadcast Telegram = circulation générale.
- ✅ Bars systématiques = régulier.
- ⚠️ **"Hot tips" sur événements** (FOMC blackout, news high-impact) — le module `news_analysis_agent.py` réagit à actualités spécifiques ⇒ risque de qualification **"market timing tips"** que la SEC traite plus durement.
- ⚠️ Tier INSTITUTIONAL avec chat Q&A → custom-tailored ⇒ casse l'exception Lowe.

**Seuils RIA** : si AUM = $0 (pas de mandat de gestion), Smart Sentinel n'a **pas** besoin d'enregistrement RIA fédéral. Mais **state-level "Investment Adviser Representative" registration** peut s'appliquer (NY BitLicense équivalent NJ, CA, MA — particulièrement strictes).

**Risque concret US** : **bloquer entièrement le marché US** est l'option la plus prudente jusqu'à no-action letter ou opinion sec-counsel.

### 2.4 Critères FCA (Royaume-Uni)

**Texte** : **Financial Services and Markets Act 2000, s.21** (financial promotions) ; **FCA PERG 8** (Perimeter Guidance manual) ; **FCA Handbook COBS 9** (suitability) ; **Financial Promotion Order 2005 (SI 2005/1529)**.

UK post-Brexit a **renforcé** le régime financial promotion (oct. 2023) :
- Toute communication B2C UK promouvant un produit financier doit être **approuvée par une firme FCA-authorised** (Section 21 approval).
- Crypto et CFDs sous régime durci 2024 — **interdiction promotion grand public** sans warnings standardisés et 24h cooling-off.

⇒ Smart Sentinel diffusant signaux XAU/CFD à des résidents UK = **financial promotion** → soit **enregistrement FCA**, soit **approval gateway** (firm tierce qui valide promo, ~£500-2000/mois), soit **bloquer UK**.

### 2.5 Critères AMF Québec (Canada)

**Texte** : **Loi sur les valeurs mobilières du Québec, art. 148** (inscription comme conseiller en valeurs) ; **Règlement 31-103** (obligations d'inscription) ; **Avis 31-352** sur l'usage de réseaux sociaux pour distribution de signaux.

Québec a régime **provincial autonome** distinct ROC (Regulator of Canada via OCRI). Tout signal trading payant à un résident québécois nécessite **inscription** sauf exemption "publisher". **AMF QC plus strict** que SEC : exemption publisher rarement accordée si signaux contiennent SL/TP.

**Recommandation** : bloquer Québec (geo-IP) en attendant opinion d'avocat québécois.

### 2.6 Critères ASIC (Australie)

**Texte** : **Corporations Act 2001, s.766B** (financial product advice) ; **ASIC Regulatory Guide RG 36** (general advice vs personal advice) ; **RG 234** (advertising financial products) ; **Regulatory Guide RG 274** Sept 2021 sur **DDO (Design and Distribution Obligations)**.

Distinction "**general advice**" (autorisée sans AFSL si disclaimer warning) vs "**personal advice**" (AFSL obligatoire). Australie **strict** sur DDO 2021 : tout produit financier requiert un **TMD (Target Market Determination)** publié.

⇒ Australie : faisable avec **General advice warning** ASIC-compliant + TMD publié. Coût AFSL solo ~AU$ 50k initial + 30k/an.

### 2.7 Synthèse multi-juridiction

| Juridiction | Régulateur | Faisabilité solo founder | Action recommandée |
|-------------|-----------|--------------------------|--------------------|
| **France (résidence)** | AMF | Faisable avec disclaimer renforcé + reformulation produit | Reformuler "signaux" → "analyses algorithmiques", ne PAS s'enregistrer CIF (frais 1k€/an + obligations) tant que <500 abonnés payants |
| **UE (passporting)** | ESMA + national | Faisable, attention durcissement 2026 finfluencers | Disclaimer ESMA-compliant par langue, cf. §3 |
| **UK** | FCA | Difficile post-Brexit | **Bloquer UK** ou s21 approval via firm tierce (~£1k/mois) |
| **US** | SEC + state IA | Risqué (Lowe exception fragile si chat Q&A) | **Bloquer US entier** jusqu'à no-action letter ou opinion |
| **Québec** | AMF QC | Très difficile | **Bloquer Québec** geo-IP |
| **Canada hors-QC** | OCRI provinces | Faisable avec disclaimer | Disclaimer EN/FR avec mention "general info" |
| **Australie** | ASIC | Faisable mais coûteux (DDO + TMD) | Reporter post-PMF |
| **Singapour / Hong Kong** | MAS / SFC | Strictes (licences nécessaires souvent) | Reporter post-PMF |
| **Reste du monde** | Variable | Cas par cas | Disclaimer générique, monitorer croissance |

---

## 3. Disclaimer "Not financial advice" — par juridiction

### 3.1 État actuel dans le code

| Fichier | Ligne | Contenu | Suffisant ? |
|---------|-------|---------|-------------|
| `src/delivery/telegram_notifier.py` | 142 | `_Smart Sentinel AI — Not financial advice_` | ❌ trop court, pas de juridiction, pas de loss warning, mono-langue EN |
| `src/delivery/discord_notifier.py` | 154 | `Smart Sentinel AI — Not financial advice` | ❌ idem |
| `src/intelligence/llm_narrative_engine.py` | 80 | "Never give financial advice. Frame as educational analysis" | ✅ niveau prompt, mais pas vu par utilisateur final |
| `src/api/routes/` | n/a | **Aucune route `/terms` implémentée** (mentionnée roadmap L1265) | ❌ **bloquant go-live payant** |
| Landing page | n/a | Inexistante dans le repo | ❌ |

### 3.2 Templates par juridiction (à valider avocat)

#### EN — UE / Generic / US-blocked

```
RISK WARNING / NOT INVESTMENT ADVICE
Smart Sentinel AI publishes algorithmic market analysis for educational purposes
only. Outputs are general information, NOT personalised investment advice within
the meaning of MiFID II Art. 4(1)(4) (EU) or the Investment Advisers Act 1940
§202(a)(11) (US — service not available in the United States and Canadian Province
of Quebec). No content constitutes a recommendation, solicitation, or offer to
buy or sell any financial instrument. Trading involves substantial risk of loss
and is not suitable for every investor. Past algorithmic performance does not
guarantee future results. By using this service you acknowledge that you trade
at your own risk and have read our Terms (https://[domain]/terms) and Privacy
Policy (https://[domain]/privacy).
```

#### FR — France / Belgique / Suisse / Canada FR

```
AVERTISSEMENT / INFORMATION GÉNÉRALE — PAS UN CONSEIL EN INVESTISSEMENT
Smart Sentinel AI publie des analyses algorithmiques de marché à finalité
éducative. Les contenus diffusés constituent une information générale et NE
SONT PAS un conseil en investissement personnalisé au sens de l'article L.541-1
du Code monétaire et financier (France) ni au sens de la directive MiFID II
2014/65/UE. Smart Sentinel AI n'est pas inscrit comme Conseiller en
Investissements Financiers (CIF) auprès de l'AMF et n'effectue aucune
recommandation personnalisée. Le trading sur instruments financiers comporte un
risque substantiel de perte en capital et n'est pas adapté à tous les
investisseurs. Les performances algorithmiques passées ne préjugent pas des
performances futures. En utilisant ce service, vous reconnaissez trader à vos
risques et avoir consulté nos CGU (https://[domain]/cgu) et notre politique de
confidentialité (https://[domain]/confidentialite).
Service non disponible aux résidents du Québec ni des États-Unis.
```

#### DE — Allemagne / Autriche / Suisse-DE

```
RISIKOHINWEIS / KEINE ANLAGEBERATUNG
Smart Sentinel AI veröffentlicht algorithmische Marktanalysen ausschließlich zu
Bildungszwecken. Die Inhalte sind allgemeine Informationen und stellen KEINE
personalisierte Anlageberatung im Sinne von §1 Abs. 1a Nr. 1a KWG (DE) bzw.
MiFID II 2014/65/EU dar. Der Handel mit Finanzinstrumenten birgt erhebliche
Verlustrisiken. Vergangene algorithmische Performance ist kein Indikator für
zukünftige Ergebnisse. Mit der Nutzung des Dienstes erkennen Sie unsere AGB
(https://[domain]/agb) und Datenschutzerklärung an.
```

#### ES — Espagne / LATAM

```
ADVERTENCIA DE RIESGO / NO ES ASESORAMIENTO FINANCIERO
Smart Sentinel AI publica análisis algorítmicos de mercado con fines puramente
educativos. Los contenidos son información general y NO constituyen
asesoramiento personalizado en el sentido del Art. 4(1)(4) MiFID II (UE) ni de
la Ley del Mercado de Valores española. La operación con instrumentos
financieros conlleva un riesgo sustancial de pérdida y no es adecuada para
todos los inversores. El rendimiento algorítmico pasado no garantiza
resultados futuros.
```

### 3.3 Implementation matrix

| Surface | Disclaimer requis | Action concrète |
|---------|-------------------|-----------------|
| Telegram (chaque message) | Court 1 ligne avec lien `/terms` | Modifier `telegram_notifier.py:142` : `_Educational analysis. Not investment advice. Terms: [domain]/terms_` |
| Discord (chaque embed) | Court footer + lien | Modifier `discord_notifier.py:154` |
| API REST (chaque réponse `/api/v1/signals`) | Champ `disclaimer` dans `SignalResponse` model | Ajouter dans `src/api/models.py` un champ obligatoire serialise depuis `config.DISCLAIMER_PER_LANG` |
| Landing page | Long disclaimer + risk warning above-the-fold | À créer (hors scope code) |
| Email marketing | CAN-SPAM (US), LCEN (FR), CASL (CA) — opt-out + adresse physique | Si Stripe/Paddle gère emailing transactionnel, OK ; marketing email = à externaliser Brevo / Mailchimp avec compliance built-in |
| Onboarding (signup) | Acceptation explicite CGU + Privacy + cookies (RGPD) — checkbox non pré-cochée | À implémenter avec auth flow |

---

## 4. RGPD — Audit et plan d'action

### 4.1 Données traitées (recensement)

| Donnée | Source | Finalité | Base légale (art. 6 RGPD) | Durée conservation |
|--------|--------|----------|----------------------------|-----------------------|
| Email utilisateur | Signup | Authentification, livraison Telegram | Exécution contrat (6.1.b) | Vie compte + 3 ans |
| Telegram chat ID | Onboarding | Livraison signaux | Exécution contrat | Vie compte |
| Discord webhook URL | Onboarding | Livraison signaux | Exécution contrat | Vie compte |
| API key | Auto-généré | Authentification | Exécution contrat | Vie compte |
| Adresse IP | Logs | Sécurité, geo-blocking | Intérêt légitime (6.1.f) | 12 mois (LCEN) |
| Logs requêtes API | Middleware | Debug, sécurité | Intérêt légitime | 12 mois |
| Stripe customer ID | Stripe webhook | Facturation | Obligation légale (6.1.c) | 10 ans (compta FR) |
| Tier subscription | DB | Gating features | Exécution contrat | Vie compte |
| Historique signaux livrés | `signal_store.py` SQLite | Traçabilité, support | Intérêt légitime | À définir — recommandé 12 mois après livraison |

⚠️ **Manquant** : Aucun **registre des activités de traitement** (Art. 30 RGPD) écrit. Solo founder < 250 employés = exemption partielle, **MAIS** traitement régulier de données personnelles + transfert hors-UE = **registre obligatoire** quand-même.

### 4.2 Transferts hors-UE

**Anthropic Inc. (US)** est destinataire indirect de données personnelles potentiellement contenues dans :
- Les **chat queries** (tier INSTITUTIONAL) — si l'utilisateur tape "I have $50k portfolio, should I…", PII embarqué.
- Les **signal payloads** envoyés à l'API Anthropic — pas de PII en théorie (juste OHLCV + scores), à confirmer par audit code `llm_narrative_engine.py:_signal_to_csv()`.

**Cadre légal applicable** :
1. **CCT 2021/914/UE** (Clauses Contractuelles Types décision Comm. Européenne 4 juin 2021) — module 2 (responsable→sous-traitant), à signer entre Smart Sentinel (controller) et Anthropic (processor).
2. **EU-US Data Privacy Framework (DPF)** entré en vigueur 10 juillet 2023 (décision d'adéquation 2023/1795). Anthropic Inc. est-il **certifié DPF** ? À vérifier sur https://www.dataprivacyframework.gov/list. *Si oui*, transfert présume adéquat ⇒ CCT optionnelle (mais recommandée en couche défensive vs Schrems III hypothétique).
3. **Schrems II (CJUE C-311/18, 16 juill. 2020)** — invalidation Privacy Shield. DPF est successeur juridique mais fragile (recours en cours devant CJUE prévus 2026-2027).

**Action concrète** :
1. Vérifier Anthropic DPF certification (probable mais pas vérifié dans cet audit — agent sans accès web).
2. Demander à Anthropic le **DPA enterprise** (Anthropic propose DPA standard pour clients API au-dessus d'un certain volume — solo dev peut être éligible via formulaire support).
3. Documenter dans politique de confidentialité : "Vos données peuvent être transférées vers les États-Unis dans le cadre du DPF UE-US et des CCT 2021/914".

### 4.3 Droits des personnes concernées

| Droit RGPD | État actuel | Implémentation requise |
|-----------|--------------|------------------------|
| Accès (art. 15) | ❌ Aucun endpoint | `GET /api/v1/me/data` retournant export JSON de toutes les données |
| Rectification (art. 16) | Partiel (changement email) | OK si signup permet update |
| Effacement / oubli (art. 17) | ❌ Aucun endpoint | `DELETE /api/v1/me` + purge SQLite + révocation Telegram chat ID |
| Portabilité (art. 20) | ❌ | Idem droit d'accès, format JSON/CSV |
| Opposition (art. 21) | ❌ | Toggle marketing emails dans settings |
| Limitation (art. 18) | ❌ | "Pause" compte qui désactive scanner sans suppression |

**Question subtile sur signaux historiques** : SQLite `signal_store` contient l'historique signaux LIVRÉS au compte. Quand utilisateur exerce droit à l'oubli :
- **Option A** : suppression complète → perte de traçabilité comptable / compliance.
- **Option B** : **anonymisation** (remplacer `user_id` par hash irréversible, garder métriques agrégées). Conforme art. 17 §3.b RGPD (intérêt à conserver à des fins statistiques). Recommandé.

### 4.4 Politique cookies & traceurs

Si landing page (à créer) utilise **Google Analytics**, **Hotjar**, **Meta Pixel**, **Stripe.js** : **bandeau cookies CNIL-compliant** obligatoire (refus aussi facile que accepter, pas de pre-tick, granularité par finalité).

**Solution simple solo founder** : utiliser **Plausible Analytics** (cookieless, RGPD-friendly, hébergé UE) → pas de bandeau requis pour analytics.

### 4.5 DPO / Représentant UE

Solo founder résidant France :
- **DPO non obligatoire** sauf si traitements à grande échelle ou catégories spéciales art. 9. → Probablement non.
- **Représentant UE non requis** (établissement déjà UE).
- **Notification CNIL** non requise pour activité standard (suppression de la déclaration préalable depuis 2018).

⚠️ Si le founder déménage hors-UE : représentant UE obligatoire (art. 27 RGPD) — coût ~50-200 €/mois (DataRep, EuroPaper, Prighter).

---

## 5. Licences data — usage commercial

### 5.1 Dukascopy

**CGU Dukascopy historical data** (https://www.dukascopy.com/swiss/english/marketwatch/historical/) :
- **Free for personal use**, **commercial use requires written agreement**.
- Téléchargements via JForex API ou web tools = même CGU.
- Le repo contient `scripts/download_dukascopy_xau.py` qui scrape leur CDN historique tick.

**Risque** : si Smart Sentinel facture des abonnés en utilisant des features dérivées (volatilité historique, backtests publiés) **calculées sur Dukascopy**, c'est un usage **commercial**.

**Mitigation** :
1. **Demander un commercial license** à Dukascopy Bank SA (Genève) — coût indicatif €200-1000/mois selon volume.
2. **OU** migrer vers un fournisseur explicitement commercial : Polygon.io ($199/mois starter), Tiingo ($30/mois starter), Databento, Norgate Data.
3. **OU** restreindre Dukascopy data au **dev/backtest interne** uniquement (jamais servi à un user payant en live).

### 5.2 MetaTrader 5 (MT5) broker feed

**Texte** : MetaQuotes EULA + CGU broker spécifique (IC Markets, Pepperstone, FTMO, etc.).

Le repo a `scripts/export_mt5_history.py` qui exporte historiques bars depuis terminal MT5.
- Données **temps réel** = propriété du broker, redistribution interdite par CGU broker (uniformément).
- Données **historiques** = même contrainte généralement.

**Mitigation** : utiliser MT5 **uniquement pour exécution / signal validation interne**, ne **pas redistribuer les bars** comme service. Si on publie sur Telegram "EUR/USD à 1.0850", c'est de la rediffusion de prix → litigieux mais marché tolère (les prix de marché ne sont pas copyrightables). Citer la **source generic** ("market price as observed") plutôt que "MT5 / IC Markets feed".

### 5.3 ForexFactory

**Texte** : ForexFactory CGU (https://www.forexfactory.com/disclaimer) — usage personnel autorisé, scraping interdit, redistribution interdite.

Le repo a `scripts/fetch_forexfactory_live.py` qui télécharge leur calendrier économique JSON.
- **Risque cease-and-desist** : ForexFactory connu pour blocker IP scrapers (Cloudflare).
- **Risque DMCA / rediffusion** : si Smart Sentinel publie "FOMC à 19h00" en se basant sur FF, faible (les événements eux-mêmes ne sont pas copyrightables, seules les annotations FF le sont).

**Mitigation** :
1. Migrer vers **Trading Economics API** (commercial, $79-499/mois selon plan).
2. **OU** **Investing.com Calendar API** (via partenariat).
3. **OU** **Federal Reserve / ECB / BLS direct feeds** (gratuits, FRED API, Eurostat) — plus de friction technique mais légalement clean.
4. **OU** rester ForexFactory mais avec **back-off respectueux** (≥ 1 req/heure, User-Agent identifiant) et préparer plan B si bloqué.

### 5.4 Anthropic API

**DPA Anthropic Commercial Terms** (https://www.anthropic.com/legal/commercial-terms) :
- Output **non training par Anthropic** sur clients API payants (politique 2024).
- DPA standard inclus pour comptes API.
- Pas de revendication propriété intellectuelle sur outputs (l'utilisateur owns les outputs).
- ⚠️ **Restriction** : ne pas utiliser pour décisions à haut risque (santé, justice, "high-stakes financial advice"). **Section "Usage Policies"** à relire — la mention "financial advice" est listée comme usage encadré (pas interdit mais avec disclaimer obligatoire à l'utilisateur final).

**Action** : ajouter dans Privacy Policy : "Nous utilisons Anthropic Claude pour générer les explications. Les requêtes envoyées à Anthropic sont anonymisées (pas d'email/ID utilisateur). Anthropic ne s'entraîne pas sur ces données. DPA disponible sur demande."

---

## 6. CGU / CGV — Structure recommandée

### 6.1 État actuel

**Aucune CGU / CGV / Privacy Policy / Cookie Policy** n'existe dans le repo. C'est **bloquant** pour go-live payant (Stripe refuse onboarding marchand sans Terms publics).

### 6.2 Structure CGU SaaS B2C — table des matières recommandée

```
1. Préambule et définitions
2. Objet du service (educational analysis, NOT advice)
3. Inscription et compte utilisateur
   3.1 Conditions (âge >= 18, capacité juridique)
   3.2 Vérification email
   3.3 Pseudonymes interdits si payant (KYC light pour facturation)
4. Description des tiers (FREE / ANALYST / STRATEGIST / INSTITUTIONAL)
5. Modalités financières
   5.1 Prix par tier
   5.2 Paiement (Stripe / Paddle — délégué)
   5.3 Renouvellement automatique + désabonnement art. L.215-1 Code Conso
   5.4 Droit de rétractation 14 jours art. L.221-18 Code Conso
       (avec exception "service entièrement exécuté" si activé immédiat)
6. Disclaimer & limitations de responsabilité
   6.1 Pas de conseil personnalisé (cf. §2 du présent rapport)
   6.2 Risque trading
   6.3 Pas de garantie résultats
   6.4 Plafond responsabilité = montant abonnement annuel
7. Propriété intellectuelle
   7.1 Smart Sentinel détient algorithmes
   7.2 Utilisateur a licence d'usage non-exclusive
8. Données personnelles → renvoi Privacy Policy
9. Suspension et résiliation
10. Force majeure (incl. panne Anthropic, panne broker feed)
11. Médiation conso (art. L.612-1 Code Conso) — Médiateur de la consommation FR obligatoire pour B2C
12. Droit applicable + juridiction (recommandé: tribunaux français + droit français)
13. Modifications CGU (préavis 30j)
```

### 6.3 Structure Privacy Policy (RGPD)

```
1. Identité du responsable (Loukmane Bessam, [adresse], [email DPO])
2. Données collectées (cf. §4.1)
3. Finalités (cf. §4.1)
4. Bases légales (cf. §4.1)
5. Destinataires (Anthropic US, Telegram, Stripe, hébergeur)
6. Transferts hors-UE (Anthropic + DPF + CCT)
7. Durées de conservation
8. Vos droits (accès, rectification, effacement, portabilité, opposition, limitation)
9. Comment exercer vos droits (email, délai 30j)
10. Réclamation auprès CNIL (https://www.cnil.fr/fr/plaintes)
11. Cookies (renvoi Cookie Policy si applicable)
12. Modifications de cette politique
```

### 6.4 Cabinets recommandés (FR, fintech / SaaS)

⚠️ **À titre purement indicatif. Le founder doit vérifier réputation, disponibilité et tarifs auprès du Barreau de Paris ou son barreau local.**

1. **Reed Smith Paris** (équipe fintech, gros budget : 500-800 €/h).
2. **De Gaulle Fleurance & Associés** (régulation financière, expérience CIF/AMF — budget moyen, 350-500 €/h).
3. **Aramis Société d'Avocats** (SaaS B2C, RGPD — adapté solo founder, 250-400 €/h).
4. **Alternative low-cost** : **Captain Contrat** (https://www.captaincontrat.com) ou **Legalstart** pour CGU/Privacy templates (~150-300 € one-shot, pas équivalent avocat mais OK pour démarrer ; revoir avec avocat avant >100 abonnés payants).

**Budget réaliste solo founder** :
- CGU + Privacy + Cookie : **2-4 k€** chez avocat junior fintech, **800 € + 2h consult** chez Captain Contrat + revue ciblée.
- Audit qualification CIF (note de cadrage) : **3-5 k€**.
- Setup multi-juridictions complet : **8-15 k€**.

---

## 7. PSD2 / KYC — Décision paiements

### 7.1 Contexte

PSD2 (**Directive UE 2015/2366**) régule les services de paiement. Tout encaissement direct = **prestataire de services de paiement (PSP)** → enregistrement ACPR (FR), licence DSP2.

### 7.2 Options

| Option | Conformité PSD2 | KYC | Complexité founder | Recommandé ? |
|--------|------------------|-----|---------------------|----------------|
| **Stripe** | Délégué Stripe (PSP US/IRL agréé ACPR) | Stripe gère KYC compte, pas KYC client final pour B2C SaaS sub | Faible | **OUI** |
| **Paddle** | Paddle = **Merchant of Record** → Paddle facture, Smart Sentinel reçoit revenu net | Délégué Paddle | Très faible (Paddle gère TVA EU/US/UK aussi) | **OUI alternative** |
| **Lemon Squeezy** | Idem Paddle | Délégué | Très faible | OUI alternative startup-friendly |
| Encaissement direct (virement / SEPA) | **Smart Sentinel devient PSP** | KYC client = obligatoire AML/CFT | Très élevé (licence) | **NON** |
| Crypto USDC/USDT | Zone grise réglementaire 2026 | KYC selon volume + MiCA UE 2024 | Moyenne | Pas avant PMF |

### 7.3 Recommandation

**Paddle (Merchant of Record)** pour démarrage :
- TVA gérée (gros gain pour solo founder vendant cross-border).
- Pas de souci PSD2.
- Pas de souci KYC client final (Paddle n'en demande pas pour SaaS sub < $1k/mois/user).
- Frais : 5% + $0.50 par transaction (vs Stripe 1.4-2.9% + 0.25€). Plus cher mais simpler.

**Stripe** si volume > 50k€/mois (frais marginaux meilleurs) — mais nécessite gérer TVA OSS soi-même (faisable mais 2-3 jours setup comptable).

### 7.4 KYC abonnés

**B2C SaaS < 1k€/mois/user** : pas de KYC obligatoire (pas un service financier au sens DSP2).
**B2B INSTITUTIONAL > 5k€/mois** : recommandé KYC light (vérification entreprise via SIREN, signature DPA mutuel) pour éviter blanchiment et limiter risque réputationnel.

---

## 8. Assurances

### 8.1 RC Pro Fintech (responsabilité civile professionnelle)

**Couvre** :
- Faute professionnelle (bug algo qui cause perte massive client → réclamation).
- Frais de défense juridique.
- Dommages immatériels consécutifs.

**Courtiers / assureurs FR** :
- **Hiscox** (leader fintech FR, devis en ligne) : **1 500-3 500 €/an** pour SaaS solo CA <100k€, plafond 1M€.
- **AIG France** via courtier (Marsh, Aon) : 2 500-5 000 €/an, plafond 2-5M€.
- **Stoïk** (insurtech FR, RC + Cyber bundle) : 1 800-3 000 €/an, plafond 500k€.
- **Wakam** (white label, via courtiers spécialisés tech) : 1 200-2 800 €/an.

**Exclusions classiques à scruter** :
- Conseil en investissement non-déclaré (rejet sinistre si requalification CIF).
- "Speculative trading losses" (souvent exclu).
- ⇒ Demander explicite : "Couverture en cas de réclamation pour perte de trading consécutive à signal algorithmique éducatif".

### 8.2 Cyber

**Couvre** :
- Rançongiciel (rachat + restauration).
- Fuite données personnelles (notification CNIL, frais investigation, sanctions RGPD).
- Interruption d'activité (panne serveur).

**Coût solo founder FR** : **800-1 800 €/an** (Stoïk, Hiscox CyberClear, AXA Cyber). Plafond typique 250 k€.

### 8.3 Total recommandé

**3-5 k€/an** pour bundle RC Pro fintech (1M€) + Cyber (250k€). À souscrire **avant** 1er paiement client.

---

## 9. Restrictions géographiques — Implémentation

### 9.1 Pays à bloquer (priorité)

| Catégorie | Pays / régions | Raison | Méthode |
|-----------|----------------|--------|---------|
| **OFAC SDN** (US sanctions) | Iran, Syrie, Corée du Nord, Cuba, Crimée, régions occupées Ukraine, etc. | Sanctions internationales | IP geo-block + KYC vérif |
| **FATF haut risque** (liste noire) | Iran, Corée du Nord, Myanmar | AML/CFT | IP geo-block |
| **FATF gris** (liste grise) | Bulgarie, Burkina Faso, Cameroun, Croatie, RDC, Gibraltar, Haïti, Jamaïque, Mali, Mozambique, Nigeria, Philippines, Sénégal, Soudan du Sud, Syrie, Turquie, Vietnam, Yémen (avr. 2025 — à mettre à jour) | Vigilance renforcée | Optionnel (KYC strict si vente) |
| **Régulation locale stricte** | États-Unis (SEC), Québec (AMF QC) | Risque inscription RIA / IA QC | IP geo-block |
| **Régulation à valider** | Royaume-Uni (FCA s21), Singapour (MAS), Hong Kong (SFC), Corée du Sud (FSC) | Inscription nécessaire ou approval | IP geo-block phase 1 |

### 9.2 Implémentation technique

**Service IP geolocation** :
1. **MaxMind GeoLite2** (gratuit, mise à jour mensuelle, accuracy ~99.5% pays) — recommandé démarrage.
2. **MaxMind GeoIP2 Precision** (payant, accuracy 99.8% + ville) — quand >10k req/jour.
3. **IPinfo.io** (alternative, free tier 50k/mois).

**Code à ajouter** (middleware FastAPI) :

```python
# src/api/middleware/geo_block.py (à créer)
import geoip2.database

BLOCKED_COUNTRIES = {"US", "IR", "KP", "SY", "CU"}
BLOCKED_REGIONS = {("CA", "QC")}  # (country, region)

class GeoBlockMiddleware:
    def __init__(self, app):
        self.app = app
        self.reader = geoip2.database.Reader("./data/GeoLite2-Country.mmdb")

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            client_ip = scope["client"][0]
            try:
                resp = self.reader.country(client_ip)
                if resp.country.iso_code in BLOCKED_COUNTRIES:
                    # 451 Unavailable For Legal Reasons
                    return await self._reject(send, 451, "Service unavailable in your jurisdiction")
            except Exception:
                pass  # fail-open ou fail-closed selon politique
        await self.app(scope, receive, send)

    async def _reject(self, send, code, msg):
        await send({"type": "http.response.start", "status": code, "headers": []})
        await send({"type": "http.response.body", "body": msg.encode()})
```

**À wirer** dans `src/api/app.py` après le rate limiter.

### 9.3 Limites du geo-blocking IP

- VPN contournement trivial. Mitigation : ajouter bloc dans **CGU article "Restrictions territoriales"** (utilisateur s'engage à ne pas utiliser depuis pays bloqué — décharge partielle).
- KYC à l'inscription pour tier payant : demander pays de résidence + IBAN (vérifier cohérence avec IP).
- Géolocalisation pas suffisante pour AML : Stripe/Paddle gèrent leur côté.

### 9.4 Décharge légale article CGU

```
Article X — Restrictions territoriales
Le Service n'est pas destiné à être utilisé par des résidents ou des personnes
physiquement présentes dans les juridictions suivantes : États-Unis d'Amérique,
Province de Québec (Canada), Iran, Syrie, Corée du Nord, Cuba, et toute
juridiction où la diffusion d'analyses algorithmiques de marché requiert une
licence non détenue par Smart Sentinel AI. L'utilisateur garantit ne pas être
résident ni présent dans une telle juridiction. Smart Sentinel AI se réserve le
droit de bloquer l'accès à tout moment, sans préavis ni remboursement, à toute
adresse IP localisée dans une juridiction interdite.
```

---

## 10. Top 5 risques bloquants — priorisés

| # | Risque | Probabilité (12 mois) | Impact | Effort mitigation | Priorité |
|---|--------|------------------------|--------|---------------------|----------|
| **1** | **Cease & desist SEC / state-IA pour signaux non-enregistrés depuis IP US** | Haute si scraping LinkedIn / SEO US | $$$ (gel comptes Stripe US, frais avocat US) | 0.5 j (geo-block + CGU) | **🔴 P0 — sous 7 jours** |
| **2** | **Plainte CNIL / RGPD pour absence Privacy Policy + transferts Anthropic non documentés** | Moyenne (faible tant que <500 users) | €€ (sanction CNIL jusqu'à 4% CA, mais souvent rappel à l'ordre d'abord) | 5-7 j (Privacy Policy + DPA Anthropic + registre Art.30) | **🔴 P0 — sous 30 jours** |
| **3** | **Cease & desist Dukascopy / ForexFactory pour usage commercial sans licence** | Moyenne (les deux ont historique de C&D actifs) | €€ (devoir migrer feed urgemment, downtime) | 10-15 j (migrer Polygon.io ou Tiingo + Trading Economics) | **🟠 P1 — sous 60 jours** |
| **4** | **Requalification CIF par AMF suite à signalement (concurrent, ex-client mécontent)** | Faible si <500 abonnés payants, croît avec visibilité | €€€ (mise en demeure 6 mois, amende administrative, interdiction commercialiser) | 15-20 j (note de cadrage avocat fintech + reformulation produit "analyses" pas "signaux") | **🟠 P1 — sous 90 jours** |
| **5** | **Réclamation client pour perte trading après signal "BUY"** | Faible mais possible, surtout USA | €€ (frais défense, dommages directs si pas RC Pro) | 1-2 sem (RC Pro Hiscox + plafond responsabilité dans CGU) | **🟡 P2 — sous 90 jours** |

### Risques latents (audit 2 ans)

6. Schrems III invalide DPF UE-US → re-négocier hébergement Anthropic ou migrer vers Mistral / autre LLM hébergé UE.
7. MiCA UE (en vigueur progressive 2024-2026) : si extension vers crypto signaux, nouveau régime de licences.
8. UE AI Act (en vigueur progressive 2024-2027) : SaaS classifié risque "limité" (transparence obligatoire — déjà couvert par disclaimer "généré par IA"). À recroiser si système devient "high-risk".
9. Litige propriété intellectuelle si reprise de techniques publiées (LuxAlgo, TradingView) — peu probable mais à scruter.
10. Réforme statut auto-entrepreneur FR (TVA intracommunautaire au-delà 36 800 € CA en 2026) — comptable à briefer.

---

## 11. Top 5 actions sous 30 jours

### Quick Wins (< 7 j cumulés)

**J1-J2 — Geo-blocking (P0 risque #1)**
- Créer `src/api/middleware/geo_block.py` (cf. §9.2).
- Bloquer US + Québec + OFAC SDN. Retour HTTP 451.
- Test : `curl --resolve [domain]:443:[US-IP] …` et `--resolve …:443:[FR-IP]`.

**J3 — Disclaimer renforcement multi-langue (P0 risque #5)**
- Modifier `src/delivery/telegram_notifier.py:142` et `src/delivery/discord_notifier.py:154` (cf. §3.2).
- Ajouter champ `disclaimer` dans `SignalResponse` model (`src/api/models.py`).
- Centraliser dans `config.DISCLAIMERS = {"en": "...", "fr": "...", ...}`.

**J4-J5 — Endpoint /terms minimal (P0 risque #2)**
- Implémenter `GET /api/v1/terms` (mentionné `SPRINT_ROADMAP.md:1265`) servant un Markdown statique.
- Implémenter `GET /api/v1/privacy` idem.
- Lier depuis tous les disclaimers.

**J6 — DPA Anthropic + vérif DPF**
- Email à support@anthropic.com pour DPA standard signé.
- Vérifier DPF certification sur https://www.dataprivacyframework.gov/list (Anthropic Inc.).

**J7 — Registre traitements (Art.30 RGPD)**
- Document interne 1 page recensant les 9 lignes du tableau §4.1.
- Stocker dans `compliance/registre_traitements_2026.md` (gitignored si sensible).

### Moyen terme (J8-J30)

**S2 J8-J14 — CGU + Privacy Policy v1**
- Achat template Captain Contrat (300 €) ou rédaction en interne en se basant sur §6.2-6.3.
- Mise en ligne sur landing page (à créer). Acceptation explicite checkbox au signup.

**S3 J15-J21 — RC Pro + Cyber**
- Devis Hiscox (en ligne 30 min) + Stoïk + Wakam.
- Souscrire avant 1er paiement.

**S4 J22-J30 — Note de cadrage avocat fintech**
- 1 RDV 2h avec avocat (De Gaulle Fleurance ou Aramis) : qualification CIF / AMF, validation CGU, couverture US/UK/QC.
- Budget 1.5-3 k€.
- Output : memo écrit, base de tous les choix futurs.

---

## 12. Trade-offs assumés

| Décision | Trade-off |
|----------|-----------|
| **Bloquer US + Québec entièrement** | Perd ~30-40% du TAM retail forex. **Mitigation** : ouvrir post-PMF avec opinion sec-counsel + RIA exemption mémo. Le risque enforcement SEC (cease & desist + restitution comptes Stripe) est >> que le gain marginal d'un retail US <500$/mois. |
| **Paddle vs Stripe au démarrage** | Paddle 5% vs Stripe 2.9%. **Justification** : économie temps founder (pas de TVA OSS à gérer, pas de PSP local à craindre). À ré-évaluer >10k€/mois MRR. |
| **Reformuler "signaux" → "analyses algorithmiques"** | Marketing un cran moins punchy. **Justification** : critique pour défense AMF / ESMA finfluencer 2026. Concurrents qui crient "Best AI signals" prennent le risque ; Smart Sentinel se positionne **"institutional market intelligence"** = plus B2B-friendly aussi. |
| **Migrer Dukascopy → Polygon.io / Tiingo** | Coût $30-200/mois additionnel. **Justification** : compliance commerciale + qualité données + SLA. Diluera <2% de la marge brute. |
| **Anonymisation signaux historiques (vs effacement)** | Code plus complexe. **Justification** : conserve métriques agrégées pour tracking performance global, conforme art. 17 §3 RGPD. |
| **Pas de DPO** | Risque marginal si traitements augmentent. **Mitigation** : ré-évaluer dès passage tier INSTITUTIONAL B2B (PII enrichi). |
| **Pas d'enregistrement CIF AMF tout de suite** | Si requalification, sanction. **Justification** : enregistrement coûte ~3-5k€/an + obligations rapport annuel + RC Pro CIF dédiée + ORIAS. Disproportionné <500 abonnés payants. À franchir aux ~1k abonnés payants (audit BPI / AMF). |

---

## 13. KPIs de compliance — mesurables

| KPI | Baseline actuel | Cible 30j | Cible 90j |
|-----|-----------------|-----------|-----------|
| Pays bloqués (geo-IP) | 0 | 5 (US, QC, IR, KP, SY) | 12 (+ FATF haut risque) |
| Disclaimers multi-langues présents | 1 (EN, basique) | 3 (EN, FR, ES) | 5 (+ DE, IT) |
| CGU + Privacy publiées | 0 | 2 (FR + EN) | 4 (+ DE, ES) |
| Endpoints RGPD (export, delete) | 0 | 1 (export) | 2 (export + delete avec anonymisation) |
| DPA signé avec Anthropic | non | oui | oui (+ MaxMind, Stripe/Paddle, Telegram BotFather) |
| Registre Art.30 RGPD | non | v1 minimal | v1.1 audité par avocat |
| RC Pro fintech souscrite | non | demande devis | active |
| Cyber souscrite | non | demande devis | active |
| Note avocat fintech sur qualification | non | RDV pris | reçue |
| Cabinets fintech contactés | 0 | 3 (devis comparé) | 1 sélectionné |
| Coût compliance / mois (assurance + outils) | 0 € | ~250 € | ~500 € |
| Coût compliance one-shot (avocat + setup) | 0 € | ~500 € (templates) | ~5 k€ (avocat + audit) |

---

## 14. Benchmarks sectoriels

- **TradingView Premium** : disclaimer présent partout, pas RIA enregistré US (exception Lowe), bloque pas US. Stratégie agressive — feasible avec leur taille (>500M users, équipe legal interne).
- **LuxAlgo** : disclaimer fort, basé Hong Kong (régulation crypto/forex plus permissive 2026), Stripe via Hong Kong entity. Block US restreint.
- **3Commas** (bot crypto) : enregistré comme "software vendor" pas "investment adviser", DPA Estonia, KYC client final sur tier > $1k/mois. Strucutre intéressante à imiter.
- **TrendSpider** : enregistré Lowe exception, disclaimer fort, propose **"signals are educational only"** systématiquement. Bloque pas US.
- **Mt5 Signals Marketplace** (MetaQuotes) : structure de marketplace = MetaQuotes est "intermédiaire", chaque signal provider est en théorie responsable. Notes : MetaQuotes fait passer une déclaration KYC à chaque signal provider mais pas de licence formelle requise. Modèle à étudier pour V2 marketplace.
- **eToro CopyTrader** : eToro est licencié CySEC/FCA/ASIC ⇒ régime totalement différent (gestion d'actifs déléguée).

**Insight** : la plupart des concurrents reposent sur **exception "publisher" Lowe v. SEC** pour les US et **disclaimer fort** pour l'UE. C'est défendable tant que :
1. Aucune personnalisation par utilisateur.
2. Disclaimer omniprésent et explicite.
3. Pas de "hot tips" sur événements.
4. Pas de gestion d'actifs ni d'exécution de trades par le SaaS.

Smart Sentinel coche **1, 2 partiellement, échoue 3 (news_analysis_agent réagit aux events)**. Mitigation : présenter news comme **"observation d'environnement"** pas **"timing recommendation"**.

---

## 15. Verdict commercial

**Aujourd'hui (2026-04-26)** :
- TESTING_MODE entre amis : **OK** (pas de paiement, pas de rediffusion publique massive).
- Beta privée payante (< 50 users, FR uniquement, opt-in explicite, RC Pro souscrite, CGU minimales) : **acceptable** après J1-J14 (geo-block + disclaimers + Privacy Policy basique).
- Public listing avec marketing SEO international + paid ads : **NON**, exposition trop forte avant audit avocat fintech (J22-J30).

**Après les 5 actions priorisées (J1-J30)** :
- Le produit devient **commercialisable en EU francophone + EN générique** (hors US/UK/QC).
- Coût additionnel : ~1.5-3 k€ one-shot + ~250-500 €/mois récurrent.
- Risque résiduel maîtrisé : surtout SEC/FCA (mitigé par geo-block) et AMF requalification CIF (mitigé par reformulation produit).

**Note d'exposition globale** :
- **3.5 / 10 aujourd'hui** (PoC entre amis acceptable, payment = pas encore).
- **6.5 / 10 projeté à J+30** (compliance "minimum viable").
- **8 / 10 projeté à J+90** (audit avocat + RC Pro + CGU pro = professionnel).
- **9 / 10 cible 12 mois** (CIF déclaration ou opinion no-action SEC + multi-juridictions).

---

## 16. Annexe — Actions concrètes file_path:line

### Code à ajouter / modifier

1. `src/delivery/telegram_notifier.py:142` — étendre disclaimer multi-langue + lien `/terms`.
2. `src/delivery/discord_notifier.py:154` — idem.
3. `src/api/models.py` — ajouter `disclaimer: str` obligatoire dans `SignalResponse`, défaut depuis `config.DISCLAIMERS[lang]`.
4. `src/api/middleware/geo_block.py` — **NEW**, middleware GeoIP MaxMind, retourne 451 pour US/QC/OFAC. Wirer dans `src/api/app.py` après rate limiter.
5. `src/api/routes/legal.py` — **NEW**, endpoints `GET /api/v1/terms`, `GET /api/v1/privacy`, `GET /api/v1/cookies` servant Markdown statique.
6. `src/api/routes/me.py` — **NEW**, endpoints `GET /api/v1/me/data` (export RGPD art. 15) et `DELETE /api/v1/me` (effacement art. 17 avec anonymisation signaux historiques).
7. `src/intelligence/llm_narrative_engine.py:80` — ajouter au system prompt : « Output MUST end with: 'Educational analysis. Not investment advice. See terms.' »
8. `config.py` — ajouter `DISCLAIMERS: dict[str, str]`, `BLOCKED_COUNTRIES: set[str]`, `BLOCKED_REGIONS: set[tuple[str, str]]`.
9. `data/GeoLite2-Country.mmdb` — télécharger (gratuit MaxMind compte signup) + ajouter au `.gitignore`.
10. `requirements.txt` — ajouter `geoip2==5.0.1`.

### Documents légaux à créer (hors-code)

11. `compliance/registre_traitements_2026.md` — Registre Art. 30 RGPD (cf. §4.1 tableau).
12. `compliance/dpa_anthropic_signed.pdf` — à demander à Anthropic support.
13. `compliance/cgu_v1_fr.md` — basé sur structure §6.2.
14. `compliance/privacy_v1_fr.md` — basé sur structure §6.3.
15. `compliance/cgu_v1_en.md` + `privacy_v1_en.md` — traductions.
16. `compliance/avocat_brief.md` — résumé pour 1er RDV avocat fintech (3-5 pages).

### Tests

17. `tests/test_geo_block.py` — **NEW**, mocks IP US/CA-QC/FR/IR, asserte 451 vs 200.
18. `tests/test_disclaimer_present.py` — **NEW**, asserte disclaimer présent dans tous les payloads `SignalResponse`, Telegram messages, Discord embeds.
19. `tests/test_legal_endpoints.py` — **NEW**, asserte `/terms`, `/privacy`, `/cookies` retournent 200 + content-type markdown.

### Configuration ops

20. Souscrire RC Pro Hiscox (devis 30 min en ligne).
21. Souscrire Cyber Stoïk (devis 30 min).
22. Email Anthropic support : DPA standard signé.
23. Email MaxMind : signup compte gratuit + télécharger DB.
24. Choisir cabinet fintech FR : prendre 3 RDV ronds (1h chacun, gratuits ou 200 €) avant de sélectionner.
25. Si Stripe : créer compte Atlas / Stripe France ; si Paddle : compte Merchant of Record.

---

## 17. Glossaire (acronymes)

- **AFSL** : Australian Financial Services Licence
- **AMF** : Autorité des Marchés Financiers (FR) ; AMF QC = Autorité des marchés financiers (Québec)
- **ASIC** : Australian Securities and Investments Commission
- **CCT** : Clauses Contractuelles Types (UE 2021/914)
- **CIF** : Conseiller en Investissements Financiers (FR, art. L.541-1 CMF)
- **CNIL** : Commission Nationale de l'Informatique et des Libertés (FR)
- **DPA** : Data Processing Agreement
- **DPF** : EU-US Data Privacy Framework (2023)
- **DPO** : Data Protection Officer
- **ESMA** : European Securities and Markets Authority
- **FATF** : Financial Action Task Force (GAFI en français)
- **FCA** : Financial Conduct Authority (UK)
- **IA Act** (US) : Investment Advisers Act of 1940
- **MiFID II** : Markets in Financial Instruments Directive II (2014/65/UE)
- **MoR** : Merchant of Record (Paddle, Lemon Squeezy)
- **OFAC** : Office of Foreign Assets Control (US Treasury)
- **PSD2 / DSP2** : Payment Services Directive 2 (UE 2015/2366)
- **RC Pro** : Responsabilité Civile Professionnelle
- **RGPD / GDPR** : Règlement Général sur la Protection des Données (UE 2016/679)
- **RIA** : Registered Investment Adviser (US, SEC enregistré)
- **SDN** : Specially Designated Nationals (OFAC list)
- **SEC** : Securities and Exchange Commission (US)
- **TMD** : Target Market Determination (ASIC AU)

---

## 18. Sources et textes cités (vérification)

| Texte | Référence | Date / version | Lien type |
|-------|-----------|----------------|-----------|
| MiFID II | Directive 2014/65/UE | 15 mai 2014 | EUR-Lex |
| MiFID II finfluencers | Directive 2024/2811 | mars 2026 application | EUR-Lex |
| Code monétaire et financier FR | Art. L.541-1 (CIF) | màj 2024 | Légifrance |
| Règlement général AMF | Livre III | 2024 | amf-france.org |
| ESMA Q&A investor protection | ESMA35-43-349 | màj 2024 | esma.europa.eu |
| AMF position-recommandation | DOC-2014-04 | màj 2023 | amf-france.org |
| Investment Advisers Act US | 15 U.S.C. §80b — §202(a)(11) | 1940, màj 2024 | sec.gov |
| Lowe v. SEC | 472 U.S. 181 | 1985 | supreme.justia.com |
| SEC Release IA-1092 | Persons Providing Investment Advice | 1987 | sec.gov |
| RGPD | Règlement (UE) 2016/679 | 27 avr. 2016 | EUR-Lex |
| CCT UE | Décision 2021/914 | 4 juin 2021 | EUR-Lex |
| Schrems II | CJUE C-311/18 | 16 juill. 2020 | curia.europa.eu |
| EU-US DPF | Décision 2023/1795 | 10 juill. 2023 | EUR-Lex |
| FCA financial promotions | FSMA 2000 s.21 + PERG 8 | 2024 | handbook.fca.org.uk |
| ASIC general advice | RG 36, RG 234, RG 274 | 2021-2024 | asic.gov.au |
| AMF QC inscription | Loi VM Québec art. 148 + Règlement 31-103 | 2024 | lautorite.qc.ca |
| FATF jurisdictions | Update April 2025 | 2025-04 | fatf-gafi.org |
| OFAC SDN list | màj continue | 2026-04-26 | treasury.gov |
| LCEN (FR) | Loi n° 2004-575 | 21 juin 2004, màj 2024 | Légifrance |
| Code Conso L.215-1 (renouvellement abo) | màj 2024 | Légifrance |
| Code Conso L.221-18 (rétractation 14j) | màj 2024 | Légifrance |
| Code Conso L.612-1 (médiation) | màj 2024 | Légifrance |

⚠️ **Toutes ces références doivent être recroisées** par un avocat avant publication CGU. L'agent IA n'a pas accès web pour vérifier dates / numéros au moment de cette rédaction.

---

**Fin du rapport. Note d'exposition globale : 3.5 / 10 aujourd'hui → 6.5 / 10 projeté J+30 → 8 / 10 projeté J+90.**
