# MIA Markets — Vision Produit V2

Document d'architecture cible
Date de rédaction : 2026-05-29
Auteur : Loukmane Bessam, fondateur MIA Markets
Statut : Validé pour exécution

## Préambule

Ce document décrit l'architecture cible de MIA Markets V2. Il fait suite à deux constats :

Le pivot positionnement du 2026-05-27 qui transforme MIA Markets de « service de signaux de trade » en « outil de compréhension augmentée du marché ».
La découverte que l'implémentation actuelle ne correspond pas à la vision commerciale : le code produit des signaux discrets sur setup détecté (~1.7 fois par mois), alors que le positionnement vendu est celui d'un indicateur de marché continu.

Ce document définit l'architecture du produit refondu pour résoudre cet écart. Il sert de référence pour le découpage en chantiers de développement et de brief pour les terminaux Claude Code qui exécuteront ces chantiers.
Toutes les décisions ci-dessous ont été validées en session interactive du 2026-05-29 entre le fondateur et Claude (LLM d'architecture).

## Section 1 — Vision Produit V2

### 1.1 Positionnement fondamental

MIA Markets est un outil de compréhension augmentée du marché destiné aux traders retail SMC/ICT. Le produit fournit une lecture continue, factuelle et structurée des conditions de marché, accompagnée d'un chatbot pédagogique permettant d'approfondir chaque élément de la lecture.
Le produit décrit. Le produit n'invente pas. Le produit ne recommande jamais d'action de trade.

### 1.2 Posture juridique — Niveau 1.5 strict

Le produit s'inscrit dans la catégorie « information financière générale + caractérisation factuelle des conditions de marché », non-réglementée selon la directive UE 2024/2811 et MiFID II.
Frontière à ne jamais franchir : le produit décrit les conditions de marché, jamais les actions à prendre.
Exemples de formulations autorisées :

- « Volatilité observée sur les 4 dernières heures : élevée. »
- « Structure SMC actuelle : BOS haussier confirmé il y a 30 minutes. »
- « News USD à fort impact attendue dans 47 minutes. »
- « Conditions techniques actuelles caractérisées par : retest en cours, volatilité élevée, news imminente. »

Exemples de formulations interdites (jamais publiées par le produit) :

- « Bon moment pour trader long. »
- « Évitez de trader maintenant. »
- « Conditions favorables / défavorables au trading. »
- « C'est risqué / dangereux / sûr de prendre position. »
- Toute formulation personnalisée (« vous devriez », « pour vous »).

### 1.3 Périmètre fonctionnel V1

Instruments couverts : XAU (Or) et EURUSD uniquement.
Timeframes supportées : 15min, 1h, 4h.
Langue d'interface : Français uniquement en V1. EN et autres langues en V2+.
Géographie : conformément au lock 4 du 2026-05-27, le produit cible FR/BE/CH/LU/Canada hors Québec/UK/AU/NZ/IE. Les territoires suivants sont géo-bloqués : États-Unis (SEC/NFA), Québec (AMF Québec), pays sous sanctions OFAC.

### 1.4 Structure tarifaire

Tier Découverte (gratuit) :

- Accès à la MarketReading actuelle (pas d'historique)
- Quota chatbot : 5 questions/jour
- Accès aux 2 instruments + 3 TFs

Tier Approfondie (9 €/mois) :

- MarketReading actuelle + historique 7 jours
- Quota chatbot : 30 questions/jour
- Filtres personnalisés activés

Tier Intégrale (19 €/mois) :

- MarketReading actuelle + historique 30 jours
- Quota chatbot : illimité (fair use)
- Filtres personnalisés + alertes personnalisées + sauvegarde multi-configurations

Tier Institutional : retiré de la grille publique V1, accessible uniquement via Calendly.

### 1.5 Critères d'acceptation Section 1

Pour considérer la Section 1 « livrée », les conditions suivantes doivent être remplies :

- La landing page actuelle communique sans ambiguïté le positionnement niveau 1.5 strict
- Aucune communication marketing ne suggère que le produit donne des signaux de trade ou des conseils
- Les CGV mentionnent explicitement le caractère informatif non-réglementé du produit
- Le bloc géo-restriction est actif sur la landing pour les 3 territoires bloqués

## Section 2 — Data Model MarketReading

### 2.1 Vue d'ensemble

Le produit publie, à chaque candle close de la timeframe regardée, un objet structuré appelé MarketReading. Cet objet remplace l'ancien InsightSignalV2 qui était orienté trade.
Une MarketReading décrit l'état actuel du marché sans jugement de valeur sur les actions à prendre.

### 2.2 Cadence de publication

À chaque candle close de la timeframe regardée :

- Si TF = 15min → une nouvelle MarketReading toutes les 15 minutes
- Si TF = 1h → toutes les heures
- Si TF = 4h → toutes les 4 heures

Indépendamment de la cadence des candles :

- Les news sont rafraîchies par polling de ForexFactory toutes les 1-2 minutes
- Le champ news_upcoming.time_to_event_min est recalculé à chaque accès utilisateur

### 2.3 Structure complète de la MarketReading

```json
{
  "schema_version": "2.0.0",
  "header": {
    "instrument": "XAUUSD",
    "timeframe": "M15",
    "candle_close_ts": "2026-05-28T14:00:00Z",
    "close_price": 2378.45
  },
  "structure": {
    "bos": {
      "direction": "bullish",
      "level": 2375.20,
      "broken_at": "2026-05-28T13:30:00Z",
      "validation_status": "confirmed"
    },
    "choch": null,
    "order_blocks": [
      {
        "id": "OB_001",
        "level_high": 2370.50,
        "level_low": 2368.20,
        "importance": "high",
        "status": "active",
        "created_at": "2026-05-26T08:00:00Z",
        "tested": false,
        "user_flagged": false
      }
    ],
    "fair_value_gaps": [
      {
        "id": "FVG_001",
        "level_high": 2378.20,
        "level_low": 2376.00,
        "status": "active",
        "created_at": "2026-05-28T12:15:00Z",
        "tested": false,
        "user_flagged": false
      }
    ],
    "retest_in_progress": {
      "level": 2375.20,
      "type": "bos_retest",
      "started_at": "2026-05-28T13:45:00Z"
    }
  },
  "regime": {
    "trend": "bullish",
    "volatility_observed": "elevated",
    "market_phase": "expansion",
    "mtf_confluence": {
      "h1": "bullish",
      "h4": "bullish"
    }
  },
  "events": {
    "news_upcoming": [
      {
        "event": "US Non-Farm Payrolls",
        "scheduled_at": "2026-05-28T14:30:00Z",
        "time_to_event_min": 30,
        "impact": "high",
        "currency": "USD",
        "potential_effect_description": "Publication majeure de l'emploi américain. Mouvement attendu sur le dollar, impact indirect sur XAU via la corrélation USD inverse classique."
      }
    ],
    "news_just_published": [],
    "technical_triggers_recent": [
      {
        "type": "bos_h1_bullish",
        "occurred_at": "2026-05-28T13:30:00Z",
        "minutes_ago": 30
      }
    ]
  },
  "conditions": {
    "tags": [
      "volatility_elevated",
      "news_imminent_high_impact",
      "structure_aligned_mtf",
      "retest_in_progress"
    ],
    "description": "Le marché XAU sur 15min est en phase de retest d'un Order Block H1, avec volatilité élevée à l'approche du NFP USD dans 30 minutes. Structure H1 et H4 alignée haussièrement.",
    "description_source": "haiku_generated"
  }
}
```

### 2.4 Règles de remplissage par champ

Section structure :

- `order_blocks` et `fair_value_gaps` : limitées aux N plus importants et récents. Le critère « important » est défini techniquement par le terminal qui code le moteur, mais doit inclure au minimum : proximité du prix actuel, volume de la candle de création, alignement avec niveaux clés. Toujours afficher actifs ET mitigated avec un flag `tested: true/false`.

Section regime :

- `volatility_observed` : valeur parmi `low`, `normal`, `elevated`. Calculée sur fenêtre rolling de 20 candles vs moyenne historique 30 jours.
- `mtf_confluence` : limitée à 2 niveaux au-dessus de la TF regardée. Si TF=15min → h1+h4. Si TF=1h → h4+daily. Si TF=4h → daily+weekly.

Section events :

- `news_upcoming` : toutes les news majeures (impact medium ou high) dans les prochaines 24h, indépendamment de la devise. Chaque news doit inclure un `potential_effect_description` qui contextualise l'impact attendu sur l'instrument regardé.
- Source de news : ForexFactory en V1.

Section conditions :

- `tags` : liste fermée de tags techniques. Le terminal qui code le moteur définira la taxonomie exacte. Exemples : `volatility_elevated`, `news_imminent_high_impact`, `structure_aligned_mtf`, `retest_in_progress`, `range_bound`, `breakout_recent`, `structure_degraded`, etc.
- `description` : phrase courte (1-3 phrases) en français qui résume les tags. Générée par Haiku 4.5 avec les 4 leviers d'optimisation ci-dessous.
- `description_source` : `haiku_generated` ou `template_fallback`.

### 2.5 Stratégie d'optimisation coût LLM pour la phrase descriptive

La génération de la phrase descriptive doit utiliser Haiku 4.5 mais dans des conditions strictes de coût. 4 leviers doivent être implémentés :

**Levier 1 — Cache par hash de tags**
Si la combinaison de tags (`["volatility_elevated", "news_imminent_high_impact", ...]`) est identique à la précédente MarketReading pour le même instrument/TF, on réutilise la description précédente. Coût : 0 appel API.

**Levier 2 — Fallback template pour cas simples**
Si la combinaison contient ≤2 tags simples (par exemple seulement `range_bound`), on utilise un template Python plutôt qu'un appel LLM. Les templates couvrent les 20-30 combinaisons les plus fréquentes.

**Levier 3 — Prompts ultra-courts**
Quand Haiku est appelé, le prompt système est court (< 200 tokens) et le contexte se limite aux tags + 2-3 chiffres clés (close price, time-to-event news, niveau BOS). Budget total par appel : < 500 tokens (input+output).

**Levier 4 — Génération à la demande utilisateur, pas systématique**
La phrase descriptive est générée uniquement lors de l'accès utilisateur, pas en background à chaque candle close. Le moteur stocke les tags à chaque candle, et la phrase est générée au moment où l'utilisateur ouvre l'app sur cette combinaison.

### 2.6 Critères d'acceptation Section 2

- Le schéma MarketReading est implémenté en TypeScript (frontend) et Python (backend) avec validation stricte
- La taxonomie des tags de conditions est documentée et stable (pas de mutation silencieuse)
- Les 4 leviers d'optimisation coût Haiku sont implémentés et testés
- Coût LLM par utilisateur actif < 0.10 $/mois en moyenne (vérifié par tracking sur première semaine de prod)

## Section 3 — Moteur de Génération

### 3.1 Architecture cible

Le moteur passe d'une logique événementielle (signal émis seulement sur setup détecté) à une logique continue par candle (MarketReading produite à chaque candle close).

Mode de fonctionnement : hybride

- **Lazy au premier accès** : si personne n'a consulté un instrument/TF depuis 24h, le backend ne calcule rien
- **Continu après premier accès** : dès qu'un utilisateur consulte XAU 15min, le backend déclenche un job récurrent qui recalcule la MarketReading XAU 15min à chaque candle close, même si l'utilisateur ne consulte plus
- **Arrêt automatique** : si plus aucun utilisateur n'a consulté XAU 15min pendant 24h consécutives, le job s'arrête

Bénéfices :

- L'historique est continu (pas de trous) pour les utilisateurs Intégrale qui regardent ensuite
- Possibilité d'évoluer vers des notifications push (V2)
- Coût LLM contenu grâce aux 4 leviers de la Section 2

### 3.2 Source de données

Backend : OANDA API v20 (compte démo gratuit)
Justification :

- API REST officielle, documentation propre, génération de client OpenAPI
- Données XAU et EURUSD quasi-temps réel (latence ~1 seconde)
- Compte démo gratuit avec accès illimité
- Pas de problème légal contrairement aux APIs non-officielles type TradingView

Endpoint principal : `GET /v3/instruments/{instrument}/candles` avec paramètres `granularity` (M15/H1/H4) et `count`.

Polling :

- À chaque candle close, le moteur appelle OANDA pour récupérer les dernières candles
- Stockage local des candles dans SQLite pour reprocessing rapide

### 3.3 Scheduler

Choix technique : APScheduler (Python)

- Bibliothèque Python intégrée dans le process FastAPI principal
- Jobs récurrents configurables par instrument/TF
- Pas de dépendance externe (Redis, Celery) pour V1

Logique :

- Quand un utilisateur consulte XAU 15min pour la première fois → enregistrement dans table `active_combinations(instrument, timeframe, last_accessed_at)`
- Scheduler interroge cette table toutes les minutes pour activer/désactiver les jobs

### 3.4 Stockage

Choix technique : SQLite avec volume persistant Fly.io

- Une seule base SQLite contenant toutes les tables (users, market_readings, active_combinations, news_cache, etc.)
- Volume Fly.io configuré dans `fly.toml` pour persistance des données
- Backup quotidien automatique vers R2 (storage object compatible S3)

Tables principales :

- `market_readings(id, instrument, timeframe, candle_close_ts, payload_json, created_at)`
- `active_combinations(instrument, timeframe, first_accessed_at, last_accessed_at)`
- `news_cache(event_id, scheduled_at, impact, currency, raw_payload, fetched_at)`

### 3.5 Rétention par tier

- Tier Découverte : MarketReading actuelle uniquement, pas d'accès historique
- Tier Approfondie : historique 7 jours
- Tier Intégrale : historique 30 jours

Implémentation : pas de suppression physique, mais filtrage à la requête selon le tier de l'utilisateur. Suppression définitive après 90 jours pour toutes les MarketReadings (purge cron).

### 3.6 Répartition des calculs

Recalculé à chaque candle close de la TF regardée :

- Prix de close
- BOS / CHOCH récents
- État des FVG (un FVG peut se combler sur cette candle)
- État des OB (un OB peut être testé sur cette candle)
- Retest en cours
- Volatilité observée (rolling 20 candles)
- Tags de conditions

Recalculé à chaque candle close de la TF supérieure :

- Tendance H1 et H4 (utilisée dans MTF confluence)
- Phase de marché
- HMM regime fitting (lent, fait en background)

Recalculé en continu (toutes les 1-2 min, indépendant des candles) :

- News upcoming (poll ForexFactory)
- News just published
- Time-to-event en minutes

Généré à la demande utilisateur :

- Phrase descriptive (Haiku, via leviers d'optimisation)

### 3.7 Critères d'acceptation Section 3

- Le moteur produit une MarketReading valide à chaque candle close pour les combinaisons actives
- La latence d'affichage à l'ouverture de l'app est < 3 secondes (premier accès) et < 500ms (accès suivants)
- Le scheduler gère correctement l'activation/désactivation des jobs sans fuite mémoire
- Tests d'intégration vérifient le bon fonctionnement sur 7 jours de simulation

## Section 4 — Chatbot Niveau 1.5 Strict

### 4.1 Architecture de génération de réponse

Le chatbot fonctionne en 3 couches de défense contre les recommandations involontaires :

**Couche 1 — Adversarial pattern detection (avant LLM)**
Avant tout appel à Haiku, la question utilisateur passe par un classifieur de patterns à risque. Si un pattern match → réponse de refus pédagogique générée par template, sans appeler Haiku.

**Couche 2 — System prompt strict + signal_summary**
Si la question passe la couche 1, Haiku est appelé avec un system prompt qui définit explicitement les règles niveau 1.5 et un contexte signal_summary condensé (cf. 4.4).

**Couche 3 — Forbidden tokens post-filter**
Après génération par Haiku, la réponse passe par un filtre de tokens interdits. Si un token est détecté → la réponse est remplacée par un refus pédagogique structuré.

### 4.2 Catégories de forbidden tokens

**Catégorie A — Verbes d'action de trading :**
`trade`, `tradez`, `achète`, `achetez`, `buy`, `vends`, `vendez`, `sell`, `entre`, `entrez`, `entry`, `sors`, `sortez`, `exit`

**Catégorie B — Verbes de recommandation :**
`je conseille`, `je déconseille`, `je recommande`, `je ne recommande pas`, `tu devrais`, `vous devriez`, `il faut`, `il faudrait`, `évite`, `évitez` (dans contexte trade)

**Catégorie C — Jugements de moment :**
`bon moment`, `mauvais moment`, `c'est le moment`, `ce n'est pas le moment`, `opportunité` (pour suggérer trade), `setup parfait`, `entrée idéale`

**Catégorie D — Jugements de valeur sur risque :**
`c'est risqué`, `c'est dangereux`, `c'est sûr`, `c'est sécurisé`, `low risk`, `high risk` (utilisés comme conseil)

Implémentation précise (liste exhaustive avec variantes, accents, fautes courantes) : à finaliser par le terminal Claude Code qui code cette section.

### 4.3 Catégories d'adversarial patterns (interception en amont)

**Patterns d'action directe :**
« est-ce que je devrais trader », « c'est le bon moment pour », « je trade ou pas », « j'entre ou pas », « tu conseilles quoi », « quel est ton avis », « ton conseil sur »

**Patterns de personnalisation :**
« avec X € de capital », « avec X lots », « j'ai X positions », « mon stop est à », « mon target est à », « si je perds X »

**Patterns de jugement de valeur :**
« c'est risqué de », « c'est dangereux de », « c'est safe de », « est-ce sûr »

**Patterns de demande de signal :**
« donne-moi un signal », « envoie un signal », « un trade à prendre », « long ou short », « bull ou bear » (pour décision)

Implémentation précise (regex avec variantes, accents, fautes courantes) : à finaliser par le terminal Claude Code.

### 4.4 Contexte chatbot — Architecture hybride

Choix : Option hybride avec tool use

Comportement par défaut : Haiku reçoit un `signal_summary` condensé (~500-700 tokens) qui synthétise la MarketReading actuelle en texte naturel.

Exemple de signal_summary :

> XAUUSD 15min, candle 14h00 close 2378.45.
> Structure : BOS haussier confirmé à 2375.20 il y a 30 minutes. FVG actif zone 2376-2378. OB H1 actif à 2370.50 (non testé). Retest en cours sur le niveau BOS.
> Régime : tendance haussière 15min, volatilité élevée, phase d'expansion. MTF : H1 haussier, H4 haussier.
> Events : NFP USD dans 30 minutes (impact élevé). Pas de news majeure récemment.
> Conditions : volatilité élevée, news imminente forte impact, structure alignée MTF, retest en cours.

Tool use sur demande : Haiku a accès à des outils pour requêter les détails précis du JSON brut si nécessaire. Outils disponibles :

- `get_order_block_details(ob_id)` : retourne les détails complets d'un OB
- `get_fvg_details(fvg_id)` : retourne les détails complets d'un FVG
- `get_bos_timing()` : retourne le timestamp exact du BOS et la candle de cassure
- `get_news_details(event_id)` : retourne les détails complets d'une news (historique, dernières publications)
- `get_volatility_metrics()` : retourne les métriques de volatilité précises (sigma, percentile historique)

Coût : la grande majorité des questions sont gérées par le signal_summary (1 appel Haiku). Seules les questions précises déclenchent un tool call (2 appels Haiku au total). Estimation moyenne : 1.2 appels par question.

### 4.5 Réponse de refus pédagogique standard

Quand un pattern adversarial est détecté (couche 1) ou qu'un token interdit est filtré (couche 3), la réponse retournée à l'utilisateur suit ce template :

> Je suis un outil de description des conditions de marché. Je ne donne pas de recommandations de trade ni d'évaluations personnalisées. C'est à vous d'évaluer si les conditions actuelles correspondent à votre méthode de trading et à votre tolérance au risque.
> Ce que je peux vous dire des conditions actuelles : [description factuelle générée à partir du signal_summary, sans appeler Haiku, en mode template].
> Si vous voulez approfondir un élément précis (BOS, FVG, OB, news, régime), n'hésitez pas à me poser une question descriptive.

### 4.6 Critères d'acceptation Section 4

- Recall de détection des patterns adversarial > 95% sur un dataset de test de 50+ questions piégées
- Précision de détection > 95% (peu de faux positifs sur questions légitimes)
- Aucun token interdit ne passe le post-filter sur un dataset de test de 100+ réponses générées
- Temps de réponse moyen du chatbot < 2 secondes sur question simple, < 4 secondes sur question avec tool call

## Section 5 — Webapp UI

### 5.1 Layout général

Choix : Dashboard 3 colonnes sur desktop, tabs sur mobile

Desktop (largeur ≥ 1024px) :

```
┌─────────────────────────────────────────────────────────┐
│  HEADER : Logo MIA Markets — Sélecteur instrument/TF — User │
├──────────┬──────────────────────────┬───────────────────┤
│          │                          │                   │
│  COL 1   │        COL 2             │      COL 3        │
│  Nav     │        Chart             │      Chatbot      │
│  (200px) │        + Cartes infos    │      + Cartes     │
│          │        (60% largeur)     │      événements   │
│          │                          │      (30%)        │
│          │                          │                   │
└──────────┴──────────────────────────┴───────────────────┘
```

**Colonne 1 — Navigation (200px) :**

- Sélecteur instrument (XAU / EURUSD)
- Sélecteur timeframe (15min / 1h / 4h)
- Historique des MarketReadings (selon tier)
- Sauvegarde de configurations multiples (Intégrale)
- Filtres personnalisés (Approfondie+)

**Colonne 2 — Chart + Structure (60% restant) :**

- Chart TradingView Lightweight (haut, ~60% hauteur écran)
- Cartes empilées sous le chart :
  - Carte Structure SMC (BOS, FVG, OB, retest)
  - Carte Régime de marché (trend, volatilité, phase, MTF)

**Colonne 3 — Events + Chatbot (30%) :**

- Carte News (upcoming + just published)
- Carte Conditions actuelles (tags + phrase descriptive)
- Chatbot sidebar permanente en bas (toujours accessible)

Mobile (largeur < 1024px) :

- Tabs horizontaux en haut : Chart / Structure / Events / Chatbot
- Sélecteur instrument/TF en header sticky
- Une tab visible à la fois, swipe pour changer

### 5.2 Composant Chart

Choix : TradingView Lightweight Charts (library MIT)

- Lib JavaScript open-source de TradingView
- Aspect visuel identique à TradingView que les traders SMC connaissent
- Données fournies par notre backend (pas par TradingView)
- Plotting des niveaux SMC (OB, FVG, BOS) en overlay

### 5.3 Composant Chatbot sidebar

- Permanent sur desktop (toujours visible dans la colonne 3)
- Tab dédiée sur mobile
- Affichage des messages précédents (scroll vertical)
- Input texte en bas avec compteur de questions restantes (selon quota tier)
- Badges visuels pour : message LLM normal, refus pédagogique, message filtré compliance
- Suggested questions dynamiques (3 questions contextuelles dérivées de la MarketReading actuelle)

### 5.4 Customisation utilisateur

V1 inclut :

- Sélection instrument + timeframe (tous tiers)
- Toggle « masquer OB/FVG mitigated » (tous tiers)
- Toggle « masquer news low impact » (tous tiers)
- Thème sombre/clair (tous tiers)
- Filtres personnalisés (Approfondie+) :
  - « Afficher uniquement les niveaux à moins de X pips du prix actuel »
  - « Afficher uniquement les FVG/OB de TF ≥ Y »
- Alertes personnalisées (Intégrale) :
  - Notification quand un BOS est détecté
  - Notification quand une news high impact est dans Z minutes
  - Notification quand un FVG est comblé
- Sauvegarde de configurations multiples (Intégrale) :
  - L'utilisateur peut sauvegarder N configurations (instrument + TF + filtres + alertes)
  - Switch rapide entre configurations

### 5.5 Authentification et tier management

- Inscription par email + mot de passe (V1)
- Tier par défaut : Découverte (gratuit)
- Upgrade payant via Stripe Checkout intégré
- Stockage tier dans table `users` (déjà implémenté côté backend)
- Refresh tier à chaque accès via API `/auth/me`

### 5.6 Critères d'acceptation Section 5

- Lighthouse score desktop ≥ 90 sur Performance/Accessibility/Best Practices/SEO
- Lighthouse score mobile ≥ 85 (mobile plus contraignant)
- Tous les composants UI ont des tests Vitest qui passent
- Tests E2E couvrent les parcours principaux : inscription, sélection instrument/TF, consultation MarketReading, question chatbot, upgrade tier

## Plan de découpage en chantiers

Le développement V2 est découpé en 6 chantiers séquentiels :

### Chantier 1 — Source de données OANDA + storage SQLite étendu

Dépendances : aucune
Effort estimé : 5-7 jours
Livrables :

- Client OANDA API généré depuis OpenAPI specs
- Table `market_readings` créée + migrations
- Volume Fly.io configuré
- Scripts de récupération initiale des données XAU et EURUSD

### Chantier 2 — Refonte du moteur (MarketReading generator)

Dépendances : Chantier 1
Effort estimé : 10-15 jours
Livrables :

- Module `market_reading_generator` qui produit le JSON MarketReading complet
- Réutilisation de la couche SMC existante (validée 99.8%)
- Implémentation taxonomie des tags de conditions
- Tests d'intégration sur 30 jours de données historiques

### Chantier 3 — Pipeline news ForexFactory + scheduler APScheduler

Dépendances : Chantier 1
Effort estimé : 4-6 jours
Livrables :

- Poller ForexFactory toutes les 1-2 minutes
- Table `news_cache` + logic de déduplication
- APScheduler configuré pour jobs récurrents
- Logic d'activation/désactivation des combinaisons actives

### Chantier 4 — Chatbot niveau 1.5 strict

Dépendances : Chantier 2 (a besoin du MarketReading pour signal_summary)
Effort estimé : 6-9 jours
Livrables :

- Couche 1 (adversarial patterns) avec recall > 95%
- Couche 2 (system prompt + signal_summary + tool use) opérationnelle
- Couche 3 (forbidden tokens) avec couverture 100%
- Tests adversariaux sur dataset de 100+ questions

### Chantier 5 — Webapp UI 3 colonnes

Dépendances : Chantier 2 (a besoin de l'endpoint MarketReading)
Effort estimé : 12-18 jours
Livrables :

- Layout 3 colonnes desktop + tabs mobile
- Intégration TradingView Lightweight Charts
- Cartes Structure / Régime / Events / Conditions
- Chatbot sidebar avec badges compliance

### Chantier 6 — Customisation utilisateur avancée

Dépendances : Chantier 5
Effort estimé : 8-12 jours
Livrables :

- Filtres personnalisés (tous tiers selon gate)
- Alertes personnalisées (Intégrale)
- Sauvegarde configurations multiples (Intégrale)
- Thème sombre/clair

### Effort total estimé

45 à 67 jours de développement pour les 6 chantiers. À raison de 8-9h/semaine de dev solo, cela représente 5 à 7 mois calendaires.

## Note importante du fondateur

Le fondateur a explicitement validé toutes les décisions ci-dessus en session du 2026-05-29, y compris l'ampleur du périmètre V1 qui inclut customisation avancée (filtres + alertes + multi-configurations). Le fondateur a indiqué « stick to my plan » sur les choix les plus ambitieux (Q1=C dashboard 3 colonnes, Q3=C customisation avancée).

Le timing initial de lancement « mi-juin 2026 » n'est pas tenable avec ce périmètre. Une décision ultérieure sur le timing de lancement devra être prise, soit en :

- Réduisant le périmètre V1 (sortir les alertes et multi-configs en V1.1)
- Acceptant un lancement décalé (août-septembre 2026 plus réaliste)

## Suivi des décisions

Ce document est vivant. Toute modification de décision doit faire l'objet d'un commit dédié avec justification dans le message. Les terminaux Claude Code qui exécutent les chantiers se réfèrent à la version git HEAD de ce document au moment de leur lancement.

Dernière mise à jour : 2026-05-29 (rédaction initiale)
Validé par : Loukmane Bessam, fondateur
