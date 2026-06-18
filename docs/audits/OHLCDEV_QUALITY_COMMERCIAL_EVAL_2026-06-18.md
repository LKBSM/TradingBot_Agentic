# OHLC.dev — Évaluation Qualité & Commerciabilité (rapport PRÉLIMINAIRE)

- **Date** : 2026-06-18
- **Branche** : `diagnostic/ohlcdev-quality-eval` (depuis `institutional-overhaul`)
- **Discipline** : 100 % lecture seule. Aucune modification produit/moteur. Aucun secret committé.
- **Statut** : **PRÉLIMINAIRE + INCOMPLET** — le test de qualité (Section 2) **n'a pas été
  exécuté** faute de clés API. Décision utilisateur : ne pas fournir les clés. Le diagnostic
  s'arrête au volet conditions (Section 3), suffisant pour trancher.

---

## 0. Objet

Évaluer OHLC.dev (revendeur de données de marché, ~15 $/mois, plan Pro) comme fournisseur
XAUUSD + EURUSD pour un produit SMC orienté **honnêteté de l'affichage**. Deux questions :
- **(A) Qualité** des bougies vs référence retail (proxy OANDA v20 démo / TradingView).
- **(B) Commerciabilité** : droit d'affichage/redistribution réellement couvert par écrit.

---

## 1. Prérequis — résultat

| Prérequis | Statut |
|---|---|
| Branche `diagnostic/ohlcdev-quality-eval` créée depuis `institutional-overhaul` | ✅ |
| `OHLCDEV_API_KEY` en variable d'environnement | ❌ **ABSENTE** (non fournie — décision utilisateur) |
| `OANDA_API_TOKEN` (compte démo fxPractice) | ❌ **ABSENTE** (non fournie — décision utilisateur) |
| Réseau → `api-fxpractice.oanda.com:443` | ✅ joignable |
| Réseau → `ohlc.dev:443` | ✅ joignable |
| Réseau → `api.ohlc.dev:443` | ❌ injoignable (pas de host API direct) |

**Note d'accès** : OHLC.dev est **revendu via RapidAPI** (le portail d'accès du site pointe
vers un lien RapidAPI). La clé « OHLCDEV_API_KEY » est donc en pratique une **clé RapidAPI**,
et les requêtes passent par un host `*.p.rapidapi.com` avec en-tête `X-RapidAPI-Key` — et non
par `api.ohlc.dev`. Ce maillon RapidAPI ajoute un **intermédiaire de plus** entre nous et la
source réelle des prix.

---

## 2. Test de qualité (alignement / offset / résidu de mèches)

**NON EXÉCUTÉ.** Bloqué sur l'absence des deux clés (impossible de tirer les bougies des deux
sources). Aucun script jetable n'a été créé. À relancer si des clés sont fournies
ultérieurement : protocole prévu = alignement par timestamp d'ouverture UTC → retrait de
l'offset systématique (Δclose médian, OHLC.dev − OANDA mid) → mesure du **résidu sur high/low**
(médiane / p95 / max, en pips/$ et en % d'ATR(14)) → accord des swings (fractal 5 barres ±1) →
complétude (barres manquantes, volume==0, spikes). Verdict prévu sur le p95 du résidu de mèches
(<10 % ATR 🟢 / 10–30 % 🟠 / >30 % 🔴).

> **Important** : même un test de qualité 🟢 ne lèverait pas les bloqueurs de la Section 3
> ci-dessous. La qualité numérique et le droit/transparence sont deux conditions **cumulatives**.

---

## 3. Commerciabilité & transparence — lecture des conditions

Lecture des pages publiques OHLC.dev (Terms, Pricing, FAQ, Home), 2026-06-18.
**Cette lecture est préliminaire ; seule une confirmation écrite du fournisseur fait foi.**

### 3.1 Classification des droits

| Droit requis | Verdict | Base (citation courte) |
|---|---|---|
| Usage commercial (plan payant) | 🟢 **Autorisé** | Terms §2/§3 : *« Pro, Ultra, and Mega plans are intended to support commercial applications »*. Free tier = non-commercial. |
| **Affichage des bougies aux abonnés payants** | 🟠 **Non écrit (ambigu)** | Aucune clause d'affichage nulle part. Sujet non traité. |
| Redistribution / revente | 🔴 **Restreint (flou)** | Terms §2 : *« Do not resell or redistribute raw API data as a competing service. »* Le qualificatif « as a competing service » est le nœud : un produit SMC B2C n'est sans doute pas un service de *données* concurrent, mais ce n'est pas écrit clairement. |
| Mise en cache / stockage en BD | 🟠 **Non écrit (ambigu)** | Aucune clause. (Ils vantent leur propre cache Redis, mais ne disent rien de *notre* droit de stocker.) |
| Suppression à la résiliation | 🟠 **Non écrit (ambigu)** | Aucune clause. |
| Attribution | ⚪ **Aucune trouvée** (probablement non requise, non confirmé) | Pas de clause d'attribution. |
| Volume / sièges | 🟢 Borné par le plan | Quotas requêtes (Pro 15 $ = 10 000/mois) + rate limits ; pas de notion de sièges. |

### 3.2 Provenance & exactitude (cœur du problème pour un produit d'honnêteté)

1. **Provenance NON divulguée** — nulle part OHLC.dev n'indique *quelle* source/LP fournit les
   prix XAUUSD/EURUSD, ni si les cotations sont en **bid / mid / ask**. Pour un produit SMC où
   l'on prétend montrer « le même graphe que ton écran », ne pas connaître la source rend
   l'offset systématique et l'alignement des mèches **non interprétables**.
2. **Exactitude déclinée par le fournisseur** — disclaimer explicite : *« Market data may not
   always be real-time or fully accurate and can differ from prices on official exchanges »* ;
   Terms §4 : *« does not guarantee uninterrupted accuracy, completeness, or timeliness »*.

### 3.3 Tarifs (référence)

| Plan | Prix | Requêtes/mois | Usage commercial |
|---|---|---|---|
| Basic (Free) | 0 $ | 500 | Non (perso/test) |
| **Pro** | **15 $/mois** | 10 000 | Oui (« commercial-ready ») |
| Ultra | 35 $/mois | 25 000 | Oui |
| Mega | 100 $/mois | 110 000 | Oui |

---

## 4. Questions OUI/NON pour le support (confirmation écrite = artefact qui fait foi)

1. Sous le plan Pro à 15 $/mois, ai-je le droit d'**afficher vos bougies XAUUSD et EURUSD à des
   abonnés payants** dans une application web B2C ? (OUI/NON)
2. Cet affichage compte-t-il comme « resell or redistribute raw API data as a competing
   service » interdit par vos Terms ? (OUI/NON)
3. Ai-je le droit de **mettre en cache et stocker l'historique** des bougies dans ma base de
   données ? (OUI/NON)
4. Une **attribution** (« Data by OHLC.dev ») est-elle requise à l'affichage ? (OUI/NON)
5. À la **résiliation**, dois-je supprimer les données déjà mises en cache/stockées ? (OUI/NON)
6. Quelle est la **source/provenance** des prix XAUUSD et EURUSD, et sont-ils en **bid, mid ou
   ask** ?
7. Y a-t-il une **limite de sièges/utilisateurs finaux** sous le plan Pro, ou seulement la
   limite de requêtes ?

---

## 5. Conclusion & recommandation

**Verdict commerciabilité : 🔴 MAUVAIS FIT pour un produit d'honnêteté — malgré le prix.**

Trois bloqueurs **structurels** (indépendants du test de qualité, qui n'a pas pu être exécuté) :

1. **Provenance non divulguée** — on ne peut pas affirmer « le même graphe que ton écran » au
   client si on ignore d'où viennent les prix et s'ils sont bid/mid/ask. Incompatible avec le
   positionnement « outil de compréhension augmentée / honnêteté ».
2. **Droit d'affichage non écrit** — l'affichage aux abonnés payants n'est couvert par **aucune
   clause** ; seule existe une interdiction floue de redistribution « as a competing service ».
   Risque juridique non borné sans confirmation écrite.
3. **Exactitude explicitement déclinée** par le fournisseur (« can differ from prices on
   official exchanges »), avec un **maillon RapidAPI supplémentaire** entre nous et la source.

Le tarif (~15 $/mois) ne compense pas ces trois défauts pour un produit dont la promesse
centrale est la fidélité et la transparence de l'affichage.

**Recommandation : privilégier un fournisseur qui DÉCLARE explicitement sa source** (LP/exchange,
bid/mid/ask) et qui couvre **par écrit** le droit d'affichage aux utilisateurs finaux — quitte
à payer plus cher. OHLC.dev ne devrait être (re)considéré que si son support confirme par écrit
les points 1, 2 et 6 ci-dessus (Section 4) de façon non ambiguë.

---

## 6. Réserves méthodologiques

- Section 2 (qualité numérique) **non testée** : verdict portant **uniquement** sur les
  conditions et la transparence. Une éventuelle bonne qualité numérique ne lèverait pas les
  bloqueurs de provenance/droit/exactitude.
- Lecture des Terms via rendu web (certaines réponses de la FAQ chargées en JS n'ont pas pu être
  lues). À compléter par le texte intégral et la réponse écrite du support avant tout engagement.
