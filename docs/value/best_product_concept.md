# Smart Sentinel AI — Le meilleur concept produit B2C

**Date** : 2026-05-17 · **Auditeur** : trio quant senior + PM SaaS B2C + growth strategist
**Source** : `docs/value/client_relevance_review.md` (Livrable 1 validé) + `client_information_explained.txt`
**Contrainte structurante** : le **chatbot est le pilier de défensabilité** (Problème #4 = moat). Tous les concepts l'intègrent comme pilier, pas comme accessoire. Le choix se fait sur **comment** il s'articule avec l'indicateur, pas sur sa présence.

---

## 0. TL;DR pour le décideur

- **3 concepts crédibles** distincts, tous compatibles avec le pipeline algo immuable et la posture compliance (edge_claim=False) :
  - **Concept A — "Le quant de garde"** : chat-first radical. Le chat est le produit, l'indicateur est minimal.
  - **Concept B — "Smart Sentinel Co-Pilot"** : lecture institutionnelle traduite en hero (PF+IC) + chatbot permanent en pilier.
  - **Concept C — "Le radar honnête"** : radical simplicity. 3 chiffres en hero, le reste délégué au chat à la demande.
- **Choix défendu : Concept B** (score pondéré 4.60/5 vs 3.85 pour A et 4.25 pour C). Maintient les 3 pépites du Livrable 1 (PF+IC hero, calendar event, conviction calibrée), traite le chatbot comme pilier permanent (pas mode dépannage), reste différenciant en <10s, et exploite tous les segments retail (débutant occupé → power-user SMC) via progressive disclosure.
- **Concept B n'est pas le plus radical** (A l'est), il est le **plus défendable commercialement** : il garde tous les leviers de valeur sans sacrifier la clarté hero. La radicalité (A) impressionne mais filtre le marché ; la simplicité (C) rassure mais perd le moat.

---

## 1. Les 3 concepts en détail

### 🅰 Concept A — "Sentinel · Le quant de garde"

**Promesse en une phrase**
*« Le seul outil de trading qui vous donne un quant junior personnel, disponible 24/7, qui lit le marché et répond à vos questions en langage humain. »*

**Client cible précis**
Retail actif intermédiaire (1-3 ans d'expérience), 25-45 ans, FR-first puis EN. Trade XAU + EUR + indices. Aime apprendre, lit Substack quant, suit Twitter FinTwit, a déjà testé LuxAlgo et n'en peut plus du jargon SMC non expliqué. Recherche un *interlocuteur*, pas un outil de plus. Volume estimé adressable FR : ~3 000-5 000 individus (overlap fort avec wedge eval_25).

**Ce qui est mis en façade** (l'écran principal)
- Une **fenêtre de chat permanente plein écran**, avec contexte de marché injecté en haut (1 ligne : *« Lecture haussière XAU M15 · STRONG · FOMC dans 18h »*).
- 4-6 questions suggérées en chips cliquables ("Pourquoi cette lecture ?", "C'est quoi un retest armé ?", "Le FOMC change quoi ?").
- Le chat est l'interaction principale. L'indicateur est un *contexte enrichi* du chat.

**Ce qui est caché**
- 100% de la richesse algo (waterfall, conformal, régime, vol forecast, structure) n'apparaît QUE via question. Le client ne voit jamais un dashboard.

**Rôle exact du chatbot**
**Il EST le produit.** L'indicateur est l'API interne du chatbot. Le chatbot a accès en temps réel à tous les champs InsightSignalV2, peut décomposer le score, expliquer un régime, comparer avec setups historiques similaires, refuser de donner un ordre.

**Modèle de monétisation cohérent**
- FREE : 5 questions/jour, 1 actif (XAU)
- ANALYST $29/mois : 50 questions/jour, 4 actifs, chat illimité en lecture passive
- STRATEGIST $79/mois : illimité + alertes proactives par chat ("Hey, le régime vient de basculer en stress")
- *Pas de tier institutional* — c'est un produit conversationnel B2C pur.

**Forces**
- Différenciation **maximale** : aucun concurrent trading retail ne propose un chat comme produit principal. Position unique.
- Le chatbot devient **l'expérience** — rétention et stickiness probablement très élevés (chaque session = engagement).
- Compliance compatible : chat permet wording fluide, refus d'ordres incarné dans la conversation.
- Faisabilité : nécessite UX chat + prompt engineering — pas de modif algo.

**Risques**
- **Tue le hook commercial <10 secondes.** Un prospect qui ouvre la landing voit un chat vide — ne comprend pas la valeur. Demande un onboarding fort + démo conversationnelle.
- **Coût LLM non-borné** : illimité = inflation OPEX. Doit avoir cache sémantique agressif (déjà en place, mais hit rate 7.8% selon eval_05_09).
- **Niche réelle plus petite** que la promesse — les "retail qui aiment converser pour apprendre" sont surreprésentés sur Twitter, sous-représentés dans la base de paiement.
- **Réduit l'importance perçue des pépites algo** : si tout passe par chat, le client n'a aucun moyen de *voir* la richesse — il doit la *demander*. Risque de sous-évaluer le produit.

---

### 🅱 Concept B — "Smart Sentinel Co-Pilot" ✅ choix recommandé

**Promesse en une phrase**
*« L'analyse de marché de niveau institutionnel, traduite pour vous — et un quant qui répond à toutes vos questions, en temps réel. »*

**Client cible précis**
Spectre large du retail engagé : **débutant motivé** (besoin de comprendre) → **trader intermédiaire** (besoin de structure) → **power-user SMC** (besoin de justification). 25-55 ans. FR-first → EN. Volume adressable FR : ~15 000-30 000 individus (wedge eval_25 + extensions adjacentes). Compatible avec tous les segments du marché retail sérieux.

**Ce qui est mis en façade** (l'écran principal — hero hiérarchisé)
1. **HERO CARD permanent** (le seul élément de plein écran à 100% de visibilité) :
   - Lecture compacte ("Lecture haussière XAU M15") + label STRONG + badge "8 facteurs analysés"
   - **Track record honnête en hero** : *« 329 setups similaires depuis 2019 · 1.30€ gagnés pour 1€ perdu · IC 95% : 1.12-1.49 · walk-forward 7 ans »*
   - **Alerte event imminente** si applicable ("FOMC Minutes dans 2h47" en rouge si <2h)
   - **CTA permanent** : *« Demander à Sentinel »* (ouvre le chatbot avec contexte)
2. **6 cartes de soutien** en grille (immédiatement visibles, pas de scroll) :
   - Conviction calibrée (jauge 72/100 + intervalle conformel "marge d'erreur 54-82")
   - Régime marché ("Tendance haussière calme · risque de retournement : faible")
   - Volatilité ("Amplitude attendue +10% vs normal")
   - Structure ("Cassure 2391.5 · zone déséquilibre 2378-2381 · retest armé · invalidation 2378")
   - Session ("New York Overlap")
   - Lecture verbale (narrative_short)
3. **Footer compliance permanent** (discret) : *« Démonstration paper-trading · Lecture algorithmique éducative · Ne constitue ni un signal ni un conseil. »*

**Ce qui est caché** (accessible mais pas imposé)
- Détail waterfall des 8 composantes (waterfall chiffré I.* avec contributions + poids) → via chatbot OU clic "Détails"
- Données techniques brutes (`hmm_posterior`, `expected_run_length`, `coverage_alpha`) → masquées en B2C
- Méthodologie complète (walk-forward, bootstrap, ACI Gibbs-Candès) → page "Méthodologie" d'un clic
- `id` SHA-1, ISO timestamps bruts → footer technique optionnel

**Rôle exact du chatbot**
**Pilier permanent** — toujours visible (sidebar droite desktop, FAB flottant mobile), accessible d'un clic depuis n'importe quel élément. Mode d'accès principal aux secondaires (Sc1-Sc5) et au détail waterfall I.*. Il :
- **Définit le jargon** à la demande (un clic sur "FVG" → bulle "C'est quoi un FVG ?")
- **Décompose la conviction** ("Pourquoi 72 ?" → réponse pédagogique structurée des 8 contributions)
- **Contextualise l'event** ("Le FOMC change quoi ?" → réponse historique + impact sur régime)
- **Compare aux setups historiques** ("Ça ressemble à quoi historiquement ?" → stats J.* enrichies)
- **Refuse les ordres** ("Donc je dois acheter ?" → refus pédagogique + recadrage compliance)
- **Apprend des sources** (Phase 2B RAG : citations académiques inline)

C'est **le chatbot qui rend le dashboard supportable** pour le débutant et **le dashboard qui rend le chatbot précieux** pour l'expert. Symbiose, pas concurrence.

**Modèle de monétisation cohérent**
- **FREE** : 1 actif (XAU), 5 lectures/jour, chatbot limité (10 questions/jour), pas d'alerte event
- **ANALYST $29/mois** : 4 actifs (XAU + EUR + indices + 1 au choix), 30 lectures/jour, chatbot 100 questions/jour, alertes event
- **STRATEGIST $79/mois** : 6 actifs, illimité, chatbot illimité (cache sémantique + Sonnet sur questions complexes), narrative_long avec sources RAG, exports CSV
- **INSTITUTIONAL $1990/mois** : API B2B brokers/family offices, JSON complet (8 composantes + métriques internes), webhook
- Cohérent avec eval_27 (grille recommandée FREE/$29/$79/$1990), avec INSTITUTIONAL pricé à valeur réelle (vs $149 sous-pricé ×13)

**Forces**
- **Maintient toutes les pépites** identifiées au Livrable 1 (H0 hero + 6 soutiens + 5 secondaires via chat).
- **Clarté <10s** : le HERO CARD dit en une vue *« lecture haussière forte, 329 setups historiques, PF 1.30, FOMC dans 2h »* — c'est immédiatement actionnable et différencié.
- **Levier chatbot maximal** : pilier permanent, pas mode dépannage. Justifie l'OPEX LLM.
- **Compliance native** : footer permanent + posture éducative + refus d'ordres incarné dans le chat.
- **Faisabilité totale** : zéro modif algo, juste recomposition de la même donnée + UX + prompt engineering chat.
- **Scalable cross-segments** : le débutant lit le hero + chatte, l'expert SMC lit la structure et ignore le chat, le power-user creuse via chat les composantes. Une UI, trois usages.
- **Compatible évolution** : sources RAG Phase 2B s'intègrent naturellement dans les réponses chat (citations inline).

**Risques**
- **Plus complexe à designer** que C (3 chiffres). Demande arbitrages UX réels (densité hero, hiérarchie typo, comportement chat).
- **Risque "encore un dashboard"** si la hero card n'est pas radicalement plus claire et plus honnête que la concurrence — le PF+IC en hero doit être traité comme un *poster*, pas comme une ligne de stats.
- **OPEX LLM significatif sur tier STRATEGIST illimité** — atténué par cache sémantique (à pousser de 7.8% à 30%+ via bump SCORE_BUCKET_PTS, cf. eval_05_09).
- **Le chatbot doit être bon dès le jour 1**. Un chatbot qui hallucine ou répond mal tue la promesse. Demande prompt engineering soigné + garde-fous post-génération (déjà en place : `contains_forbidden_token`).

---

### 🅲 Concept C — "Sentinel · Radar honnête"

**Promesse en une phrase**
*« 3 chiffres pour comprendre le marché. 1 quant pour aller plus loin. »*

**Client cible précis**
Retail **occupé** ou **débutant timide**. 30-55 ans. Pro avec une heure le soir pour trader, ou débutant intimidé par les dashboards. Cherche du calme, pas de la richesse. Volume adressable FR : ~20 000-40 000 individus, mais valeur unitaire plus faible (paie moins, churn plus haut).

**Ce qui est mis en façade**
- **3 chiffres et un mot, plein écran** :
  1. **Lecture** : "HAUSSIÈRE" (vert) / "BAISSIÈRE" (rouge) / "ILLISIBLE" (gris)
  2. **Conviction** : 72 (avec mini-jauge)
  3. **Confiance historique** : 1.30 (avec mini-IC 1.12-1.49 en dessous)
- **1 alerte event** si applicable ("FOMC dans 2h")
- **1 bouton CTA central** : *« Pourquoi ? »* → ouvre le chatbot

**Ce qui est caché**
- Tout le reste : régime, vol, structure, breakdown, conformal, sentiment. Accessible uniquement via chat ou onglet "Détails" (très peu visité).

**Rôle exact du chatbot**
**Mode dépannage** — invoqué par "Pourquoi ?" ou questions explicites. Pas permanent visuellement (icône en bas). Le chat porte 100% de la richesse, mais reste optionnel pour la lecture quotidienne.

**Modèle de monétisation cohérent**
- FREE : 1 actif, 3 lectures/jour, 3 questions/jour
- ANALYST $19/mois (sous-pricé volontairement vs B) : 3 actifs, illimité lectures, 30 questions
- *Pas de tier au-dessus* — produit volontairement simple, monétisation faible.

**Forces**
- **Clarté <10s imbattable** : 3 chiffres, c'est mieux que tout concurrent.
- **Compliance facile** : peu de surface, peu de jargon, peu de risque de glissement promesse.
- **Onboarding zéro** : compréhensible sans tutoriel.
- **Faisabilité maximale** : sous-ensemble du existant.

**Risques**
- **Perd l'argument "8 facteurs analysés"** en façade → ressemble à un signal Telegram à $20/mois. Difficile de défendre la prime de prix.
- **Sous-valorise le chatbot** : mode dépannage = engagement faible = churn élevé. Le chatbot n'est plus le moat, c'est une feature.
- **Sous-monétise** le produit : tier ceiling $19-29/mois max, vs $79-1990 sur B. ARPU plus faible × churn plus haut = LTV inférieure.
- **Tue la défensabilité long terme** : un concurrent peut copier "3 chiffres + chat" en 6 mois. La complexité hiérarchisée de B est plus difficile à imiter.
- **Frustration power-users SMC** (wedge eval_25) : ne verront pas leurs niveaux BOS/FVG/OB en façade → abandonneront pour LuxAlgo.

---

## 2. Choix défendu — Concept B "Smart Sentinel Co-Pilot"

### 2.1 — Scoring pondéré sur les 6 critères

| Critère | Pondération | Concept A — Quant de garde | Concept B — Co-Pilot | Concept C — Radar honnête |
|---|---|---|---|---|
| **1. Taille de marché retail adressable** | 15% | 3/5 (niche conversation) | **5/5** (large, multi-segments) | 4/5 (large occasionnel, LTV faible) |
| **2. Différenciation vs concurrence** | 20% | **5/5** (jamais vu) | 4/5 (mélange unique) | 3/5 (déjà tenté par d'autres) |
| **3. Clarté valeur <10s** | 20% | 2/5 (chat vide ≠ hook visuel) | 4/5 (hero card forte) | **5/5** (3 chiffres = killer) |
| **4. Compatibilité compliance** | 10% | 4/5 (chat peut déraper) | **5/5** (footer + posture incarnée) | **5/5** (peu de surface) |
| **5. Effet de levier du chatbot** | 20% | **5/5** (chat = produit) | **5/5** (chat = pilier permanent) | 4/5 (chat = mode dépannage) |
| **6. Faisabilité sans toucher code algo** | 15% | 4/5 (chat + UX) | **5/5** (recomposition pure) | **5/5** (sous-ensemble) |
| **Total pondéré /5** | 100% | **3.85** | **4.60** ✅ | **4.25** |

**Calculs détaillés**
- A : 0.15×3 + 0.20×5 + 0.20×2 + 0.10×4 + 0.20×5 + 0.15×4 = 0.45 + 1.00 + 0.40 + 0.40 + 1.00 + 0.60 = **3.85**
- B : 0.15×5 + 0.20×4 + 0.20×4 + 0.10×5 + 0.20×5 + 0.15×5 = 0.75 + 0.80 + 0.80 + 0.50 + 1.00 + 0.75 = **4.60**
- C : 0.15×4 + 0.20×3 + 0.20×5 + 0.10×5 + 0.20×4 + 0.15×5 = 0.60 + 0.60 + 1.00 + 0.50 + 0.80 + 0.75 = **4.25**

### 2.2 — Pourquoi B et pas A (le plus radical)

A est **commercialement séduisant pour le pitch** ("le chat est le produit, jamais vu en trading") mais **opérationnellement fragile** :
- Le **hook landing** s'écroule : un prospect qui ouvre une landing avec une fenêtre de chat vide ne comprend pas la valeur. Toutes les SaaS B2C qui ont essayé le "chat-first" ont dû ajouter un onboarding visuel lourd (Intercom, Drift, Replika) — A vaudra le même.
- A **réduit la perception de richesse algo** : si la décomposition 8-composantes / le PF historique / le conformal interval ne sont visibles que via question, le client ne sait pas ce qu'il achète. C'est exactement le Problème #3 du Livrable 1 (pépites noyées), version pire.
- **OPEX LLM non-borné** : l'illimité conversationnel sur tier $29 est intenable financièrement sans cap dur (qui casse alors la promesse).
- **Le marché adressable est plus petit que la promesse** : les retail qui aiment converser pour apprendre sont surreprésentés sur Twitter / FinTwit, sous-représentés dans la base de paiement effective.

A est un **excellent concept secondaire**, à creuser éventuellement en spin-off ("Smart Sentinel Chat Edition") pour un segment niche.

### 2.3 — Pourquoi B et pas C (le plus simple)

C est **séduisant pour le débutant** mais **stratégiquement perdant** :
- C **tue le hero PF+IC** au sens où on doit le réduire à un chiffre nu (1.30), perdant l'intervalle et le contexte 329 setups. Le hero "1.30" sans contexte ressemble à *n'importe quoi*.
- C **commoditise le produit** : "3 chiffres + un bouton" ressemble à un signal Telegram premium. Le PM SaaS le sait : on ne défend pas une prime de prix avec moins d'info que la concurrence.
- C **frustre la niche wedge SMC FR** (eval_25 : "XAU SMC FR-first $20-49/mo") — ils veulent voir leurs niveaux BOS/FVG/OB en façade. C les pousse vers LuxAlgo.
- C **dégrade le chatbot en accessoire** : si le chat est optionnel et caché, il ne devient jamais le moat. Le Problème #4 du Livrable 1 reste non résolu.
- C **plafonne la monétisation** : tier ceiling $19-29 = ARPU faible = LTV insuffisante pour payer le COGS LLM correctement.

C est un **bon mode dégradé** (vue "Simple" ou "Mode Lite") **à proposer en option dans B**, pas un produit principal.

### 2.4 — Pourquoi B est la bonne réponse stratégique

B résout les 5 problèmes du Livrable 1 simultanément :

| Problème Livrable 1 | Comment B le résout |
|---|---|
| #1 Surcharge cognitive | Hero card forte + 6 cartes hiérarchisées + tout le reste délégué chat → 3 niveaux clairs |
| #2 Jargon brut | Wording grand public en façade + chatbot qui définit chaque terme à la demande (clic FVG → bulle pédagogique) |
| #3 Pépites noyées | PF+IC en hero permanent, calendar event en hero conditionnel (≤2h) → la pépite n°1 est visible en <5s |
| #4 Chatbot accessoire | Chatbot **pilier permanent** (sidebar/FAB), mode d'accès principal aux secondaires + waterfall + définitions |
| #5 Valeur invisible <10s | Hero card lit *« 329 setups · 1.30 [1.12-1.49] »* + lecture + event → hook commercial frontal et factuel |

B est aussi **le plus compatible avec la roadmap Phase 2B** (sources RAG K.4 = pépite future) : les citations académiques s'intègrent naturellement dans les réponses chat, renforçant le moat anti-blackbox.

B **n'est pas le plus radical visuellement**, mais c'est le **plus défendable commercialement** : il garde tous les leviers de valeur, exploite le chatbot comme moat, et reste lisible en <10s.

### 2.5 — Nom de marque

**"Smart Sentinel Co-Pilot"** (ou variante FR : *« Smart Sentinel · Co-Pilote »*)

- Référence reconnaissable (GitHub Copilot, Microsoft Copilot) → positionne instantanément la promesse "assistant intelligent, vous restez aux commandes".
- "Co-" = collaboration explicite (vous décidez, je vous aide à comprendre) → compatible compliance.
- "Pilot" = navigation, lecture, contexte — pas exécution.
- Marketable, prononcable en FR et EN, défendable en marque déposée.

### 2.6 — Tagline & sous-tagline

- **Tagline principale (10s hook)** :
  *« L'analyse de marché de niveau institutionnel, traduite. Plus un quant qui répond à toutes vos questions. »*

- **Sous-tagline factuelle (5s scroll)** :
  *« 329 setups historiques · profit factor 1.30 [1.12-1.49] · walk-forward 7 ans · honnête sur ce qu'il ne sait pas. »*

- **Tagline anti-finfluenceur** (page méthodologie / about) :
  *« Nous ne vous disons pas quoi faire. Nous vous donnons les meilleurs outils pour décider. »*

### 2.7 — Architecture multi-vues (3 modes, 1 moteur) — résout la surcharge sans amputer la richesse

**Problème produit identifié** : trop d'information bien présentée reste *trop* d'information pour un prospect en 10 secondes, ou pour un débutant occupé qui consulte en mobilité. Mais l'amputer (Concept C) tue le moat. **Solution : 3 vues, même donnée, choix utilisateur.**

Le pipeline algo produit le même `InsightSignalV2` (immuable). Ce qui change, c'est **ce qui est mis devant les yeux du client par défaut**, et **où est délégué le reste**. Trois modes, switchables d'un clic, mémorisables par utilisateur.

#### 🅵 Mode FOCUS — *« Le coup d'œil »* (5-10s, vue indicateur de base)

**C'est ce mode qui définit visuellement le produit pour un prospect, un débutant occupé, un utilisateur mobile, ou un signal Telegram.**

Ce qui s'affiche, plein écran, **rien d'autre** :
- **Direction + label** ("HAUSSIÈRE · STRONG", couleur sémantique vert/rouge/gris)
- **Conviction calibrée** (un chiffre 72, mini-jauge sous-jacente)
- **PF historique + IC** (1.30 [1.12-1.49] · 329 setups · walk-forward 7 ans) — *unique chiffre de track record honnête*
- **Alerte event imminente** si applicable (≤4h : "FOMC dans 2h47" rouge ; sinon : caché)
- **1 ligne narrative_short** (sous le hero, posée, pédagogique)
- **1 bouton "Demander à Sentinel"** (ouvre chat overlay, ne remplace pas la vue)
- **Footer compliance** (1 ligne discrète)

Ce qui est délégué :
- Tout le reste (6 cards, waterfall, structure SMC, conformal, sources) → bouton **« Vue Co-Pilot »** en haut, ou via chat (Sentinel sait tout, suffit de demander)

Persona type : débutant motivé, trader occupé, prospect landing, lecteur Telegram.

#### 🅒 Mode CO-PILOT — *« La lecture guidée »* (30-60s, vue par défaut webapp)

**C'est le mode équilibré, déjà décrit dans la section 1 (Concept B principal).**

Ce qui s'affiche :
- Hero card (FOCUS condensé) + 6 cartes de soutien (Conviction, Régime, Volatilité, Structure SMC, Session, Lecture verbale) + Chatbot sidebar permanent

Ce qui est délégué :
- Waterfall 8 composantes détaillé, conformal interval visualisé, sources RAG → bouton **« Vue Expert »**, ou chat

Persona type : trader engagé intermédiaire/avancé, swing trader sérieux, power-user SMC qui veut le contexte.

#### 🅴 Mode EXPERT — *« Le cockpit complet »* (2-5min, vue institutionnelle)

**C'est le mode qui justifie le tier STRATEGIST $79 et l'API B2B INSTITUTIONAL $1990.**

Ce qui s'affiche, en plus du Co-Pilot :
- **Waterfall 8 composantes** chiffré, avec contributions individuelles (Smart Money, Volatilité, Liquidité, Régime, Sessions, Multi-TF, Technical, News) en barres horizontales avec poids et scores bruts
- **Conformal interval visualisé** (marge d'erreur 54-82 sur conviction 72, avec bande grise sur la jauge)
- **Indicateurs régime techniques** (HMM posterior 3-états, BOCPD `expected_run_length`, `jump_ratio`)
- **Stats historiques détaillées** (J.* enrichies : win-rate, drawdown max, exposure time, profil de distribution)
- **Sources RAG** (citations académiques inline dans le narratif Phase 2B)
- **Replay historique** (chart H4/H1/M15 avec annotations BOS, FVG, OB, event timeline) — *tier STRATEGIST*
- **Chatbot mode Pro** (peut interroger features brutes : *« montre-moi le delta vol forecast vs naive »*)

Persona type : power-user SMC professionnel, prop trader, gestionnaire en family office, intégrateur API B2B, audit interne.

#### 📋 Récapitulatif distribution multi-surfaces

| Surface | Modes disponibles | Mode par défaut | Logique |
|---|---|---|---|
| **Telegram** | FOCUS uniquement | FOCUS | Contrainte 800 chars + format conversationnel = FOCUS naturel ; le client peut cliquer un lien vers webapp pour CO-PILOT/EXPERT |
| **Webapp** | FOCUS / CO-PILOT / EXPERT (toggle persistant en haut à droite, sauvegardé par utilisateur) | CO-PILOT | Le client choisit son mode et il est mémorisé ; toggle visible mais discret |
| **API B2B** | EXPERT (JSON complet `InsightSignalV2` v2.1.0) | — | Le client B2B compose sa vue (broker, family office) à partir du JSON exhaustif |
| **Email digest** | FOCUS (avec lien vers CO-PILOT/EXPERT) | FOCUS | Format mail = synthèse, profondeur via lien clickable |

#### 🎯 Pourquoi cette architecture résout le problème commercial

| Risque commercial | Comment l'architecture multi-vues le neutralise |
|---|---|
| **Prospect intimidé en 10s** | Landing montre FOCUS par défaut → vue épurée, hook clair, 3 chiffres importants visibles |
| **Débutant noyé** | Mode FOCUS reste maison ; il découvre CO-PILOT quand il est prêt (CTA "voir le détail") ; chatbot toujours là |
| **Power-user frustré** par sur-simplification | Toggle "EXPERT" d'un clic en haut → tout déballé, niveau Bloomberg Terminal |
| **Mobile → impossibilité de lire un dashboard** | FOCUS = mobile-first natif ; CO-PILOT/EXPERT = ne s'affichent qu'au-delà d'un breakpoint, ou en orientation paysage |
| **B2B veut tout le JSON** | API = EXPERT direct, sans contrainte UI |
| **Argument anti-blackbox** ("8 facteurs analysés") | Badge "8 facteurs analysés ✓" visible dès FOCUS → la richesse est *signalée*, son détail est *à la demande* |

#### 💰 Cohérence tarifaire avec les modes

| Tier | Modes débloqués | Logique | Prix mensuel |
|---|---|---|---|
| **FREE** | FOCUS seul, 1 actif (XAU), 3 lectures/jour | Hook + démonstration valeur | $0 |
| **ANALYST** | FOCUS + CO-PILOT, 4 actifs, 30 lectures/jour, chat 100Q/jour, alertes event | Trader retail engagé | $29 |
| **STRATEGIST** | + EXPERT (waterfall, conformal, replay, RAG sources), 6 actifs, illimité, chat illimité, exports CSV | Power-user / pro retail | $79 |
| **INSTITUTIONAL** | API B2B (JSON EXPERT brut), webhooks, SLA, multi-asset, license commerciale | Broker / family office | $1990 |

**Le mode = la valeur perçue = le prix.** Le client paie pour la profondeur de vue, pas pour le nombre de signaux. C'est cohérent avec la posture compliance (`edge_claim = False`) : on ne vend pas plus de profit avec un tier supérieur, on vend plus de **compréhension**.

---

## 3. Architecture du concept B en synthèse — 3 modes en une UI

```
┌────────────────────────────────────────────────────────────────────────┐
│  HEADER  Smart Sentinel Co-Pilot · XAUUSD M15 ▾ · FR ▾ ·              │
│          MODE : [ FOCUS ] [ CO-PILOT ] [ EXPERT ]                      │
└────────────────────────────────────────────────────────────────────────┘

─── MODE FOCUS ─── (≈ vue indicateur de base, mobile-first) ───────────────
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│       🟢 LECTURE HAUSSIÈRE · STRONG     ✓ 8 facteurs analysés         │
│       XAU/USD · M15 · valable 2h47                                    │
│                                                                        │
│       ┌──── CONVICTION ────┐     ┌──── TRACK RECORD HONNÊTE ────┐    │
│       │       72           │     │   1.30€ pour 1€ perdu        │    │
│       │  ▮▮▮▮▮▮▮▯▯▯ /100   │     │   IC 95% : 1.12 - 1.49       │    │
│       │  marge ±14         │     │   329 setups · 7 ans         │    │
│       └────────────────────┘     └──────────────────────────────┘    │
│                                                                        │
│       ⚠ FOMC Minutes dans 2h47 — volatilité attendue +18%             │
│                                                                        │
│       « Cassure haussière confirmée, retest armé entre 2378 et 2381   │
│         dollars. À surveiller : compte-rendu FOMC dans 2h47. »        │
│                                                                        │
│       [💬 Demander à Sentinel]   [⤢ Voir le détail (Co-Pilot) →]     │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘

─── MODE CO-PILOT ─── (vue par défaut webapp, 30-60s lecture) ─────────────
┌──────────────────────────────────────────────────────┬─────────────────┐
│  HERO CARD (résumé FOCUS condensé)                   │                 │
│  ┌──────────────────────────────────────────────┐   │   CHATBOT       │
│  │ 🟢 LECTURE HAUSSIÈRE · STRONG               │   │   SIDEBAR       │
│  │ 1.30 [1.12-1.49] · 329 setups               │   │   PERMANENT     │
│  │ ⚠ FOMC dans 2h47                            │   │                 │
│  └──────────────────────────────────────────────┘   │   ┌──────────┐  │
│                                                      │   │ Posez une │  │
│  6 CARTES DE SOUTIEN                                │   │ question  │  │
│  ┌────────┐ ┌────────┐ ┌────────┐                   │   └──────────┘  │
│  │Convict.│ │ Régime │ │Volatil.│                   │                 │
│  │ 72/100 │ │tendance│ │ +18%   │                   │  Suggestions :  │
│  │ ±14    │ │ calme  │ │vs norm.│                   │  • Pourquoi 72 ?│
│  └────────┘ └────────┘ └────────┘                   │  • Retest armé ?│
│  ┌────────┐ ┌────────┐ ┌────────┐                   │  • FOMC change ?│
│  │Structur│ │Session │ │Lecture │                   │  • Historique ? │
│  │BOS 2391│ │ NY Ovr │ │verbale │                   │  • Acheter ?    │
│  │FVG 2378│ │        │ │ "..."  │                   │                 │
│  └────────┘ └────────┘ └────────┘                   │                 │
│                                                      │                 │
│  [⤢ Voir le détail technique (EXPERT) →]            │                 │
└──────────────────────────────────────────────────────┴─────────────────┘

─── MODE EXPERT ─── (cockpit complet, 2-5min lecture, STRATEGIST $79) ─────
┌──────────────────────────────────────────────────────┬─────────────────┐
│  HERO + 6 CARDS (Co-Pilot)                          │   CHATBOT       │
├──────────────────────────────────────────────────────┤   MODE PRO      │
│  WATERFALL 8 COMPOSANTES (contributions chiffrées)  │                 │
│  Smart Money     ████████████████ +24.5 (poids 25%)  │   Peut inter-   │
│  Volatilité      ███████████      +14.0 (poids 15%)  │   roger les     │
│  Multi-Timeframe ████████          +9.5 (poids 12%)  │   features      │
│  Liquidité       ██████            +6.0 (poids 10%)  │   brutes :      │
│  Sessions        ████              +4.0 (poids  8%)  │   • montre delta│
│  Régime          ██                +1.0 (poids 15%)  │     vol fcst    │
│  Technical       ▏                 +0.5 (poids 10%)  │   • posterior   │
│  News            ▏                 +0.0 (poids  5%)  │     HMM ?       │
│                                                      │   • run length  │
├──────────────────────────────────────────────────────┤     BOCPD ?     │
│  CONFORMAL INTERVAL  [conviction 72 · marge 54-82]   │                 │
│  ▮▮▮▮▮▮▮▯▯▯  ── couverture nominale 95% ── ACI ON   │                 │
├──────────────────────────────────────────────────────┤                 │
│  STATS HISTORIQUES J.* ENRICHIES                     │                 │
│  N=329 · WR 41.6% · DD max 8.4% · Exp time 4h12     │                 │
│  Distribution P&L : skew +0.34 · kurtosis 3.8       │                 │
├──────────────────────────────────────────────────────┤                 │
│  SOURCES RAG (citations académiques)                 │                 │
│  • Lopez de Prado 2018 ─ CPCV walk-forward          │                 │
│  • Corsi 2009 ─ HAR-RV volatility                   │                 │
│  • Gibbs & Candès 2021 ─ ACI conformal coverage     │                 │
├──────────────────────────────────────────────────────┤                 │
│  REPLAY HISTORIQUE (chart annotated H4/H1/M15)       │                 │
└──────────────────────────────────────────────────────┴─────────────────┘

──────────────────────────────────────────────────────────────────────────
  FOOTER  Démonstration paper-trading · Lecture algorithmique éducative
          Ne constitue ni un signal ni un conseil · Walk-forward 7 ans
──────────────────────────────────────────────────────────────────────────
```

**Principe de design transversal** :
- **Un même `InsightSignalV2`**, trois projections visuelles → zéro duplication algo, juste UX.
- **Toggle persistant** (cookie / preference utilisateur) : un débutant qui choisit FOCUS reste en FOCUS jusqu'à ce qu'il bascule lui-même.
- **Le bouton "Demander à Sentinel"** est présent dans les 3 modes, à des degrés de prégnance variables (CTA isolé en FOCUS, sidebar permanente en CO-PILOT, panneau pro en EXPERT).
- **Mobile** : seul FOCUS s'affiche par défaut sur viewport <768px ; CO-PILOT/EXPERT accessibles via toggle mais avec scroll vertical (pas de sidebar parallèle).
- **Compliance footer** identique dans les 3 modes, jamais masqué.

---

## 4. Ce que démontrera le Livrable 3 (démo HTML)

La démo `mockups/v3/best_concept_demo.html` (single-file, sans CDN) devra prouver concrètement :

1. **Section 1 — Le pitch** : tagline forte + comparatif visuel "Avant (Telegram brut actuel) vs Maintenant (hero card)" + 3 arguments différenciation.
2. **Section 2 — Lecture de marché repensée AVEC TOGGLE 3 MODES** :
   - Toggle interactif **FOCUS / CO-PILOT / EXPERT** en haut de la section.
   - **FOCUS** : vue épurée 5-10s (direction + conviction + PF/IC + alerte event + narratif court).
   - **CO-PILOT** : hero condensé + 6 cards + chatbot sidebar (vue par défaut).
   - **EXPERT** : tout le Co-Pilot + waterfall 8 composantes + conformal visualisé + stats J.* enrichies + sources RAG.
   - Données XAU M15 inspirées du .txt (BOS 2391.5, FVG 2378-2381, conviction 72, régime trend bullish, FOMC dans 2h47).
3. **Section 3 — Chatbot intégré fonctionnel** : sidebar/widget chat avec 6 questions scriptées (réponses pré-écrites JS) :
   - "Pourquoi la conviction n'est que de 72 ?" → décomposition 8-composantes pédagogique
   - "C'est quoi un retest armé, en simple ?" → définition + analogie
   - "Le FOMC dans 2h47, ça change quoi ?" → contextualisation event
   - "Ça ressemble à quoi historiquement, ce setup ?" → stats J.* enrichies
   - "Donc je dois acheter ?" → **refus pédagogique** visible (compliance + posture)
   - "Quelle est ta marge d'erreur sur 72 ?" → traduction conformal en langage humain
4. **Section 4 — Pourquoi ça se vend** :
   - Tableau monétisation 4-tiers avec **modes débloqués par tier** (FREE=FOCUS, ANALYST=FOCUS+CO-PILOT, STRATEGIST=+EXPERT, INSTITUTIONAL=API)
   - 3 différenciateurs commerciaux (multi-vues, chatbot moat, honnêteté statistique)
   - Comparatif vs LuxAlgo / signaux Telegram / TradingView indicators.

Design : finance premium, sobre, crédible. Pas de crypto-bro, pas de neon. Inspirations : Bloomberg Terminal (sobriété EXPERT), Linear (densité élégante CO-PILOT), Stripe (clarté FOCUS), Pitchbook (track record honnête en hero).

---

## 5. Validation utilisateur

Le concept B est désormais consolidé avec l'architecture multi-vues (3 modes) qui résout la tension richesse vs surcharge sans amputer le moteur. L'utilisateur a explicitement validé la nécessité de cette architecture le 2026-05-18.

**Direction d'exécution** :
- ✅ Concept B retenu (4.60/5)
- ✅ Multi-vues FOCUS / CO-PILOT / EXPERT intégrées (résout problème surcharge sans toucher algo)
- ✅ Toggle 3 modes sera la pièce centrale de la démo HTML (Livrable 3)
- ▶️ Livrable 3 en cours : `mockups/v3/best_concept_demo.html`
- ⏭️ Livrable 4 ensuite : recos enrichissement P0/P1/P2
