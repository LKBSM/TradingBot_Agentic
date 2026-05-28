# Pivot de positionnement — Acte officiel post-audit algo

**Date** : 2026-05-27
**Type** : décision politique majeure
**Statut** : ✅ ACTÉ par utilisateur
**Justification** : audit algorithmique 2026-05-27 (note 3/10)
**Référence** : `docs/governance/AUDIT_ALGO_2026_05_27.md`

---

## Contexte

L'audit rigoureux de l'algorithme d'analyse du marché conduit le 2026-05-27 a révélé :

- **Scoring rule-based Pearson −0.023** (zéro pouvoir prédictif sur 924 trades 7 ans)
- **Backtest 7 ans PF 0.786, return −62 %, sous-performance −318 pp vs Buy & Hold XAU**
- **Tiers PREMIUM = 1 trade / 7 ans · STANDARD = 0 trade / 7 ans** (cosmétiques, empiriquement vides)
- **Brier score 0.2551 PIRE que baseline 0.2456** (pire qu'une constante)
- **LightGBM scorer existe** (`models/scoring_v3_lgbm.pkl` 81 KB) **mais non-wired** en prod
- **Coûts transactionnels NON inclus** dans les backtests (PF gonflé +0.10-0.20)
- **Look-ahead bug B2** confirmé dans `multi_timeframe_features.py:554-566`

**Conclusion audit** : le produit en l'état est **non vendable comme "système de trading profitable"**. Le claim "PF 1.30 [1.12-1.49]" et les tiers PREMIUM/STANDARD/WEAK sont mensongers tant qu'empiriquement non substanciés.

---

## Décision

### Volet B — Repositionnement immédiat (cette semaine)

#### B.1 — Repositionnement produit officiel

**Avant** : « M.I.A. Markets · système d'analyse algorithmique » (laissait place à interprétation "trading profitable")

**Après** :
> **M.I.A. Markets — Outil de compréhension augmentée des marchés Or & FX**
>
> *Apprendre. Comprendre. Décider en autonomie.*

- Plus de positionnement "système de trading"
- Posture pédagogique 100 % assumée
- Pas de promesse de gain, pas de claim de performance
- Compréhension > prédiction
- Le chatbot Sentinel et les sources RAG académiques sont l'argument principal, pas le scoring

#### B.2 — Pricing révisé

**Avant** : FREE / $29 (STARTER) / $79 (PRO) / $1990 (INSTITUTIONAL)
**Après** : **FREE / 9 € (DÉCOUVERTE) / 19 € (APPROFONDIE)** + INSTITUTIONAL retiré de la grille publique

| Tier | Avant | Après | Cible |
|---|---|---|---|
| **FREE** | $0 | **0 €** | Découverte produit |
| ~~STARTER $29~~ | $29/mo | **DÉCOUVERTE 9 €/mo** | Apprentissage pédagogique |
| ~~PRO $79~~ | $79/mo | **APPROFONDIE 19 €/mo** | Power-user éducation |
| ~~INSTITUTIONAL $1990~~ | grille publique | **Retiré grille publique** → "Contact us" / Calendly | B2B uniquement, sur demande |

**Justification** : prix ajusté à la **valeur réelle pédagogique** du produit (et non promesse de profit non substanciée). Cohérent avec la honnêteté de l'audit.

#### B.3 — Suppression tiers algorithmiques PREMIUM/STANDARD/WEAK

Les **tiers algorithmiques** PREMIUM/STANDARD/WEAK (tiers de signal selon score 0-100) sont **retirés du produit visible** :
- Plus de label "PREMIUM" / "STANDARD" / "WEAK" affiché au client
- Plus de promesse "signaux haute conviction"
- Conservé en interne uniquement pour debug + monitoring

Les **tiers commerciaux** (FREE / DÉCOUVERTE / APPROFONDIE) restent — mais ils débloquent du **contenu/profondeur**, pas une "qualité de signal".

#### B.4 — Retrait claims de performance non substanciés

Tous les claims suivants sont **retirés** ou **taggés `LEGAL-PENDING` / `IN-SAMPLE-NON-VALIDE`** dans toutes les copies, le mockup HTML, le webapp, les emails, les briefs :

| Claim | Statut |
|---|---|
| "Profit factor 1.30" | ❌ Retiré |
| "IC 95 % : 1.12 – 1.49" | ❌ Retiré |
| "Win rate 31.9 %" | ❌ Retiré |
| "329 setups historiques" | ❌ Retiré |
| "Walk-forward 7 ans" | ⚠️ Conservé MAIS qualifié "in-sample, OOS pending" |
| "Conviction calibrée" | ⚠️ Conservé MAIS qualifié "calibration en cours, validation OOS pending" |

**Remplacement positionnement** :
> *Outil d'analyse pédagogique en phase d'accès anticipé · Méthodologie publique · Validation statistique OOS en cours*

#### B.5 — Cap utilisateurs bootstrap conservé

- **Cap 50 abonnés payants total** (DÉCOUVERTE + APPROFONDIE) pendant M0-M3
- Revenue projeté max : 50 × 19 € = 950 €/mo (cas optimiste) ou plus probable 30 × 9 € + 20 × 19 € = 650 €/mo
- Marge ARR : ~7.8-11.4 k€/an au cap → suffit à financer 1 mois avocat fintech à M3-M6 si traction
- Cohérent avec stratégie bootstrap légal documentée

---

### Volet C — Réparation algorithmique en parallèle (3-4 semaines)

Tous les blockers du Sprint 1 du MASTER_PLAN sont **maintenus et étendus** :

| Blocker | Effort | Priorité |
|---|---|---|
| **#1** — Wire `scoring_v3_lgbm.pkl` (81 KB existant) en prod via `ConfluenceDetector` | 5j | P0 absolu |
| **#2** — Brancher `DynamicSpreadModel` + `DynamicSlippageModel` dans `_build_trade()` | 3j | P0 absolu |
| **#3** — Bootstrap IC 95 % + walk-forward CPCV + PBO + DSR | 2j | P0 absolu |
| **#4** — Patcher look-ahead B2 dans `multi_timeframe_features.py:554-566` | 1j | P0 absolu |

**Total effort Volet C** : ~11 jours dev étalés sur 3-4 semaines, en parallèle des autres terminaux.

---

## Gate de promotion vers prix premium

**Condition d'augmentation de prix au-delà de 9€/19€** :

Quand les 3 critères statistiques suivants sont validés simultanément après Volet C :

| Critère | Cible | Méthode |
|---|---|---|
| **Brier skill OOS** | **> +2 %** vs baseline naïf | Walk-forward CPCV out-of-sample |
| **DSR (Deflated Sharpe Ratio)** | **> 1.0** | López de Prado 2018 |
| **PBO (Probability of Backtest Overfitting)** | **< 0.5** | Bailey-López de Prado |

**Alors et seulement alors** :
- Possibilité de monter un tier au-dessus (≥ 29€/mo) avec **preuve statistique publiable**
- Re-claim "PF 1.30" et autres métriques **substanciés OOS**
- Réintroduire tier INSTITUTIONAL dans la grille publique
- Communication marketing renforcée

**Sinon** :
- Maintien permanent pricing FREE / 9€ / 19€
- Maintien positionnement "outil de compréhension augmentée"
- Pas de claim de performance, juste pédagogie
- Potentiellement pivot Scénario D (B2B-API brokers) si pas de traction B2C à M6

---

## Échéance Volet C — Re-évaluation Brier skill

**Date cible** : 2026-06-24 (4 semaines après aujourd'hui)

À cette date :
- Sprint 1 (Volet C) complet ou substantiel
- Modèle LightGBM entraîné en walk-forward CPCV
- Coûts transactionnels intégrés
- IC bootstrap calculé
- Look-ahead B2 patché
- **Rapport `reports/scoring_v2_OOS_validation.md` produit** avec Brier skill + DSR + PBO

**Décision à cette date** :
- Si **3 critères validés** → escalade pricing premium (≥ 29 €) + claim publié
- Si **< 3 critères** → maintien pricing 9€/19€ + positionnement éducatif
- Si **PF OOS < 1.0 même avec LGBM** → pivot Scénario D (B2B-API) ou abandon edge claim

---

## Propagation aux autres terminaux

Un brief dédié (`docs/governance/PROPAGATION_BRIEF_2026_05_27.md`) sera transmis à :

1. **Terminal frontend/landing** (autre instance Claude Code) : aligner copies, pricing, retirer claims
2. **Terminal légal low-cost** (si existe) : aligner CGU/Privacy templates avec nouveau positionnement

Le brief précise exactement quoi changer, où, et pourquoi.

---

## Signature

- ✅ **APPROUVÉ par Loukmane Bessam le 2026-05-27**
- Base : audit algo `AUDIT_ALGO_2026_05_27.md`
- Stratégie : B + C en parallèle
- Gate promotion : Brier > +2 % AND DSR > 1.0 AND PBO < 0.5

---

## Conséquences pratiques immédiates

### Sur les documents

| Document | Action |
|---|---|
| `MASTER_PLAN.md` | ✏️ Pricing révisé + positionnement actualisé + Blockers #2/#3/#4 ajoutés Sprint 1 |
| `PILLARS_PERFECTION.md` | ✏️ Pilier D actualisé (retrait tiers PREMIUM/STANDARD + claims) |
| `vague1_execution/copies/landing_copy.md` | ✏️ Retrait PF 1.30, 329 setups, IC |
| `vague1_execution/copies/pricing_copy.md` | ✏️ Nouvelle grille 9€/19€, INSTITUTIONAL retiré |
| `vague1_execution/copies/methodologie_copy.md` | ✏️ Stats taggées "in-sample, OOS pending" |
| `vague1_execution/copies/emails_lifecycle.md` | ✏️ Retrait des claims dans emails marketing |
| `vague1_execution/copies/chatbot_scripted_responses.md` | ✏️ Retirer chiffres PF/win rate des réponses Q4 |
| `vague1_execution/configurations/stripe_products.md` | ✏️ Nouveaux price IDs 9€/19€, INSTITUTIONAL en commentaire |
| `vague1_execution/configurations/tier_quotas.md` | ✏️ Quotas alignés DÉCOUVERTE/APPROFONDIE |
| `vague1_execution/briefs/DG-120_landing_hero.md` | ✏️ Hero sans claims chiffrés |
| `vague1_execution/briefs/DG-132_page_pricing.md` | ✏️ 3 tiers (FREE/9€/19€) + decoy retiré |
| `vague1_execution/briefs/DG-142_track_record_public.md` | ✏️ Tableau public avec disclaimer fort |
| `legal_bootstrap_strategy_2026_05_26.md` | ✏️ Pricing à jour, cap 50 abonnés |
| `decisions/2026-05-26_5_political_locks.md` | ✏️ Pricing décision révisée |
| `README.md` (governance) | ✏️ Référence vers ce doc |
| `mockups/v3/best_concept_demo.html` | ✏️ Retrait chiffres + nouveau pricing |
| `webapp/` (autre instance) | 📨 Brief propagation transmis |

### Sur l'autre instance Claude Code

Brief de propagation explicite à transmettre. Inclut :
- Copy changes à appliquer
- Nouveaux env vars (`PRICE_DECOUVERTE_MONTHLY_ID`, `PRICE_APPROFONDIE_MONTHLY_ID`)
- Retrait composants `<Premium>` / `<Standard>` / `<Weak>` tiers
- Nouveau wording landing + pricing
- 4 blockers ajoutés Sprint 1
