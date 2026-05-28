# 5 décisions politiques à acter — M.I.A. Markets

> ⚠️ **PRICING D3 RÉVISÉ 2026-05-27** — La décision D3 originale (FREE/$29/$79/$1990) a été **remplacée** par **FREE / 9 € / 19 €** + INSTITUTIONAL retiré grille publique, suite à l'audit algorithmique `AUDIT_ALGO_2026_05_27.md`. Document actuel : `docs/governance/decisions/2026-05-27_pivot_positioning_audit.md`. Les décisions D1, D2, D4, D5 restent valides.

**Date proposée** : lundi 2026-06-01 (semaine S1 du plan révisé)
**Auteur du brief** : second instance Claude Code (revue critique decision_gate_review_v2)
**Statut** : ✅ **4/5 SIGNÉES + 1/5 REFORMULÉE EN STRATÉGIE BOOTSTRAP** (2026-05-26)
**Décisions actées** : 1, 3, 4, 5 = APPROUVÉES · 2 = DEFER M3 avec stratégie provisoire (cf. `legal_bootstrap_strategy_2026_05_26.md`)

> **Document opérationnel à signer avant tout démarrage du plan d'exécution révisé.**
> Aucune de ces décisions ne demande de dev. Toutes débloquent la roadmap entière (Vague 1 S1-S6).
> Si tu ne tranches pas, le code peut continuer, mais la roadmap reste incohérente.

---

## Décision 1 — Vision B (narrative-first) confirmée par engagement écrit

### Enjeu
Sans décision écrite formelle dans `reports/governance/kill_criteria_board.md`, la tentation de retour Vision A (RL trading bot) reste vivace. Vision A = 320h Phase 2A, Vision B = 320h Phase 2B. **Aller-retour = 6 mois perdus.**

Le verdict A1 du 2026-05-01 a été statistiquement définitif (DSR=0.000, PBO=0.5, CPCV PF=1.008 sur 7 ans XAU walk-forward). Le procès est tranché empiriquement, il reste à honorer le verdict.

### Option recommandée
**Confirmer Vision B par écrit.** Engagement formel :
- Pas de retry Vision A pendant ≥ 90 jours (à partir du 2026-06-01)
- Réouverture du débat conditionnée à : forward-test PF > 1.30 sur ≥ 90 jours paper-trading public + DSR > 1.0 + PBO < 0.5
- Probabilité d'occurrence des conditions estimée : 5-10%

### Si tu ne tranches pas
Sous la pression "pourquoi pas réessayer A1 avec X feature ?", le pivot Vision B perd sa cohérence. Roadmap incohérente = exécution diffuse = brûlage de runway sans direction.

### Alternative défendable
Aucune. A1 a été testé sur 7 ans XAU walk-forward CPCV. Statistiquement définitif.

### Signature

- ✅ **APPROUVÉ 2026-05-26** : engagement Vision B 90 jours min, conditions de retry codifiées dans `reports/governance/kill_criteria_board.md`
- ☐ Discuter modifications avant signature

Signé : **Loukmane Bessam** Date : **2026-05-26**

---

## Décision 2 — Démarrage RFQ avocat fintech FR (budget 3-5 k€)

### Enjeu
Sans CGU/CGV/Privacy signées par un avocat fintech FR ≥ 5 ans pratique, **Stripe live (DG-043) est bloqué juridiquement**. Lead time minimum :
- RFQ 3 cabinets : 1 semaine
- Sélection : 3-5 jours
- Relecture + signature : 2-3 semaines
- **Total = 5-6 semaines avant Stripe live ouvert**

Tant que tu ne démarres pas, **aucune date de premier paiement** ne peut être planifiée. Trésorerie consommée pendant le glissement = brûlée.

### Option recommandée
**GO immédiat S1 (semaine du 2026-06-01).** Budget 3-5 k€ engagé cette semaine.
- 3 RFQ envoyés en parallèle lundi : Lexing, Hashtag Avocats, Couvrelles & Marchand-Berdat (ou cabinets équivalents)
- Critères : ≥ 5 ans pratique fintech FR, ≥ 3 références B2C SaaS, parle UE 2024/2811 + MiFID + RGPD
- Sélection sous 5 jours, brief envoyé pour relecture sous 2 semaines

### Si tu ne tranches pas
Tout le calendrier Stripe live glisse de 4-6 semaines. Décision DG-043 (Stripe live activation) impossible à phaser.

### Alternative défendable (non recommandée)
Utiliser un cabinet DPO-as-a-service (Privacy seul, ~1500€) + CGU/CGV générique sans relecture spécifique fintech. **Risque** : fermeture compte Stripe sur litige client. Coût asymétrique = perte massive.

### Signature

- ☐ ~~APPROUVÉ : budget 3-5 k€ engagé, RFQ envoyés cette semaine~~ **NON RETENU** (budget non disponible)
- ✅ **DEFER M3 avec stratégie provisoire approuvée 2026-05-26**
  - Référence : `docs/governance/legal_bootstrap_strategy_2026_05_26.md`
  - Templates V0 : `docs/governance/legal_templates/`
  - Plan migration : `docs/governance/legal_migration_plan_to_lawyer.md`
  - Déclencheur réactivation : MRR ≥ $1500 stable 60j + trésorerie ≥ 4 k€
  - Exposition résiduelle estimée : ~$1 200-$2 400/an pendant 3 mois max
  - Stack bootstrap retenu : auto-entrepreneur + Iubenda Pro $30/mo + RC Pro Freelance 300-500€/an + cap 50 abonnés + geo-restrict FR/BE/CH/LU
  - Coût annuel total stack provisoire : ~750-1 640 € (vs 3-5 k€ avocat one-shot)

Signé : **Loukmane Bessam** Date : **2026-05-26**

---

## Décision 3 — Lock pricing v1 : FREE / $29 / $79 / $1990

### Enjeu
Sans grille tarifaire fixée :
- DG-132 (page pricing) impossible à développer
- DG-043 (Stripe configuration) impossible à câbler
- DG-083 (decoy effect $1990) inopérant
- DG-084 (dual trial 14j+14j) inopérant
- **Tous les items monetization aval sont bloqués.**

La grille recommandée par eval_27 (FREE / STARTER $29 / PRO $79 / INSTITUTIONAL $1990) est **défensable**, avec :
- Decoy $1990 → +25-40% conversion PRO
- INSTITUTIONAL pricé à valeur réelle (vs $149 actuel sous-pricé ×13)
- Dual trial 14j+14j → +$1168 MRR vs freemium-only
- Annual 16.7% off → réduction churn

### Option recommandée
**GO grille eval_27** :
| Tier | Prix mensuel | Prix annuel (-16.7%) |
|---|---|---|
| FREE | $0 | — |
| STARTER | $29 | $290 |
| PRO | $79 | $790 |
| INSTITUTIONAL | $1990 | $19,900 (engagement 12 mois min) |

Dual trial 14j sans CB (FREE→STARTER) + 14j avec CB (STARTER→PRO). Refund 30j first-month no-questions.

### Si tu ne tranches pas
Prospects voient un pricing flou = pas de conversion. Concurrence (LuxAlgo $39.95, TradingView $14.95) prend la décision pour nous par défaut.

### Alternative défendable (non recommandée)
Grille BP actuel ($49/$99/$149). **Pourquoi non recommandée** :
- INSTITUTIONAL sous-pricé ×13 selon eval_27 (marge réelle 48%)
- Pas de decoy effect
- Pas de B2B viable
- Plafonne ARPU

### Signature

- ✅ **APPROUVÉ 2026-05-26** : grille FREE / $29 / $79 / $1990 + annual -16.7% + dual trial + refund 30j
- ☐ Modifier (préciser ci-dessous)
- ☐ Discuter avant signature

Signé : **Loukmane Bessam** Date : **2026-05-26**

---

## Décision 4 — Instruments GA : XAU + EUR seuls

### Enjeu
**Marketing actuel "6 instruments" est un mensonge** : 5/6 presets sans CSV propre (eval_08). Risque commercial direct + compliance fragile (claim non substanciable).

Tant que tu ne tranches pas, l'effort dev reste dispersé sur 6 actifs, qualité dégradée sur tous, marketing incohérent.

### Option recommandée
**GO XAU + EUR seuls** en GA S6.
- Drop marketing "6 instruments" partout (landing, Telegram, doc API)
- Drop dev BTC, US500, JPY, GBP (presets restent en code mais marketing les ignore)
- USOIL ajouté post-S16 conditionné à validation Polygon data
- Focus : XAU M15 (asset principal) + EURUSD M15 (extension wedge eval_25)

### Si tu ne tranches pas
- Effort dev dispersé sur 6 actifs → qualité moyenne sur tous
- Compliance fragile (claims non substanciables)
- Marketing menteur expose à fermeture Stripe (Trading Standards UE)

### Alternative défendable (non recommandée immédiatement)
Garder XAU + EUR + USOIL en GA. **Bloquée** par DG-076 MODIFY (Polygon DEFER tant que pas de proof commercial XAU). À reconsidérer post-Vague 1.

### Signature

- ✅ **APPROUVÉ 2026-05-26** : XAU + EUR seuls en GA S6, drop marketing "6 instruments"
- ☐ Garder XAU + EUR + USOIL (souscrire Polygon $129/mo)
- ☐ Discuter avant signature

Signé : **Loukmane Bessam** Date : **2026-05-26**

---

## Décision 5 — Reformulation totale "signaux" → "analyses" + USP "honest confidence"

### Enjeu

**Volet 1 — Compliance non-négociable** :
MiFID directive finfluencer entre en vigueur **mars 2026**. Le mot-clé "signal trading" est un déclencheur explicite. Sans reformulation totale (landing, Telegram, narratives, API docs, emails), risque d'amende et de fermeture Stripe. Reformulation déjà partiellement faite en sprint W3, à compléter sur landing + marketing.

**Volet 2 — Positioning différenciation** :
Sans USP "honest confidence" écrit explicitement, aucun différenciateur défendable face à la concurrence (LuxAlgo, signaux Telegram, TradingView). Le seul angle défendable face à "your A1 failed" = honnêteté assumée.

### Option recommandée
**GO reformulation totale + USP "honest confidence" en hero secondaire landing.**

Vocabulaire :
- "signaux" → "analyses" / "lectures algorithmiques"
- "BUY/SELL" → "setup haussier / baissier"
- "achetez/vendez" → "lecture haussière/baissière"
- Cibles : landing, Telegram, Discord, narratives LLM, API docs, emails

USP central :
> *« L'unique indicateur qui assume ce qu'il ne sait pas. »*
> *« Nous ne vous disons pas quoi faire — nous vous donnons les meilleurs outils pour décider. »*

### Si tu ne tranches pas
- Risque amende MiFID mars 2026
- Positioning indifférencié → commodité prix face à LuxAlgo $39.95
- Le moat "honest confidence" reste implicite donc invisible commercialement

### Alternative défendable (non recommandée)
Reformulation partielle (Telegram + narratives seuls, landing inchangée). **Pourquoi non recommandée** : la landing est la première impression, doit être 100% compliance et 100% différenciée. Reformulation partielle = trou commercial visible.

### Signature

- ✅ **APPROUVÉ 2026-05-26** : reformulation totale + USP "honest confidence" hero secondaire landing
- ☐ Modifier USP (préciser ci-dessous)
- ☐ Discuter avant signature

Signé : **Loukmane Bessam** Date : **2026-05-26**

---

## Synthèse pour décision rapide

| Décision | Bloquant aval | Effort signature | Coût |
|---|---|---|---|
| **1. Vision B** | Toute la roadmap | 5 min écriture | $0 |
| **2. RFQ avocat** | Stripe live (5-6 sem lead time) | 30 min envoi 3 RFQ | 3-5 k€ engagés |
| **3. Pricing lock** | Landing + Stripe + decoy + trial | 10 min validation | $0 |
| **4. Instruments XAU+EUR** | Marketing + dev focus | 5 min validation | $0 (économie Polygon $129/mo) |
| **5. Vocabulaire + USP** | Compliance + différenciation marque | 15 min validation | $0 |

**Total : ~1h de décisions, 0 dev, débloque ~280h de Vague 1.**

---

## Validation utilisateur — ACTÉ 2026-05-26

Date de signature : **2026-05-26**

Signature utilisateur : **Loukmane Bessam**

Notes / réserves :
- Décision 2 (avocat fintech) reformulée en DEFER M3 avec stratégie provisoire de bootstrap légal documentée dans `legal_bootstrap_strategy_2026_05_26.md`
- Budget avocat 3-5 k€ non disponible cette semaine, sera engagé à M3 avec déclencheur MRR ≥ $1500 stable 60j + trésorerie ≥ 4 k€
- Exposition résiduelle bootstrap acceptée et documentée (~$1 200-2 400/an pendant 3 mois max)
- Stack bootstrap : auto-entrepreneur FR + Iubenda Pro $30/mo + RC Pro Freelance 300-500€/an + 6 templates V0 customisés (`legal_templates/`) + cap 50 abonnés + geo-restrict FR/BE/CH/LU
- Toutes les autres décisions (1, 3, 4, 5) approuvées sans réserve

**Ce document est désormais le contrat d'engagement effectif pour le démarrage de la Vague 1.**

Démarrage Vague 1 conditionné à :
1. ✅ Vision B confirmée écrit dans `kill_criteria_board.md`
2. ✅ Stratégie bootstrap légal validée
3. ⏭ Création auto-entreprise (semaine S1)
4. ⏭ Souscription Iubenda Pro + customisation templates V0 (semaine S1-S2)
5. ⏭ Souscription RC Pro Freelance (semaine S1)
6. ⏭ Mise en œuvre des 10 P0-strict-MVP (S1-S6, ~140-160h)
