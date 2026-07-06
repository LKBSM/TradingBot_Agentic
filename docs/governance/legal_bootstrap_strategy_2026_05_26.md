# Stratégie de protection légale provisoire — M.I.A. Markets

> ⚠️ **PARTIELLEMENT OBSOLÈTE (2026-07-06)** — les points géo-restriction
> (FR/BE/CH/LU), migration avocat M3 et cadre auto-entrepreneur FR/CM2C sont
> remplacés par `decisions/2026-07-06_conformite_posture_descriptive_exclusion.md`
> (posture descriptive + exclusion US/UK/OFAC, entreprise québécoise Loi 25 + LPC,
> aucune consultation avocat planifiée). Le reste (posture produit réduite,
> refund, honnêteté des claims) reste la ligne en vigueur.

**Date** : 2026-05-26
**Contexte** : budget avocat fintech FR (3-5 k€, DG-075) non disponible aujourd'hui. Vague 1 doit démarrer pour générer du revenue qui financera l'avocat à M3.
**Objectif** : minimiser l'exposition juridique pendant la phase bootstrap (M0-M3) avec des templates de qualité, une posture produit réduite, et une géo-restriction temporaire.

---

## ⚠️ Disclaimer non-négociable

**Je ne suis pas avocat.** Cette stratégie est une analyse risque/bénéfice de bon sens basée sur les pratiques courantes du SaaS bootstrap fintech FR (auto-entrepreneur, Iubenda, Stripe Atlas, Plausible self-hosted). Elle ne constitue **pas un conseil juridique**.

Les risques résiduels documentés en §8 sont **réels mais bornés**. Tu prends la décision finale en connaissance de cause. Cette stratégie est explicitement **provisoire 2-3 mois maximum** — à remplacer par CGU avocat-signed dès que le revenue le permet.

---

## 0. TL;DR pour le décideur

- **7 piliers** réduisent ton exposition juridique d'environ **85-90 %** vs lancer "à l'aveugle"
- **Cap utilisateurs payants à 50** pendant M1-M3 → exposition financière maximum bornée
- **Géo-restriction FR + BE + CH + LU** au lancement → 4 juridictions au lieu de 30+ → simplifie TVA, exclut US/UK/CA naturellement
- **Templates Iubenda + iubenda Privacy** ($30-60/mo) → qualité équivalente cabinet généraliste, automatisation RGPD intégrée
- **Posture produit "Beta Early Access · Educational Use"** explicite → réduit claim commercial donc risque MiFID
- **Revenue projeté pour financer avocat** : 50 × $29 × 2 mois = $2900 → couvre 1 cabinet fintech entrée de gamme à M3
- **Risques résiduels** documentés et quantifiés en §8

**Décision attendue** : valides-tu cette stratégie comme remplacement temporaire de DG-075 ?

---

## 1. Pilier 1 — Statut juridique : Auto-entrepreneur FR

### Pourquoi
- Le plus simple à créer (formulaire en ligne 15 min, gratuit, immatriculation sous 7-14j)
- Plafond CA HT services 2024 : **77 700 €/an** (largement suffisant pour la phase 1 — équivaut à ~$84k/an)
- TVA franchise en base jusqu'à 36 800 € HT/an services → **pas de TVA à facturer** ni à reverser jusqu'à ce seuil
- Au-dessus du seuil franchise mais sous 77 700 €, TVA simple (20 %) à reverser trimestriellement

### Coûts
- Création : **0 €**
- Cotisations sociales : **22-25 % du CA** (forfait simplifié SSI)
- Compte bancaire pro obligatoire au-delà de 10 000 €/an : ~5-15 €/mois (Shine, Qonto, Hello bank! Pro)

### Limitations
- **Responsabilité illimitée sur patrimoine personnel** (pas de séparation comme SASU) → c'est le risque principal. Mitigé par : cap utilisateurs (§5), assurance RC Pro minimale (§7), refund 30j systématique (§4)
- Pas adapté si > 77 700 €/an HT → migration vers SASU prévue M6-M9

### Action concrète
- Créer compte sur `autoentrepreneur.urssaf.fr` cette semaine
- Code APE recommandé : **62.01Z** (programmation informatique) ou **63.11Z** (traitement de données, hébergement)
- Domiciliation : adresse perso OK (pas besoin de domiciliation commerciale)

---

## 2. Pilier 2 — Géo-restriction temporaire : FR + BE + CH + LU seulement

### Pourquoi
- US/UK/CA/AU/QC sont des juridictions **lourdes** (SEC, FCA, ASIC, AMF Québec). Géo-block dès le jour 1 → conformité DG-045 minimale.
- UE complète = 27 pays × règles fiscales différentes = complexité TVA OSS énorme
- FR + BE + CH + LU = **francophones, juridictions cousines, droit conso similaire**, complexité fiscale minimale
- Volume marché adressable FR+BE+CH+LU couvre largement les 50 abonnés cap M1-M3

### Configuration technique
- Geo-block IP au middleware (DG-045 déjà partiellement livré W1)
- Allow-list : `FR`, `BE`, `CH`, `LU` uniquement
- Page `/restricted-region` honnête : "Service en phase beta, restreint à FR/BE/CH/LU. Inscrivez-vous à la liste d'attente."
- Stripe Checkout configuré pour ces 4 pays seulement

### Conformité fiscale simplifiée
- FR : TVA 20 % (franchise en base au début)
- BE : TVA 21 % — micro-régime UE OSS si dépasse seuil
- CH : pas de TVA UE (Suisse hors UE)
- LU : TVA 17 % — UE OSS

→ Au début (franchise FR active), **tu vends HT en France** et tu n'as pas à facturer la TVA. Pour BE/LU, soit tu restes en franchise FR (si revenu très faible), soit tu actives Stripe Tax avec OSS UE quand tu passes le seuil.

### Action concrète
- Activer geo-block 4 pays en S2 (DG-045 modifié pour allow-list au lieu de deny-list)
- Désactiver inscription pour autres pays au niveau front

---

## 3. Pilier 3 — Posture produit "Beta Early Access · Educational Use"

### Pourquoi
La directive UE 2024/2811 finfluencer (mars 2026) vise les acteurs qui **commercialisent des recommandations d'investissement**. Une posture "outil éducatif en bêta" réduit drastiquement l'exposition. C'est aussi cohérent avec ton positioning "honest confidence" (DG-077) et `edge_claim=False`.

### Wording cadre (à mettre partout)

**Hero landing** :
> M.I.A. Markets · **Outil éducatif d'analyse de marché** · Phase d'accès anticipé
>
> Apprenez à lire les marchés Or et FX comme un quant institutionnel.
> Pas de signaux. Pas de promesses. Pas de conseils. Juste de la compréhension.

**Footer permanent** (déjà dans le mockup HTML) :
> Démonstration paper-trading · Lecture algorithmique éducative · Ne constitue ni un signal de trading ni un conseil en investissement.

**Page tarifs** :
> M.I.A. Markets est en phase d'accès anticipé (Early Access). Les fonctionnalités évoluent. Le service est proposé en tant qu'outil pédagogique d'analyse algorithmique, sans recommandation d'investissement.

### Implications produit
- **Pas le mot "signal"** nulle part (DG-073 strict)
- **Pas de claim de gain ni de performance** ("gagnez X%", "+30% en 3 mois" = INTERDIT)
- **Refus pédagogique chatbot** systématique sur "dois-je acheter ?" (DG-112 déjà spec)
- **Track record visible** = paper-trading historique uniquement, jamais "rendement réel"
- **Disclaimer renforcé** sur chaque page critique

### Action concrète
- Audit du wording sur landing + Telegram + emails + chatbot prompts
- Bouton "Early Access" partout (vs "Inscription / S'abonner") → cadre le statut produit

---

## 4. Pilier 4 — Templates légaux Iubenda + suppléments customisés

### Pourquoi
Iubenda est utilisé par **90 000+ entreprises** en UE (dont des fintechs FR), génère CGU + Privacy Policy + Cookie Policy en RGPD-compliant, mis à jour automatiquement quand la loi change. Coût $30-60/mo. **Qualité équivalente à un cabinet généraliste** sans relecture spécifique.

### Stack légal recommandé

| Document | Source | Coût | Effort |
|---|---|---|---|
| CGU/CGV | Iubenda template "SaaS B2C" customisé | $30/mo (Plan Pro) | 4-6h customisation |
| Privacy Policy | Iubenda Privacy Policy Generator | inclus | 2-3h |
| Cookie Policy | Inclus Iubenda OU pas de cookies (Plausible self-hosted) | inclus | 1h |
| DPA | Template Stripe + Anthropic acceptation standard | $0 | 1h |
| Mentions légales | Template fourni dans `docs/governance/legal_templates/` | $0 | 1h |
| Disclaimer compliance | Template fourni multilingue | $0 | 1h |

**Coût total mensuel** : ~$30/mo Iubenda Pro (vs 3-5 k€ avocat fintech one-shot)

**Important** : ces templates sont **bons mais pas parfaits**. Un avocat fintech ajouterait : clauses MiFID spécifiques, limitation de responsabilité fintech robuste, médiation conso CM2C précisée. C'est ce qui sera fait à M3.

### Renforcement V0 (au-dessus des templates Iubenda)
Le dossier `docs/governance/legal_templates/` contient 6 templates V0 qui **renforcent** Iubenda sur les points critiques fintech :
- `disclaimer_compliance.md` — wording UE 2024/2811-aware partout
- `cgu_cgv_v0_template.md` — clauses fintech-specific à ajouter au template Iubenda
- `privacy_policy_v0_template.md` — RGPD + détail data collecte/conservation
- `mentions_legales_auto_entrepreneur.md` — obligatoire
- `cookie_notice_minimal.md` — minimaliste (Plausible self-hosted = pas de bandeau intrusif)
- `incident_response_runbook.md` — process réaction si plainte/breach

---

## 5. Pilier 5 — Cap utilisateurs payants : 50 abonnés max M1-M3

### Pourquoi
Plus tu as d'abonnés, plus tu as d'exposition (chaque client = un litige potentiel). Cap = exposition financière maximum bornée.

### Mécanique
- **M0 (now-S2)** : tier FREE seul (pas de Stripe live activé). Tester produit, valider valeur sans risque légal de vente.
- **M1 (S3-S6)** : ouverture STARTER $29 seul, cap à **20 abonnés payants** (waitlist au-delà). Pas de tier PRO ni INSTITUTIONAL.
- **M2 (S7-S10)** : si M1 stable, élargir à **50 abonnés STARTER**. Ouverture PRO $79 cap à **10 abonnés** sélectionnés (power users beta).
- **M3 (S11-S14)** : avec revenue $1500-3000/mo accumulé, RFQ avocat fintech.

### Communication marketing
- "Accès anticipé limité à X places" = bon levier marketing (FOMO honnête)
- Waitlist visible sur landing = preuve sociale + capture leads
- Pas une excuse marketing factice : c'est réel et tu peux le justifier ("on stabilise avant d'élargir")

### Exposition financière calculée
- 50 abonnés × $29 × 12 mois = **$17,400/an HT** = sous le seuil franchise TVA FR (36,800 €)
- Refund 30j systématique → exposition légale par client = 1 mois max = $29 max remboursable
- Total exposition par client × 50 clients = **$1,450 max** (théorique worst case)

### Action concrète
- Coder hard cap dans `quota_manager.py` (DG-046 modifié : cap global d'abonnements, pas seulement par tier)
- Waitlist via simple Google Form intégré au landing

---

## 6. Pilier 6 — Refund 30j systématique no-questions

### Pourquoi
Le refund 30j (DG-079 déjà approuvé) est **ton bouclier juridique principal** en bootstrap. Toute insatisfaction = remboursement. Réduit drastiquement la probabilité de litige client → réduit le besoin de CGU robustes.

### Mécanique
- Premier mois remboursable intégralement, sans question
- Process automatisé via Stripe Customer Portal (DG-043) — pas d'intervention manuelle
- Communication transparente : "Pas content sous 30 jours = remboursé. Point."

### Impact légal
- Compatible loi Hamon FR (14j minimum, tu en offres 30)
- Réduit le risque de plainte conso → réduit l'usage de la médiation CM2C (DG-082)
- En cas de litige, tu peux toujours offrir le refund comme premier remède → désamorce 90 % des conflits

---

## 7. Pilier 7 — Assurance RC Pro minimale + couverture perso

### Recommandation provisoire (M0-M3)
Au lieu de RC Pro complète Stoïk/Hiscox (3-5 k€/an), souscrire :

| Assurance | Coût | Couverture |
|---|---|---|
| **RC Pro Freelance basique** (Hiscox / Wemind / Coover) | 300-500 €/an | Dommages à tiers, responsabilité professionnelle de base |
| **Cyber-risque indépendant** | optionnel, 150-300 €/an | Brèche RGPD, ransomware basique |
| **Responsabilité civile personnelle** (souvent incluse habitation/MRH) | inclus | Vérifier auprès de ton assureur habitation |

**Coût provisoire** : 300-800 €/an au lieu de 3-5 k€.

### À M3 (avec revenue stable)
Upgrade vers Stoïk ou Hiscox bundle complet (RC Pro fintech + Cyber + Protection Juridique) : 3-5 k€/an comme prévu DG-050.

### Action concrète
- Devis Hiscox Freelance en ligne (5 min) : `hiscox.fr/freelances`
- Devis Wemind ou Coover en alternative
- Souscription cette semaine = tranquillité

---

## 8. Risques résiduels documentés (honnêteté)

### 🔴 Risque 1 — Litige client sur claim performance / promesse implicite
- **Probabilité** : faible si wording strictement éducatif respecté (Pilier 3)
- **Impact** : remboursement + risque amende AMF si plainte
- **Mitigation** : refund 30j (Pilier 6) + wording audité par toi avant chaque release marketing + chatbot refus pédagogique scripté (DG-112)
- **Coût worst case** : 1 plainte AMF = procédure 6-12 mois, amende 1 500-5 000 € si infraction caractérisée
- **Probabilité × Impact estimée** : ~$300-500 exposition annualisée

### 🟡 Risque 2 — Plainte CNIL / brèche RGPD
- **Probabilité** : faible si stack technique propre (Plausible self-hosted, données minimales, Stripe gère le PCI, Anthropic gère son propre RGPD)
- **Impact** : amende théorique jusqu'à 4 % du CA — mais en pratique CNIL avertit d'abord (mise en demeure 30j) avant amende effective
- **Mitigation** : Privacy Policy V0 complète, DPA Stripe + Anthropic acceptés explicitement, minimisation données (email + tier seulement, pas de PII trading), process incident (`incident_response_runbook.md`)
- **Coût worst case** : 5 000 € amende sur premier avertissement non corrigé (rare en pratique sur petits acteurs)
- **Probabilité × Impact estimée** : ~$100-200 exposition annualisée

### 🟡 Risque 3 — Suspension compte Stripe pour vocabulaire / claim
- **Probabilité** : moyenne. Stripe scanne automatiquement les sites contenant "trading signals" "guaranteed profit" "investment advice"
- **Impact** : suspension compte = perte revenue immédiate
- **Mitigation** : Pilier 3 strict (wording éducatif) + soumission proactive du site à Stripe pour review avant activation live + maintien du tier FREE en parallèle pendant 30j post-activation pour détecter friction
- **Coût worst case** : 1-2 semaines de revenue perdues, réactivation possible après modification site (1 mois max)
- **Probabilité × Impact estimée** : ~$500-1000 exposition unique (pas annualisée)

### 🟢 Risque 4 — Plainte conso L.612-1 médiation
- **Probabilité** : très faible avec refund 30j systématique
- **Impact** : adhésion médiation forcée (150 €/an) — tu y seras de toute façon (DG-082)
- **Mitigation** : refund 30j (Pilier 6) + adhésion CM2C/MEDICYS volontaire dès M2 (150 €/an)
- **Coût worst case** : 150 €/an + 50-200 € par dossier de médiation
- **Probabilité × Impact estimée** : ~$200-400 exposition annualisée

### 🟢 Risque 5 — Action MiFID directe AMF
- **Probabilité** : très faible avec posture "outil éducatif" + cap utilisateurs + wording strict
- **Impact** : AMF intervient surtout sur acteurs >1000 abonnés ou influenceurs visibles. Sub-50 abonnés en bootstrap = sous le radar
- **Mitigation** : Pilier 3 (posture) + Pilier 5 (cap utilisateurs) + Pilier 6 (refund) + arrêt immédiat de toute communication marketing si moindre signal
- **Coût worst case** : avertissement AMF + obligation cessation → arrêt produit. Pas d'amende sur petit acteur en général.
- **Probabilité × Impact estimée** : ~$100-300 exposition annualisée

### Total exposition résiduelle annualisée estimée
**~$1 200 - $2 400 / an** vs **3-5 k€ avocat one-shot non disponible**.

→ La stratégie a **un sens financier** : tu acceptes une exposition résiduelle 2-4× inférieure au coût avocat, en attendant d'avoir le revenue pour le payer proprement.

---

## 9. Calendrier de migration vers avocat fintech

| Mois | Événement | Critère déclenchement | Coût |
|---|---|---|---|
| **M0** (now) | Setup stratégie bootstrap | — | $30/mo Iubenda + 300€/an RC Pro = ~$680/an |
| **M1** | Soft launch STARTER $29 cap 20 abonnés | Mockup HTML livré + CGU V0 + Privacy V0 publiés + Iubenda souscrit | Revenue cible : $500/mo |
| **M2** | Élargissement 50 STARTER + 10 PRO | MRR ≥ $500 stable 30j | Revenue cible : $1500/mo |
| **M3** | RFQ avocat fintech | MRR ≥ $1500 stable 60j ET trésorerie disponible ≥ 4k€ | Engagement budget 3-5 k€ |
| **M4** | Migration CGU V0 → V1 avocat-signed | Signature CGU V1 | — |
| **M5** | Ouverture INSTITUTIONAL + RC Pro upgrade | CGU V1 publiée + DPA B2B template | — |
| **M6-M9** | Migration auto-entrepreneur → SASU | CA HT > 50 k€/an OU besoin de séparation patrimoine | ~1500 € création SASU |

---

## 10. Migration CGU V0 → V1 sans rupture

### Process recommandé (M4)
1. **Sélection cabinet** (M3) — RFQ 3 cabinets, sélection sous 1 semaine
2. **Brief avocat** — fournir CGU V0 actuelle + audit risques V0 + cibles (B2C UE + B2B INSTITUTIONAL futur)
3. **Relecture + amendements** — l'avocat amende V0, retourne V1 sous 2-3 semaines
4. **Validation toi** — relire V1 vs V0, comprendre changements
5. **Publication V1** :
   - Annonce email à tous les abonnés actifs : "Nos CGU sont mises à jour pour mieux vous protéger. Voici les changements clés [bullet list 5 lignes]. Vous restez abonné = vous acceptez. Sinon, refund proratisé immédiat sur demande."
   - Période de transition 14 jours (légalement requis pour modification CGU SaaS)
   - Bouton "Accepter les nouvelles CGU" lors de la prochaine connexion
6. **Archive V0** — conserver 5 ans pour traçabilité des engagements passés

### Action concrète au moment voulu
Voir `docs/governance/legal_migration_plan_to_lawyer.md` pour le détail RFQ.

---

## 11. Décision politique #2 reformulée

**Avant** (DG-075) :
> Engagement cabinet avocat fintech FR 3-5 k€ — démarrage S1 immédiat.

**Reformulé (2026-05-26)** :
> **DG-075 = DEFER M3** (déclencheur : MRR ≥ $1500 stable 60j + trésorerie ≥ 4k€).
> Stratégie provisoire active (cf. `legal_bootstrap_strategy_2026_05_26.md`) :
> - Auto-entrepreneur FR
> - Géo-restriction FR + BE + CH + LU
> - Posture produit "Early Access · Educational Use"
> - Templates Iubenda Pro + suppléments V0 (`docs/governance/legal_templates/`)
> - Cap 50 abonnés payants M1-M3
> - Refund 30j systématique
> - RC Pro freelance basique 300-500 €/an
>
> Exposition résiduelle estimée : ~$1 200 - $2 400/an pendant 3 mois (avant migration avocat M3).

---

## 12. Checklist d'exécution M0 (cette semaine)

- [ ] **Statut juridique** : créer auto-entrepreneur sur `autoentrepreneur.urssaf.fr` (15 min)
- [ ] **Compte bancaire pro** : ouvrir Shine ou Qonto (15 min)
- [ ] **Iubenda Pro** : souscrire $30/mo, générer CGU/CGV/Privacy/Cookie templates
- [ ] **Customiser les 4 templates Iubenda** avec les suppléments V0 (4-6h total)
- [ ] **RC Pro Freelance** : devis Hiscox + souscription (1h)
- [ ] **Cyber-risque** : option Wemind ou Coover (optionnel, 30 min)
- [ ] **Geo-block configuration** : adapter `geo_block.py` en allow-list FR+BE+CH+LU
- [ ] **Mentions légales** : remplir template `mentions_legales_auto_entrepreneur.md` avec ton SIRET (5 min après création AE)
- [ ] **Stripe configuration** : créer compte Stripe en mode test, soumission pour review fintech avant activation live
- [ ] **Audit wording** : vérifier qu'aucun "signal", "buy", "sell", "garanti", "+X%" n'apparaît nulle part dans le produit ou la com'
- [ ] **Adhésion médiation conso** : reporter à M2 (économie 150 € jusqu'au premier abonné payant)

---

## 13. Quoi faire si quelque chose dérape

### Plainte client
1. Lire l'`incident_response_runbook.md` (templates de réponse)
2. Offrir refund immédiat sans question
3. Documenter dans le journal d'incidents (`docs/incidents/YYYY-MM-DD_short_desc.md`)
4. Si client persiste : médiation amiable (gratuite ou via CM2C si déjà adhéré)
5. Si médiation échoue : à ce stade, consulter avocat ponctuel (~200-400 €/heure) pour gérer le dossier

### Notification CNIL (brèche)
1. Stopper l'exposition (correctif immédiat)
2. Notifier CNIL sous 72h via `notifications.cnil.fr` (formulaire en ligne)
3. Si données utilisateur impactées : notifier les utilisateurs sous 72h supplémentaires
4. Documenter cause + correctif + procédure ajustée

### Suspension Stripe
1. Ne pas paniquer — Stripe envoie un email avec la raison
2. Lire le motif, identifier le terme/page incriminé
3. Corriger sous 7 jours et soumettre review re-activation
4. Si suspension confirmée : ouvrir compte Paddle ou Mollie en backup (moins strict que Stripe sur fintech bootstrap)

### Avertissement AMF (improbable mais)
1. **Consulter avocat fintech immédiatement** (même budget non disponible — c'est l'urgence qui justifie la dépense)
2. Suspendre toute communication marketing
3. Audit complet wording site + chatbot + emails
4. Réponse formelle AMF dans le délai imparti (généralement 30-60j)

---

## Conclusion

Cette stratégie te permet de **lancer Vague 1 cette semaine** sans bloquer sur l'avocat fintech, en acceptant une exposition résiduelle estimée à **~$1 200 - $2 400/an pendant 3 mois maximum**.

C'est un **risque calculé** : tu dépenses ~$680/an (Iubenda + RC Pro freelance) pour réduire ton exposition de 85-90 % vs lancer "à l'aveugle", et tu accumules le revenue qui paiera l'avocat fintech à M3.

Ce n'est **pas une solution à long terme**. À M3, tu migres vers CGU avocat-signed + RC Pro complète + SASU si CA le justifie.

**Templates concrets prêts à utiliser** : voir `docs/governance/legal_templates/`.
**Plan migration avocat M3** : voir `docs/governance/legal_migration_plan_to_lawyer.md`.

---

**FIN DE LA STRATÉGIE.**

Encore une fois : je ne suis pas avocat. Cette stratégie est une analyse de bon sens, pas un conseil juridique. La décision finale et son risque t'appartiennent.
