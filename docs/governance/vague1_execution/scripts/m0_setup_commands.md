# Setup M0 — Commandes utilisateur cette semaine

Toutes les actions à exécuter par TOI (Loukmane) cette semaine pour préparer le démarrage Vague 1.

Estimation totale : **~4-6h sur 1-2 jours** réparties sur démarches en ligne + attente passive (SIRET lead time 7-14j).

---

## 🔴 Action 1 — Création auto-entreprise (15 min + 7-14j passif)

**Site** : https://www.autoentrepreneur.urssaf.fr

**Procédure** :

1. Cliquer "Créer mon auto-entreprise"
2. Choisir activité : **"Prestation de services commerciaux ou artisanaux"**
3. Code APE : **62.01Z — Programmation informatique** (ou 63.11Z traitement de données si tu préfères)
4. Activité principale : *"Édition et exploitation de logiciels d'analyse algorithmique de marchés financiers à vocation pédagogique"*
5. Domiciliation : adresse personnelle OK (économie ~360 €/an vs domiciliation commerciale)
6. Régime fiscal : **micro-entrepreneur (micro-BIC)**
7. Régime TVA : **franchise en base** (par défaut, plafond 36 800 € HT/an services)
8. Régime social : **micro-social simplifié**
9. Activité ABRS (assurance retraite des indépendants) : automatique

**Documents requis** :
- Pièce d'identité (CNI ou passeport)
- Justificatif de domicile < 3 mois

**Délai** :
- Confirmation INSEE sous 7-14 jours
- Tu recevras **SIRET + n° SIREN** par mail

**Coût** : **0 €** (création gratuite)

**Action complémentaire après réception SIRET** :
- Activer ton **compte URSSAF** sur autoentrepreneur.urssaf.fr (déclaration mensuelle ou trimestrielle au choix — trimestrielle recommandée moins de friction)

---

## 🟠 Action 2 — Compte bancaire pro (15 min, immédiat)

Au-delà de 10 000 €/an de CA, un compte bancaire dédié à l'activité pro est obligatoire (loi PACTE 2019). Mieux vaut ouvrir tout de suite.

**Recommandation** : **Shine** (5-15 €/mois, design moderne, intégration directe avec URSSAF possible).

**Alternative** : **Qonto** (9-29 €/mois, plus pro mais cher).

**Procédure Shine** :
1. https://www.shine.fr
2. Inscription en ligne 15 min
3. Vérification identité KYC sous 24-48h
4. IBAN immédiat pour les premières opérations
5. Carte bancaire physique 5-7 jours après

**Coût** : ~120 €/an (10 €/mois Shine)

---

## 🟡 Action 3 — Achat domain `mia.markets` (10 min)

**Recommandation** : **Namecheap** ou **OVH** (français).

**Procédure Namecheap** :
1. https://www.namecheap.com
2. Rechercher `mia.markets`
3. Vérifier disponibilité — si pris, alternatives à considérer :
   - `mia-markets.ai`
   - `miamarkets.io`
   - `miamarkets.app`
4. Acheter (paiement CB)
5. Activer **WHOIS Privacy** (gratuit chez Namecheap, masque tes infos perso)
6. Configurer DNS — vide pour l'instant, sera renseigné lors du deploy Fly.io/Vercel

**Coût** : **~15 €/an** (renouvellement automatique optionnel à activer)

---

## 🟢 Action 4 — Souscription emails pro (15 min)

**Recommandation** : **ProtonMail Custom Domain** (mail.proton.me / business)

**Procédure** :
1. https://proton.me/business
2. Souscrire plan "Mail Essentials" (~$5/mo, 1 utilisateur, 3 adresses)
3. Configurer `mia.markets` comme custom domain
4. Créer les 4 adresses :
   - `contact@mia.markets` (principal)
   - `privacy@mia.markets` (RGPD, DG-038)
   - `security@mia.markets` (incidents)
   - `support@mia.markets` (utilisateurs)
5. Vérifier les enregistrements DNS (MX, SPF, DKIM, DMARC) → ProtonMail te donne les valeurs à coller chez Namecheap

**Alternative économique** : **Fastmail** (~$3-5/mo) ou **Zoho Mail** (gratuit jusqu'à 5 utilisateurs, mais ergonomie moindre)

**Coût** : **~5 $/mois** soit **~60 €/an**

---

## 🟡 Action 5 — Devis + souscription RC Pro Freelance (1h)

**Trois cabinets à contacter pour devis comparatifs** :

### Option A — Hiscox Freelance
- https://www.hiscox.fr/professionnels/independants
- Devis en ligne 10 min
- Coût indicatif : **300-400 €/an** (RC Pro basique freelance)
- Bon point : leader marché freelance, gère bien fintech

### Option B — Wemind
- https://www.wemind.io
- Plateforme dédiée freelance + indépendant
- Coût indicatif : **250-350 €/an**
- Bon point : bundle possible (RC Pro + protection juridique)

### Option C — Coover
- https://www.coover.fr/assurance/rc-pro-freelance
- Devis comparateur multi-assureurs
- Coût indicatif : **300-500 €/an**

**Description activité à fournir** (pour devis correct) :
> "Édition et exploitation d'un service logiciel en ligne d'analyse algorithmique de marchés financiers (gold et FX) à vocation pédagogique. Pas de conseil en investissement (non agréé AMF). Restriction géographique FR + BE + CH + LU. ~50 abonnés payants maximum en phase d'accès anticipé."

**Couvertures minimales à demander** :
- Responsabilité civile professionnelle (RC Pro)
- Erreur ou omission dans la prestation
- Atteinte aux droits des tiers
- Mini-couverture cyber-risque si possible (+150-300 €/an)
- Géographie : FR + BE + CH + LU
- Plafond : 500 000 € minimum

**Coût** : **~300-500 €/an**

---

## 🟢 Action 6 — Iubenda Pro (optionnel — peut être différé) (30 min)

Si tu souhaites un boost de qualité légale immédiat :

**Procédure** :
1. https://www.iubenda.com
2. Souscrire plan "Pro" ($30/mo, multi-langue inclus)
3. Créer projet "M.I.A. Markets"
4. Configurer :
   - Type : SaaS B2C
   - Pays : France
   - Géographie : FR, BE, CH, LU
   - Activités : abonnement payant, communications marketing, traitement données utilisateur
5. Générer CGU/CGV + Privacy Policy + Cookie Policy
6. Copier le markdown + appliquer les **clauses fintech-specific** depuis `docs/governance/legal_templates/cgu_cgv_v0_template.md`
7. Publier sur `/cgu` et `/privacy`

**Coût** : **$30/mo** soit **~360 €/an**

**Alternative gratuite** : utiliser les **templates V0 dans `docs/governance/legal_templates/`** directement sans Iubenda. Qualité moindre mais acceptable pour bootstrap 2-3 mois.

---

## 🔵 Action 7 — Comptes services produit (15 min)

À créer en attendant le démarrage technique S2 :

### Fly.io (hébergement)
- https://fly.io/signup
- Inscription gratuite (free tier : 3 apps × 256MB RAM)
- Pas de CB requise pour démarrer

### Anthropic (déjà fait sans doute)
- https://console.anthropic.com
- Vérifier crédits + facturation

### Vercel (frontend)
- https://vercel.com/signup
- Plan Hobby gratuit pour démarrer

### Plausible self-hosted
- Pas de compte cloud à créer, sera déployé sur Fly.io en S3

### Stripe
- https://dashboard.stripe.com/register
- Création en **mode test** uniquement S1-S5
- KYC pour activation live se fera S6

### Telegram Bot
- Si pas déjà fait : créer bot via @BotFather sur Telegram, récupérer `TELEGRAM_BOT_TOKEN`
- Créer channel public `M.I.A. Markets — Public Tape` (DG-072) — mais ne pas commencer à publier avant S5 (besoin track record cohérent)

**Coût** : **0 €** sauf Anthropic API (pay-as-you-go selon usage)

---

## 📋 Checklist M0 résumée

À faire cette semaine (S1) :

- [ ] **J1** : Créer auto-entreprise URSSAF (15 min, attente 7-14j SIRET)
- [ ] **J1** : Acheter domain `mia.markets` Namecheap (10 min)
- [ ] **J1** : Souscrire compte bancaire pro Shine (15 min, KYC 24-48h)
- [ ] **J2** : Souscrire ProtonMail Custom Domain (15 min)
- [ ] **J2** : Configurer DNS domain → ProtonMail MX/SPF/DKIM/DMARC
- [ ] **J2-J3** : Devis 3 cabinets RC Pro, sélection, souscription (~1h)
- [ ] **J3** : Créer comptes services (Fly.io, Vercel, Stripe test) (15 min)
- [ ] **J3** : Créer bot Telegram via @BotFather, sauvegarder token (10 min)
- [ ] **J5** : Vérification email pro fonctionnel
- [ ] **J7-J14** : Réception SIRET → activer URSSAF en ligne

**Total temps actif** : ~3-4h
**Coût annuel cumulé** : **~750-900 €** (sans Iubenda) ou **~1100-1300 €** (avec Iubenda)
**Coût mensuel** : **~65-110 €**

---

## ⏭ Après réception SIRET (J+7-14)

- [ ] Compléter mentions légales avec SIRET réel dans `legal_templates/mentions_legales_auto_entrepreneur.md`
- [ ] Mettre à jour ProtonMail signature : "Auto-entrepreneur · SIRET XXX XXX XXX XXXXX"
- [ ] Configurer Shine avec n° SIRET
- [ ] Informer l'autre instance Claude Code que l'identité juridique est en place → démarrage Sprint 1 dev possible

---

## 🚨 Pièges à éviter

- ❌ **Mettre adresse perso comme RGS** sans WHOIS privacy : ton adresse devient publique sur whois.com
- ❌ **Activer Stripe live AVANT KYC complet** : Stripe peut suspendre. Soumettre review fintech S6 obligatoire.
- ❌ **Oublier DMARC** sur emails : tes mails partent en SPAM sur Gmail/Outlook
- ❌ **Souscrire RC Pro sans préciser activité fintech** : risque exclusion sinistre. Description précise dans le devis obligatoire.
- ❌ **Démarrer Telegram public track record AVANT signal stable** : DG-072 dit forward only 60-90j MINIMUM avant communication marketing.
- ❌ **Acheter domain `.io` au lieu de `.com`** : crédibilité B2C/B2B inférieure. Si `.com` indisponible, préférer `.ai` plutôt que `.io`.

---

## Récapitulatif coût annuel bootstrap

| Ligne | Coût annuel |
|---|---|
| Auto-entreprise | 0 € |
| Domain Namecheap | 15 € |
| Compte bancaire pro Shine | 120 € |
| ProtonMail Custom Domain | 60 € |
| RC Pro Freelance Hiscox | 300-500 € |
| Iubenda Pro (optionnel) | 360 € |
| Fly.io free tier (au démarrage) | 0 € (puis ~60-180 €/an si scaling) |
| Vercel Hobby | 0 € |
| Anthropic API | variable (~120-600 €/an selon usage) |
| Stripe | frais transaction (~3 % du revenue) |
| **Total minimum (sans Iubenda)** | **~615-1 200 €/an** |
| **Total recommandé (avec Iubenda)** | **~975-1 560 €/an** |
