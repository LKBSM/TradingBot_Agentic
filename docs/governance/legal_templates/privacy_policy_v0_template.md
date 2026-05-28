# Politique de Confidentialité V0 — Template bootstrap M.I.A. Markets

**Date** : 2026-05-26
**Statut** : V0 bootstrap — à migrer V1 avocat-signed à M3
**Usage** : combiner avec template Iubenda Privacy Policy auto-généré. Les clauses ci-dessous **renforcent** Iubenda sur les points spécifiques fintech / RGPD.

---

## ⚠️ Mode d'emploi

1. Iubenda Pro génère une Privacy Policy de base
2. Adapter les sections "Données collectées", "Base légale", "Sous-traitants", "Durée de conservation" avec le contenu ci-dessous
3. Publier sur `/privacy`
4. Lien obligatoire footer + modale signup

---

## Politique de Confidentialité — M.I.A. Markets

**Dernière mise à jour** : 2026-05-XX
**Version** : V0 (bootstrap)
**Responsable de traitement** : [Nom], micro-entrepreneur, SIRET [XXX XXX XXX XXXXX], [adresse]
**Contact** : `privacy@mia.markets` (à créer)

---

### 1. Présentation

La présente Politique décrit comment M.I.A. Markets (ci-après "le Service") collecte, utilise, conserve et protège les données personnelles des Utilisateurs, conformément au Règlement (UE) 2016/679 (RGPD) et à la loi française "Informatique et Libertés" du 6 janvier 1978 modifiée.

### 2. Données collectées

Le Service applique le principe de **minimisation** : seules les données strictement nécessaires au fonctionnement sont collectées.

#### 2.1 Données fournies par l'Utilisateur

| Donnée | Finalité | Caractère obligatoire |
|---|---|---|
| Adresse e-mail | Création de compte, authentification, communication transactionnelle | Obligatoire |
| Identifiant tier | Gestion de l'abonnement | Obligatoire si tier payant |
| Langue préférée (FR/EN/DE/ES) | Personnalisation interface | Optionnel (défaut FR) |
| Watchlist d'actifs | Personnalisation lectures (XAU / EUR uniquement en bootstrap) | Optionnel |

#### 2.2 Données générées automatiquement

| Donnée | Finalité | Conservation |
|---|---|---|
| Journaux d'usage anonymisés (events Plausible) | Analytique produit, amélioration UX | 12 mois |
| Identifiant Stripe customer | Gestion abonnement | Durée du contrat + 5 ans (obligation comptable) |
| Historique des lectures consultées | Affichage historique 50 dernières lectures | Durée du compte + 30 jours après suppression |
| Conversations chatbot | Amélioration prompts, traçabilité refus pédagogique | 30 jours puis anonymisation, archivage agrégé 5 ans |

#### 2.3 Données NON collectées

Le Service **ne collecte pas** :
- Nom, prénom (sauf si Utilisateur les ajoute volontairement à son profil)
- Données patrimoniales, situation financière, portefeuille de trading
- Numéro de carte bancaire (géré exclusivement par Stripe, le Service n'a accès qu'à un token)
- Données de localisation précise (uniquement le pays pour le geo-block)
- Données biométriques, données sensibles au sens de l'article 9 RGPD

### 3. Base légale du traitement

| Traitement | Base légale RGPD (article 6) |
|---|---|
| Création de compte + authentification | Exécution du contrat (art. 6.1.b) |
| Gestion abonnement et paiement | Exécution du contrat (art. 6.1.b) |
| Analytique produit | Intérêt légitime (art. 6.1.f) — finalité : amélioration du Service |
| Communication transactionnelle (notifications service) | Exécution du contrat (art. 6.1.b) |
| Communication marketing (digest hebdo) | Consentement (art. 6.1.a) — opt-in explicite, opt-out facile |
| Obligations comptables et fiscales | Obligation légale (art. 6.1.c) |
| Sécurité du Service, prévention de la fraude | Intérêt légitime (art. 6.1.f) |

### 4. Durée de conservation

| Donnée | Durée | Justification |
|---|---|---|
| Données de compte actif | Durée du contrat + 30j (purge) | Continuité de service |
| Données comptables (factures, paiements) | 10 ans | Obligation Code de commerce |
| Logs d'authentification | 12 mois | Sécurité, RGPD art. 32 |
| Journaux applicatifs | 30 jours | Debug, sécurité |
| Conversations chatbot | 30 jours nominatives puis anonymisation, 5 ans agrégé | Amélioration produit + traçabilité compliance |
| Cookies analytiques (Plausible) | Pas de cookies (Plausible self-hosted, fingerprintless) | — |

### 5. Sous-traitants (data processors)

Le Service utilise les sous-traitants suivants. Chacun est lié par un **Data Processing Agreement (DPA)** conforme à l'article 28 RGPD.

| Sous-traitant | Service rendu | Localisation données | DPA |
|---|---|---|---|
| **Fly.io Inc.** | Hébergement application | Paris (région cdg) avec failover EU | Acceptation lors souscription |
| **Stripe Inc.** | Traitement paiement | UE + USA (sous Data Privacy Framework) | `stripe.com/legal/dpa` |
| **Anthropic PBC** | Génération narratifs et chatbot LLM | USA (Data Privacy Framework certifié) | `anthropic.com/legal/dpa` |
| **Cloudflare R2** | Stockage modèles ML et backups | EU | Acceptation lors souscription |
| **Plausible Analytics** (self-hosted) | Analytique produit | Hébergé Fly.io UE | Pas de sous-traitant tiers (self-hosted) |
| **Provider email** (à choisir : ProtonMail / Fastmail / Postmark) | Communication transactionnelle | EU | DPA à signer |
| **Trading Economics** | Données calendrier économique (pas de PII utilisateur) | N/A | N/A — pas de données utilisateur transmises |

**Transferts hors UE** : les transferts vers les USA (Stripe, Anthropic) sont fondés sur le **Data Privacy Framework UE-USA** (décision d'adéquation 2023/1795) et complétés par des Clauses Contractuelles Types (CCT) en cas d'invalidation.

### 6. Droits des Utilisateurs

Conformément aux articles 15 à 22 du RGPD, chaque Utilisateur dispose des droits suivants :

| Droit | Procédure | SLA |
|---|---|---|
| **Accès** (art. 15) | Bouton "Exporter mes données" dans paramètres compte OU email à `privacy@mia.markets` | 30 jours |
| **Rectification** (art. 16) | Directement dans paramètres compte OU email | 30 jours |
| **Suppression / "droit à l'oubli"** (art. 17) | Bouton "Supprimer mon compte" dans paramètres OU email | 30 jours (anonymisation pour conservation comptable 10 ans) |
| **Limitation** (art. 18) | Email à `privacy@mia.markets` | 30 jours |
| **Portabilité** (art. 20) | Export JSON / CSV via bouton "Exporter mes données" | 30 jours |
| **Opposition** (art. 21) | Email à `privacy@mia.markets` | 30 jours |
| **Retrait du consentement** (art. 7) | Désinscription emails depuis lien intégré dans chaque email | Immédiat |

Pour exercer ces droits, contacter `privacy@mia.markets` avec une pièce d'identité justifiant de l'identité du demandeur.

**Réclamation auprès de l'autorité de contrôle** : tout Utilisateur peut déposer une réclamation auprès de la CNIL (www.cnil.fr) ou de l'autorité de protection des données de son pays de résidence (APD Belgique, PFPDT Suisse, CNPD Luxembourg).

### 7. Sécurité

L'Éditeur met en œuvre les mesures techniques et organisationnelles suivantes pour protéger les données :

- **Chiffrement en transit** : TLS 1.3 obligatoire sur toutes les communications (webapp, API, Telegram)
- **Chiffrement au repos** : base de données SQLite chiffrée (SQLCipher) + backups chiffrés Cloudflare R2
- **Authentification** : tokens API rotables, mots de passe hachés bcrypt
- **Accès interne minimisé** : un seul administrateur (l'Éditeur) avec accès production, MFA activée
- **Logs d'accès** : conservation 12 mois pour audit
- **Patching** : mises à jour de sécurité appliquées sous 7 jours
- **Sauvegardes** : quotidiennes, chiffrées, restaurabilité testée mensuellement

### 8. Cookies et traceurs

Le Service utilise un **stack analytique sans cookies tiers** :

- **Plausible Analytics self-hosted** : pas de cookies, pas de fingerprinting, conforme CNIL recommandation 2024 sur les exemptions
- **Cookie de session** (technique, obligatoire) : pour maintenir l'authentification utilisateur — exempté de consentement (CNIL art. 82 alinéa 2)
- **Pas de Google Analytics, pas de Meta Pixel, pas de tracker publicitaire**

En conséquence, **aucun bandeau cookies intrusif n'est nécessaire**. Une information claire dans la présente Politique suffit.

### 9. Modifications de la Politique

L'Éditeur peut être amené à modifier la présente Politique. Toute modification substantielle sera notifiée aux Utilisateurs par email **au moins 30 jours avant** son entrée en vigueur. L'Utilisateur peut résilier son compte sans frais s'il refuse les modifications.

L'historique des versions est conservé et accessible sur demande.

### 10. Contact

| Question | Contact |
|---|---|
| Exercice des droits RGPD | `privacy@mia.markets` |
| Question générale sur la confidentialité | `privacy@mia.markets` |
| Signaler une brèche de sécurité | `security@mia.markets` |
| Réclamation autorité de contrôle | CNIL : www.cnil.fr |

---

## Annexe — Registre des activités de traitement (RGPD art. 30)

L'Éditeur étant micro-entrepreneur avec un effectif inférieur à 250 personnes, le tenue d'un registre formel des activités de traitement n'est pas obligatoire au sens strict (art. 30.5), mais elle est **recommandée** et constitue une bonne pratique de conformité.

Le registre interne est tenu dans `docs/governance/legal_archive/registre_traitements.md` (à créer en bootstrap).

---

## Checklist publication Privacy V0

- [ ] Iubenda Privacy Policy générée, sections renforcées avec le contenu ci-dessus
- [ ] Remplacement `[Nom]`, `[SIRET]`, `[adresse]` par valeurs réelles
- [ ] Création comptes email : `privacy@`, `security@`, `support@` (via ProtonMail Custom Domain ~$5/mo, ou Fastmail $5/mo)
- [ ] Publication sur `/privacy`
- [ ] Lien dans footer
- [ ] Lien dans modale signup obligatoire (case à cocher "J'accepte la Politique de Confidentialité")
- [ ] Tenue du registre interne `legal_archive/registre_traitements.md`
- [ ] DPA Stripe accepté (clic acceptation dans le dashboard Stripe)
- [ ] DPA Anthropic accepté (signature électronique sur leur portal)
- [ ] DPA Fly.io accepté (clic acceptation)
- [ ] Archive PDF horodatée dans `docs/governance/legal_archive/privacy_v0_2026_05_XX.pdf`
