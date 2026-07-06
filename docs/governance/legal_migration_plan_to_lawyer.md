# Plan de migration V0 → V1 (avocat fintech FR) — M3

> ⚠️ **SUSPENDU SINE DIE (2026-07-06)** — décision fondateur : aucune
> consultation avocat planifiée. Stratégie en vigueur = posture descriptive +
> exclusion US/UK/OFAC, cf.
> `decisions/2026-07-06_conformite_posture_descriptive_exclusion.md`.
> Ce plan ne redevient pertinent que si l'ouverture des États-Unis ou du
> Royaume-Uni est un jour souhaitée (et devrait alors viser un avocat
> québécois/canadien, pas le cadre auto-entrepreneur FR décrit ici).

**Date création** : 2026-05-26
**Cible exécution** : M3 (2026-08-XX environ, ~12 semaines après lancement bootstrap)
**Référence** : `legal_bootstrap_strategy_2026_05_26.md` + `legal_templates/`

---

## 0. Critères de déclenchement

La migration vers CGU avocat-signed est déclenchée **dès que les 3 conditions sont satisfaites simultanément** :

| Critère | Cible | Vérification |
|---|---|---|
| **MRR B2C** | ≥ $1500/mois stable | Stripe Dashboard, moyenne 60 jours glissants |
| **Trésorerie disponible** | ≥ 4 000 € sur compte pro | Relevé bancaire Shine/Qonto |
| **Stabilité opérationnelle** | Aucun incident légal majeur ouvert | Journal `docs/incidents/` clean |

Si MRR < $1500 à M3 → **prolonger le bootstrap jusqu'à M4-M6**, en restant vigilant sur les risques résiduels.

Si MRR explose > $5k avant M3 → **anticiper la migration à M2** (urgence relative à l'exposition).

---

## 1. RFQ — Sélection cabinet (M3 semaine 1)

### Cabinets à contacter (3 minimum)

Trois cabinets avec spécialité fintech FR + ≥ 5 ans d'expérience B2C SaaS :

| Cabinet | Spécialité | Localisation | Email contact |
|---|---|---|---|
| **Hashtag Avocats** | Fintech, RegTech, blockchain | Paris | `contact@hashtag-avocats.com` |
| **Lexing Alain Bensoussan** | Tech & data, fintech | Paris | `lexing.tech@alain-bensoussan.com` |
| **Couvrelles & Marchand-Berdat** | Fintech, conformité financière | Paris | À identifier |
| **De Gaulle Fleurance & Associés** (alternative) | Droit financier, MiFID | Paris | Via formulaire site |
| **August Debouzy** (alternative haut de gamme) | Fintech, M&A | Paris | Via formulaire site (cher) |

### Email RFQ type (à envoyer aux 3 cabinets)

```
Objet : RFQ — Relecture CGU/CGV/Privacy SaaS fintech B2C en phase de croissance

Bonjour Maître,

Je dirige M.I.A. Markets, un SaaS B2C français en phase Early Access
qui produit des analyses algorithmiques éducatives pour les marchés Or et FX.

Statut actuel :
- Auto-entrepreneur français
- MRR : ~$XXXX/mois
- Abonnés payants : ~XX (cap actuel 50)
- Tiers : FREE / STARTER $29 / PRO $79 / INSTITUTIONAL $1990 (futur)
- Géographie : France, Belgique, Suisse, Luxembourg
- Posture : outil pédagogique, pas de conseil en investissement,
  edge_claim explicitement False, refus pédagogique chatbot scripté
- Compliance UE 2024/2811 finfluencer : wording strictement éducatif
- RGPD : Privacy Policy V0 publiée

Mission demandée :
1. Relecture et amélioration des CGU/CGV V0 actuelles (~30 articles,
   basées sur template Iubenda + ajouts fintech-specific)
2. Relecture et amélioration de la Privacy Policy V0
3. Conseil sur :
   - Limitation de responsabilité optimisée fintech
   - Clauses MiFID exemption financial influencer
   - Procédure médiation conso L.612-1 (plateforme à choisir)
   - DPA B2B template pour tier INSTITUTIONAL (futur)
4. Engagement responsabilité avocat sur la conformité des documents signés

Documents disponibles sur demande pour préparation devis :
- CGU/CGV V0 (markdown, ~5 pages)
- Privacy Policy V0 (markdown, ~4 pages)
- Stratégie de bootstrap légal (markdown, ~10 pages)
- Description du produit + screenshots
- Audit risques résiduels V0 quantifié

Devis demandé pour :
A) Mission "Audit + amendements" (sans représentation litige future)
B) Idem A + souscription RC Pro fintech (recommandation cabinet partenaire)
C) Idem A + adhésion médiateur conso (recommandation + souscription)

Budget cible : 3 000 - 5 000 € HT.
Délai cible : 3-4 semaines.

Je peux envoyer les documents préparatoires en pré-RFQ pour devis précis.

Cordialement,
[Prénom Nom]
[Adresse]
SIRET [XXX]
+33 X XX XX XX XX (sur demande)
```

### Critères de sélection (grille à scorer 1-5)

| Critère | Pondération |
|---|---|
| **Expérience fintech FR vérifiable** (≥ 5 ans, ≥ 3 références) | 25 % |
| **Budget tenable** (3-5 k€) | 20 % |
| **Délai engagé** (≤ 4 semaines) | 15 % |
| **Lisibilité du devis** (transparent, sans frais cachés) | 10 % |
| **Réactivité communication** (réponse RFQ < 5 jours ouvrés) | 10 % |
| **Couverture MiFID 2024/2811** explicite dans la lettre de mission | 10 % |
| **Engagement responsabilité** (clause de garantie sur conformité) | 10 % |

**Sélection sous 5 jours ouvrés** après réception des 3 devis.

---

## 2. Brief préparatoire avocat (M3 semaine 2)

### Documents à fournir au cabinet sélectionné

1. **CGU/CGV V0** (markdown + PDF horodaté)
2. **Privacy Policy V0** (markdown + PDF)
3. **Mentions légales** (PDF)
4. **`legal_bootstrap_strategy_2026_05_26.md`** (stratégie complète + risques résiduels documentés)
5. **Description produit** (extrait `client_information_explained.txt` ou screenshot mockup HTML)
6. **Audit wording compliance** (`docs/audits/wording_audit_M0.md` si tenu à jour)
7. **Journal incidents M0-M3** (si tu en as eu — montre maturité)
8. **Statistiques d'usage agrégées** (nombre abonnés, MRR, taux refund, plaintes)

### Questions à poser dès la première réunion

1. **MiFID 2024/2811** : ton wording actuel est-il suffisant pour bénéficier de l'exemption "outil pédagogique" ? Quelles précisions ajouter ?
2. **Limitation responsabilité** : la jurisprudence FR récente (2023-2025) a-t-elle évolué sur les plafonds B2B / B2C SaaS fintech ?
3. **Médiation conso** : quelle plateforme recommandes-tu (CM2C, MEDICYS, autres) ? Coût annuel ? Taux d'usage observé pour ton type d'activité ?
4. **DPA B2B INSTITUTIONAL** : faut-il préparer le template dès maintenant ou attendre le 1er prospect B2B confirmé ?
5. **RC Pro fintech complète** : tu recommandes Stoïk, Hiscox, ou autre ? Bundle ou polices séparées ?
6. **Migration auto-entrepreneur → SASU** : à quel niveau de CA recommandes-tu de basculer ? Conséquences sur la responsabilité ?
7. **Géographie** : sécuriser CH (hors UE) impose-t-il des clauses supplémentaires de juridiction ?

---

## 3. Délais et livrables (M3 semaines 3-5)

| Semaine | Étape | Livrable |
|---|---|---|
| M3-S3 | Avocat relit CGU + Privacy V0 + audit conformité | Premier retour synthèse |
| M3-S3 | Réunion call 1h pour discuter retour | Compte-rendu écrit |
| M3-S4 | Avocat produit V1 amendée | CGU V1 + Privacy V1 + Mentions V1 |
| M3-S4 | Toi : relecture V1 + questions résiduelles | Liste annotée |
| M3-S5 | Avocat finalise V1.1 | Versions finales signées par avocat (en-tête cabinet) |
| M3-S5 | Souscription RC Pro fintech complète (Stoïk/Hiscox) | Police signée |
| M3-S5 | Adhésion médiateur conso (CM2C ou MEDICYS) | Confirmation adhésion |

---

## 4. Migration technique V0 → V1 sans rupture (M4 semaine 1)

### Process publication V1

#### J-14 (M3-S5 final)
- V1 prête, validée toi et avocat
- Archive V0 dans `docs/governance/legal_archive/cgu_v0_2026_05_XX.pdf`
- Préparation email annonce aux abonnés

#### J-14 → J-1 (période transition légale)
- Email à tous les abonnés actifs :

```
Objet : M.I.A. Markets · Mise à jour de nos conditions générales

Bonjour [Prénom],

Nous mettons à jour nos Conditions Générales d'Utilisation et notre
Politique de Confidentialité, à compter du [DATE J+14].

Pourquoi ce changement :
Notre service grandit, et nous avons fait relire nos documents par
un cabinet d'avocats spécialisé en fintech FR. Le résultat : des
documents plus clairs, mieux protégés pour vous et pour nous.

Principales évolutions :
- Clause de médiation conso désormais nommée et opérationnelle
- Limitation de responsabilité réécrite selon la jurisprudence 2025
- Procédure d'exercice des droits RGPD plus détaillée
- [3-5 autres points clés selon avocat]

Vous pouvez consulter les nouvelles versions ici :
- CGU V1 : [URL]
- Privacy V1 : [URL]

Vous n'avez RIEN à faire pour continuer votre abonnement.
À partir du [DATE J+14], votre utilisation continue du service
vaut acceptation des nouvelles conditions.

Si vous n'êtes pas d'accord, vous pouvez résilier votre abonnement
sans frais avant le [DATE J+14] depuis votre espace client.
Vous serez intégralement remboursé au prorata des jours restants.

Pour toute question : privacy@mia.markets

Merci de votre confiance.

[Prénom]
Éditeur, M.I.A. Markets
```

- Pop-up discret dans webapp : "Nos CGU évoluent le [date]. En savoir plus"
- Lien CGU V1 publié au footer + page dédiée

#### J0 (mise en vigueur)
- Bascule des liens CGU/Privacy vers V1
- Bouton "Accepter les nouvelles CGU" lors de la prochaine connexion (un seul clic, mémorisé)
- Pas de friction excessive : si l'utilisateur ne clique pas, l'utilisation reste valide (l'utilisation continue vaut acceptation, c'est annoncé dans l'email)

#### J+30 (vérification)
- Audit : tous les utilisateurs ont vu la nouvelle CGU au moins une fois ?
- Taux de résiliation observé pendant la transition < 5 % (sinon = problème, à analyser)
- Premier feedback : appels support concernant les changements ?

### Pas de breaking change technique
- Le contrat utilisateur côté code reste identique
- Les clauses légales évoluent, mais pas les fonctionnalités
- Migration **100 % sans rupture produit**

---

## 5. Mise à jour autres documents (M4 semaine 2)

Une fois CGU V1 et Privacy V1 publiées, mettre à jour :

- [ ] **Mockup HTML** (`mockups/v3/best_concept_demo.html`) : footer compliance mis à jour avec nouveaux liens
- [ ] **Email templates** (incidents, RGPD, accusés) : références CGU V1
- [ ] **Chatbot system prompt** : références aux nouvelles CGU pour les questions compliance
- [ ] **Disclaimer marketing** (landing, Telegram, emails) : harmoniser avec V1
- [ ] **Documentation API** (futur INSTITUTIONAL) : ajouter DPA B2B template fourni par avocat
- [ ] **MEMORY.md** : ajouter entrée migration V0 → V1 effectuée

---

## 6. Souscription RC Pro fintech complète (M4 semaine 2)

### Bundle recommandé

**Option Stoïk Bundle Fintech** :
- RC Pro Fintech : couvre erreurs algorithmiques, mauvais conseil (même implicite), litige client
- Cyber-risque : ransomware, brèche RGPD, perte de données
- Protection Juridique : avocat couvert pour les premières heures de défense
- Coût indicatif : **3 000 - 5 000 € / an** selon chiffre d'affaires

**Option Hiscox** (alternative) :
- RC Pro classique freelance : moins fintech-specific mais éprouvé
- Coût ~ 1 500 - 3 000 € / an
- Recommandé si Stoïk trop cher ou trop spécialisé

### Avant souscription
- Vérifier que la police couvre :
  - ✓ Responsabilité pour erreurs algorithmiques
  - ✓ Responsabilité conseil (même si tu n'en donnes pas — précaution)
  - ✓ Cyber-risque (brèche RGPD jusqu'à 100k€ minimum)
  - ✓ Protection juridique avec franchise basse
  - ✓ Géographie FR + BE + CH + LU minimum
  - ✓ Activité "service éducatif analyse financière en ligne" (et pas "conseil en investissement" qui exclurait fintech)

---

## 7. Adhésion médiateur conso (M4 semaine 2)

### Choix plateforme

| Plateforme | Coût annuel | Délai traitement | Notes |
|---|---|---|---|
| **CM2C** | 150 € | ~60 jours | Spécialisé tech & e-commerce |
| **MEDICYS** | 150-200 € | ~60-90 jours | Médiateur national reconnu |
| **MNS** (Médiateur national de la consommation et du sport) | Variable | — | Plutôt secteur spécifique, à vérifier |

Recommandation : **CM2C** par défaut, à valider avec avocat selon ta spécialité.

### Process adhésion
1. Demande d'adhésion en ligne (CM2C : `cm2c.net`)
2. Fourniture KBis / extrait INSEE auto-entreprise
3. Paiement cotisation annuelle
4. Confirmation sous 2-4 semaines
5. **Obligation** : afficher les coordonnées de la plateforme sur le site (CGU article médiation + mentions légales)
6. **Obligation** : intégrer le lien dans tous les emails de litige

---

## 8. Migration auto-entrepreneur → SASU (M6-M12 selon CA)

### Critères de déclenchement
- CA HT annuel approche **77 700 € (plafond AE services)** → migration obligatoire
- OU séparation patrimoine personnel souhaitée (protection en cas de litige important)
- OU embauche d'un premier salarié / collaborateur
- OU levée de fonds envisagée

### Coût migration
- Création SASU : ~1 500 € (avocat + greffe + statuts)
- Comptable obligatoire : ~100-200 €/mois
- Charges sociales différentes (président SASU : régime général + dividendes)
- À budgéter en M6-M9

### Pas urgent en M3
La migration SASU n'est pas nécessaire pour la V1 légale. Reste auto-entrepreneur tant que CA < 50 000 € HT/an.

---

## 9. Checklist M3 — vue d'ensemble

- [ ] Vérifier déclencheurs (MRR ≥ $1500 stable 60j, trésorerie ≥ 4k€, no incident ouvert)
- [ ] Envoyer 3 RFQ aux cabinets sélectionnés
- [ ] Réception 3 devis sous 5-7 jours
- [ ] Sélection cabinet (grille scoring)
- [ ] Signature lettre de mission + acompte 30-50 %
- [ ] Envoi documents préparatoires
- [ ] Réunion call 1 (synthèse retour)
- [ ] Réception V1 amendée par avocat
- [ ] Relecture toi + questions résiduelles
- [ ] Réception V1.1 finale signée avocat
- [ ] Souscription RC Pro fintech complète (Stoïk/Hiscox)
- [ ] Adhésion médiateur conso (CM2C/MEDICYS)
- [ ] Envoi email annonce abonnés J-14
- [ ] Bascule technique CGU/Privacy V1 J0
- [ ] Audit J+30
- [ ] Mise à jour MEMORY.md
- [ ] Solde paiement avocat

---

## 10. Notes finales

### Si retard sur déclencheurs
Si à M3 le MRR ne dépasse pas $1500 :
- **Prolonger bootstrap M4-M6**
- **Renforcer la posture Pilier 3 (Early Access · Educational Use)** dans la communication
- **Limiter encore le cap utilisateurs** (30 au lieu de 50) pour minimiser exposition
- **Ne pas céder à la pression de monter de tier** sans budget avocat

### Si M3 atteint mais cabinet débordé
Les bons cabinets fintech FR ont parfois 1-2 mois de délai. Si délai > 4 semaines :
- Engager le 2e choix sur la grille
- Ou attendre 4 semaines supplémentaires avec bootstrap (gain de revenue compense l'attente)
- Ne **pas** signer avec un cabinet "rapide pas cher" si conformité fintech non vérifiée

### Si budget dépassé
- Négocier 50 % paiement à la signature + 50 % à livraison (échelonnement)
- Négocier l'exclusion de RC Pro fintech recommandation (économie 500-1000 €) : tu choisis toi-même Hiscox seul
- Refuser les ajouts "nice-to-have" qui font gonfler le devis (audit blockchain, audit IA, etc. non nécessaires V1)

### Conserver la trace
Tous les documents légaux signés (V0, V1) + correspondance avocat + factures + polices d'assurance sont archivés dans `docs/governance/legal_archive/` avec date et version.
