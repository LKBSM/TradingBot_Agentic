# CGU / CGV V0 — Template bootstrap M.I.A. Markets

**Date** : 2026-05-26
**Statut** : V0 bootstrap — à migrer V1 avocat-signed à M3 (cf. `legal_migration_plan_to_lawyer.md`)
**Usage** : ce template doit être combiné avec les CGU/CGV Iubenda Pro auto-générées. Les clauses ci-dessous sont les **ajouts fintech-specific** qui renforcent le template Iubenda.

---

## ⚠️ Mode d'emploi

1. Souscrire **Iubenda Pro** ($30/mo) : `iubenda.com`
2. Configurer projet "SaaS B2C avec abonnement Stripe" → génère CGU/CGV/Privacy de base
3. Récupérer le markdown / HTML des CGU/CGV générées
4. **Insérer les clauses ci-dessous aux sections indiquées** (numérotation Iubenda peut varier, adapter)
5. Publier sur le site sous `/cgu` et `/cgv`
6. Lien obligatoire dans le footer de chaque page

---

## Clauses fintech-specific à ajouter au template Iubenda

### Section "Objet du Service" — REMPLACER intégralement par :

> **Article X — Objet du Service**
>
> M.I.A. Markets (ci-après "le Service") est un **outil pédagogique d'analyse algorithmique** appliqué aux marchés financiers, exclusivement l'or (XAUUSD) et l'euro/dollar (EURUSD).
>
> Le Service produit, à partir de données de marché historiques et publiques, des lectures structurées décrivant l'état du marché : direction de la tendance, conviction calibrée statistiquement, contexte de régime de volatilité, contexte événementiel macroéconomique.
>
> Le Service est édité par **[Nom], micro-entrepreneur (SIRET XXX XXX XXX XXXXX)**, ci-après "l'Éditeur".
>
> Le Service est accessible :
> - via une interface web (webapp)
> - via le canal Telegram (selon tier souscrit)
> - via une interface conversationnelle (chatbot)
>
> Le Service est **en phase d'accès anticipé (Early Access)**. Les fonctionnalités, performances et tarifs peuvent évoluer. L'Éditeur ne garantit aucune continuité de service pendant cette phase.

### Section "Nature du Service — Non-conseil" — AJOUTER :

> **Article X+1 — Absence de conseil en investissement**
>
> Le Service ne constitue **ni un service d'investissement** au sens de l'article L. 321-1 du Code monétaire et financier, **ni une activité de conseil en investissements financiers** au sens de l'article L. 541-1 du même code. L'Éditeur n'est pas agréé par l'Autorité des marchés financiers (AMF) et n'a pas vocation à l'être pour les besoins du Service.
>
> Aucune lecture, narratif, statistique, alerte ou interaction chatbot produits par le Service ne saurait être interprété comme :
> - un conseil en investissement personnalisé ou général,
> - une recommandation d'achat, de vente, de conservation ou de gestion d'un instrument financier,
> - une incitation, directe ou indirecte, à conclure une opération sur les marchés financiers,
> - une garantie de performance ou de résultat.
>
> Le Service décrit. L'Utilisateur décide.
>
> **Article X+2 — Refus de prescription**
>
> En cas de question prescriptive ("dois-je acheter ?", "dois-je vendre ?", "fixer un stop-loss à quel niveau ?"), le chatbot est paramétré pour refuser pédagogiquement de répondre. Ce refus est constitutif de la nature pédagogique du Service.
>
> **Article X+3 — Risque de perte en capital**
>
> Les marchés financiers comportent un risque de perte en capital. Les performances passées, qu'elles soient présentées sous forme de profit factor, taux de réussite, drawdown maximal ou intervalle de confiance bootstrap, sont issues de tests rétrospectifs et **ne préjugent en rien des performances futures**.
>
> L'Utilisateur reconnaît avoir conscience de ce risque, l'avoir évalué au regard de sa situation patrimoniale, et avoir la capacité financière de supporter une perte totale du capital qu'il déciderait d'engager sur les marchés à la suite de la consultation du Service.

### Section "Géographie" — AJOUTER :

> **Article X+4 — Restriction géographique**
>
> Pendant la phase d'accès anticipé, le Service est restreint aux résidents fiscaux de :
> - France métropolitaine et DROM-COM
> - Belgique
> - Suisse
> - Luxembourg
>
> Les résidents d'autres juridictions, notamment les États-Unis, le Canada, le Royaume-Uni, l'Australie, ainsi que toute personne physique ou morale figurant sur les listes de sanctions internationales (OFAC, UE, ONU), **ne peuvent pas s'inscrire ni accéder au Service**. L'Éditeur met en œuvre un blocage géographique au niveau de l'adresse IP de l'Utilisateur.
>
> L'Utilisateur garantit qu'il réside dans une juridiction autorisée et qu'il ne contourne pas le blocage géographique (VPN, proxy). Tout contournement constitue un manquement aux présentes CGU et entraîne la résiliation immédiate du compte sans remboursement.

### Section "Abonnement et tarifs" — AJOUTER :

> **Article X+5 — Tarifs et tiers d'abonnement**
>
> Le Service est proposé selon les tiers suivants :
>
> | Tier | Prix mensuel | Engagement |
> |---|---|---|
> | FREE | 0 € | Aucun |
> | STARTER | 29 USD | Mensuel, résiliable à tout moment |
> | PRO | 79 USD | Mensuel, résiliable à tout moment |
> | INSTITUTIONAL | 1990 USD | Engagement annuel 12 mois minimum |
>
> Les tarifs sont susceptibles d'évoluer pendant la phase d'accès anticipé. Toute évolution tarifaire est communiquée 30 jours à l'avance et n'est applicable qu'à compter du renouvellement suivant.
>
> Les paiements sont gérés par Stripe Inc. L'Utilisateur accepte les conditions générales de Stripe accessibles à `stripe.com/legal`.
>
> **Article X+6 — Période d'essai et droit de rétractation renforcé**
>
> L'Utilisateur bénéficie :
> - d'une **période d'essai gratuite de 14 jours** sans engagement de carte bancaire pour le passage FREE → STARTER,
> - d'une **période d'essai de 14 jours avec carte bancaire** pour le passage STARTER → PRO,
> - d'un **droit de remboursement intégral pendant les 30 premiers jours** suivant le premier paiement effectif, sans condition ni justification ("no questions asked").
>
> Au-delà de cette période, l'abonnement peut être résilié à tout moment depuis le portail client Stripe, sans pénalité, l'accès au Service étant maintenu jusqu'à la date d'échéance déjà payée.
>
> **Article X+7 — Quotas et limites**
>
> Chaque tier ouvre un certain volume de lectures, de questions au chatbot et d'actifs analysables. Ces quotas sont précisés sur la page tarifs et sont applicables en temps réel via le système de quota interne. Tout dépassement entraîne une notification de plafond atteint et l'invitation à monter de tier ; aucun usage abusif (contournement, automatisation) n'est toléré et entraîne la résiliation immédiate sans remboursement.

### Section "Données personnelles" — AJOUTER :

> **Article X+8 — Données personnelles**
>
> Le traitement des données personnelles est régi par la Politique de Confidentialité accessible à `/privacy`. L'Éditeur s'engage à respecter le Règlement général sur la protection des données (RGPD).
>
> Les données collectées sont strictement limitées à ce qui est nécessaire au fonctionnement du Service : adresse e-mail, identifiant de tier, token Stripe (sans accès aux données de paiement), préférences linguistiques, journaux d'usage anonymisés.
>
> Aucune donnée patrimoniale, fiscale ou de portefeuille de trading n'est collectée par le Service.

### Section "Limitation de responsabilité" — REMPLACER intégralement par :

> **Article X+9 — Limitation de responsabilité**
>
> Dans toute la mesure permise par la loi applicable, et sans préjudice du droit conso impératif :
>
> a) L'Éditeur ne saurait être tenu responsable des **pertes financières** directes ou indirectes subies par l'Utilisateur à la suite de décisions d'investissement prises sur la base, en lien avec, ou ayant utilisé le Service. L'Utilisateur reconnaît expressément que le Service ne constitue pas un conseil en investissement et qu'il prend ses décisions en pleine autonomie.
>
> b) L'Éditeur ne garantit aucune **disponibilité de service** pendant la phase d'accès anticipé. Le Service est fourni "tel quel" et "selon disponibilité", sans garantie expresse ou implicite de continuité, d'exactitude, d'exhaustivité ou d'adéquation à un objectif particulier.
>
> c) La responsabilité maximale de l'Éditeur, tous chefs de préjudice confondus, est plafonnée au **montant des sommes effectivement versées par l'Utilisateur au cours des 12 derniers mois** précédant l'évènement générateur de responsabilité.
>
> d) Cette limitation ne s'applique pas en cas de **faute lourde, dol, ou atteinte à un droit impératif du consommateur** prévu par le Code de la consommation.

### Section "Médiation" — AJOUTER :

> **Article X+10 — Médiation de la consommation**
>
> En cas de litige avec un consommateur n'ayant pu être résolu par voie amiable directe, l'Utilisateur peut recourir gratuitement à un médiateur de la consommation conformément à l'article L. 612-1 du Code de la consommation.
>
> L'Éditeur adhère à la plateforme **[Nom plateforme — à compléter à M2]**, dont les coordonnées et la procédure sont disponibles à : [URL].
>
> **Article X+11 — Loi applicable et juridiction**
>
> Les présentes CGU sont régies par le **droit français**.
>
> Tout litige relatif à leur formation, leur interprétation ou leur exécution est soumis aux **tribunaux du ressort du domicile du défendeur** (article R.631-3 du Code de la consommation pour les litiges B2C), ou aux tribunaux compétents en application du Règlement Bruxelles I bis pour les Utilisateurs résidant en Belgique ou au Luxembourg. Pour les Utilisateurs résidant en Suisse, les juridictions suisses compétentes en application de la Convention de Lugano.

---

## Mentions obligatoires supplémentaires

### Identification de l'Éditeur (Mentions Légales — page séparée)

À compléter dans `mentions_legales_auto_entrepreneur.md` (template fourni).

### Hébergeur

> **Hébergeur du Service** : Fly.io Inc., 2261 Market Street #4990, San Francisco, CA 94114, USA. `fly.io`

### Contact Réclamations

> **Réclamations** : `support@mia.markets` (à créer — utiliser ProtonMail ou Fastmail pour bootstrap)
> **Délai de réponse SLA** : 5 jours ouvrés
> **Médiation** : voir Article X+10

---

## Checklist publication CGU V0

- [ ] Iubenda Pro souscrit, CGU/CGV de base générées
- [ ] Insertion des 11 articles fintech-specific ci-dessus dans le template Iubenda
- [ ] Remplacement des `[Nom]`, `[SIRET]`, `[Nom plateforme]` par les valeurs réelles
- [ ] Publication sur `/cgu`
- [ ] Lien dans footer de chaque page
- [ ] Lien dans modale signup Stripe Checkout obligatoire (case à cocher "J'accepte les CGU")
- [ ] Archive PDF horodatée des CGU V0 dans `docs/governance/legal_archive/cgu_v0_2026_05_XX.pdf`
- [ ] Notification email aux abonnés FREE (s'il y en a déjà) avec lien CGU

---

## Limitations connues de V0 vs V1 avocat

| Sujet | V0 (bootstrap) | V1 (avocat M3) |
|---|---|---|
| Clauses MiFID spécifiques | génériques | détaillées sur exemptions financial influencer |
| Limitation responsabilité | plafond 12 mois CA | optimisée selon jurisprudence fintech récente |
| Médiation | renvoi générique | plateforme nommée + procédure intégrée |
| DPA B2B | absent (pas de tier INSTITUTIONAL en V0) | template B2B complet |
| Juridiction internationale | restreint FR + BE + CH + LU (simplification) | élargi UE complète + clauses internationales |
| Audit conformité finfluencer | par toi-même | par avocat avec engagement responsabilité |

**Risque résiduel acceptable pendant 2-3 mois** si Piliers 3 (posture éducative), 5 (cap utilisateurs), 6 (refund 30j) du `legal_bootstrap_strategy_2026_05_26.md` sont strictement respectés.
