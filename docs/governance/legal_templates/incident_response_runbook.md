# Runbook — Réponse aux incidents légaux et opérationnels

**Date** : 2026-05-26
**Statut** : V0 bootstrap — à étendre si volume d'incidents augmente
**Usage** : process écrit à suivre en cas de plainte client, brèche RGPD, suspension Stripe, avertissement AMF, ou demande d'autorité publique.

---

## Principe général

**Calme. Documente. Réponds vite. Garde une trace.**

90 % des incidents se désamorcent par :
1. Une **réponse rapide** (sous 24-48h) qui prouve que tu prends au sérieux
2. Une **proposition de solution** (refund, correctif, explication)
3. Une **trace écrite** du process suivi (couverture si litige escalade)

Toute communication entrante doit déclencher la création d'un fichier de suivi dans `docs/incidents/YYYY-MM-DD_short_description.md`.

---

## 🟡 Cas 1 — Plainte client (insatisfaction, demande remboursement, contestation)

### Réception
- Canal : email `support@mia.markets`, message Telegram, ticket Stripe
- SLA réponse interne : **24h ouvrées**

### Process
1. **Accusé de réception sous 24h** (template ci-dessous)
2. **Lecture attentive de la plainte** — identifier la nature (demande refund, désaccord lecture algo, problème technique, plainte juridique)
3. **Si remboursement demandé pendant période 30j** → **refund immédiat sans question**, message empathique. Process terminé.
4. **Si désaccord sur la nature du Service** → message pédagogique avec rappel du wording éducatif (CGU article X), rappel refus pédagogique
5. **Si plainte juridique formelle** ou mention "AMF/CNIL/avocat" → escalade immédiate vers process Cas 5

### Template email réponse (réception)

```
Objet : RE: [Ticket #XXXX] Votre demande concernant M.I.A. Markets

Bonjour [Prénom],

Merci pour votre message reçu le [date]. Je l'ai bien lu et je le prends au sérieux.

[Si refund pendant 30j]
Conformément à notre engagement remboursement 30 jours sans question,
je procède au remboursement immédiat. Vous recevrez la confirmation
Stripe dans les prochaines minutes.

[Si désaccord sur nature du Service]
M.I.A. Markets est un outil pédagogique d'analyse algorithmique.
Il ne formule aucune recommandation d'investissement et ne garantit
aucun résultat. C'est précisé dans nos CGU (article X) et dans tous
nos disclaimers. Si vous souhaitez tout de même résilier et obtenir
un remboursement, c'est possible sous 30 jours sans condition.

[Si problème technique]
Je remonte le bug à mon équipe. Délai de résolution estimé : [Xh / Xj].
Je vous tiens informé dès que c'est corrigé.

[Toujours conclure par]
Vous pouvez me répondre directement si vous avez d'autres questions.
Si vous souhaitez exercer un droit RGPD (accès, suppression de données),
écrivez à privacy@mia.markets.

Cordialement,
[Prénom]
Éditeur, M.I.A. Markets
```

### Traçage
Créer `docs/incidents/YYYY-MM-DD_plainte_[ticket_id].md` avec :
- Date réception
- Identité client (email, tier)
- Nature de la plainte
- Réponse envoyée (copie)
- Résolution (refund / correctif / autre)
- Date clôture

---

## 🟠 Cas 2 — Demande RGPD (accès, rectification, suppression, portabilité)

### Réception
- Canal : email `privacy@mia.markets`
- SLA légal : **30 jours** (RGPD art. 12.3)

### Process
1. **Accusé de réception sous 72h** (template ci-dessous)
2. **Vérification identité du demandeur** : email + question de vérification basique. Si suspicion d'usurpation : demander pièce d'identité.
3. **Identification de la demande** :
   - **Accès (art. 15)** : export JSON complet du compte
   - **Rectification (art. 16)** : modification en base + confirmation
   - **Suppression (art. 17)** : suppression / anonymisation cascade (cf. §3 sous-process)
   - **Portabilité (art. 20)** : export CSV / JSON normalisé
   - **Opposition (art. 21)** : opt-out marketing + désactivation analytique pour cet utilisateur
4. **Réalisation de la demande sous 30 jours**
5. **Notification de réalisation** (template)

### Sous-process suppression compte
1. Désactivation compte immédiate (login impossible)
2. Période de "soft-delete" 30 jours (récupération possible en cas d'erreur)
3. Suppression / anonymisation effective J+30 :
   - Email → remplacé par `deleted_XXXXX@anonymous.mia.markets`
   - Préférences → effacées
   - Historique lectures → conservé anonymisé (statistiques produit)
   - Conversations chatbot → conservées 30j puis anonymisées
   - Données Stripe → conservées 10 ans (obligation comptable) mais détachées du compte utilisateur
   - Logs d'authentification → conservés 12 mois sans lien avec l'identité
4. Notification finale au demandeur
5. Trace dans `docs/incidents/YYYY-MM-DD_rgpd_suppression_[user_id_hashe].md`

### Template email accusé de réception
```
Objet : RE: [RGPD] Votre demande du [date]

Bonjour,

J'accuse réception de votre demande d'exercice du droit RGPD.
Type identifié : [Accès / Rectification / Suppression / Portabilité / Opposition].

Je traiterai votre demande dans le délai légal de 30 jours.

Si je dois vérifier votre identité (en cas de suspicion d'usurpation),
je reviendrai vers vous sous 48h.

Cordialement,
[Prénom]
Responsable de traitement
```

### Si demande complexe ou volumineuse
Le RGPD permet une prolongation de 2 mois supplémentaires (art. 12.3) avec notification motivée au demandeur. À utiliser avec mesure.

---

## 🔴 Cas 3 — Notification de brèche de sécurité

### Détection
- Indicateurs : accès non autorisé détecté dans logs, vulnérabilité signalée par chercheur sécu, perte de données, ransomware, hack compte
- **Stop the bleed first** : isoler, contenir, sauvegarder logs avant analyse

### Process
1. **T+0** — détection + containment (couper l'accès, désactiver le vecteur si identifié)
2. **T+1h** — première analyse : nature, périmètre, données impactées
3. **T+24h** — décision : faut-il notifier CNIL ?
   - **OUI si** : risque pour les droits et libertés des personnes (données personnelles compromises, identification possible des personnes affectées)
   - **NON si** : pas de PII impactée, ou impact mineur (logs anonymes leakés)
4. **T+72h max** — si CNIL → **notification via `notifications.cnil.fr`** (formulaire en ligne, structuré)
5. **T+72h max** — si risque élevé pour les personnes → **notification individuelle aux utilisateurs concernés** par email (art. 34 RGPD)
6. **Correctif en place** avant la notification (sinon CNIL escalade)
7. **Documentation complète** dans `docs/incidents/YYYY-MM-DD_breach.md` avec timeline, cause racine, correctif, leçons apprises

### Contenu notification CNIL
- Nature de la violation (intrusion, perte, divulgation, etc.)
- Catégories de données concernées
- Nombre approximatif de personnes affectées
- Conséquences probables
- Mesures prises pour atténuer
- Coordonnées du contact

### Template notification utilisateur (si requise)
```
Objet : ⚠ Incident de sécurité — votre compte M.I.A. Markets

Bonjour,

Nous vous informons d'un incident de sécurité détecté le [date] concernant
les données suivantes : [liste].

Nature de l'incident : [description neutre, factuelle]

Mesures déjà prises :
- [Action 1]
- [Action 2]
- Notification de la CNIL effectuée le [date]

Recommandations pour vous :
- Changer votre mot de passe (lien)
- Vérifier vos méthodes d'authentification
- Nous contacter à security@mia.markets si vous constatez
  une activité suspecte

Nous sommes sincèrement désolés. Vous recevrez un rapport post-mortem
public dans 7 jours.

[Prénom]
Éditeur, M.I.A. Markets
```

---

## 🟡 Cas 4 — Suspension Stripe ou avertissement compliance

### Détection
- Email Stripe avec motif (souvent vague : "high-risk", "investment advice content", "trading signals")
- Notification dans le dashboard Stripe

### Process
1. **T+0** — lecture motif. Ne pas répondre dans la précipitation.
2. **T+2-4h** — audit du site et du wording : que peut bien viser Stripe ?
   - Recherche des termes : "signal", "buy", "sell", "guaranteed", "profit", "+X%", "investment advice"
   - Vérifier chatbot prompts + narratives LLM
   - Capture d'écran de chaque endroit incriminé
3. **T+1 jour** — correctifs déployés
4. **T+2 jours** — réponse Stripe avec :
   - Liste précise des corrections apportées (URL + screenshot avant/après)
   - Rappel du caractère pédagogique du Service
   - Rappel de la cible géographique restreinte (FR+BE+CH+LU)
   - Demande de réactivation
5. **Si suspension confirmée** → ouvrir compte **Paddle** ou **Mollie** en backup (moins strict que Stripe sur fintech bootstrap), migrer en 1-2 semaines.

### Template email réponse Stripe
```
Subject: Re: Account Review - M.I.A. Markets

Hi Stripe team,

Thank you for your email regarding the review of my account.

M.I.A. Markets is an **educational algorithmic analysis tool**
for gold and forex markets. It does NOT:
- Provide investment advice
- Make buy/sell recommendations
- Guarantee any performance or returns

The Service is restricted to residents of France, Belgium, Switzerland,
and Luxembourg via IP geo-blocking. Wording compliance with EU 2024/2811
(financial influencer regulation) is enforced.

Following your review, I have made the following corrections:
1. [URL] — Changed "X" to "Y"
2. [URL] — Removed claim "Z"
3. [URL] — Added educational disclaimer
4. [Screenshot before/after attached]

You can review the current site at: [URL].

The terms of service explicitly state the educational nature
(CGU article X): [link].

I'd appreciate a re-review at your convenience. Please let me know
if any additional changes are needed.

Best regards,
[Name]
Editor, M.I.A. Markets
SIRET: [XXX]
```

---

## 🔴 Cas 5 — Avertissement AMF / autorité de régulation

### Improbable mais procédure stricte

### Process
1. **T+0** — accuser réception au plus vite (sous 24h ouvrées)
2. **T+1 jour** — **CONSULTATION AVOCAT FINTECH OBLIGATOIRE**, même si budget non disponible. Urgence justifie la dépense ponctuelle (200-400 €/heure pour 2-4 heures = 400-1600 €). Cabinets recommandés :
   - **Hashtag Avocats** (Paris, spé fintech)
   - **Lexing Alain Bensoussan** (généraliste tech avec spé fintech)
   - **Couvrelles & Marchand-Berdat** (fintech FR)
3. **T+1-3 jours** — avec l'avocat :
   - Lire ensemble la lettre AMF
   - Identifier les points contestables ou conformes
   - Préparer la réponse formelle
4. **Pendant la procédure** :
   - Suspension immédiate de toute communication marketing
   - Pas de nouvelle souscription (gel du tier payant)
   - Communication transparente aux abonnés en cours (sans alarmisme)
5. **Réponse formelle dans le délai imparti** (30-60 jours généralement) via avocat
6. **Conformer si requis** ou **contester** selon conseil avocat

### À NE PAS FAIRE
- ❌ Répondre seul à l'AMF
- ❌ Continuer la communication marketing comme si de rien n'était
- ❌ Ignorer la lettre (procédure contentieuse automatique)
- ❌ Supprimer des éléments du site avant consultation (peut être interprété comme aveu)

---

## 🟢 Cas 6 — Demande d'autorité publique (judiciaire, police, fisc)

### Réception
- Réquisition judiciaire, demande de communication CNIL, contrôle URSSAF, demande fiscale

### Process
1. **Vérification d'authenticité** : la demande vient-elle vraiment de l'autorité ? Vérifier identités, signatures, références juridiques.
2. **Délais légaux** : généralement 30 jours, parfois 7-15 jours pour réquisitions urgentes
3. **Périmètre exact** : que demande-t-on précisément ? Données techniques ? Historique ? Identité utilisateur ?
4. **Conformer dans le périmètre exact** :
   - Réquisition judiciaire → fournir données demandées au format requis
   - Demande CNIL → fournir registre traitements + DPA + dispositif sécurité
   - URSSAF → fournir compta + déclarations CA
5. **Notification utilisateur** : sauf interdiction expresse (réquisition judiciaire sous secret), informer l'utilisateur impacté que ses données ont été transmises
6. **Documentation complète** de la demande et de la réponse

---

## Journal des incidents

Tous les incidents sont consignés dans `docs/incidents/` avec template :

```markdown
# Incident YYYY-MM-DD — [Titre court]

**Catégorie** : [Plainte / RGPD / Brèche / Stripe / AMF / Autorité]
**Statut** : [Ouvert / En cours / Résolu / Escaladé]
**Sévérité** : [Faible / Moyenne / Élevée / Critique]

## Détection
- Date / heure : 
- Canal : 
- Identité demandeur : 

## Nature
[Description neutre et factuelle]

## Timeline
- T+0 : ...
- T+Xh : ...
- T+Xj : ...

## Résolution
[Actions prises]

## Leçons apprises
[Process à améliorer pour éviter récurrence]

## Documents joints
- email_initial.eml
- response.eml
- screenshots/
```

---

## Annexe — Contacts utiles à avoir sous la main

| Organisme | Usage | Lien |
|---|---|---|
| **CNIL** | RGPD, notification brèche | `notifications.cnil.fr` |
| **AMF** | Réglementation financière | `amf-france.org` |
| **Médiateur conso** | Litige client | À adhérer M2 (CM2C ou MEDICYS) |
| **Stripe Support** | Compte, fraud, suspension | `support.stripe.com` |
| **Anthropic Trust & Safety** | Sécurité API LLM | `support@anthropic.com` |
| **Fly.io Support** | Hébergement | `community.fly.io` ou support payant |
| **Cabinet avocat fintech préselectionné** | Urgence légale | À identifier M0 même si pas engagé (1-2 RFQ exploratoires) |
| **Comptable / expert-compta** | Compta AE | Recommandation Shine ou comptable indépendant ~50 €/mois |

---

## Checklist d'usage

- [ ] Lire ce runbook une fois en entier avant lancement
- [ ] Préselectionner 2 cabinets avocat fintech (sans engager) — utile en cas d'urgence Cas 5
- [ ] Créer les comptes email `support@`, `privacy@`, `security@`, `contact@` avant lancement
- [ ] Créer le dossier `docs/incidents/` (vide pour démarrer)
- [ ] Tester le formulaire CNIL une fois en mode "draft" pour familiarisation (sans soumettre)
- [ ] Préparer un email type "réponse plainte" et un type "accusé RGPD" en draft Gmail/ProtonMail pour réactivité
- [ ] Identifier sa propre couverture RC Pro (Hiscox Freelance) pour numéro de police rapide en cas d'incident
