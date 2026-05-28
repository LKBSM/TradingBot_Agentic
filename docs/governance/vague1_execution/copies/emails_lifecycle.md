# Emails — Cycle de vie utilisateur

5 emails à automatiser via DG-131 (V2). Inclus en Pass 3 du kit d'exécution pour qu'ils soient prêts dès activation cycle de vie.

**Stack recommandé** : Postmark / Resend / Brevo (SendGrid évité car cher) — choisi à M2 quand le cycle est livré.

---

## Email 1 — Welcome (J0)

**Trigger** : signup confirmé (event `signup` côté analytics)
**Délai** : immédiat
**Subject** : Bienvenue dans M.I.A. Markets

```
Bonjour,

Bienvenue dans la phase d'accès anticipé de M.I.A. Markets.

Vous accédez à un outil pédagogique qui décrit l'état des marchés Or et FX
à partir de données historiques. Vous restez seul décideur de vos investissements.

Pour démarrer :
→ Votre première lecture XAU/USD M15 est disponible : [lien]
→ Sentinel (notre chatbot) peut répondre à vos questions : ouvrez-le depuis n'importe quelle lecture
→ Trois lectures par jour en plan FREE

Si vous souhaitez débloquer plus (4 actifs, structure SMC, régime + vol) :
[Voir les tarifs →]

À votre disposition pour toute question : support@mia.markets

L'équipe Sentinel

———
Démonstration paper-trading · Lecture algorithmique éducative · Ne constitue
ni un signal de trading ni un conseil en investissement.
```

---

## Email 2 — D+3 : Découverte chatbot

**Trigger** : 3 jours après signup, si l'utilisateur n'a posé aucune question chatbot
**Subject** : Avez-vous demandé à Sentinel pourquoi ?

```
Bonjour,

J'ai remarqué que vous n'avez pas encore posé de question à Sentinel
(notre chatbot intégré).

Sentinel est conçu pour vous expliquer tout ce que l'algorithme voit :
→ "Pourquoi cette conviction n'est que de X ?"
→ "C'est quoi un retest armé ?"
→ "Le prochain event macro, ça change quoi ?"
→ "Quelle est la marge d'erreur sur cette lecture ?"

Il **refuse** par contre de vous donner un ordre d'achat ou de vente —
c'est volontaire. Sentinel décrit, vous décidez.

Essayez sur votre dernière lecture : [lien]

L'équipe Sentinel

———
Démonstration paper-trading · Lecture algorithmique éducative.
[Se désabonner]
```

---

## Email 3 — D+7 : Pourquoi M.I.A. Markets — RÉVISÉ 2026-05-27

**Trigger** : 7 jours après signup, si l'utilisateur ne s'est pas converti payant
**Subject** : Pourquoi une approche pédagogique honnête (et pas un "système de trading")

```
Bonjour,

Vous testez M.I.A. Markets depuis 7 jours. Plutôt que de vous promettre
des chiffres de performance que nous n'avons pas encore validés OOS,
voici ce que vous obtenez réellement :

→ Pipeline algorithmique transparent (5 briques scientifiques publiques)
→ Méthodologie auditable (López de Prado, Corsi, Gibbs & Candès, etc.)
→ Données 7 années XAU/USD M15 (Dukascopy 98.4 % coverage)
→ Chatbot Sentinel qui définit le jargon et refuse de vous donner un ordre

Ce que nous N'AFFIRMONS PAS encore (parce que nous le validons proprement) :
- "PF 1.30 garanti"
- "X % de profit"
- "Système rentable"

Nous travaillons activement (Sprint 1 du plan dev) sur la validation
statistique OOS du moteur de scoring. Quand les seuils Brier skill > +2 %,
DSR > 1.0, PBO < 0.5 seront atteints, nous publierons les chiffres
avec leurs intervalles de confiance et la méthodologie complète.

En attendant, le plan **Découverte (9 €/mois)** débloque la structure SMC,
le régime, la volatilité, et 50 questions/jour avec Sentinel — pour
apprendre à lire les marchés Or et FX comme un quant pédagogique.

Essai gratuit 14 jours sans carte bancaire : [Lien découverte trial →]

L'équipe Sentinel

———
Outil pédagogique d'analyse algorithmique · Phase d'accès anticipé.
Ne constitue ni un signal de trading, ni un conseil en investissement.
[Se désabonner]
```

---

## Email 4 — D+13 : Trial-end reminder (Starter)

**Trigger** : 13 jours après début essai Starter sans CB (1 jour avant expiration)
**Subject** : Votre essai Starter expire demain

```
Bonjour,

Votre essai gratuit M.I.A. Markets Starter expire demain.

Vous avez utilisé pendant 13 jours :
→ X lectures consultées
→ Y questions posées à Sentinel
→ Z sections détaillées dépliées

Pour continuer sans interruption, ajoutez une carte bancaire :
[Continuer en Starter $29/mo →]
[Passer en Pro $79/mo →]

Sinon, votre compte retourne automatiquement en plan FREE demain
(pas de débit, pas de surprise).

Vous pouvez aussi nous dire ce qui a manqué dans l'essai :
[Envoyer un retour rapide →]

L'équipe Sentinel

———
Remboursement intégral 30 jours sans question si vous changez d'avis.
[Se désabonner]
```

---

## Email 5 — Churn : Récupération abonné qui annule

**Trigger** : événement Stripe `subscription.deleted` ou `subscription.canceled`
**Subject** : Vous partez ? Une question avant de fermer.

```
Bonjour,

Vous venez de résilier votre abonnement M.I.A. Markets.
Je respecte votre décision — vraiment.

Avant que vous fermiez la porte, deux choses :

1. Vous gardez l'accès jusqu'à la fin de votre période payée
   (jusqu'au [DATE]).

2. Si quelque chose n'a pas fonctionné, je veux vraiment le savoir.
   Pas pour vous garder à tout prix, mais pour améliorer le produit.

   → Une simple réponse à ce mail suffit : qu'est-ce qui aurait dû
     être différent ?

Si vous changez d'avis sous 30 jours, je vous offre votre dernier mois
remboursé intégralement, sans question.

Merci pour le temps que vous avez passé avec nous.

L'équipe Sentinel
support@mia.markets

———
[Se désabonner de tous les emails M.I.A. Markets]
```

---

## Notes d'implémentation

### Variables dynamiques

Toutes les copies utilisent les variables suivantes (à interpoler côté serveur) :
- `{user.first_name}` — si fourni, sinon "Bonjour" seul
- `{user.signup_date}` — pour le calcul de D+N
- `{user.tier}` — FREE / Starter / Pro / Institutional
- `{user.trial_end_date}` — pour Email 4
- `{user.usage.lectures_count}`, `{user.usage.chat_questions_count}`, `{user.usage.sections_expanded}` — pour Email 4 personnalisation

### Opt-out RGPD obligatoire

Chaque email contient un lien `[Se désabonner]` qui :
1. Désactive le marketing email pour ce user
2. **N'affecte pas** les emails transactionnels obligatoires (confirmation paiement, sécurité)

### Cadence anti-spam

Pas plus d'1 email marketing tous les 3 jours sur un même user. Implémentation via `email_throttle.py` (V2 DG-131).

### Tracking

Chaque email contient `?utm_source=lifecycle&utm_campaign={template_id}` pour mesure analytique Plausible.
