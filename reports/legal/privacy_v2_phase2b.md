# Politique de Confidentialité — Smart Sentinel AI

**Version v2 (Phase 2B)** — Effective : après revue avocat.

> Document brouillon — à relire par avocat fintech / DPO avant publication (Sprint INFRA-2B.4).

## 1. Responsable du traitement

- **Smart Sentinel AI** (raison sociale à compléter)
- DPO : *à nommer* — privacy@smartsentinel.ai
- Adresse postale : *à compléter*

## 2. Données collectées

### 2.1 Données fournies par l'utilisateur

- **E-mail** (obligatoire à l'inscription) — base légale : exécution du contrat (RGPD art. 6.1.b)
- **Mot de passe** stocké chiffré (bcrypt) — base légale : exécution du contrat
- **Informations de paiement** (numéro de carte, billing address) — collectées et stockées exclusivement par Stripe, jamais par nos serveurs — base légale : exécution du contrat
- **Préférences linguistiques** (FR / EN / DE / ES) — base légale : exécution du contrat

### 2.2 Données collectées automatiquement

- **Adresse IP** (logs 30 jours pour anti-abus + geo-block) — base légale : intérêt légitime (RGPD art. 6.1.f)
- **User-Agent** + headers d'accès (logs 30 jours) — base légale : intérêt légitime
- **Historique des analyses consultées** (90 jours) — base légale : exécution du contrat
- **Interactions Chat IA** (questions + réponses, 90 jours) — base légale : exécution du contrat + amélioration du service (anonymisé après expiration)
- **Identifiant Stripe customer** (durée du contrat + 5 ans pour obligations comptables) — base légale : obligation légale (RGPD art. 6.1.c)

### 2.3 Cookies

| Cookie | Finalité | Durée | Type |
|---|---|---|---|
| `next-intl-locale` | mémoriser la langue choisie | 1 an | Strictement nécessaire |
| `session` | session authentifiée | 30 jours | Strictement nécessaire |
| `_stripe_*` | session Stripe checkout | session | Tiers (Stripe) |

**Aucun cookie publicitaire, aucun cookie tiers de tracking.** Pas de Google Analytics, pas de Facebook Pixel, pas de Meta Tags. Métriques server-side seulement (Sentry, in-house latency tracker OBS-2B.4).

## 3. Finalités du traitement

1. **Fourniture du service** (analyses, chat, dashboard, billing)
2. **Conformité réglementaire** (geo-block, audit trail compliance, archives 5 ans)
3. **Lutte contre la fraude** (rate-limit, abuse detection)
4. **Communication transactionnelle** (e-mails de paiement, expiration, factures)
5. **Amélioration du service** (analyse agrégée et anonymisée des interactions chat — opt-out possible)

**Aucun usage marketing tiers, aucun profilage publicitaire, aucune vente de données.**

## 4. Destinataires

| Destinataire | Finalité | Pays | Garantie de transfert |
|---|---|---|---|
| Vercel Inc. | Hébergement webapp | USA | DPF (EU-US Data Privacy Framework) |
| Railway Corp | Hébergement API | USA | Clauses contractuelles types |
| Anthropic PBC | Génération LLM (chat, narratives) | USA | DPF |
| Stripe Payments Europe Ltd | Paiements | Irlande | UE (interne) |
| Sentry | Logs d'erreurs | USA / UE | DPF |

**Aucune donnée n'est partagée avec des annonceurs.** Smart Sentinel AI ne vend ni ne loue jamais les données utilisateurs.

## 5. Durées de conservation

| Catégorie | Durée |
|---|---|
| Compte actif | toute la durée de l'abonnement |
| Compte inactif (FREE) | 24 mois après dernière connexion → suppression automatique |
| Logs anti-abus (IP, UA) | 30 jours |
| Historique chat | 90 jours puis anonymisation |
| Factures + Stripe ID | 5 ans (obligation comptable FR) |
| Audit trail B2B (DATA-2B.4) | 7 ans (obligation MiFID II) |

## 6. Droits de l'utilisateur

Conformément aux articles 15-22 du RGPD, l'utilisateur dispose des droits suivants, exerçables à privacy@smartsentinel.ai :

- **Accès** (art. 15) : copie complète des données dans un format lisible
- **Rectification** (art. 16) : correction des données inexactes
- **Effacement** (art. 17) : suppression du compte + données associées (sous réserve des obligations légales de conservation)
- **Portabilité** (art. 20) : export JSON/CSV des données
- **Opposition** (art. 21) : refuser le traitement basé sur intérêt légitime
- **Limitation** (art. 18) : gel temporaire d'un traitement contesté

**Délai de réponse** : 30 jours maximum. Refus motivé en cas d'obligation légale contraire.

L'utilisateur peut également **introduire une réclamation auprès de la CNIL** (https://www.cnil.fr/fr/plaintes) à tout moment, sans préjudice de tout autre recours.

## 7. Sécurité

- **Chiffrement en transit** : TLS 1.3 obligatoire sur toutes les surfaces
- **Chiffrement au repos** : disques Vercel + Railway chiffrés AES-256
- **Authentification** : sessions JWT signées, rotation hebdomadaire des clés
- **API keys** : SHA-256, jamais stockées en clair (KeyStore Sprint OBS)
- **Audit trail** : toutes les actions admin journalisées (AdminActionLog SECURITY-2B.1)
- **Sauvegardes** : quotidiennes vers Backblaze B2 (Sprint INFRA-2B.6), conservées 30 jours
- **Notification de fuite** : conformément à RGPD art. 33, signalement CNIL sous 72h en cas de fuite à risque

## 8. Mineurs

Le service est interdit aux moins de 18 ans. Aucune donnée d'enfant n'est sciemment collectée. Toute donnée détectée comme appartenant à un mineur est supprimée immédiatement et le compte fermé.

## 9. Modifications

Toute modification matérielle de la présente politique est notifiée 30 jours avant entrée en vigueur. Les modifications mineures (clarifications, mises à jour de coordonnées) ne sont pas notifiées individuellement.

## 10. Historique

| Version | Date | Modifications |
|---|---|---|
| v1 | (P29-W2) | Privacy initiale Phase 2A |
| v2 | (Phase 2B) | Refonte RGPD complète + Stripe Customer ID 5 ans + audit trail 7 ans MiFID II |
