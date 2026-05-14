# Conditions Générales d'Utilisation — Smart Sentinel AI

**Version v2 (Phase 2B)** — Effective : à signaler après revue avocat.
**Périmètre** : webapp publique smartsentinel.ai, API B2B, Telegram bot, contenus éditoriaux.

> Ce document est un **brouillon préparé en interne**. Il doit être relu par un avocat fintech français/canadien avant publication (Sprint INFRA-2B.4 — €1500 budget). En l'état, ne le mettez pas en ligne sans validation.

## 1. Identification de l'éditeur

- **Service** : Smart Sentinel AI
- **Éditeur** : *À compléter — raison sociale, RCS, capital, siège social*
- **Hébergeur webapp** : Vercel Inc., 340 S Lemon Ave #4133, Walnut, CA 91789, USA
- **Hébergeur API** : Railway Corporation, 251 Little Falls Drive, Wilmington DE 19808, USA
- **Délégué à la protection des données** : *À nommer*
- **Contact** : contact@smartsentinel.ai

## 2. Nature du service

Smart Sentinel AI est un **service d'analyse contextuelle éditoriale** sur le marché de l'or (XAU/USD) et instruments associés. Le service produit :

- des **analyses textuelles** générées par intelligence artificielle, sourcées et auditables,
- des **statistiques de contexte** (régimes de volatilité, corrélations inter-actifs, événements macro),
- une **courbe de paper-trading démonstrative** publiée en temps réel à fins éducatives.

### 2.1 Ce que Smart Sentinel AI N'EST PAS

Conformément au règlement UE 2024/2811 et à la position AMF 2024-09 sur les finfluenceurs :

- **Smart Sentinel AI n'est PAS un conseiller en investissement** (au sens MiFID II art. 4.1.4) ni un conseiller en gestion de patrimoine.
- **Smart Sentinel AI ne formule PAS de recommandations personnalisées** au sens MiFID II art. 24.
- **Smart Sentinel AI ne garantit AUCUN résultat financier**. Toute communication suggérant un edge prouvé, un rendement garanti, un signal d'achat ou de vente est interdite par les présentes CGU.
- **Smart Sentinel AI ne propose pas d'instruments financiers**, n'opère pas en tant que plateforme de courtage, et n'exécute aucun ordre pour le compte de tiers.

Le service relève de la **catégorie éditoriale** au sens de MiFID II art. 21 (information générale destinée à un large public).

## 3. Accès au service

### 3.1 Inscription

L'inscription FREE ne requiert pas de carte. Les tiers payants (LITE, PRO, PRO+) requièrent une adresse e-mail valide et un paiement Stripe. Un essai gratuit de 14 jours est offert sur LITE et PRO sans carte requise.

### 3.2 Géographie

Le service est **non disponible** aux résidents des juridictions suivantes pour raison réglementaire :

- États-Unis (toutes juridictions),
- Province de Québec (Canada),
- Royaume-Uni (à confirmer post-Brexit),
- Toute juridiction listée sur la SDN List de l'OFAC.

Le middleware Geo-Block (P29-W1) applique cette exclusion au niveau réseau. Toute tentative de contournement constitue une violation des présentes CGU.

### 3.3 Âge minimum

L'utilisateur déclare avoir au moins **18 ans** au moment de l'inscription. Les comptes mineurs détectés seront suspendus immédiatement, et les paiements remboursés.

## 4. Obligations de l'utilisateur

L'utilisateur s'engage à :

1. ne pas utiliser le service comme **substitut à un conseil financier personnalisé** délivré par un professionnel agréé,
2. ne pas considérer les analyses publiées comme des **recommandations d'achat ou de vente**,
3. effectuer **sa propre analyse** avant toute décision financière,
4. ne pas **republier ou revendre** les contenus du service sans autorisation écrite,
5. ne pas tenter de **rétro-ingénieurer** l'API, scraper les contenus, ou contourner les rate-limits.

## 5. Propriété intellectuelle

L'ensemble des contenus (analyses LLM, glossaire, articles SEO, courbes graphiques, code source webapp) est protégé par le droit d'auteur. Une licence d'utilisation **non-exclusive, non-transférable, révocable** est accordée à l'utilisateur dans le cadre de son abonnement actif.

Les sources tierces citées (papers académiques, rapports COT, FOMC minutes) restent la propriété de leurs auteurs originaux ; Smart Sentinel AI ne fait que les indexer pour citation.

## 6. Données personnelles (RGPD)

Voir le document séparé `privacy_v2_phase2b.md`. En résumé :

- données collectées : e-mail, statut d'abonnement, historique de paiement Stripe, IP (logs anti-abus 30 jours), interactions chat (90 jours),
- finalité : fourniture du service, lutte contre la fraude, conformité réglementaire,
- base légale : exécution du contrat (art. 6.1.b RGPD) et intérêt légitime (art. 6.1.f),
- transfert hors UE : oui (Vercel + Railway + Anthropic + Stripe — tous sous Privacy Shield ou clauses contractuelles types),
- droits : accès, rectification, effacement, portabilité, opposition — exerçables via privacy@smartsentinel.ai.

## 7. Limitation de responsabilité

Smart Sentinel AI ne saurait être tenue responsable :

- des **pertes financières** subies par l'utilisateur dans toute activité de trading,
- de l'**interprétation erronée** d'une analyse ou d'un signal de régime,
- des **interruptions de service** dépassant l'engagement de SLA de chaque tier,
- des **erreurs ou omissions** dans les sources tierces citées.

La responsabilité totale, tous dommages confondus, est plafonnée au **montant payé par l'utilisateur sur les 12 derniers mois**.

## 8. Suspension / résiliation

Smart Sentinel AI peut suspendre ou résilier un compte sans préavis en cas de :

- violation manifeste des présentes CGU,
- tentative de contournement géographique ou de la rate-limit,
- usage abusif (scraping, attaques),
- comportement haineux ou illégal dans le chat IA.

L'utilisateur peut résilier son abonnement à tout moment via le panneau de paramètres ; remboursement au prorata du mois entamé.

## 9. Droit applicable

Les présentes CGU sont régies par le **droit français**. Tout litige relève de la compétence des **tribunaux de Paris** (consommateurs UE : juridiction du lieu de résidence applicable).

## 10. Modification des CGU

Smart Sentinel AI se réserve le droit de modifier les présentes CGU. Les modifications matérielles sont notifiées par e-mail 30 jours avant entrée en vigueur. L'utilisation continue après ce délai vaut acceptation.

## 11. Historique

| Version | Date | Modifications |
|---|---|---|
| v1 | (P29-W2) | CGU initiale Phase 2A |
| v2 | (Phase 2B) | Refonte UE 2024/2811 + paragraphe finfluencer + B2B section + RGPD section |

---

*Document préparé par : équipe Smart Sentinel AI (Théo / Sofia).*
*À relire par : avocat fintech (TBD), session Sprint INFRA-2B.4.*
