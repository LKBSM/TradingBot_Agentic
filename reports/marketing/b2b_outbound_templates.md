# B2B Outbound — Templates & Prospect List

> Sprint COMM-2B.3 (Karim) — 6h, 5 prospects ciblés, ≥ 2 réponses, ≥ 1 démo, 1 contrat M12.

## Positionnement

**Smart Sentinel B2B "Data Quality Enrichment"** — enrichissez vos signaux clients existants avec un contexte LLM sourcé : régime de marché, corrélations, narrative auditable, multi-langue FR/EN/DE/ES.

**Pas un revendeur de signaux.** On enrichit les vôtres.

**Prix d'appel** : 499€/mo basic 1k req / 1500€/mo pro 10k req / 3000€+ enterprise custom.

## Prospect list (5 cibles)

### 1. IC Markets (broker AU + global)
- **Pitch angle** : reach FR-Quebec + DE-Suisse — leur copy-trading "ZuluTrade clone" manque de contexte narratif. Notre `/enrich` ajoute en 200ms.
- **Contact** : sales@icmarkets.com + partenariats LinkedIn
- **Signaux d'intérêt** : présence Twitter active, blog avec contenus éducatifs

### 2. Exness (broker CY + global, gros volume XAU)
- **Pitch angle** : XAU est leur instrument #1 (~40% du volume FX). Notre spécialisation gold + multi-lang correspond.
- **Contact** : ib-partnerships@exness.com
- **Signaux** : sponsoring conférences forex, app mobile avec signaux internes

### 3. Pepperstone (broker AU + UK)
- **Pitch angle** : UK MiFID + Australie ASIC, deux régulateurs stricts. Notre audit trail B2B (hash-chained) répond à un besoin compliance réel.
- **Contact** : partnerships@pepperstone.com

### 4. Darwinex (broker ES, copy-trading factor-based)
- **Pitch angle** : leur modèle factor analytics manque de narrative humanisée. Notre LLM peut traduire facteurs algo → texte client en ES + EN + DE.
- **Contact** : business@darwinex.com

### 5. Forex Tester / MQL5 vendors
- **Pitch angle** : pas un broker, mais un **EA marketplace**. Les dev d'EA pourraient utiliser notre `/audit-quality` (QUANT-2B.3) pour valider leurs claims de performance.
- **Contact** : business@mql5.com + 3-4 vendors top sur leur marketplace

## Email template — outreach initial

```
Sujet : Enrichir vos signaux XAU avec un contexte LLM sourcé (audit-trail inclus)

Bonjour [Prénom],

Je suis [Nom], fondateur de Smart Sentinel AI. Nous avons construit
un service d'enrichissement contextuel pour les signaux de trading :
notre API /enrich prend un signal client {instrument, direction,
prix, time} et retourne en 200ms :

- regime de marché (HMM 3-états sur returns)
- corrélations cross-asset (XAU vs DXY, real yields, BTC)
- narrative LLM sourcée en FR/EN/DE/ES (UE 2024/2811 compliant)
- audit-trail hash-chained pour traçabilité réglementaire

Ce n'est PAS un service de signaux concurrent — c'est une couche
d'enrichissement par-dessus les vôtres. Cas d'usage typique :
votre copy-trading affiche aujourd'hui "buy XAU @ 2350, SL 2340" ;
demain il affiche le même signal + 3 paragraphes de contexte
auditable que vos clients peuvent lire avant de copier.

Pricing : 499€/mo basic (1k req) → 3000€/mo enterprise (illimité).

Démo 15min cette semaine ? Lien Cal.com : [URL]

Disponible aussi par téléphone : [tel]

Best,
[Nom]
Founder, Smart Sentinel AI
[website] · [LinkedIn]
```

## Follow-up template — J+5 sans réponse

```
Sujet : Re: Enrichir vos signaux XAU avec un contexte LLM sourcé

Bonjour [Prénom],

Suite à mon email du [date], voici un exemple concret de payload :

[insertion d'un mockup JSON `/enrich` réponse avec narrative FR]

Si l'angle n'est pas pertinent, dites-le moi — je m'efface.
Sinon, 15min de démo ?

Best,
[Nom]
```

## Démo agenda (15min)

1. **0-2min** : qui on est, pourquoi narrative-first et pas "signaux" (verdict A1, transparence radicale).
2. **2-7min** : démo live `/enrich` — un signal de leur côté, retour JSON contextuel, narrative générée.
3. **7-10min** : audit-trail demo — verify chain hash, montrer qu'on peut prouver "vous avez bien reçu cet enrichissement à 14:23:01 UTC".
4. **10-13min** : Q&A et pricing.
5. **13-15min** : prochaine étape — POC 1 mois gratuit sur 100 req/jour, puis contrat ou pas.

## KPI tracking

| Métrique | Cible M9 | Cible M12 |
|---|---|---|
| Emails envoyés | 5 | 15 (3 par mois M10-M12) |
| Réponses obtenues | ≥ 2 | ≥ 5 |
| Démos données | ≥ 1 | ≥ 3 |
| POC signés (gratuit) | 0 | ≥ 2 |
| Contrats payants | 0 | ≥ 1 |

## Kill criteria

- Si 0 réponse sur les 5 premiers prospects M9 → revue pitch avec Sofia + revisite positioning avant relance.
- Si 1 réponse mais aucune démo M11 → friction au pitch, simplifier les bénéfices.
- Si 1 démo mais aucun POC signé M12 → pricing trop élevé OU produit pas mature, pivoter vers tier B2B Basic à 199€/mo en M13.
