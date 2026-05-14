# Claims Checklist — Marketing Phase 2B

> Sprint INFRA-2B.4 (Théo) + cross-check Sofia (RISK-2B.2)
>
> Cette checklist est **opposable** : avant toute publication (SEO, YouTube, LinkedIn, Telegram, e-mail, pricing, copy webapp), le contenu doit être confronté à cette liste. Le `ComplianceChecker` (`src/risk/compliance_checker.py`) automatise la majeure partie.

## ❌ INTERDIT — UE 2024/2811 + AMF 2024-09

### Catégorie A — Recommandations directes
- ❌ "Achetez maintenant"
- ❌ "Vendez maintenant"
- ❌ "Vous devriez acheter / vendre"
- ❌ "Signal d'achat" / "signal de vente"
- ❌ "Buy this", "Sell this"
- ❌ Toute formulation impérative invitant à exécuter un ordre

### Catégorie B — Garanties de résultat
- ❌ "Garanti", "garantie de performance"
- ❌ "100% sûr", "100% safe"
- ❌ "Sans risque"
- ❌ "Profits assurés"

### Catégorie C — Performance/edge
- ❌ "Edge prouvé", "alpha démontré"
- ❌ "+X%/mois", "+X%/an", "X% de rendement" (sauf disclaimer paper-trading démonstratif)
- ❌ "Notre algo bat le marché"
- ❌ "Outperforme [benchmark]"

### Catégorie D — Conseil personnalisé
- ❌ "Adapté à votre situation"
- ❌ "Investissement recommandé pour vous"
- ❌ "Stratégie sur-mesure"

### Catégorie E — Comparatif déloyal
- ❌ Comparaison nominative à un concurrent sans disclaimer factuel sourcé
- ❌ "Meilleur que [concurrent]"

## ✅ AUTORISÉ — formulation éditoriale

### Substituts pour Catégorie A
- ✅ "Configuration haussière détectée" (vs "achetez")
- ✅ "Setup baissier identifié" (vs "vendez")
- ✅ "Confluence de facteurs alignés" (vs "signal fort")
- ✅ "Les indicateurs suggèrent un biais haussier" (avec citation source)

### Substituts pour Catégorie B
- ✅ "Probabilité historique de retour à la moyenne" (avec n, période, source)
- ✅ "Hit rate observé sur les 6 dernières années : X%"

### Substituts pour Catégorie C
- ✅ "Aucun edge revendiqué — voir notre transparence en direct"
- ✅ "Démonstration paper-trading éducative" (sur toute courbe)

### Substituts pour Catégorie D
- ✅ "Information générale destinée à un large public"
- ✅ "Faites votre propre analyse avant toute décision"

## Disclaimers obligatoires

### Sur toute analyse publique
> "Analyse algorithmique éducative. Pas un conseil en investissement."

### Sur toute courbe d'équité
> "Démonstration paper-trading. Smart Sentinel ne prétend PAS posséder un edge. Cette courbe est éducative."

### Sur toute communication marketing
> "Smart Sentinel AI produit des analyses éditoriales contextuelles, pas des recommandations personnalisées. Conformément à UE 2024/2811."

### Sur Telegram (chaque message d'analyse)
> "📊 Analyse algorithmique • Aucune recommandation • Faites votre propre due-diligence"

## Workflow de validation

1. Rédaction → 2. ComplianceChecker (CI) → 3. Si KO, fix → 4. Revue Sofia (RISK-2B.2 cycle mensuel) → 5. Publication

Toute publication contournant ce workflow est **interdite**. Une violation détectée post-publication déclenche le runbook RISK-2B.4 §1 (correction sous 1h + audit).

## Exemples revus

### Exemple OK (publication autorisée)

> "Sur XAU/USD H1, une configuration de haute confluence est détectée : break of structure ce matin, retest d'un order block sur l'open de Londres, RSI sortant de la zone de sur-vente, et corrélation DXY à -0.78 (rolling 30j). Les rapports COT [source:cot-2026-05-10] montrent une augmentation des positions longues commerciales sur 3 semaines consécutives.
>
> *Analyse algorithmique éducative. Pas un conseil en investissement.*"

### Exemple KO (publication interdite)

> "🚀 XAU à 2400$ cette semaine ! Notre algo a déjà détecté ce signal d'achat avec une précision de 92%. Achetez maintenant avant la cassure !"
>
> Violations : signal d'achat, précision 92% (claim performance), "achetez maintenant" (impératif), absence de disclaimer.

## Fréquence de revue

- **Quotidien** : ComplianceChecker en CI sur tout commit touchant `content/` ou `narratives/`
- **Hebdomadaire** : Sofia revue échantillon 10% des publications
- **Mensuel** : RISK-2B.2 LLM-as-judge sur l'intégralité du corpus publié
- **Trimestriel** : revue formelle avec l'avocat fintech (post-INFRA-2B.4)
