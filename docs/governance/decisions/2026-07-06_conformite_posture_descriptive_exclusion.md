# Décision — Stratégie de conformité : posture descriptive + exclusion US/UK/OFAC, sans consultation avocat planifiée

**Date** : 2026-07-06 · **Décideur** : fondateur (session live)
**Remplace, sur les points géo/avocat** : `legal_bootstrap_strategy_2026_05_26.md`
(géo-restriction FR/BE/CH/LU + migration avocat M3), `legal_migration_plan_to_lawyer.md`
(plan avocat entier), et toute liste « 9 pays Phase 1 » incluant le Royaume-Uni
(`MASTER_PLAN.md`, briefs dérivés).

---

## 1. La décision

L'objectif de conformité du produit est : **« rester légal partout dans le monde »,
sans consultation d'avocat planifiée.** Le moyen retenu est le seul qui rende cet
objectif atteignable sans avis juridique :

1. **Posture produit descriptive verrouillée** (protection principale, mondiale) :
   - aucune recommandation personnalisée, aucun signal, aucun impératif
     achetez/vendez (verrouillé dans les prompts moteur + refus explicite du
     chatbot) ;
   - aucune promesse de gain ni claim de performance (purge claims 2026-07-04,
     test permanent anti-régression) ;
   - avertissement de risque et posture éducative partout.
   La quasi-totalité des régimes financiers mondiaux visent le conseil
   personnalisé, les signaux et les promesses de performance — un produit
   d'information générale qui n'en fait aucun reste hors périmètre réglementé
   dans la grande majorité des juridictions.

2. **Exclusion des régimes agressifs** où même la promotion commerciale d'un
   service financier étranger est encadrée :
   - **États-Unis** (SEC, portée extraterritoriale) ;
   - **Royaume-Uni** (FCA financial promotion regime, FSMA 2000 s.21) ;
   - **pays sous sanctions OFAC SDN** (CU/IR/KP/RU/SY/BY).
   Implémentation : `GeoBlockMiddleware` (HTTP 451) + divulgation alignée dans
   les CGU §4 et `/api/v1/terms`. L'exclusion est une conformité parfaite et
   gratuite : on ne peut rien reprocher à un service qui refuse de servir.

3. **Québec et reste du monde : servis.** Le Québec est la juridiction de
   rattachement (Loi 25 + LPC) — décision du 2026-07-05, PR #32,
   verrouillée par `TestQuebecServed`.

## 2. Ce que cette décision annule

| Point des docs de mai 2026 | Statut |
|---|---|
| Géo-restriction FR/BE/CH/LU (bootstrap §pilier) | **Annulé** — périmètre servi = monde entier moins US/UK/OFAC |
| « 9 pays Phase 1 » incluant le Royaume-Uni | **Annulé** — UK exclu tant que la décision n'est pas révisée |
| Migration avocat M3 (déclencheurs MRR/trésorerie) | **Suspendu sine die** — aucune consultation planifiée |
| Cadre auto-entrepreneur FR / CM2C / médiation FR | **Caduc** — l'entreprise est québécoise (Loi 25 + LPC) ; aucune adhésion médiateur FR |

Les documents concernés restent en place comme archives ; un en-tête de
supersession y renvoie ici.

## 3. Conditions de révision

Cette décision est stable tant que le produit reste descriptif. Elle DOIT être
révisée si l'un de ces événements survient :

- **Ouverture voulue des États-Unis ou du Royaume-Uni** → il n'existe pas de
  chemin gratuit : soit avis juridique local, soit statu quo (exclusion).
- **Toute fonctionnalité qui rapproche le produit d'une recommandation**
  (score de conviction actionnable, alertes d'action, copy-trading, backtest
  vendu comme performance) → c'est le facteur qui crée du risque légal mondial,
  pas la géographie. La revue doit précéder le développement, pas le suivre.
- Mise en demeure, plainte ou contact d'un régulateur, où que ce soit.

## 4. Garde-fous techniques existants

- `GeoBlockMiddleware` + tests (`tests/test_geo_block.py`) — deny-list = US/GB/OFAC, régions vide.
- Wording moteur : interdits impératifs achetez/vendez multi-langues (`src/intelligence/llm_narrative_engine.py`, `template_narrative_engine.py`, `compliance_checker.py`).
- Claims frontend : test permanent chaînes interdites (`webapp/tests/claims-cleanup.test.ts`).
- Versionnage de consentement : `LAST_UPDATED` (`src/api/routes/legal.py`) = source unique, bump obligatoire à chaque changement de texte légal.

## 5. Reste à faire (hérité, inchangé par cette décision)

- Purger les prompts RAG qui citent le règlement inexistant « UE 2024/2811 »
  (`src/intelligence/rag/prompts.py`, 4 langues) + l'entrée fabriquée
  `ue-2024-2811` de `data/rag/sources_manifest.yaml`.
- Rebrand des terms legacy « Smart Sentinel AI » → MIA Markets (`/api/v1/terms`).
- Pages `/mentions-legales` et politique de confidentialité définitive
  (adaptées au cadre québécois, pas au cadre auto-entrepreneur FR des templates).
