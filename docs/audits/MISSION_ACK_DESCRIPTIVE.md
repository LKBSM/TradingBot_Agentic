# MISSION_ACK — Descriptive Quality Assessment (MIA Markets)

**Date** : 2026-05-27
**Mission** : auditer MIA Markets **comme indicateur descriptif**, pas comme système de trading.

1. **L'audit du 2026-05-27 (`AUDIT_ALGO_2026_05_27.md`) répond à la mauvaise question pour ce produit.** Il mesure la rentabilité (PF 0.786, return −62 %, Pearson score↔PnL −0.023) — légitime, mais hors-sujet pour un indicateur descriptif qui ne dit jamais « achète » ou « vends ».
2. **La bonne question commerciale** : « quand l'algo annonce un événement (BOS, FVG, OB, régime, vol, jump, blackout, intervalle conformel), cet événement est-il réellement présent, stable, et — pour les claims probabilistes — bien calibré ? »
3. **Périmètre strict** : structure SMC (BOS/CHOCH/FVG/OB/retest/invalidation), HMM, BOCPD, jump ratio, HAR-RV forecast, intervalle conformel (vol + conviction), calendrier événementiel/blackouts, décomposition 8 composantes, niveaux exposés, métadonnées. **Hors-périmètre** : PF, hit rate, return, tiers PREMIUM/STANDARD, LLM, architecture, compliance, pricing, frontend.
4. **Trois questions par bloc** : Q1 justesse factuelle (l'événement existe-t-il ?), Q2 stabilité temporelle (l'info reste-t-elle valide sur la fenêtre annoncée ?), Q3 calibration (les probas/intervalles ont-ils la couverture promise ?). Honnêteté brutale, pas de fausse précision, OOS obligatoire — si non mesurable rigoureusement, je le déclare « non évaluable » plutôt que d'inventer.
5. **Différence claire avec l'audit précédent** : si je dérive vers du PF/return/hit rate, je me recadre — ce n'est pas la mission. La mesure ici est : **MIA décrit-il bien ce qu'il dit voir, oui ou non**.

Livrables : `docs/audits/descriptive_quality_assessment.md` (méthodo + 5 parties) + `docs/audits/descriptive_quality_data.json` (métriques brutes reproductibles) + `docs/audits/OUT_OF_SCOPE.md` pour les sujets hors-mission rencontrés.

**Prochaine étape** : présenter la méthodologie (Partie 1) et attendre validation utilisateur avant exécution.
